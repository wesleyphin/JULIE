import logging
import math
import json
import hashlib
import os
import datetime as dt
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import CONFIG
from de3_v3_family_builder import (
    build_and_write_de3_v3_family_inventory,
    load_de3_v3_family_inventory,
)
from de3_v3_family_schema import (
    ACTIVE_FAMILY_CONTEXT_DIMENSIONS,
    build_active_context_joint_key,
    build_family_key_from_candidate,
    canonical_competition_status,
    canonical_evidence_support_tier,
    competition_status_is_eligible,
    effective_local_support_tier,
    family_id_from_key,
    normalize_context_buckets,
    safe_float,
    support_tier_from_sample_count,
    support_weight_for_tier,
)


class DE3V3FamilyRuntime:
    """Family-first DE3 runtime helper for v3 selection."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        base_cfg = cfg if isinstance(cfg, dict) else {}
        self.cfg = base_cfg
        self.enabled = bool(base_cfg.get("enabled", False))

        de3_v2_cfg = CONFIG.get("DE3_V2", {}) if isinstance(CONFIG.get("DE3_V2", {}), dict) else {}
        self.member_db_path = self._resolve_path(
            str(base_cfg.get("member_db_path") or de3_v2_cfg.get("db_path") or "dynamic_engine3_strategies_v2.json")
        )
        self.family_db_path = self._resolve_path(str(base_cfg.get("family_db_path") or "dynamic_engine3_families_v3.json"))
        self.auto_build_family_db = bool(base_cfg.get("auto_build_family_db", True))

        cp_cfg = base_cfg.get("context_profiles", {}) if isinstance(base_cfg.get("context_profiles"), dict) else {}
        self.context_profiles_enabled = bool(cp_cfg.get("enabled", True))
        self.cp_decisions_path = self._resolve_path(cp_cfg.get("decision_csv_path", "reports/de3_decisions.csv"))
        self.cp_trade_attr_path = self._resolve_path(
            cp_cfg.get("trade_attribution_csv_path", "reports/de3_decisions_trade_attribution.csv")
        )
        self.cp_min_bucket_samples = max(1, int(safe_float(cp_cfg.get("min_bucket_samples", 12), 12)))
        self.cp_strong_bucket_samples = max(
            self.cp_min_bucket_samples,
            int(safe_float(cp_cfg.get("strong_bucket_samples", 40), 40)),
        )
        self.cp_allow_legacy_parse = bool(cp_cfg.get("allow_parse_legacy_context_inputs", True))
        self.cp_require_enriched_for_runtime = bool(cp_cfg.get("require_enriched_export_for_runtime", False))

        ref_cfg = base_cfg.get("refined_universe", {}) if isinstance(base_cfg.get("refined_universe"), dict) else {}
        self.refined_universe_enabled = bool(ref_cfg.get("enabled", True))
        self.runtime_use_refined = bool(
            self.refined_universe_enabled and bool(ref_cfg.get("runtime_use_refined", True))
        )
        self.allow_runtime_raw_universe_override = bool(
            ref_cfg.get("allow_runtime_raw_universe_override", True)
        )
        self.require_meaningful_context_support_for_context_weight = bool(
            ref_cfg.get("require_meaningful_context_support_for_context_weight", True)
        )
        self.low_support_context_weight_cap = self._clip(
            safe_float(ref_cfg.get("low_support_context_weight_cap", 0.02), 0.02),
            0.0,
            0.25,
        )
        core_cfg = (
            base_cfg.get("de3v3_core", {})
            if isinstance(base_cfg.get("de3v3_core"), dict)
            else {}
        )
        core_ids_cfg = (
            core_cfg.get("anchor_family_ids", [])
            if isinstance(core_cfg.get("anchor_family_ids"), (list, tuple, set))
            else []
        )
        core_ids = [
            str(v).strip()
            for v in core_ids_cfg
            if str(v).strip()
        ]
        if not core_ids:
            core_ids = ["5min|09-12|long|Long_Rev|T6"]
        self.core_enabled = bool(core_cfg.get("enabled", True))
        self.core_anchor_family_ids: Tuple[str, ...] = tuple(core_ids)
        runtime_mode_raw = str(
            core_cfg.get(
                "default_runtime_mode",
                core_cfg.get("core_mode", "core_plus_satellites"),
            )
            or "core_plus_satellites"
        ).strip().lower()
        if runtime_mode_raw == "anchor_plus_satellites":
            runtime_mode_raw = "core_plus_satellites"
        runtime_mode_override = str(
            os.environ.get("DE3_V3_RUNTIME_MODE_OVERRIDE", "")
            or ""
        ).strip().lower()
        if runtime_mode_override in {"core_only", "core_plus_satellites", "satellites_only"}:
            runtime_mode_raw = runtime_mode_override
        if runtime_mode_raw not in {"core_only", "core_plus_satellites", "satellites_only"}:
            runtime_mode_raw = "core_plus_satellites"
        self.core_runtime_mode = str(runtime_mode_raw)
        self.force_anchor_when_eligible = bool(core_cfg.get("force_anchor_when_eligible", True))
        sat_cfg = (
            base_cfg.get("de3v3_satellites", {})
            if isinstance(base_cfg.get("de3v3_satellites"), dict)
            else {}
        )
        self.satellites_enabled = bool(sat_cfg.get("enabled", True))
        self.satellites_discovery_enabled = bool(sat_cfg.get("discovery_enabled", True))
        self.sat_min_standalone_viability = safe_float(
            sat_cfg.get("min_standalone_viability", 0.20),
            0.20,
        )
        self.sat_min_incremental_value = safe_float(
            sat_cfg.get("min_incremental_value_over_core", 0.05),
            0.05,
        )
        self.sat_max_retained = max(
            0,
            int(safe_float(sat_cfg.get("max_retained_satellites", 6), 6)),
        )
        self.sat_require_orthogonality = bool(sat_cfg.get("require_orthogonality", True))
        self.sat_max_overlap_with_core = self._clip(
            safe_float(sat_cfg.get("max_overlap_with_core", 0.80), 0.80),
            0.0,
            1.0,
        )
        self.sat_max_bad_overlap_with_core = self._clip(
            safe_float(sat_cfg.get("max_bad_overlap_with_core", 0.55), 0.55),
            0.0,
            1.0,
        )
        self.sat_allow_near_core_variants_if_incremental = bool(
            sat_cfg.get("allow_near_core_variants_if_incremental", True)
        )
        bloat_cfg = (
            base_cfg.get("bloat_control", {})
            if isinstance(base_cfg.get("bloat_control"), dict)
            else {}
        )
        self.enable_family_competition_balancing = bool(
            bloat_cfg.get("enable_family_competition_balancing", False)
        )
        self.enable_exploration_bonus = bool(
            bloat_cfg.get("enable_exploration_bonus", False)
        )
        self.enable_dominance_penalty = bool(
            bloat_cfg.get("enable_dominance_penalty", False)
        )
        self.enable_monopoly_canonical_force = bool(
            bloat_cfg.get("enable_monopoly_canonical_force", False)
        )
        self.enable_compatibility_tier_slot_pressure = bool(
            bloat_cfg.get("enable_compatibility_tier_slot_pressure", False)
        )
        self.core_satellite_state: Dict[str, Any] = {}
        self.core_family_ids_loaded: set = set(self.core_anchor_family_ids)
        self.satellite_retained_family_ids_loaded: set = set()
        self.satellite_suppressed_family_ids_loaded: set = set()

        prior_cfg = (
            base_cfg.get("prior_eligibility", {})
            if isinstance(base_cfg.get("prior_eligibility"), dict)
            else (
                base_cfg.get("family_eligibility", {})
                if isinstance(base_cfg.get("family_eligibility"), dict)
                else {}
            )
        )
        self.prior_eligibility_enabled = bool(prior_cfg.get("enabled", True))
        self.prior_min_support = safe_float(prior_cfg.get("min_total_support_trades", 25), 25)
        self.prior_min_best_pf = safe_float(prior_cfg.get("min_best_member_profit_factor", 0.90), 0.90)
        self.prior_min_best_pbr = safe_float(prior_cfg.get("min_best_member_profitable_block_ratio", 0.45), 0.45)
        self.prior_min_best_worst_pf = safe_float(prior_cfg.get("min_best_member_worst_block_pf", 0.70), 0.70)
        self.prior_min_best_worst_avg = safe_float(prior_cfg.get("min_best_member_worst_block_avg_pnl", -0.85), -0.85)
        self.prior_max_median_dd = safe_float(prior_cfg.get("max_median_drawdown_norm", 1.60), 1.60)
        self.prior_max_median_loss = safe_float(prior_cfg.get("max_median_loss_share", 0.85), 0.85)
        self.prior_min_median_structural = safe_float(prior_cfg.get("min_median_member_structural_score", -2.0), -2.0)
        self.prior_log_rejections = bool(prior_cfg.get("log_rejections", True))

        # Hard catastrophic caps are intentionally strict and only used for hard exclusion.
        self.prior_catastrophic_min_best_pf = safe_float(prior_cfg.get("catastrophic_min_best_member_profit_factor", 0.50), 0.50)
        self.prior_catastrophic_min_best_worst_pf = safe_float(prior_cfg.get("catastrophic_min_best_member_worst_block_pf", 0.25), 0.25)
        self.prior_catastrophic_max_dd = safe_float(prior_cfg.get("catastrophic_max_median_drawdown_norm", 2.30), 2.30)
        self.prior_catastrophic_max_loss = safe_float(prior_cfg.get("catastrophic_max_median_loss_share", 0.95), 0.95)

        usable_cfg = (
            base_cfg.get("usable_family_universe", {})
            if isinstance(base_cfg.get("usable_family_universe"), dict)
            else {}
        )
        self.usability_enabled = bool(usable_cfg.get("enabled", True))
        self.usability_reject_suppressed = bool(usable_cfg.get("reject_suppressed_families", True))
        self.usability_exclude_only_suppressed = bool(usable_cfg.get("exclude_only_suppressed_families", True))
        self.usability_low_support_fully_competitive = bool(usable_cfg.get("low_support_fully_competitive", True))
        self.usability_fallback_requires_weak_or_thin = bool(
            usable_cfg.get("fallback_only_requires_weak_or_thin_competition", True)
        )
        self.usability_fallback_min_active_competitors = max(
            1,
            int(safe_float(usable_cfg.get("min_active_competitors_for_fallback_restriction", 1), 1)),
        )

        support_cfg = usable_cfg.get("evidence_support", {}) if isinstance(usable_cfg.get("evidence_support"), dict) else {}
        self.evidence_mid_samples = max(
            1,
            int(
                safe_float(
                    support_cfg.get("min_mid_samples", usable_cfg.get("min_trades_for_state", 8)),
                    8,
                )
            ),
        )
        self.evidence_strong_samples = max(
            self.evidence_mid_samples,
            int(
                safe_float(
                    support_cfg.get("strong_samples", usable_cfg.get("min_trades_for_active", 20)),
                    20,
                )
            ),
        )

        adjustment_cfg = usable_cfg.get("evidence_adjustment", {}) if isinstance(usable_cfg.get("evidence_adjustment"), dict) else {}
        base_by_tier = adjustment_cfg.get("support_tier_base", {}) if isinstance(adjustment_cfg.get("support_tier_base"), dict) else {}
        quality_scale_by_tier = adjustment_cfg.get("support_tier_quality_scale", {}) if isinstance(adjustment_cfg.get("support_tier_quality_scale"), dict) else {}
        self.evidence_adjustment_base = {
            "none": safe_float(base_by_tier.get("none", 0.0), 0.0),
            "low": safe_float(base_by_tier.get("low", -0.005), -0.005),
            "mid": safe_float(base_by_tier.get("mid", 0.03), 0.03),
            "strong": safe_float(base_by_tier.get("strong", 0.08), 0.08),
        }
        self.evidence_adjustment_quality_scale = {
            "none": safe_float(quality_scale_by_tier.get("none", 0.0), 0.0),
            "low": safe_float(quality_scale_by_tier.get("low", 0.01), 0.01),
            "mid": safe_float(quality_scale_by_tier.get("mid", 0.08), 0.08),
            "strong": safe_float(quality_scale_by_tier.get("strong", 0.15), 0.15),
        }
        self.fallback_only_penalty = safe_float(adjustment_cfg.get("fallback_only_penalty", -0.05), -0.05)
        self.suppressed_adjustment = safe_float(adjustment_cfg.get("suppressed_adjustment", -0.20), -0.20)
        self.low_tier_min_adjustment = safe_float(adjustment_cfg.get("low_tier_min_adjustment", -0.02), -0.02)
        self.usability_quality_confidence_trades = max(
            1,
            int(
                safe_float(
                    adjustment_cfg.get("quality_confidence_trades", usable_cfg.get("quality_confidence_trades", 40)),
                    40,
                )
            ),
        )
        max_abs_adj = abs(safe_float(adjustment_cfg.get("max_abs_adjustment", 0.20), 0.20))
        self.usability_adj_min = -float(max_abs_adj)
        self.usability_adj_max = float(max_abs_adj)
        context_scale_cfg = adjustment_cfg.get("context_scale_by_evidence_tier", {}) if isinstance(adjustment_cfg.get("context_scale_by_evidence_tier"), dict) else {}
        self.context_scale_by_evidence_tier = {
            "none": self._clip(safe_float(context_scale_cfg.get("none", 0.0), 0.0), 0.0, 1.5),
            "low": self._clip(safe_float(context_scale_cfg.get("low", 0.15), 0.15), 0.0, 1.5),
            "mid": self._clip(safe_float(context_scale_cfg.get("mid", 0.60), 0.60), 0.0, 1.5),
            "strong": self._clip(safe_float(context_scale_cfg.get("strong", 1.00), 1.00), 0.0, 1.5),
        }

        fam_scoring = base_cfg.get("family_scoring", {}) if isinstance(base_cfg.get("family_scoring"), dict) else {}
        fam_weights = fam_scoring.get("weights", {}) if isinstance(fam_scoring.get("weights"), dict) else {}
        self.w_context_expectancy = safe_float(
            fam_weights.get("context_profile_expectancy", fam_weights.get("context_ev", 0.45)),
            0.45,
        )
        self.w_context_confidence = safe_float(
            fam_weights.get("context_profile_confidence", fam_weights.get("confidence", 0.20)),
            0.20,
        )
        self.w_prior = safe_float(fam_weights.get("family_prior", 0.50), 0.50)
        self.w_usability = safe_float(fam_weights.get("v3_realized_usability", fam_weights.get("usability_adjustment", 1.0)), 1.0)
        self.w_adaptive = safe_float(fam_weights.get("adaptive_policy", 0.10), 0.10)
        self.fs_normalize_prior_component = bool(
            fam_scoring.get("normalize_prior_component", True)
        )
        self.fs_cap_context_advantage_when_single_strong_family = bool(
            fam_scoring.get("cap_context_advantage_when_single_strong_family", True)
        )
        self.fs_single_strong_family_context_cap = max(
            0.0,
            safe_float(
                fam_scoring.get("single_strong_family_context_cap", 0.10),
                0.10,
            ),
        )
        self.fs_compatible_band_penalty = min(
            0.0,
            safe_float(fam_scoring.get("compatible_band_penalty", -0.03), -0.03),
        )
        self.fs_close_competition_margin = max(
            0.0,
            safe_float(fam_scoring.get("close_competition_margin", 0.24), 0.24),
        )
        self.fs_max_competition_adjustment_close = max(
            0.0,
            safe_float(
                fam_scoring.get("max_competition_adjustment_close", 0.12),
                0.12,
            ),
        )
        self.fs_max_competition_adjustment_far = max(
            0.0,
            safe_float(
                fam_scoring.get("max_competition_adjustment_far", 0.03),
                0.03,
            ),
        )
        self.fs_dominance_penalty_curve = str(
            fam_scoring.get("dominance_penalty_curve", "quadratic") or "quadratic"
        ).strip().lower()
        self.fs_exploration_bonus_curve = str(
            fam_scoring.get("exploration_bonus_curve", "quadratic_decay")
            or "quadratic_decay"
        ).strip().lower()
        self.fs_log_score_delta_ladder = bool(
            fam_scoring.get("log_score_delta_ladder", True)
        )

        comp_cfg = (
            base_cfg.get("family_competition", {})
            if isinstance(base_cfg.get("family_competition"), dict)
            else {}
        )
        self.use_bootstrap_family_competition_floor = bool(
            comp_cfg.get("use_bootstrap_family_competition_floor", True)
        )
        self.bootstrap_min_competing_families = max(
            1,
            int(safe_float(comp_cfg.get("bootstrap_min_competing_families", 3), 3)),
        )
        self.include_exact_and_compatible_only = bool(
            comp_cfg.get("include_exact_and_compatible_only", True)
        )
        raw_block_thresholds = (
            comp_cfg.get("temporary_excluded_thresholds", [])
            if isinstance(comp_cfg.get("temporary_excluded_thresholds"), (list, tuple, set))
            else []
        )
        raw_block_family_ids = (
            comp_cfg.get("temporary_excluded_family_ids", [])
            if isinstance(comp_cfg.get("temporary_excluded_family_ids"), (list, tuple, set))
            else []
        )
        self.temp_excluded_thresholds = {
            str(v).strip().upper()
            for v in raw_block_thresholds
            if str(v).strip()
        }
        self.temp_excluded_family_ids = {
            str(v).strip()
            for v in raw_block_family_ids
            if str(v).strip()
        }
        cap_cfg = (
            comp_cfg.get("family_candidate_cap", {})
            if isinstance(comp_cfg.get("family_candidate_cap"), dict)
            else {}
        )
        legacy_max_total = int(
            safe_float(comp_cfg.get("max_family_candidates_per_decision", 6), 6)
        )
        legacy_max_compatible = int(
            safe_float(comp_cfg.get("compatible_family_max_count", 4), 4)
        )
        legacy_compatible_penalty = -abs(
            safe_float(comp_cfg.get("compatible_family_penalty", -0.06), -0.06)
        )
        self.family_candidate_cap_enabled = bool(cap_cfg.get("enabled", True))
        self.cap_max_total_candidates = max(
            1,
            int(
                safe_float(
                    cap_cfg.get("max_total_candidates", legacy_max_total),
                    legacy_max_total,
                )
            ),
        )
        self.cap_min_exact_match_candidates = max(
            0,
            int(safe_float(cap_cfg.get("min_exact_match_candidates", 2), 2)),
        )
        self.cap_min_compatible_band_candidates = max(
            0,
            int(safe_float(cap_cfg.get("min_compatible_band_candidates", 2), 2)),
        )
        self.cap_max_exact_match_candidates = max(
            0,
            int(
                safe_float(
                    cap_cfg.get("max_exact_match_candidates", self.cap_max_total_candidates),
                    self.cap_max_total_candidates,
                )
            ),
        )
        self.cap_max_compatible_band_candidates = max(
            0,
            int(
                safe_float(
                    cap_cfg.get("max_compatible_band_candidates", legacy_max_compatible),
                    legacy_max_compatible,
                )
            ),
        )
        self.cap_use_preliminary_score_for_cap = bool(
            cap_cfg.get("use_preliminary_score_for_cap", True)
        )
        self.cap_log_pre_cap_post_cap = bool(
            cap_cfg.get("log_pre_cap_post_cap", True)
        )
        self.cap_preliminary_compatibility_penalty_exact = safe_float(
            cap_cfg.get("compatibility_penalty_exact", 0.0),
            0.0,
        )
        self.cap_preliminary_compatibility_penalty_compatible = safe_float(
            cap_cfg.get("compatibility_penalty_compatible", legacy_compatible_penalty),
            legacy_compatible_penalty,
        )
        # Legacy aliases remain populated for compatibility with existing logs/tooling.
        self.max_family_candidates_per_decision = int(self.cap_max_total_candidates)
        self.compatible_family_max_count = int(self.cap_max_compatible_band_candidates)
        self.compatible_family_penalty = float(legacy_compatible_penalty)
        compatibility_cfg = (
            comp_cfg.get("compatibility_bands", {})
            if isinstance(comp_cfg.get("compatibility_bands"), dict)
            else {}
        )
        self.session_nearby_max_hour_distance = self._clip(
            safe_float(compatibility_cfg.get("session_nearby_max_hour_distance", 6.0), 6.0),
            0.0,
            12.0,
        )
        self.timeframe_nearby_max_minutes_delta = max(
            0,
            int(safe_float(compatibility_cfg.get("timeframe_nearby_max_minutes_delta", 10), 10)),
        )
        self.timeframe_nearby_max_ratio = max(
            1.0,
            safe_float(compatibility_cfg.get("timeframe_nearby_max_ratio", 3.0), 3.0),
        )
        self.strategy_type_allow_related = bool(
            compatibility_cfg.get("strategy_type_allow_related", True)
        )
        balance_cfg = (
            comp_cfg.get("family_competition_balance", {})
            if isinstance(comp_cfg.get("family_competition_balance"), dict)
            else {}
        )

        def _balance_value(key: str, default: Any) -> Any:
            if key in balance_cfg:
                return balance_cfg.get(key)
            return comp_cfg.get(key, default)

        self.family_competition_balance_enabled = bool(_balance_value("enabled", True))
        if not self.enable_family_competition_balancing:
            self.family_competition_balance_enabled = False
        self.dominance_window_size = max(5, int(safe_float(_balance_value("dominance_window_size", 160), 160)))
        self.dominance_penalty_start_share = self._clip(
            safe_float(_balance_value("dominance_penalty_start_share", 0.55), 0.55),
            0.0,
            1.0,
        )
        self.dominance_penalty_max = abs(
            safe_float(
                _balance_value("dominance_penalty_max", _balance_value("max_dominance_penalty", 0.12)),
                0.12,
            )
        )
        self.max_dominance_penalty = abs(safe_float(_balance_value("max_dominance_penalty", self.dominance_penalty_max), self.dominance_penalty_max))
        self.low_support_exploration_bonus = max(
            0.0,
            safe_float(_balance_value("low_support_exploration_bonus", _balance_value("max_exploration_bonus", 0.08)), 0.08),
        )
        self.max_exploration_bonus = max(
            0.0,
            safe_float(_balance_value("max_exploration_bonus", self.low_support_exploration_bonus), self.low_support_exploration_bonus),
        )
        self.exploration_bonus_decay_threshold = max(
            1,
            int(safe_float(_balance_value("exploration_bonus_decay_threshold", 20), 20)),
        )
        self.competition_margin_points = max(
            0.0,
            safe_float(_balance_value("competition_margin_points", 0.22), 0.22),
        )
        self.cap_context_advantage_in_close_competition = bool(
            _balance_value("cap_context_advantage_in_close_competition", True)
        )
        self.max_context_advantage_cap = max(
            0.0,
            safe_float(_balance_value("max_context_advantage_cap", 0.12), 0.12),
        )

        configured_dims = fam_scoring.get("active_context_dimensions")
        if isinstance(configured_dims, (list, tuple)):
            dims = [str(dim).strip() for dim in configured_dims if str(dim).strip()]
        else:
            dims = list(ACTIVE_FAMILY_CONTEXT_DIMENSIONS)
        self.active_context_dimensions: Tuple[str, ...] = tuple(dims or list(ACTIVE_FAMILY_CONTEXT_DIMENSIONS))
        self.use_context_profiles = bool(fam_scoring.get("use_context_profiles", True))
        self.fallback_to_priors_when_profile_weak = bool(fam_scoring.get("fallback_to_priors_when_profile_weak", True))
        self.use_joint_context_profiles = bool(fam_scoring.get("use_joint_context_profiles", False))

        self.min_context_bucket_samples = max(
            1,
            int(safe_float(fam_scoring.get("min_context_bucket_samples", self.cp_min_bucket_samples), self.cp_min_bucket_samples)),
        )
        self.strong_context_bucket_samples = max(
            self.min_context_bucket_samples,
            int(
                safe_float(
                    fam_scoring.get("strong_context_bucket_samples", self.cp_strong_bucket_samples),
                    self.cp_strong_bucket_samples,
                )
            ),
        )
        self.context_profile_weight_strong = self._clip(
            safe_float(fam_scoring.get("context_profile_weight_strong", 1.0), 1.0),
            0.0,
            2.0,
        )
        self.context_profile_weight_mid = self._clip(
            safe_float(fam_scoring.get("context_profile_weight_mid", 0.45), 0.45),
            0.0,
            2.0,
        )
        self.context_profile_weight_low_or_none = self._clip(
            safe_float(fam_scoring.get("context_profile_weight_low_or_none", 0.05), 0.05),
            0.0,
            2.0,
        )
        if self.require_meaningful_context_support_for_context_weight:
            self.context_profile_weight_low_or_none = min(
                float(self.context_profile_weight_low_or_none),
                float(self.low_support_context_weight_cap),
            )
            self.context_scale_by_evidence_tier["low"] = min(
                float(self.context_scale_by_evidence_tier.get("low", 0.0)),
                float(self.low_support_context_weight_cap),
            )

        fam_gates = fam_scoring.get("gates", {}) if isinstance(fam_scoring.get("gates"), dict) else {}
        self.gates_enabled = bool(fam_gates.get("enabled", True))
        self.min_family_score = safe_float(fam_gates.get("min_family_score", 0.05), 0.05)
        self.min_adaptive_component = safe_float(fam_gates.get("min_adaptive_component", -2.0), -2.0)
        self.log_decisions = bool(fam_scoring.get("log_decisions", True))
        self.log_top_k = max(1, int(safe_float(fam_scoring.get("log_top_k", 3), 3)))

        member_cfg = (
            base_cfg.get("local_member_selection", {})
            if isinstance(base_cfg.get("local_member_selection"), dict)
            else {}
        )
        member_weights = member_cfg.get("weights", {}) if isinstance(member_cfg.get("weights"), dict) else {}
        self.w_local_edge = safe_float(member_weights.get("edge_points", 0.55), 0.55)
        self.w_local_struct = safe_float(member_weights.get("structural_score", 0.30), 0.30)
        self.w_local_payoff = safe_float(member_weights.get("payoff", 0.10), 0.10)
        self.w_local_context_bracket = safe_float(member_weights.get("context_bracket_suitability", 0.10), 0.10)
        self.w_local_conf = safe_float(member_weights.get("confidence", 0.05), 0.05)
        self.local_rr_target = safe_float(member_cfg.get("target_rr", 1.50), 1.50)
        self.local_rr_tolerance = max(0.1, safe_float(member_cfg.get("rr_tolerance", 1.50), 1.50))
        self.local_target_rr_expanding = safe_float(member_cfg.get("target_rr_expanding", 2.00), 2.00)
        self.local_target_rr_compressed = safe_float(member_cfg.get("target_rr_compressed", 1.15), 1.15)
        self.local_target_sl_atr_expanding = safe_float(member_cfg.get("target_sl_atr_expanding", 1.40), 1.40)
        self.local_target_sl_atr_neutral = safe_float(member_cfg.get("target_sl_atr_neutral", 1.10), 1.10)
        self.local_target_sl_atr_compressed = safe_float(member_cfg.get("target_sl_atr_compressed", 0.85), 0.85)
        self.local_sl_atr_tolerance = max(0.1, safe_float(member_cfg.get("sl_atr_tolerance", 0.90), 0.90))
        self.local_context_scale_mid = self._clip(
            safe_float(member_cfg.get("context_bracket_weight_scale_mid", 0.45), 0.45),
            0.0,
            1.0,
        )
        self.local_context_scale_low = self._clip(
            safe_float(member_cfg.get("context_bracket_weight_scale_low", 0.0), 0.0),
            0.0,
            1.0,
        )
        self.local_allow_context_mid = bool(member_cfg.get("allow_context_adaptation_mid_support", True))
        self.local_mid_noncanonical_penalty = safe_float(
            member_cfg.get("mid_support_noncanonical_penalty", 0.05),
            0.05,
        )
        self.local_freeze_to_canonical_low = bool(member_cfg.get("freeze_to_canonical_when_low_support", True))
        self.force_canonical_when_family_monopoly = bool(
            member_cfg.get("force_canonical_when_family_monopoly", True)
        )
        if not self.enable_monopoly_canonical_force:
            self.force_canonical_when_family_monopoly = False
        self.monopoly_share_threshold = self._clip(
            safe_float(member_cfg.get("monopoly_share_threshold", 0.75), 0.75),
            0.0,
            1.0,
        )
        self.monopoly_lookback_window = max(
            5,
            int(safe_float(member_cfg.get("monopoly_lookback_window", 140), 140)),
        )
        self.local_full_adaptation_min_support_tier = str(
            member_cfg.get("full_adaptation_min_support_tier", "strong") or "strong"
        ).strip().lower()
        self.local_conservative_adaptation_min_support_tier = str(
            member_cfg.get("conservative_adaptation_min_support_tier", "mid") or "mid"
        ).strip().lower()
        if self.local_full_adaptation_min_support_tier not in {"low", "mid", "strong"}:
            self.local_full_adaptation_min_support_tier = "strong"
        if self.local_conservative_adaptation_min_support_tier not in {"low", "mid", "strong"}:
            self.local_conservative_adaptation_min_support_tier = "mid"
        self.local_min_score = safe_float(member_cfg.get("min_local_score", -999.0), -999.0)
        observability_cfg = (
            base_cfg.get("observability", {})
            if isinstance(base_cfg.get("observability"), dict)
            else {}
        )
        self.observability_enabled = bool(observability_cfg.get("enabled", True))
        self.observability_emit_family_score_trace = bool(
            observability_cfg.get("emit_family_score_trace", True)
        )
        self.observability_emit_member_resolution_audit = bool(
            observability_cfg.get("emit_member_resolution_audit", True)
        )
        self.observability_emit_choice_path_audit = bool(
            observability_cfg.get("emit_choice_path_audit", True)
        )
        self.observability_emit_score_path_audit = bool(
            observability_cfg.get("emit_score_path_audit", True)
        )
        self.observability_strict_score_path_assertions = bool(
            observability_cfg.get("strict_score_path_assertions", False)
        )
        self.observability_family_score_trace_max_rows = max(
            1000,
            int(safe_float(observability_cfg.get("family_score_trace_max_rows", 300000), 300000)),
        )
        self.observability_member_resolution_max_rows = max(
            1000,
            int(
                safe_float(
                    observability_cfg.get("member_resolution_trace_max_rows", 300000),
                    300000,
                )
            ),
        )
        self._recent_chosen_family_ids: List[str] = []

        self.family_inventory: Dict[str, Any] = {}
        self.families_by_id: Dict[str, Dict[str, Any]] = {}
        self._bundle_sections_present: List[str] = []
        self._bundle_section_sizes: Dict[str, Any] = {}
        self._artifact_fingerprint: Dict[str, Any] = {}
        self._path_counters: Dict[str, int] = self._init_path_counters()
        self._score_path_audit: Dict[str, Any] = self._init_score_path_audit()
        self._choice_path_audit: Dict[str, Any] = self._init_choice_path_audit()
        self._family_candidate_size_histogram: Dict[str, int] = {}
        self._family_score_trace_rows: List[Dict[str, Any]] = []
        self._member_resolution_trace_rows: List[Dict[str, Any]] = []
        self._family_eligibility_trace_rows: List[Dict[str, Any]] = []
        self._pre_cap_candidate_audit_rows: List[Dict[str, Any]] = []
        self._family_reachability_by_family: Dict[str, Dict[str, Any]] = {}
        self._bundle_usage_audit_cache: Optional[Dict[str, Any]] = None
        self._config_usage_audit_cache: Optional[Dict[str, Any]] = None
        self.last_load_status: Dict[str, Any] = {
            "enabled": bool(self.enabled),
            "family_artifact_path": str(self.family_db_path),
            "artifact_kind": "family_inventory",
            "bundle_loaded": False,
            "bundle_version": "",
            "bundle_build_timestamp": None,
            "runtime_use_refined": bool(self.runtime_use_refined),
            "core_enabled": bool(self.core_enabled),
            "core_runtime_mode": str(self.core_runtime_mode),
            "core_anchor_family_ids": list(self.core_anchor_family_ids),
            "core_family_count": 0,
            "satellite_family_count": 0,
            "active_runtime_family_count": 0,
            "core_satellite_state_loaded": False,
            "loaded_universe": "raw",
            "raw_family_count": 0,
            "retained_family_count": 0,
            "suppressed_family_count": 0,
            "raw_member_count": 0,
            "retained_member_count": 0,
            "suppressed_member_count": 0,
            "anchor_member_count": 0,
            "family_artifact_loaded": False,
            "family_count": 0,
            "auto_built": False,
            "load_error": "",
            "context_profiles_loaded": False,
            "enriched_export_required": False,
            "fallback_to_priors_enabled": bool(self.fallback_to_priors_when_profile_weak),
            "prior_eligibility_enabled": bool(self.prior_eligibility_enabled),
            "competition_floor": {
                "enabled": bool(self.use_bootstrap_family_competition_floor),
                "bootstrap_min_competing_families": int(self.bootstrap_min_competing_families),
                "include_exact_and_compatible_only": bool(self.include_exact_and_compatible_only),
                "temporary_excluded_thresholds": sorted(
                    list(self.temp_excluded_thresholds)
                ),
                "temporary_excluded_family_ids": sorted(
                    list(self.temp_excluded_family_ids)
                ),
                "max_family_candidates_per_decision": int(self.max_family_candidates_per_decision),
                "compatible_family_max_count": int(self.compatible_family_max_count),
                "compatible_family_penalty": float(self.compatible_family_penalty),
                "family_candidate_cap": {
                    "enabled": bool(self.family_candidate_cap_enabled),
                    "max_total_candidates": int(self.cap_max_total_candidates),
                    "min_exact_match_candidates": int(self.cap_min_exact_match_candidates),
                    "min_compatible_band_candidates": int(self.cap_min_compatible_band_candidates),
                    "max_exact_match_candidates": int(self.cap_max_exact_match_candidates),
                    "max_compatible_band_candidates": int(self.cap_max_compatible_band_candidates),
                    "use_preliminary_score_for_cap": bool(self.cap_use_preliminary_score_for_cap),
                    "compatibility_penalty_exact": float(self.cap_preliminary_compatibility_penalty_exact),
                    "compatibility_penalty_compatible": float(self.cap_preliminary_compatibility_penalty_compatible),
                    "log_pre_cap_post_cap": bool(self.cap_log_pre_cap_post_cap),
                },
                "compatibility_bands": {
                    "session_nearby_max_hour_distance": float(self.session_nearby_max_hour_distance),
                    "timeframe_nearby_max_minutes_delta": int(self.timeframe_nearby_max_minutes_delta),
                    "timeframe_nearby_max_ratio": float(self.timeframe_nearby_max_ratio),
                    "strategy_type_allow_related": bool(self.strategy_type_allow_related),
                },
            },
            "competition_balance": {
                "enabled": bool(self.family_competition_balance_enabled),
                "dominance_window_size": int(self.dominance_window_size),
                "dominance_penalty_start_share": float(self.dominance_penalty_start_share),
                "dominance_penalty_max": float(self.dominance_penalty_max),
                "max_dominance_penalty": float(self.max_dominance_penalty),
                "low_support_exploration_bonus": float(self.low_support_exploration_bonus),
                "max_exploration_bonus": float(self.max_exploration_bonus),
                "exploration_bonus_decay_threshold": int(self.exploration_bonus_decay_threshold),
                "competition_margin_points": float(self.competition_margin_points),
                "cap_context_advantage_in_close_competition": bool(self.cap_context_advantage_in_close_competition),
                "max_context_advantage_cap": float(self.max_context_advantage_cap),
            },
            "bloat_control": {
                "enable_family_competition_balancing": bool(
                    self.enable_family_competition_balancing
                ),
                "enable_exploration_bonus": bool(self.enable_exploration_bonus),
                "enable_dominance_penalty": bool(self.enable_dominance_penalty),
                "enable_monopoly_canonical_force": bool(
                    self.enable_monopoly_canonical_force
                ),
                "enable_compatibility_tier_slot_pressure": bool(
                    self.enable_compatibility_tier_slot_pressure
                ),
            },
            "runtime_mode_summary": {
                "mode": str(self.core_runtime_mode),
                "core_family_ids": list(self.core_anchor_family_ids),
                "retained_satellite_family_ids": [],
                "suppressed_satellite_family_ids": [],
                "force_anchor_when_eligible": bool(self.force_anchor_when_eligible),
            },
            "usability_competition": {
                "low_support_fully_competitive": bool(self.usability_low_support_fully_competitive),
                "exclude_only_suppressed_families": bool(self.usability_exclude_only_suppressed),
            },
            "evidence_model": {
                "min_mid_samples": int(self.evidence_mid_samples),
                "strong_samples": int(self.evidence_strong_samples),
                "fallback_only_penalty": float(self.fallback_only_penalty),
                "suppressed_adjustment": float(self.suppressed_adjustment),
                "low_tier_min_adjustment": float(self.low_tier_min_adjustment),
                "context_scale_by_evidence_tier": dict(self.context_scale_by_evidence_tier),
            },
            "active_context_dimensions": list(self.active_context_dimensions),
            "context_trust": {
                "min_context_bucket_samples": int(self.min_context_bucket_samples),
                "strong_context_bucket_samples": int(self.strong_context_bucket_samples),
                "context_profile_weight_strong": float(self.context_profile_weight_strong),
                "context_profile_weight_mid": float(self.context_profile_weight_mid),
                "context_profile_weight_low_or_none": float(self.context_profile_weight_low_or_none),
                "require_meaningful_context_support_for_context_weight": bool(
                    self.require_meaningful_context_support_for_context_weight
                ),
                "low_support_context_weight_cap": float(self.low_support_context_weight_cap),
            },
            "family_scoring_controls": {
                "normalize_prior_component": bool(self.fs_normalize_prior_component),
                "cap_context_advantage_when_single_strong_family": bool(
                    self.fs_cap_context_advantage_when_single_strong_family
                ),
                "single_strong_family_context_cap": float(
                    self.fs_single_strong_family_context_cap
                ),
                "compatible_band_penalty": float(self.fs_compatible_band_penalty),
                "close_competition_margin": float(self.fs_close_competition_margin),
                "max_competition_adjustment_close": float(
                    self.fs_max_competition_adjustment_close
                ),
                "max_competition_adjustment_far": float(
                    self.fs_max_competition_adjustment_far
                ),
                "dominance_penalty_curve": str(self.fs_dominance_penalty_curve),
                "exploration_bonus_curve": str(self.fs_exploration_bonus_curve),
                "log_score_delta_ladder": bool(self.fs_log_score_delta_ladder),
            },
            "local_bracket_freeze": {
                "freeze_to_canonical_when_low_support": bool(self.local_freeze_to_canonical_low),
                "force_canonical_when_family_monopoly": bool(self.force_canonical_when_family_monopoly),
                "monopoly_share_threshold": float(self.monopoly_share_threshold),
                "monopoly_lookback_window": int(self.monopoly_lookback_window),
                "context_bracket_weight_scale_mid": float(self.local_context_scale_mid),
                "context_bracket_weight_scale_low": float(self.local_context_scale_low),
                "full_adaptation_min_support_tier": str(self.local_full_adaptation_min_support_tier),
                "conservative_adaptation_min_support_tier": str(self.local_conservative_adaptation_min_support_tier),
            },
            "observability": {
                "enabled": bool(self.observability_enabled),
                "emit_family_score_trace": bool(self.observability_emit_family_score_trace),
                "emit_member_resolution_audit": bool(
                    self.observability_emit_member_resolution_audit
                ),
                "emit_choice_path_audit": bool(self.observability_emit_choice_path_audit),
                "emit_score_path_audit": bool(self.observability_emit_score_path_audit),
                "strict_score_path_assertions": bool(
                    self.observability_strict_score_path_assertions
                ),
                "family_score_trace_max_rows": int(
                    self.observability_family_score_trace_max_rows
                ),
                "member_resolution_trace_max_rows": int(
                    self.observability_member_resolution_max_rows
                ),
            },
        }
        self._load_or_build_family_inventory()

    @staticmethod
    def _init_path_counters() -> Dict[str, int]:
        return {
            "runtime_invocations": 0,
            "refined_universe_loaded_count": 0,
            "raw_universe_fallback_count": 0,
            "family_first_candidate_construction_count": 0,
            "member_first_fallback_count": 0,
            "suppressed_family_skip_count": 0,
            "prior_ineligible_skip_count": 0,
            "context_profile_used_count": 0,
            "context_profile_fallback_to_priors_count": 0,
            "strong_support_decision_count": 0,
            "mid_support_decision_count": 0,
            "low_or_none_support_decision_count": 0,
            "local_bracket_full_mode_count": 0,
            "local_bracket_conservative_mode_count": 0,
            "local_bracket_frozen_mode_count": 0,
            "monopoly_canonical_force_count": 0,
            "exploration_bonus_applied_count": 0,
            "dominance_penalty_applied_count": 0,
            "context_advantage_capped_count": 0,
            "close_competition_decision_count": 0,
            "family_candidate_set_size_eq_1_count": 0,
            "family_candidate_set_size_gt_1_count": 0,
            "score_path_inconsistency_warning_count": 0,
            "decisions_chosen_by_score_ranking_count": 0,
            "decisions_chosen_by_single_candidate_count": 0,
            "decisions_chosen_by_fallback_default_count": 0,
            "decisions_with_score_export_inconsistency_count": 0,
            "decisions_with_multiple_candidates_but_zero_score_delta_count": 0,
            "bootstrap_floor_inclusion_count": 0,
            "family_first_candidate_set_empty_count": 0,
            "local_member_resolution_failed_count": 0,
            "anchor_member_used_count": 0,
            "non_anchor_member_used_count": 0,
            "support_tier_none_count": 0,
            "support_tier_low_count": 0,
            "support_tier_mid_count": 0,
            "support_tier_strong_count": 0,
            "retained_families_total_last": 0,
            "retained_families_scanned_total": 0,
            "retained_families_eligible_total": 0,
            "retained_families_excluded_total": 0,
            "retained_families_unscanned_total": 0,
            "exact_match_eligible_count": 0,
            "compatible_band_eligible_count": 0,
            "incompatible_excluded_count": 0,
            "temporary_exclusion_skip_count": 0,
            "candidate_cap_excluded_count": 0,
            "pre_cap_candidate_total": 0,
            "post_cap_candidate_total": 0,
            "pre_cap_exact_eligible_total": 0,
            "pre_cap_compatible_eligible_total": 0,
            "post_cap_exact_survived_total": 0,
            "post_cap_compatible_survived_total": 0,
            "compatible_dropped_by_cap_total": 0,
            "decisions_with_compatible_pre_cap_all_dropped_count": 0,
            "decisions_with_cap_applied_count": 0,
        }

    @staticmethod
    def _init_score_path_audit() -> Dict[str, Any]:
        return {
            "decision_count": 0,
            "warning_count": 0,
            "warnings": [],
            "summary": {
                "multi_candidate_decisions": 0,
                "multi_candidate_all_zero_final_score_count": 0,
                "multi_candidate_all_equal_final_score_count": 0,
                "multiple_candidates_but_zero_score_delta_count": 0,
                "adjustments_applied_but_component_zero_count": 0,
            },
            "rule_violations": {},
        }

    @staticmethod
    def _init_choice_path_audit() -> Dict[str, Any]:
        return {
            "decision_count": 0,
            "decisions_chosen_by_score_ranking": 0,
            "decisions_chosen_by_single_candidate": 0,
            "decisions_chosen_by_fallback_default": 0,
            "decisions_chosen_by_canonical_force": 0,
            "decisions_with_score_export_inconsistency": 0,
            "decisions_with_multiple_candidates_but_zero_score_delta": 0,
            "decisions_where_local_member_resolution_failed": 0,
            "decisions_where_family_first_candidate_set_was_empty": 0,
        }

    def _bump_counter(self, key: str, inc: int = 1) -> None:
        if key not in self._path_counters:
            self._path_counters[key] = 0
        self._path_counters[key] = int(self._path_counters.get(key, 0) + int(inc))

    def _record_candidate_size_histogram(self, candidate_count: int) -> None:
        key = str(max(0, int(candidate_count)))
        self._family_candidate_size_histogram[key] = int(self._family_candidate_size_histogram.get(key, 0) + 1)

    @staticmethod
    def _resolve_path(raw_path: Any) -> Path:
        out = Path(str(raw_path or "").strip())
        if not out.is_absolute():
            out = Path(__file__).resolve().parent / out
        return out

    @staticmethod
    def _clip(value: float, lo: float, hi: float) -> float:
        return float(max(lo, min(hi, value)))

    @staticmethod
    def _squash_component(value: float, scale: float = 1.0) -> float:
        safe_scale = max(1e-6, float(scale))
        if not math.isfinite(float(value)):
            return 0.0
        return float(math.tanh(float(value) / safe_scale))

    @staticmethod
    def _curve_value(progress: float, curve: str) -> float:
        p = float(max(0.0, min(1.0, progress)))
        c = str(curve or "").strip().lower()
        if c in {"quadratic", "square"}:
            return float(p * p)
        if c in {"sqrt", "square_root"}:
            return float(math.sqrt(p))
        if c in {"cubic"}:
            return float(p * p * p)
        if c in {"quadratic_decay"}:
            # High when progress is low, quickly decays as progress grows.
            return float(max(0.0, 1.0 - (p * p)))
        return float(p)

    @staticmethod
    def _support_tier_rank(tier: Any) -> int:
        raw = str(tier or "").strip().lower()
        if raw == "strong":
            return 2
        if raw == "mid":
            return 1
        return 0

    @staticmethod
    def _legacy_state_to_competition_status(value: Any) -> str:
        raw = str(value or "").strip().lower()
        if raw == "suppressed":
            return "suppressed"
        if raw == "fallback_only":
            return "fallback_only"
        if raw in {"active", "low_support"}:
            return "competitive"
        return "competitive"

    @staticmethod
    def _legacy_state_to_evidence_tier(value: Any) -> str:
        raw = str(value or "").strip().lower()
        if raw == "active":
            return "mid"
        if raw == "low_support":
            return "low"
        if raw == "fallback_only":
            return "mid"
        if raw == "suppressed":
            return "low"
        return "none"

    def _load_or_build_family_inventory(self) -> None:
        if not self.enabled:
            return
        auto_built = False
        load_error = ""
        if not self.family_db_path.exists() and self.auto_build_family_db:
            try:
                name_lower = str(self.family_db_path.name or "").strip().lower()
                use_bundle_auto_build = ("bundle" in name_lower)
                if use_bundle_auto_build:
                    from de3_v3_pipeline import build_and_write_de3_v3_bundle

                    build_and_write_de3_v3_bundle(
                        source_v2_path=self.member_db_path,
                        out_bundle_path=self.family_db_path,
                        decisions_csv_path=self.cp_decisions_path,
                        trade_attribution_csv_path=self.cp_trade_attr_path,
                        out_families_path=None,
                        write_legacy_family_artifact=False,
                        emit_aux_reports=False,
                        min_bucket_samples=int(self.cp_min_bucket_samples),
                        strong_bucket_samples=int(self.cp_strong_bucket_samples),
                        context_profiles_enabled=bool(self.context_profiles_enabled),
                        allow_parse_legacy_context_inputs=bool(self.cp_allow_legacy_parse),
                        mode="full",
                    )
                else:
                    build_and_write_de3_v3_family_inventory(
                        source_v2_path=self.member_db_path,
                        out_path=self.family_db_path,
                        decision_csv_path=self.cp_decisions_path,
                        trade_attribution_csv_path=self.cp_trade_attr_path,
                        min_bucket_samples=self.cp_min_bucket_samples,
                        strong_bucket_samples=self.cp_strong_bucket_samples,
                        context_profiles_enabled=self.context_profiles_enabled,
                        allow_parse_legacy_context_inputs=self.cp_allow_legacy_parse,
                    )
                auto_built = True
            except Exception as exc:
                load_error = f"auto_build_failed:{exc}"
                logging.warning("DE3 v3 family auto-build failed: %s", exc)

        if self.family_db_path.exists():
            try:
                bundle_inspect = self._inspect_bundle_sections(self.family_db_path)
                self._bundle_sections_present = list(bundle_inspect.get("sections_present", []))
                self._bundle_section_sizes = dict(bundle_inspect.get("section_sizes", {}))
                self._artifact_fingerprint = self._file_fingerprint(self.family_db_path)
                self._bundle_usage_audit_cache = None
                self._config_usage_audit_cache = None
                payload = load_de3_v3_family_inventory(
                    self.family_db_path,
                    prefer_refined=bool(self.runtime_use_refined),
                )
                families = payload.get("families") if isinstance(payload.get("families"), list) else []
                self.family_inventory = payload
                self.families_by_id = {
                    str(item.get("family_id", "")): item
                    for item in families
                    if isinstance(item, dict) and str(item.get("family_id", "")).strip()
                }
                runtime_core_sat = (
                    payload.get("_bundle_runtime_core_satellite_state", {})
                    if isinstance(payload.get("_bundle_runtime_core_satellite_state"), dict)
                    else {}
                )
                core_ids_loaded_raw = (
                    runtime_core_sat.get("core_family_ids", [])
                    if isinstance(runtime_core_sat.get("core_family_ids"), (list, tuple, set))
                    else []
                )
                core_ids_loaded = {
                    str(v).strip()
                    for v in core_ids_loaded_raw
                    if str(v).strip()
                }
                if not core_ids_loaded:
                    core_ids_loaded = set(self.core_anchor_family_ids)
                sat_retained_raw = (
                    runtime_core_sat.get("retained_satellite_family_ids", [])
                    if isinstance(runtime_core_sat.get("retained_satellite_family_ids"), (list, tuple, set))
                    else []
                )
                sat_suppressed_raw = (
                    runtime_core_sat.get("suppressed_satellite_family_ids", [])
                    if isinstance(runtime_core_sat.get("suppressed_satellite_family_ids"), (list, tuple, set))
                    else []
                )
                self.core_family_ids_loaded = set(core_ids_loaded)
                self.satellite_retained_family_ids_loaded = {
                    str(v).strip()
                    for v in sat_retained_raw
                    if str(v).strip()
                }
                self.satellite_suppressed_family_ids_loaded = {
                    str(v).strip()
                    for v in sat_suppressed_raw
                    if str(v).strip()
                }
                self.core_satellite_state = dict(runtime_core_sat)
                cp_meta = payload.get("family_context_profile_build", {}) if isinstance(payload.get("family_context_profile_build"), dict) else {}
                runtime_meta = payload.get("family_runtime_state_build", {}) if isinstance(payload.get("family_runtime_state_build"), dict) else {}
                audit = cp_meta.get("audit", {}) if isinstance(cp_meta.get("audit"), dict) else {}
                runtime_state_meta = (
                    runtime_meta.get("runtime_state_meta", {})
                    if isinstance(runtime_meta.get("runtime_state_meta"), dict)
                    else {}
                )
                artifact_kind = str(payload.get("_artifact_kind", "family_inventory") or "family_inventory")
                bundle_meta = payload.get("_bundle_metadata", {}) if isinstance(payload.get("_bundle_metadata"), dict) else {}
                bundle_version = str(payload.get("_bundle_version", "") or "")
                loaded_universe = str(payload.get("_bundle_selected_universe", "raw") or "raw")
                if self.runtime_use_refined and loaded_universe in {"retained_runtime", "refined", "refined_retained_filter"}:
                    self._bump_counter("refined_universe_loaded_count", 1)
                elif self.runtime_use_refined:
                    self._bump_counter("raw_universe_fallback_count", 1)
                refinement_summary = (
                    payload.get("_bundle_refinement_summary", {})
                    if isinstance(payload.get("_bundle_refinement_summary"), dict)
                    else {}
                )
                raw_family_count = int(
                    safe_float(
                        refinement_summary.get(
                            "raw_family_count",
                            payload.get("_bundle_raw_family_count", len(families)),
                        ),
                        len(families),
                    )
                )
                retained_family_count = int(
                    safe_float(
                        refinement_summary.get(
                            "retained_family_count",
                            payload.get("_bundle_refined_family_count", len(families)),
                        ),
                        len(families),
                    )
                )
                suppressed_family_count = int(
                    safe_float(
                        refinement_summary.get("suppressed_family_count", max(0, raw_family_count - retained_family_count)),
                        max(0, raw_family_count - retained_family_count),
                    )
                )
                raw_member_count = int(
                    safe_float(
                        refinement_summary.get(
                            "raw_member_count",
                            sum(
                                int(safe_float((row or {}).get("raw_member_count", (row or {}).get("member_count", 0)), 0))
                                for row in families
                                if isinstance(row, dict)
                            ),
                        ),
                        0,
                    )
                )
                retained_member_count = int(
                    safe_float(
                        refinement_summary.get(
                            "retained_member_count",
                            sum(
                                int(safe_float((row or {}).get("member_count", 0), 0))
                                for row in families
                                if isinstance(row, dict)
                            ),
                        ),
                        0,
                    )
                )
                suppressed_member_count = int(
                    safe_float(
                        refinement_summary.get("suppressed_member_count", max(0, raw_member_count - retained_member_count)),
                        max(0, raw_member_count - retained_member_count),
                    )
                )
                anchor_member_count = int(
                    safe_float(
                        refinement_summary.get(
                            "anchor_member_count",
                            sum(
                                1
                                for row in families
                                if isinstance(row, dict)
                                and str(((row.get("canonical_representative_member") or {}).get("member_id", "")).strip())
                            ),
                        ),
                        0,
                    )
                )
                context_loaded = bool(cp_meta.get("enabled", False)) and any(
                    bool((row.get("family_context_profiles", {}).get("_meta", {}) or {}).get("has_profile_data", False))
                    for row in self.families_by_id.values()
                    if isinstance(row, dict)
                )
                core_family_count = int(
                    sum(
                        1
                        for fid in self.families_by_id.keys()
                        if str(fid or "").strip() in set(self.core_family_ids_loaded)
                    )
                )
                satellite_family_count = int(max(0, len(self.families_by_id) - core_family_count))
                runtime_state_loaded = bool(str(runtime_state_meta.get("status", "")).lower() == "ok")
                self.last_load_status = {
                    "enabled": bool(self.enabled),
                    "family_artifact_path": str(self.family_db_path),
                    "artifact_kind": str(artifact_kind),
                    "bundle_loaded": bool(artifact_kind == "bundle"),
                    "bundle_version": str(bundle_version),
                    "bundle_build_timestamp": bundle_meta.get("build_timestamp"),
                    "artifact_fingerprint": dict(self._artifact_fingerprint),
                    "bundle_sections_present": list(self._bundle_sections_present),
                    "bundle_section_sizes": dict(self._bundle_section_sizes),
                    "runtime_use_refined": bool(self.runtime_use_refined),
                    "core_enabled": bool(self.core_enabled),
                    "core_runtime_mode": str(self.core_runtime_mode),
                    "core_anchor_family_ids": list(self.core_anchor_family_ids),
                    "core_family_count": int(core_family_count),
                    "satellite_family_count": int(satellite_family_count),
                    "active_runtime_family_count": int(len(self._runtime_universe_family_items())),
                    "core_satellite_state_loaded": bool(runtime_core_sat),
                    "loaded_universe": str(loaded_universe),
                    "raw_family_count": int(raw_family_count),
                    "retained_family_count": int(retained_family_count),
                    "suppressed_family_count": int(suppressed_family_count),
                    "raw_member_count": int(raw_member_count),
                    "retained_member_count": int(retained_member_count),
                    "suppressed_member_count": int(suppressed_member_count),
                    "anchor_member_count": int(anchor_member_count),
                    "family_artifact_loaded": True,
                    "family_count": int(len(self.families_by_id)),
                    "auto_built": bool(auto_built),
                    "load_error": str(load_error or ""),
                    "context_profiles_loaded": bool(context_loaded),
                    "enriched_export_required": bool(audit.get("enriched_export_required_for_full_bucketing", False)),
                    "context_profile_build": cp_meta,
                    "runtime_state_build": runtime_meta,
                    "runtime_state_loaded": bool(runtime_state_loaded),
                    "fallback_to_priors_enabled": bool(self.fallback_to_priors_when_profile_weak),
                    "prior_eligibility_enabled": bool(self.prior_eligibility_enabled),
                    "competition_floor": {
                        "enabled": bool(self.use_bootstrap_family_competition_floor),
                        "bootstrap_min_competing_families": int(self.bootstrap_min_competing_families),
                        "include_exact_and_compatible_only": bool(self.include_exact_and_compatible_only),
                        "max_family_candidates_per_decision": int(self.max_family_candidates_per_decision),
                        "compatible_family_max_count": int(self.compatible_family_max_count),
                        "compatible_family_penalty": float(self.compatible_family_penalty),
                        "family_candidate_cap": {
                            "enabled": bool(self.family_candidate_cap_enabled),
                            "max_total_candidates": int(self.cap_max_total_candidates),
                            "min_exact_match_candidates": int(self.cap_min_exact_match_candidates),
                            "min_compatible_band_candidates": int(self.cap_min_compatible_band_candidates),
                            "max_exact_match_candidates": int(self.cap_max_exact_match_candidates),
                            "max_compatible_band_candidates": int(self.cap_max_compatible_band_candidates),
                            "use_preliminary_score_for_cap": bool(self.cap_use_preliminary_score_for_cap),
                            "compatibility_penalty_exact": float(self.cap_preliminary_compatibility_penalty_exact),
                            "compatibility_penalty_compatible": float(self.cap_preliminary_compatibility_penalty_compatible),
                            "log_pre_cap_post_cap": bool(self.cap_log_pre_cap_post_cap),
                        },
                        "compatibility_bands": {
                            "session_nearby_max_hour_distance": float(self.session_nearby_max_hour_distance),
                            "timeframe_nearby_max_minutes_delta": int(self.timeframe_nearby_max_minutes_delta),
                            "timeframe_nearby_max_ratio": float(self.timeframe_nearby_max_ratio),
                            "strategy_type_allow_related": bool(self.strategy_type_allow_related),
                        },
                    },
                    "competition_balance": {
                        "enabled": bool(self.family_competition_balance_enabled),
                        "dominance_window_size": int(self.dominance_window_size),
                        "dominance_penalty_start_share": float(self.dominance_penalty_start_share),
                        "dominance_penalty_max": float(self.dominance_penalty_max),
                        "max_dominance_penalty": float(self.max_dominance_penalty),
                        "low_support_exploration_bonus": float(self.low_support_exploration_bonus),
                        "max_exploration_bonus": float(self.max_exploration_bonus),
                        "exploration_bonus_decay_threshold": int(self.exploration_bonus_decay_threshold),
                        "competition_margin_points": float(self.competition_margin_points),
                        "cap_context_advantage_in_close_competition": bool(self.cap_context_advantage_in_close_competition),
                        "max_context_advantage_cap": float(self.max_context_advantage_cap),
                    },
                    "bloat_control": {
                        "enable_family_competition_balancing": bool(
                            self.enable_family_competition_balancing
                        ),
                        "enable_exploration_bonus": bool(
                            self.enable_exploration_bonus
                        ),
                        "enable_dominance_penalty": bool(
                            self.enable_dominance_penalty
                        ),
                        "enable_monopoly_canonical_force": bool(
                            self.enable_monopoly_canonical_force
                        ),
                        "enable_compatibility_tier_slot_pressure": bool(
                            self.enable_compatibility_tier_slot_pressure
                        ),
                    },
                    "runtime_mode_summary": {
                        "mode": str(self.core_runtime_mode),
                        "core_family_ids": sorted(list(self.core_family_ids_loaded)),
                        "retained_satellite_family_ids": sorted(
                            list(self.satellite_retained_family_ids_loaded)
                        ),
                        "suppressed_satellite_family_ids": sorted(
                            list(self.satellite_suppressed_family_ids_loaded)
                        ),
                        "force_anchor_when_eligible": bool(
                            self.force_anchor_when_eligible
                        ),
                    },
                    "usability_competition": {
                        "low_support_fully_competitive": bool(self.usability_low_support_fully_competitive),
                        "exclude_only_suppressed_families": bool(self.usability_exclude_only_suppressed),
                    },
                    "evidence_model": {
                        "min_mid_samples": int(self.evidence_mid_samples),
                        "strong_samples": int(self.evidence_strong_samples),
                        "fallback_only_penalty": float(self.fallback_only_penalty),
                        "suppressed_adjustment": float(self.suppressed_adjustment),
                        "low_tier_min_adjustment": float(self.low_tier_min_adjustment),
                        "context_scale_by_evidence_tier": dict(self.context_scale_by_evidence_tier),
                    },
                    "active_context_dimensions": list(self.active_context_dimensions),
                    "context_trust": {
                        "min_context_bucket_samples": int(self.min_context_bucket_samples),
                        "strong_context_bucket_samples": int(self.strong_context_bucket_samples),
                        "context_profile_weight_strong": float(self.context_profile_weight_strong),
                        "context_profile_weight_mid": float(self.context_profile_weight_mid),
                        "context_profile_weight_low_or_none": float(self.context_profile_weight_low_or_none),
                        "require_meaningful_context_support_for_context_weight": bool(
                            self.require_meaningful_context_support_for_context_weight
                        ),
                        "low_support_context_weight_cap": float(self.low_support_context_weight_cap),
                    },
                    "local_bracket_freeze": {
                        "freeze_to_canonical_when_low_support": bool(self.local_freeze_to_canonical_low),
                        "force_canonical_when_family_monopoly": bool(self.force_canonical_when_family_monopoly),
                        "monopoly_share_threshold": float(self.monopoly_share_threshold),
                        "monopoly_lookback_window": int(self.monopoly_lookback_window),
                        "context_bracket_weight_scale_mid": float(self.local_context_scale_mid),
                        "context_bracket_weight_scale_low": float(self.local_context_scale_low),
                        "full_adaptation_min_support_tier": str(self.local_full_adaptation_min_support_tier),
                        "conservative_adaptation_min_support_tier": str(self.local_conservative_adaptation_min_support_tier),
                    },
                }
                return
            except Exception as exc:
                load_error = f"load_failed:{exc}"
                logging.warning("DE3 v3 family inventory load failed (%s): %s", self.family_db_path, exc)

        self.family_inventory = {}
        self.families_by_id = {}
        self._bundle_sections_present = []
        self._bundle_section_sizes = {}
        self._artifact_fingerprint = self._file_fingerprint(self.family_db_path)
        self._bundle_usage_audit_cache = None
        self._config_usage_audit_cache = None
        self.last_load_status = {
            "enabled": bool(self.enabled),
            "family_artifact_path": str(self.family_db_path),
            "artifact_kind": "family_inventory",
            "bundle_loaded": False,
            "bundle_version": "",
            "bundle_build_timestamp": None,
            "artifact_fingerprint": dict(self._artifact_fingerprint),
            "bundle_sections_present": [],
            "bundle_section_sizes": {},
            "runtime_use_refined": bool(self.runtime_use_refined),
            "core_enabled": bool(self.core_enabled),
            "core_runtime_mode": str(self.core_runtime_mode),
            "core_anchor_family_ids": list(self.core_anchor_family_ids),
            "core_family_count": 0,
            "satellite_family_count": 0,
            "active_runtime_family_count": 0,
            "core_satellite_state_loaded": False,
            "loaded_universe": "raw",
            "raw_family_count": 0,
            "retained_family_count": 0,
            "suppressed_family_count": 0,
            "raw_member_count": 0,
            "retained_member_count": 0,
            "suppressed_member_count": 0,
            "anchor_member_count": 0,
            "family_artifact_loaded": False,
            "family_count": 0,
            "auto_built": bool(auto_built),
            "load_error": str(load_error or "missing_family_artifact"),
            "context_profiles_loaded": False,
            "enriched_export_required": False,
            "runtime_state_loaded": False,
            "fallback_to_priors_enabled": bool(self.fallback_to_priors_when_profile_weak),
            "prior_eligibility_enabled": bool(self.prior_eligibility_enabled),
            "competition_floor": {
                "enabled": bool(self.use_bootstrap_family_competition_floor),
                "bootstrap_min_competing_families": int(self.bootstrap_min_competing_families),
                "include_exact_and_compatible_only": bool(self.include_exact_and_compatible_only),
                "max_family_candidates_per_decision": int(self.max_family_candidates_per_decision),
                "compatible_family_max_count": int(self.compatible_family_max_count),
                "compatible_family_penalty": float(self.compatible_family_penalty),
                "family_candidate_cap": {
                    "enabled": bool(self.family_candidate_cap_enabled),
                    "max_total_candidates": int(self.cap_max_total_candidates),
                    "min_exact_match_candidates": int(self.cap_min_exact_match_candidates),
                    "min_compatible_band_candidates": int(self.cap_min_compatible_band_candidates),
                    "max_exact_match_candidates": int(self.cap_max_exact_match_candidates),
                    "max_compatible_band_candidates": int(self.cap_max_compatible_band_candidates),
                    "use_preliminary_score_for_cap": bool(self.cap_use_preliminary_score_for_cap),
                    "compatibility_penalty_exact": float(self.cap_preliminary_compatibility_penalty_exact),
                    "compatibility_penalty_compatible": float(self.cap_preliminary_compatibility_penalty_compatible),
                    "log_pre_cap_post_cap": bool(self.cap_log_pre_cap_post_cap),
                },
                "compatibility_bands": {
                    "session_nearby_max_hour_distance": float(self.session_nearby_max_hour_distance),
                    "timeframe_nearby_max_minutes_delta": int(self.timeframe_nearby_max_minutes_delta),
                    "timeframe_nearby_max_ratio": float(self.timeframe_nearby_max_ratio),
                    "strategy_type_allow_related": bool(self.strategy_type_allow_related),
                },
            },
            "competition_balance": {
                "enabled": bool(self.family_competition_balance_enabled),
                "dominance_window_size": int(self.dominance_window_size),
                "dominance_penalty_start_share": float(self.dominance_penalty_start_share),
                "dominance_penalty_max": float(self.dominance_penalty_max),
                "max_dominance_penalty": float(self.max_dominance_penalty),
                "low_support_exploration_bonus": float(self.low_support_exploration_bonus),
                "max_exploration_bonus": float(self.max_exploration_bonus),
                "exploration_bonus_decay_threshold": int(self.exploration_bonus_decay_threshold),
                "competition_margin_points": float(self.competition_margin_points),
                "cap_context_advantage_in_close_competition": bool(self.cap_context_advantage_in_close_competition),
                "max_context_advantage_cap": float(self.max_context_advantage_cap),
            },
            "bloat_control": {
                "enable_family_competition_balancing": bool(
                    self.enable_family_competition_balancing
                ),
                "enable_exploration_bonus": bool(self.enable_exploration_bonus),
                "enable_dominance_penalty": bool(self.enable_dominance_penalty),
                "enable_monopoly_canonical_force": bool(
                    self.enable_monopoly_canonical_force
                ),
                "enable_compatibility_tier_slot_pressure": bool(
                    self.enable_compatibility_tier_slot_pressure
                ),
            },
            "runtime_mode_summary": {
                "mode": str(self.core_runtime_mode),
                "core_family_ids": sorted(list(self.core_family_ids_loaded)),
                "retained_satellite_family_ids": sorted(
                    list(self.satellite_retained_family_ids_loaded)
                ),
                "suppressed_satellite_family_ids": sorted(
                    list(self.satellite_suppressed_family_ids_loaded)
                ),
                "force_anchor_when_eligible": bool(self.force_anchor_when_eligible),
            },
            "usability_competition": {
                "low_support_fully_competitive": bool(self.usability_low_support_fully_competitive),
                "exclude_only_suppressed_families": bool(self.usability_exclude_only_suppressed),
            },
            "evidence_model": {
                "min_mid_samples": int(self.evidence_mid_samples),
                "strong_samples": int(self.evidence_strong_samples),
                "fallback_only_penalty": float(self.fallback_only_penalty),
                "suppressed_adjustment": float(self.suppressed_adjustment),
                "low_tier_min_adjustment": float(self.low_tier_min_adjustment),
                "context_scale_by_evidence_tier": dict(self.context_scale_by_evidence_tier),
            },
            "active_context_dimensions": list(self.active_context_dimensions),
            "context_trust": {
                "min_context_bucket_samples": int(self.min_context_bucket_samples),
                "strong_context_bucket_samples": int(self.strong_context_bucket_samples),
                "context_profile_weight_strong": float(self.context_profile_weight_strong),
                "context_profile_weight_mid": float(self.context_profile_weight_mid),
                "context_profile_weight_low_or_none": float(self.context_profile_weight_low_or_none),
                "require_meaningful_context_support_for_context_weight": bool(
                    self.require_meaningful_context_support_for_context_weight
                ),
                "low_support_context_weight_cap": float(self.low_support_context_weight_cap),
            },
            "local_bracket_freeze": {
                "freeze_to_canonical_when_low_support": bool(self.local_freeze_to_canonical_low),
                "force_canonical_when_family_monopoly": bool(self.force_canonical_when_family_monopoly),
                "monopoly_share_threshold": float(self.monopoly_share_threshold),
                "monopoly_lookback_window": int(self.monopoly_lookback_window),
                "context_bracket_weight_scale_mid": float(self.local_context_scale_mid),
                "context_bracket_weight_scale_low": float(self.local_context_scale_low),
                "full_adaptation_min_support_tier": str(self.local_full_adaptation_min_support_tier),
                "conservative_adaptation_min_support_tier": str(self.local_conservative_adaptation_min_support_tier),
            },
        }

    def get_runtime_status(self) -> Dict[str, Any]:
        out = dict(self.last_load_status)
        out["runtime_path_counters"] = dict(self._path_counters)
        out["score_path_audit"] = self.get_score_path_audit()
        out["choice_path_audit"] = self.get_choice_path_audit()
        out["bundle_sections_present"] = list(self._bundle_sections_present)
        out["bundle_section_sizes"] = dict(self._bundle_section_sizes)
        out["artifact_fingerprint"] = dict(self._artifact_fingerprint)
        out["family_score_trace_rows"] = int(len(self._family_score_trace_rows))
        out["member_resolution_trace_rows"] = int(len(self._member_resolution_trace_rows))
        out["family_eligibility_trace_rows"] = int(len(self._family_eligibility_trace_rows))
        out["pre_cap_candidate_audit_rows"] = int(len(self._pre_cap_candidate_audit_rows))
        return out

    def get_runtime_mode_summary(self) -> Dict[str, Any]:
        bundle_mode = (
            self.family_inventory.get("_bundle_runtime_mode_summary", {})
            if isinstance(self.family_inventory.get("_bundle_runtime_mode_summary"), dict)
            else {}
        )
        if bundle_mode:
            base = dict(bundle_mode)
        else:
            base = {}
        active_items = self._runtime_universe_family_items()
        active_family_ids = [str(fid) for fid, _ in active_items]
        core_active = [fid for fid in active_family_ids if self._family_role(fid) == "core"]
        sat_active = [fid for fid in active_family_ids if self._family_role(fid) == "satellite"]
        out = {
            "mode": str(self.core_runtime_mode),
            "core_enabled": bool(self.core_enabled),
            "satellites_enabled": bool(self.satellites_enabled),
            "force_anchor_when_eligible": bool(self.force_anchor_when_eligible),
            "core_anchor_family_ids": list(self.core_anchor_family_ids),
            "core_family_ids_loaded": sorted(list(self.core_family_ids_loaded)),
            "retained_satellite_family_ids": sorted(
                list(self.satellite_retained_family_ids_loaded)
            ),
            "suppressed_satellite_family_ids": sorted(
                list(self.satellite_suppressed_family_ids_loaded)
            ),
            "active_runtime_family_ids": list(active_family_ids),
            "active_runtime_family_count": int(len(active_family_ids)),
            "active_core_family_count": int(len(core_active)),
            "active_satellite_family_count": int(len(sat_active)),
            "bloat_control": {
                "enable_family_competition_balancing": bool(
                    self.enable_family_competition_balancing
                ),
                "enable_exploration_bonus": bool(self.enable_exploration_bonus),
                "enable_dominance_penalty": bool(self.enable_dominance_penalty),
                "enable_monopoly_canonical_force": bool(
                    self.enable_monopoly_canonical_force
                ),
                "enable_compatibility_tier_slot_pressure": bool(
                    self.enable_compatibility_tier_slot_pressure
                ),
            },
        }
        base.update(out)
        return base

    def _runtime_metrics_for_family_ids(self, family_ids: List[str]) -> Dict[str, Any]:
        net = 0.0
        trades = 0
        gp = 0.0
        gl = 0.0
        chosen = 0
        ids = {str(fid).strip() for fid in family_ids if str(fid).strip()}
        for fid in ids:
            row = self.families_by_id.get(fid)
            if not isinstance(row, dict):
                continue
            state = self._extract_runtime_state(row)
            metrics = state.get("metrics") if isinstance(state.get("metrics"), dict) else {}
            pnl = float(safe_float(metrics.get("realized_net_pnl", 0.0), 0.0))
            trade_count = int(max(0, safe_float(metrics.get("executed_trade_count", 0), 0)))
            chosen_count = int(max(0, safe_float(metrics.get("chosen_count", 0), 0)))
            trades += trade_count
            chosen += chosen_count
            net += pnl
            if pnl >= 0:
                gp += pnl
            else:
                gl += abs(pnl)
        pf = gp / gl if gl > 1e-12 else (999.0 if gp > 0.0 else 0.0)
        return {
            "family_count": int(len(ids)),
            "chosen_count": int(chosen),
            "trade_count": int(trades),
            "net_pnl": float(net),
            "profit_factor": float(pf),
            "gross_profit": float(gp),
            "gross_loss_abs": float(gl),
        }

    def get_core_summary(self) -> Dict[str, Any]:
        core_ids = sorted(list(self.core_family_ids_loaded or set(self.core_anchor_family_ids)))
        metrics = self._runtime_metrics_for_family_ids(core_ids)
        out = {
            "created_at": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(),
            "runtime_mode": str(self.core_runtime_mode),
            "core_enabled": bool(self.core_enabled),
            "core_family_ids": list(core_ids),
            "core_metrics": dict(metrics),
            "source_section": dict(
                self.family_inventory.get("_bundle_core_families", {})
                if isinstance(self.family_inventory.get("_bundle_core_families"), dict)
                else {}
            ),
        }
        bundle_core_summary = (
            self.family_inventory.get("_bundle_core_summary", {})
            if isinstance(self.family_inventory.get("_bundle_core_summary"), dict)
            else {}
        )
        if bundle_core_summary:
            merged = dict(bundle_core_summary)
            merged.update(out)
            return merged
        return out

    def get_t6_anchor_report(self) -> Dict[str, Any]:
        t6_id = "5min|09-12|long|Long_Rev|T6"
        core_ids = set(self.core_family_ids_loaded or set(self.core_anchor_family_ids))
        row = self.families_by_id.get(t6_id, {})
        state = self._extract_runtime_state(row if isinstance(row, dict) else None)
        metrics = state.get("metrics") if isinstance(state.get("metrics"), dict) else {}
        out = {
            "anchor_family_id": str(t6_id),
            "configured_as_core_anchor": bool(t6_id in core_ids),
            "present_in_runtime_inventory": bool(isinstance(row, dict) and bool(row)),
            "runtime_role": str(self._family_role(t6_id)),
            "runtime_mode": str(self.core_runtime_mode),
            "metrics": dict(metrics),
        }
        bundle_anchor = (
            self.family_inventory.get("_bundle_t6_anchor_report", {})
            if isinstance(self.family_inventory.get("_bundle_t6_anchor_report"), dict)
            else {}
        )
        if bundle_anchor:
            merged = dict(bundle_anchor)
            merged.update(out)
            return merged
        return out

    def get_satellite_quality_report(self) -> Dict[str, Any]:
        section = (
            self.family_inventory.get("_bundle_satellite_quality_summary", {})
            if isinstance(self.family_inventory.get("_bundle_satellite_quality_summary"), dict)
            else {}
        )
        if section:
            return dict(section)
        # Fallback summary when loaded artifact is legacy/non-bundle.
        core_ids = set(self.core_family_ids_loaded or set(self.core_anchor_family_ids))
        rows = []
        for fid, row in sorted(self.families_by_id.items()):
            if fid in core_ids or not isinstance(row, dict):
                continue
            quality = safe_float(row.get("family_quality_score", 0.0), 0.0)
            cls = str(row.get("family_quality_classification", "unknown") or "unknown")
            rows.append(
                {
                    "family_id": str(fid),
                    "standalone_viability_component": float(quality),
                    "satellite_quality_score": float(quality),
                    "satellite_classification": "keep_satellite"
                    if quality >= self.sat_min_standalone_viability
                    else "suppress_satellite",
                    "source_family_quality_classification": str(cls),
                }
            )
        return {
            "created_at": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(),
            "satellite_count": int(len(rows)),
            "satellites": rows,
        }

    def get_portfolio_increment_report(self) -> Dict[str, Any]:
        section = (
            self.family_inventory.get("_bundle_portfolio_incremental_tests", {})
            if isinstance(self.family_inventory.get("_bundle_portfolio_incremental_tests"), dict)
            else {}
        )
        if section:
            return dict(section)
        core_ids = sorted(list(self.core_family_ids_loaded or set(self.core_anchor_family_ids)))
        sat_ids = sorted(
            [
                fid
                for fid in self.families_by_id.keys()
                if str(fid).strip() and fid not in set(core_ids)
            ]
        )
        core_metrics = self._runtime_metrics_for_family_ids(core_ids)
        satellite_rows = []
        for fid in sat_ids:
            sat_metrics = self._runtime_metrics_for_family_ids([fid])
            combined_pf = (
                (core_metrics.get("gross_profit", 0.0) + sat_metrics.get("gross_profit", 0.0))
                / max(
                    1e-12,
                    (core_metrics.get("gross_loss_abs", 0.0) + sat_metrics.get("gross_loss_abs", 0.0)),
                )
            )
            satellite_rows.append(
                {
                    "family_id": str(fid),
                    "delta_net_pnl_vs_core": float(sat_metrics.get("net_pnl", 0.0)),
                    "delta_trade_count_vs_core": int(sat_metrics.get("trade_count", 0)),
                    "delta_profit_factor_vs_core": float(
                        float(combined_pf) - float(core_metrics.get("profit_factor", 0.0))
                    ),
                }
            )
        return {
            "created_at": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(),
            "core_baseline": dict(core_metrics),
            "satellite_increment_tests": satellite_rows,
        }

    def get_runtime_path_counters(self) -> Dict[str, Any]:
        counters = dict(self._path_counters)
        counters["loaded_universe"] = str(self.last_load_status.get("loaded_universe", "raw") or "raw")
        counters["runtime_use_refined"] = bool(self.last_load_status.get("runtime_use_refined", self.runtime_use_refined))
        counters["family_count_loaded"] = int(self.last_load_status.get("family_count", 0) or 0)
        counters["family_candidate_set_size_histogram"] = dict(self._family_candidate_size_histogram)
        counters["family_candidate_count_eq_1"] = int(counters.get("family_candidate_set_size_eq_1_count", 0) or 0)
        counters["family_candidate_count_gt_1"] = int(counters.get("family_candidate_set_size_gt_1_count", 0) or 0)
        counters["chosen_by_single_candidate_count"] = int(counters.get("decisions_chosen_by_single_candidate_count", 0) or 0)
        counters["chosen_by_ranked_score_count"] = int(counters.get("decisions_chosen_by_score_ranking_count", 0) or 0)
        counters["chosen_by_fallback_default_count"] = int(counters.get("decisions_chosen_by_fallback_default_count", 0) or 0)
        counters["support_tier_counts"] = {
            "none": int(counters.get("support_tier_none_count", 0) or 0),
            "low": int(counters.get("support_tier_low_count", 0) or 0),
            "mid": int(counters.get("support_tier_mid_count", 0) or 0),
            "strong": int(counters.get("support_tier_strong_count", 0) or 0),
        }
        counters["compatibility_outcome_counts"] = {
            "exact_match_eligible": int(counters.get("exact_match_eligible_count", 0) or 0),
            "compatible_band_eligible": int(counters.get("compatible_band_eligible_count", 0) or 0),
            "incompatible_excluded": int(counters.get("incompatible_excluded_count", 0) or 0),
            "temporary_exclusion_skipped": int(
                counters.get("temporary_exclusion_skip_count", 0) or 0
            ),
            "candidate_cap_excluded": int(counters.get("candidate_cap_excluded_count", 0) or 0),
        }
        invocations = int(counters.get("runtime_invocations", 0) or 0)
        scanned_total = int(counters.get("retained_families_scanned_total", 0) or 0)
        eligible_total = int(counters.get("retained_families_eligible_total", 0) or 0)
        excluded_total = int(counters.get("retained_families_excluded_total", 0) or 0)
        unscanned_total = int(counters.get("retained_families_unscanned_total", 0) or 0)
        counters["retained_families_total"] = int(
            counters.get("retained_families_total_last", 0) or 0
        )
        counters["retained_families_scanned_per_decision"] = float(
            scanned_total / float(max(1, invocations))
        )
        counters["retained_families_eligible_per_decision"] = float(
            eligible_total / float(max(1, invocations))
        )
        counters["retained_families_excluded_per_decision"] = float(
            excluded_total / float(max(1, invocations))
        )
        counters["retained_families_unscanned_per_decision"] = float(
            unscanned_total / float(max(1, invocations))
        )
        counters["pre_cap_candidate_count_avg"] = float(
            safe_float(counters.get("pre_cap_candidate_total", 0), 0.0) / float(max(1, invocations))
        )
        counters["post_cap_candidate_count_avg"] = float(
            safe_float(counters.get("post_cap_candidate_total", 0), 0.0) / float(max(1, invocations))
        )
        counters["pre_cap_exact_eligible_avg"] = float(
            safe_float(counters.get("pre_cap_exact_eligible_total", 0), 0.0) / float(max(1, invocations))
        )
        counters["pre_cap_compatible_eligible_avg"] = float(
            safe_float(counters.get("pre_cap_compatible_eligible_total", 0), 0.0) / float(max(1, invocations))
        )
        counters["post_cap_exact_survived_avg"] = float(
            safe_float(counters.get("post_cap_exact_survived_total", 0), 0.0) / float(max(1, invocations))
        )
        counters["post_cap_compatible_survived_avg"] = float(
            safe_float(counters.get("post_cap_compatible_survived_total", 0), 0.0) / float(max(1, invocations))
        )
        counters["compatible_dropped_by_cap_avg"] = float(
            safe_float(counters.get("compatible_dropped_by_cap_total", 0), 0.0) / float(max(1, invocations))
        )
        counters["local_bracket_mode_counts"] = {
            "frozen": int(counters.get("local_bracket_frozen_mode_count", 0) or 0),
            "conservative": int(counters.get("local_bracket_conservative_mode_count", 0) or 0),
            "full": int(counters.get("local_bracket_full_mode_count", 0) or 0),
        }
        hist = self._recent_chosen_family_ids
        total_hist = int(len(hist))
        share_inputs = {
            "window_size": int(total_hist),
            "top_family_id": "",
            "top_family_share": 0.0,
            "top_2_family_share": 0.0,
        }
        if total_hist > 0:
            counts: Dict[str, int] = defaultdict(int)
            for fid in hist:
                key = str(fid or "").strip()
                if key:
                    counts[key] = int(counts.get(key, 0) + 1)
            counters["chosen_family_unique_count"] = int(len(counts))
            if counts:
                sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)
                top_family_id, top_count = sorted_counts[0]
                share_inputs["top_family_id"] = str(top_family_id)
                share_inputs["top_family_share"] = float(top_count / float(total_hist))
                top_two = int(top_count)
                if len(sorted_counts) > 1:
                    top_two += int(sorted_counts[1][1])
                share_inputs["top_2_family_share"] = float(top_two / float(total_hist))
        else:
            counters["chosen_family_unique_count"] = 0
        counters["top_family_share_inputs"] = share_inputs
        return counters

    def get_score_path_audit(self) -> Dict[str, Any]:
        out = dict(self._score_path_audit)
        warnings = self._score_path_audit.get("warnings", [])
        out["warnings"] = list(warnings if isinstance(warnings, list) else [])
        out["summary"] = dict(self._score_path_audit.get("summary", {}))
        out["rule_violations"] = dict(self._score_path_audit.get("rule_violations", {}))
        return out

    def get_choice_path_audit(self) -> Dict[str, Any]:
        out = dict(self._choice_path_audit)
        out["decisions_chosen_by_default_fallback"] = int(
            out.get("decisions_chosen_by_fallback_default", 0)
        )
        return out

    def _append_family_score_trace_rows(self, rows: List[Dict[str, Any]]) -> None:
        if (not self.observability_enabled) or (not self.observability_emit_family_score_trace):
            return
        if not rows:
            return
        self._family_score_trace_rows.extend(rows)
        overflow = int(len(self._family_score_trace_rows) - self.observability_family_score_trace_max_rows)
        if overflow > 0:
            del self._family_score_trace_rows[:overflow]

    def _append_member_resolution_trace_row(self, row: Dict[str, Any]) -> None:
        if (not self.observability_enabled) or (not self.observability_emit_member_resolution_audit):
            return
        if not isinstance(row, dict):
            return
        self._member_resolution_trace_rows.append(dict(row))
        overflow = int(len(self._member_resolution_trace_rows) - self.observability_member_resolution_max_rows)
        if overflow > 0:
            del self._member_resolution_trace_rows[:overflow]

    def _append_family_eligibility_trace_rows(self, rows: List[Dict[str, Any]]) -> None:
        if (not self.observability_enabled) or (not self.observability_emit_choice_path_audit):
            return
        if not rows:
            return
        max_rows = int(max(1000, self.observability_family_score_trace_max_rows))
        self._family_eligibility_trace_rows.extend([dict(r) for r in rows if isinstance(r, dict)])
        overflow = int(len(self._family_eligibility_trace_rows) - max_rows)
        if overflow > 0:
            del self._family_eligibility_trace_rows[:overflow]

    def _finalize_family_reachability_observability(
        self,
        *,
        family_rows: List[Dict[str, Any]],
        normalized_context: Dict[str, str],
    ) -> None:
        if not isinstance(family_rows, list) or not family_rows:
            return
        decision_invocation = int(self._path_counters.get("runtime_invocations", 0))
        context_joint = str(
            build_active_context_joint_key(
                normalized_context if isinstance(normalized_context, dict) else {}
            )
            or ""
        )
        trace_rows: List[Dict[str, Any]] = []
        for row in family_rows:
            if not isinstance(row, dict):
                continue
            trace_rows.append(
                {
                    "decision_invocation": int(decision_invocation),
                    "family_id": str(row.get("family_id", "") or ""),
                    "retained_runtime": bool(row.get("retained_runtime", False)),
                    "evaluated_for_eligibility": bool(row.get("evaluated_for_eligibility", False)),
                    "coarse_eligible": bool(row.get("coarse_eligible", False)),
                    "eligible_for_candidate_set": bool(
                        row.get("final_competition_pool_flag", row.get("competition_eligible", False))
                    ),
                    "failure_reason": str(
                        row.get(
                            "eligibility_failure_reason",
                            row.get("competition_eligibility_reason", ""),
                        )
                        or ""
                    ),
                    "coarse_failure_reason": str(
                        row.get("coarse_eligibility_failure_reason", "") or ""
                    ),
                    "competition_status": str(
                        row.get(
                            "competition_status",
                            row.get("family_competition_status", "competitive"),
                        )
                        or "competitive"
                    ),
                    "competition_eligibility_reason": str(
                        row.get("competition_eligibility_reason", "") or ""
                    ),
                    "excluded_by_session_mismatch": bool(
                        row.get("excluded_by_session_mismatch", False)
                    ),
                    "excluded_by_side_mismatch": bool(
                        row.get("excluded_by_side_mismatch", False)
                    ),
                    "excluded_by_timeframe_mismatch": bool(
                        row.get("excluded_by_timeframe_mismatch", False)
                    ),
                    "excluded_by_strategy_type_mismatch": bool(
                        row.get("excluded_by_strategy_type_mismatch", False)
                    ),
                    "excluded_by_context_gate": bool(
                        row.get("excluded_by_context_gate", False)
                    ),
                    "excluded_by_adaptive_policy_gate": bool(
                        row.get("excluded_by_adaptive_policy_gate", False)
                    ),
                    "excluded_by_no_local_member_available": bool(
                        row.get("excluded_by_no_local_member_available", False)
                    ),
                    "excluded_by_temporary_exclusion": bool(
                        row.get("excluded_by_temporary_exclusion", False)
                    ),
                    "excluded_by_candidate_cap": bool(
                        row.get("excluded_by_candidate_cap", False)
                    ),
                    "compatibility_tier": str(
                        row.get("compatibility_tier", "") or ""
                    ),
                    "session_compatibility_tier": str(
                        row.get("session_compatibility_tier", "") or ""
                    ),
                    "timeframe_compatibility_tier": str(
                        row.get("timeframe_compatibility_tier", "") or ""
                    ),
                    "strategy_type_compatibility_tier": str(
                        row.get("strategy_type_compatibility_tier", "") or ""
                    ),
                    "family_compatibility_component": float(
                        safe_float(row.get("family_compatibility_component", 0.0), 0.0)
                    ),
                    "exact_match_eligible": bool(row.get("exact_match_eligible", False)),
                    "compatible_band_eligible": bool(row.get("compatible_band_eligible", False)),
                    "incompatible_excluded": bool(row.get("incompatible_excluded", False)),
                    "entered_via_compatible_band": bool(
                        row.get("entered_via_compatible_band", False)
                    ),
                    "eligibility_tier": str(row.get("eligibility_tier", "incompatible") or "incompatible"),
                    "preliminary_family_score": float(
                        safe_float(row.get("preliminary_family_score", 0.0), 0.0)
                    ),
                    "preliminary_compatibility_penalty_component": float(
                        safe_float(
                            row.get("preliminary_compatibility_penalty_component", 0.0),
                            0.0,
                        )
                    ),
                    "entered_pre_cap_pool": bool(row.get("entered_pre_cap_pool", False)),
                    "survived_cap": bool(row.get("survived_cap", False)),
                    "cap_drop_reason": str(row.get("cap_drop_reason", "") or ""),
                    "cap_tier_slot_used": str(row.get("cap_tier_slot_used", "") or ""),
                    "final_competition_pool_flag": bool(
                        row.get("final_competition_pool_flag", row.get("competition_eligible", False))
                    ),
                    "session": str(row.get("coarse_compatibility_session", "") or ""),
                    "side": str(row.get("coarse_compatibility_side", "") or ""),
                    "timeframe": str(row.get("coarse_compatibility_timeframe", "") or ""),
                    "de3_strategy_type": str(
                        row.get("coarse_compatibility_strategy_type", "") or ""
                    ),
                    "threshold": str(
                        row.get("coarse_compatibility_threshold", "") or ""
                    ),
                    "member_candidates_seen_count": int(
                        safe_float(row.get("member_candidates_seen_count", 0), 0)
                    ),
                    "feasible_member_count": int(
                        safe_float(row.get("feasible_member_count", 0), 0)
                    ),
                    "member_filtered_out_count": int(
                        safe_float(row.get("member_filtered_out_count", 0), 0)
                    ),
                    "scanned": True,
                    "chosen": bool(row.get("family_chosen_flag", False)),
                    "retained_families_total": int(
                        safe_float(row.get("retained_families_total", 0), 0)
                    ),
                    "retained_families_scanned": int(
                        safe_float(row.get("retained_families_scanned", 0), 0)
                    ),
                    "retained_families_eligible": int(
                        safe_float(row.get("retained_families_eligible", 0), 0)
                    ),
                    "retained_families_excluded": int(
                        safe_float(row.get("retained_families_excluded", 0), 0)
                    ),
                    "retained_families_unscanned": int(
                        safe_float(row.get("retained_families_unscanned", 0), 0)
                    ),
                    "retained_family_scan_guarantee_pass": bool(
                        row.get("retained_family_scan_guarantee_pass", False)
                    ),
                    "context_bucket_joint": context_joint,
                }
            )
            row.pop("entries", None)
        self._append_family_eligibility_trace_rows(trace_rows)
        self._update_family_reachability_stats(
            family_rows=family_rows,
            normalized_context=normalized_context,
        )

    def _update_family_reachability_stats(
        self,
        *,
        family_rows: List[Dict[str, Any]],
        normalized_context: Dict[str, str],
    ) -> None:
        if not family_rows:
            return
        context_joint = build_active_context_joint_key(
            normalized_context if isinstance(normalized_context, dict) else {}
        )
        exclusion_context = str(context_joint or "").strip() or "none"
        for row in family_rows:
            if not isinstance(row, dict):
                continue
            family_id = str(row.get("family_id", "") or "").strip()
            if not family_id:
                continue
            stats = self._family_reachability_by_family.get(family_id)
            if not isinstance(stats, dict):
                stats = {
                    "scanned_count": 0,
                    "eligible_count": 0,
                    "chosen_count": 0,
                    "exact_match_count": 0,
                    "compatible_band_count": 0,
                    "incompatible_exclusion_count": 0,
                    "pre_cap_eligible_count": 0,
                    "post_cap_survived_count": 0,
                    "dropped_by_cap_count": 0,
                    "preliminary_score_sum": 0.0,
                    "preliminary_score_count": 0,
                    "eligibility_tier_counts": {},
                    "cap_drop_reason_counts": {},
                    "failure_reason_counts": {},
                    "exclusion_context_counts": {},
                    "incompatibility_reason_counts": {},
                }
            stats["scanned_count"] = int(stats.get("scanned_count", 0) + 1)
            if bool(row.get("competition_eligible", False)):
                stats["eligible_count"] = int(stats.get("eligible_count", 0) + 1)
            if bool(row.get("family_chosen_flag", False)):
                stats["chosen_count"] = int(stats.get("chosen_count", 0) + 1)
            if bool(row.get("coarse_eligible", False)) and str(row.get("compatibility_tier", "")).strip().lower() == "exact":
                stats["exact_match_count"] = int(stats.get("exact_match_count", 0) + 1)
            if bool(row.get("coarse_eligible", False)) and str(row.get("compatibility_tier", "")).strip().lower() == "compatible":
                stats["compatible_band_count"] = int(stats.get("compatible_band_count", 0) + 1)
            if (not bool(row.get("coarse_eligible", False))) and str(row.get("compatibility_tier", "")).strip().lower() == "incompatible":
                stats["incompatible_exclusion_count"] = int(
                    stats.get("incompatible_exclusion_count", 0) + 1
                )
            if bool(row.get("entered_pre_cap_pool", False)):
                stats["pre_cap_eligible_count"] = int(
                    stats.get("pre_cap_eligible_count", 0) + 1
                )
                stats["preliminary_score_sum"] = float(
                    safe_float(stats.get("preliminary_score_sum", 0.0), 0.0)
                    + safe_float(row.get("preliminary_family_score", 0.0), 0.0)
                )
                stats["preliminary_score_count"] = int(
                    safe_float(stats.get("preliminary_score_count", 0), 0) + 1
                )
                tier_counts = (
                    stats.get("eligibility_tier_counts")
                    if isinstance(stats.get("eligibility_tier_counts"), dict)
                    else {}
                )
                tier_key = str(row.get("eligibility_tier", "incompatible") or "incompatible")
                tier_counts[tier_key] = int(tier_counts.get(tier_key, 0) + 1)
                stats["eligibility_tier_counts"] = tier_counts
            if bool(row.get("survived_cap", False)):
                stats["post_cap_survived_count"] = int(
                    stats.get("post_cap_survived_count", 0) + 1
                )
            if bool(row.get("excluded_by_candidate_cap", False)):
                stats["dropped_by_cap_count"] = int(
                    stats.get("dropped_by_cap_count", 0) + 1
                )
                cap_reason_counts = (
                    stats.get("cap_drop_reason_counts")
                    if isinstance(stats.get("cap_drop_reason_counts"), dict)
                    else {}
                )
                cap_reason = str(row.get("cap_drop_reason", "") or "").strip()
                if cap_reason:
                    cap_reason_counts[cap_reason] = int(cap_reason_counts.get(cap_reason, 0) + 1)
                stats["cap_drop_reason_counts"] = cap_reason_counts
            reason = str(row.get("eligibility_failure_reason", "") or "").strip()
            if reason:
                reason_counts = (
                    stats.get("failure_reason_counts")
                    if isinstance(stats.get("failure_reason_counts"), dict)
                    else {}
                )
                reason_counts[reason] = int(reason_counts.get(reason, 0) + 1)
                stats["failure_reason_counts"] = reason_counts
                ctx_counts = (
                    stats.get("exclusion_context_counts")
                    if isinstance(stats.get("exclusion_context_counts"), dict)
                    else {}
                )
                ctx_counts[exclusion_context] = int(ctx_counts.get(exclusion_context, 0) + 1)
                stats["exclusion_context_counts"] = ctx_counts
            incompat_reason = str(row.get("coarse_eligibility_failure_reason", "") or "").strip()
            if str(row.get("compatibility_tier", "")).strip().lower() == "incompatible" and incompat_reason:
                incompat_counts = (
                    stats.get("incompatibility_reason_counts")
                    if isinstance(stats.get("incompatibility_reason_counts"), dict)
                    else {}
                )
                incompat_counts[incompat_reason] = int(incompat_counts.get(incompat_reason, 0) + 1)
                stats["incompatibility_reason_counts"] = incompat_counts
            self._family_reachability_by_family[family_id] = stats

    def get_family_score_trace(self) -> Dict[str, Any]:
        rows = list(self._family_score_trace_rows)
        candidate_hist: Dict[str, int] = defaultdict(int)
        chosen_families: Dict[str, int] = defaultdict(int)
        for row in rows:
            candidate_count = int(safe_float(row.get("family_candidate_count", 0), 0))
            candidate_hist[str(max(0, candidate_count))] = int(
                candidate_hist.get(str(max(0, candidate_count)), 0) + 1
            )
            if bool(row.get("chosen_flag", False)):
                family_id = str(row.get("family_id", "") or "").strip()
                if family_id:
                    chosen_families[family_id] = int(chosen_families.get(family_id, 0) + 1)
        return {
            "row_count": int(len(rows)),
            "decision_count_estimate": int(
                len({int(safe_float(row.get("decision_invocation", -1), -1)) for row in rows if int(safe_float(row.get("decision_invocation", -1), -1)) >= 0})
            ),
            "rows": rows,
            "family_candidate_count_histogram": dict(candidate_hist),
            "chosen_family_frequency": dict(chosen_families),
        }

    def get_family_score_component_summary(self) -> Dict[str, Any]:
        rows = [
            row
            for row in self._family_score_trace_rows
            if isinstance(row, dict) and bool(row.get("final_competition_pool_flag", False))
        ]
        if not rows:
            return {
                "row_count": 0,
                "decision_count_estimate": 0,
                "families": {},
                "overall": {},
            }

        decision_rows: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for row in rows:
            inv = int(safe_float(row.get("decision_invocation", -1), -1))
            if inv >= 0:
                decision_rows[inv].append(row)

        winner_by_decision: Dict[int, Dict[str, Any]] = {}
        runner_by_decision: Dict[int, Optional[Dict[str, Any]]] = {}
        for inv, inv_rows in decision_rows.items():
            ranked = sorted(
                inv_rows,
                key=lambda r: safe_float(r.get("final_family_score", float("-inf")), float("-inf")),
                reverse=True,
            )
            if ranked:
                winner_by_decision[inv] = ranked[0]
                runner_by_decision[inv] = ranked[1] if len(ranked) > 1 else None

        grouped: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            family_id = str(row.get("family_id", "") or "").strip()
            if not family_id:
                continue
            stats = grouped.get(family_id)
            if not isinstance(stats, dict):
                stats = {
                    "eligible_count": 0,
                    "chosen_count": 0,
                    "sum_prior": 0.0,
                    "sum_context": 0.0,
                    "sum_evidence": 0.0,
                    "sum_adaptive": 0.0,
                    "sum_diversity": 0.0,
                    "sum_compatibility": 0.0,
                    "sum_pre": 0.0,
                    "sum_final": 0.0,
                    "delta_vs_winner_values": [],
                    "delta_vs_runner_values": [],
                    "support_tier_counts": defaultdict(int),
                    "context_trusted_count": 0,
                }
                grouped[family_id] = stats
            stats["eligible_count"] = int(stats["eligible_count"] + 1)
            chosen_flag = bool(row.get("chosen_flag", False))
            if chosen_flag:
                stats["chosen_count"] = int(stats["chosen_count"] + 1)
            stats["sum_prior"] += float(safe_float(row.get("prior_component", 0.0), 0.0))
            stats["sum_context"] += float(
                safe_float(row.get("trusted_context_component", 0.0), 0.0)
            )
            stats["sum_evidence"] += float(
                safe_float(row.get("evidence_adjustment", 0.0), 0.0)
            )
            stats["sum_adaptive"] += float(
                safe_float(row.get("adaptive_component", 0.0), 0.0)
            )
            stats["sum_diversity"] += float(
                safe_float(row.get("competition_diversity_adjustment", 0.0), 0.0)
            )
            stats["sum_compatibility"] += float(
                safe_float(row.get("family_compatibility_component", 0.0), 0.0)
            )
            stats["sum_pre"] += float(
                safe_float(row.get("pre_adjustment_score", 0.0), 0.0)
            )
            stats["sum_final"] += float(
                safe_float(row.get("final_family_score", 0.0), 0.0)
            )
            tier = str(row.get("support_tier", "low") or "low").strip().lower()
            stats["support_tier_counts"][tier] = int(
                stats["support_tier_counts"].get(tier, 0) + 1
            )
            if bool(row.get("context_trusted_flag", False)):
                stats["context_trusted_count"] = int(stats["context_trusted_count"] + 1)

            inv = int(safe_float(row.get("decision_invocation", -1), -1))
            winner = winner_by_decision.get(inv)
            if isinstance(winner, dict) and not chosen_flag:
                winner_score = float(
                    safe_float(winner.get("final_family_score", 0.0), 0.0)
                )
                row_score = float(safe_float(row.get("final_family_score", 0.0), 0.0))
                stats["delta_vs_winner_values"].append(float(row_score - winner_score))
            runner = runner_by_decision.get(inv)
            if isinstance(runner, dict):
                runner_score = float(
                    safe_float(runner.get("final_family_score", 0.0), 0.0)
                )
                row_score = float(safe_float(row.get("final_family_score", 0.0), 0.0))
                stats["delta_vs_runner_values"].append(float(row_score - runner_score))

        families_out: Dict[str, Any] = {}
        for family_id, stats in grouped.items():
            eligible_count = int(stats.get("eligible_count", 0) or 0)
            denom = float(max(1, eligible_count))
            winner_vals = (
                list(stats.get("delta_vs_winner_values", []))
                if isinstance(stats.get("delta_vs_winner_values"), list)
                else []
            )
            runner_vals = (
                list(stats.get("delta_vs_runner_values", []))
                if isinstance(stats.get("delta_vs_runner_values"), list)
                else []
            )
            families_out[family_id] = {
                "eligible_count": int(eligible_count),
                "chosen_count": int(stats.get("chosen_count", 0) or 0),
                "mean_prior_component": float(stats.get("sum_prior", 0.0) / denom),
                "mean_trusted_context_component": float(
                    stats.get("sum_context", 0.0) / denom
                ),
                "mean_evidence_adjustment": float(
                    stats.get("sum_evidence", 0.0) / denom
                ),
                "mean_adaptive_component": float(
                    stats.get("sum_adaptive", 0.0) / denom
                ),
                "mean_competition_diversity_adjustment": float(
                    stats.get("sum_diversity", 0.0) / denom
                ),
                "mean_family_compatibility_component": float(
                    stats.get("sum_compatibility", 0.0) / denom
                ),
                "mean_pre_adjustment_score": float(stats.get("sum_pre", 0.0) / denom),
                "mean_final_family_score": float(stats.get("sum_final", 0.0) / denom),
                "mean_score_delta_vs_winner_when_not_chosen": float(
                    (sum(winner_vals) / float(max(1, len(winner_vals))))
                    if winner_vals
                    else 0.0
                ),
                "mean_score_delta_vs_runner_up": float(
                    (sum(runner_vals) / float(max(1, len(runner_vals))))
                    if runner_vals
                    else 0.0
                ),
                "support_tier_distribution": {
                    str(k): int(v)
                    for k, v in sorted(
                        (stats.get("support_tier_counts", {}) or {}).items()
                    )
                },
                "context_trusted_rate": float(
                    safe_float(stats.get("context_trusted_count", 0), 0.0) / denom
                ),
            }

        all_rows = float(max(1, len(rows)))
        overall = {
            "family_count": int(len(families_out)),
            "mean_prior_component": float(
                sum(float(safe_float(row.get("prior_component", 0.0), 0.0)) for row in rows)
                / all_rows
            ),
            "mean_trusted_context_component": float(
                sum(
                    float(
                        safe_float(row.get("trusted_context_component", 0.0), 0.0)
                    )
                    for row in rows
                )
                / all_rows
            ),
            "mean_evidence_adjustment": float(
                sum(float(safe_float(row.get("evidence_adjustment", 0.0), 0.0)) for row in rows)
                / all_rows
            ),
            "mean_adaptive_component": float(
                sum(float(safe_float(row.get("adaptive_component", 0.0), 0.0)) for row in rows)
                / all_rows
            ),
            "mean_competition_diversity_adjustment": float(
                sum(
                    float(
                        safe_float(row.get("competition_diversity_adjustment", 0.0), 0.0)
                    )
                    for row in rows
                )
                / all_rows
            ),
            "mean_family_compatibility_component": float(
                sum(
                    float(
                        safe_float(row.get("family_compatibility_component", 0.0), 0.0)
                    )
                    for row in rows
                )
                / all_rows
            ),
        }
        return {
            "row_count": int(len(rows)),
            "decision_count_estimate": int(len(decision_rows)),
            "families": families_out,
            "overall": overall,
        }

    def get_family_score_delta_ladder(self) -> Dict[str, Any]:
        rows = [
            row
            for row in self._family_score_trace_rows
            if isinstance(row, dict) and bool(row.get("final_competition_pool_flag", False))
        ]
        if not rows:
            return {
                "row_count": 0,
                "decision_count_estimate": 0,
                "families": {},
                "frequently_eligible_rarely_chosen": [],
                "dominant_gap_component_overall": "unknown",
            }

        decision_rows: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for row in rows:
            inv = int(safe_float(row.get("decision_invocation", -1), -1))
            if inv >= 0:
                decision_rows[inv].append(row)

        family_stats: Dict[str, Dict[str, Any]] = {}
        overall_component_gap = defaultdict(float)
        overall_component_count = defaultdict(int)

        for inv_rows in decision_rows.values():
            ranked = sorted(
                inv_rows,
                key=lambda r: safe_float(r.get("final_family_score", float("-inf")), float("-inf")),
                reverse=True,
            )
            if not ranked:
                continue
            winner = ranked[0]
            winner_components = {
                "prior": float(safe_float(winner.get("prior_component", 0.0), 0.0)),
                "context": float(
                    safe_float(winner.get("trusted_context_component", 0.0), 0.0)
                ),
                "evidence": float(
                    safe_float(winner.get("evidence_adjustment", 0.0), 0.0)
                ),
                "adaptive": float(safe_float(winner.get("adaptive_component", 0.0), 0.0)),
                "compatibility": float(
                    safe_float(winner.get("family_compatibility_component", 0.0), 0.0)
                ),
                "competition": float(
                    safe_float(
                        winner.get("competition_diversity_adjustment", 0.0), 0.0
                    )
                ),
            }
            winner_score = float(
                safe_float(winner.get("final_family_score", float("-inf")), float("-inf"))
            )
            if not math.isfinite(winner_score):
                continue
            for row in ranked:
                family_id = str(row.get("family_id", "") or "").strip()
                if not family_id:
                    continue
                stats = family_stats.get(family_id)
                if not isinstance(stats, dict):
                    stats = {
                        "eligible_count": 0,
                        "chosen_count": 0,
                        "score_gap_values": [],
                        "component_gap_sums": defaultdict(float),
                        "component_gap_counts": defaultdict(int),
                    }
                    family_stats[family_id] = stats
                stats["eligible_count"] = int(stats["eligible_count"] + 1)
                chosen_flag = bool(row.get("chosen_flag", False))
                if chosen_flag:
                    stats["chosen_count"] = int(stats["chosen_count"] + 1)
                    continue

                row_score = float(
                    safe_float(row.get("final_family_score", float("-inf")), float("-inf"))
                )
                if not math.isfinite(row_score):
                    continue
                gap = float(winner_score - row_score)
                stats["score_gap_values"].append(gap)

                row_components = {
                    "prior": float(safe_float(row.get("prior_component", 0.0), 0.0)),
                    "context": float(
                        safe_float(row.get("trusted_context_component", 0.0), 0.0)
                    ),
                    "evidence": float(
                        safe_float(row.get("evidence_adjustment", 0.0), 0.0)
                    ),
                    "adaptive": float(
                        safe_float(row.get("adaptive_component", 0.0), 0.0)
                    ),
                    "compatibility": float(
                        safe_float(row.get("family_compatibility_component", 0.0), 0.0)
                    ),
                    "competition": float(
                        safe_float(
                            row.get("competition_diversity_adjustment", 0.0), 0.0
                        )
                    ),
                }
                for cname, winner_val in winner_components.items():
                    cgap = float(winner_val - row_components.get(cname, 0.0))
                    stats["component_gap_sums"][cname] = float(
                        stats["component_gap_sums"].get(cname, 0.0) + cgap
                    )
                    stats["component_gap_counts"][cname] = int(
                        stats["component_gap_counts"].get(cname, 0) + 1
                    )
                    overall_component_gap[cname] = float(
                        overall_component_gap.get(cname, 0.0) + cgap
                    )
                    overall_component_count[cname] = int(
                        overall_component_count.get(cname, 0) + 1
                    )

        families_out: Dict[str, Any] = {}
        frequent_rare: List[Dict[str, Any]] = []
        for family_id, stats in family_stats.items():
            gaps = sorted(
                [
                    float(v)
                    for v in (stats.get("score_gap_values", []) or [])
                    if math.isfinite(float(v))
                ]
            )
            if gaps:
                n = len(gaps)

                def _pct(p: float) -> float:
                    idx = int(round((p / 100.0) * float(max(0, n - 1))))
                    idx = max(0, min(n - 1, idx))
                    return float(gaps[idx])
            else:

                def _pct(p: float) -> float:
                    _ = p
                    return 0.0

            comp_means: Dict[str, float] = {}
            for cname, sum_v in (stats.get("component_gap_sums", {}) or {}).items():
                cnt = int((stats.get("component_gap_counts", {}) or {}).get(cname, 0) or 0)
                comp_means[str(cname)] = float(sum_v / float(max(1, cnt)))
            dominant_component = "none"
            if comp_means:
                dominant_component = str(
                    max(comp_means.items(), key=lambda item: float(item[1]))[0]
                )
            eligible_count = int(stats.get("eligible_count", 0) or 0)
            chosen_count = int(stats.get("chosen_count", 0) or 0)
            chosen_rate = float(chosen_count / float(max(1, eligible_count)))
            family_payload = {
                "eligible_count": int(eligible_count),
                "chosen_count": int(chosen_count),
                "chosen_rate": float(chosen_rate),
                "average_score_gap_to_winner": float(sum(gaps) / float(max(1, len(gaps))))
                if gaps
                else 0.0,
                "median_score_gap_to_winner": float(_pct(50)),
                "p10_score_gap_to_winner": float(_pct(10)),
                "p25_score_gap_to_winner": float(_pct(25)),
                "p75_score_gap_to_winner": float(_pct(75)),
                "p90_score_gap_to_winner": float(_pct(90)),
                "component_gap_means": comp_means,
                "dominant_gap_component": str(dominant_component),
            }
            families_out[family_id] = family_payload
            if eligible_count >= 25 and chosen_rate <= 0.05:
                frequent_rare.append({"family_id": family_id, **family_payload})

        overall_component_means: Dict[str, float] = {}
        for cname, total in overall_component_gap.items():
            cnt = int(overall_component_count.get(cname, 0) or 0)
            overall_component_means[str(cname)] = float(total / float(max(1, cnt)))
        dominant_overall = "unknown"
        if overall_component_means:
            dominant_overall = str(
                max(
                    overall_component_means.items(),
                    key=lambda item: float(item[1]),
                )[0]
            )

        frequent_rare.sort(
            key=lambda row: float(row.get("average_score_gap_to_winner", 0.0)),
            reverse=True,
        )
        return {
            "row_count": int(len(rows)),
            "decision_count_estimate": int(len(decision_rows)),
            "families": families_out,
            "frequently_eligible_rarely_chosen": frequent_rare,
            "overall_component_gap_means": overall_component_means,
            "dominant_gap_component_overall": str(dominant_overall),
        }

    def get_member_resolution_audit(self) -> Dict[str, Any]:
        rows = list(self._member_resolution_trace_rows)
        summary = {
            "row_count": int(len(rows)),
            "anchor_selected_count": 0,
            "non_anchor_selected_count": 0,
            "unknown_anchor_alignment_count": 0,
            "frozen_mode_count": 0,
            "conservative_mode_count": 0,
            "full_mode_count": 0,
            "no_local_alternative_count": 0,
            "canonical_fallback_used_count": 0,
        }
        for row in rows:
            mode = str(row.get("local_member_selection_mode", "none") or "none").strip().lower()
            if mode == "frozen":
                summary["frozen_mode_count"] = int(summary["frozen_mode_count"] + 1)
            elif mode == "conservative":
                summary["conservative_mode_count"] = int(summary["conservative_mode_count"] + 1)
            elif mode == "full":
                summary["full_mode_count"] = int(summary["full_mode_count"] + 1)
            anchor_member_id = str(row.get("anchor_member_id", "") or "").strip()
            chosen_member_id = str(row.get("chosen_member_id", "") or "").strip()
            anchor_selected = bool(row.get("anchor_selected", False))
            if anchor_selected:
                summary["anchor_selected_count"] = int(summary["anchor_selected_count"] + 1)
            else:
                if anchor_member_id and chosen_member_id:
                    summary["non_anchor_selected_count"] = int(summary["non_anchor_selected_count"] + 1)
                else:
                    summary["unknown_anchor_alignment_count"] = int(
                        summary["unknown_anchor_alignment_count"] + 1
                    )
            if bool(row.get("no_local_alternative", False)):
                summary["no_local_alternative_count"] = int(summary["no_local_alternative_count"] + 1)
            if bool(row.get("canonical_fallback_used", False)):
                summary["canonical_fallback_used_count"] = int(
                    summary["canonical_fallback_used_count"] + 1
                )
        return {"rows": rows, "summary": summary}

    def get_family_eligibility_trace(self) -> Dict[str, Any]:
        rows = list(self._family_eligibility_trace_rows)
        return {
            "row_count": int(len(rows)),
            "decision_count_estimate": int(
                len(
                    {
                        int(safe_float(row.get("decision_invocation", -1), -1))
                        for row in rows
                        if int(safe_float(row.get("decision_invocation", -1), -1)) >= 0
                    }
                )
            ),
            "rows": rows,
        }

    def get_family_reachability_summary(self) -> Dict[str, Any]:
        retained_ids = [
            str(fid or "").strip()
            for fid, row in self.families_by_id.items()
            if str(fid or "").strip() and self._family_retained_for_runtime(row)
        ]
        retained_set = set(retained_ids)
        scanned_ids = {
            fid
            for fid, stats in self._family_reachability_by_family.items()
            if str(fid or "").strip()
            and int(((stats or {}).get("scanned_count", 0) or 0)) > 0
        }
        eligible_ids = {
            fid
            for fid, stats in self._family_reachability_by_family.items()
            if str(fid or "").strip()
            and int(((stats or {}).get("eligible_count", 0) or 0)) > 0
        }
        chosen_ids = {
            fid
            for fid, stats in self._family_reachability_by_family.items()
            if str(fid or "").strip()
            and int(((stats or {}).get("chosen_count", 0) or 0)) > 0
        }
        exact_eligible_ids = {
            fid
            for fid, stats in self._family_reachability_by_family.items()
            if str(fid or "").strip()
            and int(((stats or {}).get("exact_match_count", 0) or 0)) > 0
        }
        compatible_eligible_ids = {
            fid
            for fid, stats in self._family_reachability_by_family.items()
            if str(fid or "").strip()
            and int(((stats or {}).get("compatible_band_count", 0) or 0)) > 0
        }
        incompatible_only_ids = {
            fid
            for fid, stats in self._family_reachability_by_family.items()
            if str(fid or "").strip()
            and int(((stats or {}).get("scanned_count", 0) or 0)) > 0
            and int(((stats or {}).get("exact_match_count", 0) or 0)) <= 0
            and int(((stats or {}).get("compatible_band_count", 0) or 0)) <= 0
            and int(((stats or {}).get("incompatible_exclusion_count", 0) or 0)) > 0
        }
        never_scanned = sorted(fid for fid in retained_set if fid not in scanned_ids)
        scanned_but_never_eligible = sorted(fid for fid in scanned_ids if fid in retained_set and fid not in eligible_ids)
        eligible_but_never_chosen = sorted(fid for fid in eligible_ids if fid not in chosen_ids)

        top_reason_counts: Dict[str, int] = defaultdict(int)
        top_compat_failure_counts: Dict[str, int] = defaultdict(int)
        per_family: Dict[str, Any] = {}
        for fid in retained_ids:
            stats = self._family_reachability_by_family.get(fid, {}) if isinstance(self._family_reachability_by_family.get(fid, {}), dict) else {}
            reason_counts = stats.get("failure_reason_counts") if isinstance(stats.get("failure_reason_counts"), dict) else {}
            context_counts = stats.get("exclusion_context_counts") if isinstance(stats.get("exclusion_context_counts"), dict) else {}
            incompat_reason_counts = stats.get("incompatibility_reason_counts") if isinstance(stats.get("incompatibility_reason_counts"), dict) else {}
            for reason, cnt in reason_counts.items():
                top_reason_counts[str(reason)] = int(top_reason_counts.get(str(reason), 0) + int(cnt or 0))
            for reason, cnt in incompat_reason_counts.items():
                top_compat_failure_counts[str(reason)] = int(
                    top_compat_failure_counts.get(str(reason), 0) + int(cnt or 0)
                )
            most_common_failure_reason = ""
            most_common_context = ""
            most_common_compat_failure_reason = ""
            if reason_counts:
                most_common_failure_reason = str(
                    max(reason_counts.items(), key=lambda item: int(item[1] or 0))[0]
                )
            if context_counts:
                most_common_context = str(
                    max(context_counts.items(), key=lambda item: int(item[1] or 0))[0]
                )
            if incompat_reason_counts:
                most_common_compat_failure_reason = str(
                    max(incompat_reason_counts.items(), key=lambda item: int(item[1] or 0))[0]
                )
            per_family[fid] = {
                "scanned_count": int(stats.get("scanned_count", 0) or 0),
                "eligible_count": int(stats.get("eligible_count", 0) or 0),
                "chosen_count": int(stats.get("chosen_count", 0) or 0),
                "exact_match_count": int(stats.get("exact_match_count", 0) or 0),
                "compatible_band_count": int(stats.get("compatible_band_count", 0) or 0),
                "incompatible_exclusion_count": int(
                    stats.get("incompatible_exclusion_count", 0) or 0
                ),
                "pre_cap_eligible_count": int(stats.get("pre_cap_eligible_count", 0) or 0),
                "post_cap_survived_count": int(stats.get("post_cap_survived_count", 0) or 0),
                "dropped_by_cap_count": int(stats.get("dropped_by_cap_count", 0) or 0),
                "average_preliminary_score": float(
                    safe_float(stats.get("preliminary_score_sum", 0.0), 0.0)
                    / float(max(1, int(safe_float(stats.get("preliminary_score_count", 0), 0))))
                ),
                "eligibility_tier_counts": dict(
                    stats.get("eligibility_tier_counts", {})
                    if isinstance(stats.get("eligibility_tier_counts"), dict)
                    else {}
                ),
                "cap_drop_reason_counts": dict(
                    stats.get("cap_drop_reason_counts", {})
                    if isinstance(stats.get("cap_drop_reason_counts"), dict)
                    else {}
                ),
                "most_common_failure_reason": str(most_common_failure_reason),
                "most_common_compatibility_failure_reason": str(
                    most_common_compat_failure_reason
                ),
                "most_common_context_bucket_when_excluded": str(most_common_context),
            }

        invocations = int(self._path_counters.get("runtime_invocations", 0) or 0)
        return {
            "retained_runtime_family_count": int(len(retained_ids)),
            "families_ever_scanned_count": int(len(scanned_ids)),
            "families_ever_eligible_count": int(len(eligible_ids)),
            "families_ever_exact_match_eligible_count": int(len(exact_eligible_ids)),
            "families_ever_compatible_band_eligible_count": int(len(compatible_eligible_ids)),
            "families_ever_incompatible_only_count": int(len(incompatible_only_ids)),
            "families_never_scanned_count": int(len(never_scanned)),
            "families_never_scanned": never_scanned,
            "families_scanned_but_never_eligible_count": int(len(scanned_but_never_eligible)),
            "families_scanned_but_never_eligible": scanned_but_never_eligible,
            "families_eligible_but_never_chosen_count": int(len(eligible_but_never_chosen)),
            "families_eligible_but_never_chosen": eligible_but_never_chosen,
            "top_exclusion_reasons": {
                str(k): int(v)
                for k, v in sorted(top_reason_counts.items(), key=lambda item: item[1], reverse=True)
            },
            "top_compatibility_failure_reasons": {
                str(k): int(v)
                for k, v in sorted(top_compat_failure_counts.items(), key=lambda item: item[1], reverse=True)
            },
            "average_pre_cap_family_candidate_count": float(
                safe_float(self._path_counters.get("pre_cap_candidate_total", 0), 0.0)
                / float(max(1, invocations))
            ),
            "average_post_cap_family_candidate_count": float(
                safe_float(self._path_counters.get("post_cap_candidate_total", 0), 0.0)
                / float(max(1, invocations))
            ),
            "average_compatible_band_families_pre_cap": float(
                safe_float(self._path_counters.get("pre_cap_compatible_eligible_total", 0), 0.0)
                / float(max(1, invocations))
            ),
            "average_compatible_band_families_post_cap": float(
                safe_float(self._path_counters.get("post_cap_compatible_survived_total", 0), 0.0)
                / float(max(1, invocations))
            ),
            "decisions_where_compatible_existed_but_all_dropped_by_cap": int(
                safe_float(
                    self._path_counters.get(
                        "decisions_with_compatible_pre_cap_all_dropped_count", 0
                    ),
                    0,
                )
            ),
            "per_family": per_family,
        }

    def get_family_compatibility_audit(self) -> Dict[str, Any]:
        retained_ids = [
            str(fid or "").strip()
            for fid, row in self.families_by_id.items()
            if str(fid or "").strip() and self._family_retained_for_runtime(row)
        ]
        counts = {
            "exact_match_eligible": 0,
            "compatible_band_eligible": 0,
            "incompatible_excluded": 0,
        }
        top_incompat_reasons: Dict[str, int] = defaultdict(int)
        per_family: Dict[str, Any] = {}
        for fid in retained_ids:
            stats = self._family_reachability_by_family.get(fid, {}) if isinstance(self._family_reachability_by_family.get(fid, {}), dict) else {}
            exact_count = int(stats.get("exact_match_count", 0) or 0)
            compatible_count = int(stats.get("compatible_band_count", 0) or 0)
            incompatible_count = int(stats.get("incompatible_exclusion_count", 0) or 0)
            incompat_reason_counts = (
                stats.get("incompatibility_reason_counts")
                if isinstance(stats.get("incompatibility_reason_counts"), dict)
                else {}
            )
            for reason, value in incompat_reason_counts.items():
                top_incompat_reasons[str(reason)] = int(
                    top_incompat_reasons.get(str(reason), 0) + int(value or 0)
                )
            most_common_reason = ""
            if incompat_reason_counts:
                most_common_reason = str(
                    max(incompat_reason_counts.items(), key=lambda item: int(item[1] or 0))[0]
                )
            counts["exact_match_eligible"] = int(
                counts["exact_match_eligible"] + exact_count
            )
            counts["compatible_band_eligible"] = int(
                counts["compatible_band_eligible"] + compatible_count
            )
            counts["incompatible_excluded"] = int(
                counts["incompatible_excluded"] + incompatible_count
            )
            per_family[fid] = {
                "exact_match_count": int(exact_count),
                "compatible_band_count": int(compatible_count),
                "incompatible_exclusion_count": int(incompatible_count),
                "most_common_incompatibility_reason": str(most_common_reason),
            }
        return {
            "retained_runtime_family_count": int(len(retained_ids)),
            "eligibility_outcome_counts": counts,
            "exclusion_reasons_after_broadening": {
                str(k): int(v)
                for k, v in sorted(top_incompat_reasons.items(), key=lambda item: item[1], reverse=True)
            },
            "pre_cap_post_cap_summary": {
                "pre_cap_candidate_count_avg": float(
                    safe_float(self._path_counters.get("pre_cap_candidate_total", 0), 0.0)
                    / float(max(1, int(self._path_counters.get("runtime_invocations", 0) or 0)))
                ),
                "post_cap_candidate_count_avg": float(
                    safe_float(self._path_counters.get("post_cap_candidate_total", 0), 0.0)
                    / float(max(1, int(self._path_counters.get("runtime_invocations", 0) or 0)))
                ),
                "compatible_pre_cap_avg": float(
                    safe_float(self._path_counters.get("pre_cap_compatible_eligible_total", 0), 0.0)
                    / float(max(1, int(self._path_counters.get("runtime_invocations", 0) or 0)))
                ),
                "compatible_post_cap_avg": float(
                    safe_float(self._path_counters.get("post_cap_compatible_survived_total", 0), 0.0)
                    / float(max(1, int(self._path_counters.get("runtime_invocations", 0) or 0)))
                ),
                "decisions_with_compatible_pre_cap_all_dropped_count": int(
                    self._path_counters.get(
                        "decisions_with_compatible_pre_cap_all_dropped_count", 0
                    )
                    or 0
                ),
            },
            "per_family": per_family,
            "candidate_set_size_histogram_current": dict(self._family_candidate_size_histogram),
            "candidate_set_size_histogram_before_broadening": None,
        }

    def get_pre_cap_candidate_audit(self) -> Dict[str, Any]:
        decision_rows = list(self._pre_cap_candidate_audit_rows)
        per_family: Dict[str, Any] = {}
        for fid, row in self.families_by_id.items():
            family_id = str(fid or "").strip()
            if not family_id or not self._family_retained_for_runtime(row):
                continue
            stats = (
                self._family_reachability_by_family.get(family_id, {})
                if isinstance(self._family_reachability_by_family.get(family_id, {}), dict)
                else {}
            )
            tier_counts = (
                stats.get("eligibility_tier_counts")
                if isinstance(stats.get("eligibility_tier_counts"), dict)
                else {}
            )
            dominant_tier = ""
            if tier_counts:
                dominant_tier = str(
                    max(tier_counts.items(), key=lambda item: int(item[1] or 0))[0]
                )
            per_family[family_id] = {
                "pre_cap_eligible_count": int(stats.get("pre_cap_eligible_count", 0) or 0),
                "post_cap_survived_count": int(stats.get("post_cap_survived_count", 0) or 0),
                "dropped_by_cap_count": int(stats.get("dropped_by_cap_count", 0) or 0),
                "average_preliminary_score": float(
                    safe_float(stats.get("preliminary_score_sum", 0.0), 0.0)
                    / float(max(1, int(safe_float(stats.get("preliminary_score_count", 0), 0))))
                ),
                "eligibility_tier": str(dominant_tier or "incompatible"),
                "eligibility_tier_counts": dict(tier_counts),
            }

        summary = {
            "decision_count": int(len(decision_rows)),
            "pre_cap_candidate_count_avg": 0.0,
            "post_cap_candidate_count_avg": 0.0,
            "exact_match_eligible_count_avg": 0.0,
            "compatible_band_eligible_count_avg": 0.0,
            "exact_match_survived_count_avg": 0.0,
            "compatible_band_survived_count_avg": 0.0,
            "compatible_band_dropped_by_cap_count_avg": 0.0,
        }
        if decision_rows:
            denom = float(max(1, len(decision_rows)))
            summary["pre_cap_candidate_count_avg"] = float(
                sum(int(safe_float(r.get("pre_cap_candidate_count", 0), 0)) for r in decision_rows)
                / denom
            )
            summary["post_cap_candidate_count_avg"] = float(
                sum(int(safe_float(r.get("post_cap_candidate_count", 0), 0)) for r in decision_rows)
                / denom
            )
            summary["exact_match_eligible_count_avg"] = float(
                sum(int(safe_float(r.get("exact_match_eligible_count", 0), 0)) for r in decision_rows)
                / denom
            )
            summary["compatible_band_eligible_count_avg"] = float(
                sum(int(safe_float(r.get("compatible_band_eligible_count", 0), 0)) for r in decision_rows)
                / denom
            )
            summary["exact_match_survived_count_avg"] = float(
                sum(int(safe_float(r.get("exact_match_survived_count", 0), 0)) for r in decision_rows)
                / denom
            )
            summary["compatible_band_survived_count_avg"] = float(
                sum(int(safe_float(r.get("compatible_band_survived_count", 0), 0)) for r in decision_rows)
                / denom
            )
            summary["compatible_band_dropped_by_cap_count_avg"] = float(
                sum(int(safe_float(r.get("compatible_band_dropped_by_cap_count", 0), 0)) for r in decision_rows)
                / denom
            )

        return {
            "summary": summary,
            "decisions": decision_rows,
            "per_family": per_family,
        }

    @staticmethod
    def _family_score_object_from_row(row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
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
                "support_tier": "low",
                "context_trusted_flag": False,
                "exploration_bonus_applied": False,
                "dominance_penalty_applied": False,
                "context_advantage_capped": False,
                "any_post_processing_applied": False,
                "compatibility_tier": "incompatible",
                "session_compatibility_tier": "incompatible",
                "timeframe_compatibility_tier": "incompatible",
                "strategy_type_compatibility_tier": "incompatible",
                "entered_via_compatible_band": False,
                "eligibility_tier": "incompatible",
                "preliminary_family_score": 0.0,
                "preliminary_compatibility_penalty_component": 0.0,
                "entered_pre_cap_pool": False,
                "survived_cap": False,
                "cap_drop_reason": "",
                "cap_tier_slot_used": "",
                "final_competition_pool_flag": False,
                "candidate_rank_before_adjustments": 0,
                "family_candidate_source": "",
                "family_runtime_role": "satellite",
                "is_core_family": False,
                "is_satellite_family": True,
                "chosen_flag": False,
                "runner_up_flag": False,
            }
        comps = row.get("family_score_components") if isinstance(row.get("family_score_components"), dict) else {}
        prior_component = float(
            safe_float(
                comps.get(
                    "family_prior_component_normalized",
                    comps.get(
                        "family_prior_component_weighted",
                        comps.get("family_prior_component", 0.0),
                    ),
                ),
                0.0,
            )
        )
        trusted_context_component = float(
            safe_float(
                comps.get(
                    "context_total_component_normalized",
                    comps.get(
                        "context_total_component",
                        safe_float(
                            comps.get(
                                "context_profile_expectancy_component_weighted", 0.0
                            ),
                            0.0,
                        )
                        + safe_float(
                            comps.get(
                                "context_profile_confidence_component_weighted", 0.0
                            ),
                            0.0,
                        ),
                    ),
                ),
                0.0,
            )
        )
        evidence_adjustment = float(
            safe_float(
                comps.get(
                    "v3_realized_usability_component_normalized",
                    comps.get(
                        "v3_realized_usability_component",
                        comps.get("usability_adjustment", 0.0),
                    ),
                ),
                0.0,
            )
        )
        adaptive_component = float(
            safe_float(
                comps.get("adaptive_policy_component_weighted", comps.get("adaptive_policy_component", 0.0)),
                0.0,
            )
        )
        diversity_adjustment = float(
            safe_float(
                comps.get(
                    "competition_diversity_adjustment",
                    row.get("competition_diversity_adjustment", row.get("diversity_adjustment", 0.0)),
                ),
                0.0,
            )
        )
        family_compatibility_component = float(
            safe_float(
                comps.get(
                    "family_compatibility_component",
                    row.get("family_compatibility_component", 0.0),
                ),
                0.0,
            )
        )
        pre_adjustment_score = float(
            safe_float(
                row.get("base_family_score", comps.get("base_family_score", row.get("family_score", 0.0))),
                0.0,
            )
        )
        final_family_score = float(
            safe_float(
                row.get("family_score", comps.get("final_family_score", pre_adjustment_score + diversity_adjustment)),
                0.0,
            )
        )
        context_trusted_flag = bool(comps.get("trusted_context_used", row.get("profile_trusted", False)))
        support_tier = str(
            row.get(
                "family_local_support_tier",
                comps.get("context_support_tier", row.get("context_support_tier", "low")),
            )
            or "low"
        ).strip().lower()
        exploration_bonus_applied = bool(
            comps.get(
                "exploration_bonus_applied",
                bool(safe_float(row.get("exploration_bonus", 0.0), 0.0) > 1e-9),
            )
        )
        dominance_penalty_applied = bool(
            comps.get(
                "dominance_penalty_applied",
                bool(safe_float(row.get("dominance_penalty", 0.0), 0.0) < -1e-9),
            )
        )
        context_advantage_capped = bool(comps.get("context_advantage_capped", row.get("context_advantage_capped", False)))
        any_post = bool(abs(diversity_adjustment) > 1e-9 or context_advantage_capped)
        return {
            "family_id": str(row.get("family_id", "") or ""),
            "prior_component": float(prior_component),
            "trusted_context_component": float(trusted_context_component),
            "evidence_adjustment": float(evidence_adjustment),
            "adaptive_component": float(adaptive_component),
            "competition_diversity_adjustment": float(diversity_adjustment),
            "family_compatibility_component": float(family_compatibility_component),
            "pre_adjustment_score": float(pre_adjustment_score),
            "final_family_score": float(final_family_score),
            "support_tier": str(support_tier),
            "context_trusted_flag": bool(context_trusted_flag),
            "exploration_bonus_applied": bool(exploration_bonus_applied),
            "dominance_penalty_applied": bool(dominance_penalty_applied),
            "context_advantage_capped": bool(context_advantage_capped),
            "any_post_processing_applied": bool(any_post),
            "compatibility_tier": str(
                row.get(
                    "compatibility_tier",
                    comps.get("compatibility_tier", "incompatible"),
                )
                or "incompatible"
            ).strip().lower(),
            "session_compatibility_tier": str(
                row.get(
                    "session_compatibility_tier",
                    comps.get("session_compatibility_tier", "incompatible"),
                )
                or "incompatible"
            ).strip().lower(),
            "timeframe_compatibility_tier": str(
                row.get(
                    "timeframe_compatibility_tier",
                    comps.get("timeframe_compatibility_tier", "incompatible"),
                )
                or "incompatible"
            ).strip().lower(),
            "strategy_type_compatibility_tier": str(
                row.get(
                    "strategy_type_compatibility_tier",
                    comps.get("strategy_type_compatibility_tier", "incompatible"),
                )
                or "incompatible"
            ).strip().lower(),
            "entered_via_compatible_band": bool(
                row.get(
                    "entered_via_compatible_band",
                    comps.get("entered_via_compatible_band", False),
                )
            ),
            "eligibility_tier": str(
                row.get(
                    "eligibility_tier",
                    comps.get("eligibility_tier", "incompatible"),
                )
                or "incompatible"
            ).strip().lower(),
            "preliminary_family_score": float(
                safe_float(
                    row.get(
                        "preliminary_family_score",
                        comps.get("preliminary_family_score", pre_adjustment_score),
                    ),
                    pre_adjustment_score,
                )
            ),
            "preliminary_compatibility_penalty_component": float(
                safe_float(
                    row.get(
                        "preliminary_compatibility_penalty_component",
                        comps.get("preliminary_compatibility_penalty_component", 0.0),
                    ),
                    0.0,
                )
            ),
            "entered_pre_cap_pool": bool(
                row.get(
                    "entered_pre_cap_pool",
                    comps.get("entered_pre_cap_pool", False),
                )
            ),
            "survived_cap": bool(
                row.get(
                    "survived_cap",
                    comps.get("survived_cap", bool(row.get("competition_eligible", False))),
                )
            ),
            "cap_drop_reason": str(
                row.get(
                    "cap_drop_reason",
                    comps.get("cap_drop_reason", ""),
                )
                or ""
            ),
            "cap_tier_slot_used": str(
                row.get(
                    "cap_tier_slot_used",
                    comps.get("cap_tier_slot_used", ""),
                )
                or ""
            ),
            "final_competition_pool_flag": bool(
                row.get(
                    "final_competition_pool_flag",
                    comps.get("final_competition_pool_flag", bool(row.get("competition_eligible", False))),
                )
            ),
            "candidate_rank_before_adjustments": int(
                safe_float(row.get("candidate_rank_before_adjustments", 0), 0)
            ),
            "family_candidate_source": str(row.get("family_candidate_source", "") or ""),
            "family_runtime_role": str(
                row.get(
                    "family_runtime_role",
                    comps.get("family_runtime_role", "satellite"),
                )
                or "satellite"
            ),
            "is_core_family": bool(
                row.get(
                    "is_core_family",
                    comps.get("is_core_family", False),
                )
            ),
            "is_satellite_family": bool(
                row.get(
                    "is_satellite_family",
                    comps.get("is_satellite_family", True),
                )
            ),
            "chosen_flag": bool(row.get("family_chosen_flag", False)),
            "runner_up_flag": bool(row.get("family_runner_up_flag", False)),
        }

    def _sync_family_score_object(
        self,
        row: Optional[Dict[str, Any]],
        *,
        chosen_flag: bool = False,
        runner_up_flag: bool = False,
    ) -> Dict[str, Any]:
        score_obj = self._family_score_object_from_row(row)
        score_obj["chosen_flag"] = bool(chosen_flag)
        score_obj["runner_up_flag"] = bool(runner_up_flag)
        if not isinstance(row, dict):
            return score_obj
        row["family_score_object"] = dict(score_obj)
        row["base_family_score"] = float(score_obj.get("pre_adjustment_score", 0.0))
        row["family_score"] = float(score_obj.get("final_family_score", 0.0))
        row["final_family_score"] = float(score_obj.get("final_family_score", 0.0))
        row["competition_diversity_adjustment"] = float(score_obj.get("competition_diversity_adjustment", 0.0))
        row["eligibility_tier"] = str(score_obj.get("eligibility_tier", "incompatible"))
        row["preliminary_family_score"] = float(score_obj.get("preliminary_family_score", 0.0))
        row["preliminary_compatibility_penalty_component"] = float(
            score_obj.get("preliminary_compatibility_penalty_component", 0.0)
        )
        row["entered_pre_cap_pool"] = bool(score_obj.get("entered_pre_cap_pool", False))
        row["survived_cap"] = bool(score_obj.get("survived_cap", False))
        row["cap_drop_reason"] = str(score_obj.get("cap_drop_reason", "") or "")
        row["cap_tier_slot_used"] = str(score_obj.get("cap_tier_slot_used", "") or "")
        row["final_competition_pool_flag"] = bool(
            score_obj.get("final_competition_pool_flag", False)
        )
        row["family_chosen_flag"] = bool(chosen_flag)
        row["family_runner_up_flag"] = bool(runner_up_flag)
        row["score_post_processing_applied"] = bool(score_obj.get("any_post_processing_applied", False))
        row["context_trusted_flag"] = bool(score_obj.get("context_trusted_flag", False))
        row["score_support_tier"] = str(score_obj.get("support_tier", "low"))
        row["candidate_rank_before_adjustments"] = int(
            safe_float(score_obj.get("candidate_rank_before_adjustments", row.get("candidate_rank_before_adjustments", 0)), 0)
        )
        if not str(row.get("family_candidate_source", "")).strip():
            row["family_candidate_source"] = str(score_obj.get("family_candidate_source", "") or "")
        row["family_runtime_role"] = str(score_obj.get("family_runtime_role", "satellite") or "satellite")
        row["is_core_family"] = bool(score_obj.get("is_core_family", False))
        row["is_satellite_family"] = bool(score_obj.get("is_satellite_family", True))
        comps = row.get("family_score_components") if isinstance(row.get("family_score_components"), dict) else {}
        comps["canonical_score_object"] = dict(score_obj)
        comps["base_family_score"] = float(score_obj.get("pre_adjustment_score", 0.0))
        comps["final_family_score"] = float(score_obj.get("final_family_score", 0.0))
        comps["competition_diversity_adjustment"] = float(score_obj.get("competition_diversity_adjustment", 0.0))
        comps["family_compatibility_component"] = float(score_obj.get("family_compatibility_component", 0.0))
        comps["trusted_context_used"] = bool(score_obj.get("context_trusted_flag", False))
        comps["local_support_tier"] = str(score_obj.get("support_tier", "low"))
        comps["score_post_processing_applied"] = bool(score_obj.get("any_post_processing_applied", False))
        comps["compatibility_tier"] = str(score_obj.get("compatibility_tier", "incompatible"))
        comps["family_runtime_role"] = str(score_obj.get("family_runtime_role", "satellite") or "satellite")
        comps["is_core_family"] = bool(score_obj.get("is_core_family", False))
        comps["is_satellite_family"] = bool(score_obj.get("is_satellite_family", True))
        comps["session_compatibility_tier"] = str(score_obj.get("session_compatibility_tier", "incompatible"))
        comps["timeframe_compatibility_tier"] = str(score_obj.get("timeframe_compatibility_tier", "incompatible"))
        comps["strategy_type_compatibility_tier"] = str(score_obj.get("strategy_type_compatibility_tier", "incompatible"))
        comps["entered_via_compatible_band"] = bool(score_obj.get("entered_via_compatible_band", False))
        comps["eligibility_tier"] = str(score_obj.get("eligibility_tier", "incompatible"))
        comps["preliminary_family_score"] = float(score_obj.get("preliminary_family_score", 0.0))
        comps["preliminary_compatibility_penalty_component"] = float(
            score_obj.get("preliminary_compatibility_penalty_component", 0.0)
        )
        comps["entered_pre_cap_pool"] = bool(score_obj.get("entered_pre_cap_pool", False))
        comps["survived_cap"] = bool(score_obj.get("survived_cap", False))
        comps["cap_drop_reason"] = str(score_obj.get("cap_drop_reason", "") or "")
        comps["cap_tier_slot_used"] = str(score_obj.get("cap_tier_slot_used", "") or "")
        comps["final_competition_pool_flag"] = bool(
            score_obj.get("final_competition_pool_flag", False)
        )
        row["family_score_components"] = comps
        return score_obj

    def _record_score_path_warning(self, *, invocation: int, code: str, detail: Dict[str, Any]) -> None:
        severity = "medium"
        likely_cause = "unknown"
        if code in {"multi_candidate_all_zero_final_scores", "trusted_context_flagged_but_context_component_zero"}:
            severity = "high"
            likely_cause = "export_bug_or_wrong_variable_binding"
        elif code in {"adjustments_flagged_but_diversity_component_zero"}:
            severity = "high"
            likely_cause = "dead_scoring_branch_or_overwritten_values"
        elif code in {"multiple_candidates_but_zero_score_delta", "multi_candidate_all_equal_final_scores"}:
            severity = "medium"
            likely_cause = "fallback_path_dominance_or_neutral_scoring"
        warnings = self._score_path_audit.setdefault("warnings", [])
        if not isinstance(warnings, list):
            warnings = []
            self._score_path_audit["warnings"] = warnings
        warnings.append(
            {
                "invocation": int(invocation),
                "code": str(code),
                "severity": str(severity),
                "likely_cause": str(likely_cause),
                "detail": dict(detail),
            }
        )
        max_warnings = 1500
        if len(warnings) > max_warnings:
            self._score_path_audit["warnings"] = warnings[-max_warnings:]
        self._score_path_audit["warning_count"] = int(self._score_path_audit.get("warning_count", 0) + 1)
        violations = self._score_path_audit.setdefault("rule_violations", {})
        row = violations.get(code, {}) if isinstance(violations.get(code), dict) else {}
        examples = list(row.get("example_invocations", [])) if isinstance(row.get("example_invocations"), list) else []
        if int(invocation) not in examples and len(examples) < 25:
            examples.append(int(invocation))
        row = {
            "count": int(safe_float(row.get("count", 0), 0) + 1),
            "severity": str(severity),
            "likely_cause": str(likely_cause),
            "example_invocations": examples,
        }
        violations[str(code)] = row
        self._score_path_audit["rule_violations"] = violations
        self._bump_counter("score_path_inconsistency_warning_count", 1)
        if int(self._score_path_audit.get("warning_count", 0)) <= 25:
            logging.warning("DE3 v3 score-path inconsistency: %s | detail=%s", code, detail)
        if self.observability_enabled and self.observability_strict_score_path_assertions:
            raise RuntimeError(f"DE3 v3 strict score-path assertion failed: {code} | {detail}")

    def _assess_score_path_consistency(
        self,
        *,
        invocation: int,
        eligible_rows: List[Dict[str, Any]],
        chosen_row: Optional[Dict[str, Any]],
        runner_up_row: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        candidate_count = int(len(eligible_rows))
        score_objects: List[Dict[str, Any]] = [self._family_score_object_from_row(row) for row in eligible_rows]
        score_obj_by_family = {
            str(obj.get("family_id", "") or ""): obj for obj in score_objects if str(obj.get("family_id", "") or "")
        }
        chosen_obj = self._family_score_object_from_row(chosen_row)
        if str(chosen_obj.get("family_id", "") or "") in score_obj_by_family:
            chosen_obj = score_obj_by_family.get(str(chosen_obj.get("family_id", "") or ""), chosen_obj)
        runner_up_obj = self._family_score_object_from_row(runner_up_row)
        if str(runner_up_obj.get("family_id", "") or "") in score_obj_by_family:
            runner_up_obj = score_obj_by_family.get(str(runner_up_obj.get("family_id", "") or ""), runner_up_obj)
        warnings: List[Dict[str, Any]] = []
        all_final_scores = [float(obj.get("final_family_score", 0.0)) for obj in score_objects]
        all_zero = bool(candidate_count > 1 and all(abs(score) <= 1e-12 for score in all_final_scores))
        all_equal = bool(
            candidate_count > 1
            and len(all_final_scores) > 1
            and all(abs(score - all_final_scores[0]) <= 1e-12 for score in all_final_scores[1:])
        )
        delta = float(chosen_obj.get("final_family_score", 0.0) - runner_up_obj.get("final_family_score", 0.0))
        zero_delta = bool(candidate_count > 1 and abs(delta) <= 1e-12)
        any_trusted_context = any(bool(obj.get("context_trusted_flag", False)) for obj in score_objects)
        any_adjustments_applied = any(
            bool(obj.get("any_post_processing_applied", False)) for obj in score_objects
        )
        any_diversity_nonzero = any(
            abs(float(obj.get("competition_diversity_adjustment", 0.0))) > 1e-12 for obj in score_objects
        )
        if all_zero:
            warnings.append(
                {
                    "code": "multi_candidate_all_zero_final_scores",
                    "candidate_count": int(candidate_count),
                }
            )
        if all_equal:
            warnings.append(
                {
                    "code": "multi_candidate_all_equal_final_scores",
                    "candidate_count": int(candidate_count),
                    "shared_score": float(all_final_scores[0]) if all_final_scores else 0.0,
                }
            )
        if zero_delta:
            warnings.append(
                {
                    "code": "multiple_candidates_but_zero_score_delta",
                    "candidate_count": int(candidate_count),
                    "chosen_family_id": str(chosen_obj.get("family_id", "")),
                    "runner_up_family_id": str(runner_up_obj.get("family_id", "")),
                }
            )
        if any_trusted_context and all(
            abs(float(obj.get("trusted_context_component", 0.0))) <= 1e-12 for obj in score_objects
        ):
            warnings.append(
                {
                    "code": "trusted_context_flagged_but_context_component_zero",
                    "candidate_count": int(candidate_count),
                }
            )
        if any_adjustments_applied and not any_diversity_nonzero:
            warnings.append(
                {
                    "code": "adjustments_flagged_but_diversity_component_zero",
                    "candidate_count": int(candidate_count),
                }
            )

        summary = self._score_path_audit.setdefault("summary", {})
        self._score_path_audit["decision_count"] = int(self._score_path_audit.get("decision_count", 0) + 1)
        if candidate_count > 1:
            summary["multi_candidate_decisions"] = int(summary.get("multi_candidate_decisions", 0) + 1)
        if all_zero:
            summary["multi_candidate_all_zero_final_score_count"] = int(
                summary.get("multi_candidate_all_zero_final_score_count", 0) + 1
            )
        if all_equal:
            summary["multi_candidate_all_equal_final_score_count"] = int(
                summary.get("multi_candidate_all_equal_final_score_count", 0) + 1
            )
        if zero_delta:
            summary["multiple_candidates_but_zero_score_delta_count"] = int(
                summary.get("multiple_candidates_but_zero_score_delta_count", 0) + 1
            )
        if any_adjustments_applied and not any_diversity_nonzero:
            summary["adjustments_applied_but_component_zero_count"] = int(
                summary.get("adjustments_applied_but_component_zero_count", 0) + 1
            )
        self._score_path_audit["summary"] = summary

        for warning in warnings:
            self._record_score_path_warning(
                invocation=int(invocation),
                code=str(warning.get("code", "unknown")),
                detail={
                    "candidate_count": int(candidate_count),
                    "chosen_family_id": str(chosen_obj.get("family_id", "")),
                    "runner_up_family_id": str(runner_up_obj.get("family_id", "")),
                    **dict(warning),
                },
            )
        return {
            "candidate_count": int(candidate_count),
            "has_multiple_candidates": bool(candidate_count > 1),
            "chosen_family_id": str(chosen_obj.get("family_id", "")),
            "runner_up_family_id": str(runner_up_obj.get("family_id", "")),
            "chosen_final_family_score": float(chosen_obj.get("final_family_score", 0.0)),
            "runner_up_final_family_score": float(runner_up_obj.get("final_family_score", 0.0)),
            "chosen_vs_runner_up_score_delta": float(delta if candidate_count > 1 else 0.0),
            "all_zero_final_scores": bool(all_zero),
            "all_equal_final_scores": bool(all_equal),
            "warnings": warnings,
        }

    def _flatten_config(self, value: Any, prefix: str = "") -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if isinstance(value, dict):
            for key, sub in value.items():
                key_text = str(key)
                next_prefix = f"{prefix}.{key_text}" if prefix else key_text
                out.update(self._flatten_config(sub, next_prefix))
        else:
            out[str(prefix)] = value
        return out

    def _config_snapshot_relevant(self) -> Dict[str, Any]:
        cfg = self.cfg if isinstance(self.cfg, dict) else {}
        out = {
            "enabled": bool(cfg.get("enabled", False)),
            "member_db_path": str(self.member_db_path),
            "family_db_path": str(self.family_db_path),
            "auto_build_family_db": bool(self.auto_build_family_db),
            "context_profiles": dict(cfg.get("context_profiles", {}) if isinstance(cfg.get("context_profiles"), dict) else {}),
            "refined_universe": dict(cfg.get("refined_universe", {}) if isinstance(cfg.get("refined_universe"), dict) else {}),
            "de3v3_core": dict(cfg.get("de3v3_core", {}) if isinstance(cfg.get("de3v3_core"), dict) else {}),
            "de3v3_satellites": dict(cfg.get("de3v3_satellites", {}) if isinstance(cfg.get("de3v3_satellites"), dict) else {}),
            "bloat_control": dict(cfg.get("bloat_control", {}) if isinstance(cfg.get("bloat_control"), dict) else {}),
            "prior_eligibility": dict(
                cfg.get("prior_eligibility", cfg.get("family_eligibility", {}))
                if isinstance(cfg.get("prior_eligibility", cfg.get("family_eligibility", {})), dict)
                else {}
            ),
            "usable_family_universe": dict(cfg.get("usable_family_universe", {}) if isinstance(cfg.get("usable_family_universe"), dict) else {}),
            "family_competition": dict(cfg.get("family_competition", {}) if isinstance(cfg.get("family_competition"), dict) else {}),
            "family_scoring": dict(cfg.get("family_scoring", {}) if isinstance(cfg.get("family_scoring"), dict) else {}),
            "local_member_selection": dict(cfg.get("local_member_selection", {}) if isinstance(cfg.get("local_member_selection"), dict) else {}),
            "observability": dict(cfg.get("observability", {}) if isinstance(cfg.get("observability"), dict) else {}),
        }
        return out

    @staticmethod
    def _file_fingerprint(path: Path) -> Dict[str, Any]:
        out = {
            "path": str(path),
            "exists": bool(path.exists()),
            "sha256": "",
            "size_bytes": 0,
            "mtime": None,
        }
        if not path.exists():
            return out
        try:
            hasher = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    hasher.update(chunk)
            st = path.stat()
            out["sha256"] = hasher.hexdigest()
            out["size_bytes"] = int(getattr(st, "st_size", 0) or 0)
            out["mtime"] = dt.datetime.fromtimestamp(float(getattr(st, "st_mtime", 0.0))).isoformat()
        except Exception as exc:
            out["error"] = str(exc)
        return out

    def _inspect_bundle_sections(self, path: Path) -> Dict[str, Any]:
        required_sections = [
            "raw_family_inventory",
            "refined_family_inventory",
            "core_summary",
            "t6_anchor_report",
            "core_families",
            "core_members",
            "satellite_candidates_raw",
            "satellite_candidates_refined",
            "satellite_quality_summary",
            "portfolio_incremental_tests",
            "orthogonality_summary",
            "runtime_core_satellite_state",
            "runtime_mode_summary",
            "family_quality_summary",
            "member_quality_summary",
            "anchor_members",
            "context_profiles",
            "runtime_state_defaults",
            "refinement_summary",
            "diagnostics_summary",
            "retained_runtime_universe",
            "suppressed_families",
            "suppressed_members",
            "family_cluster_distinctiveness",
            "metadata",
            "legacy_family_inventory",
        ]
        out = {
            "artifact_kind": "family_inventory",
            "sections_present": [],
            "section_sizes": {},
        }
        if not path.exists():
            return out
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return out
        if not isinstance(payload, dict):
            return out
        if isinstance(payload.get("families"), list):
            out["artifact_kind"] = "family_inventory"
            out["sections_present"] = ["families"]
            out["section_sizes"] = {"families": int(len(payload.get("families", []) or []))}
            return out
        out["artifact_kind"] = "bundle"
        present = []
        sizes: Dict[str, Any] = {}
        for section in required_sections:
            if section not in payload:
                continue
            present.append(section)
            section_val = payload.get(section)
            if isinstance(section_val, dict):
                if isinstance(section_val.get("families"), list):
                    sizes[section] = int(len(section_val.get("families", []) or []))
                elif isinstance(section_val.get("members"), list):
                    sizes[section] = int(len(section_val.get("members", []) or []))
                elif isinstance(section_val.get("count"), (int, float)):
                    sizes[section] = int(safe_float(section_val.get("count", 0), 0))
                else:
                    sizes[section] = int(len(section_val))
            elif isinstance(section_val, list):
                sizes[section] = int(len(section_val))
            else:
                sizes[section] = 1
        out["sections_present"] = present
        out["section_sizes"] = sizes
        return out

    def get_activation_audit(self) -> Dict[str, Any]:
        status = self.get_runtime_status()
        status_config_snapshot = self._config_snapshot_relevant()
        out = {
            "created_at": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(),
            "active_de3_version": "v3",
            "family_mode_active": bool(self.enabled),
            "active_mode_flags": {
                "family_mode_active": bool(self.enabled),
                "context_profiles_enabled": bool(self.context_profiles_enabled),
                "runtime_use_refined": bool(self.runtime_use_refined),
                "core_enabled": bool(self.core_enabled),
                "satellites_enabled": bool(self.satellites_enabled),
                "gates_enabled": bool(self.gates_enabled),
                "observability_enabled": bool(self.observability_enabled),
            },
            "family_artifact_path": str(status.get("family_artifact_path", self.family_db_path)),
            "family_artifact_kind": str(status.get("artifact_kind", "family_inventory")),
            "artifact_fingerprint": dict(self._artifact_fingerprint),
            "bundle_sections_present": list(self._bundle_sections_present),
            "bundle_section_sizes": dict(self._bundle_section_sizes),
            "refined_universe_mode_enabled": bool(self.refined_universe_enabled),
            "runtime_use_refined": bool(self.runtime_use_refined),
            "loaded_universe": str(status.get("loaded_universe", "raw")),
            "raw_universe_override_allowed": bool(self.allow_runtime_raw_universe_override),
            "raw_universe_override_active": bool(
                self.runtime_use_refined and str(status.get("loaded_universe", "raw")) in {"raw", "legacy"}
            ),
            "family_candidate_construction_mode": "family_first",
            "runtime_mode": str(self.core_runtime_mode),
            "core_anchor_family_ids": list(self.core_anchor_family_ids),
            "force_anchor_when_eligible": bool(self.force_anchor_when_eligible),
            "member_first_fallback_path_active": False,
            "fallback_paths": {
                "fallback_to_priors_when_profile_weak": bool(self.fallback_to_priors_when_profile_weak),
                "context_profiles_enabled": bool(self.context_profiles_enabled),
                "require_enriched_export_for_runtime": bool(self.cp_require_enriched_for_runtime),
                "enriched_export_required": bool(status.get("enriched_export_required", False)),
            },
            "family_scoring_weights": {
                "context_profile_expectancy": float(self.w_context_expectancy),
                "context_profile_confidence": float(self.w_context_confidence),
                "family_prior": float(self.w_prior),
                "v3_realized_usability": float(self.w_usability),
                "adaptive_policy": float(self.w_adaptive),
            },
            "family_scoring_controls": {
                "normalize_prior_component": bool(self.fs_normalize_prior_component),
                "cap_context_advantage_when_single_strong_family": bool(
                    self.fs_cap_context_advantage_when_single_strong_family
                ),
                "single_strong_family_context_cap": float(
                    self.fs_single_strong_family_context_cap
                ),
                "compatible_band_penalty": float(self.fs_compatible_band_penalty),
                "close_competition_margin": float(self.fs_close_competition_margin),
                "max_competition_adjustment_close": float(
                    self.fs_max_competition_adjustment_close
                ),
                "max_competition_adjustment_far": float(
                    self.fs_max_competition_adjustment_far
                ),
                "dominance_penalty_curve": str(self.fs_dominance_penalty_curve),
                "exploration_bonus_curve": str(self.fs_exploration_bonus_curve),
                "log_score_delta_ladder": bool(self.fs_log_score_delta_ladder),
            },
            "local_member_scoring_weights": {
                "edge_points": float(self.w_local_edge),
                "structural_score": float(self.w_local_struct),
                "payoff": float(self.w_local_payoff),
                "context_bracket_suitability": float(self.w_local_context_bracket),
                "confidence": float(self.w_local_conf),
            },
            "active_context_dimensions": list(self.active_context_dimensions),
            "context_trust": {
                "min_context_bucket_samples": int(self.min_context_bucket_samples),
                "strong_context_bucket_samples": int(self.strong_context_bucket_samples),
                "context_profile_weight_strong": float(self.context_profile_weight_strong),
                "context_profile_weight_mid": float(self.context_profile_weight_mid),
                "context_profile_weight_low_or_none": float(self.context_profile_weight_low_or_none),
                "require_meaningful_context_support_for_context_weight": bool(
                    self.require_meaningful_context_support_for_context_weight
                ),
                "low_support_context_weight_cap": float(self.low_support_context_weight_cap),
                "context_scale_by_evidence_tier": dict(self.context_scale_by_evidence_tier),
            },
            "local_bracket_adaptation": {
                "full_adaptation_min_support_tier": str(self.local_full_adaptation_min_support_tier),
                "conservative_adaptation_min_support_tier": str(self.local_conservative_adaptation_min_support_tier),
                "freeze_to_canonical_when_low_support": bool(self.local_freeze_to_canonical_low),
                "force_canonical_when_family_monopoly": bool(self.force_canonical_when_family_monopoly),
                "monopoly_share_threshold": float(self.monopoly_share_threshold),
                "monopoly_lookback_window": int(self.monopoly_lookback_window),
            },
            "family_competition_balance": {
                "enabled": bool(self.family_competition_balance_enabled),
                "dominance_window_size": int(self.dominance_window_size),
                "dominance_penalty_start_share": float(self.dominance_penalty_start_share),
                "dominance_penalty_max": float(self.dominance_penalty_max),
                "max_dominance_penalty": float(self.max_dominance_penalty),
                "low_support_exploration_bonus": float(self.low_support_exploration_bonus),
                "max_exploration_bonus": float(self.max_exploration_bonus),
                "exploration_bonus_decay_threshold": int(self.exploration_bonus_decay_threshold),
                "competition_margin_points": float(self.competition_margin_points),
                "cap_context_advantage_in_close_competition": bool(self.cap_context_advantage_in_close_competition),
                "max_context_advantage_cap": float(self.max_context_advantage_cap),
            },
            "bloat_control_flags": {
                "enable_family_competition_balancing": bool(
                    self.enable_family_competition_balancing
                ),
                "enable_exploration_bonus": bool(self.enable_exploration_bonus),
                "enable_dominance_penalty": bool(self.enable_dominance_penalty),
                "enable_monopoly_canonical_force": bool(
                    self.enable_monopoly_canonical_force
                ),
                "enable_compatibility_tier_slot_pressure": bool(
                    self.enable_compatibility_tier_slot_pressure
                ),
            },
            "family_candidate_compatibility": {
                "include_exact_and_compatible_only": bool(self.include_exact_and_compatible_only),
                "temporary_excluded_thresholds": sorted(
                    list(self.temp_excluded_thresholds)
                ),
                "temporary_excluded_family_ids": sorted(
                    list(self.temp_excluded_family_ids)
                ),
                "max_family_candidates_per_decision": int(self.max_family_candidates_per_decision),
                "compatible_family_max_count": int(self.compatible_family_max_count),
                "compatible_family_penalty": float(self.compatible_family_penalty),
                "family_candidate_cap": {
                    "enabled": bool(self.family_candidate_cap_enabled),
                    "max_total_candidates": int(self.cap_max_total_candidates),
                    "min_exact_match_candidates": int(self.cap_min_exact_match_candidates),
                    "min_compatible_band_candidates": int(self.cap_min_compatible_band_candidates),
                    "max_exact_match_candidates": int(self.cap_max_exact_match_candidates),
                    "max_compatible_band_candidates": int(self.cap_max_compatible_band_candidates),
                    "use_preliminary_score_for_cap": bool(self.cap_use_preliminary_score_for_cap),
                    "compatibility_penalty_exact": float(self.cap_preliminary_compatibility_penalty_exact),
                    "compatibility_penalty_compatible": float(self.cap_preliminary_compatibility_penalty_compatible),
                    "log_pre_cap_post_cap": bool(self.cap_log_pre_cap_post_cap),
                },
                "session_nearby_max_hour_distance": float(self.session_nearby_max_hour_distance),
                "timeframe_nearby_max_minutes_delta": int(self.timeframe_nearby_max_minutes_delta),
                "timeframe_nearby_max_ratio": float(self.timeframe_nearby_max_ratio),
                "strategy_type_allow_related": bool(self.strategy_type_allow_related),
            },
            "observability_settings_used": {
                "enabled": bool(self.observability_enabled),
                "emit_family_score_trace": bool(self.observability_emit_family_score_trace),
                "emit_member_resolution_audit": bool(self.observability_emit_member_resolution_audit),
                "emit_choice_path_audit": bool(self.observability_emit_choice_path_audit),
                "emit_score_path_audit": bool(self.observability_emit_score_path_audit),
                "strict_score_path_assertions": bool(self.observability_strict_score_path_assertions),
                "family_score_trace_max_rows": int(self.observability_family_score_trace_max_rows),
                "member_resolution_trace_max_rows": int(self.observability_member_resolution_max_rows),
            },
            "config_snapshot_relevant": status_config_snapshot,
            "config_snapshot_flat": self._flatten_config(status_config_snapshot, "DE3_V3"),
        }
        return out

    def get_bundle_usage_audit(self) -> Dict[str, Any]:
        if isinstance(self._bundle_usage_audit_cache, dict):
            return dict(self._bundle_usage_audit_cache)
        status = self.get_runtime_status()
        loaded_universe = str(status.get("loaded_universe", "raw") or "raw")
        present = set(self._bundle_sections_present)
        is_bundle = str(status.get("artifact_kind", "")) == "bundle"
        def _section_row(
            section_name: str,
            *,
            loaded: bool,
            referenced: bool,
            family_scoring: bool,
            member_scoring: bool,
            branching: bool,
        ) -> Dict[str, Any]:
            present_flag = bool(section_name in present)
            row = {
                "present": bool(present_flag),
                "loaded": bool(loaded and present_flag),
                "referenced_by_runtime": bool(referenced and present_flag),
                "affects_family_scoring": bool(family_scoring and present_flag),
                "affects_member_scoring": bool(member_scoring and present_flag),
                "affects_branching": bool(branching and present_flag),
                # Backward-compatible aliases.
                "used_in_family_scoring": bool(family_scoring and present_flag),
                "used_in_member_scoring": bool(member_scoring and present_flag),
                "used_in_branching": bool(branching and present_flag),
            }
            row["likely_inert"] = bool(
                row["present"]
                and (not row["referenced_by_runtime"])
                and (not row["affects_family_scoring"])
                and (not row["affects_member_scoring"])
                and (not row["affects_branching"])
            )
            return row

        section_usage = {
            "metadata": _section_row(
                "metadata",
                loaded=True,
                referenced=True,
                family_scoring=False,
                member_scoring=False,
                branching=True,
            ),
            "raw_family_inventory": _section_row(
                "raw_family_inventory",
                loaded=bool(loaded_universe == "raw"),
                referenced=bool(loaded_universe == "raw"),
                family_scoring=bool(loaded_universe == "raw"),
                member_scoring=bool(loaded_universe == "raw"),
                branching=False,
            ),
            "refined_family_inventory": _section_row(
                "refined_family_inventory",
                loaded=bool(str(loaded_universe).startswith("refined")),
                referenced=bool(str(loaded_universe).startswith("refined")),
                family_scoring=bool(str(loaded_universe).startswith("refined")),
                member_scoring=bool(str(loaded_universe).startswith("refined")),
                branching=bool(str(loaded_universe).startswith("refined")),
            ),
            "retained_runtime_universe": _section_row(
                "retained_runtime_universe",
                loaded=bool(loaded_universe == "retained_runtime"),
                referenced=bool(loaded_universe == "retained_runtime"),
                family_scoring=bool(loaded_universe == "retained_runtime"),
                member_scoring=bool(loaded_universe == "retained_runtime"),
                branching=bool(loaded_universe == "retained_runtime"),
            ),
            "core_families": _section_row(
                "core_families",
                loaded=bool(self.core_enabled),
                referenced=True,
                family_scoring=True,
                member_scoring=False,
                branching=True,
            ),
            "core_summary": _section_row(
                "core_summary",
                loaded=bool(self.core_enabled),
                referenced=True,
                family_scoring=False,
                member_scoring=False,
                branching=True,
            ),
            "t6_anchor_report": _section_row(
                "t6_anchor_report",
                loaded=bool(self.core_enabled),
                referenced=True,
                family_scoring=False,
                member_scoring=False,
                branching=True,
            ),
            "core_members": _section_row(
                "core_members",
                loaded=bool(self.core_enabled),
                referenced=True,
                family_scoring=False,
                member_scoring=True,
                branching=True,
            ),
            "satellite_candidates_raw": _section_row(
                "satellite_candidates_raw",
                loaded=bool(self.satellites_enabled),
                referenced=False,
                family_scoring=False,
                member_scoring=False,
                branching=False,
            ),
            "satellite_candidates_refined": _section_row(
                "satellite_candidates_refined",
                loaded=bool(self.satellites_enabled),
                referenced=True,
                family_scoring=True,
                member_scoring=True,
                branching=True,
            ),
            "satellite_quality_summary": _section_row(
                "satellite_quality_summary",
                loaded=bool(self.satellites_enabled),
                referenced=True,
                family_scoring=True,
                member_scoring=False,
                branching=True,
            ),
            "portfolio_incremental_tests": _section_row(
                "portfolio_incremental_tests",
                loaded=bool(self.satellites_enabled),
                referenced=True,
                family_scoring=False,
                member_scoring=False,
                branching=True,
            ),
            "orthogonality_summary": _section_row(
                "orthogonality_summary",
                loaded=bool(self.satellites_enabled),
                referenced=True,
                family_scoring=False,
                member_scoring=False,
                branching=True,
            ),
            "runtime_core_satellite_state": _section_row(
                "runtime_core_satellite_state",
                loaded=True,
                referenced=True,
                family_scoring=False,
                member_scoring=False,
                branching=True,
            ),
            "runtime_mode_summary": _section_row(
                "runtime_mode_summary",
                loaded=True,
                referenced=True,
                family_scoring=False,
                member_scoring=False,
                branching=True,
            ),
            "family_quality_summary": _section_row(
                "family_quality_summary",
                loaded=False,
                referenced=False,
                family_scoring=False,
                member_scoring=False,
                branching=False,
            ),
            "member_quality_summary": _section_row(
                "member_quality_summary",
                loaded=False,
                referenced=False,
                family_scoring=False,
                member_scoring=False,
                branching=False,
            ),
            "anchor_members": _section_row(
                "anchor_members",
                loaded=False,
                referenced=True,
                family_scoring=False,
                member_scoring=True,
                branching=True,
            ),
            "context_profiles": _section_row(
                "context_profiles",
                loaded=bool(self.context_profiles_enabled),
                referenced=bool(self.context_profiles_enabled),
                family_scoring=bool(self.use_context_profiles and self.context_profiles_enabled),
                member_scoring=False,
                branching=bool(self.use_context_profiles and self.context_profiles_enabled),
            ),
            "runtime_state_defaults": _section_row(
                "runtime_state_defaults",
                loaded=True,
                referenced=True,
                family_scoring=True,
                member_scoring=False,
                branching=True,
            ),
            "refinement_summary": _section_row(
                "refinement_summary",
                loaded=False,
                referenced=False,
                family_scoring=False,
                member_scoring=False,
                branching=False,
            ),
            "diagnostics_summary": _section_row(
                "diagnostics_summary",
                loaded=False,
                referenced=False,
                family_scoring=False,
                member_scoring=False,
                branching=False,
            ),
        }
        unused_sections = [sec for sec, row in section_usage.items() if bool(row.get("likely_inert", False))]
        sample_family_fields = set()
        for idx, row in enumerate(self.families_by_id.values()):
            if not isinstance(row, dict):
                continue
            sample_family_fields.update(str(k) for k in row.keys())
            if idx >= 7:
                break
        used_family_fields = {
            "family_id",
            "family_key",
            "family_priors",
            "family_context_profiles",
            "family_runtime_state",
            "members",
            "retained_member_ids",
            "canonical_representative_member",
            "family_retained",
            "family_quality_classification",
            "member_count",
        }
        unused_family_fields = sorted([field for field in sample_family_fields if field not in used_family_fields])
        payload = {
            "artifact_kind": str(status.get("artifact_kind", "family_inventory")),
            "is_bundle": bool(is_bundle),
            "loaded_universe": str(loaded_universe),
            "sections_present": list(self._bundle_sections_present),
            "section_sizes": dict(self._bundle_section_sizes),
            "section_usage": section_usage,
            "unused_or_inert_sections": unused_sections,
            "family_row_fields_sampled": sorted(list(sample_family_fields)),
            "family_row_fields_used_in_runtime": sorted(list(used_family_fields)),
            "family_row_fields_likely_inert": unused_family_fields,
        }
        self._bundle_usage_audit_cache = dict(payload)
        return payload

    def get_config_usage_audit(self) -> Dict[str, Any]:
        if isinstance(self._config_usage_audit_cache, dict):
            return dict(self._config_usage_audit_cache)
        cfg = self._config_snapshot_relevant()
        flat = self._flatten_config(cfg, "DE3_V3")
        read_map: Dict[str, Dict[str, bool]] = {
            "DE3_V3.enabled": {"scoring": False, "branching": True},
            "DE3_V3.member_db_path": {"scoring": False, "branching": True},
            "DE3_V3.family_db_path": {"scoring": False, "branching": True},
            "DE3_V3.auto_build_family_db": {"scoring": False, "branching": True},
            "DE3_V3.refined_universe.enabled": {"scoring": False, "branching": True},
            "DE3_V3.refined_universe.runtime_use_refined": {"scoring": False, "branching": True},
            "DE3_V3.refined_universe.allow_runtime_raw_universe_override": {"scoring": False, "branching": True},
            "DE3_V3.refined_universe.require_meaningful_context_support_for_context_weight": {"scoring": True, "branching": True},
            "DE3_V3.refined_universe.low_support_context_weight_cap": {"scoring": True, "branching": False},
            "DE3_V3.de3v3_core.enabled": {"scoring": False, "branching": True},
            "DE3_V3.de3v3_core.anchor_family_ids": {"scoring": False, "branching": True},
            "DE3_V3.de3v3_core.default_runtime_mode": {"scoring": False, "branching": True},
            "DE3_V3.de3v3_core.core_mode": {"scoring": False, "branching": True},
            "DE3_V3.de3v3_core.force_anchor_when_eligible": {"scoring": False, "branching": True},
            "DE3_V3.de3v3_satellites.enabled": {"scoring": False, "branching": True},
            "DE3_V3.de3v3_satellites.discovery_enabled": {"scoring": False, "branching": True},
            "DE3_V3.de3v3_satellites.min_standalone_viability": {"scoring": False, "branching": True},
            "DE3_V3.de3v3_satellites.min_incremental_value_over_core": {"scoring": False, "branching": True},
            "DE3_V3.de3v3_satellites.max_retained_satellites": {"scoring": False, "branching": True},
            "DE3_V3.de3v3_satellites.require_orthogonality": {"scoring": False, "branching": True},
            "DE3_V3.de3v3_satellites.max_overlap_with_core": {"scoring": False, "branching": True},
            "DE3_V3.de3v3_satellites.max_bad_overlap_with_core": {"scoring": False, "branching": True},
            "DE3_V3.de3v3_satellites.allow_near_core_variants_if_incremental": {"scoring": False, "branching": True},
            "DE3_V3.bloat_control.enable_family_competition_balancing": {"scoring": False, "branching": True},
            "DE3_V3.bloat_control.enable_exploration_bonus": {"scoring": True, "branching": True},
            "DE3_V3.bloat_control.enable_dominance_penalty": {"scoring": True, "branching": True},
            "DE3_V3.bloat_control.enable_monopoly_canonical_force": {"scoring": False, "branching": True},
            "DE3_V3.bloat_control.enable_compatibility_tier_slot_pressure": {"scoring": False, "branching": True},
            "DE3_V3.context_profiles.enabled": {"scoring": False, "branching": True},
            "DE3_V3.context_profiles.min_bucket_samples": {"scoring": True, "branching": True},
            "DE3_V3.context_profiles.strong_bucket_samples": {"scoring": True, "branching": True},
            "DE3_V3.context_profiles.require_enriched_export_for_runtime": {"scoring": False, "branching": True},
            "DE3_V3.usable_family_universe.enabled": {"scoring": False, "branching": True},
            "DE3_V3.usable_family_universe.exclude_only_suppressed_families": {"scoring": False, "branching": True},
            "DE3_V3.usable_family_universe.low_support_fully_competitive": {"scoring": False, "branching": True},
            "DE3_V3.usable_family_universe.evidence_support.min_mid_samples": {"scoring": False, "branching": True},
            "DE3_V3.usable_family_universe.evidence_support.strong_samples": {"scoring": False, "branching": True},
            "DE3_V3.usable_family_universe.evidence_adjustment.context_scale_by_evidence_tier.none": {"scoring": True, "branching": False},
            "DE3_V3.usable_family_universe.evidence_adjustment.context_scale_by_evidence_tier.low": {"scoring": True, "branching": False},
            "DE3_V3.usable_family_universe.evidence_adjustment.context_scale_by_evidence_tier.mid": {"scoring": True, "branching": False},
            "DE3_V3.usable_family_universe.evidence_adjustment.context_scale_by_evidence_tier.strong": {"scoring": True, "branching": False},
            "DE3_V3.family_competition.use_bootstrap_family_competition_floor": {"scoring": False, "branching": True},
            "DE3_V3.family_competition.bootstrap_min_competing_families": {"scoring": False, "branching": True},
            "DE3_V3.family_competition.include_exact_and_compatible_only": {"scoring": False, "branching": True},
            "DE3_V3.family_competition.temporary_excluded_thresholds": {"scoring": False, "branching": True},
            "DE3_V3.family_competition.temporary_excluded_family_ids": {"scoring": False, "branching": True},
            "DE3_V3.family_competition.max_family_candidates_per_decision": {"scoring": False, "branching": True},
            "DE3_V3.family_competition.compatible_family_max_count": {"scoring": False, "branching": True},
            "DE3_V3.family_competition.compatible_family_penalty": {"scoring": True, "branching": False},
            "DE3_V3.family_competition.family_candidate_cap.enabled": {"scoring": False, "branching": True},
            "DE3_V3.family_competition.family_candidate_cap.max_total_candidates": {"scoring": False, "branching": True},
            "DE3_V3.family_competition.family_candidate_cap.min_exact_match_candidates": {"scoring": False, "branching": True},
            "DE3_V3.family_competition.family_candidate_cap.min_compatible_band_candidates": {"scoring": False, "branching": True},
            "DE3_V3.family_competition.family_candidate_cap.max_exact_match_candidates": {"scoring": False, "branching": True},
            "DE3_V3.family_competition.family_candidate_cap.max_compatible_band_candidates": {"scoring": False, "branching": True},
            "DE3_V3.family_competition.family_candidate_cap.use_preliminary_score_for_cap": {"scoring": False, "branching": True},
            "DE3_V3.family_competition.family_candidate_cap.compatibility_penalty_exact": {"scoring": True, "branching": False},
            "DE3_V3.family_competition.family_candidate_cap.compatibility_penalty_compatible": {"scoring": True, "branching": False},
            "DE3_V3.family_competition.family_candidate_cap.log_pre_cap_post_cap": {"scoring": False, "branching": True},
            "DE3_V3.family_competition.compatibility_bands.session_nearby_max_hour_distance": {"scoring": False, "branching": True},
            "DE3_V3.family_competition.compatibility_bands.timeframe_nearby_max_minutes_delta": {"scoring": False, "branching": True},
            "DE3_V3.family_competition.compatibility_bands.timeframe_nearby_max_ratio": {"scoring": False, "branching": True},
            "DE3_V3.family_competition.compatibility_bands.strategy_type_allow_related": {"scoring": False, "branching": True},
            "DE3_V3.family_competition.family_competition_balance.enabled": {"scoring": False, "branching": True},
            "DE3_V3.family_competition.family_competition_balance.dominance_window_size": {"scoring": False, "branching": True},
            "DE3_V3.family_competition.family_competition_balance.dominance_penalty_start_share": {"scoring": True, "branching": False},
            "DE3_V3.family_competition.family_competition_balance.dominance_penalty_max": {"scoring": True, "branching": False},
            "DE3_V3.family_competition.family_competition_balance.low_support_exploration_bonus": {"scoring": True, "branching": False},
            "DE3_V3.family_competition.family_competition_balance.competition_margin_points": {"scoring": False, "branching": True},
            "DE3_V3.family_competition.family_competition_balance.cap_context_advantage_in_close_competition": {"scoring": False, "branching": True},
            "DE3_V3.family_competition.family_competition_balance.max_context_advantage_cap": {"scoring": True, "branching": False},
            "DE3_V3.family_scoring.weights.context_profile_expectancy": {"scoring": True, "branching": False},
            "DE3_V3.family_scoring.weights.context_profile_confidence": {"scoring": True, "branching": False},
            "DE3_V3.family_scoring.weights.family_prior": {"scoring": True, "branching": False},
            "DE3_V3.family_scoring.weights.v3_realized_usability": {"scoring": True, "branching": False},
            "DE3_V3.family_scoring.weights.adaptive_policy": {"scoring": True, "branching": False},
            "DE3_V3.family_scoring.normalize_prior_component": {"scoring": True, "branching": False},
            "DE3_V3.family_scoring.cap_context_advantage_when_single_strong_family": {"scoring": True, "branching": True},
            "DE3_V3.family_scoring.single_strong_family_context_cap": {"scoring": True, "branching": False},
            "DE3_V3.family_scoring.compatible_band_penalty": {"scoring": True, "branching": False},
            "DE3_V3.family_scoring.close_competition_margin": {"scoring": False, "branching": True},
            "DE3_V3.family_scoring.max_competition_adjustment_close": {"scoring": True, "branching": False},
            "DE3_V3.family_scoring.max_competition_adjustment_far": {"scoring": True, "branching": False},
            "DE3_V3.family_scoring.dominance_penalty_curve": {"scoring": True, "branching": False},
            "DE3_V3.family_scoring.exploration_bonus_curve": {"scoring": True, "branching": False},
            "DE3_V3.family_scoring.log_score_delta_ladder": {"scoring": False, "branching": False},
            "DE3_V3.family_scoring.active_context_dimensions": {"scoring": True, "branching": True},
            "DE3_V3.family_scoring.use_context_profiles": {"scoring": False, "branching": True},
            "DE3_V3.family_scoring.fallback_to_priors_when_profile_weak": {"scoring": False, "branching": True},
            "DE3_V3.family_scoring.min_context_bucket_samples": {"scoring": True, "branching": True},
            "DE3_V3.family_scoring.strong_context_bucket_samples": {"scoring": True, "branching": True},
            "DE3_V3.family_scoring.context_profile_weight_strong": {"scoring": True, "branching": False},
            "DE3_V3.family_scoring.context_profile_weight_mid": {"scoring": True, "branching": False},
            "DE3_V3.family_scoring.context_profile_weight_low_or_none": {"scoring": True, "branching": False},
            "DE3_V3.family_scoring.gates.enabled": {"scoring": False, "branching": True},
            "DE3_V3.family_scoring.gates.min_family_score": {"scoring": False, "branching": True},
            "DE3_V3.family_scoring.gates.min_adaptive_component": {"scoring": False, "branching": True},
            "DE3_V3.local_member_selection.weights.edge_points": {"scoring": True, "branching": False},
            "DE3_V3.local_member_selection.weights.structural_score": {"scoring": True, "branching": False},
            "DE3_V3.local_member_selection.weights.payoff": {"scoring": True, "branching": False},
            "DE3_V3.local_member_selection.weights.context_bracket_suitability": {"scoring": True, "branching": False},
            "DE3_V3.local_member_selection.weights.confidence": {"scoring": True, "branching": False},
            "DE3_V3.local_member_selection.full_adaptation_min_support_tier": {"scoring": False, "branching": True},
            "DE3_V3.local_member_selection.conservative_adaptation_min_support_tier": {"scoring": False, "branching": True},
            "DE3_V3.local_member_selection.freeze_to_canonical_when_low_support": {"scoring": False, "branching": True},
            "DE3_V3.local_member_selection.force_canonical_when_family_monopoly": {"scoring": False, "branching": True},
            "DE3_V3.local_member_selection.monopoly_share_threshold": {"scoring": False, "branching": True},
            "DE3_V3.local_member_selection.monopoly_lookback_window": {"scoring": False, "branching": True},
            "DE3_V3.observability.enabled": {"scoring": False, "branching": True},
            "DE3_V3.observability.emit_family_score_trace": {"scoring": False, "branching": True},
            "DE3_V3.observability.emit_member_resolution_audit": {"scoring": False, "branching": True},
            "DE3_V3.observability.emit_choice_path_audit": {"scoring": False, "branching": True},
            "DE3_V3.observability.emit_score_path_audit": {"scoring": False, "branching": True},
            "DE3_V3.observability.strict_score_path_assertions": {"scoring": False, "branching": True},
            "DE3_V3.observability.family_score_trace_max_rows": {"scoring": False, "branching": True},
            "DE3_V3.observability.member_resolution_trace_max_rows": {"scoring": False, "branching": True},
        }
        all_keys = sorted(set(list(flat.keys()) + list(read_map.keys())))
        rows: Dict[str, Dict[str, Any]] = {}
        nested_balance = self.cfg.get("family_competition", {}) if isinstance(self.cfg.get("family_competition"), dict) else {}
        nested_balance_cfg = nested_balance.get("family_competition_balance", {}) if isinstance(nested_balance.get("family_competition_balance"), dict) else {}
        for key in all_keys:
            defined = key in flat
            read_meta = read_map.get(key, None)
            read = bool(read_meta is not None)
            affects_scoring = bool((read_meta or {}).get("scoring", False))
            affects_member_scoring = bool(
                (read_meta or {}).get("member_scoring", False)
                or key.startswith("DE3_V3.local_member_selection.weights.")
            )
            affects_branching = bool((read_meta or {}).get("branching", False))
            overridden = False
            if key == "DE3_V3.refined_universe.runtime_use_refined" and (not self.refined_universe_enabled):
                overridden = True
            if key.endswith(".context_profile_weight_low_or_none") and self.require_meaningful_context_support_for_context_weight:
                raw_val = safe_float(flat.get(key, self.context_profile_weight_low_or_none), self.context_profile_weight_low_or_none)
                if raw_val > self.low_support_context_weight_cap:
                    overridden = True
            if key.endswith(".context_scale_by_evidence_tier.low") and self.require_meaningful_context_support_for_context_weight:
                raw_val = safe_float(flat.get(key, self.context_scale_by_evidence_tier.get("low", 0.0)), self.context_scale_by_evidence_tier.get("low", 0.0))
                if raw_val > self.low_support_context_weight_cap:
                    overridden = True
            if key.startswith("DE3_V3.family_competition.") and ".family_competition_balance." not in key:
                tail = key.split("DE3_V3.family_competition.", 1)[-1]
                if tail in {"dominance_window_size", "dominance_penalty_start_share", "dominance_penalty_max", "max_dominance_penalty", "low_support_exploration_bonus", "max_exploration_bonus", "exploration_bonus_decay_threshold", "competition_margin_points", "cap_context_advantage_in_close_competition", "max_context_advantage_cap"}:
                    if tail in nested_balance_cfg:
                        overridden = True
            defaulted = bool(read and (not defined))
            likely_inert = bool(
                (not read)
                or overridden
                or (
                    read
                    and (not affects_scoring)
                    and (not affects_member_scoring)
                    and (not affects_branching)
                )
            )
            if not read:
                reason_if_inert = "not_read_by_runtime"
            elif overridden:
                reason_if_inert = "overridden_or_shadowed"
            elif (not affects_scoring) and (not affects_member_scoring) and (not affects_branching):
                reason_if_inert = "read_but_non_behavioral"
            else:
                reason_if_inert = ""
            rows[key] = {
                "defined_in_config": bool(defined),
                "read_by_runtime": bool(read),
                "affects_scoring": bool(affects_scoring),
                "affects_family_scoring": bool(affects_scoring),
                "affects_member_scoring": bool(affects_member_scoring),
                "affects_branching": bool(affects_branching),
                "overridden_or_shadowed": bool(overridden),
                "defaulted": bool(defaulted),
                "likely_inert": bool(likely_inert),
                "reason_if_inert": str(reason_if_inert),
                "resolved_value": flat.get(key),
            }
        payload = {
            "created_at": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(),
            "key_count": int(len(rows)),
            "keys": rows,
            "likely_inert_keys": [k for k, v in rows.items() if bool(v.get("likely_inert", False))],
        }
        self._config_usage_audit_cache = dict(payload)
        return payload

    def annotate_candidate_family(
        self,
        candidate: Dict[str, Any],
        *,
        default_session: Optional[str],
    ) -> Tuple[str, Dict[str, Any], Optional[Dict[str, Any]]]:
        family_key = build_family_key_from_candidate(candidate, default_session=default_session)
        family_id = family_id_from_key(family_key)
        inventory_family = self.families_by_id.get(family_id)
        candidate["de3_family_id"] = family_id
        candidate["de3_family_key"] = family_key
        return family_id, family_key, inventory_family

    @staticmethod
    def _canonical_member_id(family_inventory_row: Optional[Dict[str, Any]]) -> str:
        if not isinstance(family_inventory_row, dict):
            return ""
        cached = family_inventory_row.get("_cached_canonical_member_id")
        if isinstance(cached, str):
            return cached
        canonical = (
            family_inventory_row.get("canonical_representative_member")
            if isinstance(family_inventory_row.get("canonical_representative_member"), dict)
            else {}
        )
        out = str(canonical.get("member_id", "") or "").strip()
        family_inventory_row["_cached_canonical_member_id"] = out
        return out

    def _canonical_member_shape_key(self, family_inventory_row: Optional[Dict[str, Any]]) -> str:
        if not isinstance(family_inventory_row, dict):
            return ""
        cached = family_inventory_row.get("_cached_canonical_member_shape_key")
        if isinstance(cached, str):
            return cached
        canonical = (
            family_inventory_row.get("canonical_representative_member")
            if isinstance(family_inventory_row.get("canonical_representative_member"), dict)
            else {}
        )
        key = self._member_shape_key(canonical.get("sl"), canonical.get("tp"))
        family_inventory_row["_cached_canonical_member_shape_key"] = key
        return key

    def _family_retained_for_runtime(
        self,
        family_inventory_row: Optional[Dict[str, Any]],
        *,
        family_id: Optional[str] = None,
    ) -> bool:
        fid = str(family_id or "").strip()
        if self.core_enabled and fid and fid in set(self.core_family_ids_loaded or self.core_anchor_family_ids):
            return True
        if not self.runtime_use_refined:
            return True
        if not isinstance(family_inventory_row, dict):
            return False
        if "family_retained" in family_inventory_row:
            return bool(family_inventory_row.get("family_retained", False))
        quality_class = str(family_inventory_row.get("family_quality_classification", "") or "").strip().lower()
        if quality_class in {"strong_family", "keep_family"}:
            return True
        if quality_class in {"weak_family", "suppress_family"}:
            return False
        # If no explicit refinement fields exist, keep compatible behavior.
        return True

    def _family_role(self, family_id: str) -> str:
        fid = str(family_id or "").strip()
        core_ids = set(self.core_family_ids_loaded or self.core_anchor_family_ids)
        if self.core_enabled and fid and fid in core_ids:
            return "core"
        return "satellite"

    def _runtime_universe_family_items(self) -> List[Tuple[str, Dict[str, Any]]]:
        items: List[Tuple[str, Dict[str, Any]]] = []
        core_ids = set(self.core_family_ids_loaded or self.core_anchor_family_ids)
        retained_sat_ids = set(self.satellite_retained_family_ids_loaded or set())
        mode = str(self.core_runtime_mode or "core_plus_satellites").strip().lower()
        if mode not in {"core_only", "core_plus_satellites", "satellites_only"}:
            mode = "core_plus_satellites"
        for family_id, row in self.families_by_id.items():
            if not isinstance(row, dict):
                continue
            fid = str(family_id or "").strip()
            if not fid:
                continue
            role = "core" if (self.core_enabled and fid in core_ids) else "satellite"
            include = False
            if mode == "core_only":
                include = bool(role == "core")
            elif mode == "satellites_only":
                include = bool(role == "satellite" and self.satellites_enabled)
            else:
                include = bool(
                    (role == "core")
                    or (role == "satellite" and self.satellites_enabled)
                )
            if not include:
                continue
            if role == "satellite":
                if retained_sat_ids and fid not in retained_sat_ids:
                    continue
                if not self._family_retained_for_runtime(row, family_id=fid):
                    continue
            items.append((fid, row))
            row["_runtime_family_role"] = role
        items.sort(
            key=lambda item: (
                0 if self._family_role(item[0]) == "core" else 1,
                item[0],
            )
        )
        return items

    @staticmethod
    def _entry_member_id_candidates(entry: Dict[str, Any]) -> List[str]:
        out: List[str] = []
        cand_id = str(entry.get("cand_id", "") or "").strip()
        if cand_id:
            out.append(cand_id)
        cand = entry.get("cand") if isinstance(entry.get("cand"), dict) else {}
        strategy_id = str(cand.get("strategy_id", "") or "").strip()
        member_id = str(cand.get("member_id", "") or "").strip()
        if strategy_id:
            out.append(strategy_id)
        if member_id and member_id not in out:
            out.append(member_id)
        return out

    @staticmethod
    def _member_shape_key(sl_value: Any, tp_value: Any) -> str:
        sl = safe_float(sl_value, float("nan"))
        tp = safe_float(tp_value, float("nan"))
        if not (math.isfinite(sl) and math.isfinite(tp)):
            return ""
        return f"SL{sl:.6f}|TP{tp:.6f}"

    def _entry_member_shape_key(self, entry: Dict[str, Any]) -> str:
        cand = entry.get("cand") if isinstance(entry.get("cand"), dict) else {}
        return self._member_shape_key(cand.get("sl"), cand.get("tp"))

    def _retained_member_shape_keys(self, family_inventory_row: Optional[Dict[str, Any]]) -> Optional[set]:
        if not self.runtime_use_refined or not isinstance(family_inventory_row, dict):
            return None
        if "_cached_retained_member_shape_keys" in family_inventory_row:
            cached = family_inventory_row.get("_cached_retained_member_shape_keys")
            return cached if isinstance(cached, set) else None
        out: set = set()
        members = family_inventory_row.get("members")
        if isinstance(members, list):
            for member in members:
                if not isinstance(member, dict):
                    continue
                member_retained = bool(member.get("member_retained", True))
                if not member_retained:
                    continue
                key = self._member_shape_key(member.get("sl"), member.get("tp"))
                if key:
                    out.add(key)
        if not out:
            # Fallback for artifacts that only provide retained_member_ids.
            canonical = (
                family_inventory_row.get("canonical_representative_member")
                if isinstance(family_inventory_row.get("canonical_representative_member"), dict)
                else {}
            )
            key = self._member_shape_key(canonical.get("sl"), canonical.get("tp"))
            if key:
                out.add(key)
        family_inventory_row["_cached_retained_member_shape_keys"] = out if out else None
        return out if out else None

    def _retained_member_ids(self, family_inventory_row: Optional[Dict[str, Any]]) -> Optional[set]:
        if not self.runtime_use_refined or not isinstance(family_inventory_row, dict):
            return None
        if "_cached_retained_member_ids" in family_inventory_row:
            cached = family_inventory_row.get("_cached_retained_member_ids")
            return cached if isinstance(cached, set) else None
        ids: set = set()
        raw_ids = family_inventory_row.get("retained_member_ids")
        if isinstance(raw_ids, list):
            for value in raw_ids:
                key = str(value or "").strip()
                if key:
                    ids.add(key)
        members = family_inventory_row.get("members")
        if isinstance(members, list):
            for member in members:
                if not isinstance(member, dict):
                    continue
                member_retained = bool(member.get("member_retained", True))
                if not member_retained and ids:
                    continue
                key = str(member.get("member_id", "") or member.get("strategy_id", "") or "").strip()
                if key:
                    ids.add(key)
        out = ids if ids else None
        family_inventory_row["_cached_retained_member_ids"] = out
        return out

    def _entry_member_allowed_in_refined_universe(
        self,
        *,
        entry: Dict[str, Any],
        family_inventory_row: Optional[Dict[str, Any]],
    ) -> bool:
        if not self.runtime_use_refined:
            return True
        if not self._family_retained_for_runtime(family_inventory_row):
            return False
        retained_ids = self._retained_member_ids(family_inventory_row)
        retained_shapes = self._retained_member_shape_keys(family_inventory_row)
        if not retained_ids and not retained_shapes:
            return True
        member_ids = self._entry_member_id_candidates(entry)
        if retained_ids and any(member_id in retained_ids for member_id in member_ids):
            return True
        entry_shape = self._entry_member_shape_key(entry)
        if retained_shapes and entry_shape and entry_shape in retained_shapes:
            return True
        return False

    # Backward-compatible alias.
    def _entry_allowed_in_refined_universe(
        self,
        *,
        entry: Dict[str, Any],
        family_inventory_row: Optional[Dict[str, Any]],
    ) -> bool:
        return self._entry_member_allowed_in_refined_universe(
            entry=entry,
            family_inventory_row=family_inventory_row,
        )

    @staticmethod
    def _session_window_hours(session_value: Any) -> Optional[Tuple[int, int]]:
        raw = str(session_value or "").strip()
        if not raw or "-" not in raw:
            return None
        try:
            left, right = raw.split("-", 1)
            start = int(str(left).strip())
            end = int(str(right).strip())
        except Exception:
            return None
        if not (0 <= start <= 24 and 0 <= end <= 24):
            return None
        return start, end

    @staticmethod
    def _hour_in_session_window(hour_et: int, session_window: Optional[Tuple[int, int]]) -> bool:
        if session_window is None:
            return True
        start, end = session_window
        h = int(hour_et) % 24
        s = int(start) % 24
        e = int(end) % 24
        if s == e:
            return True
        if s < e:
            return bool(s <= h < e)
        return bool(h >= s or h < e)

    @staticmethod
    def _session_window_center(session_window: Optional[Tuple[int, int]]) -> Optional[float]:
        if session_window is None:
            return None
        start, end = session_window
        s = int(start) % 24
        e = int(end) % 24
        if s == e:
            return 12.0
        span = (e - s) % 24
        return float((s + (0.5 * span)) % 24)

    @staticmethod
    def _window_center_distance_hours(
        left: Optional[Tuple[int, int]],
        right: Optional[Tuple[int, int]],
    ) -> Optional[float]:
        c_left = DE3V3FamilyRuntime._session_window_center(left)
        c_right = DE3V3FamilyRuntime._session_window_center(right)
        if c_left is None or c_right is None:
            return None
        delta = abs(float(c_left) - float(c_right))
        return float(min(delta, 24.0 - delta))

    def _session_compatibility_tier(
        self,
        *,
        family_session: Any,
        decision_session: Any,
        sessions_seen: Optional[set],
        decision_hour_et: Optional[int],
    ) -> str:
        fam = str(family_session or "").strip()
        dec = str(decision_session or "").strip()
        if not fam:
            return "exact"
        if dec and fam == dec:
            return "exact"
        seen_sessions = sessions_seen if isinstance(sessions_seen, set) else set()
        if fam in seen_sessions:
            return "exact"
        fam_window = self._session_window_hours(fam)
        if (
            decision_hour_et is not None
            and self._hour_in_session_window(int(decision_hour_et), fam_window)
        ):
            return "compatible"
        # Coarse fallback: allow overlapping/nearby windows if both parse cleanly.
        dec_window = self._session_window_hours(dec)
        if fam_window is not None and dec_window is not None:
            for hour in range(24):
                if self._hour_in_session_window(hour, fam_window) and self._hour_in_session_window(hour, dec_window):
                    return "compatible"
            center_distance = self._window_center_distance_hours(fam_window, dec_window)
            if (
                center_distance is not None
                and center_distance <= float(self.session_nearby_max_hour_distance)
            ):
                return "compatible"
        if not dec and decision_hour_et is None and not seen_sessions:
            return "compatible"
        return "incompatible"

    @staticmethod
    def _parse_timeframe_minutes(value: Any) -> Optional[int]:
        raw = str(value or "").strip().lower()
        if not raw:
            return None
        digits = "".join(ch for ch in raw if ch.isdigit())
        if not digits:
            return None
        try:
            minutes = int(digits)
        except Exception:
            return None
        if minutes <= 0:
            return None
        return int(minutes)

    def _timeframe_compatibility_tier(
        self,
        *,
        family_timeframe: Any,
        timeframes_seen: Optional[set],
    ) -> str:
        fam = str(family_timeframe or "").strip().lower()
        seen = timeframes_seen if isinstance(timeframes_seen, set) else set()
        seen_norm = {str(v or "").strip().lower() for v in seen if str(v or "").strip()}
        if not fam:
            return "exact"
        if fam in seen_norm:
            return "exact"
        fam_minutes = self._parse_timeframe_minutes(fam)
        seen_minutes = [
            self._parse_timeframe_minutes(v) for v in seen_norm
        ]
        seen_minutes = [int(v) for v in seen_minutes if isinstance(v, int) and v > 0]
        if not seen_norm:
            return "exact"
        if fam_minutes is None or not seen_minutes:
            return "compatible"
        min_delta = min(abs(int(fam_minutes) - int(v)) for v in seen_minutes)
        min_ratio = min(
            max(float(fam_minutes), float(v)) / float(max(1, min(int(fam_minutes), int(v))))
            for v in seen_minutes
        )
        if min_delta == 0:
            return "exact"
        if (
            min_delta <= int(self.timeframe_nearby_max_minutes_delta)
            or min_ratio <= float(self.timeframe_nearby_max_ratio)
        ):
            return "compatible"
        return "incompatible"

    @staticmethod
    def _strategy_type_core_token(value: Any) -> str:
        raw = str(value or "").strip().upper()
        if not raw:
            return ""
        token = raw.replace("-", "_").replace(" ", "_")
        parts = [p for p in token.split("_") if p]
        parts = [p for p in parts if p not in {"LONG", "SHORT"}]
        if not parts:
            return token
        return parts[-1]

    def _strategy_type_compatibility_tier(
        self,
        *,
        family_strategy_type: Any,
        strategy_types_seen: Optional[set],
    ) -> str:
        fam = str(family_strategy_type or "").strip()
        seen = strategy_types_seen if isinstance(strategy_types_seen, set) else set()
        seen_norm = {str(v or "").strip() for v in seen if str(v or "").strip()}
        if not fam:
            return "exact"
        if fam in seen_norm:
            return "exact"
        if not seen_norm:
            return "exact"
        fam_core = self._strategy_type_core_token(fam)
        seen_core = {self._strategy_type_core_token(v) for v in seen_norm}
        if fam_core and fam_core in seen_core:
            return "compatible"
        if not self.strategy_type_allow_related:
            return "incompatible"
        related_groups = [
            {"REV", "REVERSION", "MEANREV", "MEAN_REVERSION", "MR"},
            {"TREND", "MOM", "MOMO", "MOMENTUM", "CONT", "CONTINUATION"},
            {"BRK", "BREAK", "BREAKOUT", "BO"},
        ]
        for group in related_groups:
            if fam_core in group and any(core in group for core in seen_core):
                return "compatible"
        # Broad fallback: if no clear contradictory subtype, keep as compatible-band.
        return "compatible"

    def _compatibility_component_from_tiers(
        self,
        *,
        session_tier: str,
        timeframe_tier: str,
        strategy_type_tier: str,
    ) -> float:
        compatible_dims = 0
        for tier in (session_tier, timeframe_tier, strategy_type_tier):
            if str(tier or "").strip().lower() == "compatible":
                compatible_dims += 1
        if compatible_dims <= 0:
            return 0.0
        # Compatible-band families should be penalized, but not auto-defeated.
        per_dim_penalty = float(self.fs_compatible_band_penalty)
        return float(per_dim_penalty * (float(compatible_dims) / 3.0))

    def _decision_signature(
        self,
        *,
        feasible_candidates: List[Dict[str, Any]],
        default_session: Optional[str],
        context_inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        sessions: set = set()
        sides: set = set()
        timeframes: set = set()
        strategy_types: set = set()
        thresholds: set = set()
        for entry in feasible_candidates:
            cand = entry.get("cand") if isinstance(entry.get("cand"), dict) else {}
            family_id, family_key, _ = self.annotate_candidate_family(cand, default_session=default_session)
            _ = family_id  # keeps annotate side effects for consistency.
            if isinstance(family_key, dict):
                session_v = str(family_key.get("session", "") or "").strip()
                side_v = str(family_key.get("side", "") or "").strip().lower()
                tf_v = str(family_key.get("timeframe", "") or "").strip().lower()
                type_v = str(family_key.get("de3_strategy_type", "") or "").strip()
                thresh_v = str(family_key.get("threshold", "") or "").strip().upper()
                if session_v:
                    sessions.add(session_v)
                if side_v:
                    sides.add(side_v)
                if tf_v:
                    timeframes.add(tf_v)
                if type_v:
                    strategy_types.add(type_v)
                if thresh_v:
                    thresholds.add(thresh_v)
        decision_session = str(
            (
                context_inputs.get("session")
                if isinstance(context_inputs, dict) and context_inputs.get("session") is not None
                else (default_session or "")
            )
            or ""
        ).strip()
        if decision_session:
            sessions.add(decision_session)
        decision_hour = None
        if isinstance(context_inputs, dict):
            try:
                decision_hour = int(safe_float(context_inputs.get("hour_et"), float("nan")))
                if decision_hour < 0 or decision_hour > 23:
                    decision_hour = None
            except Exception:
                decision_hour = None
        return {
            "sessions": sessions,
            "sides": sides,
            "timeframes": timeframes,
            "strategy_types": strategy_types,
            "thresholds": thresholds,
            "decision_session": decision_session,
            "decision_hour_et": decision_hour,
        }

    def _evaluate_family_coarse_eligibility(
        self,
        *,
        family_id: str,
        family_key: Dict[str, Any],
        family_inventory_row: Optional[Dict[str, Any]],
        signature: Dict[str, Any],
    ) -> Dict[str, Any]:
        timeframe = str((family_key or {}).get("timeframe", "") or "").strip().lower()
        session = str((family_key or {}).get("session", "") or "").strip()
        side = str((family_key or {}).get("side", "") or "").strip().lower()
        strategy_type = str((family_key or {}).get("de3_strategy_type", "") or "").strip()
        threshold = str((family_key or {}).get("threshold", "") or "").strip().upper()
        retained_runtime = bool(self._family_retained_for_runtime(family_inventory_row))
        total_inventory_members = int(
            safe_float(
                ((family_inventory_row or {}).get("family_priors") or {}).get(
                    "family_member_count",
                    (family_inventory_row or {}).get("member_count", 0),
                ),
                0,
            )
        )
        if total_inventory_members <= 0 and isinstance(family_inventory_row, dict):
            members = family_inventory_row.get("members")
            if isinstance(members, list):
                total_inventory_members = int(len(members))

        sessions = signature.get("sessions") if isinstance(signature.get("sessions"), set) else set()
        sides = signature.get("sides") if isinstance(signature.get("sides"), set) else set()
        timeframes = signature.get("timeframes") if isinstance(signature.get("timeframes"), set) else set()
        strategy_types = (
            signature.get("strategy_types")
            if isinstance(signature.get("strategy_types"), set)
            else set()
        )
        decision_session = str(signature.get("decision_session", "") or "").strip()
        decision_hour = signature.get("decision_hour_et")
        decision_hour_int = int(decision_hour) if isinstance(decision_hour, int) else None

        session_tier = self._session_compatibility_tier(
            family_session=session,
            decision_session=decision_session,
            sessions_seen=sessions,
            decision_hour_et=decision_hour_int,
        )
        timeframe_tier = self._timeframe_compatibility_tier(
            family_timeframe=timeframe,
            timeframes_seen=timeframes,
        )
        strategy_tier = self._strategy_type_compatibility_tier(
            family_strategy_type=strategy_type,
            strategy_types_seen=strategy_types,
        )
        side_tier = "exact"
        if sides and side and side not in sides:
            side_tier = "incompatible"

        eligible = True
        reason = ""
        compatibility_tier = "exact"
        if not retained_runtime:
            eligible = False
            reason = "not_retained_runtime"
        elif total_inventory_members <= 0:
            eligible = False
            reason = "no_retained_members"
        elif side_tier == "incompatible":
            eligible = False
            reason = "side_mismatch"
            compatibility_tier = "incompatible"
        elif session_tier == "incompatible":
            eligible = False
            reason = "session_mismatch"
            compatibility_tier = "incompatible"
        elif timeframe_tier == "incompatible":
            eligible = False
            reason = "timeframe_mismatch"
            compatibility_tier = "incompatible"
        elif strategy_tier == "incompatible":
            eligible = False
            reason = "strategy_type_mismatch"
            compatibility_tier = "incompatible"
        else:
            if (
                str(session_tier).lower() == "exact"
                and str(timeframe_tier).lower() == "exact"
                and str(strategy_tier).lower() == "exact"
            ):
                compatibility_tier = "exact"
            else:
                compatibility_tier = "compatible"

        if (
            eligible
            and self.include_exact_and_compatible_only
            and compatibility_tier not in {"exact", "compatible"}
        ):
            eligible = False
            reason = "incompatible"
            compatibility_tier = "incompatible"

        compatibility_component = self._compatibility_component_from_tiers(
            session_tier=session_tier,
            timeframe_tier=timeframe_tier,
            strategy_type_tier=strategy_tier,
        )
        exact_match_eligible = bool(eligible and compatibility_tier == "exact")
        compatible_band_eligible = bool(eligible and compatibility_tier == "compatible")
        incompatible_excluded = bool((not eligible) and compatibility_tier == "incompatible")

        return {
            "family_id": str(family_id),
            "eligible": bool(eligible),
            "failure_reason": str(reason),
            "compatibility_tier": str(compatibility_tier),
            "session_compatibility_tier": str(session_tier),
            "timeframe_compatibility_tier": str(timeframe_tier),
            "strategy_type_compatibility_tier": str(strategy_tier),
            "side_compatibility_tier": str(side_tier),
            "exact_match_eligible": bool(exact_match_eligible),
            "compatible_band_eligible": bool(compatible_band_eligible),
            "incompatible_excluded": bool(incompatible_excluded),
            "entered_via_compatible_band": bool(compatible_band_eligible),
            "family_compatibility_component": float(compatibility_component),
            "coarse_fields": {
                "timeframe": str(timeframe),
                "session": str(session),
                "side": str(side),
                "de3_strategy_type": str(strategy_type),
                "threshold": str(threshold),
            },
            "decision_signature": {
                "decision_session": str(decision_session),
                "decision_hour_et": decision_hour_int,
                "sessions_seen": sorted(list(sessions)),
                "sides_seen": sorted(list(sides)),
                "timeframes_seen": sorted(list(timeframes)),
                "strategy_types_seen": sorted(list(strategy_types)),
            },
            "total_inventory_members": int(total_inventory_members),
        }

    def _temporary_family_block_reason(
        self, *, family_id: str, family_key: Optional[Dict[str, Any]]
    ) -> str:
        fid = str(family_id or "").strip()
        if fid and fid in self.temp_excluded_family_ids:
            return "temporary_exclusion:family_id"
        threshold = ""
        if isinstance(family_key, dict):
            threshold = str(family_key.get("threshold", "") or "").strip().upper()
        if threshold and threshold in self.temp_excluded_thresholds:
            return f"temporary_exclusion:threshold_{threshold}"
        return ""

    def _extract_runtime_state(self, family_inventory_row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not isinstance(family_inventory_row, dict):
            return {
                "evidence_support_tier": "none",
                "competition_status": "competitive",
                "usability_adjustment": 0.0,
                "suppression_reason": "",
                "metrics": {},
                "source": "",
            }
        cached = family_inventory_row.get("_cached_runtime_state_eval")
        if isinstance(cached, dict):
            return {
                "evidence_support_tier": str(cached.get("evidence_support_tier", "none") or "none"),
                "competition_status": str(cached.get("competition_status", "competitive") or "competitive"),
                "usability_adjustment": float(safe_float(cached.get("usability_adjustment", 0.0), 0.0)),
                "suppression_reason": str(cached.get("suppression_reason", "") or ""),
                "metrics": dict(cached.get("metrics", {}) if isinstance(cached.get("metrics"), dict) else {}),
                "source": str(cached.get("source", "") or ""),
            }
        state = family_inventory_row.get("family_runtime_state")
        if not isinstance(state, dict):
            return {
                "evidence_support_tier": "none",
                "competition_status": "competitive",
                "usability_adjustment": 0.0,
                "suppression_reason": "",
                "metrics": {},
                "source": "",
            }
        legacy_state = str(state.get("usability_state", "") or "")
        evidence_support_tier = canonical_evidence_support_tier(
            state.get("evidence_support_tier", self._legacy_state_to_evidence_tier(legacy_state)),
            sample_count=(state.get("metrics", {}) or {}).get("executed_trade_count", 0),
            min_samples=self.evidence_mid_samples,
            strong_samples=self.evidence_strong_samples,
        )
        competition_status = canonical_competition_status(
            state.get("competition_status", self._legacy_state_to_competition_status(legacy_state)),
            default="competitive",
        )
        if (not competition_status_is_eligible(competition_status)) and competition_status != "suppressed":
            competition_status = "suppressed"
        out = {
            "evidence_support_tier": str(evidence_support_tier),
            "competition_status": str(competition_status),
            "usability_adjustment": safe_float(
                state.get("usability_adjustment", state.get("usability_component_hint", 0.0)),
                0.0,
            ),
            "suppression_reason": str(state.get("suppression_reason", "") or ""),
            "metrics": state.get("metrics", {}) if isinstance(state.get("metrics"), dict) else {},
            "source": str(state.get("source", "") or ""),
        }
        family_inventory_row["_cached_runtime_state_eval"] = dict(out)
        return out

    @staticmethod
    def _competition_cluster_key(row: Dict[str, Any]) -> str:
        family_key = row.get("family_key") if isinstance(row.get("family_key"), dict) else {}
        timeframe = str(family_key.get("timeframe", "") or "")
        session = str(family_key.get("session", "") or "")
        side = str(family_key.get("side", "") or "")
        strategy_type = str(family_key.get("de3_strategy_type", "") or "")
        return "|".join([timeframe, session, side, strategy_type])

    def _recent_history_window_max(self) -> int:
        return int(max(10, self.dominance_window_size, self.monopoly_lookback_window))

    def _recent_share_from_history(self, family_ids: List[str], *, window: int) -> Dict[str, float]:
        out = {fid: 0.0 for fid in family_ids}
        if not family_ids:
            return out
        if not self._recent_chosen_family_ids:
            return out
        hist = self._recent_chosen_family_ids[-int(max(1, window)) :]
        total = len(hist)
        if total <= 0:
            return out
        counts: Dict[str, int] = defaultdict(int)
        for fid in hist:
            key = str(fid or "").strip()
            if key:
                counts[key] = int(counts.get(key, 0) + 1)
        for fid in family_ids:
            out[fid] = float(counts.get(fid, 0) / float(total))
        return out

    @staticmethod
    def _recent_share_from_runtime_metrics(rows: List[Dict[str, Any]]) -> Dict[str, float]:
        totals = 0.0
        counts: Dict[str, float] = {}
        for row in rows:
            fid = str(row.get("family_id", "") or "").strip()
            metrics = row.get("family_usability_metrics") if isinstance(row.get("family_usability_metrics"), dict) else {}
            chosen_n = max(0.0, safe_float(metrics.get("chosen_count", 0.0), 0.0))
            counts[fid] = float(chosen_n)
            totals += float(chosen_n)
        out = {str(row.get("family_id", "") or "").strip(): 0.0 for row in rows}
        if totals <= 1e-9:
            return out
        for fid, val in counts.items():
            out[fid] = float(val / totals)
        return out

    def _recent_share_map(self, rows: List[Dict[str, Any]]) -> Dict[str, float]:
        family_ids = [str(row.get("family_id", "") or "").strip() for row in rows if str(row.get("family_id", "") or "").strip()]
        if not family_ids:
            return {}
        history_window = int(max(1, self.dominance_window_size))
        min_history = int(max(8, min(history_window, self.bootstrap_min_competing_families * 6)))
        if len(self._recent_chosen_family_ids) >= min_history:
            return self._recent_share_from_history(family_ids, window=history_window)
        return self._recent_share_from_runtime_metrics(rows)

    def _compute_monopoly_state(self) -> Dict[str, Any]:
        window = int(max(5, self.monopoly_lookback_window))
        hist = self._recent_chosen_family_ids[-window:]
        total = len(hist)
        if total <= 0:
            return {
                "active": False,
                "top_family_id": "",
                "top_share": 0.0,
                "unique_count": 0,
                "window_size": int(window),
            }
        counts: Dict[str, int] = defaultdict(int)
        for family_id in hist:
            key = str(family_id or "").strip()
            if key:
                counts[key] = int(counts.get(key, 0) + 1)
        if not counts:
            return {
                "active": False,
                "top_family_id": "",
                "top_share": 0.0,
                "unique_count": 0,
                "window_size": int(window),
            }
        top_family_id, top_count = max(counts.items(), key=lambda item: item[1])
        top_share = float(top_count / float(total))
        return {
            "active": bool(top_share >= self.monopoly_share_threshold and total >= max(10, window // 4)),
            "top_family_id": str(top_family_id),
            "top_share": float(top_share),
            "unique_count": int(len(counts)),
            "window_size": int(window),
        }

    def _record_chosen_family(self, family_id: str) -> None:
        key = str(family_id or "").strip()
        if not key:
            return
        self._recent_chosen_family_ids.append(key)
        max_window = self._recent_history_window_max()
        if len(self._recent_chosen_family_ids) > max_window:
            self._recent_chosen_family_ids = self._recent_chosen_family_ids[-max_window:]

    @staticmethod
    def _eligibility_tier_from_row(row: Optional[Dict[str, Any]]) -> str:
        if not isinstance(row, dict):
            return "incompatible"
        tier = str(
            row.get(
                "eligibility_tier",
                row.get("compatibility_tier", "incompatible"),
            )
            or "incompatible"
        ).strip().lower()
        if tier not in {"exact", "compatible"}:
            return "incompatible"
        return tier

    def _preliminary_compatibility_penalty(self, tier: str) -> float:
        t = str(tier or "").strip().lower()
        if t == "exact":
            return float(self.cap_preliminary_compatibility_penalty_exact)
        if t == "compatible":
            return float(self.cap_preliminary_compatibility_penalty_compatible)
        return float(-0.50)

    def _preliminary_cap_score(self, row: Optional[Dict[str, Any]]) -> float:
        if not isinstance(row, dict):
            return float("-inf")
        tier = self._eligibility_tier_from_row(row)
        base_score = safe_float(
            row.get(
                "base_family_score",
                row.get("family_score", float("-inf")),
            ),
            float("-inf"),
        )
        penalty = self._preliminary_compatibility_penalty(tier)
        row["eligibility_tier"] = str(tier)
        row["preliminary_compatibility_penalty_component"] = float(penalty)
        if self.cap_use_preliminary_score_for_cap:
            prelim = float(base_score + penalty)
        else:
            prelim = float(base_score)
        row["preliminary_family_score"] = float(prelim)
        comps = row.get("family_score_components") if isinstance(row.get("family_score_components"), dict) else {}
        comps["eligibility_tier"] = str(tier)
        comps["preliminary_compatibility_penalty_component"] = float(penalty)
        comps["preliminary_family_score"] = float(prelim)
        row["family_score_components"] = comps
        return float(prelim)

    def _append_pre_cap_candidate_audit_row(self, payload: Dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            return
        if (not self.observability_enabled) or (not self.observability_emit_choice_path_audit):
            return
        self._pre_cap_candidate_audit_rows.append(dict(payload))
        max_rows = int(max(1000, self.observability_family_score_trace_max_rows))
        overflow = int(len(self._pre_cap_candidate_audit_rows) - max_rows)
        if overflow > 0:
            del self._pre_cap_candidate_audit_rows[:overflow]

    def _mark_candidate_cap_exclusion(self, row: Dict[str, Any], reason: str, *, cap_drop_reason: str = "") -> None:
        if not isinstance(row, dict):
            return
        row["competition_eligible"] = False
        row["eligible_for_candidate_set"] = False
        row["excluded_by_candidate_cap"] = True
        row["survived_cap"] = False
        row["final_competition_pool_flag"] = False
        row["cap_drop_reason"] = str(cap_drop_reason or reason or "candidate_cap_excluded")
        row["eligibility_failure_reason"] = str(reason or "candidate_cap_excluded")
        row["competition_eligibility_reason"] = str(reason or "candidate_cap_excluded")
        comps = row.get("family_score_components") if isinstance(row.get("family_score_components"), dict) else {}
        comps["candidate_cap_excluded"] = True
        comps["competition_eligibility_reason"] = str(reason or "candidate_cap_excluded")
        comps["survived_cap"] = False
        comps["final_competition_pool_flag"] = False
        comps["cap_drop_reason"] = str(cap_drop_reason or reason or "candidate_cap_excluded")
        row["family_score_components"] = comps
        self._bump_counter("candidate_cap_excluded_count", 1)

    def _apply_candidate_set_bounds(self, family_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        meta = {
            "pre_cap_candidate_count": 0,
            "post_cap_candidate_count": 0,
            "exact_match_eligible_count": 0,
            "compatible_band_eligible_count": 0,
            "exact_match_survived_count": 0,
            "compatible_band_survived_count": 0,
            "compatible_band_dropped_by_cap_count": 0,
            "cap_applied": False,
        }
        if not isinstance(family_rows, list) or not family_rows:
            return meta

        for row in family_rows:
            if not isinstance(row, dict):
                continue
            row["entered_pre_cap_pool"] = False
            row["survived_cap"] = False
            row["final_competition_pool_flag"] = False
            row["cap_drop_reason"] = ""
            row["cap_tier_slot_used"] = ""
            row["excluded_by_candidate_cap"] = False
            self._preliminary_cap_score(row)

        eligible = [row for row in family_rows if bool(row.get("competition_eligible", False))]
        if self.include_exact_and_compatible_only:
            for row in list(eligible):
                tier = self._eligibility_tier_from_row(row)
                if tier not in {"exact", "compatible"}:
                    self._mark_candidate_cap_exclusion(
                        row,
                        "candidate_cap:incompatible_tier",
                        cap_drop_reason="incompatible_tier",
                    )
            eligible = [row for row in family_rows if bool(row.get("competition_eligible", False))]

        for row in eligible:
            row["entered_pre_cap_pool"] = True
            comps = row.get("family_score_components") if isinstance(row.get("family_score_components"), dict) else {}
            comps["entered_pre_cap_pool"] = True
            row["family_score_components"] = comps

        exact_rows = [row for row in eligible if self._eligibility_tier_from_row(row) == "exact"]
        compatible_rows = [row for row in eligible if self._eligibility_tier_from_row(row) == "compatible"]
        other_rows = [
            row for row in eligible if self._eligibility_tier_from_row(row) not in {"exact", "compatible"}
        ]
        meta["pre_cap_candidate_count"] = int(len(eligible))
        meta["exact_match_eligible_count"] = int(len(exact_rows))
        meta["compatible_band_eligible_count"] = int(len(compatible_rows))
        if not eligible:
            return meta

        rank_key = lambda row: (
            float(safe_float(row.get("preliminary_family_score", float("-inf")), float("-inf"))),
            str(row.get("family_id", "") or ""),
        )
        exact_rows.sort(key=rank_key, reverse=True)
        compatible_rows.sort(key=rank_key, reverse=True)
        other_rows.sort(key=rank_key, reverse=True)
        exact_rank = {id(row): idx + 1 for idx, row in enumerate(exact_rows)}
        compatible_rank = {id(row): idx + 1 for idx, row in enumerate(compatible_rows)}

        if (not self.family_candidate_cap_enabled) or self.cap_max_total_candidates <= 0:
            for row in eligible:
                row["survived_cap"] = True
                row["final_competition_pool_flag"] = True
                comps = row.get("family_score_components") if isinstance(row.get("family_score_components"), dict) else {}
                comps["survived_cap"] = True
                comps["final_competition_pool_flag"] = True
                row["family_score_components"] = comps
            meta["post_cap_candidate_count"] = int(len(eligible))
            meta["exact_match_survived_count"] = int(len(exact_rows))
            meta["compatible_band_survived_count"] = int(len(compatible_rows))
            return meta

        if not self.enable_compatibility_tier_slot_pressure:
            max_total = int(max(1, self.cap_max_total_candidates))
            ranked_all = sorted(list(eligible), key=rank_key, reverse=True)
            selected_ids = {id(row) for row in ranked_all[:max_total]}
            meta["cap_applied"] = bool(len(eligible) > len(selected_ids))
            if meta["cap_applied"]:
                self._bump_counter("decisions_with_cap_applied_count", 1)
            for row in eligible:
                if id(row) in selected_ids:
                    row["survived_cap"] = True
                    row["final_competition_pool_flag"] = True
                    row["cap_drop_reason"] = ""
                    row["cap_tier_slot_used"] = "global_preliminary_rank"
                    row["excluded_by_candidate_cap"] = False
                    comps = row.get("family_score_components") if isinstance(row.get("family_score_components"), dict) else {}
                    comps["survived_cap"] = True
                    comps["final_competition_pool_flag"] = True
                    comps["cap_tier_slot_used"] = "global_preliminary_rank"
                    comps["cap_drop_reason"] = ""
                    row["family_score_components"] = comps
                    continue
                self._mark_candidate_cap_exclusion(
                    row,
                    "candidate_cap:global_cap_preliminary_rank",
                    cap_drop_reason="global_cap_preliminary_rank",
                )
            post_eligible = [row for row in family_rows if bool(row.get("competition_eligible", False))]
            post_exact = [row for row in post_eligible if self._eligibility_tier_from_row(row) == "exact"]
            post_compatible = [row for row in post_eligible if self._eligibility_tier_from_row(row) == "compatible"]
            meta["post_cap_candidate_count"] = int(len(post_eligible))
            meta["exact_match_survived_count"] = int(len(post_exact))
            meta["compatible_band_survived_count"] = int(len(post_compatible))
            meta["compatible_band_dropped_by_cap_count"] = int(
                max(0, meta["compatible_band_eligible_count"] - meta["compatible_band_survived_count"])
            )
            return meta

        max_total = int(max(1, self.cap_max_total_candidates))
        exact_capacity = int(len(exact_rows))
        compatible_capacity = int(len(compatible_rows))
        if self.cap_max_exact_match_candidates > 0:
            exact_capacity = int(min(exact_capacity, self.cap_max_exact_match_candidates))
        if self.cap_max_compatible_band_candidates > 0:
            compatible_capacity = int(min(compatible_capacity, self.cap_max_compatible_band_candidates))

        reserve_exact = int(min(self.cap_min_exact_match_candidates, exact_capacity))
        reserve_compatible = int(min(self.cap_min_compatible_band_candidates, compatible_capacity))
        while reserve_exact + reserve_compatible > max_total:
            if reserve_exact >= reserve_compatible and reserve_exact > 0:
                reserve_exact -= 1
            elif reserve_compatible > 0:
                reserve_compatible -= 1
            else:
                break

        selected: List[Dict[str, Any]] = []
        selected_ids: set = set()

        def _select_rows(rows: List[Dict[str, Any]], limit: int, slot: str) -> None:
            if limit <= 0:
                return
            for row in rows[:limit]:
                row_id = id(row)
                if row_id in selected_ids:
                    continue
                selected.append(row)
                selected_ids.add(row_id)
                row["cap_tier_slot_used"] = str(slot)

        _select_rows(exact_rows, reserve_exact, "reserved_exact_match")
        _select_rows(compatible_rows, reserve_compatible, "reserved_compatible_band")

        exact_remaining = [row for row in exact_rows if id(row) not in selected_ids][:exact_capacity]
        compatible_remaining = [
            row for row in compatible_rows if id(row) not in selected_ids
        ][:compatible_capacity]
        other_remaining = [row for row in other_rows if id(row) not in selected_ids]
        open_pool = sorted(exact_remaining + compatible_remaining + other_remaining, key=rank_key, reverse=True)
        open_slots = int(max(0, max_total - len(selected)))
        for row in open_pool[:open_slots]:
            row_id = id(row)
            if row_id in selected_ids:
                continue
            selected.append(row)
            selected_ids.add(row_id)
            tier = self._eligibility_tier_from_row(row)
            if tier == "exact":
                row["cap_tier_slot_used"] = "open_exact_match"
            elif tier == "compatible":
                row["cap_tier_slot_used"] = "open_compatible_band"
            else:
                row["cap_tier_slot_used"] = "open_other"

        meta["cap_applied"] = bool(len(eligible) > len(selected))
        if meta["cap_applied"]:
            self._bump_counter("decisions_with_cap_applied_count", 1)

        for row in eligible:
            if id(row) in selected_ids:
                row["survived_cap"] = True
                row["final_competition_pool_flag"] = True
                row["cap_drop_reason"] = ""
                row["excluded_by_candidate_cap"] = False
                comps = row.get("family_score_components") if isinstance(row.get("family_score_components"), dict) else {}
                comps["survived_cap"] = True
                comps["final_competition_pool_flag"] = True
                comps["cap_tier_slot_used"] = str(row.get("cap_tier_slot_used", "") or "")
                comps["cap_drop_reason"] = ""
                row["family_score_components"] = comps
                continue

            tier = self._eligibility_tier_from_row(row)
            rank_in_tier = int(
                exact_rank.get(id(row), 0) if tier == "exact" else compatible_rank.get(id(row), 0)
            )
            if tier == "exact" and self.cap_max_exact_match_candidates > 0 and rank_in_tier > self.cap_max_exact_match_candidates:
                cap_drop_reason = "over_cap_exact_match"
            elif tier == "compatible" and self.cap_max_compatible_band_candidates > 0 and rank_in_tier > self.cap_max_compatible_band_candidates:
                cap_drop_reason = "over_cap_compatible_band"
            elif tier in {"exact", "compatible"}:
                cap_drop_reason = "lower_preliminary_score_in_tier"
            else:
                cap_drop_reason = "global_cap_after_tier_merge"
            self._mark_candidate_cap_exclusion(
                row,
                f"candidate_cap:{cap_drop_reason}",
                cap_drop_reason=cap_drop_reason,
            )

        post_eligible = [row for row in family_rows if bool(row.get("competition_eligible", False))]
        post_exact = [row for row in post_eligible if self._eligibility_tier_from_row(row) == "exact"]
        post_compatible = [row for row in post_eligible if self._eligibility_tier_from_row(row) == "compatible"]
        meta["post_cap_candidate_count"] = int(len(post_eligible))
        meta["exact_match_survived_count"] = int(len(post_exact))
        meta["compatible_band_survived_count"] = int(len(post_compatible))
        meta["compatible_band_dropped_by_cap_count"] = int(
            max(0, meta["compatible_band_eligible_count"] - meta["compatible_band_survived_count"])
        )
        return meta

    def _apply_competition_balance(self, family_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        eligible_rows = [row for row in family_rows if bool(row.get("competition_eligible", False))]
        if not eligible_rows:
            return {
                "close_competition_decision": False,
                "bootstrap_competition_used": False,
            }

        close_margin = float(
            max(0.0, self.fs_close_competition_margin or self.competition_margin_points)
        )
        recent_shares = self._recent_share_map(eligible_rows)
        cluster_map: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in eligible_rows:
            row["recent_chosen_share"] = float(recent_shares.get(str(row.get("family_id", "")), 0.0))
            row["competition_margin_qualified"] = False
            row["context_advantage_capped"] = False
            row["context_advantage_cap_delta"] = 0.0
            row["diversity_adjustment"] = 0.0
            row["competition_diversity_adjustment"] = 0.0
            row["exploration_bonus"] = 0.0
            row["dominance_penalty"] = 0.0
            row["base_family_score"] = float(safe_float(row.get("family_score", 0.0), 0.0))
            cluster_key = self._competition_cluster_key(row)
            cluster_map[cluster_key].append(row)

        if not self.family_competition_balance_enabled:
            bootstrap_used = bool(
                any(bool(row.get("bootstrap_competition_included", False)) for row in eligible_rows)
            )
            for row in eligible_rows:
                base_score = float(
                    safe_float(row.get("base_family_score", row.get("family_score", 0.0)), 0.0)
                )
                row["family_score"] = float(base_score)
                row["final_family_score"] = float(base_score)
                row["close_competition_decision"] = False
                row["bootstrap_competition_used_decision"] = bool(bootstrap_used)
                comps = row.get("family_score_components") if isinstance(row.get("family_score_components"), dict) else {}
                comps["base_family_score"] = float(base_score)
                comps["final_family_score"] = float(base_score)
                comps["close_competition_decision"] = False
                comps["bootstrap_competition_used_decision"] = bool(bootstrap_used)
                comps["competition_diversity_adjustment"] = 0.0
                row["family_score_components"] = comps
            return {
                "close_competition_decision": False,
                "bootstrap_competition_used": bool(bootstrap_used),
            }

        close_competition_decision = False
        for cluster_rows in cluster_map.values():
            if not cluster_rows:
                continue
            cluster_rows.sort(
                key=lambda row: safe_float(
                    row.get("base_family_score", float("-inf")), float("-inf")
                ),
                reverse=True,
            )
            leader_score = safe_float(
                cluster_rows[0].get("base_family_score", float("-inf")), float("-inf")
            )
            if not math.isfinite(leader_score):
                continue
            second_score = safe_float(
                cluster_rows[1].get("base_family_score", float("-inf")),
                float("-inf"),
            ) if len(cluster_rows) > 1 else float("-inf")
            top_gap = (
                float(leader_score - second_score)
                if math.isfinite(second_score)
                else float("inf")
            )

            strong_pool_rows = [
                row
                for row in cluster_rows
                if str(
                    row.get(
                        "evidence_support_tier",
                        row.get("family_evidence_support_tier", "none"),
                    )
                    or "none"
                )
                .strip()
                .lower()
                == "strong"
            ]
            should_cap_single_strong = bool(
                self.fs_cap_context_advantage_when_single_strong_family
                and len(cluster_rows) >= 2
                and len(strong_pool_rows) == 1
                and (
                    (top_gap <= close_margin)
                    or (not math.isfinite(top_gap))
                    or close_margin <= 0.0
                )
            )
            if should_cap_single_strong:
                strong_row = strong_pool_rows[0]
                comps = (
                    strong_row.get("family_score_components")
                    if isinstance(strong_row.get("family_score_components"), dict)
                    else {}
                )
                context_total = safe_float(
                    comps.get(
                        "context_total_component_normalized",
                        comps.get("context_total_component", 0.0),
                    ),
                    0.0,
                )
                context_cap = float(self.fs_single_strong_family_context_cap)
                if context_total > context_cap:
                    cap_delta = float(context_cap - context_total)
                    strong_row["base_family_score"] = float(
                        safe_float(strong_row.get("base_family_score", 0.0), 0.0)
                        + cap_delta
                    )
                    strong_row["context_advantage_capped"] = True
                    strong_row["context_advantage_cap_delta"] = float(cap_delta)
                    comps["context_total_component_capped"] = float(context_cap)
                    comps["context_advantage_capped"] = True
                    comps["context_advantage_cap_delta"] = float(cap_delta)
                    comps["single_strong_context_cap_applied"] = True
                    strong_row["family_score_components"] = comps

            # Re-sort in case context cap changed ordering.
            cluster_rows.sort(
                key=lambda row: safe_float(
                    row.get("base_family_score", float("-inf")), float("-inf")
                ),
                reverse=True,
            )
            leader_score = safe_float(
                cluster_rows[0].get("base_family_score", float("-inf")), float("-inf")
            )
            if not math.isfinite(leader_score):
                continue
            margin_rows = (
                [
                    row
                    for row in cluster_rows
                    if (
                        leader_score
                        - safe_float(row.get("base_family_score", float("-inf")), float("-inf"))
                    )
                    <= close_margin
                ]
                if close_margin > 0.0
                else list(cluster_rows)
            )
            margin_ids = {
                str(row.get("family_id", "") or "").strip()
                for row in margin_rows
                if str(row.get("family_id", "") or "").strip()
            }
            if len(margin_rows) >= 2:
                close_competition_decision = True

            for row in cluster_rows:
                base_score_now = float(
                    safe_float(row.get("base_family_score", float("-inf")), float("-inf"))
                )
                gap_to_leader = (
                    float(leader_score - base_score_now)
                    if math.isfinite(base_score_now)
                    else float("inf")
                )
                is_margin = str(row.get("family_id", "") or "").strip() in margin_ids
                row["competition_margin_qualified"] = bool(is_margin)
                row["score_gap_to_cluster_leader"] = float(gap_to_leader)
                comps = (
                    row.get("family_score_components")
                    if isinstance(row.get("family_score_components"), dict)
                    else {}
                )
                comps["competition_margin_qualified"] = bool(is_margin)
                comps["score_gap_to_cluster_leader"] = float(gap_to_leader)
                comps["recent_chosen_share"] = float(row.get("recent_chosen_share", 0.0))
                row["family_score_components"] = comps

            if len(cluster_rows) < 2:
                continue
            for row in cluster_rows:
                evidence_tier = str(row.get("evidence_support_tier", row.get("family_evidence_support_tier", "none")) or "none").strip().lower()
                competition_status = str(row.get("competition_status", row.get("family_competition_status", "competitive")) or "competitive").strip().lower()
                prior_eligible = bool(row.get("prior_eligible", False))
                metrics = row.get("family_usability_metrics") if isinstance(row.get("family_usability_metrics"), dict) else {}
                executed_count = max(0.0, safe_float(metrics.get("executed_trade_count", 0.0), 0.0))
                recent_share = float(safe_float(row.get("recent_chosen_share", 0.0), 0.0))
                base_score_now = float(
                    safe_float(row.get("base_family_score", float("-inf")), float("-inf"))
                )
                gap_to_leader = (
                    float(leader_score - base_score_now)
                    if math.isfinite(base_score_now)
                    else float("inf")
                )
                is_close = bool(
                    close_margin <= 0.0
                    or (math.isfinite(gap_to_leader) and gap_to_leader <= close_margin)
                )
                if close_margin > 0.0 and math.isfinite(gap_to_leader):
                    closeness = self._clip(
                        1.0 - (gap_to_leader / float(close_margin)),
                        0.0,
                        1.0,
                    )
                else:
                    closeness = 1.0 if is_close else 0.0

                exploration_bonus = 0.0
                if (
                    self.enable_exploration_bonus
                    and prior_eligible
                    and competition_status != "suppressed"
                    and evidence_tier in {"none", "low"}
                ):
                    executed_progress = self._clip(
                        executed_count / float(max(1, self.exploration_bonus_decay_threshold)),
                        0.0,
                        1.0,
                    )
                    decay_scale = self._curve_value(
                        executed_progress,
                        self.fs_exploration_bonus_curve,
                    )
                    exploration_bonus = float(
                        self._clip(
                            self.low_support_exploration_bonus * decay_scale,
                            0.0,
                            self.max_exploration_bonus,
                        )
                    )
                    exploration_bonus *= (
                        max(0.20, closeness) if is_close else 0.25
                    )

                dominance_penalty = 0.0
                if self.enable_dominance_penalty and recent_share > self.dominance_penalty_start_share:
                    denom = max(1e-9, 1.0 - self.dominance_penalty_start_share)
                    penalty_progress = self._clip(
                        (recent_share - self.dominance_penalty_start_share) / denom,
                        0.0,
                        1.0,
                    )
                    penalty_scale = self._curve_value(
                        penalty_progress,
                        self.fs_dominance_penalty_curve,
                    )
                    dominance_penalty = float(
                        -self._clip(
                            self.dominance_penalty_max * penalty_scale,
                            0.0,
                            self.max_dominance_penalty,
                        )
                    )
                    dominance_penalty *= (
                        max(0.20, closeness) if is_close else 0.25
                    )

                diversity_adjustment_raw = float(exploration_bonus + dominance_penalty)
                max_abs_adjustment = (
                    self.fs_max_competition_adjustment_close
                    if is_close
                    else self.fs_max_competition_adjustment_far
                )
                diversity_adjustment = float(
                    self._clip(
                        diversity_adjustment_raw,
                        -max_abs_adjustment,
                        max_abs_adjustment,
                    )
                )
                row["exploration_bonus"] = float(exploration_bonus)
                row["dominance_penalty"] = float(dominance_penalty)
                row["diversity_adjustment"] = float(diversity_adjustment)
                row["competition_diversity_adjustment"] = float(diversity_adjustment)
                comps = row.get("family_score_components") if isinstance(row.get("family_score_components"), dict) else {}
                comps["exploration_bonus"] = float(exploration_bonus)
                comps["dominance_penalty"] = float(dominance_penalty)
                comps["diversity_adjustment_raw"] = float(diversity_adjustment_raw)
                comps["diversity_adjustment"] = float(diversity_adjustment)
                comps["competition_diversity_adjustment"] = float(diversity_adjustment)
                comps["exploration_bonus_applied"] = bool(exploration_bonus > 1e-9)
                comps["dominance_penalty_applied"] = bool(dominance_penalty < -1e-9)
                comps["max_competition_adjustment_applied"] = float(max_abs_adjustment)
                comps["close_competition_margin"] = float(close_margin)
                row["family_score_components"] = comps

        for row in eligible_rows:
            base_score = float(safe_float(row.get("base_family_score", row.get("family_score", 0.0)), 0.0))
            diversity_adjustment = float(safe_float(row.get("diversity_adjustment", 0.0), 0.0))
            final_score = float(base_score + diversity_adjustment)
            row["base_family_score"] = float(base_score)
            row["family_score"] = float(final_score)
            row["final_family_score"] = float(final_score)
            row["close_competition_decision"] = bool(close_competition_decision)
            row["exploration_bonus_applied"] = bool(safe_float(row.get("exploration_bonus", 0.0), 0.0) > 1e-9)
            row["dominance_penalty_applied"] = bool(safe_float(row.get("dominance_penalty", 0.0), 0.0) < -1e-9)
            comps = row.get("family_score_components") if isinstance(row.get("family_score_components"), dict) else {}
            comps["base_family_score"] = float(base_score)
            comps["diversity_adjustment"] = float(diversity_adjustment)
            comps["final_family_score"] = float(final_score)
            comps["context_advantage_capped"] = bool(row.get("context_advantage_capped", False))
            comps["context_advantage_cap_delta"] = float(safe_float(row.get("context_advantage_cap_delta", 0.0), 0.0))
            comps["close_competition_decision"] = bool(close_competition_decision)
            row["family_score_components"] = comps

        bootstrap_used = bool(any(bool(row.get("bootstrap_competition_included", False)) for row in eligible_rows))
        for row in eligible_rows:
            row["bootstrap_competition_used_decision"] = bool(bootstrap_used)
            comps = row.get("family_score_components") if isinstance(row.get("family_score_components"), dict) else {}
            comps["bootstrap_competition_used_decision"] = bool(bootstrap_used)
            row["family_score_components"] = comps
        return {
            "close_competition_decision": bool(close_competition_decision),
            "bootstrap_competition_used": bool(bootstrap_used),
        }

    def _prior_eligibility(self, family_inventory_row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not isinstance(family_inventory_row, dict):
            return {
                "prior_eligible": False,
                "prior_reason": "missing_family_inventory",
                "catastrophic_prior": True,
            }
        cached = family_inventory_row.get("_cached_prior_eval")
        if isinstance(cached, dict):
            return {
                "prior_eligible": bool(cached.get("prior_eligible", False)),
                "prior_reason": str(cached.get("prior_reason", "") or ""),
                "catastrophic_prior": bool(cached.get("catastrophic_prior", False)),
            }
        priors = family_inventory_row.get("family_priors") if isinstance(family_inventory_row.get("family_priors"), dict) else {}
        if not priors:
            out = {
                "prior_eligible": False,
                "prior_reason": "missing_family_priors",
                "catastrophic_prior": True,
            }
            family_inventory_row["_cached_prior_eval"] = dict(out)
            return out

        best_pf = safe_float(priors.get("best_member_profit_factor", 0.0), 0.0)
        best_pbr = safe_float(priors.get("best_member_profitable_block_ratio", 0.0), 0.0)
        best_worst_pf = safe_float(priors.get("best_member_worst_block_pf", 0.0), 0.0)
        best_worst_avg = safe_float(priors.get("best_member_worst_block_avg_pnl", 0.0), 0.0)
        support = safe_float(priors.get("total_support_trades", 0.0), 0.0)
        median_dd = safe_float(priors.get("median_drawdown_norm", 0.0), 0.0)
        median_loss = safe_float(priors.get("median_loss_share", 0.0), 0.0)
        median_struct = safe_float(priors.get("median_member_structural_score", 0.0), 0.0)

        catastrophic = bool(
            (best_pf < self.prior_catastrophic_min_best_pf)
            or (best_worst_pf < self.prior_catastrophic_min_best_worst_pf)
            or (median_dd > self.prior_catastrophic_max_dd)
            or (median_loss > self.prior_catastrophic_max_loss)
        )
        if catastrophic:
            out = {
                "prior_eligible": False,
                "prior_reason": "catastrophic_prior_failure",
                "catastrophic_prior": True,
            }
            family_inventory_row["_cached_prior_eval"] = dict(out)
            return out
        if not self.prior_eligibility_enabled:
            out = {
                "prior_eligible": True,
                "prior_reason": "",
                "catastrophic_prior": False,
            }
            family_inventory_row["_cached_prior_eval"] = dict(out)
            return out

        reasons: List[str] = []
        if support < self.prior_min_support:
            reasons.append("support_below_min")
        if best_pf < self.prior_min_best_pf:
            reasons.append("best_pf_below_min")
        if best_pbr < self.prior_min_best_pbr:
            reasons.append("best_pbr_below_min")
        if best_worst_pf < self.prior_min_best_worst_pf:
            reasons.append("best_worst_pf_below_min")
        if best_worst_avg < self.prior_min_best_worst_avg:
            reasons.append("worst_block_avg_too_low")
        if median_dd > self.prior_max_median_dd:
            reasons.append("median_drawdown_too_high")
        if median_loss > self.prior_max_median_loss:
            reasons.append("median_loss_share_too_high")
        if median_struct < self.prior_min_median_structural:
            reasons.append("median_structural_too_low")

        out = {
            "prior_eligible": bool(len(reasons) == 0),
            "prior_reason": ",".join(reasons),
            "catastrophic_prior": False,
        }
        family_inventory_row["_cached_prior_eval"] = dict(out)
        return out

    def _prior_component(self, family_inventory_row: Optional[Dict[str, Any]]) -> float:
        if not isinstance(family_inventory_row, dict):
            return 0.0
        cached = family_inventory_row.get("_cached_prior_component")
        if isinstance(cached, (int, float)):
            return float(cached)
        priors = family_inventory_row.get("family_priors") if isinstance(family_inventory_row.get("family_priors"), dict) else {}
        struct_term = self._clip(safe_float(priors.get("median_member_structural_score", 0.0), 0.0) / 2.0, -2.0, 2.0)
        avg_term = self._clip(safe_float(priors.get("median_member_avg_pnl", 0.0), 0.0) / 1.5, -2.0, 2.0)
        pf_term = self._clip((safe_float(priors.get("median_member_profit_factor", 0.0), 0.0) - 1.0) / 0.5, -2.0, 2.0)
        pbr_term = self._clip((safe_float(priors.get("median_member_profitable_block_ratio", 0.0), 0.0) - 0.5) / 0.25, -2.0, 2.0)
        dd_penalty = self._clip(safe_float(priors.get("median_drawdown_norm", 0.0), 0.0) / 0.8, 0.0, 2.0)
        stop_penalty = self._clip(safe_float(priors.get("median_stop_like_share", 0.0), 0.0) / 0.5, 0.0, 2.0)
        loss_penalty = self._clip(safe_float(priors.get("median_loss_share", 0.0), 0.0) / 0.6, 0.0, 2.0)
        support = max(0.0, safe_float(priors.get("total_support_trades", 0.0), 0.0))
        support_term = self._clip(math.log1p(support) / math.log1p(400.0), 0.0, 1.0)
        out = float(
            (0.38 * struct_term) + (0.20 * avg_term) + (0.15 * pf_term) + (0.12 * pbr_term) + (0.15 * support_term)
            - (0.18 * dd_penalty) - (0.07 * stop_penalty) - (0.05 * loss_penalty)
        )
        family_inventory_row["_cached_prior_component"] = float(out)
        return out

    def _usability_component(self, family_inventory_row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        state_info = self._extract_runtime_state(family_inventory_row)
        evidence_support_tier = canonical_evidence_support_tier(state_info.get("evidence_support_tier", "none"))
        competition_status = canonical_competition_status(state_info.get("competition_status", "competitive"))
        suppression_reason = str(state_info.get("suppression_reason", "") or "")
        metrics = state_info["metrics"] if isinstance(state_info["metrics"], dict) else {}
        raw_component = safe_float(state_info.get("usability_adjustment", 0.0), 0.0)
        if not self.usability_enabled:
            return {
                "component": 0.0,
                "competition_status": competition_status,
                "evidence_support_tier": evidence_support_tier,
                "suppression_reason": suppression_reason,
                "metrics": metrics,
                "raw": float(raw_component),
            }

        if not math.isfinite(raw_component):
            raw_component = 0.0
        component = self._clip(raw_component, self.usability_adj_min, self.usability_adj_max)
        if competition_status == "suppressed":
            component = min(float(component), float(self.suppressed_adjustment))
        if competition_status != "suppressed" and evidence_support_tier in {"none", "low"}:
            component = max(float(component), float(self.low_tier_min_adjustment))
        return {
            "component": float(component),
            "competition_status": str(competition_status),
            "evidence_support_tier": str(evidence_support_tier),
            "suppression_reason": str(suppression_reason),
            "metrics": metrics,
            "raw": float(raw_component),
        }

    def _context_stat_expectancy(self, stat: Dict[str, Any]) -> float:
        exp = self._clip(safe_float(stat.get("avg_pnl", 0.0), 0.0) / 1.5, -2.0, 2.0)
        exp += self._clip((safe_float(stat.get("profit_factor", 0.0), 0.0) - 1.0) / 0.5, -2.0, 2.0) * 0.8
        exp -= self._clip((safe_float(stat.get("stop_rate", 0.0), 0.0) - 0.35) / 0.35, -2.0, 2.0) * 0.6
        exp -= self._clip((safe_float(stat.get("stop_gap_rate", 0.0), 0.0) - 0.10) / 0.20, -2.0, 2.0) * 0.3
        return float(exp)

    def _context_profile_components(
        self,
        family_inventory_row: Optional[Dict[str, Any]],
        context_inputs: Dict[str, Any],
        *,
        normalized_context: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        out = {
            "expectancy": 0.0,
            "confidence": 0.0,
            "support_ratio": 0.0,
            "support_tier": "low",
            "sample_count": 0,
            "trust_weight": float(self.context_profile_weight_low_or_none),
            "used_profile_data": False,
            "trusted_profile_used": False,
            "used_fallback_priors": True,
            "active_context_buckets": {},
        }
        if not (self.use_context_profiles and self.context_profiles_enabled):
            return out
        if self.cp_require_enriched_for_runtime and bool(self.last_load_status.get("enriched_export_required", False)):
            return out
        if not isinstance(family_inventory_row, dict):
            return out

        profiles = family_inventory_row.get("family_context_profiles") if isinstance(family_inventory_row.get("family_context_profiles"), dict) else {}
        if not profiles:
            return out

        buckets = normalized_context if isinstance(normalized_context, dict) else normalize_context_buckets(
            context_inputs if isinstance(context_inputs, dict) else {}
        )
        active_buckets = {dim: str(buckets.get(dim, "") or "") for dim in self.active_context_dimensions}
        out["active_context_buckets"] = dict(active_buckets)

        components: List[Tuple[float, float]] = []
        sample_counts: List[int] = []
        for dim in self.active_context_dimensions:
            bucket = active_buckets.get(dim, "")
            dim_map = profiles.get(dim) if isinstance(profiles.get(dim), dict) else {}
            stat = dim_map.get(bucket) if bucket and isinstance(dim_map.get(bucket), dict) else None
            sample_n = int(max(0, safe_float((stat or {}).get("sample_count", 0.0), 0.0)))
            sample_counts.append(sample_n)
            if not isinstance(stat, dict) or sample_n <= 0:
                continue
            components.append((self._context_stat_expectancy(stat), max(1.0, math.log1p(float(sample_n)))))

        if self.use_joint_context_profiles:
            joint_profiles = profiles.get("joint_profiles") if isinstance(profiles.get("joint_profiles"), dict) else {}
            if joint_profiles:
                joint_key = build_active_context_joint_key(active_buckets)
                out["active_joint_key"] = joint_key
                joint_stat = joint_profiles.get(joint_key) if isinstance(joint_profiles.get(joint_key), dict) else None
                joint_n = int(max(0, safe_float((joint_stat or {}).get("sample_count", 0.0), 0.0)))
                if isinstance(joint_stat, dict) and joint_n > 0:
                    components.append((self._context_stat_expectancy(joint_stat), max(1.0, math.log1p(float(joint_n)))))
                    sample_counts.append(joint_n)

        if not sample_counts:
            return out

        sample_floor = int(min(sample_counts))
        support_tier = support_tier_from_sample_count(
            sample_floor,
            min_samples=self.min_context_bucket_samples,
            strong_samples=self.strong_context_bucket_samples,
        )
        trust_weight = support_weight_for_tier(
            support_tier,
            strong_weight=self.context_profile_weight_strong,
            mid_weight=self.context_profile_weight_mid,
            low_weight=self.context_profile_weight_low_or_none,
        )
        support_ratio = self._clip(
            float(sample_floor) / float(max(1, self.strong_context_bucket_samples)),
            0.0,
            1.0,
        )
        confidence_raw = self._clip((support_ratio - 0.5) / 0.25, -2.0, 2.0)

        if components:
            total_w = sum(max(0.0, w) for _, w in components)
            if total_w > 0.0:
                raw_expectancy = sum(v * w for v, w in components) / total_w
            else:
                raw_expectancy = 0.0
            out["expectancy"] = float(raw_expectancy * trust_weight)
            out["confidence"] = float(confidence_raw * trust_weight)
            out["used_profile_data"] = True

        out["support_ratio"] = float(support_ratio)
        out["support_tier"] = str(support_tier)
        out["sample_count"] = int(sample_floor)
        out["trust_weight"] = float(trust_weight)
        out["trusted_profile_used"] = bool(out["used_profile_data"] and support_tier in {"mid", "strong"})
        out["used_fallback_priors"] = bool(not out["trusted_profile_used"])
        return out

    def _local_member_score_with_components(
        self,
        entry: Dict[str, Any],
        context_inputs: Dict[str, Any],
        *,
        normalized_context: Optional[Dict[str, str]] = None,
        context_scale: float = 1.0,
        allow_context_adaptation: bool = True,
    ) -> Dict[str, float]:
        edge_term = self._clip(safe_float(entry.get("edge_points", 0.0), 0.0) / 0.75, -2.0, 2.0)
        struct_term = self._clip(safe_float(entry.get("structural_score", 0.0), 0.0) / 2.0, -2.0, 2.0)
        conf_term = self._clip((safe_float(entry.get("edge_confidence", 0.5), 0.5) - 0.5) / 0.25, -2.0, 2.0)
        cand = entry.get("cand") if isinstance(entry.get("cand"), dict) else {}
        tp = max(1e-9, safe_float(cand.get("tp", 0.0), 0.0))
        sl = max(1e-9, safe_float(cand.get("sl", 0.0), 0.0))
        rr = tp / sl

        buckets = normalized_context if isinstance(normalized_context, dict) else normalize_context_buckets(context_inputs)
        regime = str(buckets.get("compression_expansion_regime", "neutral")) if allow_context_adaptation else "neutral"
        rr_target = self.local_rr_target
        if regime == "expanding":
            rr_target = self.local_target_rr_expanding
        elif regime == "compressed":
            rr_target = self.local_target_rr_compressed
        payoff_term = 1.0 - self._clip(abs(rr - rr_target) / self.local_rr_tolerance, 0.0, 1.0)

        atr_now = safe_float(context_inputs.get("atr_5m", 0.0), 0.0)
        sl_atr_term = 0.5
        if atr_now > 1e-9:
            sl_atr = sl / atr_now
            if regime == "expanding":
                target_sl_atr = self.local_target_sl_atr_expanding
            elif regime == "compressed":
                target_sl_atr = self.local_target_sl_atr_compressed
            else:
                target_sl_atr = self.local_target_sl_atr_neutral
            sl_atr_term = 1.0 - self._clip(abs(sl_atr - target_sl_atr) / self.local_sl_atr_tolerance, 0.0, 1.0)
        context_bracket_term = (0.65 * payoff_term) + (0.35 * sl_atr_term)
        final_score = float(
            (self.w_local_edge * edge_term)
            + (self.w_local_struct * struct_term)
            + (self.w_local_payoff * payoff_term)
            + (self.w_local_context_bracket * context_bracket_term * float(max(0.0, context_scale)))
            + (self.w_local_conf * conf_term)
        )
        return {
            "edge_component": float(self.w_local_edge * edge_term),
            "structural_component": float(self.w_local_struct * struct_term),
            "payoff_component": float(self.w_local_payoff * payoff_term),
            "context_bracket_suitability_component": float(
                self.w_local_context_bracket
                * context_bracket_term
                * float(max(0.0, context_scale))
            ),
            "confidence_component": float(self.w_local_conf * conf_term),
            "context_bracket_term_raw": float(context_bracket_term),
            "payoff_term_raw": float(payoff_term),
            "sl_atr_term_raw": float(sl_atr_term),
            "final_member_score": float(final_score),
        }

    def _local_member_score(
        self,
        entry: Dict[str, Any],
        context_inputs: Dict[str, Any],
        *,
        normalized_context: Optional[Dict[str, str]] = None,
        context_scale: float = 1.0,
        allow_context_adaptation: bool = True,
    ) -> float:
        comps = self._local_member_score_with_components(
            entry,
            context_inputs,
            normalized_context=normalized_context,
            context_scale=context_scale,
            allow_context_adaptation=allow_context_adaptation,
        )
        return float(safe_float(comps.get("final_member_score", 0.0), 0.0))

    def _select_preview_entry(self, entries: List[Dict[str, Any]], family_inventory_row: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not entries:
            return None
        canonical_id = self._canonical_member_id(family_inventory_row)
        canonical_shape = self._canonical_member_shape_key(family_inventory_row)
        if canonical_id:
            for entry in entries:
                cand_id = str(entry.get("cand_id", "") or "").strip()
                cand = entry.get("cand") if isinstance(entry.get("cand"), dict) else {}
                strategy_id = str(cand.get("strategy_id", "") or "").strip()
                if cand_id == canonical_id or strategy_id == canonical_id:
                    return entry
        if canonical_shape:
            for entry in entries:
                if self._entry_member_shape_key(entry) == canonical_shape:
                    return entry
        return max(
            entries,
            key=lambda e: (
                safe_float(e.get("structural_score", 0.0), 0.0),
                safe_float(e.get("edge_points", 0.0), 0.0),
            ),
        )

    def _select_local_member(
        self,
        *,
        entries: List[Dict[str, Any]],
        family_inventory_row: Optional[Dict[str, Any]],
        context_inputs: Dict[str, Any],
        normalized_context: Dict[str, str],
        context_support_tier: str,
        force_mode: str = "",
    ) -> Dict[str, Any]:
        if not entries:
            return {
                "chosen_entry": None,
                "chosen_score": float("-inf"),
                "mode": "none",
                "adaptation_enabled": False,
                "override_applied": False,
                "canonical_member_id": "",
                "anchor_selected": False,
            }
        tier = str(context_support_tier or "low").strip().lower()
        tier_rank = self._support_tier_rank(tier)
        full_rank = self._support_tier_rank(self.local_full_adaptation_min_support_tier)
        conservative_rank = self._support_tier_rank(self.local_conservative_adaptation_min_support_tier)
        canonical_id = self._canonical_member_id(family_inventory_row)
        canonical_shape = self._canonical_member_shape_key(family_inventory_row)
        canonical_entry = None
        if canonical_id:
            for entry in entries:
                cand_id = str(entry.get("cand_id", "") or "").strip()
                cand = entry.get("cand") if isinstance(entry.get("cand"), dict) else {}
                strategy_id = str(cand.get("strategy_id", "") or "").strip()
                if cand_id == canonical_id or strategy_id == canonical_id:
                    canonical_entry = entry
                    break
        if canonical_entry is None and canonical_shape:
            for entry in entries:
                if self._entry_member_shape_key(entry) == canonical_shape:
                    canonical_entry = entry
                    break

        forced_mode = str(force_mode or "").strip().lower()
        if forced_mode not in {"full", "conservative", "frozen"}:
            forced_mode = ""

        if forced_mode == "full":
            mode = "full"
            adaptation_enabled = True
            context_scale = 1.0
            allow_context_adaptation = True
        elif forced_mode == "conservative":
            mode = "conservative"
            adaptation_enabled = bool(self.local_context_scale_mid > 0.0)
            context_scale = self.local_context_scale_mid
            allow_context_adaptation = bool(self.local_allow_context_mid)
        elif forced_mode == "frozen":
            mode = "frozen"
            adaptation_enabled = False
            context_scale = self.local_context_scale_low
            allow_context_adaptation = False
            if self.local_freeze_to_canonical_low and canonical_entry is not None:
                chosen_components = self._local_member_score_with_components(
                    canonical_entry,
                    context_inputs,
                    normalized_context=normalized_context,
                    context_scale=context_scale,
                    allow_context_adaptation=False,
                )
                chosen_score = float(
                    safe_float(chosen_components.get("final_member_score", float("-inf")), float("-inf"))
                )
                canonical_entry["de3_member_local_score"] = float(chosen_score)
                canonical_entry["de3_member_local_score_components"] = dict(chosen_components)
                return {
                    "chosen_entry": canonical_entry,
                    "chosen_score": float(chosen_score),
                    "chosen_score_components": dict(chosen_components),
                    "mode": mode,
                    "adaptation_enabled": adaptation_enabled,
                    "override_applied": False,
                    "canonical_member_id": canonical_id,
                    "canonical_fallback_used": True,
                    "why_anchor_forced": "forced_mode_frozen",
                    "why_non_anchor_beat_anchor": "",
                    "no_local_alternative": bool(len(entries) <= 1),
                    "anchor_selected": True,
                }
        elif tier_rank >= full_rank:
            mode = "full"
            adaptation_enabled = True
            context_scale = 1.0
            allow_context_adaptation = True
        elif tier_rank >= conservative_rank:
            mode = "conservative"
            adaptation_enabled = bool(self.local_context_scale_mid > 0.0)
            context_scale = self.local_context_scale_mid
            allow_context_adaptation = bool(self.local_allow_context_mid)
        else:
            mode = "frozen"
            adaptation_enabled = False
            context_scale = self.local_context_scale_low
            allow_context_adaptation = False
            if self.local_freeze_to_canonical_low and canonical_entry is not None:
                chosen_components = self._local_member_score_with_components(
                    canonical_entry,
                    context_inputs,
                    normalized_context=normalized_context,
                    context_scale=context_scale,
                    allow_context_adaptation=False,
                )
                chosen_score = float(
                    safe_float(chosen_components.get("final_member_score", float("-inf")), float("-inf"))
                )
                canonical_entry["de3_member_local_score"] = float(chosen_score)
                canonical_entry["de3_member_local_score_components"] = dict(chosen_components)
                return {
                    "chosen_entry": canonical_entry,
                    "chosen_score": float(chosen_score),
                    "chosen_score_components": dict(chosen_components),
                    "mode": mode,
                    "adaptation_enabled": adaptation_enabled,
                    "override_applied": False,
                    "canonical_member_id": canonical_id,
                    "canonical_fallback_used": True,
                    "why_anchor_forced": "low_support_frozen_mode",
                    "why_non_anchor_beat_anchor": "",
                    "no_local_alternative": bool(len(entries) <= 1),
                    "anchor_selected": True,
                }

        best_entry = None
        best_score = float("-inf")
        best_components: Dict[str, Any] = {}
        canonical_score = float("-inf")
        canonical_components: Dict[str, Any] = {}
        for entry in entries:
            components = self._local_member_score_with_components(
                entry,
                context_inputs,
                normalized_context=normalized_context,
                context_scale=context_scale,
                allow_context_adaptation=allow_context_adaptation,
            )
            score = float(safe_float(components.get("final_member_score", float("-inf")), float("-inf")))
            if tier == "mid" and canonical_id:
                cand_id = str(entry.get("cand_id", "") or "").strip()
                cand = entry.get("cand") if isinstance(entry.get("cand"), dict) else {}
                strategy_id = str(cand.get("strategy_id", "") or "").strip()
                is_canonical = cand_id == canonical_id or strategy_id == canonical_id
                if not is_canonical:
                    score -= abs(self.local_mid_noncanonical_penalty)
                    components = dict(components)
                    components["mid_support_noncanonical_penalty"] = float(
                        -abs(self.local_mid_noncanonical_penalty)
                    )
                    components["final_member_score"] = float(score)
            entry["de3_member_local_score"] = float(score)
            entry["de3_member_local_score_components"] = dict(components)
            if canonical_entry is not None and entry is canonical_entry:
                canonical_score = float(score)
                canonical_components = dict(components)
            if best_entry is None or score > best_score:
                best_entry = entry
                best_score = float(score)
                best_components = dict(components)

        override_applied = False
        why_non_anchor_beat_anchor = ""
        canonical_fallback_used = False
        if canonical_entry is not None and isinstance(best_entry, dict):
            chosen_id = str(best_entry.get("cand_id", "") or "").strip()
            chosen_cand = best_entry.get("cand") if isinstance(best_entry.get("cand"), dict) else {}
            chosen_strategy_id = str(chosen_cand.get("strategy_id", "") or "").strip()
            chosen_shape = self._entry_member_shape_key(best_entry)
            canonical_shape_match = bool(canonical_shape and chosen_shape == canonical_shape)
            canonical_id_match = bool(canonical_id and (chosen_id == canonical_id or chosen_strategy_id == canonical_id))
            override_applied = bool(not canonical_shape_match and not canonical_id_match)
            if override_applied:
                why_non_anchor_beat_anchor = (
                    f"non_anchor_score_better:{best_score:.4f}>{canonical_score:.4f}"
                    if canonical_score > float("-inf")
                    else "non_anchor_selected"
                )
            else:
                canonical_fallback_used = True

        anchor_selected = bool(canonical_entry is not None and not override_applied)
        return {
            "chosen_entry": best_entry,
            "chosen_score": float(best_score),
            "chosen_score_components": dict(best_components),
            "mode": mode,
            "adaptation_enabled": adaptation_enabled,
            "override_applied": bool(override_applied),
            "canonical_member_id": canonical_id,
            "canonical_fallback_used": bool(canonical_fallback_used),
            "why_anchor_forced": "",
            "why_non_anchor_beat_anchor": str(why_non_anchor_beat_anchor),
            "no_local_alternative": bool(len(entries) <= 1),
            "canonical_score": float(canonical_score) if canonical_score > float("-inf") else None,
            "canonical_score_components": dict(canonical_components),
            "anchor_selected": bool(anchor_selected),
        }

    def select_family_and_member(self, *, feasible_candidates: List[Dict[str, Any]], default_session: Optional[str], context_inputs: Dict[str, Any]) -> Dict[str, Any]:
        self._bump_counter("runtime_invocations", 1)
        if not feasible_candidates:
            self._choice_path_audit["decision_count"] = int(self._choice_path_audit.get("decision_count", 0) + 1)
            self._choice_path_audit["decisions_chosen_by_fallback_default"] = int(
                self._choice_path_audit.get("decisions_chosen_by_fallback_default", 0) + 1
            )
            self._choice_path_audit["decisions_where_family_first_candidate_set_was_empty"] = int(
                self._choice_path_audit.get("decisions_where_family_first_candidate_set_was_empty", 0) + 1
            )
            self._bump_counter("decisions_chosen_by_fallback_default_count", 1)
            self._bump_counter("family_first_candidate_set_empty_count", 1)
            return {
                "chosen_entry": None,
                "chosen_family_id": "",
                "chosen_member_local_score": None,
                "feasible_family_rows": [],
                "abstain_reason": "no_feasible_members",
                "choice_path_mode": "fallback_default",
                "score_path_consistency": {
                    "candidate_count": 0,
                    "has_multiple_candidates": False,
                    "chosen_vs_runner_up_score_delta": 0.0,
                    "all_zero_final_scores": False,
                    "all_equal_final_scores": False,
                    "warnings": [],
                },
            }
        self._bump_counter("family_first_candidate_construction_count", 1)

        normalized_context = normalize_context_buckets(
            context_inputs if isinstance(context_inputs, dict) else {}
        )
        signature = self._decision_signature(
            feasible_candidates=feasible_candidates,
            default_session=default_session,
            context_inputs=context_inputs if isinstance(context_inputs, dict) else {},
        )
        runtime_universe_items = self._runtime_universe_family_items()
        universe_family_ids = {fid for fid, _ in runtime_universe_items}
        grouped_all_entries: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        grouped_retained_entries: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        grouped_member_filtered_out_count: Dict[str, int] = defaultdict(int)
        grouped_family_keys: Dict[str, Dict[str, Any]] = {}

        for entry in feasible_candidates:
            cand = entry.get("cand") if isinstance(entry.get("cand"), dict) else {}
            family_id, family_key, inventory_row = self.annotate_candidate_family(
                cand,
                default_session=default_session,
            )
            if family_id not in universe_family_ids:
                continue
            entry["family_id"] = family_id
            entry["family_key"] = family_key
            grouped_all_entries[family_id].append(entry)
            grouped_family_keys[family_id] = dict(family_key or {})
            if self._entry_member_allowed_in_refined_universe(
                entry=entry,
                family_inventory_row=inventory_row,
            ):
                grouped_retained_entries[family_id].append(entry)
            else:
                grouped_member_filtered_out_count[family_id] = int(
                    grouped_member_filtered_out_count.get(family_id, 0) + 1
                )

        family_rows: List[Dict[str, Any]] = []
        for family_id, inventory_row in runtime_universe_items:
            family_key = (
                inventory_row.get("family_key")
                if isinstance(inventory_row.get("family_key"), dict)
                else grouped_family_keys.get(family_id, {})
            )
            family_key = dict(family_key or {})
            family_role = self._family_role(family_id)
            all_entries = list(grouped_all_entries.get(family_id, []))
            entries = list(grouped_retained_entries.get(family_id, []))
            member_filtered_out_count = int(grouped_member_filtered_out_count.get(family_id, 0))
            family_retained = bool(self._family_retained_for_runtime(inventory_row))
            coarse_eval = self._evaluate_family_coarse_eligibility(
                family_id=family_id,
                family_key=family_key,
                family_inventory_row=inventory_row,
                signature=signature,
            )
            coarse_eligible = bool(coarse_eval.get("eligible", False))
            coarse_failure_reason = str(coarse_eval.get("failure_reason", "") or "")
            compatibility_tier = str(coarse_eval.get("compatibility_tier", "incompatible") or "incompatible").strip().lower()
            session_compatibility_tier = str(
                coarse_eval.get("session_compatibility_tier", "incompatible") or "incompatible"
            ).strip().lower()
            timeframe_compatibility_tier = str(
                coarse_eval.get("timeframe_compatibility_tier", "incompatible") or "incompatible"
            ).strip().lower()
            strategy_type_compatibility_tier = str(
                coarse_eval.get("strategy_type_compatibility_tier", "incompatible") or "incompatible"
            ).strip().lower()
            family_compatibility_component = float(
                safe_float(coarse_eval.get("family_compatibility_component", 0.0), 0.0)
            )
            exact_match_eligible = bool(coarse_eval.get("exact_match_eligible", False))
            compatible_band_eligible = bool(coarse_eval.get("compatible_band_eligible", False))
            incompatible_excluded = bool(coarse_eval.get("incompatible_excluded", False))
            entered_via_compatible_band = bool(coarse_eval.get("entered_via_compatible_band", False))
            eligibility_tier = str(
                compatibility_tier if coarse_eligible else "incompatible"
            ).strip().lower()

            prior_eval = self._prior_eligibility(inventory_row)
            prior_eligible = bool(prior_eval.get("prior_eligible", False))
            prior_reason = str(prior_eval.get("prior_reason", "") or "")
            catastrophic_prior = bool(prior_eval.get("catastrophic_prior", False))
            if (not prior_eligible) and self.prior_log_rejections and prior_reason:
                logging.info("DE3 v3 prior ineligible: %s | reason=%s", family_id, prior_reason)

            edges = [safe_float(e.get("edge_points", 0.0), 0.0) for e in entries]
            confs = [
                self._clip(safe_float(e.get("edge_confidence", 0.5), 0.5), 0.0, 1.0)
                for e in entries
            ]
            if edges:
                adaptive_component = (0.8 * self._clip(max(edges) / 0.75, -2.0, 2.0)) + (
                    0.2
                    * self._clip(
                        ((sum(confs) / max(1, len(confs))) - 0.5) / 0.25,
                        -2.0,
                        2.0,
                    )
                )
            else:
                adaptive_component = 0.0
            prior_component = self._prior_component(inventory_row)
            profile = self._context_profile_components(
                inventory_row,
                context_inputs,
                normalized_context=normalized_context,
            )
            expectancy_component = safe_float(profile.get("expectancy", 0.0), 0.0)
            confidence_component = safe_float(profile.get("confidence", 0.0), 0.0)
            support_ratio = safe_float(profile.get("support_ratio", 0.0), 0.0)
            context_tier = str(profile.get("support_tier", "low") or "low")
            context_sample_count = int(safe_float(profile.get("sample_count", 0.0), 0.0))
            context_trust_weight = safe_float(
                profile.get("trust_weight", self.context_profile_weight_low_or_none),
                self.context_profile_weight_low_or_none,
            )
            if self.fallback_to_priors_when_profile_weak and context_tier == "low":
                expectancy_component *= context_trust_weight
                confidence_component *= context_trust_weight

            usability = self._usability_component(inventory_row)
            usability_component = safe_float(usability.get("component", 0.0), 0.0)
            evidence_support_tier = canonical_evidence_support_tier(
                usability.get("evidence_support_tier", "none"),
                sample_count=(usability.get("metrics", {}) or {}).get("executed_trade_count", 0),
                min_samples=self.evidence_mid_samples,
                strong_samples=self.evidence_strong_samples,
            )
            competition_status = canonical_competition_status(
                usability.get("competition_status", "competitive"),
                default="competitive",
            )
            usability_metrics = usability.get("metrics", {}) if isinstance(usability.get("metrics"), dict) else {}
            suppression_reason = str(usability.get("suppression_reason", "") or "")
            if (not prior_eligible) or catastrophic_prior:
                competition_status = "suppressed"
                suppression_reason = (
                    "catastrophic_prior_failure"
                    if catastrophic_prior
                    else (f"prior_ineligible:{prior_reason}" if prior_reason else "prior_ineligible")
                )
            context_evidence_scale = safe_float(
                self.context_scale_by_evidence_tier.get(evidence_support_tier, 0.0),
                0.0,
            )
            context_expectancy_component = float(expectancy_component * float(context_evidence_scale))
            context_confidence_component = float(confidence_component * float(context_evidence_scale))
            local_support_tier = effective_local_support_tier(
                context_support_tier=context_tier,
                evidence_support_tier=evidence_support_tier,
            )

            catastrophic_prior_flag = bool(catastrophic_prior)
            if self.usability_exclude_only_suppressed:
                competition_eligible = bool(
                    coarse_eligible
                    and prior_eligible
                    and competition_status_is_eligible(competition_status)
                    and not catastrophic_prior_flag
                )
            else:
                competition_eligible = bool(
                    coarse_eligible
                    and prior_eligible
                    and competition_status in {"competitive", "competitive_bootstrap"}
                    and not catastrophic_prior_flag
                )
            temporary_exclusion_reason = self._temporary_family_block_reason(
                family_id=family_id,
                family_key=family_key,
            )
            if temporary_exclusion_reason:
                competition_eligible = False
                self._bump_counter("temporary_exclusion_skip_count", 1)
            competition_reason = ""
            if not competition_eligible:
                if temporary_exclusion_reason:
                    competition_reason = str(temporary_exclusion_reason)
                elif not coarse_eligible:
                    competition_reason = (
                        f"coarse_ineligible:{coarse_failure_reason}"
                        if coarse_failure_reason
                        else "coarse_ineligible"
                    )
                elif not prior_eligible:
                    competition_reason = f"prior_ineligible:{prior_reason}" if prior_reason else "prior_ineligible"
                elif competition_status == "suppressed" or catastrophic_prior_flag:
                    competition_reason = suppression_reason or "suppressed"
                else:
                    competition_reason = "non_competitive_status"

            weighted_usability_component = float(self.w_usability * usability_component)
            weighted_prior_component = float(self.w_prior * prior_component)
            weighted_context_expectancy_component = float(self.w_context_expectancy * context_expectancy_component)
            weighted_context_confidence_component = float(self.w_context_confidence * context_confidence_component)
            weighted_adaptive_component = float(self.w_adaptive * adaptive_component)
            context_total_component_raw = float(
                weighted_context_expectancy_component + weighted_context_confidence_component
            )
            if self.fs_normalize_prior_component:
                normalized_prior_component = float(
                    self._squash_component(weighted_prior_component, 0.35)
                )
                normalized_context_component = float(
                    self._squash_component(context_total_component_raw, 0.35)
                )
                normalized_evidence_component = float(
                    self._squash_component(weighted_usability_component, 0.25)
                )
            else:
                normalized_prior_component = float(weighted_prior_component)
                normalized_context_component = float(context_total_component_raw)
                normalized_evidence_component = float(weighted_usability_component)
            context_total_component = float(normalized_context_component)
            base_family_score = (
                normalized_prior_component
                + normalized_context_component
                + normalized_evidence_component
                + weighted_adaptive_component
                + float(family_compatibility_component)
            )
            preliminary_compatibility_penalty_component = float(
                self._preliminary_compatibility_penalty(eligibility_tier)
            )
            preliminary_family_score = float(
                base_family_score + preliminary_compatibility_penalty_component
                if self.cap_use_preliminary_score_for_cap
                else base_family_score
            )
            preview_entry = self._select_preview_entry(entries, inventory_row)
            preview_member_id = ""
            if isinstance(preview_entry, dict):
                preview_member_id = str(preview_entry.get("cand_id", "") or "").strip()
                if not preview_member_id:
                    cand = preview_entry.get("cand") if isinstance(preview_entry.get("cand"), dict) else {}
                    preview_member_id = str(cand.get("strategy_id", "") or "").strip()

            family_rows.append(
                {
                    "family_id": family_id,
                    "family_key": dict(family_key or {}),
                    "family_runtime_role": str(family_role),
                    "is_core_family": bool(family_role == "core"),
                    "is_satellite_family": bool(family_role == "satellite"),
                    "feasible_member_count": int(len(entries)),
                    "member_candidates_seen_count": int(len(all_entries)),
                    "member_filtered_out_count": int(member_filtered_out_count),
                    "inventory_member_count": int(
                        safe_float(
                            ((inventory_row or {}).get("family_priors") or {}).get("family_member_count", len(entries)),
                            len(entries),
                        )
                    ),
                    "family_score": float(base_family_score),
                    "base_family_score": float(base_family_score),
                    "final_family_score": float(base_family_score),
                    "candidate_rank_before_adjustments": 0,
                    "family_candidate_source": str(self.last_load_status.get("loaded_universe", "raw") or "raw"),
                    "eligibility_tier": str(eligibility_tier),
                    "preliminary_family_score": float(preliminary_family_score),
                    "preliminary_compatibility_penalty_component": float(
                        preliminary_compatibility_penalty_component
                    ),
                    "entered_pre_cap_pool": False,
                    "survived_cap": False,
                    "cap_drop_reason": "",
                    "cap_tier_slot_used": "",
                    "final_competition_pool_flag": False,
                    "diversity_adjustment": 0.0,
                    "competition_diversity_adjustment": 0.0,
                    "recent_chosen_share": 0.0,
                    "exploration_bonus_applied": False,
                    "dominance_penalty_applied": False,
                    "competition_margin_qualified": False,
                    "context_advantage_capped": False,
                    "close_competition_decision": False,
                    "bootstrap_competition_used_decision": False,
                    "family_monopoly_active": False,
                    "family_monopoly_top_share": 0.0,
                    "family_monopoly_top_family_id": "",
                    "family_monopoly_unique_count": 0,
                    "family_monopoly_window_size": 0,
                    "monopoly_canonical_force_applied": False,
                    "family_score_components": {
                        "family_runtime_role": str(family_role),
                        "is_core_family": bool(family_role == "core"),
                        "is_satellite_family": bool(family_role == "satellite"),
                        "family_prior_component": float(prior_component),
                        "family_prior_component_weighted": float(weighted_prior_component),
                        "family_prior_component_normalized": float(
                            normalized_prior_component
                        ),
                        "context_profile_expectancy_component": float(context_expectancy_component),
                        "context_profile_expectancy_component_weighted": float(weighted_context_expectancy_component),
                        "context_profile_confidence_component": float(context_confidence_component),
                        "context_profile_confidence_component_weighted": float(weighted_context_confidence_component),
                        "context_total_component_raw": float(context_total_component_raw),
                        "context_total_component_normalized": float(
                            normalized_context_component
                        ),
                        "v3_realized_usability_component": float(weighted_usability_component),
                        "v3_realized_usability_component_normalized": float(
                            normalized_evidence_component
                        ),
                        "v3_realized_usability_raw_adjustment": float(usability_component),
                        "adaptive_policy_component": float(adaptive_component),
                        "adaptive_policy_component_weighted": float(weighted_adaptive_component),
                        "context_support_ratio": float(support_ratio),
                        "context_support_tier": str(context_tier),
                        "context_sample_count": int(context_sample_count),
                        "context_profile_weight": float(context_trust_weight),
                        "trusted_context_used": bool(profile.get("trusted_profile_used", False)),
                        "fallback_to_priors": bool(profile.get("used_fallback_priors", True)),
                        "prior_eligible": bool(prior_eligible),
                        "prior_eligibility_reason": str(prior_reason),
                        "catastrophic_prior": bool(catastrophic_prior_flag),
                        "evidence_support_tier": str(evidence_support_tier),
                        "competition_status": str(competition_status),
                        "family_retained": bool(family_retained),
                        "usability_adjustment": float(usability_component),
                        "suppression_reason": str(suppression_reason),
                        "context_evidence_scale": float(context_evidence_scale),
                        "context_total_component": float(context_total_component),
                        "family_compatibility_component": float(family_compatibility_component),
                        "eligibility_tier": str(eligibility_tier),
                        "preliminary_family_score": float(preliminary_family_score),
                        "preliminary_compatibility_penalty_component": float(
                            preliminary_compatibility_penalty_component
                        ),
                        "entered_pre_cap_pool": False,
                        "survived_cap": False,
                        "cap_drop_reason": "",
                        "cap_tier_slot_used": "",
                        "final_competition_pool_flag": False,
                        "compatibility_tier": str(compatibility_tier),
                        "session_compatibility_tier": str(session_compatibility_tier),
                        "timeframe_compatibility_tier": str(timeframe_compatibility_tier),
                        "strategy_type_compatibility_tier": str(strategy_type_compatibility_tier),
                        "exact_match_eligible": bool(exact_match_eligible),
                        "compatible_band_eligible": bool(compatible_band_eligible),
                        "incompatible_excluded": bool(incompatible_excluded),
                        "entered_via_compatible_band": bool(entered_via_compatible_band),
                        "base_family_score": float(base_family_score),
                        "candidate_rank_before_adjustments": 0,
                        "family_candidate_source": str(
                            self.last_load_status.get("loaded_universe", "raw") or "raw"
                        ),
                        "family_runtime_role": str(family_role),
                        "is_core_family": bool(family_role == "core"),
                        "is_satellite_family": bool(family_role == "satellite"),
                        "diversity_adjustment": 0.0,
                        "competition_diversity_adjustment": 0.0,
                        "final_family_score": float(base_family_score),
                        "recent_chosen_share": 0.0,
                        "exploration_bonus": 0.0,
                        "dominance_penalty": 0.0,
                        "exploration_bonus_applied": False,
                        "dominance_penalty_applied": False,
                        "competition_margin_qualified": False,
                        "context_advantage_capped": False,
                        "context_advantage_cap_delta": 0.0,
                        "close_competition_decision": False,
                        "bootstrap_competition_used_decision": False,
                        "family_monopoly_active": False,
                        "family_monopoly_top_share": 0.0,
                        "family_monopoly_top_family_id": "",
                        "family_monopoly_unique_count": 0,
                        "monopoly_canonical_force_applied": False,
                        "local_support_tier": str(local_support_tier),
                        "bootstrap_competition_included": False,
                        # Backward-compatible alias.
                        "usability_state": (
                            "suppressed"
                            if competition_status == "suppressed"
                            else ("fallback_only" if competition_status == "fallback_only" else ("low_support" if evidence_support_tier in {"none", "low"} else "active"))
                        ),
                        "context_ev_component": float(context_expectancy_component),
                        "confidence_component": float(context_confidence_component),
                        "context_profile_component": float(context_expectancy_component),
                    },
                    "profile_used": bool(profile.get("used_profile_data", False)),
                    "profile_trusted": bool(profile.get("trusted_profile_used", False)),
                    "profile_fallback": bool(profile.get("used_fallback_priors", True)),
                    "context_support_tier": str(context_tier),
                    "context_sample_count": int(context_sample_count),
                    "context_profile_weight": float(context_trust_weight),
                    "active_context_buckets": dict(profile.get("active_context_buckets", {}) or {}),
                    "active_context_dimensions": list(self.active_context_dimensions),
                    "retained_runtime": bool(family_retained),
                    "evaluated_for_eligibility": True,
                    "coarse_eligibility_checked": True,
                    "coarse_eligible": bool(coarse_eligible),
                    "eligibility_tier": str(eligibility_tier),
                    "compatibility_tier": str(compatibility_tier),
                    "session_compatibility_tier": str(session_compatibility_tier),
                    "timeframe_compatibility_tier": str(timeframe_compatibility_tier),
                    "strategy_type_compatibility_tier": str(strategy_type_compatibility_tier),
                    "family_compatibility_component": float(family_compatibility_component),
                    "preliminary_family_score": float(preliminary_family_score),
                    "preliminary_compatibility_penalty_component": float(
                        preliminary_compatibility_penalty_component
                    ),
                    "exact_match_eligible": bool(exact_match_eligible),
                    "compatible_band_eligible": bool(compatible_band_eligible),
                    "incompatible_excluded": bool(incompatible_excluded),
                    "entered_via_compatible_band": bool(entered_via_compatible_band),
                    "eligible_for_candidate_set": bool(competition_eligible),
                    "eligibility_failure_reason": (
                        str(competition_reason)
                        if not competition_eligible
                        else ""
                    ),
                    "coarse_eligibility_failure_reason": str(coarse_failure_reason),
                    "excluded_by_session_mismatch": bool(
                        ("session_mismatch" in str(competition_reason))
                        or str(coarse_failure_reason) == "session_mismatch"
                    ),
                    "excluded_by_side_mismatch": bool(
                        ("side_mismatch" in str(competition_reason))
                        or str(coarse_failure_reason) == "side_mismatch"
                    ),
                    "excluded_by_timeframe_mismatch": bool(
                        ("timeframe_mismatch" in str(competition_reason))
                        or str(coarse_failure_reason) == "timeframe_mismatch"
                    ),
                    "excluded_by_strategy_type_mismatch": bool(
                        ("strategy_type_mismatch" in str(competition_reason))
                        or str(coarse_failure_reason) == "strategy_type_mismatch"
                    ),
                    "excluded_by_context_gate": bool("context_gate" in str(competition_reason)),
                    "excluded_by_adaptive_policy_gate": bool(
                        ("adaptive_component" in str(competition_reason))
                        or ("adaptive_policy" in str(competition_reason))
                    ),
                    "excluded_by_no_local_member_available": bool(
                        ("no_local_member" in str(competition_reason))
                        or ("no_member" in str(competition_reason))
                    ),
                    "excluded_by_temporary_exclusion": bool(
                        "temporary_exclusion:" in str(competition_reason)
                    ),
                    "excluded_by_candidate_cap": False,
                    "coarse_compatibility_timeframe": str(
                        (coarse_eval.get("coarse_fields", {}) or {}).get("timeframe", "")
                    ),
                    "coarse_compatibility_session": str(
                        (coarse_eval.get("coarse_fields", {}) or {}).get("session", "")
                    ),
                    "coarse_compatibility_side": str(
                        (coarse_eval.get("coarse_fields", {}) or {}).get("side", "")
                    ),
                    "coarse_compatibility_strategy_type": str(
                        (coarse_eval.get("coarse_fields", {}) or {}).get(
                            "de3_strategy_type", ""
                        )
                    ),
                    "coarse_compatibility_threshold": str(
                        (coarse_eval.get("coarse_fields", {}) or {}).get("threshold", "")
                    ),
                    "coarse_signature_sessions_seen": ",".join(
                        list((coarse_eval.get("decision_signature", {}) or {}).get("sessions_seen", []))
                    ),
                    "coarse_signature_sides_seen": ",".join(
                        list((coarse_eval.get("decision_signature", {}) or {}).get("sides_seen", []))
                    ),
                    "coarse_signature_timeframes_seen": ",".join(
                        list(
                            (coarse_eval.get("decision_signature", {}) or {}).get(
                                "timeframes_seen", []
                            )
                        )
                    ),
                    "coarse_signature_strategy_types_seen": ",".join(
                        list(
                            (coarse_eval.get("decision_signature", {}) or {}).get(
                                "strategy_types_seen", []
                            )
                        )
                    ),
                    "coarse_signature_decision_session": str(
                        (coarse_eval.get("decision_signature", {}) or {}).get(
                            "decision_session", ""
                        )
                    ),
                    "coarse_signature_decision_hour_et": (
                        (coarse_eval.get("decision_signature", {}) or {}).get(
                            "decision_hour_et"
                        )
                    ),
                    "evidence_support_tier": str(evidence_support_tier),
                    "competition_status": str(competition_status),
                    "family_evidence_support_tier": str(evidence_support_tier),
                    "family_competition_status": str(competition_status),
                    "family_usability_adjustment": float(usability_component),
                    "family_retained": bool(family_retained),
                    "family_suppression_reason": str(suppression_reason),
                    "family_local_support_tier": str(local_support_tier),
                    # Backward-compatible alias.
                    "family_usability_state": (
                        "suppressed"
                        if competition_status == "suppressed"
                        else ("fallback_only" if competition_status == "fallback_only" else ("low_support" if evidence_support_tier in {"none", "low"} else "active"))
                    ),
                    "family_usability_component": float(weighted_usability_component),
                    "family_usability_raw_adjustment": float(usability_component),
                    "family_usability_metrics": dict(usability_metrics),
                    "prior_eligible": bool(prior_eligible),
                    "prior_eligibility_reason": str(prior_reason),
                    "catastrophic_prior": bool(catastrophic_prior_flag),
                    "suppressed_by_state": bool(competition_status == "suppressed"),
                    "competition_eligible": bool(competition_eligible),
                    "competition_eligibility_reason": str(competition_reason),
                    "bootstrap_included": False,
                    "bootstrap_competition_included": False,
                    "preview_entry": preview_entry,
                    "preview_member_id": preview_member_id,
                    "inventory_row": inventory_row,
                    "entries": entries,
                }
            )

        retained_total = int(len(runtime_universe_items))
        scanned_count = int(len(family_rows))
        eligible_count = int(sum(1 for row in family_rows if bool(row.get("competition_eligible", False))))
        excluded_count = int(max(0, scanned_count - eligible_count))
        unscanned_count = int(max(0, retained_total - scanned_count))
        exact_match_eligible_count = int(
            sum(
                1
                for row in family_rows
                if bool(row.get("coarse_eligible", False))
                and str(row.get("compatibility_tier", "")).strip().lower() == "exact"
            )
        )
        compatible_band_eligible_count = int(
            sum(
                1
                for row in family_rows
                if bool(row.get("coarse_eligible", False))
                and str(row.get("compatibility_tier", "")).strip().lower() == "compatible"
            )
        )
        incompatible_excluded_count = int(
            sum(
                1
                for row in family_rows
                if (not bool(row.get("coarse_eligible", False)))
                and str(row.get("compatibility_tier", "")).strip().lower() == "incompatible"
            )
        )
        self._path_counters["retained_families_total_last"] = int(retained_total)
        self._bump_counter("retained_families_scanned_total", scanned_count)
        self._bump_counter("retained_families_eligible_total", eligible_count)
        self._bump_counter("retained_families_excluded_total", excluded_count)
        self._bump_counter("retained_families_unscanned_total", unscanned_count)
        self._bump_counter("exact_match_eligible_count", exact_match_eligible_count)
        self._bump_counter("compatible_band_eligible_count", compatible_band_eligible_count)
        self._bump_counter("incompatible_excluded_count", incompatible_excluded_count)
        if unscanned_count > 0:
            logging.warning(
                "DE3 v3 retained-family scan gap: total=%s scanned=%s unscanned=%s",
                retained_total,
                scanned_count,
                unscanned_count,
            )
        for row in family_rows:
            row["retained_families_total"] = int(retained_total)
            row["retained_families_scanned"] = int(scanned_count)
            row["retained_families_eligible"] = int(eligible_count)
            row["retained_families_excluded"] = int(excluded_count)
            row["retained_families_unscanned"] = int(unscanned_count)
            row["retained_family_scan_guarantee_pass"] = bool(unscanned_count == 0)
            row["exact_match_eligible_count"] = int(exact_match_eligible_count)
            row["compatible_band_eligible_count"] = int(compatible_band_eligible_count)
            row["incompatible_excluded_count"] = int(incompatible_excluded_count)

        family_scan_count = int(len(family_rows))
        pre_rank_rows = sorted(
            family_rows,
            key=lambda row: safe_float(
                row.get(
                    "base_family_score",
                    (row.get("family_score_components", {}) or {}).get(
                        "base_family_score", float("-inf")
                    ),
                ),
                float("-inf"),
            ),
            reverse=True,
        )
        pre_rank_by_family: Dict[str, int] = {}
        for rank_idx, row in enumerate(pre_rank_rows, start=1):
            family_id = str(row.get("family_id", "") or "").strip()
            if family_id and family_id not in pre_rank_by_family:
                pre_rank_by_family[family_id] = int(rank_idx)
        for row in family_rows:
            row["family_scanned_count"] = int(family_scan_count)
            row["family_candidate_count"] = int(family_scan_count)
            row["family_candidate_set_gt_1"] = bool(family_scan_count > 1)
            row["candidate_rank_before_adjustments"] = int(
                pre_rank_by_family.get(str(row.get("family_id", "") or "").strip(), 0)
            )
            if not str(row.get("family_candidate_source", "")).strip():
                row["family_candidate_source"] = str(
                    self.last_load_status.get("loaded_universe", "raw") or "raw"
                )
            self._sync_family_score_object(row, chosen_flag=False, runner_up_flag=False)
        suppressed_skips = 0
        prior_ineligible_skips = 0
        for row in family_rows:
            if bool(row.get("prior_eligible", False)) is False:
                prior_ineligible_skips += 1
            if str(row.get("competition_status", row.get("family_competition_status", "competitive")) or "competitive").strip().lower() == "suppressed":
                suppressed_skips += 1
        if prior_ineligible_skips > 0:
            self._bump_counter("prior_ineligible_skip_count", int(prior_ineligible_skips))
        if suppressed_skips > 0:
            self._bump_counter("suppressed_family_skip_count", int(suppressed_skips))

        base_competitive = [row for row in family_rows if bool(row.get("competition_eligible", False))]
        if self.use_bootstrap_family_competition_floor and len(base_competitive) < self.bootstrap_min_competing_families:
            remaining = [
                row for row in family_rows
                if not bool(row.get("competition_eligible", False))
                and bool(row.get("prior_eligible", False))
                and not bool(row.get("catastrophic_prior", False))
                and not (self.usability_exclude_only_suppressed and bool(row.get("suppressed_by_state", False)))
            ]
            remaining.sort(
                key=lambda row: (
                    safe_float((row.get("family_score_components", {}) or {}).get("family_prior_component", float("-inf")), float("-inf")),
                    safe_float(row.get("family_score", float("-inf")), float("-inf")),
                ),
                reverse=True,
            )
            needed = max(0, self.bootstrap_min_competing_families - len(base_competitive))
            for row in remaining[:needed]:
                row["competition_eligible"] = True
                row["competition_eligibility_reason"] = "bootstrap_competition_floor"
                row["eligible_for_candidate_set"] = True
                row["eligibility_failure_reason"] = ""
                row["excluded_by_session_mismatch"] = False
                row["excluded_by_side_mismatch"] = False
                row["excluded_by_timeframe_mismatch"] = False
                row["excluded_by_strategy_type_mismatch"] = False
                row["excluded_by_context_gate"] = False
                row["excluded_by_adaptive_policy_gate"] = False
                row["excluded_by_no_local_member_available"] = False
                row["excluded_by_temporary_exclusion"] = False
                row["excluded_by_candidate_cap"] = False
                row["bootstrap_included"] = True
                row["bootstrap_competition_included"] = True
                row["competition_status"] = "competitive_bootstrap"
                row["family_competition_status"] = "competitive_bootstrap"
                components = row.get("family_score_components") if isinstance(row.get("family_score_components"), dict) else {}
                components["competition_status"] = "competitive_bootstrap"
                components["bootstrap_competition_included"] = True
                components["competition_eligibility_reason"] = "bootstrap_competition_floor"
                row["family_score_components"] = components
            if needed > 0:
                self._bump_counter("bootstrap_floor_inclusion_count", int(needed))

        cap_meta = self._apply_candidate_set_bounds(family_rows)
        pre_cap_count = int(safe_float(cap_meta.get("pre_cap_candidate_count", 0), 0))
        post_cap_count = int(safe_float(cap_meta.get("post_cap_candidate_count", 0), 0))
        pre_cap_exact = int(safe_float(cap_meta.get("exact_match_eligible_count", 0), 0))
        pre_cap_compatible = int(
            safe_float(cap_meta.get("compatible_band_eligible_count", 0), 0)
        )
        post_cap_exact = int(safe_float(cap_meta.get("exact_match_survived_count", 0), 0))
        post_cap_compatible = int(
            safe_float(cap_meta.get("compatible_band_survived_count", 0), 0)
        )
        compatible_dropped_by_cap = int(
            safe_float(cap_meta.get("compatible_band_dropped_by_cap_count", 0), 0)
        )
        self._bump_counter("pre_cap_candidate_total", pre_cap_count)
        self._bump_counter("post_cap_candidate_total", post_cap_count)
        self._bump_counter("pre_cap_exact_eligible_total", pre_cap_exact)
        self._bump_counter("pre_cap_compatible_eligible_total", pre_cap_compatible)
        self._bump_counter("post_cap_exact_survived_total", post_cap_exact)
        self._bump_counter("post_cap_compatible_survived_total", post_cap_compatible)
        self._bump_counter("compatible_dropped_by_cap_total", compatible_dropped_by_cap)
        if pre_cap_compatible > 0 and post_cap_compatible <= 0:
            self._bump_counter("decisions_with_compatible_pre_cap_all_dropped_count", 1)
        cap_decision_payload = {
            "decision_invocation": int(self._path_counters.get("runtime_invocations", 0)),
            "pre_cap_candidate_count": int(pre_cap_count),
            "post_cap_candidate_count": int(post_cap_count),
            "exact_match_eligible_count": int(pre_cap_exact),
            "compatible_band_eligible_count": int(pre_cap_compatible),
            "exact_match_survived_count": int(post_cap_exact),
            "compatible_band_survived_count": int(post_cap_compatible),
            "compatible_band_dropped_by_cap_count": int(compatible_dropped_by_cap),
            "cap_applied": bool(cap_meta.get("cap_applied", False)),
        }
        self._append_pre_cap_candidate_audit_row(cap_decision_payload)
        for row in family_rows:
            row["pre_cap_candidate_count"] = int(pre_cap_count)
            row["post_cap_candidate_count"] = int(post_cap_count)
            row["exact_match_eligible_count"] = int(pre_cap_exact)
            row["compatible_band_eligible_count"] = int(pre_cap_compatible)
            row["exact_match_survived_count"] = int(post_cap_exact)
            row["compatible_band_survived_count"] = int(post_cap_compatible)
            row["compatible_band_dropped_by_cap_count"] = int(compatible_dropped_by_cap)
            row["pre_cap_post_cap_logged"] = bool(self.cap_log_pre_cap_post_cap)
            comps = row.get("family_score_components") if isinstance(row.get("family_score_components"), dict) else {}
            comps["pre_cap_candidate_count"] = int(pre_cap_count)
            comps["post_cap_candidate_count"] = int(post_cap_count)
            comps["exact_match_eligible_count"] = int(pre_cap_exact)
            comps["compatible_band_eligible_count"] = int(pre_cap_compatible)
            comps["exact_match_survived_count"] = int(post_cap_exact)
            comps["compatible_band_survived_count"] = int(post_cap_compatible)
            comps["compatible_band_dropped_by_cap_count"] = int(compatible_dropped_by_cap)
            row["family_score_components"] = comps
        competition_balance_meta = self._apply_competition_balance(family_rows)
        if bool(competition_balance_meta.get("close_competition_decision", False)):
            self._bump_counter("close_competition_decision_count", 1)
        if any(bool(row.get("exploration_bonus_applied", False)) for row in family_rows):
            self._bump_counter("exploration_bonus_applied_count", 1)
        if any(bool(row.get("dominance_penalty_applied", False)) for row in family_rows):
            self._bump_counter("dominance_penalty_applied_count", 1)
        if any(bool(row.get("context_advantage_capped", False)) for row in family_rows):
            self._bump_counter("context_advantage_capped_count", 1)

        family_rows.sort(
            key=lambda row: (
                bool(row.get("competition_eligible", False)),
                safe_float(row.get("family_score", float("-inf")), float("-inf")),
            ),
            reverse=True,
        )
        eligible_rows = [row for row in family_rows if bool(row.get("competition_eligible", False))]
        family_candidate_count = int(len(eligible_rows))
        if family_candidate_count <= 1:
            self._bump_counter("family_candidate_set_size_eq_1_count", 1)
        else:
            self._bump_counter("family_candidate_set_size_gt_1_count", 1)
        self._record_candidate_size_histogram(family_candidate_count)
        for row in family_rows:
            row["family_candidate_count"] = int(family_candidate_count)
            row["family_candidate_set_gt_1"] = bool(family_candidate_count > 1)
        if not eligible_rows:
            trace_rows_no_choice: List[Dict[str, Any]] = []
            self._choice_path_audit["decision_count"] = int(self._choice_path_audit.get("decision_count", 0) + 1)
            self._choice_path_audit["decisions_chosen_by_fallback_default"] = int(
                self._choice_path_audit.get("decisions_chosen_by_fallback_default", 0) + 1
            )
            self._choice_path_audit["decisions_where_family_first_candidate_set_was_empty"] = int(
                self._choice_path_audit.get("decisions_where_family_first_candidate_set_was_empty", 0) + 1
            )
            self._bump_counter("decisions_chosen_by_fallback_default_count", 1)
            self._bump_counter("family_first_candidate_set_empty_count", 1)
            for row in family_rows:
                row["choice_path_mode"] = "fallback_default"
                row["score_path_inconsistency_flag"] = False
                row["family_candidate_count"] = int(family_candidate_count)
                row["family_candidate_set_gt_1"] = bool(family_candidate_count > 1)
                score_obj = (
                    row.get("family_score_object")
                    if isinstance(row.get("family_score_object"), dict)
                    else self._family_score_object_from_row(row)
                )
                trace_rows_no_choice.append(
                    {
                        "decision_invocation": int(self._path_counters.get("runtime_invocations", 0)),
                        "family_id": str(row.get("family_id", "") or ""),
                        "family_candidate_count": int(family_candidate_count),
                        "candidate_rank_before_adjustments": int(
                            safe_float(row.get("candidate_rank_before_adjustments", 0), 0)
                        ),
                        "family_candidate_source": str(
                            row.get(
                                "family_candidate_source",
                                str(self.last_load_status.get("loaded_universe", "raw") or "raw"),
                            )
                            or ""
                        ),
                        "family_runtime_role": str(score_obj.get("family_runtime_role", "satellite") or "satellite"),
                        "is_core_family": bool(score_obj.get("is_core_family", False)),
                        "is_satellite_family": bool(score_obj.get("is_satellite_family", True)),
                        "prior_component": float(score_obj.get("prior_component", 0.0) or 0.0),
                        "trusted_context_component": float(
                            score_obj.get("trusted_context_component", 0.0) or 0.0
                        ),
                        "evidence_adjustment": float(score_obj.get("evidence_adjustment", 0.0) or 0.0),
                        "adaptive_component": float(score_obj.get("adaptive_component", 0.0) or 0.0),
                        "competition_diversity_adjustment": float(
                            score_obj.get("competition_diversity_adjustment", 0.0) or 0.0
                        ),
                        "family_compatibility_component": float(
                            score_obj.get("family_compatibility_component", 0.0) or 0.0
                        ),
                        "pre_adjustment_score": float(score_obj.get("pre_adjustment_score", 0.0) or 0.0),
                        "final_family_score": float(score_obj.get("final_family_score", 0.0) or 0.0),
                        "compatibility_tier": str(score_obj.get("compatibility_tier", "incompatible") or "incompatible"),
                        "session_compatibility_tier": str(score_obj.get("session_compatibility_tier", "incompatible") or "incompatible"),
                        "timeframe_compatibility_tier": str(score_obj.get("timeframe_compatibility_tier", "incompatible") or "incompatible"),
                        "strategy_type_compatibility_tier": str(score_obj.get("strategy_type_compatibility_tier", "incompatible") or "incompatible"),
                        "entered_via_compatible_band": bool(score_obj.get("entered_via_compatible_band", False)),
                        "eligibility_tier": str(score_obj.get("eligibility_tier", "incompatible") or "incompatible"),
                        "preliminary_family_score": float(score_obj.get("preliminary_family_score", 0.0) or 0.0),
                        "preliminary_compatibility_penalty_component": float(
                            score_obj.get("preliminary_compatibility_penalty_component", 0.0) or 0.0
                        ),
                        "entered_pre_cap_pool": bool(score_obj.get("entered_pre_cap_pool", False)),
                        "survived_cap": bool(score_obj.get("survived_cap", False)),
                        "cap_drop_reason": str(score_obj.get("cap_drop_reason", "") or ""),
                        "cap_tier_slot_used": str(score_obj.get("cap_tier_slot_used", "") or ""),
                        "final_competition_pool_flag": bool(score_obj.get("final_competition_pool_flag", False)),
                        "support_tier": str(score_obj.get("support_tier", "low") or "low"),
                        "context_trusted_flag": bool(score_obj.get("context_trusted_flag", False)),
                        "exploration_bonus_applied": bool(score_obj.get("exploration_bonus_applied", False)),
                        "dominance_penalty_applied": bool(score_obj.get("dominance_penalty_applied", False)),
                        "context_advantage_capped": bool(score_obj.get("context_advantage_capped", False)),
                        "chosen_flag": False,
                        "runner_up_flag": False,
                    }
                )
            self._append_family_score_trace_rows(trace_rows_no_choice)
            self._finalize_family_reachability_observability(
                family_rows=family_rows,
                normalized_context=normalized_context,
            )
            return {
                "chosen_entry": None,
                "chosen_family_id": "",
                "chosen_member_local_score": None,
                "feasible_family_rows": family_rows,
                "abstain_reason": "no_competing_families",
                "choice_path_mode": "fallback_default",
                "score_path_consistency": {
                    "candidate_count": int(family_candidate_count),
                    "has_multiple_candidates": bool(family_candidate_count > 1),
                    "all_zero_final_scores": False,
                    "all_equal_final_scores": False,
                    "warnings": [],
                    "chosen_vs_runner_up_score_delta": 0.0,
                },
            }

        best_family = eligible_rows[0]
        runner_up_family = eligible_rows[1] if len(eligible_rows) > 1 else None
        for row in family_rows:
            fid = str(row.get("family_id", "") or "")
            chosen_flag = bool(fid == str(best_family.get("family_id", "") or ""))
            runner_up_flag = bool(
                isinstance(runner_up_family, dict)
                and fid == str(runner_up_family.get("family_id", "") or "")
            )
            self._sync_family_score_object(row, chosen_flag=chosen_flag, runner_up_flag=runner_up_flag)
        score_path_consistency = self._assess_score_path_consistency(
            invocation=int(self._path_counters.get("runtime_invocations", 0)),
            eligible_rows=eligible_rows,
            chosen_row=best_family,
            runner_up_row=runner_up_family if isinstance(runner_up_family, dict) else None,
        )
        if bool(score_path_consistency.get("warnings", [])):
            self._choice_path_audit["decisions_with_score_export_inconsistency"] = int(
                self._choice_path_audit.get("decisions_with_score_export_inconsistency", 0) + 1
            )
            self._bump_counter("decisions_with_score_export_inconsistency_count", 1)
        if bool(score_path_consistency.get("has_multiple_candidates", False)) and abs(
            safe_float(score_path_consistency.get("chosen_vs_runner_up_score_delta", 0.0), 0.0)
        ) <= 1e-12:
            self._choice_path_audit["decisions_with_multiple_candidates_but_zero_score_delta"] = int(
                self._choice_path_audit.get("decisions_with_multiple_candidates_but_zero_score_delta", 0) + 1
            )
            self._bump_counter("decisions_with_multiple_candidates_but_zero_score_delta_count", 1)
        comps = best_family.get("family_score_components") if isinstance(best_family.get("family_score_components"), dict) else {}
        if self.gates_enabled:
            if safe_float(best_family.get("family_score", 0.0), 0.0) < self.min_family_score:
                self._choice_path_audit["decision_count"] = int(self._choice_path_audit.get("decision_count", 0) + 1)
                self._choice_path_audit["decisions_chosen_by_fallback_default"] = int(
                    self._choice_path_audit.get("decisions_chosen_by_fallback_default", 0) + 1
                )
                self._bump_counter("decisions_chosen_by_fallback_default_count", 1)
                for row in family_rows:
                    row["choice_path_mode"] = "fallback_default"
                    row["score_path_inconsistency_flag"] = bool(score_path_consistency.get("warnings", []))
                self._finalize_family_reachability_observability(
                    family_rows=family_rows,
                    normalized_context=normalized_context,
                )
                return {
                    "chosen_entry": None,
                    "chosen_family_id": str(best_family.get("family_id", "")),
                    "chosen_member_local_score": None,
                    "feasible_family_rows": family_rows,
                    "abstain_reason": f"family_score {safe_float(best_family.get('family_score', 0.0), 0.0):.3f} < {self.min_family_score:.3f}",
                    "choice_path_mode": "fallback_default",
                    "score_path_consistency": dict(score_path_consistency),
                }
            if safe_float(comps.get("adaptive_policy_component", 0.0), 0.0) < self.min_adaptive_component:
                self._choice_path_audit["decision_count"] = int(self._choice_path_audit.get("decision_count", 0) + 1)
                self._choice_path_audit["decisions_chosen_by_fallback_default"] = int(
                    self._choice_path_audit.get("decisions_chosen_by_fallback_default", 0) + 1
                )
                self._bump_counter("decisions_chosen_by_fallback_default_count", 1)
                for row in family_rows:
                    row["choice_path_mode"] = "fallback_default"
                    row["score_path_inconsistency_flag"] = bool(score_path_consistency.get("warnings", []))
                self._finalize_family_reachability_observability(
                    family_rows=family_rows,
                    normalized_context=normalized_context,
                )
                return {
                    "chosen_entry": None,
                    "chosen_family_id": str(best_family.get("family_id", "")),
                    "chosen_member_local_score": None,
                    "feasible_family_rows": family_rows,
                    "abstain_reason": f"adaptive_component {safe_float(comps.get('adaptive_policy_component', 0.0), 0.0):.3f} < {self.min_adaptive_component:.3f}",
                    "choice_path_mode": "fallback_default",
                    "score_path_consistency": dict(score_path_consistency),
                }

        monopoly_state = self._compute_monopoly_state()
        local_force_mode = ""
        if self.force_canonical_when_family_monopoly and bool(monopoly_state.get("active", False)):
            local_tier = str(best_family.get("family_local_support_tier", best_family.get("context_support_tier", "low")) or "low").strip().lower()
            if self._support_tier_rank(local_tier) >= self._support_tier_rank("strong"):
                local_force_mode = "conservative"
            else:
                local_force_mode = "frozen"
        best_family["family_monopoly_active"] = bool(monopoly_state.get("active", False))
        best_family["family_monopoly_top_share"] = float(safe_float(monopoly_state.get("top_share", 0.0), 0.0))
        best_family["family_monopoly_top_family_id"] = str(monopoly_state.get("top_family_id", "") or "")
        best_family["family_monopoly_unique_count"] = int(safe_float(monopoly_state.get("unique_count", 0), 0))
        best_family["family_monopoly_window_size"] = int(safe_float(monopoly_state.get("window_size", self.monopoly_lookback_window), self.monopoly_lookback_window))
        best_family["monopoly_canonical_force_applied"] = bool(local_force_mode in {"conservative", "frozen"})

        local_pick = self._select_local_member(
            entries=list(best_family.get("entries") or []),
            family_inventory_row=best_family.get("inventory_row") if isinstance(best_family.get("inventory_row"), dict) else None,
            context_inputs=context_inputs,
            normalized_context=normalized_context,
            context_support_tier=str(best_family.get("family_local_support_tier", best_family.get("context_support_tier", "low"))),
            force_mode=local_force_mode,
        )
        chosen_entry = local_pick.get("chosen_entry") if isinstance(local_pick.get("chosen_entry"), dict) else None
        chosen_local_score = safe_float(local_pick.get("chosen_score", float("-inf")), float("-inf"))
        if chosen_entry is None:
            self._choice_path_audit["decision_count"] = int(self._choice_path_audit.get("decision_count", 0) + 1)
            self._choice_path_audit["decisions_chosen_by_fallback_default"] = int(
                self._choice_path_audit.get("decisions_chosen_by_fallback_default", 0) + 1
            )
            self._choice_path_audit["decisions_where_local_member_resolution_failed"] = int(
                self._choice_path_audit.get("decisions_where_local_member_resolution_failed", 0) + 1
            )
            self._bump_counter("decisions_chosen_by_fallback_default_count", 1)
            self._bump_counter("local_member_resolution_failed_count", 1)
            for row in family_rows:
                row["choice_path_mode"] = "fallback_default"
                row["score_path_inconsistency_flag"] = bool(score_path_consistency.get("warnings", []))
            self._finalize_family_reachability_observability(
                family_rows=family_rows,
                normalized_context=normalized_context,
            )
            return {
                "chosen_entry": None,
                "chosen_family_id": str(best_family.get("family_id", "")),
                "chosen_member_local_score": None,
                "feasible_family_rows": family_rows,
                "abstain_reason": "chosen_family_has_no_member",
                "choice_path_mode": "fallback_default",
                "score_path_consistency": dict(score_path_consistency),
            }
        if chosen_local_score < self.local_min_score:
            self._choice_path_audit["decision_count"] = int(self._choice_path_audit.get("decision_count", 0) + 1)
            self._choice_path_audit["decisions_chosen_by_fallback_default"] = int(
                self._choice_path_audit.get("decisions_chosen_by_fallback_default", 0) + 1
            )
            self._bump_counter("decisions_chosen_by_fallback_default_count", 1)
            for row in family_rows:
                row["choice_path_mode"] = "fallback_default"
                row["score_path_inconsistency_flag"] = bool(score_path_consistency.get("warnings", []))
            self._finalize_family_reachability_observability(
                family_rows=family_rows,
                normalized_context=normalized_context,
            )
            return {
                "chosen_entry": None,
                "chosen_family_id": str(best_family.get("family_id", "")),
                "chosen_member_local_score": float(chosen_local_score),
                "feasible_family_rows": family_rows,
                "abstain_reason": f"local_member_score {chosen_local_score:.3f} < {self.local_min_score:.3f}",
                "choice_path_mode": "fallback_default",
                "score_path_consistency": dict(score_path_consistency),
            }

        chosen_entry["de3_member_local_score"] = float(chosen_local_score)
        chosen_local_components = (
            local_pick.get("chosen_score_components")
            if isinstance(local_pick.get("chosen_score_components"), dict)
            else {}
        )
        if isinstance(chosen_local_components, dict):
            chosen_entry["de3_member_local_score_components"] = dict(chosen_local_components)
        chosen_member_id = str(chosen_entry.get("cand_id", "") or "").strip()
        if not chosen_member_id:
            chosen_member_id = str(
                (
                    (chosen_entry.get("cand") if isinstance(chosen_entry.get("cand"), dict) else {})
                    .get("strategy_id", "")
                )
                or ""
            ).strip()
        best_family["chosen_member_id"] = str(chosen_member_id)
        best_family["chosen_member_local_score"] = float(chosen_local_score)
        best_family["local_member_count_within_family"] = int(
            len(best_family.get("entries", []) if isinstance(best_family.get("entries"), list) else [])
        )
        best_family["local_edge_component"] = float(
            safe_float(chosen_local_components.get("edge_component", 0.0), 0.0)
        )
        best_family["local_structural_component"] = float(
            safe_float(chosen_local_components.get("structural_component", 0.0), 0.0)
        )
        best_family["local_bracket_suitability_component"] = float(
            safe_float(
                chosen_local_components.get("context_bracket_suitability_component", 0.0),
                0.0,
            )
        )
        best_family["local_confidence_component"] = float(
            safe_float(chosen_local_components.get("confidence_component", 0.0), 0.0)
        )
        best_family["local_payoff_component"] = float(
            safe_float(chosen_local_components.get("payoff_component", 0.0), 0.0)
        )
        best_family["local_final_member_score"] = float(
            safe_float(chosen_local_components.get("final_member_score", chosen_local_score), chosen_local_score)
        )
        best_family["local_bracket_adaptation_mode"] = str(local_pick.get("mode", "none") or "none")
        best_family["local_bracket_adaptation_enabled"] = bool(local_pick.get("adaptation_enabled", False))
        best_family["local_bracket_override_applied"] = bool(local_pick.get("override_applied", False))
        best_family["canonical_member_id"] = str(local_pick.get("canonical_member_id", "") or "")
        best_family["canonical_fallback_used"] = bool(local_pick.get("canonical_fallback_used", False))
        best_family["anchor_selected"] = bool(
            local_pick.get(
                "anchor_selected",
                (
                    str(best_family.get("chosen_member_id", "") or "").strip()
                    and str(best_family.get("chosen_member_id", "") or "").strip()
                    == str(best_family.get("canonical_member_id", "") or "").strip()
                ),
            )
        )
        best_family["why_non_anchor_beat_anchor"] = str(local_pick.get("why_non_anchor_beat_anchor", "") or "")
        best_family["why_anchor_forced"] = str(local_pick.get("why_anchor_forced", "") or "")
        best_family["no_local_alternative"] = bool(local_pick.get("no_local_alternative", False))
        best_family["competition_close_call_decision"] = bool(competition_balance_meta.get("close_competition_decision", False))
        best_family["bootstrap_competition_used_decision"] = bool(competition_balance_meta.get("bootstrap_competition_used", False))
        best_family["family_chosen_flag"] = True
        best_family["choice_path_mode"] = "single_candidate" if len(eligible_rows) == 1 else "score_ranking"

        chosen_components = best_family.get("family_score_components") if isinstance(best_family.get("family_score_components"), dict) else {}
        if bool(chosen_components.get("trusted_context_used", False)):
            self._bump_counter("context_profile_used_count", 1)
        if bool(chosen_components.get("fallback_to_priors", True)):
            self._bump_counter("context_profile_fallback_to_priors_count", 1)
        chosen_tier = str(
            best_family.get(
                "family_local_support_tier",
                best_family.get("context_support_tier", "low"),
            )
            or "low"
        ).strip().lower()
        if chosen_tier == "strong":
            self._bump_counter("strong_support_decision_count", 1)
        elif chosen_tier == "mid":
            self._bump_counter("mid_support_decision_count", 1)
        else:
            self._bump_counter("low_or_none_support_decision_count", 1)
        mode_key = str(local_pick.get("mode", "none") or "none").strip().lower()
        if mode_key == "full":
            self._bump_counter("local_bracket_full_mode_count", 1)
        elif mode_key == "conservative":
            self._bump_counter("local_bracket_conservative_mode_count", 1)
        elif mode_key == "frozen":
            self._bump_counter("local_bracket_frozen_mode_count", 1)
        if bool(best_family.get("monopoly_canonical_force_applied", False)):
            self._bump_counter("monopoly_canonical_force_count", 1)
            self._choice_path_audit["decisions_chosen_by_canonical_force"] = int(
                self._choice_path_audit.get("decisions_chosen_by_canonical_force", 0) + 1
            )
        chosen_member_id = str(best_family.get("chosen_member_id", "") or "").strip()
        canonical_member_id = str(best_family.get("canonical_member_id", "") or "").strip()
        anchor_selected_flag = bool(best_family.get("anchor_selected", False))
        if anchor_selected_flag:
            self._bump_counter("anchor_member_used_count", 1)
        elif chosen_member_id and canonical_member_id:
            self._bump_counter("non_anchor_member_used_count", 1)
        self._append_member_resolution_trace_row(
            {
                "decision_invocation": int(self._path_counters.get("runtime_invocations", 0)),
                "chosen_family_id": str(best_family.get("family_id", "") or ""),
                "local_member_count_within_family": int(best_family.get("local_member_count_within_family", 0) or 0),
                "anchor_member_id": str(best_family.get("canonical_member_id", "") or ""),
                "chosen_member_id": str(best_family.get("chosen_member_id", "") or ""),
                "local_member_selection_mode": str(best_family.get("local_bracket_adaptation_mode", "none") or "none"),
                "local_edge_component": float(best_family.get("local_edge_component", 0.0) or 0.0),
                "local_structural_component": float(best_family.get("local_structural_component", 0.0) or 0.0),
                "local_bracket_suitability_component": float(
                    best_family.get("local_bracket_suitability_component", 0.0) or 0.0
                ),
                "local_final_member_score": float(best_family.get("local_final_member_score", chosen_local_score) or chosen_local_score),
                "canonical_fallback_used": bool(best_family.get("canonical_fallback_used", False)),
                "why_non_anchor_beat_anchor": str(best_family.get("why_non_anchor_beat_anchor", "") or ""),
                "why_anchor_forced": str(best_family.get("why_anchor_forced", "") or ""),
                "no_local_alternative": bool(best_family.get("no_local_alternative", False)),
                "anchor_selected": bool(anchor_selected_flag),
            }
        )
        chosen_evidence_tier = str(
            best_family.get(
                "family_evidence_support_tier",
                best_family.get("evidence_support_tier", "none"),
            )
            or "none"
        ).strip().lower()
        if chosen_evidence_tier == "strong":
            self._bump_counter("support_tier_strong_count", 1)
        elif chosen_evidence_tier == "mid":
            self._bump_counter("support_tier_mid_count", 1)
        elif chosen_evidence_tier == "low":
            self._bump_counter("support_tier_low_count", 1)
        else:
            self._bump_counter("support_tier_none_count", 1)
        self._choice_path_audit["decision_count"] = int(self._choice_path_audit.get("decision_count", 0) + 1)
        if len(eligible_rows) <= 1:
            self._choice_path_audit["decisions_chosen_by_single_candidate"] = int(
                self._choice_path_audit.get("decisions_chosen_by_single_candidate", 0) + 1
            )
            self._bump_counter("decisions_chosen_by_single_candidate_count", 1)
        else:
            self._choice_path_audit["decisions_chosen_by_score_ranking"] = int(
                self._choice_path_audit.get("decisions_chosen_by_score_ranking", 0) + 1
            )
            self._bump_counter("decisions_chosen_by_score_ranking_count", 1)

        best_components = best_family.get("family_score_components") if isinstance(best_family.get("family_score_components"), dict) else {}
        best_components["family_monopoly_active"] = bool(best_family.get("family_monopoly_active", False))
        best_components["family_monopoly_top_share"] = float(safe_float(best_family.get("family_monopoly_top_share", 0.0), 0.0))
        best_components["family_monopoly_top_family_id"] = str(best_family.get("family_monopoly_top_family_id", "") or "")
        best_components["family_monopoly_unique_count"] = int(safe_float(best_family.get("family_monopoly_unique_count", 0), 0))
        best_components["monopoly_canonical_force_applied"] = bool(best_family.get("monopoly_canonical_force_applied", False))
        best_components["competition_close_call_decision"] = bool(best_family.get("competition_close_call_decision", False))
        best_components["bootstrap_competition_used_decision"] = bool(best_family.get("bootstrap_competition_used_decision", False))
        best_family["family_score_components"] = best_components

        self._record_chosen_family(str(best_family.get("family_id", "") or ""))

        trace_rows: List[Dict[str, Any]] = []
        for row in family_rows:
            row["family_chosen_flag"] = bool(str(row.get("family_id", "")) == str(best_family.get("family_id", "")))
            row["choice_path_mode"] = str(best_family.get("choice_path_mode", "score_ranking"))
            row["score_path_inconsistency_flag"] = bool(score_path_consistency.get("warnings", []))
            row["chosen_vs_runner_up_score_delta"] = float(
                safe_float(score_path_consistency.get("chosen_vs_runner_up_score_delta", 0.0), 0.0)
            )
            row["family_candidate_count"] = int(score_path_consistency.get("candidate_count", family_candidate_count))
            row["family_candidate_set_gt_1"] = bool(score_path_consistency.get("has_multiple_candidates", family_candidate_count > 1))
            self._sync_family_score_object(
                row,
                chosen_flag=bool(row.get("family_chosen_flag", False)),
                runner_up_flag=bool(row.get("family_runner_up_flag", False)),
            )
            score_obj = (
                row.get("family_score_object")
                if isinstance(row.get("family_score_object"), dict)
                else self._family_score_object_from_row(row)
            )
            trace_rows.append(
                {
                    "decision_invocation": int(self._path_counters.get("runtime_invocations", 0)),
                    "family_id": str(row.get("family_id", "") or ""),
                    "family_candidate_count": int(row.get("family_candidate_count", family_candidate_count) or family_candidate_count),
                    "candidate_rank_before_adjustments": int(
                        safe_float(
                            score_obj.get(
                                "candidate_rank_before_adjustments",
                                row.get("candidate_rank_before_adjustments", 0),
                            ),
                            0,
                        )
                    ),
                    "family_candidate_source": str(
                        score_obj.get(
                            "family_candidate_source",
                            row.get(
                                "family_candidate_source",
                                str(self.last_load_status.get("loaded_universe", "raw") or "raw"),
                            ),
                        )
                        or ""
                    ),
                    "family_runtime_role": str(score_obj.get("family_runtime_role", "satellite") or "satellite"),
                    "is_core_family": bool(score_obj.get("is_core_family", False)),
                    "is_satellite_family": bool(score_obj.get("is_satellite_family", True)),
                    "prior_component": float(score_obj.get("prior_component", 0.0) or 0.0),
                    "trusted_context_component": float(
                        score_obj.get("trusted_context_component", 0.0) or 0.0
                    ),
                    "evidence_adjustment": float(score_obj.get("evidence_adjustment", 0.0) or 0.0),
                    "adaptive_component": float(score_obj.get("adaptive_component", 0.0) or 0.0),
                    "competition_diversity_adjustment": float(
                        score_obj.get("competition_diversity_adjustment", 0.0) or 0.0
                    ),
                    "family_compatibility_component": float(
                        score_obj.get("family_compatibility_component", 0.0) or 0.0
                    ),
                    "pre_adjustment_score": float(score_obj.get("pre_adjustment_score", 0.0) or 0.0),
                    "final_family_score": float(score_obj.get("final_family_score", 0.0) or 0.0),
                    "compatibility_tier": str(score_obj.get("compatibility_tier", "incompatible") or "incompatible"),
                    "session_compatibility_tier": str(score_obj.get("session_compatibility_tier", "incompatible") or "incompatible"),
                    "timeframe_compatibility_tier": str(score_obj.get("timeframe_compatibility_tier", "incompatible") or "incompatible"),
                    "strategy_type_compatibility_tier": str(score_obj.get("strategy_type_compatibility_tier", "incompatible") or "incompatible"),
                    "entered_via_compatible_band": bool(score_obj.get("entered_via_compatible_band", False)),
                    "eligibility_tier": str(score_obj.get("eligibility_tier", "incompatible") or "incompatible"),
                    "preliminary_family_score": float(score_obj.get("preliminary_family_score", 0.0) or 0.0),
                    "preliminary_compatibility_penalty_component": float(
                        score_obj.get("preliminary_compatibility_penalty_component", 0.0) or 0.0
                    ),
                    "entered_pre_cap_pool": bool(score_obj.get("entered_pre_cap_pool", False)),
                    "survived_cap": bool(score_obj.get("survived_cap", False)),
                    "cap_drop_reason": str(score_obj.get("cap_drop_reason", "") or ""),
                    "cap_tier_slot_used": str(score_obj.get("cap_tier_slot_used", "") or ""),
                    "final_competition_pool_flag": bool(score_obj.get("final_competition_pool_flag", False)),
                    "support_tier": str(score_obj.get("support_tier", "low") or "low"),
                    "context_trusted_flag": bool(score_obj.get("context_trusted_flag", False)),
                    "exploration_bonus_applied": bool(score_obj.get("exploration_bonus_applied", False)),
                    "dominance_penalty_applied": bool(score_obj.get("dominance_penalty_applied", False)),
                    "context_advantage_capped": bool(score_obj.get("context_advantage_capped", False)),
                    "chosen_flag": bool(score_obj.get("chosen_flag", False)),
                    "runner_up_flag": bool(score_obj.get("runner_up_flag", False)),
                }
            )
        self._append_family_score_trace_rows(trace_rows)
        self._finalize_family_reachability_observability(
            family_rows=family_rows,
            normalized_context=normalized_context,
        )

        return {
            "chosen_entry": chosen_entry,
            "chosen_family_id": str(best_family.get("family_id", "")),
            "chosen_member_local_score": float(chosen_local_score),
            "chosen_family_row": best_family,
            "chosen_family_context_support_tier": str(best_family.get("context_support_tier", "low")),
            "chosen_family_local_support_tier": str(best_family.get("family_local_support_tier", "low")),
            "chosen_family_evidence_support_tier": str(best_family.get("family_evidence_support_tier", "none")),
            "chosen_family_competition_status": str(best_family.get("family_competition_status", "competitive")),
            "chosen_family_usability_state": str(best_family.get("family_usability_state", "low_support")),
            "local_bracket_adaptation_mode": str(best_family.get("local_bracket_adaptation_mode", "none")),
            "local_bracket_adaptation_enabled": bool(best_family.get("local_bracket_adaptation_enabled", False)),
            "local_bracket_override_applied": bool(best_family.get("local_bracket_override_applied", False)),
            "competition_close_call_decision": bool(best_family.get("competition_close_call_decision", False)),
            "bootstrap_competition_used_decision": bool(best_family.get("bootstrap_competition_used_decision", False)),
            "monopoly_canonical_force_applied": bool(best_family.get("monopoly_canonical_force_applied", False)),
            "choice_path_mode": str(best_family.get("choice_path_mode", "score_ranking")),
            "score_path_consistency": dict(score_path_consistency),
            "feasible_family_rows": family_rows,
            "abstain_reason": "",
        }
