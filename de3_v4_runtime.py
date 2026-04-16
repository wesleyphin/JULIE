import datetime as dt
import json
import logging
import math
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from de3_v4_bracket_module import DE3V4BracketModule
from de3_v4_lane_selector import DE3V4LaneSelector
from de3_v4_router import DE3V4Router
from de3_v4_schema import (
    LANE_LONG_REV,
    LANE_NO_TRADE,
    build_family_id,
    clip,
    format_threshold,
    lane_to_side,
    safe_div,
    safe_float,
    safe_int,
    strategy_type_to_lane,
)


NY_TZ = ZoneInfo("America/New_York")


class DE3V4Runtime:
    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        self.cfg = dict(cfg) if isinstance(cfg, dict) else {}
        runtime_cfg = self.cfg.get("runtime", {}) if isinstance(self.cfg.get("runtime"), dict) else {}
        core_cfg = self.cfg.get("core", {}) if isinstance(self.cfg.get("core"), dict) else {}
        self.core_enabled = bool(core_cfg.get("enabled", True))
        self.core_anchor_family_ids = [
            str(v).strip()
            for v in (core_cfg.get("anchor_family_ids", []) if isinstance(core_cfg.get("anchor_family_ids"), (list, tuple, set)) else [])
            if str(v).strip()
        ]
        # Only auto-inject the known core anchor when core mode is enabled.
        if self.core_enabled and not self.core_anchor_family_ids:
            self.core_anchor_family_ids = ["5min|09-12|long|Long_Rev|T6"]
        runtime_mode = str(
            core_cfg.get("default_runtime_mode", "core_plus_satellites") or "core_plus_satellites"
        ).strip().lower()
        if runtime_mode not in {"core_only", "core_plus_satellites", "satellites_only"}:
            runtime_mode = "core_plus_satellites"
        # Hard safety: disabling core implies satellites-only runtime.
        if not self.core_enabled:
            runtime_mode = "satellites_only"
        self.runtime_mode = runtime_mode
        self.force_anchor_when_eligible = bool(core_cfg.get("force_anchor_when_eligible", False)) and bool(
            self.core_enabled
        )
        self._core_anchor_set = {str(v).strip() for v in self.core_anchor_family_ids if str(v).strip()}
        self._core_anchor_signatures = [
            sig for sig in (self._parse_family_id(v) for v in self._core_anchor_set) if isinstance(sig, dict)
        ]
        self._core_anchor_signature_index: Dict[Tuple[str, str, str, str], set[str]] = defaultdict(set)
        for sig in self._core_anchor_signatures:
            key = (
                str(sig.get("timeframe", "") or ""),
                str(sig.get("strategy_type", "") or ""),
                str(sig.get("side", "") or "").lower(),
                str(sig.get("threshold_text", "") or ""),
            )
            self._core_anchor_signature_index[key].add(str(sig.get("session", "") or "").strip())
        self._core_anchor_variant_tokens = {
            (
                f"{str(sig.get('timeframe', '')).lower()}_{str(sig.get('session', '')).lower()}_"
                f"{str(sig.get('strategy_type', '')).lower()}_t{str(sig.get('threshold_text', '')).lower()}"
            )
            for sig in self._core_anchor_signatures
            if isinstance(sig, dict)
        }
        runtime_excluded_families = (
            runtime_cfg.get("excluded_family_ids", [])
            if isinstance(runtime_cfg.get("excluded_family_ids", []), (list, tuple, set))
            else []
        )
        runtime_excluded_variant_patterns = (
            runtime_cfg.get("excluded_variant_patterns", [])
            if isinstance(runtime_cfg.get("excluded_variant_patterns", []), (list, tuple, set))
            else []
        )
        self.excluded_family_ids = {
            str(v).strip() for v in runtime_excluded_families if str(v).strip()
        }
        # Core-disabled safety: always exclude configured core anchors from runtime candidates.
        if not self.core_enabled:
            self.excluded_family_ids.update(self._core_anchor_set)
        self.excluded_variant_patterns = {
            str(v).strip().lower() for v in runtime_excluded_variant_patterns if str(v).strip()
        }
        if not self.core_enabled:
            for sig in self._core_anchor_signatures:
                self.excluded_variant_patterns.add(
                    f"{str(sig.get('timeframe', '')).lower()}_{str(sig.get('session', '')).lower()}_"
                    f"{str(sig.get('strategy_type', '')).lower()}_t{str(sig.get('threshold_text', '')).lower()}"
                )
        self._runtime_excluded_family_count = 0
        self._runtime_excluded_variant_pattern_count = 0
        self._candidate_variant_filter_reject_count = 0
        self._candidate_variant_filter_reject_reason_counts: Counter[str] = Counter()
        self._book_gate_decision_count = 0
        self._book_gate_selected_book_counts: Counter[str] = Counter()
        self._book_gate_scope_match_counts: Counter[str] = Counter()
        self._book_gate_default_selection_count = 0
        self._book_gate_empty_fallback_count = 0
        self._book_gate_context_key_counts: Counter[str] = Counter()
        self.bundle_path = self._resolve_bundle_path(
            self.cfg.get("bundle_path", "dynamic_engine3_v4_bundle.json")
        )
        self.bundle = self._load_bundle(self.bundle_path)
        self.bundle_loaded = bool(self.bundle)
        self.direct_decision_cfg = (
            dict(runtime_cfg.get("direct_decision_model", {}))
            if isinstance(runtime_cfg.get("direct_decision_model"), dict)
            else {}
        )
        self._bundle_decision_policy_model = (
            self.bundle.get("decision_policy_model", {})
            if isinstance(self.bundle.get("decision_policy_model", {}), dict)
            else {}
        )
        self._bundle_book_gate_model = (
            self.bundle.get("book_gate_model", {})
            if isinstance(self.bundle.get("book_gate_model", {}), dict)
            else {}
        )
        self._bundle_conflict_side_model = (
            self.bundle.get("conflict_side_model", {})
            if isinstance(self.bundle.get("conflict_side_model", {}), dict)
            else {}
        )
        self._bundle_decision_side_model = (
            self.bundle.get("decision_side_model", {})
            if isinstance(self.bundle.get("decision_side_model", {}), dict)
            else {}
        )
        self._bundle_decision_side_models = (
            [
                dict(item)
                for item in self.bundle.get("decision_side_models", [])
                if isinstance(item, dict)
            ]
            if isinstance(self.bundle.get("decision_side_models"), list)
            else []
        )
        self._direct_decision_runtime_enabled = bool(self.direct_decision_cfg.get("enabled", False))
        self._direct_decision_use_bundle_model = bool(
            self.direct_decision_cfg.get("use_bundle_model", True)
        )
        self.direct_decision_model_cfg = (
            dict(self._bundle_decision_policy_model)
            if self._direct_decision_use_bundle_model and isinstance(self._bundle_decision_policy_model, dict)
            else {}
        )
        bundle_candidate_variant_filter_cfg = (
            dict(self.direct_decision_model_cfg.get("candidate_variant_filter", {}))
            if isinstance(self.direct_decision_model_cfg.get("candidate_variant_filter"), dict)
            else {}
        )
        runtime_candidate_variant_filter_cfg = (
            dict(self.direct_decision_cfg.get("candidate_variant_filter", {}))
            if isinstance(self.direct_decision_cfg.get("candidate_variant_filter"), dict)
            else {}
        )
        self._runtime_candidate_variant_filter_cfg = dict(runtime_candidate_variant_filter_cfg)
        self.candidate_variant_filter_cfg = dict(bundle_candidate_variant_filter_cfg)
        if runtime_candidate_variant_filter_cfg:
            merged_side_overrides = (
                dict(self.candidate_variant_filter_cfg.get("side_overrides", {}))
                if isinstance(self.candidate_variant_filter_cfg.get("side_overrides"), dict)
                else {}
            )
            runtime_side_overrides = (
                dict(runtime_candidate_variant_filter_cfg.get("side_overrides", {}))
                if isinstance(runtime_candidate_variant_filter_cfg.get("side_overrides"), dict)
                else {}
            )
            self.candidate_variant_filter_cfg.update(runtime_candidate_variant_filter_cfg)
            if runtime_side_overrides:
                merged_side_overrides.update(runtime_side_overrides)
                self.candidate_variant_filter_cfg["side_overrides"] = merged_side_overrides
        self.candidate_variant_filter_enabled = bool(
            self.candidate_variant_filter_cfg.get("enabled", False)
        )
        runtime_conflict_side_cfg = (
            dict(runtime_cfg.get("conflict_side_model", {}))
            if isinstance(runtime_cfg.get("conflict_side_model"), dict)
            else {}
        )
        self.conflict_side_model_cfg = dict(self._bundle_conflict_side_model)
        if runtime_conflict_side_cfg:
            self.conflict_side_model_cfg.update(runtime_conflict_side_cfg)
        self.conflict_side_model_enabled = bool(self.conflict_side_model_cfg.get("enabled", False))
        self._conflict_side_eval_count = 0
        self._conflict_side_override_count = 0
        self._conflict_side_abstain_count = 0
        self._conflict_side_predicted_side_counts: Counter[str] = Counter()
        self._conflict_side_scope_match_counts: Counter[str] = Counter()
        runtime_decision_side_cfg = (
            dict(runtime_cfg.get("decision_side_model", {}))
            if isinstance(runtime_cfg.get("decision_side_model"), dict)
            else {}
        )
        self.decision_side_model_cfgs: List[Dict[str, Any]] = []
        if self._bundle_decision_side_models:
            self.decision_side_model_cfgs = [dict(item) for item in self._bundle_decision_side_models]
        elif self._bundle_decision_side_model:
            self.decision_side_model_cfgs = [dict(self._bundle_decision_side_model)]
        if runtime_decision_side_cfg:
            if self.decision_side_model_cfgs:
                merged_primary = dict(self.decision_side_model_cfgs[0])
                merged_primary.update(runtime_decision_side_cfg)
                self.decision_side_model_cfgs[0] = merged_primary
            else:
                self.decision_side_model_cfgs = [dict(runtime_decision_side_cfg)]
        self.decision_side_model_cfg = (
            dict(self.decision_side_model_cfgs[0])
            if self.decision_side_model_cfgs
            else {}
        )
        self.decision_side_model_enabled = any(
            bool((cfg or {}).get("enabled", False))
            for cfg in self.decision_side_model_cfgs
            if isinstance(cfg, dict)
        )
        self._decision_side_eval_count = 0
        self._decision_side_override_count = 0
        self._decision_side_abstain_count = 0
        self._decision_side_predicted_action_counts: Counter[str] = Counter()
        self._decision_side_scope_match_counts: Counter[str] = Counter()
        runtime_book_gate_cfg = (
            dict(runtime_cfg.get("book_gate_model", {}))
            if isinstance(runtime_cfg.get("book_gate_model"), dict)
            else {}
        )
        self.book_gate_model_cfg = dict(self._bundle_book_gate_model)
        if runtime_book_gate_cfg:
            merged_books = (
                dict(self.book_gate_model_cfg.get("books", {}))
                if isinstance(self.book_gate_model_cfg.get("books"), dict)
                else {}
            )
            runtime_books = (
                dict(runtime_book_gate_cfg.get("books", {}))
                if isinstance(runtime_book_gate_cfg.get("books"), dict)
                else {}
            )
            self.book_gate_model_cfg.update(runtime_book_gate_cfg)
            if runtime_books:
                merged_books.update(runtime_books)
                self.book_gate_model_cfg["books"] = merged_books
        self.book_gate_enabled = bool(self.book_gate_model_cfg.get("enabled", False))
        self.direct_decision_selection_mode = str(
            self.direct_decision_model_cfg.get(
                "selection_mode",
                self.direct_decision_cfg.get("selection_mode", "replace_router_lane"),
            )
            or "replace_router_lane"
        ).strip().lower()
        self.direct_decision_enabled = bool(
            self._direct_decision_runtime_enabled
            and isinstance(self.direct_decision_model_cfg, dict)
            and bool(self.direct_decision_model_cfg.get("enabled", False))
            and self.direct_decision_selection_mode
            in {"replace_router_lane", "hybrid_fallback_router_lane", "hybrid_compare_baseline"}
        )
        self._direct_decision_score_margin_scale = max(
            1e-6,
            safe_float(self.direct_decision_cfg.get("score_margin_scale", 0.12), 0.12),
        )
        self._direct_decision_conf_floor = clip(
            safe_float(self.direct_decision_cfg.get("confidence_floor", 0.05), 0.05),
            0.0,
            1.0,
        )
        self._direct_decision_conf_cap = clip(
            safe_float(self.direct_decision_cfg.get("confidence_cap", 0.95), 0.95),
            self._direct_decision_conf_floor,
            1.0,
        )
        self.execution_filters_cfg = (
            dict(runtime_cfg.get("execution_filters", {}))
            if isinstance(runtime_cfg.get("execution_filters"), dict)
            else {}
        )
        self.execution_policy_cfg = self._build_execution_policy_cfg(
            runtime_cfg=runtime_cfg,
            legacy_execution_filters_cfg=self.execution_filters_cfg,
            bundle_entry_policy_model=(
                self.bundle.get("entry_policy_model", {})
                if isinstance(self.bundle.get("entry_policy_model"), dict)
                else {}
            ),
        )
        self.execution_policy_enabled = bool(self.execution_policy_cfg.get("enabled", False))
        self.calibrated_entry_model_cfg = (
            dict(self.execution_policy_cfg.get("calibrated_entry_model", {}))
            if isinstance(self.execution_policy_cfg.get("calibrated_entry_model"), dict)
            else {}
        )
        self._entry_model_variant_stats = (
            self.calibrated_entry_model_cfg.get("variant_stats", {})
            if isinstance(self.calibrated_entry_model_cfg.get("variant_stats"), dict)
            else {}
        )
        self._entry_model_lane_stats = (
            self.calibrated_entry_model_cfg.get("lane_stats", {})
            if isinstance(self.calibrated_entry_model_cfg.get("lane_stats"), dict)
            else {}
        )
        self._entry_model_global_stats = (
            self.calibrated_entry_model_cfg.get("global_stats", {})
            if isinstance(self.calibrated_entry_model_cfg.get("global_stats"), dict)
            else {}
        )
        self.calibrated_entry_model_enabled = bool(
            self.calibrated_entry_model_cfg.get("enabled", False)
        )
        try:
            self.trace_max_rows = int(runtime_cfg.get("trace_max_rows", 250000) or 250000)
        except Exception:
            self.trace_max_rows = 250000
        if self.trace_max_rows < 0:
            self.trace_max_rows = 0
        trace_row_maxlen = self.trace_max_rows if self.trace_max_rows > 0 else None
        # Backward-compatible alias for legacy report fields.
        self.execution_filters_enabled = bool(self.execution_policy_enabled)
        self._execution_policy_reject_count = 0
        self._execution_policy_reject_reason_counts: Counter[str] = Counter()
        self._execution_policy_tier_counts: Counter[str] = Counter()
        self._execution_policy_soft_pass_count = 0
        self._execution_policy_soft_pass_reason_counts: Counter[str] = Counter()
        self._execution_policy_rows = deque(maxlen=trace_row_maxlen)
        self._execution_policy_rows_total = 0
        self._execution_policy_allowed_count = 0
        self._entry_model_eval_count = 0
        self._entry_model_reject_count = 0
        self._entry_model_reject_reason_counts: Counter[str] = Counter()
        self._entry_model_missing_stats_count = 0
        self._entry_model_scope_counts: Counter[str] = Counter()
        self.router = DE3V4Router(
            self.bundle.get("router_model_or_router_rules", {})
            if isinstance(self.bundle.get("router_model_or_router_rules"), dict)
            else {},
            runtime_cfg=(self.cfg.get("runtime", {}) if isinstance(self.cfg.get("runtime"), dict) else {}).get("router", {}),
        )
        self.lane_selector = DE3V4LaneSelector(
            lane_inventory=self.bundle.get("lane_inventory", {}),
            lane_variant_quality=self.bundle.get("lane_variant_quality", {}),
            lane_anchor_variants=self.bundle.get("lane_anchor_variants", {}),
            runtime_cfg=(self.cfg.get("runtime", {}) if isinstance(self.cfg.get("runtime"), dict) else {}).get("lane_selector", {}),
        )
        self.bracket_module = DE3V4BracketModule(
            bracket_defaults=self.bundle.get("bracket_defaults", {}),
            bracket_modes=self.bundle.get("bracket_modes", {}),
            family_bracket_selector=self.bundle.get("family_bracket_selector", {}),
            runtime_cfg=(self.cfg.get("runtime", {}) if isinstance(self.cfg.get("runtime"), dict) else {}).get("bracket_module", {}),
        )
        self._runtime_invocations = 0
        self._router_rows = deque(maxlen=trace_row_maxlen)
        self._lane_rows = deque(maxlen=trace_row_maxlen)
        self._bracket_rows = deque(maxlen=trace_row_maxlen)
        self._choice_rows = deque(maxlen=trace_row_maxlen)
        self._router_rows_total = 0
        self._lane_rows_total = 0
        self._bracket_rows_total = 0
        self._choice_rows_total = 0
        self._route_decision_counts: Counter[str] = Counter()
        self._route_reason_counts: Counter[str] = Counter()
        self._runtime_mode_counts: Counter[str] = Counter()
        self._chosen_family_counts: Counter[str] = Counter()
        self._lane_selected_counts: Counter[str] = Counter()
        self._variant_selected_counts: Counter[str] = Counter()
        self._bracket_mode_counts: Counter[str] = Counter()
        self._canonical_default_usage_count = 0
        self._route_conf_sum = 0.0
        self._route_conf_min: Optional[float] = None
        self._route_conf_max: Optional[float] = None
        self._route_margin_sum = 0.0
        self._route_margin_min: Optional[float] = None
        self._route_margin_max: Optional[float] = None
        self._lane_candidate_count_sum = 0.0
        self._bracket_sl_sum = 0.0
        self._bracket_tp_sum = 0.0
        self._direct_decision_selected_count = 0
        self._direct_decision_abstain_count = 0
        self._direct_decision_fallback_count = 0
        self._direct_decision_baseline_compare_fallback_count = 0

        logging.info(
            (
                "DE3v4 runtime initialized | bundle=%s loaded=%s mode=%s core_enabled=%s "
                "core_anchors=%s router=%s lane_selector=%s bracket_module=%s "
                "execution_policy_enabled=%s calibrated_entry_model_enabled=%s direct_decision_enabled=%s"
            ),
            self.bundle_path,
            self.bundle_loaded,
            self.runtime_mode,
            self.core_enabled,
            self.core_anchor_family_ids,
            True,
            True,
            True,
            bool(self.execution_policy_enabled),
            bool(self.calibrated_entry_model_enabled),
            bool(self.direct_decision_enabled),
        )

    @staticmethod
    def _resolve_bundle_path(raw_path: Any) -> Path:
        p = Path(str(raw_path or "").strip())
        if not p.is_absolute():
            p = Path(__file__).resolve().parent / p
        return p

    @staticmethod
    def _load_bundle(path: Path) -> Dict[str, Any]:
        if not path.exists():
            logging.warning("DE3v4 bundle missing: %s", path)
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception as exc:
            logging.warning("DE3v4 bundle unreadable (%s): %s", path, exc)
            return {}
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _parse_family_id(value: Any) -> Optional[Dict[str, Any]]:
        text = str(value or "").strip()
        if not text:
            return None
        parts = text.split("|")
        if len(parts) < 5:
            return None
        timeframe = str(parts[0] or "").strip()
        session = str(parts[1] or "").strip()
        side = str(parts[2] or "").strip().lower()
        strategy_type = str(parts[3] or "").strip()
        thresh_part = str(parts[4] or "").strip()
        family_tag = ""
        if len(parts) >= 6:
            family_tag = str(parts[5] or "").strip()
            if family_tag.upper().startswith("F"):
                family_tag = family_tag[1:]
        if thresh_part.upper().startswith("T"):
            thresh_part = thresh_part[1:]
        threshold_value = safe_float(thresh_part, float("nan"))
        threshold_text = format_threshold(thresh_part)
        if not timeframe or not strategy_type:
            return None
        return {
            "family_id": text,
            "timeframe": timeframe,
            "session": session,
            "side": side,
            "strategy_type": strategy_type,
            "threshold_value": threshold_value,
            "threshold_text": threshold_text,
            "family_tag": family_tag,
        }

    def _candidate_family_id(self, entry: Dict[str, Any], default_session: str) -> str:
        cand = entry.get("cand", {}) if isinstance(entry.get("cand"), dict) else {}
        cand_session = str(cand.get("session", "") or "").strip()
        family_session = cand_session or str(default_session or "").strip()
        return build_family_id(
            timeframe=cand.get("timeframe", ""),
            session=family_session,
            strategy_type=cand.get("strategy_type", ""),
            threshold=cand.get("thresh", 0.0),
            family_tag=cand.get("family_tag", cand.get("FamilyTag", "")),
        )

    def _candidate_variant_id(self, entry: Dict[str, Any]) -> str:
        cand = entry.get("cand", {}) if isinstance(entry.get("cand"), dict) else {}
        return str(entry.get("cand_id") or cand.get("strategy_id") or "").strip()

    def _candidate_lane(self, entry: Dict[str, Any]) -> str:
        cand = entry.get("cand", {}) if isinstance(entry.get("cand"), dict) else {}
        return strategy_type_to_lane(cand.get("strategy_type", ""))

    def _record_router_row(self, row: Dict[str, Any]) -> None:
        self._router_rows_total += 1
        route_decision = str(row.get("route_decision", "") or "")
        route_reason = str(row.get("route_reason", "") or "")
        route_conf = safe_float(row.get("route_confidence", 0.0), 0.0)
        route_margin = safe_float(row.get("route_margin", 0.0), 0.0)
        self._route_decision_counts[route_decision] += 1
        self._route_reason_counts[route_reason] += 1
        self._route_conf_sum += float(route_conf)
        self._route_margin_sum += float(route_margin)
        self._route_conf_min = (
            float(route_conf)
            if self._route_conf_min is None
            else float(min(self._route_conf_min, float(route_conf)))
        )
        self._route_conf_max = (
            float(route_conf)
            if self._route_conf_max is None
            else float(max(self._route_conf_max, float(route_conf)))
        )
        self._route_margin_min = (
            float(route_margin)
            if self._route_margin_min is None
            else float(min(self._route_margin_min, float(route_margin)))
        )
        self._route_margin_max = (
            float(route_margin)
            if self._route_margin_max is None
            else float(max(self._route_margin_max, float(route_margin)))
        )
        self._router_rows.append(row)

    def _record_choice_row(self, row: Dict[str, Any]) -> None:
        self._choice_rows_total += 1
        runtime_mode = str(row.get("runtime_mode", "") or "")
        if runtime_mode:
            self._runtime_mode_counts[runtime_mode] += 1
        chosen_family_id = str(row.get("chosen_family_id", "") or "").strip()
        if chosen_family_id:
            self._chosen_family_counts[chosen_family_id] += 1
        self._choice_rows.append(row)

    def _record_lane_row(self, row: Dict[str, Any]) -> None:
        self._lane_rows_total += 1
        selected_lane = str(row.get("selected_lane", "") or "")
        selected_variant_id = str(row.get("selected_variant_id", "") or "")
        self._lane_selected_counts[selected_lane] += 1
        self._variant_selected_counts[selected_variant_id] += 1
        lane_candidate_count = int(safe_float(row.get("lane_candidate_count", 0), 0))
        self._lane_candidate_count_sum += float(lane_candidate_count)
        self._lane_rows.append(row)

    def _record_bracket_row(self, row: Dict[str, Any]) -> None:
        self._bracket_rows_total += 1
        bracket_mode = str(row.get("bracket_mode", "") or "")
        self._bracket_mode_counts[bracket_mode] += 1
        selected_sl = safe_float(row.get("selected_sl", 0.0), 0.0)
        selected_tp = safe_float(row.get("selected_tp", 0.0), 0.0)
        self._bracket_sl_sum += float(selected_sl)
        self._bracket_tp_sum += float(selected_tp)
        if bool(row.get("canonical_default_used", False)):
            self._canonical_default_usage_count += 1
        self._bracket_rows.append(row)

    def _record_execution_policy_row(self, row: Dict[str, Any]) -> None:
        self._execution_policy_rows_total += 1
        if bool(row.get("allow", False)):
            self._execution_policy_allowed_count += 1
        self._execution_policy_rows.append(row)

    def _candidate_is_core_family(
        self,
        *,
        entry: Dict[str, Any],
        family_id: str,
        variant_id: str,
        default_session: str,
    ) -> bool:
        family_norm = str(family_id or "").strip()
        if family_norm and family_norm in self._core_anchor_set:
            return True
        cand = entry.get("cand", {}) if isinstance(entry.get("cand"), dict) else {}
        cand_tf = str(cand.get("timeframe", "") or "").strip()
        cand_stype = str(cand.get("strategy_type", "") or "").strip()
        cand_lane = strategy_type_to_lane(cand_stype)
        cand_side = lane_to_side(cand_lane)
        cand_session = str(cand.get("session", "") or "").strip() or str(default_session or "").strip()
        cand_thresh_text = format_threshold(cand.get("thresh", 0.0))
        sig_sessions = self._core_anchor_signature_index.get(
            (cand_tf, cand_stype, str(cand_side or "").lower(), cand_thresh_text),
            set(),
        )
        if sig_sessions:
            if "" in sig_sessions or not cand_session or cand_session in sig_sessions:
                return True
        # Fallback to variant-id signature matching when family-id/session mapping drifts.
        var_norm = str(variant_id or "").strip().lower()
        if var_norm:
            for token in self._core_anchor_variant_tokens:
                if token and token in var_norm:
                    return True
        return False

    def _candidate_excluded_by_runtime_filters(self, *, family_id: str, variant_id: str) -> Optional[str]:
        family_norm = str(family_id or "").strip()
        if family_norm and family_norm in self.excluded_family_ids:
            return "excluded_family_id"
        var_norm = str(variant_id or "").strip().lower()
        if var_norm:
            for pattern in self.excluded_variant_patterns:
                if pattern and pattern in var_norm:
                    return "excluded_variant_pattern"
        return None

    def _candidate_variant_filter_side(self, candidate_entry: Dict[str, Any]) -> str:
        side_raw = str(candidate_entry.get("side_considered", "") or "").strip().lower()
        if side_raw in {"long", "buy"}:
            return "long"
        if side_raw in {"short", "sell"}:
            return "short"
        lane = self._entry_model_lane(candidate_entry)
        lane_side = str(lane_to_side(lane) or "").strip().lower()
        if lane_side in {"long", "short"}:
            return lane_side
        side_fallback = str(candidate_entry.get("side", "") or "").strip().lower()
        if side_fallback in {"long", "buy"}:
            return "long"
        if side_fallback in {"short", "sell"}:
            return "short"
        return ""

    @staticmethod
    def _candidate_variant_filter_thresholds_from_cfg(
        *,
        filter_cfg: Optional[Dict[str, Any]],
        side_considered: str,
    ) -> Dict[str, Any]:
        cfg = dict(filter_cfg) if isinstance(filter_cfg, dict) else {}
        side_overrides = (
            cfg.get("side_overrides", {})
            if isinstance(cfg.get("side_overrides"), dict)
            else {}
        )
        default_cfg = (
            side_overrides.get("default", {})
            if isinstance(side_overrides.get("default"), dict)
            else {}
        )
        out = dict(default_cfg)
        for key, value in cfg.items():
            if key == "side_overrides":
                continue
            out[key] = value
        if side_considered:
            side_cfg = (
                side_overrides.get(side_considered, {})
                if isinstance(side_overrides.get(side_considered), dict)
                else {}
            )
            out.update(side_cfg)
        return out

    def _candidate_variant_filter_thresholds(self, *, side_considered: str) -> Dict[str, Any]:
        return self._candidate_variant_filter_thresholds_from_cfg(
            filter_cfg=self.candidate_variant_filter_cfg,
            side_considered=side_considered,
        )

    def _evaluate_candidate_variant_filter_with_cfg(
        self,
        *,
        candidate_entry: Dict[str, Any],
        filter_cfg: Optional[Dict[str, Any]],
        variant_stats: Optional[Dict[str, Any]] = None,
        enabled: Optional[bool] = None,
        context_label: str = "candidate_variant_filter",
    ) -> Dict[str, Any]:
        filter_enabled = bool(enabled) if enabled is not None else bool(
            isinstance(filter_cfg, dict) and filter_cfg.get("enabled", False)
        )
        if not filter_enabled:
            return {"enabled": False, "allow": True, "reason": "", "details": {}}
        stats_source = (
            variant_stats
            if isinstance(variant_stats, dict)
            else (
                self.direct_decision_model_cfg.get("variant_stats", {})
                if isinstance(self.direct_decision_model_cfg.get("variant_stats"), dict)
                else {}
            )
        )
        variant_id = self._entry_model_variant_id(candidate_entry)
        side_considered = self._candidate_variant_filter_side(candidate_entry)
        thresholds = self._candidate_variant_filter_thresholds_from_cfg(
            filter_cfg=filter_cfg,
            side_considered=side_considered,
        )
        allow_on_missing = bool(thresholds.get("allow_on_missing_variant_stats", True))
        stats = stats_source.get(variant_id, {}) if variant_id else {}
        if not isinstance(stats, dict) or not stats:
            return {
                "enabled": True,
                "allow": bool(allow_on_missing),
                "reason": "" if allow_on_missing else "missing_variant_stats",
                "details": {
                    "context_label": str(context_label),
                    "variant_id": str(variant_id),
                    "side_considered": str(side_considered),
                    "thresholds": dict(thresholds),
                },
            }

        n_trades = int(safe_float(stats.get("n_trades", 0), 0))
        profit_factor = float(safe_float(stats.get("profit_factor", 0.0), 0.0))
        quality_lcb = float(safe_float(stats.get("quality_lcb_score", 0.0), 0.0))
        year_coverage = int(safe_float(stats.get("year_coverage", 0), 0))
        p_win_lcb = float(safe_float(stats.get("p_win_lcb", 0.0), 0.0))
        loss_share = float(safe_float(stats.get("loss_share", 0.0), 0.0))
        stop_like_share = float(safe_float(stats.get("stop_like_share", 0.0), 0.0))
        drawdown_norm = float(safe_float(stats.get("drawdown_norm", 0.0), 0.0))
        worst_block_avg_pnl = float(safe_float(stats.get("worst_block_avg_pnl", 0.0), 0.0))

        checks = [
            ("n_trades_below_min", n_trades, thresholds.get("min_n_trades"), lambda a, b: a >= safe_int(b, 0)),
            ("profit_factor_below_min", profit_factor, thresholds.get("min_profit_factor"), lambda a, b: a >= safe_float(b, 0.0)),
            ("quality_lcb_below_min", quality_lcb, thresholds.get("min_quality_lcb_score"), lambda a, b: a >= safe_float(b, float("-inf"))),
            ("year_coverage_below_min", year_coverage, thresholds.get("min_year_coverage"), lambda a, b: a >= safe_int(b, 0)),
            ("p_win_lcb_below_min", p_win_lcb, thresholds.get("min_p_win_lcb"), lambda a, b: a >= safe_float(b, 0.0)),
            ("loss_share_above_max", loss_share, thresholds.get("max_loss_share"), lambda a, b: a <= safe_float(b, 1.0)),
            ("stop_like_share_above_max", stop_like_share, thresholds.get("max_stop_like_share"), lambda a, b: a <= safe_float(b, 1.0)),
            ("drawdown_norm_above_max", drawdown_norm, thresholds.get("max_drawdown_norm"), lambda a, b: a <= safe_float(b, float("inf"))),
            ("worst_block_avg_pnl_below_min", worst_block_avg_pnl, thresholds.get("min_worst_block_avg_pnl"), lambda a, b: a >= safe_float(b, float("-inf"))),
        ]
        for reason, actual_value, threshold_value, comparator in checks:
            if threshold_value is None:
                continue
            if not comparator(actual_value, threshold_value):
                return {
                    "enabled": True,
                    "allow": False,
                    "reason": str(reason),
                    "details": {
                        "context_label": str(context_label),
                        "variant_id": str(variant_id),
                        "side_considered": str(side_considered),
                        "actual_value": float(actual_value),
                        "threshold_value": float(safe_float(threshold_value, 0.0)),
                        "threshold_key": str(reason),
                        "stats": {
                            "n_trades": int(n_trades),
                            "profit_factor": float(profit_factor),
                            "quality_lcb_score": float(quality_lcb),
                            "year_coverage": int(year_coverage),
                            "p_win_lcb": float(p_win_lcb),
                            "loss_share": float(loss_share),
                            "stop_like_share": float(stop_like_share),
                            "drawdown_norm": float(drawdown_norm),
                            "worst_block_avg_pnl": float(worst_block_avg_pnl),
                        },
                    },
                }
        return {
            "enabled": True,
            "allow": True,
            "reason": "",
            "details": {
                "context_label": str(context_label),
                "variant_id": str(variant_id),
                "side_considered": str(side_considered),
                "stats": {
                    "n_trades": int(n_trades),
                    "profit_factor": float(profit_factor),
                    "quality_lcb_score": float(quality_lcb),
                    "year_coverage": int(year_coverage),
                    "p_win_lcb": float(p_win_lcb),
                    "loss_share": float(loss_share),
                    "stop_like_share": float(stop_like_share),
                    "drawdown_norm": float(drawdown_norm),
                    "worst_block_avg_pnl": float(worst_block_avg_pnl),
                },
            },
        }

    def _evaluate_candidate_variant_filter(
        self,
        *,
        candidate_entry: Dict[str, Any],
    ) -> Dict[str, Any]:
        return self._evaluate_candidate_variant_filter_with_cfg(
            candidate_entry=candidate_entry,
            filter_cfg=self.candidate_variant_filter_cfg,
            enabled=self.candidate_variant_filter_enabled,
            context_label="candidate_variant_filter",
        )

    @staticmethod
    def _normalize_book_gate_value(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return str(value).strip().lower()
        if isinstance(value, bool):
            return "true" if value else "false"
        try:
            numeric = float(value)
        except Exception:
            return str(value).strip().lower()
        if not math.isfinite(numeric):
            return ""
        rounded = round(numeric)
        if abs(numeric - rounded) <= 1e-9:
            return str(int(rounded))
        return f"{numeric:.4f}".rstrip("0").rstrip(".")

    @classmethod
    def _book_gate_bucket_key(
        cls,
        *,
        context_values: Dict[str, str],
        fields: List[str],
    ) -> str:
        if not fields:
            return "__global__"
        parts: List[str] = []
        for field in fields:
            key = str(field or "").strip()
            if not key:
                continue
            value = cls._normalize_book_gate_value(context_values.get(key, ""))
            if not value:
                return ""
            parts.append(f"{key}={value}")
        return "|".join(parts)

    def _resolve_book_gate_context(
        self,
        *,
        context_inputs: Optional[Dict[str, Any]],
        default_session: str,
        current_time: Optional[Any],
    ) -> Dict[str, str]:
        ctx = context_inputs if isinstance(context_inputs, dict) else {}
        current_ts = _to_timestamp(current_time)
        hour_default = current_ts.astimezone(NY_TZ).hour if isinstance(current_ts, dt.datetime) else None
        return {
            "session": self._normalize_book_gate_value(default_session),
            "ctx_volatility_regime": self._normalize_book_gate_value(ctx.get("volatility_regime")),
            "ctx_chop_trend_regime": self._normalize_book_gate_value(ctx.get("chop_trend_regime")),
            "ctx_compression_expansion_regime": self._normalize_book_gate_value(
                ctx.get("compression_expansion_regime")
            ),
            "ctx_confidence_band": self._normalize_book_gate_value(ctx.get("confidence_band")),
            "ctx_rvol_liquidity_state": self._normalize_book_gate_value(ctx.get("rvol_liquidity_state")),
            "ctx_session_substate": self._normalize_book_gate_value(ctx.get("session_substate")),
            "ctx_price_location": self._normalize_book_gate_value(ctx.get("price_location")),
            "ctx_hour_et": self._normalize_book_gate_value(ctx.get("hour_et", hour_default)),
            "side_considered": self._normalize_book_gate_value(ctx.get("side_considered")),
            "timeframe": self._normalize_book_gate_value(ctx.get("timeframe")),
            "strategy_type": self._normalize_book_gate_value(ctx.get("strategy_type")),
            "sub_strategy": self._normalize_book_gate_value(ctx.get("sub_strategy")),
        }

    def _book_gate_uses_row_context_fields(self) -> bool:
        if not bool(self.book_gate_enabled):
            return False
        if self._book_gate_supports_decision_model_switching():
            return False
        scope_priority = (
            self.book_gate_model_cfg.get("scope_priority", [])
            if isinstance(self.book_gate_model_cfg.get("scope_priority"), list)
            else []
        )
        row_fields = {"side_considered", "timeframe", "strategy_type", "sub_strategy"}
        for raw_scope in scope_priority:
            scope = raw_scope if isinstance(raw_scope, dict) else {}
            fields = (
                [str(v).strip() for v in scope.get("fields", []) if str(v).strip()]
                if isinstance(scope.get("fields"), list)
                else []
            )
            if any(field in row_fields for field in fields):
                return True
        return False

    def _book_gate_payload_for_book(self, book_name: str) -> Dict[str, Any]:
        books = (
            self.book_gate_model_cfg.get("books", {})
            if isinstance(self.book_gate_model_cfg.get("books"), dict)
            else {}
        )
        payload = books.get(book_name, {}) if isinstance(books.get(book_name, {}), dict) else {}
        return dict(payload) if isinstance(payload, dict) else {}

    def _merge_candidate_variant_filter_cfg(
        self,
        *,
        filter_cfg: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        base_cfg = dict(filter_cfg) if isinstance(filter_cfg, dict) else {}
        runtime_cfg = (
            dict(self._runtime_candidate_variant_filter_cfg)
            if isinstance(self._runtime_candidate_variant_filter_cfg, dict)
            else {}
        )
        if not runtime_cfg:
            return base_cfg
        merged_side_overrides = (
            dict(base_cfg.get("side_overrides", {}))
            if isinstance(base_cfg.get("side_overrides"), dict)
            else {}
        )
        runtime_side_overrides = (
            dict(runtime_cfg.get("side_overrides", {}))
            if isinstance(runtime_cfg.get("side_overrides"), dict)
            else {}
        )
        for key, value in runtime_cfg.items():
            if key == "side_overrides":
                continue
            base_cfg[key] = value
        if runtime_side_overrides:
            merged_side_overrides.update(runtime_side_overrides)
            base_cfg["side_overrides"] = merged_side_overrides
        return base_cfg

    def _book_gate_filter_cfg_for_book(self, book_name: str) -> Dict[str, Any]:
        payload = self._book_gate_payload_for_book(book_name)
        raw_filter_cfg = (
            dict(payload.get("candidate_variant_filter", {}))
            if isinstance(payload.get("candidate_variant_filter"), dict)
            else {}
        )
        if not raw_filter_cfg and str(book_name or "").strip() == str(
            self.book_gate_model_cfg.get("default_book", "") or ""
        ).strip():
            raw_filter_cfg = (
                dict(self.direct_decision_model_cfg.get("candidate_variant_filter", {}))
                if isinstance(self.direct_decision_model_cfg.get("candidate_variant_filter"), dict)
                else {}
            )
        return self._merge_candidate_variant_filter_cfg(filter_cfg=raw_filter_cfg)

    def _book_gate_decision_model_cfg_for_book(self, book_name: str) -> Dict[str, Any]:
        payload = self._book_gate_payload_for_book(book_name)
        decision_model_cfg = (
            dict(payload.get("decision_policy_model", {}))
            if isinstance(payload.get("decision_policy_model"), dict)
            else {}
        )
        if not decision_model_cfg and str(book_name or "").strip() == str(
            self.book_gate_model_cfg.get("default_book", "") or ""
        ).strip():
            decision_model_cfg = dict(self.direct_decision_model_cfg)
        if decision_model_cfg and not isinstance(
            decision_model_cfg.get("candidate_variant_filter"), dict
        ):
            filter_cfg = self._book_gate_filter_cfg_for_book(book_name)
            if filter_cfg:
                decision_model_cfg["candidate_variant_filter"] = dict(filter_cfg)
        return decision_model_cfg

    def _book_gate_supports_decision_model_switching(self) -> bool:
        if not bool(self.book_gate_enabled):
            return False
        books = (
            self.book_gate_model_cfg.get("books", {})
            if isinstance(self.book_gate_model_cfg.get("books"), dict)
            else {}
        )
        for payload in books.values():
            if not isinstance(payload, dict):
                continue
            decision_model_cfg = payload.get("decision_policy_model", {})
            if isinstance(decision_model_cfg, dict) and decision_model_cfg:
                return True
        return False

    def _record_book_gate_selection(self, meta: Optional[Dict[str, Any]]) -> None:
        if not isinstance(meta, dict):
            return
        if not bool(meta.get("enabled", False)):
            return
        if bool(meta.get("stats_recorded", False)):
            return
        self._book_gate_decision_count += 1
        final_book = str(meta.get("selected_book", "") or meta.get("default_book", "") or "")
        default_book = str(meta.get("default_book", "") or "")
        if final_book == default_book:
            self._book_gate_default_selection_count += 1
        if final_book:
            self._book_gate_selected_book_counts[final_book] += 1
        matched_scope = str(meta.get("matched_scope", "") or "")
        if matched_scope:
            self._book_gate_scope_match_counts[matched_scope] += 1
        if bool(meta.get("fallback_to_default", False)):
            self._book_gate_empty_fallback_count += 1
        meta["stats_recorded"] = True

    def _book_gate_context_proxy_row(
        self,
        *,
        candidate_rows: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        rows = [dict(row) for row in candidate_rows if isinstance(row, dict)]
        if not rows:
            return None
        return max(
            rows,
            key=lambda row: (
                1 if safe_int(row.get("candidate_rank_before_adjustments", 0), 0) > 0 else 0,
                -safe_int(row.get("candidate_rank_before_adjustments", 0), 0),
                safe_float(
                    row.get("runtime_rank_score", row.get("edge_points", row.get("lane_score", 0.0))),
                    0.0,
                ),
                safe_float(row.get("structural_score", 0.0), 0.0),
            ),
        )

    def _candidate_row_context_value(self, row: Optional[Dict[str, Any]], key: str) -> Any:
        if not isinstance(row, dict):
            return None
        key_norm = str(key or "").strip()
        if not key_norm:
            return None
        if key_norm in row and row.get(key_norm) is not None and str(row.get(key_norm)).strip():
            return row.get(key_norm)
        cand = row.get("cand", {}) if isinstance(row.get("cand"), dict) else {}
        if key_norm in {"timeframe", "strategy_type", "thresh"}:
            value = cand.get(key_norm)
            if value is not None and str(value).strip():
                return value
        if key_norm == "sub_strategy":
            for alt_key in ("sub_strategy", "selected_variant_id", "variant_id", "strategy_id"):
                value = row.get(alt_key)
                if value is not None and str(value).strip():
                    return value
            value = cand.get("strategy_id")
            if value is not None and str(value).strip():
                return value
        if key_norm == "side_considered":
            value = row.get("side_considered")
            if value is not None and str(value).strip():
                return value
            side_raw = str(row.get("side", "") or cand.get("signal", "") or "").strip().lower()
            if side_raw in {"long", "buy"}:
                return "long"
            if side_raw in {"short", "sell"}:
                return "short"
            lane = str(row.get("lane", "") or "")
            lane_side = lane_to_side(lane)
            if lane_side:
                return lane_side
        return None

    def _filter_candidate_rows_with_cfg(
        self,
        *,
        candidate_rows: List[Dict[str, Any]],
        filter_cfg: Optional[Dict[str, Any]],
        variant_stats: Optional[Dict[str, Any]] = None,
        context_label: str,
        annotate_prefix: str,
        count_rejections: bool,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        filter_enabled = bool(isinstance(filter_cfg, dict) and filter_cfg.get("enabled", False))
        if not filter_enabled:
            copied = [dict(row) for row in candidate_rows if isinstance(row, dict)]
            return copied, {
                "enabled": False,
                "excluded": 0,
                "reject_reason_counts": {},
            }

        filtered_rows: List[Dict[str, Any]] = []
        excluded = 0
        reject_reasons: Counter[str] = Counter()
        for raw_row in candidate_rows:
            if not isinstance(raw_row, dict):
                continue
            row = dict(raw_row)
            filter_eval = self._evaluate_candidate_variant_filter_with_cfg(
                candidate_entry=row,
                filter_cfg=filter_cfg,
                variant_stats=variant_stats,
                enabled=filter_enabled,
                context_label=context_label,
            )
            row[f"{annotate_prefix}_allow"] = bool(filter_eval.get("allow", True))
            row[f"{annotate_prefix}_reason"] = str(filter_eval.get("reason", "") or "")
            row[f"{annotate_prefix}_details"] = (
                dict(filter_eval.get("details", {}))
                if isinstance(filter_eval.get("details", {}), dict)
                else {}
            )
            if not bool(filter_eval.get("allow", True)):
                excluded += 1
                reason = str(filter_eval.get("reason", "") or "filtered")
                reject_reasons[reason] += 1
                if count_rejections:
                    self._candidate_variant_filter_reject_count += 1
                    self._candidate_variant_filter_reject_reason_counts[reason] += 1
                continue
            filtered_rows.append(row)
        return filtered_rows, {
            "enabled": True,
            "excluded": int(excluded),
            "reject_reason_counts": dict(reject_reasons),
        }

    def _select_book_gate(
        self,
        *,
        context_inputs: Optional[Dict[str, Any]],
        default_session: str,
        current_time: Optional[Any],
    ) -> Dict[str, Any]:
        books = (
            self.book_gate_model_cfg.get("books", {})
            if isinstance(self.book_gate_model_cfg.get("books"), dict)
            else {}
        )
        default_book = str(self.book_gate_model_cfg.get("default_book", "") or "").strip()
        if not default_book and books:
            default_book = str(next(iter(books.keys())))
        if (not self.book_gate_enabled) or (not books):
            return {
                "enabled": False,
                "selected_book": str(default_book),
                "requested_book": str(default_book),
                "default_book": str(default_book),
                "matched_scope": "",
                "matched_bucket": "",
                "reason": "disabled",
                "context": self._resolve_book_gate_context(
                    context_inputs=context_inputs,
                    default_session=default_session,
                    current_time=current_time,
                ),
            }

        context = self._resolve_book_gate_context(
            context_inputs=context_inputs,
            default_session=default_session,
            current_time=current_time,
        )
        scope_priority = (
            self.book_gate_model_cfg.get("scope_priority", [])
            if isinstance(self.book_gate_model_cfg.get("scope_priority"), list)
            else []
        )
        bucket_overrides = (
            self.book_gate_model_cfg.get("bucket_overrides", {})
            if isinstance(self.book_gate_model_cfg.get("bucket_overrides"), dict)
            else {}
        )
        selected_book = str(default_book)
        matched_scope = ""
        matched_bucket = ""
        reason = "default_book"
        for raw_scope in scope_priority:
            scope = raw_scope if isinstance(raw_scope, dict) else {}
            fields = (
                [str(v).strip() for v in scope.get("fields", []) if str(v).strip()]
                if isinstance(scope.get("fields"), list)
                else []
            )
            scope_name = str(scope.get("name", "") or "").strip()
            if not scope_name:
                scope_name = "__".join(fields) if fields else "global"
            bucket_key = self._book_gate_bucket_key(context_values=context, fields=fields)
            if not bucket_key:
                continue
            scope_map = (
                bucket_overrides.get(scope_name, {})
                if isinstance(bucket_overrides.get(scope_name), dict)
                else {}
            )
            payload = scope_map.get(bucket_key, {}) if isinstance(scope_map.get(bucket_key, {}), dict) else {}
            book_name = str(payload.get("book", "") or "").strip()
            if not book_name or book_name not in books:
                continue
            selected_book = str(book_name)
            matched_scope = str(scope_name)
            matched_bucket = str(bucket_key)
            reason = "bucket_override"
            break

        return {
            "enabled": True,
            "selected_book": str(selected_book),
            "requested_book": str(selected_book),
            "default_book": str(default_book),
            "matched_scope": str(matched_scope),
            "matched_bucket": str(matched_bucket),
            "reason": str(reason),
            "context": dict(context),
        }

    def _select_with_model_stack(
        self,
        *,
        candidate_rows: List[Dict[str, Any]],
        default_session: str,
        decision_model_cfg: Optional[Dict[str, Any]],
        track_counters: bool,
    ) -> Dict[str, Any]:
        lane_candidates: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in candidate_rows:
            if not isinstance(row, dict):
                continue
            lane_candidates[str(row.get("lane", ""))].append(row)
        timeframe_hint = self._dominant_timeframe(candidate_rows)
        direct_result = self._select_with_direct_decision_model(
            filtered_rows=candidate_rows,
            default_session=str(default_session or ""),
            model_cfg=decision_model_cfg,
            track_counters=track_counters,
        )
        selection_mode = str(
            (
                decision_model_cfg.get("selection_mode", "")
                if isinstance(decision_model_cfg, dict)
                else ""
            )
            or self.direct_decision_selection_mode
            or "replace_router_lane"
        ).strip().lower()
        baseline_result = (
            self._select_with_router_lane_stack(
                lane_candidates=lane_candidates,
                default_session=str(default_session or ""),
                timeframe_hint=str(timeframe_hint or ""),
            )
            if selection_mode == "hybrid_compare_baseline"
            else None
        )
        if selection_mode == "hybrid_compare_baseline":
            direct_result = self._maybe_override_baseline_with_direct(
                direct_result=direct_result,
                baseline_result=baseline_result or {},
                default_session=str(default_session or ""),
                model_cfg=decision_model_cfg,
                track_counters=track_counters,
            )
        final_result = direct_result if isinstance(direct_result, dict) else baseline_result
        return {
            "selection_mode": str(selection_mode),
            "direct_result": direct_result,
            "baseline_result": baseline_result,
            "final_result": final_result,
            "lane_candidates": lane_candidates,
            "timeframe_hint": str(timeframe_hint or ""),
        }

    def _apply_book_gate_decision_switch(
        self,
        *,
        candidate_rows: List[Dict[str, Any]],
        context_inputs: Optional[Dict[str, Any]],
        default_session: str,
        current_time: Optional[Any],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        selection = self._select_book_gate(
            context_inputs=context_inputs,
            default_session=default_session,
            current_time=current_time,
        )
        if not bool(selection.get("enabled", False)):
            return candidate_rows, selection

        default_book = str(selection.get("default_book", "") or "")
        if not default_book:
            default_book = str(selection.get("selected_book", "") or "")
        provisional_book = str(default_book or selection.get("selected_book", "") or "")
        provisional_filter_cfg = self._book_gate_filter_cfg_for_book(provisional_book)
        provisional_decision_model_cfg = self._book_gate_decision_model_cfg_for_book(provisional_book)
        provisional_variant_stats = (
            provisional_decision_model_cfg.get("variant_stats", {})
            if isinstance(provisional_decision_model_cfg.get("variant_stats"), dict)
            else {}
        )
        provisional_rows, _ = self._filter_candidate_rows_with_cfg(
            candidate_rows=candidate_rows,
            filter_cfg=provisional_filter_cfg,
            variant_stats=provisional_variant_stats,
            context_label=f"book_gate_provisional:{provisional_book}",
            annotate_prefix="book_gate_provisional_filter",
            count_rejections=False,
        )
        provisional_stack = self._select_with_model_stack(
            candidate_rows=provisional_rows,
            default_session=default_session,
            decision_model_cfg=provisional_decision_model_cfg,
            track_counters=False,
        )
        provisional_result = (
            dict(provisional_stack.get("final_result", {}))
            if isinstance(provisional_stack.get("final_result"), dict)
            else {}
        )
        chosen_context_row = (
            dict(provisional_result.get("chosen_entry", {}))
            if isinstance(provisional_result.get("chosen_entry"), dict)
            else None
        )
        base_context_row = self._book_gate_context_proxy_row(
            candidate_rows=provisional_rows if provisional_rows else candidate_rows
        )

        merged_context: Dict[str, Any] = {}
        if isinstance(context_inputs, dict):
            merged_context.update(context_inputs)
        if isinstance(base_context_row, dict):
            for key in ("timeframe", "strategy_type", "sub_strategy"):
                value = self._candidate_row_context_value(base_context_row, key)
                if value is not None and str(value).strip():
                    merged_context[key] = value
        if isinstance(chosen_context_row, dict):
            for key in ("side_considered", "sub_strategy"):
                value = self._candidate_row_context_value(chosen_context_row, key)
                if value is not None and str(value).strip():
                    merged_context[key] = value
        elif isinstance(base_context_row, dict):
            value = self._candidate_row_context_value(base_context_row, "side_considered")
            if value is not None and str(value).strip():
                merged_context["side_considered"] = value
        selection = self._select_book_gate(
            context_inputs=merged_context,
            default_session=default_session,
            current_time=current_time,
        )
        context_signature = self._book_gate_bucket_key(
            context_values=self._resolve_book_gate_context(
                context_inputs=merged_context,
                default_session=default_session,
                current_time=current_time,
            ),
            fields=["session", "side_considered", "strategy_type"],
        )
        if context_signature:
            self._book_gate_context_key_counts[context_signature] += 1
        meta = dict(selection)
        meta.update(
            {
                "decision_policy_switching": True,
                "filter_enabled": False,
                "fallback_to_default": False,
                "candidate_count_before_filter": int(len(candidate_rows)),
                "candidate_count_excluded": 0,
                "candidate_count_after_filter": int(len(candidate_rows)),
                "reject_reason_counts": {},
                "provisional_book": str(provisional_book),
                "provisional_route_decision": str(
                    provisional_result.get("route_decision", LANE_NO_TRADE) or LANE_NO_TRADE
                ),
                "provisional_selected_variant_id": str(
                    provisional_result.get("selected_variant_id", "") or ""
                ),
                "provisional_selection_mode": str(
                    provisional_stack.get("selection_mode", "") or ""
                ),
                "stats_recorded": False,
            }
        )
        for row in candidate_rows:
            if not isinstance(row, dict):
                continue
            row["book_gate_requested_book"] = str(meta.get("requested_book", "") or "")
            row["book_gate_selected_book"] = str(meta.get("selected_book", "") or "")
            row["book_gate_scope"] = str(meta.get("matched_scope", "") or "")
            row["book_gate_bucket"] = str(meta.get("matched_bucket", "") or "")
            row["book_gate_reason"] = str(meta.get("reason", "") or "")
            row["book_gate_fallback_to_default"] = False
        return candidate_rows, meta

    def _apply_book_gate(
        self,
        *,
        candidate_rows: List[Dict[str, Any]],
        context_inputs: Optional[Dict[str, Any]],
        default_session: str,
        current_time: Optional[Any],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if self._book_gate_supports_decision_model_switching():
            return self._apply_book_gate_decision_switch(
                candidate_rows=candidate_rows,
                context_inputs=context_inputs,
                default_session=default_session,
                current_time=current_time,
            )
        if self._book_gate_uses_row_context_fields():
            return self._apply_book_gate_rowwise(
                candidate_rows=candidate_rows,
                context_inputs=context_inputs,
                default_session=default_session,
                current_time=current_time,
            )

        selection = self._select_book_gate(
            context_inputs=context_inputs,
            default_session=default_session,
            current_time=current_time,
        )
        if not bool(selection.get("enabled", False)):
            return candidate_rows, selection

        self._book_gate_decision_count += 1
        selected_book = str(selection.get("selected_book", "") or "")
        default_book = str(selection.get("default_book", "") or "")
        selected_filter_cfg = self._book_gate_filter_cfg_for_book(selected_book)
        selected_filter_enabled = bool(selected_filter_cfg.get("enabled", False))

        pre_rows = list(candidate_rows)
        filtered_rows = list(pre_rows)
        excluded = 0
        reject_reasons: Counter[str] = Counter()
        if selected_filter_enabled:
            filtered_rows = []
            for row in pre_rows:
                if not isinstance(row, dict):
                    continue
                filter_eval = self._evaluate_candidate_variant_filter_with_cfg(
                    candidate_entry=row,
                    filter_cfg=selected_filter_cfg,
                    context_label=f"book_gate:{selected_book}",
                )
                row["book_gate_filter_allow"] = bool(filter_eval.get("allow", True))
                row["book_gate_filter_reason"] = str(filter_eval.get("reason", "") or "")
                row["book_gate_filter_details"] = (
                    dict(filter_eval.get("details", {}))
                    if isinstance(filter_eval.get("details"), dict)
                    else {}
                )
                row["book_gate_requested_book"] = str(selected_book)
                if not bool(filter_eval.get("allow", True)):
                    excluded += 1
                    reject_reasons[str(filter_eval.get("reason", "") or "filtered")] += 1
                    continue
                filtered_rows.append(row)

        fallback_to_default = False
        if (
            (not filtered_rows)
            and bool(self.book_gate_model_cfg.get("fallback_to_default_on_empty", True))
            and default_book
            and selected_book
            and selected_book != default_book
        ):
            fallback_to_default = True
            self._book_gate_empty_fallback_count += 1
            selected_book = str(default_book)
            selected_filter_cfg = self._book_gate_filter_cfg_for_book(selected_book)
            selected_filter_enabled = bool(selected_filter_cfg.get("enabled", False))
            filtered_rows = list(pre_rows)
            excluded = 0
            reject_reasons = Counter()
            if selected_filter_enabled:
                filtered_rows = []
                for row in pre_rows:
                    if not isinstance(row, dict):
                        continue
                    filter_eval = self._evaluate_candidate_variant_filter_with_cfg(
                        candidate_entry=row,
                        filter_cfg=selected_filter_cfg,
                        context_label=f"book_gate:{selected_book}",
                    )
                    row["book_gate_filter_allow"] = bool(filter_eval.get("allow", True))
                    row["book_gate_filter_reason"] = str(filter_eval.get("reason", "") or "")
                    row["book_gate_filter_details"] = (
                        dict(filter_eval.get("details", {}))
                        if isinstance(filter_eval.get("details"), dict)
                        else {}
                    )
                    row["book_gate_requested_book"] = str(selection.get("requested_book", ""))
                    row["book_gate_selected_book"] = str(selected_book)
                    if not bool(filter_eval.get("allow", True)):
                        excluded += 1
                        reject_reasons[str(filter_eval.get("reason", "") or "filtered")] += 1
                        continue
                    filtered_rows.append(row)

        final_book = str(selected_book or default_book)
        if final_book == default_book:
            self._book_gate_default_selection_count += 1
        self._book_gate_selected_book_counts[final_book] += 1
        matched_scope = str(selection.get("matched_scope", "") or "")
        if matched_scope:
            self._book_gate_scope_match_counts[matched_scope] += 1

        meta = dict(selection)
        meta.update(
            {
                "selected_book": str(final_book),
                "filter_enabled": bool(selected_filter_enabled),
                "fallback_to_default": bool(fallback_to_default),
                "candidate_count_before_filter": int(len(pre_rows)),
                "candidate_count_excluded": int(excluded),
                "candidate_count_after_filter": int(len(filtered_rows)),
                "reject_reason_counts": dict(reject_reasons),
                "stats_recorded": True,
            }
        )
        for row in filtered_rows:
            if not isinstance(row, dict):
                continue
            row["book_gate_selected_book"] = str(final_book)
            row["book_gate_scope"] = str(meta.get("matched_scope", "") or "")
            row["book_gate_bucket"] = str(meta.get("matched_bucket", "") or "")
            row["book_gate_reason"] = str(meta.get("reason", "") or "")
            row["book_gate_fallback_to_default"] = bool(fallback_to_default)
        return filtered_rows, meta

    def _apply_book_gate_rowwise(
        self,
        *,
        candidate_rows: List[Dict[str, Any]],
        context_inputs: Optional[Dict[str, Any]],
        default_session: str,
        current_time: Optional[Any],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        books = (
            self.book_gate_model_cfg.get("books", {})
            if isinstance(self.book_gate_model_cfg.get("books"), dict)
            else {}
        )
        default_book = str(self.book_gate_model_cfg.get("default_book", "") or "").strip()
        if not default_book and books:
            default_book = str(next(iter(books.keys())))
        if (not bool(self.book_gate_enabled)) or (not books):
            return candidate_rows, {
                "enabled": False,
                "selected_book": str(default_book),
                "default_book": str(default_book),
                "matched_scope": "",
                "matched_bucket": "",
                "reason": "disabled",
            }

        self._book_gate_decision_count += 1
        pre_rows = list(candidate_rows)
        filtered_rows: List[Dict[str, Any]] = []
        selected_books: Counter[str] = Counter()
        matched_scopes: Counter[str] = Counter()
        reject_reasons: Counter[str] = Counter()
        excluded = 0

        for raw_row in pre_rows:
            if not isinstance(raw_row, dict):
                continue
            row = dict(raw_row)
            merged_context: Dict[str, Any] = {}
            if isinstance(context_inputs, dict):
                merged_context.update(context_inputs)
            for key in ("side_considered", "timeframe", "strategy_type", "sub_strategy"):
                if key in row:
                    merged_context[key] = row.get(key)

            selection = self._select_book_gate(
                context_inputs=merged_context,
                default_session=default_session,
                current_time=current_time,
            )
            row_book = str(selection.get("selected_book", "") or default_book)
            if not row_book:
                row_book = str(default_book)
            selected_books[row_book] += 1
            matched_scope = str(selection.get("matched_scope", "") or "")
            if matched_scope:
                matched_scopes[matched_scope] += 1

            selected_filter_cfg = self._book_gate_filter_cfg_for_book(row_book)
            selected_filter_enabled = bool(selected_filter_cfg.get("enabled", False))
            allow = True
            filter_eval: Dict[str, Any] = {}
            if selected_filter_enabled:
                filter_eval = self._evaluate_candidate_variant_filter_with_cfg(
                    candidate_entry=row,
                    filter_cfg=selected_filter_cfg,
                    context_label=f"book_gate_row:{row_book}",
                )
                allow = bool(filter_eval.get("allow", True))
                row["book_gate_filter_allow"] = bool(allow)
                row["book_gate_filter_reason"] = str(filter_eval.get("reason", "") or "")
                row["book_gate_filter_details"] = (
                    dict(filter_eval.get("details", {}))
                    if isinstance(filter_eval.get("details"), dict)
                    else {}
                )
            row["book_gate_requested_book"] = str(selection.get("requested_book", row_book) or row_book)
            row["book_gate_selected_book"] = str(row_book)
            row["book_gate_scope"] = str(matched_scope)
            row["book_gate_bucket"] = str(selection.get("matched_bucket", "") or "")
            row["book_gate_reason"] = str(selection.get("reason", "") or "")
            row["book_gate_fallback_to_default"] = False
            if not allow:
                excluded += 1
                reject_reasons[str(filter_eval.get("reason", "") or "filtered")] += 1
                continue
            filtered_rows.append(row)

        fallback_to_default = False
        if (
            (not filtered_rows)
            and bool(self.book_gate_model_cfg.get("fallback_to_default_on_empty", True))
            and default_book
        ):
            fallback_to_default = True
            self._book_gate_empty_fallback_count += 1
            filtered_rows = list(pre_rows)
            selected_books = Counter({str(default_book): len(filtered_rows)})
            matched_scopes = Counter()
            reject_reasons = Counter()
            excluded = 0

        dominant_book = str(default_book)
        if selected_books:
            dominant_book = str(selected_books.most_common(1)[0][0] or default_book)
        if dominant_book == default_book:
            self._book_gate_default_selection_count += 1
        self._book_gate_selected_book_counts.update(selected_books)
        self._book_gate_scope_match_counts.update(matched_scopes)

        meta = {
            "enabled": True,
            "selected_book": str(dominant_book if not fallback_to_default else default_book),
            "default_book": str(default_book),
            "matched_scope": "",
            "matched_bucket": "",
            "reason": "row_bucket_filter" if not fallback_to_default else "fallback_to_default",
            "fallback_to_default": bool(fallback_to_default),
            "selected_book_counts": dict(selected_books),
            "scope_match_counts": dict(matched_scopes),
            "candidate_count_before_filter": int(len(pre_rows)),
            "candidate_count_excluded": int(excluded),
            "candidate_count_after_filter": int(len(filtered_rows)),
            "reject_reason_counts": dict(reject_reasons),
            "stats_recorded": True,
        }
        return filtered_rows, meta

    def _mode_filter(
        self,
        *,
        candidate_rows: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if self.runtime_mode == "core_plus_satellites":
            return candidate_rows
        if self.runtime_mode == "core_only":
            return [row for row in candidate_rows if bool(row.get("is_core_family", False))]
        if self.runtime_mode == "satellites_only":
            return [row for row in candidate_rows if not bool(row.get("is_core_family", False))]
        return candidate_rows

    @staticmethod
    def _dominant_timeframe(rows: List[Dict[str, Any]]) -> str:
        counts: Dict[str, int] = {}
        for row in rows:
            cand = row.get("cand", {}) if isinstance(row.get("cand"), dict) else {}
            tf = str(cand.get("timeframe", "") or "")
            if not tf:
                continue
            counts[tf] = int(counts.get(tf, 0) + 1)
        if not counts:
            return ""
        return max(counts, key=counts.get)

    @staticmethod
    def _abstain_result(
        *,
        reason: str,
        feasible_rows: List[Dict[str, Any]],
        decision_export_rows: Optional[List[Dict[str, Any]]] = None,
        route_decision: str,
        route_confidence: float,
        route_margin: float,
        route_scores: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        return {
            "abstained": True,
            "abstain_reason": str(reason or ""),
            "feasible_rows": feasible_rows,
            "decision_export_rows": (
                decision_export_rows
                if isinstance(decision_export_rows, list)
                else feasible_rows
            ),
            "route_decision": str(route_decision or LANE_NO_TRADE),
            "route_confidence": float(route_confidence),
            "route_margin": float(route_margin),
            "route_scores": dict(route_scores or {}),
        }

    @staticmethod
    def _snapshot_candidate_rows_for_export(
        candidate_rows: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        snapshot: List[Dict[str, Any]] = []
        for raw_row in candidate_rows:
            if not isinstance(raw_row, dict):
                continue
            row = dict(raw_row)
            for key in (
                "cand",
                "exec",
                "lane_score_inputs",
                "family_context_inputs",
            ):
                value = row.get(key)
                if isinstance(value, dict):
                    row[key] = dict(value)
            for key in (
                "feasible_family_ids",
                "core_anchor_family_ids",
                "decision_side_model_matched_scopes",
                "conflict_side_model_matched_scopes",
            ):
                value = row.get(key)
                if isinstance(value, list):
                    row[key] = list(value)
            snapshot.append(row)
        return snapshot

    @staticmethod
    def _optional_float(value: Any) -> Optional[float]:
        try:
            out = float(value)
        except Exception:
            return None
        if not math.isfinite(out):
            return None
        return float(out)

    @classmethod
    def _cfg_float(cls, mapping: Dict[str, Any], key: str, default: Optional[float] = None) -> Optional[float]:
        if not isinstance(mapping, dict):
            return default
        if key not in mapping:
            return default
        raw = cls._optional_float(mapping.get(key))
        if raw is None:
            return default
        return float(raw)

    @staticmethod
    def _bounded_unit(value: float, lo: float, hi: float, *, invert: bool = False) -> float:
        if not math.isfinite(value):
            return 0.0
        if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
            return 0.0
        score = (float(value) - float(lo)) / (float(hi) - float(lo))
        if invert:
            score = 1.0 - score
        return float(max(0.0, min(1.0, score)))

    def _build_execution_policy_cfg(
        self,
        *,
        runtime_cfg: Dict[str, Any],
        legacy_execution_filters_cfg: Dict[str, Any],
        bundle_entry_policy_model: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        policy_cfg_raw = (
            dict(runtime_cfg.get("execution_policy", {}))
            if isinstance(runtime_cfg.get("execution_policy"), dict)
            else {}
        )
        bundle_entry_model = (
            dict(bundle_entry_policy_model)
            if isinstance(bundle_entry_policy_model, dict)
            else {}
        )
        hard_cfg = (
            dict(policy_cfg_raw.get("hard_limits", {}))
            if isinstance(policy_cfg_raw.get("hard_limits"), dict)
            else {}
        )
        weights_cfg = (
            dict(policy_cfg_raw.get("weights", {}))
            if isinstance(policy_cfg_raw.get("weights"), dict)
            else {}
        )
        ranges_cfg = (
            dict(policy_cfg_raw.get("ranges", {}))
            if isinstance(policy_cfg_raw.get("ranges"), dict)
            else {}
        )

        def _pick_float(
            key: str,
            *,
            hard_key: Optional[str] = None,
            default: Optional[float] = None,
        ) -> Optional[float]:
            if hard_key and hard_key in hard_cfg:
                return self._cfg_float(hard_cfg, hard_key, default)
            if key in policy_cfg_raw:
                return self._cfg_float(policy_cfg_raw, key, default)
            if key in legacy_execution_filters_cfg:
                return self._cfg_float(legacy_execution_filters_cfg, key, default)
            return default

        policy_enabled = bool(
            policy_cfg_raw.get(
                "enabled",
                legacy_execution_filters_cfg.get("enabled", False),
            )
        )
        calibrated_model_cfg_raw = (
            dict(policy_cfg_raw.get("calibrated_entry_model", {}))
            if isinstance(policy_cfg_raw.get("calibrated_entry_model"), dict)
            else {}
        )
        calibrated_model_enabled = bool(
            calibrated_model_cfg_raw.get(
                "enabled",
                bundle_entry_model.get("enabled", False),
            )
        )
        use_bundle_entry_model = bool(
            calibrated_model_cfg_raw.get("use_bundle_model", False)
            and isinstance(bundle_entry_model, dict)
            and bool(bundle_entry_model)
        )
        calibrated_model_param_cfg = {} if use_bundle_entry_model else calibrated_model_cfg_raw

        weights = {
            "route_confidence": self._cfg_float(weights_cfg, "route_confidence", 0.22),
            "edge_points": self._cfg_float(weights_cfg, "edge_points", 0.28),
            "lane_score": self._cfg_float(weights_cfg, "lane_score", 0.22),
            "structural_score": self._cfg_float(weights_cfg, "structural_score", 0.12),
            "variant_quality_prior": self._cfg_float(weights_cfg, "variant_quality_prior", 0.08),
            "loss_quality": self._cfg_float(weights_cfg, "loss_quality", 0.04),
            "stop_quality": self._cfg_float(weights_cfg, "stop_quality", 0.04),
        }
        weight_total = sum(float(v or 0.0) for v in weights.values())
        if weight_total <= 0.0:
            weights = {
                "route_confidence": 0.25,
                "edge_points": 0.25,
                "lane_score": 0.20,
                "structural_score": 0.10,
                "variant_quality_prior": 0.10,
                "loss_quality": 0.05,
                "stop_quality": 0.05,
            }
            weight_total = 1.0

        def _range(name: str, lo_default: float, hi_default: float) -> Dict[str, float]:
            raw = ranges_cfg.get(name, {}) if isinstance(ranges_cfg.get(name), dict) else {}
            lo = self._cfg_float(raw, "min", lo_default)
            hi = self._cfg_float(raw, "max", hi_default)
            if lo is None:
                lo = lo_default
            if hi is None:
                hi = hi_default
            if hi <= lo:
                hi = lo + 1e-9
            return {"min": float(lo), "max": float(hi)}

        score_components_cfg = (
            dict(bundle_entry_model.get("score_components", {}))
            if isinstance(bundle_entry_model.get("score_components"), dict)
            else {}
        )
        minimums_cfg = (
            dict(bundle_entry_model.get("minimums", {}))
            if isinstance(bundle_entry_model.get("minimums"), dict)
            else {}
        )
        scope_offsets_cfg = (
            dict(bundle_entry_model.get("scope_threshold_offsets", {}))
            if isinstance(bundle_entry_model.get("scope_threshold_offsets"), dict)
            else {}
        )
        shape_model_cfg = (
            dict(bundle_entry_model.get("shape_penalty_model", {}))
            if isinstance(bundle_entry_model.get("shape_penalty_model"), dict)
            else {}
        )
        scope_offsets_override = (
            dict(calibrated_model_param_cfg.get("scope_threshold_offsets", {}))
            if isinstance(calibrated_model_param_cfg.get("scope_threshold_offsets"), dict)
            else {}
        )
        shape_model_override = (
            dict(calibrated_model_param_cfg.get("shape_penalty_model", {}))
            if isinstance(calibrated_model_param_cfg.get("shape_penalty_model"), dict)
            else {}
        )
        fallback_guard_cfg_raw = (
            dict(calibrated_model_cfg_raw.get("fallback_scope_guard", {}))
            if isinstance(calibrated_model_cfg_raw.get("fallback_scope_guard"), dict)
            else {}
        )

        def _scope_list_from_cfg(
            primary: Dict[str, Any],
            key: str,
            fallback_primary: Dict[str, Any],
            fallback_key: str,
            default_scopes: List[str],
        ) -> List[str]:
            raw = primary.get(key, None)
            if not isinstance(raw, (list, tuple, set)):
                raw = fallback_primary.get(fallback_key, None)
            if not isinstance(raw, (list, tuple, set)):
                raw = list(default_scopes)
            out = []
            for item in raw:
                name = str(item or "").strip().lower()
                if name in {"variant", "lane", "global", "missing"}:
                    out.append(name)
            return sorted(set(out))

        fallback_guard_scopes = _scope_list_from_cfg(
            fallback_guard_cfg_raw,
            "scopes",
            calibrated_model_cfg_raw,
            "fallback_guard_scopes",
            ["lane", "global"],
        )
        allowed_scopes = _scope_list_from_cfg(
            calibrated_model_cfg_raw,
            "allowed_scopes",
            bundle_entry_model,
            "allowed_scopes",
            ["variant", "lane", "global", "missing"],
        )
        calibrated_entry_model_cfg = {
            "enabled": bool(calibrated_model_enabled),
            "enforce_veto": bool(calibrated_model_cfg_raw.get("enforce_veto", True)),
            "allow_on_missing_stats": bool(
                calibrated_model_param_cfg.get(
                    "allow_on_missing_stats",
                    minimums_cfg.get("allow_on_missing_stats", True),
                )
            ),
            "min_variant_trades": int(
                max(
                    1,
                    safe_float(
                        calibrated_model_param_cfg.get(
                            "min_variant_trades",
                            minimums_cfg.get("min_variant_trades", 25),
                        ),
                        25,
                    ),
                )
            ),
            "min_lane_trades": int(
                max(
                    1,
                    safe_float(
                        calibrated_model_param_cfg.get(
                            "min_lane_trades",
                            minimums_cfg.get("min_lane_trades", 120),
                        ),
                        120,
                    ),
                )
            ),
            "selected_threshold": float(
                self._cfg_float(
                    calibrated_model_param_cfg,
                    "selected_threshold",
                    safe_float(bundle_entry_model.get("selected_threshold", 0.0), 0.0),
                )
                or 0.0
            ),
            "conservative_buffer": float(
                self._cfg_float(
                    calibrated_model_param_cfg,
                    "conservative_buffer",
                    safe_float(minimums_cfg.get("conservative_buffer", 0.035), 0.035),
                )
                or 0.0
            ),
            "use_bundle_model": bool(use_bundle_entry_model),
            "fallback_scope_guard_enabled": bool(
                fallback_guard_cfg_raw.get(
                    "enabled",
                    calibrated_model_cfg_raw.get("fallback_scope_guard_enabled", True),
                )
            ),
            "allowed_scopes": list(allowed_scopes),
            "fallback_guard_scopes": list(fallback_guard_scopes),
            "fallback_min_ev_lcb_points": float(
                self._cfg_float(
                    fallback_guard_cfg_raw,
                    "min_ev_lcb_points",
                    self._cfg_float(calibrated_model_cfg_raw, "fallback_min_ev_lcb_points", -0.05),
                )
                or 0.0
            ),
            "fallback_min_quality_lcb_score": float(
                self._cfg_float(
                    fallback_guard_cfg_raw,
                    "min_quality_lcb_score",
                    self._cfg_float(calibrated_model_cfg_raw, "fallback_min_quality_lcb_score", -0.02),
                )
                or 0.0
            ),
            "fallback_min_p_win_lcb": float(
                self._cfg_float(
                    fallback_guard_cfg_raw,
                    "min_p_win_lcb",
                    self._cfg_float(calibrated_model_cfg_raw, "fallback_min_p_win_lcb", 0.30),
                )
                or 0.0
            ),
            "fallback_min_worst_block_avg_pnl": float(
                self._cfg_float(
                    fallback_guard_cfg_raw,
                    "min_worst_block_avg_pnl",
                    self._cfg_float(
                        calibrated_model_cfg_raw,
                        "fallback_min_worst_block_avg_pnl",
                        -60.0,
                    ),
                )
                or 0.0
            ),
            "fallback_min_year_coverage": int(
                max(
                    0,
                    safe_float(
                        fallback_guard_cfg_raw.get(
                            "min_year_coverage",
                            calibrated_model_cfg_raw.get("fallback_min_year_coverage", 5),
                        ),
                        5,
                    ),
                )
            ),
            "fallback_min_variant_quality_prior": float(
                self._cfg_float(
                    fallback_guard_cfg_raw,
                    "min_variant_quality_prior",
                    self._cfg_float(
                        calibrated_model_cfg_raw,
                        "fallback_min_variant_quality_prior",
                        0.0,
                    ),
                )
                or 0.0
            ),
            "fallback_allow_global_scope": bool(
                fallback_guard_cfg_raw.get(
                    "allow_global_scope",
                    calibrated_model_cfg_raw.get("fallback_allow_global_scope", False),
                )
            ),
            "scope_threshold_offsets": {
                "variant": float(
                    self._cfg_float(
                        scope_offsets_override,
                        "variant",
                        safe_float(scope_offsets_cfg.get("variant", 0.0), 0.0),
                    )
                    or 0.0
                ),
                "lane": float(
                    self._cfg_float(
                        scope_offsets_override,
                        "lane",
                        safe_float(scope_offsets_cfg.get("lane", 0.06), 0.06),
                    )
                    or 0.0
                ),
                "global": float(
                    self._cfg_float(
                        scope_offsets_override,
                        "global",
                        safe_float(scope_offsets_cfg.get("global", 0.12), 0.12),
                    )
                    or 0.0
                ),
                "missing": float(
                    self._cfg_float(
                        scope_offsets_override,
                        "missing",
                        safe_float(scope_offsets_cfg.get("missing", 0.15), 0.15),
                    )
                    or 0.0
                ),
                "default": float(
                    self._cfg_float(
                        scope_offsets_override,
                        "default",
                        safe_float(scope_offsets_cfg.get("default", 0.0), 0.0),
                    )
                    or 0.0
                ),
            },
            "score_components": {
                "weight_quality_lcb": float(
                    self._cfg_float(
                        calibrated_model_param_cfg,
                        "weight_quality_lcb",
                        safe_float(score_components_cfg.get("weight_quality_lcb", 0.65), 0.65),
                    )
                    or 0.65
                ),
                "weight_route_confidence": float(
                    self._cfg_float(
                        calibrated_model_param_cfg,
                        "weight_route_confidence",
                        safe_float(score_components_cfg.get("weight_route_confidence", 0.20), 0.20),
                    )
                    or 0.20
                ),
                "weight_edge_points": float(
                    self._cfg_float(
                        calibrated_model_param_cfg,
                        "weight_edge_points",
                        safe_float(score_components_cfg.get("weight_edge_points", 0.10), 0.10),
                    )
                    or 0.10
                ),
                "weight_structural_score": float(
                    self._cfg_float(
                        calibrated_model_param_cfg,
                        "weight_structural_score",
                        safe_float(score_components_cfg.get("weight_structural_score", 0.05), 0.05),
                    )
                    or 0.05
                ),
                "weight_profit_factor_component": float(
                    self._cfg_float(
                        calibrated_model_param_cfg,
                        "weight_profit_factor_component",
                        safe_float(score_components_cfg.get("weight_profit_factor_component", 0.0), 0.0),
                    )
                    or 0.0
                ),
                "weight_year_coverage_component": float(
                    self._cfg_float(
                        calibrated_model_param_cfg,
                        "weight_year_coverage_component",
                        safe_float(score_components_cfg.get("weight_year_coverage_component", 0.0), 0.0),
                    )
                    or 0.0
                ),
                "weight_loss_share_penalty": float(
                    self._cfg_float(
                        calibrated_model_param_cfg,
                        "weight_loss_share_penalty",
                        safe_float(score_components_cfg.get("weight_loss_share_penalty", 0.12), 0.12),
                    )
                    or 0.0
                ),
                "weight_stop_like_share_penalty": float(
                    self._cfg_float(
                        calibrated_model_param_cfg,
                        "weight_stop_like_share_penalty",
                        safe_float(score_components_cfg.get("weight_stop_like_share_penalty", 0.08), 0.08),
                    )
                    or 0.0
                ),
                "weight_drawdown_penalty": float(
                    self._cfg_float(
                        calibrated_model_param_cfg,
                        "weight_drawdown_penalty",
                        safe_float(score_components_cfg.get("weight_drawdown_penalty", 0.06), 0.06),
                    )
                    or 0.0
                ),
                "weight_worst_block_penalty": float(
                    self._cfg_float(
                        calibrated_model_param_cfg,
                        "weight_worst_block_penalty",
                        safe_float(score_components_cfg.get("weight_worst_block_penalty", 0.08), 0.08),
                    )
                    or 0.0
                ),
                "weight_shape_penalty_component": float(
                    self._cfg_float(
                        calibrated_model_param_cfg,
                        "weight_shape_penalty_component",
                        safe_float(score_components_cfg.get("weight_shape_penalty_component", 0.0), 0.0),
                    )
                    or 0.0
                ),
                "route_confidence_center": float(
                    self._cfg_float(
                        calibrated_model_param_cfg,
                        "route_confidence_center",
                        safe_float(score_components_cfg.get("route_confidence_center", 0.05), 0.05),
                    )
                    or 0.05
                ),
                "edge_scale_points": float(
                    max(
                        1e-9,
                        self._cfg_float(
                            calibrated_model_param_cfg,
                            "edge_scale_points",
                            safe_float(score_components_cfg.get("edge_scale_points", 0.40), 0.40),
                        )
                        or 0.40,
                    )
                ),
                "structural_scale": float(
                    max(
                        1e-9,
                        self._cfg_float(
                            calibrated_model_param_cfg,
                            "structural_scale",
                            safe_float(score_components_cfg.get("structural_scale", 0.80), 0.80),
                        )
                        or 0.80,
                    )
                ),
                "profit_factor_center": float(
                    self._cfg_float(
                        calibrated_model_param_cfg,
                        "profit_factor_center",
                        safe_float(score_components_cfg.get("profit_factor_center", 1.10), 1.10),
                    )
                    or 1.10
                ),
                "profit_factor_scale": float(
                    max(
                        1e-9,
                        self._cfg_float(
                            calibrated_model_param_cfg,
                            "profit_factor_scale",
                            safe_float(score_components_cfg.get("profit_factor_scale", 0.35), 0.35),
                        )
                        or 0.35,
                    )
                ),
                "year_coverage_full_years": float(
                    max(
                        1.0,
                        self._cfg_float(
                            calibrated_model_param_cfg,
                            "year_coverage_full_years",
                            safe_float(score_components_cfg.get("year_coverage_full_years", 8.0), 8.0),
                        )
                        or 8.0,
                    )
                ),
                "loss_share_center": float(
                    self._cfg_float(
                        calibrated_model_param_cfg,
                        "loss_share_center",
                        safe_float(score_components_cfg.get("loss_share_center", 0.52), 0.52),
                    )
                    or 0.52
                ),
                "loss_share_scale": float(
                    max(
                        1e-9,
                        self._cfg_float(
                            calibrated_model_param_cfg,
                            "loss_share_scale",
                            safe_float(score_components_cfg.get("loss_share_scale", 0.22), 0.22),
                        )
                        or 0.22,
                    )
                ),
                "stop_like_share_center": float(
                    self._cfg_float(
                        calibrated_model_param_cfg,
                        "stop_like_share_center",
                        safe_float(score_components_cfg.get("stop_like_share_center", 0.62), 0.62),
                    )
                    or 0.62
                ),
                "stop_like_share_scale": float(
                    max(
                        1e-9,
                        self._cfg_float(
                            calibrated_model_param_cfg,
                            "stop_like_share_scale",
                            safe_float(score_components_cfg.get("stop_like_share_scale", 0.25), 0.25),
                        )
                        or 0.25,
                    )
                ),
                "drawdown_scale": float(
                    max(
                        1e-9,
                        self._cfg_float(
                            calibrated_model_param_cfg,
                            "drawdown_scale",
                            safe_float(score_components_cfg.get("drawdown_scale", 6.0), 6.0),
                        )
                        or 6.0,
                    )
                ),
                "shape_penalty_scale": float(
                    max(
                        1e-9,
                        self._cfg_float(
                            calibrated_model_param_cfg,
                            "shape_penalty_scale",
                            safe_float(score_components_cfg.get("shape_penalty_scale", 1.0), 1.0),
                        )
                        or 1.0,
                    )
                ),
                "shape_penalty_cap": float(
                    max(
                        0.0,
                        self._cfg_float(
                            calibrated_model_param_cfg,
                            "shape_penalty_cap",
                            safe_float(score_components_cfg.get("shape_penalty_cap", 2.0), 2.0),
                        )
                        or 0.0,
                    )
                ),
                "worst_block_scale_points": float(
                    max(
                        1e-9,
                        self._cfg_float(
                            calibrated_model_param_cfg,
                            "worst_block_scale_points",
                            safe_float(score_components_cfg.get("worst_block_scale_points", 3.0), 3.0),
                        )
                        or 3.0,
                    )
                ),
            },
            "selected_threshold_source": str(
                calibrated_model_param_cfg.get(
                    "selected_threshold_source",
                    bundle_entry_model.get("selected_threshold_source", ""),
                )
                or ""
            ),
            "variant_stats": (
                dict(bundle_entry_model.get("variant_stats", {}))
                if isinstance(bundle_entry_model.get("variant_stats"), dict)
                else {}
            ),
            "lane_stats": (
                dict(bundle_entry_model.get("lane_stats", {}))
                if isinstance(bundle_entry_model.get("lane_stats"), dict)
                else {}
            ),
            "global_stats": (
                dict(bundle_entry_model.get("global_stats", {}))
                if isinstance(bundle_entry_model.get("global_stats"), dict)
                else {}
            ),
            "fit_windows": (
                dict(bundle_entry_model.get("fit_windows", {}))
                if isinstance(bundle_entry_model.get("fit_windows"), dict)
                else {}
            ),
            "shape_penalty_model": {
                "enabled": bool(
                    shape_model_override.get(
                        "enabled",
                        shape_model_cfg.get("enabled", False),
                    )
                ),
                "scope_mode": str(
                    shape_model_override.get(
                        "scope_mode",
                        shape_model_cfg.get("scope_mode", "lane_timeframe"),
                    )
                    or "lane_timeframe"
                ).strip().lower(),
                "max_rules_per_row": int(
                    max(
                        0,
                        safe_float(
                            shape_model_override.get(
                                "max_rules_per_row",
                                shape_model_cfg.get("max_rules_per_row", 0),
                            ),
                            0,
                        ),
                    )
                ),
                "rules": (
                    list(shape_model_override.get("rules", []))
                    if isinstance(shape_model_override.get("rules"), list)
                    else (
                        list(shape_model_cfg.get("rules", []))
                        if isinstance(shape_model_cfg.get("rules"), list)
                        else []
                    )
                ),
            },
        }

        return {
            "enabled": bool(policy_enabled or calibrated_model_enabled),
            "base_policy_enabled": bool(policy_enabled),
            "calibrated_entry_model_enabled": bool(calibrated_model_enabled),
            "source": (
                "execution_policy"
                if bool(policy_cfg_raw)
                else ("legacy_execution_filters" if bool(legacy_execution_filters_cfg) else "defaults")
            ),
            "enforce_veto": bool(policy_cfg_raw.get("enforce_veto", True)),
            "soft_tier_on_reject": str(
                policy_cfg_raw.get("soft_tier_on_reject", "conservative") or "conservative"
            ).strip().lower(),
            "hard_limits": {
                "min_route_confidence": _pick_float("min_route_confidence", hard_key="min_route_confidence"),
                "min_edge_points": _pick_float("min_edge_points", hard_key="min_edge_points"),
                "min_structural_score": _pick_float("min_structural_score", hard_key="min_structural_score"),
                "min_lane_score": _pick_float("min_lane_score", hard_key="min_lane_score"),
                "min_variant_quality_prior": _pick_float("min_variant_quality_prior", hard_key="min_variant_quality_prior"),
                "max_loss_share": _pick_float("max_loss_share", hard_key="max_loss_share"),
                "max_stop_like_share": _pick_float("max_stop_like_share", hard_key="max_stop_like_share"),
            },
            "quality": {
                "reject_quality_score_below": self._cfg_float(
                    policy_cfg_raw,
                    "reject_quality_score_below",
                    0.30,
                ),
                "conservative_quality_score_below": self._cfg_float(
                    policy_cfg_raw,
                    "conservative_quality_score_below",
                    0.48,
                ),
                "weights": {k: float(v or 0.0) for k, v in weights.items()},
                "weight_total": float(weight_total),
                "ranges": {
                    "route_confidence": _range("route_confidence", 0.00, 0.20),
                    "edge_points": _range("edge_points", -0.10, 0.70),
                    "lane_score": _range("lane_score", -0.10, 0.70),
                    "structural_score": _range("structural_score", -0.50, 1.50),
                    "variant_quality_prior": _range("variant_quality_prior", 0.00, 1.00),
                    "loss_share": _range("loss_share", 0.30, 0.85),
                    "stop_like_share": _range("stop_like_share", 0.30, 0.90),
                },
            },
            "calibrated_entry_model": calibrated_entry_model_cfg,
        }

    def _execution_policy_feature_vector(
        self,
        *,
        chosen_entry: Dict[str, Any],
        route_confidence: float,
    ) -> Dict[str, float]:
        edge_points = safe_float(chosen_entry.get("edge_points", 0.0), 0.0)
        structural_score = safe_float(chosen_entry.get("structural_score", 0.0), 0.0)
        lane_score = safe_float(
            chosen_entry.get("lane_score", chosen_entry.get("selection_score", edge_points)),
            edge_points,
        )
        lane_inputs = (
            chosen_entry.get("lane_score_inputs", {})
            if isinstance(chosen_entry.get("lane_score_inputs"), dict)
            else {}
        )
        variant_prior = safe_float(lane_inputs.get("variant_quality_prior", 0.0), 0.0)
        loss_share = safe_float(chosen_entry.get("loss_share", 0.0), 0.0)
        stop_like_share = safe_float(chosen_entry.get("stop_like_share", 0.0), 0.0)
        return {
            "route_confidence": float(route_confidence),
            "edge_points": float(edge_points),
            "structural_score": float(structural_score),
            "lane_score": float(lane_score),
            "variant_quality_prior": float(variant_prior),
            "loss_share": float(loss_share),
            "stop_like_share": float(stop_like_share),
        }

    @staticmethod
    def _entry_model_variant_id(chosen_entry: Dict[str, Any]) -> str:
        cand = chosen_entry.get("cand", {}) if isinstance(chosen_entry.get("cand"), dict) else {}
        return str(
            chosen_entry.get("selected_variant_id")
            or chosen_entry.get("variant_id")
            or chosen_entry.get("cand_id")
            or cand.get("strategy_id")
            or ""
        ).strip()

    @staticmethod
    def _entry_model_lane(chosen_entry: Dict[str, Any]) -> str:
        lane = str(chosen_entry.get("lane", "") or "").strip()
        if lane:
            return lane
        cand = chosen_entry.get("cand", {}) if isinstance(chosen_entry.get("cand"), dict) else {}
        return strategy_type_to_lane(cand.get("strategy_type", ""))

    @staticmethod
    def _entry_model_timeframe(chosen_entry: Dict[str, Any]) -> str:
        timeframe = str(chosen_entry.get("timeframe", "") or "").strip()
        if timeframe:
            return timeframe
        cand = chosen_entry.get("cand", {}) if isinstance(chosen_entry.get("cand"), dict) else {}
        return str(cand.get("timeframe", "") or "").strip()

    @staticmethod
    def _entry_model_derive_session_substate(*, session_name: str, hour_value: Any) -> str:
        session_text = str(session_name or "").strip()
        if not session_text or "-" not in session_text:
            return ""
        parts = [part.strip() for part in session_text.split("-", 1)]
        if len(parts) != 2:
            return ""
        try:
            start_hour = int(float(parts[0]))
            hour_et = int(round(float(hour_value)))
        except Exception:
            return ""
        rel_hour = (hour_et - start_hour) % 24
        if rel_hour < 1:
            return "open"
        if rel_hour < 2:
            return "mid"
        return "late"

    def _entry_model_context_bucket_key(
        self,
        *,
        chosen_entry: Dict[str, Any],
        fields: List[str],
    ) -> str:
        if not fields:
            return "__global__"
        cand = chosen_entry.get("cand", {}) if isinstance(chosen_entry.get("cand"), dict) else {}
        parts: List[str] = []
        for field in fields:
            key = str(field or "").strip()
            if not key:
                continue
            value = chosen_entry.get(key, cand.get(key, ""))
            value_norm = self._normalize_book_gate_value(value)
            if not value_norm:
                return ""
            parts.append(f"{key}={value_norm}")
        return "|".join(parts)

    @staticmethod
    def _entry_model_numeric_value(chosen_entry: Dict[str, Any], key: str) -> float:
        cand = chosen_entry.get("cand", {}) if isinstance(chosen_entry.get("cand"), dict) else {}
        return safe_float(chosen_entry.get(key, cand.get(key, float("nan"))), float("nan"))

    @staticmethod
    def _entry_model_bucket_close_pos(value: float) -> str:
        if not math.isfinite(value):
            return ""
        if value <= 0.15:
            return "very_low"
        if value <= 0.35:
            return "low"
        if value <= 0.65:
            return "mid"
        if value <= 0.85:
            return "high"
        return "very_high"

    @staticmethod
    def _entry_model_bucket_balance(
        value: float,
        *,
        strong_neg: float,
        neg: float,
        pos: float,
        strong_pos: float,
    ) -> str:
        if not math.isfinite(value):
            return ""
        if value <= strong_neg:
            return "strong_neg"
        if value <= neg:
            return "neg"
        if value < pos:
            return "balanced"
        if value < strong_pos:
            return "pos"
        return "strong_pos"

    @staticmethod
    def _entry_model_bucket_body_ratio(value: float) -> str:
        if not math.isfinite(value):
            return ""
        if value <= 0.20:
            return "small"
        if value <= 0.45:
            return "medium"
        return "large"

    @staticmethod
    def _entry_model_bucket_vol_ratio(value: float) -> str:
        if not math.isfinite(value):
            return ""
        if value <= 0.75:
            return "low"
        if value <= 1.10:
            return "mid"
        if value <= 1.50:
            return "high"
        return "extreme"

    @staticmethod
    def _entry_model_bucket_range_ratio(value: float) -> str:
        if not math.isfinite(value):
            return ""
        if value <= 1.20:
            return "compressed"
        if value <= 2.10:
            return "normal"
        return "expanded"

    @staticmethod
    def _entry_model_bucket_down3(value: float) -> str:
        if not math.isfinite(value):
            return ""
        raw = int(round(value))
        if raw <= 1:
            return "0_1"
        if raw == 2:
            return "2"
        return "3"

    @staticmethod
    def _entry_model_directionless_strategy_style(value: Any) -> str:
        text = str(value or "").strip().lower()
        if not text:
            return ""
        for prefix in ("long_", "short_", "buy_", "sell_"):
            if text.startswith(prefix):
                text = text[len(prefix) :]
                break
        if "_" in text:
            parts = [part for part in text.split("_") if part]
            if len(parts) >= 2 and parts[0] in {"long", "short", "buy", "sell"}:
                text = "_".join(parts[1:])
        return str(text).strip("_")

    def _entry_model_short_term_context_values(
        self,
        *,
        chosen_entry: Dict[str, Any],
    ) -> Dict[str, str]:
        cand = chosen_entry.get("cand", {}) if isinstance(chosen_entry.get("cand"), dict) else {}
        lane = self._entry_model_lane(chosen_entry)
        session_name = str(chosen_entry.get("session", cand.get("session", "")) or "").strip().lower()
        timeframe = self._entry_model_timeframe(chosen_entry).lower()
        strategy_type = str(
            chosen_entry.get("strategy_type", cand.get("strategy_type", "")) or ""
        ).strip().lower()
        side_considered = str(chosen_entry.get("side_considered", cand.get("side_considered", "")) or "").strip().lower()
        if side_considered not in {"long", "short"}:
            lane_side = str(lane_to_side(lane) or "").strip().lower()
            if lane_side in {"long", "short"}:
                side_considered = lane_side
        hour_value = safe_float(
            chosen_entry.get("ctx_hour_et", cand.get("ctx_hour_et", float("nan"))),
            float("nan"),
        )
        session_substate = str(
            chosen_entry.get("ctx_session_substate", cand.get("ctx_session_substate", "")) or ""
        ).strip().lower()
        if (not session_substate) and session_name and math.isfinite(hour_value):
            session_substate = self._entry_model_derive_session_substate(
                session_name=session_name,
                hour_value=hour_value,
            )

        upper_wick = self._entry_model_numeric_value(chosen_entry, "de3_entry_upper_wick_ratio")
        lower_wick = self._entry_model_numeric_value(chosen_entry, "de3_entry_lower_wick_ratio")
        close_pos = self._entry_model_numeric_value(chosen_entry, "de3_entry_close_pos1")
        body_ratio = self._entry_model_numeric_value(chosen_entry, "de3_entry_body1_ratio")
        ret1_atr = self._entry_model_numeric_value(chosen_entry, "de3_entry_ret1_atr")
        vol_rel20 = self._entry_model_numeric_value(chosen_entry, "de3_entry_vol1_rel20")
        range10_atr = self._entry_model_numeric_value(chosen_entry, "de3_entry_range10_atr")
        dist_low = self._entry_model_numeric_value(chosen_entry, "de3_entry_dist_low5_atr")
        dist_high = self._entry_model_numeric_value(chosen_entry, "de3_entry_dist_high5_atr")
        down3 = self._entry_model_numeric_value(chosen_entry, "de3_entry_down3")
        wick_bias = lower_wick - upper_wick
        location_bias = dist_high - dist_low
        ret_bucket = self._entry_model_bucket_balance(
            ret1_atr,
            strong_neg=-0.45,
            neg=-0.12,
            pos=0.12,
            strong_pos=0.45,
        )
        wick_bucket = self._entry_model_bucket_balance(
            wick_bias,
            strong_neg=-0.28,
            neg=-0.10,
            pos=0.10,
            strong_pos=0.28,
        )
        side_pattern = str(
            chosen_entry.get("side_pattern", cand.get("side_pattern", "")) or ""
        ).strip().lower()
        return {
            "session": str(session_name),
            "timeframe": str(timeframe),
            "strategy_type": str(strategy_type),
            "strategy_style": self._entry_model_directionless_strategy_style(strategy_type),
            "side_considered": str(side_considered),
            "side_pattern": str(side_pattern),
            "sub_strategy": str(
                chosen_entry.get("sub_strategy", cand.get("strategy_id", cand.get("sub_strategy", ""))) or ""
            ).strip().lower(),
            "ctx_session_substate": str(session_substate),
            "ctx_hour_bucket": (
                str(int(round(hour_value)))
                if math.isfinite(hour_value)
                else ""
            ),
            "st_close_bucket": self._entry_model_bucket_close_pos(close_pos),
            "st_wick_bias_bucket": wick_bucket,
            "st_body_bucket": self._entry_model_bucket_body_ratio(body_ratio),
            "st_ret_bucket": ret_bucket,
            "st_vol_bucket": self._entry_model_bucket_vol_ratio(vol_rel20),
            "st_range_bucket": self._entry_model_bucket_range_ratio(range10_atr),
            "st_location_bucket": self._entry_model_bucket_balance(
                location_bias,
                strong_neg=-1.20,
                neg=-0.35,
                pos=0.35,
                strong_pos=1.20,
            ),
            "st_down3_bucket": self._entry_model_bucket_down3(down3),
            "st_pressure_bucket": (
                str(ret_bucket).strip() + "|" + str(wick_bucket).strip()
                if ret_bucket and wick_bucket
                else ""
            ),
        }

    def _entry_model_short_term_bucket_key(
        self,
        *,
        chosen_entry: Dict[str, Any],
        fields: List[str],
    ) -> str:
        if not fields:
            return ""
        context_values = self._entry_model_short_term_context_values(chosen_entry=chosen_entry)
        parts: List[str] = []
        for field in fields:
            key = str(field or "").strip()
            if not key:
                continue
            value_norm = self._normalize_book_gate_value(context_values.get(key, ""))
            if not value_norm:
                return ""
            parts.append(value_norm)
        return "|".join(parts)

    @classmethod
    def _conflict_side_bucket_key(
        cls,
        *,
        context_row: Dict[str, Any],
        fields: List[str],
    ) -> str:
        if not fields:
            return ""
        parts: List[str] = []
        for field in fields:
            key = str(field or "").strip()
            if not key:
                continue
            value_norm = cls._normalize_book_gate_value(context_row.get(key, ""))
            if not value_norm:
                return ""
            parts.append(value_norm)
        return "|".join(parts)

    @staticmethod
    def _conflict_side_direction_bucket(
        value: float,
        *,
        neg_hi: float,
        neg_lo: float,
        pos_lo: float,
        pos_hi: float,
    ) -> str:
        if not math.isfinite(value):
            return ""
        if value <= neg_hi:
            return "strong_short"
        if value <= neg_lo:
            return "short"
        if value < pos_lo:
            return "balanced"
        if value < pos_hi:
            return "long"
        return "strong_long"

    @staticmethod
    def _conflict_side_top_row_for_side(
        *,
        candidate_rows: List[Dict[str, Any]],
        side_name: str,
    ) -> Optional[Dict[str, Any]]:
        side_norm = str(side_name or "").strip().lower()
        pool = [
            dict(row)
            for row in candidate_rows
            if isinstance(row, dict) and str(row.get("side_considered", "") or "").strip().lower() == side_norm
        ]
        if not pool:
            return None

        def _sort_key(row: Dict[str, Any]) -> Tuple[int, float, float, float]:
            rank_value = safe_int(
                row.get("candidate_rank_before_adjustments", row.get("rank", 0)),
                0,
            )
            rank_key = int(rank_value) if int(rank_value) > 0 else 999999
            final_score = safe_float(
                row.get(
                    "final_score",
                    row.get(
                        "decision_policy_score",
                        row.get("selection_score", row.get("lane_score", 0.0)),
                    ),
                ),
                0.0,
            )
            edge_points = safe_float(
                row.get("edge_points", row.get("runtime_rank_score", 0.0)),
                0.0,
            )
            structural_score = safe_float(row.get("structural_score", 0.0), 0.0)
            return (
                int(rank_key),
                float(-final_score),
                float(-edge_points),
                float(-structural_score),
            )

        pool.sort(key=_sort_key)
        return dict(pool[0]) if pool else None

    def _evaluate_conflict_side_model(
        self,
        *,
        candidate_rows: List[Dict[str, Any]],
        default_session: str,
        current_time: Optional[Any],
    ) -> Dict[str, Any]:
        model_cfg = (
            dict(self.conflict_side_model_cfg)
            if isinstance(self.conflict_side_model_cfg, dict)
            else {}
        )
        if not bool(self.conflict_side_model_enabled) or not model_cfg:
            return {
                "enabled": False,
                "match_count": 0,
                "matched_scopes": [],
                "predicted_side": "",
            }
        long_row = self._conflict_side_top_row_for_side(candidate_rows=candidate_rows, side_name="long")
        short_row = self._conflict_side_top_row_for_side(candidate_rows=candidate_rows, side_name="short")
        if not isinstance(long_row, dict) or not isinstance(short_row, dict):
            return {
                "enabled": True,
                "match_count": 0,
                "matched_scopes": [],
                "predicted_side": "",
            }
        current_ts = _to_timestamp(current_time)
        fallback_hour = current_ts.astimezone(NY_TZ).hour if isinstance(current_ts, dt.datetime) else None
        session_name = str(
            long_row.get("session") or short_row.get("session") or default_session or ""
        ).strip()
        hour_value = long_row.get("ctx_hour_et", short_row.get("ctx_hour_et", fallback_hour))
        session_substate = str(
            long_row.get("ctx_session_substate")
            or short_row.get("ctx_session_substate")
            or self._entry_model_derive_session_substate(
                session_name=session_name,
                hour_value=hour_value,
            )
            or ""
        ).strip().lower()
        ctx_hour_bucket = (
            str(int(round(float(hour_value))))
            if self._optional_float(hour_value) is not None
            else ""
        )
        long_rank = safe_float(
            long_row.get("candidate_rank_before_adjustments", long_row.get("rank", 0)),
            0.0,
        )
        short_rank = safe_float(
            short_row.get("candidate_rank_before_adjustments", short_row.get("rank", 0)),
            0.0,
        )
        long_final_score = safe_float(
            long_row.get(
                "final_score",
                long_row.get("decision_policy_score", long_row.get("selection_score", long_row.get("lane_score", 0.0))),
            ),
            0.0,
        )
        short_final_score = safe_float(
            short_row.get(
                "final_score",
                short_row.get("decision_policy_score", short_row.get("selection_score", short_row.get("lane_score", 0.0))),
            ),
            0.0,
        )
        long_edge_points = safe_float(long_row.get("edge_points", long_row.get("runtime_rank_score", 0.0)), 0.0)
        short_edge_points = safe_float(short_row.get("edge_points", short_row.get("runtime_rank_score", 0.0)), 0.0)
        long_structural_score = safe_float(long_row.get("structural_score", 0.0), 0.0)
        short_structural_score = safe_float(short_row.get("structural_score", 0.0), 0.0)
        rank_adv = float(short_rank - long_rank)
        score_adv = float(long_final_score - short_final_score)
        edge_adv = float(long_edge_points - short_edge_points)
        struct_adv = float(long_structural_score - short_structural_score)
        context_row = {
            "session": str(session_name),
            "ctx_session_substate": str(session_substate),
            "ctx_hour_bucket": str(ctx_hour_bucket),
            "long_strategy_type": str(long_row.get("strategy_type", "") or ""),
            "short_strategy_type": str(short_row.get("strategy_type", "") or ""),
            "long_timeframe": str(long_row.get("timeframe", "") or ""),
            "short_timeframe": str(short_row.get("timeframe", "") or ""),
            "rank_adv_bucket": self._conflict_side_direction_bucket(
                rank_adv,
                neg_hi=-1.5,
                neg_lo=-0.5,
                pos_lo=0.5,
                pos_hi=1.5,
            ),
            "score_adv_bucket": self._conflict_side_direction_bucket(
                score_adv,
                neg_hi=-1.0,
                neg_lo=-0.25,
                pos_lo=0.25,
                pos_hi=1.0,
            ),
            "edge_adv_bucket": self._conflict_side_direction_bucket(
                edge_adv,
                neg_hi=-1.0,
                neg_lo=-0.25,
                pos_lo=0.25,
                pos_hi=1.0,
            ),
            "struct_adv_bucket": self._conflict_side_direction_bucket(
                struct_adv,
                neg_hi=-1.0,
                neg_lo=-0.25,
                pos_lo=0.25,
                pos_hi=1.0,
            ),
        }
        context_row["score_edge_combo"] = (
            f"{context_row.get('score_adv_bucket', '')}|{context_row.get('edge_adv_bucket', '')}"
        )
        context_row["rank_score_combo"] = (
            f"{context_row.get('rank_adv_bucket', '')}|{context_row.get('score_adv_bucket', '')}"
        )
        scopes = model_cfg.get("scopes", []) if isinstance(model_cfg.get("scopes"), list) else []
        max_abs_score = max(0.10, safe_float(model_cfg.get("max_abs_score", 1.25), 1.25))
        max_scopes_per_row = max(1, int(safe_float(model_cfg.get("max_scopes_per_row", 2), 2)))
        matches: List[Dict[str, Any]] = []
        for raw_scope in scopes:
            scope = raw_scope if isinstance(raw_scope, dict) else {}
            fields = (
                [str(v).strip() for v in scope.get("fields", []) if str(v).strip()]
                if isinstance(scope.get("fields"), list)
                else []
            )
            if not fields:
                continue
            bucket_key = self._conflict_side_bucket_key(context_row=context_row, fields=fields)
            if not bucket_key:
                continue
            buckets = scope.get("buckets", {}) if isinstance(scope.get("buckets"), dict) else {}
            bucket = buckets.get(bucket_key, {}) if isinstance(buckets.get(bucket_key, {}), dict) else {}
            if not bucket:
                continue
            weight = max(0.0, safe_float(scope.get("weight", 1.0), 1.0))
            if weight <= 0.0:
                continue
            score = safe_float(bucket.get("score", 0.0), 0.0)
            matches.append(
                {
                    "scope_name": str(scope.get("name", "") or ""),
                    "weight": float(weight),
                    "score": float(score),
                    "weighted_abs": float(abs(score) * weight),
                    "long_mean": float(safe_float(bucket.get("long_mean_pnl_points", 0.0), 0.0)),
                    "short_mean": float(safe_float(bucket.get("short_mean_pnl_points", 0.0), 0.0)),
                    "long_positive_rate": float(safe_float(bucket.get("long_positive_rate", 0.0), 0.0)),
                    "short_positive_rate": float(safe_float(bucket.get("short_positive_rate", 0.0), 0.0)),
                }
            )
        if not matches:
            return {
                "enabled": True,
                "match_count": 0,
                "matched_scopes": [],
                "predicted_side": "",
                "top_long_variant_id": str(long_row.get("variant_id", "") or ""),
                "top_short_variant_id": str(short_row.get("variant_id", "") or ""),
            }
        matches.sort(
            key=lambda item: (
                safe_float(item.get("weighted_abs", 0.0), 0.0),
                abs(safe_float(item.get("score", 0.0), 0.0)),
            ),
            reverse=True,
        )
        matches = matches[:max_scopes_per_row]
        total_weight = sum(max(0.0, safe_float(item.get("weight", 0.0), 0.0)) for item in matches)
        score = safe_div(
            sum(
                safe_float(item.get("score", 0.0), 0.0)
                * max(0.0, safe_float(item.get("weight", 0.0), 0.0))
                for item in matches
            ),
            total_weight,
            0.0,
        )
        long_mean = safe_div(
            sum(
                safe_float(item.get("long_mean", 0.0), 0.0)
                * max(0.0, safe_float(item.get("weight", 0.0), 0.0))
                for item in matches
            ),
            total_weight,
            0.0,
        )
        short_mean = safe_div(
            sum(
                safe_float(item.get("short_mean", 0.0), 0.0)
                * max(0.0, safe_float(item.get("weight", 0.0), 0.0))
                for item in matches
            ),
            total_weight,
            0.0,
        )
        long_positive_rate = safe_div(
            sum(
                safe_float(item.get("long_positive_rate", 0.0), 0.0)
                * max(0.0, safe_float(item.get("weight", 0.0), 0.0))
                for item in matches
            ),
            total_weight,
            0.0,
        )
        short_positive_rate = safe_div(
            sum(
                safe_float(item.get("short_positive_rate", 0.0), 0.0)
                * max(0.0, safe_float(item.get("weight", 0.0), 0.0))
                for item in matches
            ),
            total_weight,
            0.0,
        )
        positive_match_count = sum(1 for item in matches if safe_float(item.get("score", 0.0), 0.0) > 0.0)
        negative_match_count = sum(1 for item in matches if safe_float(item.get("score", 0.0), 0.0) < 0.0)
        consensus_ratio = (
            float(max(positive_match_count, negative_match_count) / max(1, len(matches)))
            if matches
            else 0.0
        )
        score = float(clip(score, -max_abs_score, max_abs_score))
        threshold = max(0.0, safe_float(model_cfg.get("selected_threshold", 0.0), 0.0))
        min_side_mean = safe_float(model_cfg.get("min_side_mean_points", 0.25), 0.25)
        min_advantage_gap = max(0.0, safe_float(model_cfg.get("min_advantage_gap_points", 0.0), 0.0))
        min_match_count = max(1, int(safe_float(model_cfg.get("min_match_count_to_override", 1), 1)))
        min_consensus_ratio = safe_float(model_cfg.get("min_consensus_ratio", 0.0), 0.0)
        allow_no_trade = bool(model_cfg.get("allow_no_trade", False))
        no_trade_max_side_mean = safe_float(model_cfg.get("no_trade_max_side_mean_points", 0.0), 0.0)
        no_trade_max_positive_rate = safe_float(model_cfg.get("no_trade_max_positive_rate", 0.0), 0.0)
        predicted_side = ""
        if (
            allow_no_trade
            and abs(score) >= threshold
            and max(long_mean, short_mean) <= no_trade_max_side_mean
            and max(long_positive_rate, short_positive_rate) <= no_trade_max_positive_rate
            and int(len(matches)) >= min_match_count
            and consensus_ratio >= min_consensus_ratio
        ):
            predicted_side = "no_trade"
        elif (
            int(len(matches)) >= min_match_count
            and consensus_ratio >= min_consensus_ratio
            and score >= threshold
            and long_mean >= min_side_mean
            and (long_mean - short_mean) >= min_advantage_gap
        ):
            predicted_side = "long"
        elif (
            int(len(matches)) >= min_match_count
            and consensus_ratio >= min_consensus_ratio
            and score <= -threshold
            and short_mean >= min_side_mean
            and (short_mean - long_mean) >= min_advantage_gap
        ):
            predicted_side = "short"
        return {
            "enabled": True,
            "score": float(score),
            "threshold": float(threshold),
            "match_count": int(len(matches)),
            "positive_match_count": int(positive_match_count),
            "negative_match_count": int(negative_match_count),
            "consensus_ratio": float(consensus_ratio),
            "matched_scopes": [
                str(item.get("scope_name", "") or "")
                for item in matches
                if str(item.get("scope_name", "") or "")
            ],
            "predicted_side": str(predicted_side),
            "long_mean": float(long_mean),
            "short_mean": float(short_mean),
            "long_positive_rate": float(long_positive_rate),
            "short_positive_rate": float(short_positive_rate),
            "context_row": dict(context_row),
            "top_long_variant_id": str(long_row.get("variant_id", "") or ""),
            "top_short_variant_id": str(short_row.get("variant_id", "") or ""),
        }

    def _evaluate_decision_side_model(
        self,
        *,
        candidate_rows: List[Dict[str, Any]],
        default_session: str,
        current_time: Optional[Any],
        model_cfg: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        model_cfg = dict(model_cfg) if isinstance(model_cfg, dict) else {}
        if not bool(model_cfg.get("enabled", False)) or not model_cfg:
            return {"enabled": False, "match_count": 0, "matched_scopes": [], "predicted_action": ""}
        long_row = self._conflict_side_top_row_for_side(candidate_rows=candidate_rows, side_name="long")
        short_row = self._conflict_side_top_row_for_side(candidate_rows=candidate_rows, side_name="short")
        if not isinstance(long_row, dict) and not isinstance(short_row, dict):
            return {"enabled": True, "match_count": 0, "matched_scopes": [], "predicted_action": ""}
        base_row = dict(long_row) if isinstance(long_row, dict) else dict(short_row)
        context_row = self._entry_model_short_term_context_values(chosen_entry=base_row)
        current_ts = _to_timestamp(current_time)
        fallback_hour = current_ts.astimezone(NY_TZ).hour if isinstance(current_ts, dt.datetime) else None
        if not str(context_row.get("session", "") or "").strip():
            context_row["session"] = str(base_row.get("session") or default_session or "").strip().lower()
        if not str(context_row.get("ctx_session_substate", "") or "").strip():
            hour_text = str(context_row.get("ctx_hour_bucket", "") or "").strip()
            hour_value = safe_float(hour_text, float("nan"))
            if (not math.isfinite(hour_value)) and fallback_hour is not None:
                hour_value = float(fallback_hour)
                context_row["ctx_hour_bucket"] = str(int(fallback_hour))
            if str(context_row.get("session", "") or "").strip() and math.isfinite(hour_value):
                context_row["ctx_session_substate"] = self._entry_model_derive_session_substate(
                    session_name=str(context_row.get("session", "") or ""),
                    hour_value=hour_value,
                )
        side_pattern = (
            "both"
            if isinstance(long_row, dict) and isinstance(short_row, dict)
            else ("long_only" if isinstance(long_row, dict) else "short_only")
        )
        allowed_side_patterns = (
            [
                str(v).strip().lower()
                for v in model_cfg.get("apply_side_patterns", [])
                if str(v).strip()
            ]
            if isinstance(model_cfg.get("apply_side_patterns"), list)
            else []
        )
        if allowed_side_patterns and str(side_pattern).strip().lower() not in allowed_side_patterns:
            return {
                "enabled": True,
                "match_count": 0,
                "matched_scopes": [],
                "predicted_action": "",
                "side_pattern": str(side_pattern),
            }
        context_row["side_pattern"] = str(side_pattern)
        current_session = str(context_row.get("session", "") or "").strip().lower()
        allowed_sessions = (
            [
                str(v).strip().lower()
                for v in model_cfg.get("apply_sessions", [])
                if str(v).strip()
            ]
            if isinstance(model_cfg.get("apply_sessions"), list)
            else []
        )
        if allowed_sessions and current_session not in allowed_sessions:
            return {
                "enabled": True,
                "match_count": 0,
                "matched_scopes": [],
                "predicted_action": "",
                "side_pattern": str(side_pattern),
            }
        blocked_sessions = (
            [
                str(v).strip().lower()
                for v in model_cfg.get("exclude_sessions", [])
                if str(v).strip()
            ]
            if isinstance(model_cfg.get("exclude_sessions"), list)
            else []
        )
        if blocked_sessions and current_session in blocked_sessions:
            return {
                "enabled": True,
                "match_count": 0,
                "matched_scopes": [],
                "predicted_action": "",
                "side_pattern": str(side_pattern),
            }
        current_hour_bucket = str(context_row.get("ctx_hour_bucket", "") or "").strip().lower()
        allowed_hour_buckets = (
            [
                str(v).strip().lower()
                for v in model_cfg.get("apply_hour_buckets", [])
                if str(v).strip()
            ]
            if isinstance(model_cfg.get("apply_hour_buckets"), list)
            else []
        )
        if allowed_hour_buckets and current_hour_bucket not in allowed_hour_buckets:
            return {
                "enabled": True,
                "match_count": 0,
                "matched_scopes": [],
                "predicted_action": "",
                "side_pattern": str(side_pattern),
            }
        blocked_hour_buckets = (
            [
                str(v).strip().lower()
                for v in model_cfg.get("exclude_hour_buckets", [])
                if str(v).strip()
            ]
            if isinstance(model_cfg.get("exclude_hour_buckets"), list)
            else []
        )
        if blocked_hour_buckets and current_hour_bucket in blocked_hour_buckets:
            return {
                "enabled": True,
                "match_count": 0,
                "matched_scopes": [],
                "predicted_action": "",
                "side_pattern": str(side_pattern),
            }
        current_close_bucket = str(context_row.get("st_close_bucket", "") or "").strip().lower()
        allowed_close_buckets = (
            [
                str(v).strip().lower()
                for v in model_cfg.get("apply_st_close_buckets", [])
                if str(v).strip()
            ]
            if isinstance(model_cfg.get("apply_st_close_buckets"), list)
            else []
        )
        if allowed_close_buckets and current_close_bucket not in allowed_close_buckets:
            return {
                "enabled": True,
                "match_count": 0,
                "matched_scopes": [],
                "predicted_action": "",
                "side_pattern": str(side_pattern),
            }
        blocked_close_buckets = (
            [
                str(v).strip().lower()
                for v in model_cfg.get("exclude_st_close_buckets", [])
                if str(v).strip()
            ]
            if isinstance(model_cfg.get("exclude_st_close_buckets"), list)
            else []
        )
        if blocked_close_buckets and current_close_bucket in blocked_close_buckets:
            return {
                "enabled": True,
                "match_count": 0,
                "matched_scopes": [],
                "predicted_action": "",
                "side_pattern": str(side_pattern),
            }
        context_row["ctx_volatility_regime"] = str(base_row.get("ctx_volatility_regime", "") or "").strip().lower()
        context_row["ctx_price_loc_bucket"] = self._entry_model_bucket_balance(
            safe_float(base_row.get("ctx_price_location", float("nan")), float("nan")),
            strong_neg=-1.25,
            neg=-0.35,
            pos=0.35,
            strong_pos=1.25,
        )
        long_strategy_type = str((long_row or {}).get("strategy_type", "") or "").strip().lower()
        short_strategy_type = str((short_row or {}).get("strategy_type", "") or "").strip().lower()
        long_sub_strategy = str((long_row or {}).get("sub_strategy", "") or "").strip().lower()
        short_sub_strategy = str((short_row or {}).get("sub_strategy", "") or "").strip().lower()
        context_row["long_timeframe"] = str((long_row or {}).get("timeframe", "") or "").strip().lower()
        context_row["short_timeframe"] = str((short_row or {}).get("timeframe", "") or "").strip().lower()
        context_row["long_sub_strategy"] = str(long_sub_strategy)
        context_row["short_sub_strategy"] = str(short_sub_strategy)
        context_row["long_strategy_style"] = self._entry_model_directionless_strategy_style(long_strategy_type)
        context_row["short_strategy_style"] = self._entry_model_directionless_strategy_style(short_strategy_type)
        long_rank = safe_float((long_row or {}).get("candidate_rank_before_adjustments", (long_row or {}).get("rank", float("nan"))), float("nan"))
        short_rank = safe_float((short_row or {}).get("candidate_rank_before_adjustments", (short_row or {}).get("rank", float("nan"))), float("nan"))
        long_score = safe_float((long_row or {}).get("final_score", (long_row or {}).get("decision_policy_score", (long_row or {}).get("selection_score", (long_row or {}).get("lane_score", float("nan"))))), float("nan"))
        short_score = safe_float((short_row or {}).get("final_score", (short_row or {}).get("decision_policy_score", (short_row or {}).get("selection_score", (short_row or {}).get("lane_score", float("nan"))))), float("nan"))
        long_edge = safe_float((long_row or {}).get("edge_points", (long_row or {}).get("runtime_rank_score", float("nan"))), float("nan"))
        short_edge = safe_float((short_row or {}).get("edge_points", (short_row or {}).get("runtime_rank_score", float("nan"))), float("nan"))
        baseline_side_guess = ""
        if isinstance(long_row, dict) and isinstance(short_row, dict):
            if math.isfinite(long_rank) and math.isfinite(short_rank) and long_rank != short_rank:
                baseline_side_guess = "long" if long_rank < short_rank else "short"
            elif math.isfinite(long_score) and math.isfinite(short_score) and long_score != short_score:
                baseline_side_guess = "long" if long_score > short_score else "short"
        elif isinstance(long_row, dict):
            baseline_side_guess = "long"
        elif isinstance(short_row, dict):
            baseline_side_guess = "short"
        if baseline_side_guess:
            context_row["chosen_side"] = str(baseline_side_guess)
            context_row["baseline_side_guess"] = str(baseline_side_guess)
            if baseline_side_guess == "long":
                context_row["chosen_sub_strategy"] = str(long_sub_strategy)
            elif baseline_side_guess == "short":
                context_row["chosen_sub_strategy"] = str(short_sub_strategy)
        allowed_baseline_sides = (
            [
                str(v).strip().lower()
                for v in model_cfg.get("apply_chosen_sides", [])
                if str(v).strip()
            ]
            if isinstance(model_cfg.get("apply_chosen_sides"), list)
            else []
        )
        if allowed_baseline_sides and str(baseline_side_guess).strip().lower() not in allowed_baseline_sides:
            return {
                "enabled": True,
                "match_count": 0,
                "matched_scopes": [],
                "predicted_action": "",
                "side_pattern": str(side_pattern),
            }
        score_gap_abs_max = safe_float(model_cfg.get("apply_score_gap_abs_max", float("nan")), float("nan"))
        if (
            isinstance(long_row, dict)
            and isinstance(short_row, dict)
            and math.isfinite(score_gap_abs_max)
            and math.isfinite(long_score)
            and math.isfinite(short_score)
            and abs(long_score - short_score) > max(0.0, score_gap_abs_max)
        ):
            return {
                "enabled": True,
                "match_count": 0,
                "matched_scopes": [],
                "predicted_action": "",
                "side_pattern": str(side_pattern),
            }
        rank_gap_abs_max = safe_float(model_cfg.get("apply_rank_gap_abs_max", float("nan")), float("nan"))
        if (
            isinstance(long_row, dict)
            and isinstance(short_row, dict)
            and math.isfinite(rank_gap_abs_max)
            and math.isfinite(long_rank)
            and math.isfinite(short_rank)
            and abs(long_rank - short_rank) > max(0.0, rank_gap_abs_max)
        ):
            return {
                "enabled": True,
                "match_count": 0,
                "matched_scopes": [],
                "predicted_action": "",
                "side_pattern": str(side_pattern),
            }
        if isinstance(long_row, dict) and isinstance(short_row, dict):
            context_row["rank_score_combo"] = (
                self._conflict_side_direction_bucket(short_rank - long_rank, neg_hi=-1.5, neg_lo=-0.5, pos_lo=0.5, pos_hi=1.5)
                + "|"
                + self._conflict_side_direction_bucket(long_score - short_score, neg_hi=-1.0, neg_lo=-0.25, pos_lo=0.25, pos_hi=1.0)
            )
            context_row["score_edge_combo"] = (
                self._conflict_side_direction_bucket(long_score - short_score, neg_hi=-1.0, neg_lo=-0.25, pos_lo=0.25, pos_hi=1.0)
                + "|"
                + self._conflict_side_direction_bucket(long_edge - short_edge, neg_hi=-1.0, neg_lo=-0.25, pos_lo=0.25, pos_hi=1.0)
            )
        else:
            context_row["rank_score_combo"] = ""
            context_row["score_edge_combo"] = ""
        scopes = model_cfg.get("scopes", []) if isinstance(model_cfg.get("scopes"), list) else []
        max_scopes_per_row = max(1, int(safe_float(model_cfg.get("max_scopes_per_row", 3), 3)))
        matches: List[Dict[str, Any]] = []
        for raw_scope in scopes:
            scope = raw_scope if isinstance(raw_scope, dict) else {}
            fields = [str(v).strip() for v in scope.get("fields", []) if str(v).strip()] if isinstance(scope.get("fields"), list) else []
            if not fields:
                continue
            bucket_key = self._conflict_side_bucket_key(context_row=context_row, fields=fields)
            if not bucket_key:
                continue
            buckets = scope.get("buckets", {}) if isinstance(scope.get("buckets"), dict) else {}
            bucket = buckets.get(bucket_key, {}) if isinstance(buckets.get(bucket_key, {}), dict) else {}
            if not bucket:
                continue
            weight = max(0.0, safe_float(scope.get("weight", 1.0), 1.0))
            if weight <= 0.0:
                continue
            max_abs = max(
                abs(safe_float(bucket.get("long_score", 0.0), 0.0)),
                abs(safe_float(bucket.get("short_score", 0.0), 0.0)),
                abs(safe_float(bucket.get("no_trade_score", 0.0), 0.0)),
            )
            matches.append(
                {
                    "scope_name": str(scope.get("name", "") or ""),
                    "weight": float(weight),
                    "weighted_abs": float(max_abs * weight),
                    "long_score": float(safe_float(bucket.get("long_score", 0.0), 0.0)),
                    "short_score": float(safe_float(bucket.get("short_score", 0.0), 0.0)),
                    "no_trade_score": float(safe_float(bucket.get("no_trade_score", 0.0), 0.0)),
                }
            )
        if not matches:
            return {
                "enabled": True,
                "match_count": 0,
                "matched_scopes": [],
                "predicted_action": "",
                "session": str(current_session).upper(),
                "side_pattern": str(side_pattern),
                "baseline_side_guess": str(baseline_side_guess),
            }
        matches.sort(key=lambda item: safe_float(item.get("weighted_abs", 0.0), 0.0), reverse=True)
        matches = matches[:max_scopes_per_row]
        total_weight = sum(max(0.0, safe_float(item.get("weight", 0.0), 0.0)) for item in matches)
        long_side_score = safe_div(sum(safe_float(item.get("long_score", 0.0), 0.0) * max(0.0, safe_float(item.get("weight", 0.0), 0.0)) for item in matches), total_weight, 0.0)
        short_side_score = safe_div(sum(safe_float(item.get("short_score", 0.0), 0.0) * max(0.0, safe_float(item.get("weight", 0.0), 0.0)) for item in matches), total_weight, 0.0)
        no_trade_score = safe_div(sum(safe_float(item.get("no_trade_score", 0.0), 0.0) * max(0.0, safe_float(item.get("weight", 0.0), 0.0)) for item in matches), total_weight, 0.0)
        side_scores: Dict[str, float] = {}
        if isinstance(long_row, dict):
            side_scores["long"] = float(long_side_score)
        if isinstance(short_row, dict):
            side_scores["short"] = float(short_side_score)
        predicted_action = ""
        if side_scores:
            best_side = max(side_scores.items(), key=lambda item: float(item[1]))[0]
            best_score = float(side_scores[best_side])
            other_scores = [float(v) for k, v in side_scores.items() if k != best_side]
            second_score = max(other_scores) if other_scores else float("-inf")
            trade_threshold = safe_float(model_cfg.get("selected_trade_threshold", float("inf")), float("inf"))
            side_margin = max(0.0, safe_float(model_cfg.get("selected_side_margin", 0.0), 0.0))
            no_trade_threshold = safe_float(model_cfg.get("selected_no_trade_threshold", float("inf")), float("inf"))
            no_trade_margin = max(0.0, safe_float(model_cfg.get("selected_no_trade_margin", 0.0), 0.0))
            if no_trade_score >= no_trade_threshold and no_trade_score >= (best_score + no_trade_margin):
                predicted_action = "no_trade"
            elif best_score >= trade_threshold and (not math.isfinite(second_score) or (best_score - second_score) >= side_margin):
                predicted_action = str(best_side)
        return {
            "enabled": True,
            "match_count": int(len(matches)),
            "matched_scopes": [str(item.get("scope_name", "") or "") for item in matches if str(item.get("scope_name", "") or "")],
            "predicted_action": str(predicted_action),
            "long_score": float(long_side_score),
            "short_score": float(short_side_score),
            "no_trade_score": float(no_trade_score),
            "session": str(current_session).upper(),
            "side_pattern": str(side_pattern),
            "baseline_side_guess": str(baseline_side_guess),
        }

    def _evaluate_context_prior_model(
        self,
        *,
        chosen_entry: Dict[str, Any],
        model_cfg: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not isinstance(model_cfg, dict) or not bool(model_cfg.get("enabled", False)):
            return {"score": 0.0, "match_count": 0, "matched_scopes": []}
        scopes = model_cfg.get("scopes", [])
        if not isinstance(scopes, list) or not scopes:
            return {"score": 0.0, "match_count": 0, "matched_scopes": []}
        max_abs_score = max(0.05, safe_float(model_cfg.get("max_abs_score", 1.25), 1.25))
        side_advantage_mode = str(model_cfg.get("side_advantage_mode", "off") or "off").strip().lower()
        if side_advantage_mode not in {"off", "prefer", "only"}:
            side_advantage_mode = "off"
        max_scopes_per_row = max(0, int(safe_float(model_cfg.get("max_scopes_per_row", 0), 0)))
        matches: List[Dict[str, Any]] = []
        for raw_scope in scopes:
            scope = raw_scope if isinstance(raw_scope, dict) else {}
            fields = (
                [str(v).strip() for v in scope.get("fields", []) if str(v).strip()]
                if isinstance(scope.get("fields"), list)
                else []
            )
            if not fields:
                continue
            bucket_key = self._entry_model_context_bucket_key(
                chosen_entry=chosen_entry,
                fields=fields,
            )
            if not bucket_key:
                continue
            buckets = scope.get("buckets", {}) if isinstance(scope.get("buckets"), dict) else {}
            bucket = buckets.get(bucket_key, {})
            if not isinstance(bucket, dict):
                continue
            weight = max(0.0, safe_float(scope.get("weight", 1.0), 1.0))
            if weight <= 0.0:
                continue
            prior_score = float(safe_float(bucket.get("prior_score", 0.0), 0.0))
            effective_score = prior_score
            if side_advantage_mode != "off" and "side_considered" in fields:
                side_advantage_score = safe_float(
                    bucket.get("side_advantage_score", float("nan")),
                    float("nan"),
                )
                if math.isfinite(side_advantage_score):
                    effective_score = float(side_advantage_score)
                elif side_advantage_mode == "only":
                    continue
            matches.append(
                {
                    "scope_name": str(scope.get("name", "") or ""),
                    "bucket_key": str(bucket_key),
                    "weight": float(weight),
                    "prior_score": float(effective_score),
                    "weighted_abs": float(abs(effective_score) * weight),
                }
            )
        if not matches:
            return {"score": 0.0, "match_count": 0, "matched_scopes": []}
        matches.sort(
            key=lambda item: (
                safe_float(item.get("weighted_abs", 0.0), 0.0),
                abs(safe_float(item.get("prior_score", 0.0), 0.0)),
            ),
            reverse=True,
        )
        if max_scopes_per_row > 0:
            matches = matches[:max_scopes_per_row]
        total_weight = sum(max(0.0, safe_float(item.get("weight", 0.0), 0.0)) for item in matches)
        score = safe_div(
            sum(
                safe_float(item.get("prior_score", 0.0), 0.0)
                * max(0.0, safe_float(item.get("weight", 0.0), 0.0))
                for item in matches
            ),
            total_weight,
            0.0,
        )
        score = float(clip(score, -max_abs_score, max_abs_score))
        return {
            "score": float(score),
            "match_count": int(len(matches)),
            "matched_scopes": [
                str(item.get("scope_name", "") or "")
                for item in matches
                if str(item.get("scope_name", "") or "")
            ],
        }

    def _evaluate_short_term_condition_model(
        self,
        *,
        chosen_entry: Dict[str, Any],
        model_cfg: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not isinstance(model_cfg, dict) or not bool(model_cfg.get("enabled", False)):
            return {"score": 0.0, "match_count": 0, "matched_scopes": []}
        scopes = model_cfg.get("scopes", [])
        if not isinstance(scopes, list) or not scopes:
            return {"score": 0.0, "match_count": 0, "matched_scopes": []}
        side = str(chosen_entry.get("side_considered", "") or "").strip().lower()
        if side not in {"long", "short"}:
            lane_side = str(lane_to_side(self._entry_model_lane(chosen_entry)) or "").strip().lower()
            side = lane_side if lane_side in {"long", "short"} else ""
        if side not in {"long", "short"}:
            return {"score": 0.0, "match_count": 0, "matched_scopes": []}
        score_key = f"{side}_score"
        max_abs_score = max(0.05, safe_float(model_cfg.get("max_abs_score", 1.35), 1.35))
        max_scopes_per_row = max(1, int(safe_float(model_cfg.get("max_scopes_per_row", 3), 3)))
        matches: List[Dict[str, Any]] = []
        for raw_scope in scopes:
            scope = raw_scope if isinstance(raw_scope, dict) else {}
            fields = (
                [str(v).strip() for v in scope.get("fields", []) if str(v).strip()]
                if isinstance(scope.get("fields"), list)
                else []
            )
            if not fields:
                continue
            bucket_key = self._entry_model_short_term_bucket_key(
                chosen_entry=chosen_entry,
                fields=fields,
            )
            if not bucket_key:
                continue
            buckets = scope.get("buckets", {}) if isinstance(scope.get("buckets"), dict) else {}
            bucket = buckets.get(bucket_key, {})
            if not isinstance(bucket, dict):
                continue
            side_score = safe_float(bucket.get(score_key, float("nan")), float("nan"))
            if not math.isfinite(side_score):
                continue
            weight = max(0.0, safe_float(scope.get("weight", 1.0), 1.0))
            if weight <= 0.0:
                continue
            matches.append(
                {
                    "scope_name": str(scope.get("name", "") or ""),
                    "bucket_key": str(bucket_key),
                    "weight": float(weight),
                    "side_score": float(side_score),
                    "weighted_abs": float(abs(side_score) * weight),
                }
            )
        if not matches:
            return {"score": 0.0, "match_count": 0, "matched_scopes": []}
        matches.sort(
            key=lambda item: (
                safe_float(item.get("weighted_abs", 0.0), 0.0),
                abs(safe_float(item.get("side_score", 0.0), 0.0)),
            ),
            reverse=True,
        )
        matches = matches[:max_scopes_per_row]
        total_weight = sum(max(0.0, safe_float(item.get("weight", 0.0), 0.0)) for item in matches)
        score = safe_div(
            sum(
                safe_float(item.get("side_score", 0.0), 0.0)
                * max(0.0, safe_float(item.get("weight", 0.0), 0.0))
                for item in matches
            ),
            total_weight,
            0.0,
        )
        score = float(clip(score, -max_abs_score, max_abs_score))
        return {
            "score": float(score),
            "match_count": int(len(matches)),
            "matched_scopes": [
                str(item.get("scope_name", "") or "")
                for item in matches
                if str(item.get("scope_name", "") or "")
            ],
        }

    def _evaluate_action_condition_model(
        self,
        *,
        chosen_entry: Dict[str, Any],
        model_cfg: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not isinstance(model_cfg, dict) or not bool(model_cfg.get("enabled", False)):
            return {"score": 0.0, "match_count": 0, "matched_scopes": []}
        if bool(model_cfg.get("apply_only_top_side_candidate", False)) and (
            not bool(chosen_entry.get("action_condition_top_side_candidate", False))
        ):
            return {"score": 0.0, "match_count": 0, "matched_scopes": []}
        scopes = model_cfg.get("scopes", [])
        if not isinstance(scopes, list) or not scopes:
            return {"score": 0.0, "match_count": 0, "matched_scopes": []}
        max_abs_score = max(0.05, safe_float(model_cfg.get("max_abs_score", 1.35), 1.35))
        max_scopes_per_row = max(1, int(safe_float(model_cfg.get("max_scopes_per_row", 3), 3)))
        matches: List[Dict[str, Any]] = []
        for raw_scope in scopes:
            scope = raw_scope if isinstance(raw_scope, dict) else {}
            fields = (
                [str(v).strip() for v in scope.get("fields", []) if str(v).strip()]
                if isinstance(scope.get("fields"), list)
                else []
            )
            if not fields:
                continue
            bucket_key = self._entry_model_short_term_bucket_key(
                chosen_entry=chosen_entry,
                fields=fields,
            )
            if not bucket_key:
                continue
            buckets = scope.get("buckets", {}) if isinstance(scope.get("buckets"), dict) else {}
            bucket = buckets.get(bucket_key, {})
            if not isinstance(bucket, dict):
                continue
            score = safe_float(bucket.get("score", float("nan")), float("nan"))
            if not math.isfinite(score):
                continue
            weight = max(0.0, safe_float(scope.get("weight", 1.0), 1.0))
            if weight <= 0.0:
                continue
            matches.append(
                {
                    "scope_name": str(scope.get("name", "") or ""),
                    "bucket_key": str(bucket_key),
                    "weight": float(weight),
                    "score": float(score),
                    "weighted_abs": float(abs(score) * weight),
                }
            )
        if not matches:
            return {"score": 0.0, "match_count": 0, "matched_scopes": []}
        matches.sort(
            key=lambda item: (
                safe_float(item.get("weighted_abs", 0.0), 0.0),
                abs(safe_float(item.get("score", 0.0), 0.0)),
            ),
            reverse=True,
        )
        matches = matches[:max_scopes_per_row]
        total_weight = sum(max(0.0, safe_float(item.get("weight", 0.0), 0.0)) for item in matches)
        score = safe_div(
            sum(
                safe_float(item.get("score", 0.0), 0.0)
                * max(0.0, safe_float(item.get("weight", 0.0), 0.0))
                for item in matches
            ),
            total_weight,
            0.0,
        )
        score = float(clip(score, -max_abs_score, max_abs_score))
        return {
            "score": float(score),
            "match_count": int(len(matches)),
            "matched_scopes": [
                str(item.get("scope_name", "") or "")
                for item in matches
                if str(item.get("scope_name", "") or "")
            ],
        }

    @staticmethod
    def _entry_model_shape_scope_key(*, lane: str, timeframe: str, scope_mode: str) -> str:
        lane_text = str(lane or "").strip()
        timeframe_text = str(timeframe or "").strip()
        mode = str(scope_mode or "lane_timeframe").strip().lower()
        if mode == "lane":
            return lane_text
        if mode == "timeframe":
            return timeframe_text
        if lane_text and timeframe_text:
            return f"{lane_text}|{timeframe_text}"
        return lane_text or timeframe_text

    def _evaluate_shape_penalty_model(
        self,
        *,
        chosen_entry: Dict[str, Any],
        model_cfg: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not isinstance(model_cfg, dict) or not bool(model_cfg.get("enabled", False)):
            return {"strength": 0.0, "match_count": 0, "scope_key": ""}
        rules = model_cfg.get("rules", [])
        if not isinstance(rules, list) or not rules:
            return {"strength": 0.0, "match_count": 0, "scope_key": ""}
        lane = self._entry_model_lane(chosen_entry)
        timeframe = self._entry_model_timeframe(chosen_entry)
        scope_key = self._entry_model_shape_scope_key(
            lane=lane,
            timeframe=timeframe,
            scope_mode=str(model_cfg.get("scope_mode", "lane_timeframe") or "lane_timeframe"),
        )
        if not scope_key:
            return {"strength": 0.0, "match_count": 0, "scope_key": ""}
        max_rules_per_row = max(0, int(safe_float(model_cfg.get("max_rules_per_row", 0), 0)))
        cand = chosen_entry.get("cand", {}) if isinstance(chosen_entry.get("cand"), dict) else {}
        strength = 0.0
        match_count = 0
        for rule in rules:
            if str(rule.get("scope_key", "") or "") != scope_key:
                continue
            feature = str(rule.get("feature", "") or "").strip()
            operator = str(rule.get("operator", "") or "").strip()
            threshold = safe_float(rule.get("threshold", float("nan")), float("nan"))
            value = safe_float(
                chosen_entry.get(feature, cand.get(feature, float("nan"))),
                float("nan"),
            )
            if (not feature) or (not math.isfinite(threshold)) or (not math.isfinite(value)):
                continue
            hit = (operator == "<=" and value <= threshold) or (operator == ">=" and value >= threshold)
            if not hit:
                continue
            strength += max(0.0, safe_float(rule.get("penalty_strength", 0.0), 0.0))
            match_count += 1
            if max_rules_per_row > 0 and match_count >= max_rules_per_row:
                break
        return {
            "strength": float(strength),
            "match_count": int(match_count),
            "scope_key": str(scope_key),
        }

    def _resolve_scored_model_stats(
        self,
        *,
        chosen_entry: Dict[str, Any],
        model_cfg: Dict[str, Any],
        variant_stats: Dict[str, Any],
        lane_stats: Dict[str, Any],
        global_stats: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], str]:
        variant_id = self._entry_model_variant_id(chosen_entry)
        lane = self._entry_model_lane(chosen_entry)
        min_variant = max(1, int(safe_float(model_cfg.get("min_variant_trades", 25), 25)))
        min_lane = max(1, int(safe_float(model_cfg.get("min_lane_trades", 120), 120)))
        v_stats = variant_stats.get(variant_id, {}) if variant_id else {}
        if isinstance(v_stats, dict) and int(safe_float(v_stats.get("n_trades", 0), 0)) >= min_variant:
            return v_stats, "variant"
        l_stats = lane_stats.get(lane, {}) if lane else {}
        if isinstance(l_stats, dict) and int(safe_float(l_stats.get("n_trades", 0), 0)) >= min_lane:
            return l_stats, "lane"
        if isinstance(global_stats, dict) and int(safe_float(global_stats.get("n_trades", 0), 0)) > 0:
            return global_stats, "global"
        return {}, "missing"

    def _direct_decision_lane_prior(
        self,
        *,
        lane: str,
        session_name: str,
        timeframe_hint: str,
        model_cfg: Dict[str, Any],
    ) -> float:
        router_payload = (
            model_cfg.get("router_model_or_router_rules", {})
            if isinstance(model_cfg.get("router_model_or_router_rules", {}), dict)
            else {}
        )
        session_map = (
            router_payload.get("lane_priors_by_session", {})
            if isinstance(router_payload.get("lane_priors_by_session", {}), dict)
            else {}
        )
        timeframe_map = (
            router_payload.get("lane_priors_by_timeframe", {})
            if isinstance(router_payload.get("lane_priors_by_timeframe", {}), dict)
            else {}
        )
        global_map = (
            router_payload.get("lane_priors_global", {})
            if isinstance(router_payload.get("lane_priors_global", {}), dict)
            else {}
        )
        session_val = 0.0
        if session_name and isinstance(session_map.get(session_name), dict):
            session_val = safe_float((session_map.get(session_name) or {}).get(lane, 0.0), 0.0)
        timeframe_val = 0.0
        if timeframe_hint and isinstance(timeframe_map.get(timeframe_hint), dict):
            timeframe_val = safe_float((timeframe_map.get(timeframe_hint) or {}).get(lane, 0.0), 0.0)
        global_val = safe_float(global_map.get(lane, 0.0), 0.0)
        return float((0.55 * session_val) + (0.20 * timeframe_val) + (0.25 * global_val))

    def _evaluate_direct_decision_model(
        self,
        *,
        candidate_entry: Dict[str, Any],
        default_session: str,
        model_cfg: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        model_cfg = (
            dict(model_cfg)
            if isinstance(model_cfg, dict)
            else dict(self.direct_decision_model_cfg)
        )
        if not bool(model_cfg.get("enabled", False)):
            return {
                "enabled": False,
                "allow": True,
                "score": 0.0,
                "threshold": 0.0,
                "scope": "disabled",
                "components": {},
                "stats": {},
                "fallback_scope_guard": {},
            }

        variant_stats = (
            model_cfg.get("variant_stats", {})
            if isinstance(model_cfg.get("variant_stats", {}), dict)
            else {}
        )
        lane_stats = (
            model_cfg.get("lane_stats", {})
            if isinstance(model_cfg.get("lane_stats", {}), dict)
            else {}
        )
        global_stats = (
            model_cfg.get("global_stats", {})
            if isinstance(model_cfg.get("global_stats", {}), dict)
            else {}
        )
        stats, scope = self._resolve_scored_model_stats(
            chosen_entry=candidate_entry,
            model_cfg=model_cfg,
            variant_stats=variant_stats,
            lane_stats=lane_stats,
            global_stats=global_stats,
        )
        scope_norm = str(scope or "").strip().lower()
        allow_on_missing = bool(model_cfg.get("allow_on_missing_stats", True))
        if scope_norm == "missing":
            return {
                "enabled": True,
                "allow": bool(allow_on_missing),
                "score": 0.0,
                "threshold": float(safe_float(model_cfg.get("selected_threshold", 0.0), 0.0)),
                "threshold_base": float(safe_float(model_cfg.get("selected_threshold", 0.0), 0.0)),
                "threshold_scope_offset": 0.0,
                "scope": "missing",
                "stats": {},
                "components": {},
                "fallback_scope_guard": {
                    "scope": "missing",
                    "allow_on_missing_stats": bool(allow_on_missing),
                },
            }

        lane = self._entry_model_lane(candidate_entry)
        timeframe = self._entry_model_timeframe(candidate_entry)
        session_name = str(candidate_entry.get("session", "") or "").strip() or str(default_session or "")
        variant_id = self._entry_model_variant_id(candidate_entry)
        side_considered = str(candidate_entry.get("side_considered", "") or "").strip().lower()
        if side_considered not in {"long", "short"}:
            side_considered = str(lane_to_side(lane) or "").strip().lower()
        if side_considered not in {"long", "short"}:
            side_raw = str(candidate_entry.get("side", "") or "").strip().lower()
            if side_raw in {"long", "buy"}:
                side_considered = "long"
            elif side_raw in {"short", "sell"}:
                side_considered = "short"
            else:
                side_considered = ""
        score_cfg = (
            model_cfg.get("score_components", {})
            if isinstance(model_cfg.get("score_components", {}), dict)
            else {}
        )
        context_prior_model = (
            model_cfg.get("context_prior_model", {})
            if isinstance(model_cfg.get("context_prior_model", {}), dict)
            else {}
        )
        short_term_condition_model = (
            model_cfg.get("short_term_condition_model", {})
            if isinstance(model_cfg.get("short_term_condition_model", {}), dict)
            else {}
        )
        action_condition_model = (
            model_cfg.get("action_condition_model", {})
            if isinstance(model_cfg.get("action_condition_model", {}), dict)
            else {}
        )
        variant_priors = (
            model_cfg.get("variant_quality_priors", {})
            if isinstance(model_cfg.get("variant_quality_priors", {}), dict)
            else {}
        )
        lane_prior = self._direct_decision_lane_prior(
            lane=lane,
            session_name=session_name,
            timeframe_hint=timeframe,
            model_cfg=model_cfg,
        )
        variant_prior = safe_float(variant_priors.get(variant_id, 0.0), 0.0)

        lane_inputs = (
            dict(candidate_entry.get("lane_score_inputs", {}))
            if isinstance(candidate_entry.get("lane_score_inputs", {}), dict)
            else {}
        )
        lane_inputs["variant_quality_prior"] = float(variant_prior)
        candidate_entry["lane_score_inputs"] = lane_inputs

        fallback_allow, fallback_reason, fallback_guard_meta = self._evaluate_fallback_scope_guard(
            chosen_entry=candidate_entry,
            stats=stats,
            scope=scope,
            model_cfg=model_cfg,
        )
        if not fallback_allow:
            return {
                "enabled": True,
                "allow": False,
                "score": 0.0,
                "threshold": float(safe_float(model_cfg.get("selected_threshold", 0.0), 0.0)),
                "threshold_base": float(safe_float(model_cfg.get("selected_threshold", 0.0), 0.0)),
                "threshold_scope_offset": 0.0,
                "scope": str(scope_norm),
                "stats": dict(stats) if isinstance(stats, dict) else {},
                "components": {},
                "fallback_scope_guard": dict(fallback_guard_meta),
                "reason": str(fallback_reason),
            }

        edge_points = safe_float(
            candidate_entry.get("edge_points", candidate_entry.get("runtime_rank_score", 0.0)),
            0.0,
        )
        structural = safe_float(candidate_entry.get("structural_score", 0.0), 0.0)

        w_quality = safe_float(score_cfg.get("weight_quality_lcb", 0.56), 0.56)
        w_lane_prior = safe_float(score_cfg.get("weight_lane_prior", 0.10), 0.10)
        w_variant_prior = safe_float(score_cfg.get("weight_variant_quality_prior", 0.10), 0.10)
        w_edge = safe_float(score_cfg.get("weight_edge_points", 0.12), 0.12)
        w_struct = safe_float(score_cfg.get("weight_structural_score", 0.06), 0.06)
        w_pf = safe_float(score_cfg.get("weight_profit_factor_component", 0.10), 0.10)
        w_year = safe_float(score_cfg.get("weight_year_coverage_component", 0.06), 0.06)
        w_loss = safe_float(score_cfg.get("weight_loss_share_penalty", 0.16), 0.16)
        w_stop = safe_float(score_cfg.get("weight_stop_like_share_penalty", 0.10), 0.10)
        w_drawdown = safe_float(score_cfg.get("weight_drawdown_penalty", 0.10), 0.10)
        w_worst = safe_float(score_cfg.get("weight_worst_block_penalty", 0.12), 0.12)
        w_shape = safe_float(score_cfg.get("weight_shape_penalty_component", 0.18), 0.18)
        w_context = safe_float(score_cfg.get("weight_context_prior_component", 0.0), 0.0)
        w_short_term = safe_float(score_cfg.get("weight_short_term_condition_component", 0.0), 0.0)
        w_action = safe_float(score_cfg.get("weight_action_condition_component", 0.0), 0.0)
        w_decision_side = safe_float(
            score_cfg.get(
                "weight_decision_side_prior_component",
                (
                    self.decision_side_model_cfg.get("prior_component_weight", 0.0)
                    if isinstance(self.decision_side_model_cfg, dict)
                    else 0.0
                ),
            ),
            0.0,
        )
        side_bias_cfg = (
            score_cfg.get("side_score_bias", {})
            if isinstance(score_cfg.get("side_score_bias"), dict)
            else {}
        )
        side_bias_component = float(
            safe_float(
                side_bias_cfg.get(
                    side_considered,
                    side_bias_cfg.get("default", 0.0),
                ),
                0.0,
            )
        )
        lane_prior_center = safe_float(score_cfg.get("lane_prior_center", 0.15), 0.15)
        lane_prior_scale = max(1e-9, safe_float(score_cfg.get("lane_prior_scale", 0.08), 0.08))
        variant_prior_center = safe_float(
            score_cfg.get("variant_quality_prior_center", 0.27),
            0.27,
        )
        variant_prior_scale = max(
            1e-9,
            safe_float(score_cfg.get("variant_quality_prior_scale", 0.12), 0.12),
        )
        edge_scale = max(1e-9, safe_float(score_cfg.get("edge_scale_points", 0.40), 0.40))
        struct_scale = max(1e-9, safe_float(score_cfg.get("structural_scale", 0.80), 0.80))
        profit_factor_center = safe_float(score_cfg.get("profit_factor_center", 1.10), 1.10)
        profit_factor_scale = max(
            1e-9,
            safe_float(score_cfg.get("profit_factor_scale", 0.35), 0.35),
        )
        year_coverage_full = max(
            1.0,
            safe_float(score_cfg.get("year_coverage_full_years", 8.0), 8.0),
        )
        loss_center = safe_float(score_cfg.get("loss_share_center", 0.52), 0.52)
        loss_scale = max(1e-9, safe_float(score_cfg.get("loss_share_scale", 0.22), 0.22))
        stop_center = safe_float(score_cfg.get("stop_like_share_center", 0.62), 0.62)
        stop_scale = max(1e-9, safe_float(score_cfg.get("stop_like_share_scale", 0.25), 0.25))
        drawdown_scale = max(1e-9, safe_float(score_cfg.get("drawdown_scale", 6.0), 6.0))
        worst_block_scale = max(
            1e-9,
            safe_float(score_cfg.get("worst_block_scale_points", 3.0), 3.0),
        )
        shape_penalty_scale = max(
            1e-9,
            safe_float(score_cfg.get("shape_penalty_scale", 1.0), 1.0),
        )
        shape_penalty_cap = max(0.0, safe_float(score_cfg.get("shape_penalty_cap", 2.0), 2.0))
        context_prior_scale = max(
            1e-9,
            safe_float(score_cfg.get("context_prior_scale", 0.35), 0.35),
        )
        short_term_condition_scale = max(
            1e-9,
            safe_float(score_cfg.get("short_term_condition_scale", 0.60), 0.60),
        )
        action_condition_scale = max(
            1e-9,
            safe_float(score_cfg.get("action_condition_scale", 0.60), 0.60),
        )
        decision_side_scale = max(
            1e-9,
            safe_float(
                score_cfg.get(
                    "decision_side_prior_scale",
                    (
                        self.decision_side_model_cfg.get("prior_component_scale", 0.60)
                        if isinstance(self.decision_side_model_cfg, dict)
                        else 0.60
                    ),
                ),
                0.60,
            ),
        )

        base_quality = safe_float(stats.get("quality_lcb_score", 0.0), 0.0)
        loss_share = clip(safe_float(stats.get("loss_share", loss_center), loss_center), 0.0, 1.0)
        stop_like_share = clip(
            safe_float(stats.get("stop_like_share", stop_center), stop_center),
            0.0,
            1.0,
        )
        profit_factor = max(
            0.0,
            safe_float(stats.get("profit_factor", profit_factor_center), profit_factor_center),
        )
        year_coverage = max(0.0, safe_float(stats.get("year_coverage", 0.0), 0.0))
        year_coverage_ratio = clip(safe_div(year_coverage, year_coverage_full, 0.0), 0.0, 1.0)
        drawdown_norm = max(0.0, safe_float(stats.get("drawdown_norm", 0.0), 0.0))
        worst_block_avg_pnl = safe_float(stats.get("worst_block_avg_pnl", 0.0), 0.0)
        loss_excess = max(0.0, loss_share - loss_center)
        stop_excess = max(0.0, stop_like_share - stop_center)
        worst_block_shortfall = max(0.0, -worst_block_avg_pnl)
        shape_eval = self._evaluate_shape_penalty_model(
            chosen_entry=candidate_entry,
            model_cfg=(
                model_cfg.get("shape_penalty_model", {})
                if isinstance(model_cfg.get("shape_penalty_model", {}), dict)
                else {}
            ),
        )
        shape_strength = min(
            float(shape_penalty_cap),
            max(0.0, safe_float(shape_eval.get("strength", 0.0), 0.0)),
        )
        context_eval = self._evaluate_context_prior_model(
            chosen_entry=candidate_entry,
            model_cfg=context_prior_model,
        )
        context_prior_score = safe_float(context_eval.get("score", 0.0), 0.0)
        short_term_eval = self._evaluate_short_term_condition_model(
            chosen_entry=candidate_entry,
            model_cfg=short_term_condition_model,
        )
        short_term_condition_score = safe_float(short_term_eval.get("score", 0.0), 0.0)
        action_condition_eval = self._evaluate_action_condition_model(
            chosen_entry=candidate_entry,
            model_cfg=action_condition_model,
        )
        action_condition_score = safe_float(action_condition_eval.get("score", 0.0), 0.0)
        decision_side_prior_score = safe_float(
            candidate_entry.get("decision_side_model_prior_score", 0.0),
            0.0,
        )
        decision_side_prior_component_total = safe_float(
            candidate_entry.get("decision_side_model_prior_component_total", float("nan")),
            float("nan"),
        )
        decision_side_component_override = (
            math.isfinite(decision_side_prior_component_total)
            and "weight_decision_side_prior_component" not in score_cfg
            and "decision_side_prior_scale" not in score_cfg
        )
        if decision_side_component_override:
            decision_side_prior_component = float(decision_side_prior_component_total)
        else:
            decision_side_prior_component = float(
                w_decision_side * math.tanh(decision_side_prior_score / decision_side_scale)
            )

        components = {
            "quality_lcb_component": float(w_quality * base_quality),
            "lane_prior_component": float(
                w_lane_prior * math.tanh((lane_prior - lane_prior_center) / lane_prior_scale)
            ),
            "variant_quality_prior_component": float(
                w_variant_prior
                * math.tanh((variant_prior - variant_prior_center) / variant_prior_scale)
            ),
            "edge_points_component": float(w_edge * math.tanh(edge_points / edge_scale)),
            "structural_component": float(w_struct * math.tanh(structural / struct_scale)),
            "profit_factor_component": float(
                w_pf * math.tanh((profit_factor - profit_factor_center) / profit_factor_scale)
            ),
            "year_coverage_component": float(w_year * ((2.0 * year_coverage_ratio) - 1.0)),
            "loss_share_penalty_component": float(-w_loss * math.tanh(loss_excess / loss_scale)),
            "stop_like_share_penalty_component": float(-w_stop * math.tanh(stop_excess / stop_scale)),
            "drawdown_penalty_component": float(-w_drawdown * math.tanh(drawdown_norm / drawdown_scale)),
            "worst_block_penalty_component": float(
                -w_worst * math.tanh(worst_block_shortfall / worst_block_scale)
            ),
            "shape_penalty_component": float(
                -w_shape * math.tanh(shape_strength / shape_penalty_scale)
            ),
            "context_prior_component": float(
                w_context * math.tanh(context_prior_score / context_prior_scale)
            ),
            "short_term_condition_component": float(
                w_short_term * math.tanh(short_term_condition_score / short_term_condition_scale)
            ),
            "action_condition_component": float(
                w_action * math.tanh(action_condition_score / action_condition_scale)
            ),
            "decision_side_prior_component": float(decision_side_prior_component),
            "side_bias_component": float(side_bias_component),
        }
        score = float(sum(float(v or 0.0) for v in components.values()))
        threshold_base = float(safe_float(model_cfg.get("selected_threshold", 0.0), 0.0))
        scope_offsets_cfg = (
            model_cfg.get("scope_threshold_offsets", {})
            if isinstance(model_cfg.get("scope_threshold_offsets"), dict)
            else {}
        )
        scope_offset = float(
            safe_float(
                scope_offsets_cfg.get(
                    str(scope),
                    scope_offsets_cfg.get("default", 0.0),
                ),
                0.0,
            )
        )
        threshold = float(threshold_base + scope_offset)
        allow = bool(score >= threshold)
        return {
            "enabled": True,
            "allow": bool(allow),
            "score": float(score),
            "threshold": float(threshold),
            "threshold_base": float(threshold_base),
            "threshold_scope_offset": float(scope_offset),
            "scope": str(scope),
            "stats": {
                "n_trades": int(safe_float(stats.get("n_trades", 0), 0)),
                "p_win_lcb": float(safe_float(stats.get("p_win_lcb", 0.0), 0.0)),
                "ev_lcb": float(safe_float(stats.get("ev_lcb", 0.0), 0.0)),
                "quality_lcb_score": float(base_quality),
                "profit_factor": float(profit_factor),
                "loss_share": float(loss_share),
                "stop_like_share": float(stop_like_share),
                "drawdown_norm": float(drawdown_norm),
                "worst_block_avg_pnl": float(worst_block_avg_pnl),
                "year_coverage": int(safe_float(stats.get("year_coverage", 0), 0)),
                "shape_penalty_strength": float(shape_strength),
                "shape_rule_hit_count": int(safe_float(shape_eval.get("match_count", 0), 0)),
                "shape_scope_key": str(shape_eval.get("scope_key", "") or ""),
                "context_prior_score": float(context_prior_score),
                "context_prior_match_count": int(safe_float(context_eval.get("match_count", 0), 0)),
                "context_prior_scopes": "|".join(
                    [
                        str(v).strip()
                        for v in (
                            context_eval.get("matched_scopes", [])
                            if isinstance(context_eval.get("matched_scopes", []), list)
                            else []
                        )
                        if str(v).strip()
                    ]
                ),
                "short_term_condition_score": float(short_term_condition_score),
                "short_term_condition_match_count": int(
                    safe_float(short_term_eval.get("match_count", 0), 0)
                ),
                "short_term_condition_scopes": "|".join(
                    [
                        str(v).strip()
                        for v in (
                            short_term_eval.get("matched_scopes", [])
                            if isinstance(short_term_eval.get("matched_scopes", []), list)
                            else []
                        )
                        if str(v).strip()
                    ]
                ),
                "action_condition_score": float(action_condition_score),
                "action_condition_match_count": int(
                    safe_float(action_condition_eval.get("match_count", 0), 0)
                ),
                "action_condition_scopes": "|".join(
                    [
                        str(v).strip()
                        for v in (
                            action_condition_eval.get("matched_scopes", [])
                            if isinstance(action_condition_eval.get("matched_scopes", []), list)
                            else []
                        )
                        if str(v).strip()
                    ]
                ),
                "decision_side_prior_score": float(decision_side_prior_score),
                "lane_prior": float(lane_prior),
                "variant_quality_prior": float(variant_prior),
                "side_considered": str(side_considered),
            },
            "components": dict(components),
            "fallback_scope_guard": dict(fallback_guard_meta),
        }

    def _select_with_direct_decision_model(
        self,
        *,
        filtered_rows: List[Dict[str, Any]],
        default_session: str,
        model_cfg: Optional[Dict[str, Any]] = None,
        track_counters: bool = True,
    ) -> Optional[Dict[str, Any]]:
        model_cfg = (
            dict(model_cfg)
            if isinstance(model_cfg, dict)
            else dict(self.direct_decision_model_cfg)
        )
        selection_mode = str(
            model_cfg.get(
                "selection_mode",
                self.direct_decision_selection_mode or "replace_router_lane",
            )
            or "replace_router_lane"
        ).strip().lower()
        direct_decision_enabled = bool(
            self._direct_decision_runtime_enabled
            and isinstance(model_cfg, dict)
            and bool(model_cfg.get("enabled", False))
            and selection_mode
            in {"replace_router_lane", "hybrid_fallback_router_lane", "hybrid_compare_baseline"}
        )
        if (not direct_decision_enabled) or (not filtered_rows):
            return None
        top_side_rows: Dict[str, Dict[str, Any]] = {}
        for row in filtered_rows:
            if not isinstance(row, dict):
                continue
            side_text = str(row.get("side_considered", "") or "").strip().lower()
            if side_text not in {"long", "short"}:
                continue
            current_best = top_side_rows.get(side_text)
            current_rank = safe_int(
                row.get("candidate_rank_before_adjustments", row.get("rank", 0)),
                0,
            )
            current_key = (
                current_rank if current_rank > 0 else 10**9,
                -safe_float(row.get("runtime_rank_score", row.get("edge_points", 0.0)), 0.0),
                -safe_float(row.get("edge_points", 0.0), 0.0),
                -safe_float(row.get("structural_score", 0.0), 0.0),
            )
            if not isinstance(current_best, dict):
                top_side_rows[side_text] = row
                continue
            best_rank = safe_int(
                current_best.get("candidate_rank_before_adjustments", current_best.get("rank", 0)),
                0,
            )
            best_key = (
                best_rank if best_rank > 0 else 10**9,
                -safe_float(current_best.get("runtime_rank_score", current_best.get("edge_points", 0.0)), 0.0),
                -safe_float(current_best.get("edge_points", 0.0), 0.0),
                -safe_float(current_best.get("structural_score", 0.0), 0.0),
            )
            if current_key < best_key:
                top_side_rows[side_text] = row
        for row in filtered_rows:
            if not isinstance(row, dict):
                continue
            side_text = str(row.get("side_considered", "") or "").strip().lower()
            row["action_condition_top_side_candidate"] = bool(
                side_text in top_side_rows and top_side_rows.get(side_text) is row
            )
        long_available = "long" in top_side_rows
        short_available = "short" in top_side_rows
        if long_available and short_available:
            side_pattern = "both"
        elif long_available:
            side_pattern = "long_only"
        elif short_available:
            side_pattern = "short_only"
        else:
            side_pattern = ""
        for row in filtered_rows:
            if not isinstance(row, dict):
                continue
            side_text = str(row.get("side_considered", "") or "").strip().lower()
            row["side_pattern"] = str(side_pattern)
            row["long_available"] = bool(long_available)
            row["short_available"] = bool(short_available)
            row["opposite_available"] = bool(
                (side_text == "long" and short_available)
                or (side_text == "short" and long_available)
            )
        scored_rows: List[Dict[str, Any]] = []
        lane_scores_map: Dict[str, float] = {}
        for row in filtered_rows:
            if not isinstance(row, dict):
                continue
            eval_result = self._evaluate_direct_decision_model(
                candidate_entry=row,
                default_session=default_session,
                model_cfg=model_cfg,
            )
            row_scored = dict(row)
            row_scored["decision_policy_score"] = float(safe_float(eval_result.get("score", 0.0), 0.0))
            row_scored["decision_policy_allow"] = bool(eval_result.get("allow", True))
            row_scored["decision_policy_threshold"] = float(
                safe_float(eval_result.get("threshold", 0.0), 0.0)
            )
            row_scored["decision_policy_scope"] = str(eval_result.get("scope", "") or "")
            row_scored["decision_policy_components"] = (
                dict(eval_result.get("components", {}))
                if isinstance(eval_result.get("components", {}), dict)
                else {}
            )
            row_scored["decision_policy_stats"] = (
                dict(eval_result.get("stats", {}))
                if isinstance(eval_result.get("stats", {}), dict)
                else {}
            )
            row_scored["decision_policy_fallback_scope_guard"] = (
                dict(eval_result.get("fallback_scope_guard", {}))
                if isinstance(eval_result.get("fallback_scope_guard", {}), dict)
                else {}
            )
            scored_rows.append(row_scored)
            lane = str(row_scored.get("lane", "") or "")
            if lane:
                lane_scores_map[lane] = max(
                    float(row_scored.get("decision_policy_score", float("-inf")) or float("-inf")),
                    float(lane_scores_map.get(lane, float("-inf"))),
                )

        if not scored_rows:
            return None
        scored_rows.sort(
            key=lambda r: (
                float(r.get("decision_policy_score", float("-inf")) or float("-inf")),
                float(r.get("edge_points", 0.0) or 0.0),
                float(r.get("structural_score", 0.0) or 0.0),
            ),
            reverse=True,
        )
        best = scored_rows[0]
        runner_up = scored_rows[1] if len(scored_rows) > 1 else None
        best_score = float(best.get("decision_policy_score", 0.0) or 0.0)
        best_threshold = float(best.get("decision_policy_threshold", 0.0) or 0.0)
        best_allow = bool(best.get("decision_policy_allow", True))
        second_score = float(runner_up.get("decision_policy_score", best_score) or best_score) if isinstance(runner_up, dict) else best_score
        score_margin = float(max(0.0, best_score - second_score))
        score_delta = float(best_score - best_threshold)
        confidence_raw = 0.5 + (0.25 * math.tanh(score_delta / self._direct_decision_score_margin_scale)) + (
            0.20 * math.tanh(score_margin / self._direct_decision_score_margin_scale)
        )
        confidence = float(
            clip(confidence_raw, self._direct_decision_conf_floor, self._direct_decision_conf_cap)
        )
        min_confidence_to_override = clip(
            safe_float(
                model_cfg.get(
                    "min_confidence_to_override",
                    self.direct_decision_cfg.get("min_confidence_to_override", 0.0),
                ),
                0.0,
            ),
            0.0,
            1.0,
        )
        min_score_delta_to_override = max(
            0.0,
            safe_float(
                model_cfg.get(
                    "min_score_delta_to_override",
                    self.direct_decision_cfg.get("min_score_delta_to_override", 0.0),
                ),
                0.0,
            ),
        )
        min_score_margin_to_override = max(
            0.0,
            safe_float(
                model_cfg.get(
                    "min_score_margin_to_override",
                    self.direct_decision_cfg.get("min_score_margin_to_override", 0.0),
                ),
                0.0,
            ),
        )
        route_scores = {
            lane: float(score)
            for lane, score in lane_scores_map.items()
            if math.isfinite(float(score))
        }
        if (
            selection_mode == "hybrid_fallback_router_lane"
            and (
                (not best_allow)
                or (best_score < best_threshold)
                or (confidence < min_confidence_to_override)
                or (score_delta < min_score_delta_to_override)
                or (score_margin < min_score_margin_to_override)
            )
        ):
            if track_counters:
                self._direct_decision_fallback_count += 1
            return None
        if (not best_allow) or (best_score < best_threshold):
            if track_counters:
                self._direct_decision_abstain_count += 1
            return {
                "abstained": True,
                "abstain_reason": "direct_decision_model_score_below_threshold",
                "route_decision": LANE_NO_TRADE,
                "route_confidence": float(confidence),
                "route_margin": float(score_margin),
                "route_scores": dict(route_scores),
                "route_reason": "direct_decision_model_no_trade",
                "selected_lane": "",
                "selected_variant_id": "",
                "decision_policy_score": float(best_score),
                "decision_policy_threshold": float(best_threshold),
                "decision_policy_score_delta": float(score_delta),
                "lane_candidate_count": int(len(scored_rows)),
                "lane_scores": [],
                "runner_up_within_lane": "",
                "lane_selection_reason": "direct_decision_model_no_trade",
                "chosen_family_id": "",
                "chosen_entry": None,
                "direct_decision_runner_up_variant_id": (
                    str(runner_up.get("variant_id", "")) if isinstance(runner_up, dict) else ""
                ),
            }

        if track_counters:
            self._direct_decision_selected_count += 1
        lane = str(best.get("lane", "") or "")
        lane_rows = [r for r in scored_rows if str(r.get("lane", "") or "") == lane]
        lane_scores = [
            {
                "variant_id": str(r.get("variant_id", "")),
                "lane_score": float(r.get("decision_policy_score", 0.0) or 0.0),
                "components": dict(r.get("decision_policy_components", {})),
                "inputs": {
                    "edge_points": float(safe_float(r.get("edge_points", 0.0), 0.0)),
                    "structural_score": float(safe_float(r.get("structural_score", 0.0), 0.0)),
                    "variant_quality_prior": float(
                        safe_float(
                            (
                                (r.get("decision_policy_stats", {}) or {})
                                if isinstance(r.get("decision_policy_stats", {}), dict)
                                else {}
                            ).get("variant_quality_prior", 0.0),
                            0.0,
                        )
                    ),
                    "lane_prior": float(
                        safe_float(
                            (
                                (r.get("decision_policy_stats", {}) or {})
                                if isinstance(r.get("decision_policy_stats", {}), dict)
                                else {}
                            ).get("lane_prior", 0.0),
                            0.0,
                        )
                    ),
                },
            }
            for r in lane_rows
        ]
        lane_runner_up = lane_rows[1] if len(lane_rows) > 1 else None
        return {
            "abstained": False,
            "abstain_reason": "",
            "route_decision": str(lane or LANE_NO_TRADE),
            "route_confidence": float(confidence),
            "route_margin": float(score_margin),
            "route_scores": dict(route_scores),
            "route_reason": "direct_decision_model_select",
            "selected_lane": str(lane),
            "selected_variant_id": str(best.get("variant_id", "") or ""),
            "decision_policy_score": float(best_score),
            "decision_policy_threshold": float(best_threshold),
            "decision_policy_score_delta": float(score_delta),
            "lane_candidate_count": int(len(lane_rows)),
            "lane_scores": lane_scores,
            "runner_up_within_lane": str(
                lane_runner_up.get("variant_id", "") if isinstance(lane_runner_up, dict) else ""
            ),
            "lane_selection_reason": "direct_decision_model_select",
            "chosen_family_id": str(best.get("family_id", "") or ""),
            "chosen_entry": dict(best),
            "direct_decision_runner_up_variant_id": (
                str(runner_up.get("variant_id", "")) if isinstance(runner_up, dict) else ""
            ),
        }

    def _select_with_router_lane_stack(
        self,
        *,
        lane_candidates: Dict[str, List[Dict[str, Any]]],
        default_session: str,
        timeframe_hint: str,
    ) -> Dict[str, Any]:
        route_result = self.router.route(
            lane_candidates=lane_candidates,
            session_name=str(default_session or ""),
            timeframe_hint=str(timeframe_hint or ""),
        )
        route_decision = str(route_result.get("route_decision", LANE_NO_TRADE) or LANE_NO_TRADE)
        route_conf = safe_float(route_result.get("route_confidence", 0.0), 0.0)
        route_margin = safe_float(route_result.get("route_margin", 0.0), 0.0)
        route_scores = (
            dict(route_result.get("route_scores", {}))
            if isinstance(route_result.get("route_scores", {}), dict)
            else {}
        )
        route_reason = str(route_result.get("route_reason", "") or "")
        lane_rows: List[Dict[str, Any]] = []
        lane_result: Dict[str, Any] = {
            "lane_candidate_count": 0,
            "lane_scores": [],
            "selected_variant_id": "",
            "lane_selection_reason": "",
        }
        chosen_entry = None
        runner_up_entry = None
        selected_variant_id = ""
        lane_selection_reason = ""
        if route_decision != LANE_NO_TRADE:
            lane_rows = lane_candidates.get(route_decision, [])
            lane_result = self.lane_selector.select(
                lane=route_decision,
                candidates=lane_rows,
            )
            chosen_entry = (
                lane_result.get("chosen_entry")
                if isinstance(lane_result.get("chosen_entry"), dict)
                else None
            )
            runner_up_entry = (
                lane_result.get("runner_up_entry")
                if isinstance(lane_result.get("runner_up_entry"), dict)
                else None
            )
            selected_variant_id = str(lane_result.get("selected_variant_id", "") or "")
            lane_selection_reason = str(lane_result.get("lane_selection_reason", "") or "")

            if (
                self.force_anchor_when_eligible
                and route_decision == LANE_LONG_REV
                and route_conf <= 0.05
            ):
                core_candidates = [
                    row for row in lane_rows if bool(row.get("is_core_family", False))
                ]
                if core_candidates:
                    core_candidates.sort(
                        key=lambda r: safe_float((r.get("lane_score", r.get("edge_points", 0.0))), 0.0),
                        reverse=True,
                    )
                    chosen_entry = dict(core_candidates[0])
                    selected_variant_id = str(
                        chosen_entry.get("selected_variant_id", chosen_entry.get("variant_id", ""))
                    )
                    lane_selection_reason = "force_core_anchor_when_eligible"
        return {
            "route_decision": route_decision,
            "route_confidence": float(route_conf),
            "route_margin": float(route_margin),
            "route_scores": dict(route_scores),
            "route_reason": str(route_reason),
            "lane_result": dict(lane_result),
            "chosen_entry": dict(chosen_entry) if isinstance(chosen_entry, dict) else None,
            "runner_up_entry": dict(runner_up_entry) if isinstance(runner_up_entry, dict) else None,
            "selected_variant_id": str(selected_variant_id),
            "lane_selection_reason": str(lane_selection_reason),
            "lane_rows": list(lane_rows),
        }

    def _maybe_override_baseline_with_direct(
        self,
        *,
        direct_result: Optional[Dict[str, Any]],
        baseline_result: Dict[str, Any],
        default_session: str,
        model_cfg: Optional[Dict[str, Any]] = None,
        track_counters: bool = True,
    ) -> Optional[Dict[str, Any]]:
        model_cfg = (
            dict(model_cfg)
            if isinstance(model_cfg, dict)
            else dict(self.direct_decision_model_cfg)
        )
        if not isinstance(direct_result, dict):
            return None
        if not isinstance(baseline_result, dict):
            return direct_result

        direct_route = str(direct_result.get("route_decision", LANE_NO_TRADE) or LANE_NO_TRADE)
        baseline_route = str(
            baseline_result.get("route_decision", LANE_NO_TRADE) or LANE_NO_TRADE
        )
        if direct_route == LANE_NO_TRADE:
            if track_counters:
                self._direct_decision_baseline_compare_fallback_count += 1
            return None

        allow_override_when_baseline_no_trade = bool(
            model_cfg.get(
                "allow_override_when_baseline_no_trade",
                self.direct_decision_cfg.get("allow_override_when_baseline_no_trade", False),
            )
        )
        min_confidence_to_override_no_trade = clip(
            safe_float(
                model_cfg.get(
                    "min_confidence_to_override_when_baseline_no_trade",
                    self.direct_decision_cfg.get(
                        "min_confidence_to_override_when_baseline_no_trade",
                        model_cfg.get(
                            "min_confidence_to_override",
                            self.direct_decision_cfg.get("min_confidence_to_override", 0.0),
                        ),
                    ),
                ),
                0.0,
            ),
            0.0,
            1.0,
        )
        min_score_delta_to_override_no_trade = max(
            0.0,
            safe_float(
                model_cfg.get(
                    "min_score_delta_to_override_when_baseline_no_trade",
                    self.direct_decision_cfg.get(
                        "min_score_delta_to_override_when_baseline_no_trade",
                        model_cfg.get(
                            "min_score_delta_to_override",
                            self.direct_decision_cfg.get("min_score_delta_to_override", 0.0),
                        ),
                    ),
                ),
                0.0,
            ),
        )
        min_score_margin_to_override_no_trade = max(
            0.0,
            safe_float(
                model_cfg.get(
                    "min_score_margin_to_override_when_baseline_no_trade",
                    self.direct_decision_cfg.get(
                        "min_score_margin_to_override_when_baseline_no_trade",
                        model_cfg.get(
                            "min_score_margin_to_override",
                            self.direct_decision_cfg.get("min_score_margin_to_override", 0.0),
                        ),
                    ),
                ),
                0.0,
            ),
        )
        min_baseline_score_advantage = max(
            0.0,
            safe_float(
                model_cfg.get(
                    "min_baseline_score_advantage_to_override",
                    self.direct_decision_cfg.get("min_baseline_score_advantage_to_override", 0.0),
                ),
                0.0,
            ),
        )
        min_baseline_score_delta_advantage = max(
            0.0,
            safe_float(
                model_cfg.get(
                    "min_baseline_score_delta_advantage_to_override",
                    self.direct_decision_cfg.get(
                        "min_baseline_score_delta_advantage_to_override",
                        0.0,
                    ),
                ),
                0.0,
            ),
        )
        direct_conf = float(safe_float(direct_result.get("route_confidence", 0.0), 0.0))
        direct_score = float(safe_float(direct_result.get("decision_policy_score", 0.0), 0.0))
        direct_threshold = float(
            safe_float(direct_result.get("decision_policy_threshold", 0.0), 0.0)
        )
        direct_score_delta = float(
            safe_float(
                direct_result.get("decision_policy_score_delta", direct_score - direct_threshold),
                direct_score - direct_threshold,
            )
        )
        direct_score_margin = float(safe_float(direct_result.get("route_margin", 0.0), 0.0))
        if baseline_route == LANE_NO_TRADE:
            if (
                (not allow_override_when_baseline_no_trade)
                or (direct_conf < min_confidence_to_override_no_trade)
                or (direct_score_delta < min_score_delta_to_override_no_trade)
                or (direct_score_margin < min_score_margin_to_override_no_trade)
            ):
                if track_counters:
                    self._direct_decision_baseline_compare_fallback_count += 1
                return None
            out = dict(direct_result)
            out["route_reason"] = "direct_decision_override_baseline_no_trade"
            out["baseline_compare_baseline_route"] = str(baseline_route)
            out["baseline_compare_baseline_selected_variant_id"] = ""
            out["baseline_compare_direct_score"] = float(direct_score)
            out["baseline_compare_direct_score_delta"] = float(direct_score_delta)
            out["baseline_compare_score_advantage"] = float(direct_score)
            out["baseline_compare_score_delta_advantage"] = float(direct_score_delta)
            return out

        baseline_entry = (
            dict(baseline_result.get("chosen_entry", {}))
            if isinstance(baseline_result.get("chosen_entry"), dict)
            else None
        )
        baseline_variant_id = str(baseline_result.get("selected_variant_id", "") or "")
        direct_variant_id = str(direct_result.get("selected_variant_id", "") or "")
        if baseline_entry is None:
            if track_counters:
                self._direct_decision_baseline_compare_fallback_count += 1
            return None
        if baseline_variant_id and direct_variant_id and baseline_variant_id == direct_variant_id:
            out = dict(direct_result)
            out["route_reason"] = "direct_decision_matches_baseline_variant"
            out["baseline_compare_baseline_route"] = str(baseline_route)
            out["baseline_compare_baseline_selected_variant_id"] = str(baseline_variant_id)
            out["baseline_compare_direct_score"] = float(direct_score)
            out["baseline_compare_direct_score_delta"] = float(direct_score_delta)
            out["baseline_compare_score_advantage"] = 0.0
            out["baseline_compare_score_delta_advantage"] = 0.0
            return out

        baseline_eval = self._evaluate_direct_decision_model(
            candidate_entry=baseline_entry,
            default_session=default_session,
            model_cfg=model_cfg,
        )
        baseline_score = float(safe_float(baseline_eval.get("score", 0.0), 0.0))
        baseline_threshold = float(safe_float(baseline_eval.get("threshold", 0.0), 0.0))
        baseline_score_delta = float(baseline_score - baseline_threshold)
        score_advantage = float(direct_score - baseline_score)
        score_delta_advantage = float(direct_score_delta - baseline_score_delta)
        if (
            score_advantage < min_baseline_score_advantage
            or score_delta_advantage < min_baseline_score_delta_advantage
        ):
            if track_counters:
                self._direct_decision_baseline_compare_fallback_count += 1
            return None
        out = dict(direct_result)
        out["route_reason"] = "direct_decision_override_baseline_variant"
        out["baseline_compare_baseline_route"] = str(baseline_route)
        out["baseline_compare_baseline_selected_variant_id"] = str(baseline_variant_id)
        out["baseline_compare_baseline_score"] = float(baseline_score)
        out["baseline_compare_baseline_score_delta"] = float(baseline_score_delta)
        out["baseline_compare_direct_score"] = float(direct_score)
        out["baseline_compare_direct_score_delta"] = float(direct_score_delta)
        out["baseline_compare_score_advantage"] = float(score_advantage)
        out["baseline_compare_score_delta_advantage"] = float(score_delta_advantage)
        return out

    def _resolve_entry_model_stats(
        self,
        *,
        chosen_entry: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], str]:
        variant_id = self._entry_model_variant_id(chosen_entry)
        lane = self._entry_model_lane(chosen_entry)
        min_variant = max(1, int(safe_float(self.calibrated_entry_model_cfg.get("min_variant_trades", 25), 25)))
        min_lane = max(1, int(safe_float(self.calibrated_entry_model_cfg.get("min_lane_trades", 120), 120)))
        variant_stats = self._entry_model_variant_stats
        lane_stats = self._entry_model_lane_stats
        global_stats = self._entry_model_global_stats
        v_stats = variant_stats.get(variant_id, {}) if variant_id else {}
        if isinstance(v_stats, dict) and int(safe_float(v_stats.get("n_trades", 0), 0)) >= min_variant:
            return v_stats, "variant"
        l_stats = lane_stats.get(lane, {}) if lane else {}
        if isinstance(l_stats, dict) and int(safe_float(l_stats.get("n_trades", 0), 0)) >= min_lane:
            return l_stats, "lane"
        if isinstance(global_stats, dict) and int(safe_float(global_stats.get("n_trades", 0), 0)) > 0:
            return global_stats, "global"
        return {}, "missing"

    def _evaluate_fallback_scope_guard(
        self,
        *,
        chosen_entry: Dict[str, Any],
        stats: Dict[str, Any],
        scope: str,
        model_cfg: Dict[str, Any],
    ) -> Tuple[bool, str, Dict[str, Any]]:
        scope_norm = str(scope or "").strip().lower()
        guard_enabled = bool(model_cfg.get("fallback_scope_guard_enabled", True))
        if not guard_enabled:
            return True, "", {"guard_enabled": False, "scope": scope_norm}
        guarded_scopes = (
            [str(v).strip().lower() for v in model_cfg.get("fallback_guard_scopes", [])]
            if isinstance(model_cfg.get("fallback_guard_scopes"), list)
            else ["lane", "global"]
        )
        guarded_scope_set = {
            v for v in guarded_scopes if v in {"variant", "lane", "global", "missing"}
        }
        if scope_norm not in guarded_scope_set:
            return True, "", {
                "guard_enabled": True,
                "scope": scope_norm,
                "guarded_scopes": list(sorted(guarded_scope_set)),
                "scope_guard_applied": False,
            }

        allow_global_scope = bool(model_cfg.get("fallback_allow_global_scope", False))
        ev_lcb = safe_float(stats.get("ev_lcb", 0.0), 0.0)
        quality_lcb = safe_float(stats.get("quality_lcb_score", 0.0), 0.0)
        p_win_lcb = safe_float(stats.get("p_win_lcb", 0.0), 0.0)
        worst_block = safe_float(stats.get("worst_block_avg_pnl", 0.0), 0.0)
        year_coverage = int(safe_float(stats.get("year_coverage", 0), 0))
        lane_inputs = (
            chosen_entry.get("lane_score_inputs", {})
            if isinstance(chosen_entry.get("lane_score_inputs"), dict)
            else {}
        )
        has_variant_quality_prior = "variant_quality_prior" in lane_inputs
        variant_quality_prior = safe_float(lane_inputs.get("variant_quality_prior", 0.0), 0.0)
        year_coverage_available = "year_coverage" in stats

        min_ev_lcb = safe_float(model_cfg.get("fallback_min_ev_lcb_points", -0.05), -0.05)
        min_quality_lcb = safe_float(model_cfg.get("fallback_min_quality_lcb_score", -0.02), -0.02)
        min_p_win_lcb = safe_float(model_cfg.get("fallback_min_p_win_lcb", 0.30), 0.30)
        min_worst_block = safe_float(model_cfg.get("fallback_min_worst_block_avg_pnl", -60.0), -60.0)
        min_year_coverage = int(max(0, safe_float(model_cfg.get("fallback_min_year_coverage", 5), 5)))
        min_variant_quality_prior = safe_float(
            model_cfg.get("fallback_min_variant_quality_prior", 0.0),
            0.0,
        )

        guard_meta = {
            "guard_enabled": True,
            "scope": scope_norm,
            "guarded_scopes": list(sorted(guarded_scope_set)),
            "scope_guard_applied": True,
            "allow_global_scope": bool(allow_global_scope),
            "ev_lcb": float(ev_lcb),
            "quality_lcb_score": float(quality_lcb),
            "p_win_lcb": float(p_win_lcb),
            "worst_block_avg_pnl": float(worst_block),
            "year_coverage": int(year_coverage),
            "year_coverage_available": bool(year_coverage_available),
            "variant_quality_prior": float(variant_quality_prior),
            "variant_quality_prior_available": bool(has_variant_quality_prior),
            "min_ev_lcb_points": float(min_ev_lcb),
            "min_quality_lcb_score": float(min_quality_lcb),
            "min_p_win_lcb": float(min_p_win_lcb),
            "min_worst_block_avg_pnl": float(min_worst_block),
            "min_year_coverage": int(min_year_coverage),
            "min_variant_quality_prior": float(min_variant_quality_prior),
        }

        if (scope_norm == "global") and (not allow_global_scope):
            return False, "calibrated_entry_model_global_fallback_disabled", guard_meta
        if ev_lcb < min_ev_lcb:
            return False, "calibrated_entry_model_fallback_ev_lcb_below_min", guard_meta
        if quality_lcb < min_quality_lcb:
            return False, "calibrated_entry_model_fallback_quality_lcb_below_min", guard_meta
        if p_win_lcb < min_p_win_lcb:
            return False, "calibrated_entry_model_fallback_p_win_lcb_below_min", guard_meta
        if worst_block < min_worst_block:
            return False, "calibrated_entry_model_fallback_worst_block_below_min", guard_meta
        if year_coverage_available and year_coverage < min_year_coverage:
            return False, "calibrated_entry_model_fallback_year_coverage_below_min", guard_meta
        if has_variant_quality_prior and variant_quality_prior < min_variant_quality_prior:
            return False, "calibrated_entry_model_fallback_variant_prior_below_min", guard_meta
        return True, "", guard_meta

    def _evaluate_calibrated_entry_model(
        self,
        *,
        chosen_entry: Dict[str, Any],
        route_confidence: float,
    ) -> Dict[str, Any]:
        model_cfg = self.calibrated_entry_model_cfg
        if not bool(model_cfg.get("enabled", False)):
            return {
                "enabled": False,
                "allow": True,
                "tier": "disabled",
                "reason": "calibrated_entry_model_disabled",
                "score": 0.0,
                "threshold": 0.0,
                "scope": "disabled",
                "stats": {},
                "components": {},
            }

        self._entry_model_eval_count += 1
        stats, scope = self._resolve_entry_model_stats(chosen_entry=chosen_entry)
        scope_norm = str(scope or "").strip().lower()
        self._entry_model_scope_counts[str(scope_norm)] += 1
        if scope_norm == "missing":
            self._entry_model_missing_stats_count += 1
        allowed_scopes = (
            [str(v).strip().lower() for v in model_cfg.get("allowed_scopes", [])]
            if isinstance(model_cfg.get("allowed_scopes"), list)
            else ["variant", "lane", "global", "missing"]
        )
        allowed_scope_set = {
            v for v in allowed_scopes if v in {"variant", "lane", "global", "missing"}
        }
        if (not allowed_scope_set) or (scope_norm not in allowed_scope_set):
            self._entry_model_reject_count += 1
            reject_reason = f"calibrated_entry_model_scope_not_allowed:{scope_norm or 'unknown'}"
            self._entry_model_reject_reason_counts[reject_reason] += 1
            return {
                "enabled": True,
                "allow": False,
                "tier": "reject",
                "reason": reject_reason,
                "score": 0.0,
                "threshold": float(safe_float(model_cfg.get("selected_threshold", 0.0), 0.0)),
                "threshold_base": float(safe_float(model_cfg.get("selected_threshold", 0.0), 0.0)),
                "threshold_scope_offset": 0.0,
                "scope": str(scope_norm),
                "stats": dict(stats) if isinstance(stats, dict) else {},
                "components": {},
                "fallback_scope_guard": {
                    "scope": str(scope_norm),
                    "allowed_scopes": sorted(list(allowed_scope_set)),
                    "scope_allowed": False,
                    "scope_guard_applied": True,
                },
            }
        if scope_norm == "missing":
            allow_on_missing = bool(model_cfg.get("allow_on_missing_stats", True))
            if not allow_on_missing:
                self._entry_model_reject_count += 1
                self._entry_model_reject_reason_counts["calibrated_entry_model_missing_stats"] += 1
            return {
                "enabled": True,
                "allow": bool(allow_on_missing),
                "tier": "full" if allow_on_missing else "reject",
                "reason": "calibrated_entry_model_missing_stats",
                "score": 0.0,
                "threshold": float(safe_float(model_cfg.get("selected_threshold", 0.0), 0.0)),
                "scope": "missing",
                "stats": {},
                "components": {},
            }

        fallback_allow, fallback_reason, fallback_guard_meta = self._evaluate_fallback_scope_guard(
            chosen_entry=chosen_entry,
            stats=stats,
            scope=scope,
            model_cfg=model_cfg,
        )
        if not fallback_allow:
            self._entry_model_reject_count += 1
            self._entry_model_reject_reason_counts[str(fallback_reason)] += 1
            return {
                "enabled": True,
                "allow": False,
                "tier": "reject",
                "reason": str(fallback_reason),
                "score": 0.0,
                "threshold": float(safe_float(model_cfg.get("selected_threshold", 0.0), 0.0)),
                "threshold_base": float(safe_float(model_cfg.get("selected_threshold", 0.0), 0.0)),
                "threshold_scope_offset": 0.0,
                "scope": str(scope),
                "stats": dict(stats),
                "components": {},
                "fallback_scope_guard": dict(fallback_guard_meta),
            }

        score_cfg = model_cfg.get("score_components", {}) if isinstance(model_cfg.get("score_components"), dict) else {}
        quality_lcb = safe_float(stats.get("quality_lcb_score", 0.0), 0.0)
        route_center = safe_float(score_cfg.get("route_confidence_center", 0.05), 0.05)
        edge_scale = max(1e-9, safe_float(score_cfg.get("edge_scale_points", 0.40), 0.40))
        struct_scale = max(1e-9, safe_float(score_cfg.get("structural_scale", 0.80), 0.80))
        w_quality = safe_float(score_cfg.get("weight_quality_lcb", 0.65), 0.65)
        w_route = safe_float(score_cfg.get("weight_route_confidence", 0.20), 0.20)
        w_edge = safe_float(score_cfg.get("weight_edge_points", 0.10), 0.10)
        w_struct = safe_float(score_cfg.get("weight_structural_score", 0.05), 0.05)
        w_pf = safe_float(score_cfg.get("weight_profit_factor_component", 0.0), 0.0)
        w_year = safe_float(score_cfg.get("weight_year_coverage_component", 0.0), 0.0)
        w_loss = safe_float(score_cfg.get("weight_loss_share_penalty", 0.12), 0.12)
        w_stop = safe_float(score_cfg.get("weight_stop_like_share_penalty", 0.08), 0.08)
        w_drawdown = safe_float(score_cfg.get("weight_drawdown_penalty", 0.06), 0.06)
        w_worst_block = safe_float(score_cfg.get("weight_worst_block_penalty", 0.08), 0.08)
        profit_factor_center = safe_float(score_cfg.get("profit_factor_center", 1.10), 1.10)
        profit_factor_scale = max(1e-9, safe_float(score_cfg.get("profit_factor_scale", 0.35), 0.35))
        year_coverage_full = max(1.0, safe_float(score_cfg.get("year_coverage_full_years", 8.0), 8.0))
        loss_center = safe_float(score_cfg.get("loss_share_center", 0.52), 0.52)
        loss_scale = max(1e-9, safe_float(score_cfg.get("loss_share_scale", 0.22), 0.22))
        stop_center = safe_float(score_cfg.get("stop_like_share_center", 0.62), 0.62)
        stop_scale = max(1e-9, safe_float(score_cfg.get("stop_like_share_scale", 0.25), 0.25))
        drawdown_scale = max(1e-9, safe_float(score_cfg.get("drawdown_scale", 6.0), 6.0))
        shape_penalty_scale = max(1e-9, safe_float(score_cfg.get("shape_penalty_scale", 1.0), 1.0))
        shape_penalty_cap = max(0.0, safe_float(score_cfg.get("shape_penalty_cap", 2.0), 2.0))
        worst_block_scale = max(
            1e-9,
            safe_float(score_cfg.get("worst_block_scale_points", 3.0), 3.0),
        )
        w_shape = safe_float(score_cfg.get("weight_shape_penalty_component", 0.0), 0.0)
        edge_points = safe_float(
            chosen_entry.get("edge_points", chosen_entry.get("lane_score", chosen_entry.get("selection_score", 0.0))),
            0.0,
        )
        structural = safe_float(chosen_entry.get("structural_score", 0.0), 0.0)
        loss_share = clip(safe_float(stats.get("loss_share", loss_center), loss_center), 0.0, 1.0)
        stop_like_share = clip(
            safe_float(stats.get("stop_like_share", stop_center), stop_center),
            0.0,
            1.0,
        )
        profit_factor = max(0.0, safe_float(stats.get("profit_factor", profit_factor_center), profit_factor_center))
        year_coverage = max(0.0, safe_float(stats.get("year_coverage", 0.0), 0.0))
        year_coverage_ratio = clip(safe_div(year_coverage, year_coverage_full, 0.0), 0.0, 1.0)
        drawdown_norm = max(0.0, safe_float(stats.get("drawdown_norm", 0.0), 0.0))
        worst_block_avg_pnl = safe_float(stats.get("worst_block_avg_pnl", 0.0), 0.0)
        loss_excess = max(0.0, loss_share - loss_center)
        stop_excess = max(0.0, stop_like_share - stop_center)
        worst_block_shortfall = max(0.0, -worst_block_avg_pnl)
        shape_eval = self._evaluate_shape_penalty_model(
            chosen_entry=chosen_entry,
            model_cfg=(
                model_cfg.get("shape_penalty_model", {})
                if isinstance(model_cfg.get("shape_penalty_model"), dict)
                else {}
            ),
        )
        shape_strength = min(
            float(shape_penalty_cap),
            max(0.0, safe_float(shape_eval.get("strength", 0.0), 0.0)),
        )
        components = {
            "quality_lcb_component": float(w_quality * quality_lcb),
            "route_confidence_component": float(w_route * (safe_float(route_confidence, 0.0) - route_center)),
            "edge_points_component": float(w_edge * math.tanh(edge_points / edge_scale)),
            "structural_component": float(w_struct * math.tanh(structural / struct_scale)),
            "profit_factor_component": float(
                w_pf * math.tanh((profit_factor - profit_factor_center) / profit_factor_scale)
            ),
            "year_coverage_component": float(w_year * ((2.0 * year_coverage_ratio) - 1.0)),
            "loss_share_penalty_component": float(-w_loss * math.tanh(loss_excess / loss_scale)),
            "stop_like_share_penalty_component": float(-w_stop * math.tanh(stop_excess / stop_scale)),
            "drawdown_penalty_component": float(-w_drawdown * math.tanh(drawdown_norm / drawdown_scale)),
            "worst_block_penalty_component": float(
                -w_worst_block * math.tanh(worst_block_shortfall / worst_block_scale)
            ),
            "shape_penalty_component": float(
                -w_shape * math.tanh(shape_strength / shape_penalty_scale)
            ),
        }
        score = float(sum(float(v or 0.0) for v in components.values()))
        threshold_base = float(safe_float(model_cfg.get("selected_threshold", 0.0), 0.0))
        scope_offsets_cfg = (
            model_cfg.get("scope_threshold_offsets", {})
            if isinstance(model_cfg.get("scope_threshold_offsets"), dict)
            else {}
        )
        scope_offset = float(
            safe_float(
                scope_offsets_cfg.get(
                    str(scope),
                    scope_offsets_cfg.get("default", 0.0),
                ),
                0.0,
            )
        )
        threshold = float(threshold_base + scope_offset)
        conservative_buffer = max(0.0, safe_float(model_cfg.get("conservative_buffer", 0.035), 0.035))
        allow = bool(score >= threshold)
        if not allow:
            self._entry_model_reject_count += 1
            self._entry_model_reject_reason_counts["calibrated_entry_model_score_below_threshold"] += 1
        tier = "reject"
        if allow:
            tier = "conservative" if score < (threshold + conservative_buffer) else "full"
        return {
            "enabled": True,
            "allow": bool(allow),
            "tier": str(tier),
            "reason": "calibrated_entry_model_pass" if allow else "calibrated_entry_model_score_below_threshold",
            "score": float(score),
            "threshold": float(threshold),
            "threshold_base": float(threshold_base),
            "threshold_scope_offset": float(scope_offset),
            "scope": str(scope),
            "stats": {
                "n_trades": int(safe_float(stats.get("n_trades", 0), 0)),
                "p_win_lcb": float(safe_float(stats.get("p_win_lcb", 0.0), 0.0)),
                "ev_lcb": float(safe_float(stats.get("ev_lcb", 0.0), 0.0)),
                "quality_lcb_score": float(quality_lcb),
                "profit_factor": float(profit_factor),
                "loss_share": float(loss_share),
                "stop_like_share": float(stop_like_share),
                "drawdown_norm": float(drawdown_norm),
                "worst_block_avg_pnl": float(worst_block_avg_pnl),
                "year_coverage": int(safe_float(stats.get("year_coverage", 0), 0)),
                "first_year": str(stats.get("first_year", "") or ""),
                "last_year": str(stats.get("last_year", "") or ""),
                "shape_penalty_strength": float(shape_strength),
                "shape_rule_hit_count": int(safe_float(shape_eval.get("match_count", 0), 0)),
                "shape_scope_key": str(shape_eval.get("scope_key", "") or ""),
            },
            "components": components,
            "fallback_scope_guard": dict(fallback_guard_meta),
        }

    def _evaluate_execution_policy(
        self,
        *,
        chosen_entry: Dict[str, Any],
        route_confidence: float,
    ) -> Dict[str, Any]:
        if not isinstance(chosen_entry, dict):
            return {
                "enabled": True,
                "allow": False,
                "tier": "reject",
                "reason": "invalid_chosen_entry",
                "enforce_veto": True,
                "quality_score": 0.0,
                "hard_limit_triggered": True,
                "hard_limit_reason": "invalid_chosen_entry",
                "components": {},
                "inputs": {},
                "entry_model_enabled": bool(self.calibrated_entry_model_enabled),
                "entry_model_allow": False,
                "entry_model_tier": "reject",
                "entry_model_reason": "invalid_chosen_entry",
                "entry_model_score": 0.0,
                "entry_model_threshold": 0.0,
                "entry_model_threshold_base": 0.0,
                "entry_model_threshold_scope_offset": 0.0,
                "entry_model_scope": "invalid",
                "entry_model_stats": {},
                "entry_model_fallback_scope_guard": {},
            }
        base_policy_enabled = bool(self.execution_policy_cfg.get("base_policy_enabled", False))
        entry_model_enabled = bool(self.execution_policy_cfg.get("calibrated_entry_model_enabled", False))
        if not base_policy_enabled and not entry_model_enabled:
            return {
                "enabled": False,
                "allow": True,
                "tier": "disabled",
                "reason": "execution_policy_disabled",
                "enforce_veto": False,
                "quality_score": 1.0,
                "hard_limit_triggered": False,
                "hard_limit_reason": "",
                "components": {},
                "inputs": {},
                "entry_model_enabled": False,
                "entry_model_allow": True,
                "entry_model_tier": "disabled",
                "entry_model_reason": "calibrated_entry_model_disabled",
                "entry_model_score": 0.0,
                "entry_model_threshold": 0.0,
                "entry_model_threshold_base": 0.0,
                "entry_model_threshold_scope_offset": 0.0,
                "entry_model_scope": "disabled",
                "entry_model_stats": {},
                "entry_model_fallback_scope_guard": {},
            }

        inputs = self._execution_policy_feature_vector(
            chosen_entry=chosen_entry,
            route_confidence=route_confidence,
        )
        base_policy_enforce_veto = bool(self.execution_policy_cfg.get("enforce_veto", True))
        base_allow = True
        base_tier = "full"
        base_reason = "base_policy_disabled"
        base_quality_score = 1.0
        base_hard_limit_triggered = False
        base_hard_limit_reason = ""
        base_components: Dict[str, Any] = {}

        hard_limits = (
            self.execution_policy_cfg.get("hard_limits", {})
            if isinstance(self.execution_policy_cfg.get("hard_limits"), dict)
            else {}
        )

        hard_limit_checks = (
            ("min_route_confidence", "route_confidence", "below_min"),
            ("min_edge_points", "edge_points", "below_min"),
            ("min_structural_score", "structural_score", "below_min"),
            ("min_lane_score", "lane_score", "below_min"),
            ("min_variant_quality_prior", "variant_quality_prior", "below_min"),
            ("max_loss_share", "loss_share", "above_max"),
            ("max_stop_like_share", "stop_like_share", "above_max"),
        )
        if base_policy_enabled:
            for limit_key, input_key, mode in hard_limit_checks:
                bound = self._cfg_float(hard_limits, limit_key, None)
                if bound is None:
                    continue
                observed = float(inputs.get(input_key, 0.0))
                if mode == "below_min" and observed < float(bound):
                    base_allow = False
                    base_tier = "reject"
                    base_reason = f"{input_key}_below_min"
                    base_hard_limit_triggered = True
                    base_hard_limit_reason = f"{observed:.4f} < {float(bound):.4f}"
                    base_quality_score = 0.0
                    break
                if mode == "above_max" and observed > float(bound):
                    base_allow = False
                    base_tier = "reject"
                    base_reason = f"{input_key}_above_max"
                    base_hard_limit_triggered = True
                    base_hard_limit_reason = f"{observed:.4f} > {float(bound):.4f}"
                    base_quality_score = 0.0
                    break

        if base_policy_enabled and base_allow:
            quality_cfg = (
                self.execution_policy_cfg.get("quality", {})
                if isinstance(self.execution_policy_cfg.get("quality"), dict)
                else {}
            )
            ranges_cfg = quality_cfg.get("ranges", {}) if isinstance(quality_cfg.get("ranges"), dict) else {}
            weights_cfg = quality_cfg.get("weights", {}) if isinstance(quality_cfg.get("weights"), dict) else {}
            weight_total = float(self._cfg_float(quality_cfg, "weight_total", 1.0) or 1.0)
            if weight_total <= 0.0:
                weight_total = 1.0
            w_route = float(self._cfg_float(weights_cfg, "route_confidence", 0.0) or 0.0)
            w_edge = float(self._cfg_float(weights_cfg, "edge_points", 0.0) or 0.0)
            w_lane = float(self._cfg_float(weights_cfg, "lane_score", 0.0) or 0.0)
            w_struct = float(self._cfg_float(weights_cfg, "structural_score", 0.0) or 0.0)
            w_prior = float(self._cfg_float(weights_cfg, "variant_quality_prior", 0.0) or 0.0)
            w_loss = float(self._cfg_float(weights_cfg, "loss_quality", 0.0) or 0.0)
            w_stop = float(self._cfg_float(weights_cfg, "stop_quality", 0.0) or 0.0)

            def _r(name: str) -> Dict[str, float]:
                raw = ranges_cfg.get(name, {}) if isinstance(ranges_cfg.get(name), dict) else {}
                lo = self._cfg_float(raw, "min", 0.0)
                hi = self._cfg_float(raw, "max", 1.0)
                if lo is None:
                    lo = 0.0
                if hi is None or hi <= lo:
                    hi = lo + 1e-9
                return {"min": float(lo), "max": float(hi)}

            route_r = _r("route_confidence")
            edge_r = _r("edge_points")
            lane_r = _r("lane_score")
            struct_r = _r("structural_score")
            prior_r = _r("variant_quality_prior")
            loss_r = _r("loss_share")
            stop_r = _r("stop_like_share")
            normalized = {
                "route_confidence": self._bounded_unit(inputs["route_confidence"], route_r["min"], route_r["max"]),
                "edge_points": self._bounded_unit(inputs["edge_points"], edge_r["min"], edge_r["max"]),
                "lane_score": self._bounded_unit(inputs["lane_score"], lane_r["min"], lane_r["max"]),
                "structural_score": self._bounded_unit(inputs["structural_score"], struct_r["min"], struct_r["max"]),
                "variant_quality_prior": self._bounded_unit(inputs["variant_quality_prior"], prior_r["min"], prior_r["max"]),
                "loss_quality": self._bounded_unit(inputs["loss_share"], loss_r["min"], loss_r["max"], invert=True),
                "stop_quality": self._bounded_unit(inputs["stop_like_share"], stop_r["min"], stop_r["max"], invert=True),
            }

            base_components = {
                "route_confidence_component": float(normalized["route_confidence"] * w_route),
                "edge_points_component": float(normalized["edge_points"] * w_edge),
                "lane_score_component": float(normalized["lane_score"] * w_lane),
                "structural_score_component": float(normalized["structural_score"] * w_struct),
                "variant_quality_prior_component": float(normalized["variant_quality_prior"] * w_prior),
                "loss_quality_component": float(normalized["loss_quality"] * w_loss),
                "stop_quality_component": float(normalized["stop_quality"] * w_stop),
            }
            weighted_total = float(sum(float(v or 0.0) for v in base_components.values()))
            base_quality_score = float(weighted_total / weight_total) if weight_total > 0.0 else 0.0

            reject_floor = float(self._cfg_float(quality_cfg, "reject_quality_score_below", 0.30) or 0.30)
            conservative_floor = float(self._cfg_float(quality_cfg, "conservative_quality_score_below", 0.48) or 0.48)
            if base_quality_score < reject_floor:
                base_allow = False
                base_tier = "reject"
                base_reason = "quality_score_below_reject_floor"
            else:
                base_tier = "conservative" if base_quality_score < conservative_floor else "full"
                base_reason = "policy_allow"

        entry_model_result = self._evaluate_calibrated_entry_model(
            chosen_entry=chosen_entry,
            route_confidence=route_confidence,
        )
        entry_model_allow = bool(entry_model_result.get("allow", True))
        entry_model_tier = str(entry_model_result.get("tier", "disabled") or "disabled")
        entry_model_reason = str(entry_model_result.get("reason", "calibrated_entry_model_disabled") or "calibrated_entry_model_disabled")
        entry_model_enforce_veto = bool(
            (
                self.execution_policy_cfg.get("calibrated_entry_model", {})
                if isinstance(self.execution_policy_cfg.get("calibrated_entry_model"), dict)
                else {}
            ).get("enforce_veto", True)
        )

        if base_policy_enabled and (not base_allow):
            combined_enforce_veto = bool(base_policy_enforce_veto) or bool(
                entry_model_enabled and (not entry_model_allow) and entry_model_enforce_veto
            )
            combined_reason = (
                str(entry_model_reason)
                if (entry_model_enabled and (not entry_model_allow) and entry_model_enforce_veto)
                else str(base_reason)
            )
            combined_components = dict(base_components)
            for key, val in (entry_model_result.get("components", {}) if isinstance(entry_model_result.get("components"), dict) else {}).items():
                combined_components[f"entry_model_{str(key)}"] = float(safe_float(val, 0.0))
            return {
                "enabled": True,
                "allow": False,
                "tier": "reject",
                "reason": str(combined_reason),
                "enforce_veto": bool(combined_enforce_veto),
                "quality_score": float(base_quality_score),
                "hard_limit_triggered": bool(base_hard_limit_triggered),
                "hard_limit_reason": str(base_hard_limit_reason),
                "components": combined_components,
                "inputs": inputs,
                "entry_model_enabled": bool(entry_model_enabled),
                "entry_model_allow": bool(entry_model_allow),
                "entry_model_tier": str(entry_model_tier),
                "entry_model_reason": str(entry_model_reason),
                "entry_model_score": float(safe_float(entry_model_result.get("score", 0.0), 0.0)),
                "entry_model_threshold": float(safe_float(entry_model_result.get("threshold", 0.0), 0.0)),
                "entry_model_threshold_base": float(
                    safe_float(entry_model_result.get("threshold_base", 0.0), 0.0)
                ),
                "entry_model_threshold_scope_offset": float(
                    safe_float(entry_model_result.get("threshold_scope_offset", 0.0), 0.0)
                ),
                "entry_model_scope": str(entry_model_result.get("scope", "")),
                "entry_model_stats": dict(entry_model_result.get("stats", {}) if isinstance(entry_model_result.get("stats"), dict) else {}),
                "entry_model_components": dict(entry_model_result.get("components", {}) if isinstance(entry_model_result.get("components"), dict) else {}),
                "entry_model_fallback_scope_guard": dict(
                    entry_model_result.get("fallback_scope_guard", {})
                    if isinstance(entry_model_result.get("fallback_scope_guard"), dict)
                    else {}
                ),
            }

        if entry_model_enabled and (not entry_model_allow):
            combined_components = dict(base_components)
            for key, val in (entry_model_result.get("components", {}) if isinstance(entry_model_result.get("components"), dict) else {}).items():
                combined_components[f"entry_model_{str(key)}"] = float(safe_float(val, 0.0))
            return {
                "enabled": True,
                "allow": False,
                "tier": "reject",
                "reason": str(entry_model_reason),
                "enforce_veto": bool(entry_model_enforce_veto),
                "quality_score": float(base_quality_score),
                "hard_limit_triggered": bool(base_hard_limit_triggered),
                "hard_limit_reason": str(base_hard_limit_reason),
                "components": combined_components,
                "inputs": inputs,
                "entry_model_enabled": True,
                "entry_model_allow": False,
                "entry_model_tier": str(entry_model_tier),
                "entry_model_reason": str(entry_model_reason),
                "entry_model_score": float(safe_float(entry_model_result.get("score", 0.0), 0.0)),
                "entry_model_threshold": float(safe_float(entry_model_result.get("threshold", 0.0), 0.0)),
                "entry_model_threshold_base": float(
                    safe_float(entry_model_result.get("threshold_base", 0.0), 0.0)
                ),
                "entry_model_threshold_scope_offset": float(
                    safe_float(entry_model_result.get("threshold_scope_offset", 0.0), 0.0)
                ),
                "entry_model_scope": str(entry_model_result.get("scope", "")),
                "entry_model_stats": dict(entry_model_result.get("stats", {}) if isinstance(entry_model_result.get("stats"), dict) else {}),
                "entry_model_components": dict(entry_model_result.get("components", {}) if isinstance(entry_model_result.get("components"), dict) else {}),
                "entry_model_fallback_scope_guard": dict(
                    entry_model_result.get("fallback_scope_guard", {})
                    if isinstance(entry_model_result.get("fallback_scope_guard"), dict)
                    else {}
                ),
            }

        final_tier = "full"
        if base_tier == "conservative" or entry_model_tier == "conservative":
            final_tier = "conservative"
        combined_components = dict(base_components)
        for key, val in (entry_model_result.get("components", {}) if isinstance(entry_model_result.get("components"), dict) else {}).items():
            combined_components[f"entry_model_{str(key)}"] = float(safe_float(val, 0.0))
        return {
            "enabled": True,
            "allow": True,
            "tier": str(final_tier),
            "reason": "policy_allow",
            "enforce_veto": False,
            "quality_score": float(base_quality_score),
            "hard_limit_triggered": bool(base_hard_limit_triggered),
            "hard_limit_reason": str(base_hard_limit_reason),
            "components": combined_components,
            "inputs": inputs,
            "entry_model_enabled": bool(entry_model_enabled),
            "entry_model_allow": bool(entry_model_allow),
            "entry_model_tier": str(entry_model_tier),
            "entry_model_reason": str(entry_model_reason),
            "entry_model_score": float(safe_float(entry_model_result.get("score", 0.0), 0.0)),
            "entry_model_threshold": float(safe_float(entry_model_result.get("threshold", 0.0), 0.0)),
            "entry_model_threshold_base": float(
                safe_float(entry_model_result.get("threshold_base", 0.0), 0.0)
            ),
            "entry_model_threshold_scope_offset": float(
                safe_float(entry_model_result.get("threshold_scope_offset", 0.0), 0.0)
            ),
            "entry_model_scope": str(entry_model_result.get("scope", "")),
            "entry_model_stats": dict(entry_model_result.get("stats", {}) if isinstance(entry_model_result.get("stats"), dict) else {}),
            "entry_model_components": dict(entry_model_result.get("components", {}) if isinstance(entry_model_result.get("components"), dict) else {}),
            "entry_model_fallback_scope_guard": dict(
                entry_model_result.get("fallback_scope_guard", {})
                if isinstance(entry_model_result.get("fallback_scope_guard"), dict)
                else {}
            ),
        }

    def select_route_and_variant(
        self,
        *,
        feasible_candidates: List[Dict[str, Any]],
        default_session: str,
        context_inputs: Optional[Dict[str, Any]] = None,
        current_time: Optional[Any] = None,
    ) -> Dict[str, Any]:
        self._runtime_invocations += 1
        ctx = context_inputs if isinstance(context_inputs, dict) else {}
        decision_ts = (
            str(pd_ts.isoformat())
            if (pd_ts := (None if current_time is None else _to_timestamp(current_time))) is not None
            else dt.datetime.now(NY_TZ).isoformat()
        )

        candidate_rows: List[Dict[str, Any]] = []
        decision_excluded_by_runtime_filter = 0
        for entry in feasible_candidates:
            if not isinstance(entry, dict):
                continue
            lane = self._candidate_lane(entry)
            if not lane:
                continue
            family_id = self._candidate_family_id(entry, default_session=default_session)
            variant_id = self._candidate_variant_id(entry)
            row = dict(entry)
            cand = row.get("cand", {}) if isinstance(row.get("cand"), dict) else {}
            row["lane"] = lane
            row["family_id"] = family_id
            row["variant_id"] = variant_id
            row["session"] = str(row.get("session", default_session) or default_session).strip()
            row["timeframe"] = str(
                row.get("timeframe")
                or cand.get("timeframe")
                or ""
            ).strip()
            row["strategy_type"] = str(
                row.get("strategy_type")
                or cand.get("strategy_type")
                or entry.get("strategy_type")
                or ""
            ).strip()
            row["sub_strategy"] = str(
                row.get("sub_strategy")
                or cand.get("sub_strategy")
                or entry.get("sub_strategy")
                or ""
            ).strip()
            row["side_considered"] = str(
                row.get("side_considered")
                or cand.get("side_considered")
                or lane_to_side(lane)
                or ""
            ).strip().lower()
            row["ctx_hour_et"] = (
                row.get("ctx_hour_et")
                if row.get("ctx_hour_et") is not None
                else cand.get("ctx_hour_et", cand.get("hour_et", ctx.get("hour_et")))
            )
            row["ctx_volatility_regime"] = row.get(
                "ctx_volatility_regime",
                cand.get("ctx_volatility_regime", cand.get("volatility_regime", ctx.get("volatility_regime"))),
            )
            row["ctx_chop_trend_regime"] = row.get(
                "ctx_chop_trend_regime",
                cand.get("ctx_chop_trend_regime", cand.get("chop_trend_regime", ctx.get("chop_trend_regime"))),
            )
            row["ctx_compression_expansion_regime"] = row.get(
                "ctx_compression_expansion_regime",
                cand.get(
                    "ctx_compression_expansion_regime",
                    cand.get("compression_expansion_regime", ctx.get("compression_expansion_regime")),
                ),
            )
            row["ctx_confidence_band"] = row.get(
                "ctx_confidence_band",
                cand.get("ctx_confidence_band", cand.get("confidence_band", ctx.get("confidence_band"))),
            )
            row["ctx_rvol_liquidity_state"] = row.get(
                "ctx_rvol_liquidity_state",
                cand.get("ctx_rvol_liquidity_state", cand.get("rvol_liquidity_state", ctx.get("rvol_liquidity_state"))),
            )
            row["ctx_price_location"] = row.get(
                "ctx_price_location",
                cand.get("ctx_price_location", cand.get("price_location", ctx.get("price_location"))),
            )
            row["ctx_session_substate"] = str(
                row.get("ctx_session_substate")
                or cand.get("ctx_session_substate")
                or cand.get("session_substate")
                or ctx.get("session_substate")
                or self._entry_model_derive_session_substate(
                    session_name=str(row.get("session", default_session) or default_session),
                    hour_value=row.get("ctx_hour_et"),
                )
                or ""
            ).strip().lower()
            row["is_core_family"] = bool(
                self._candidate_is_core_family(
                    entry=entry,
                    family_id=family_id,
                    variant_id=variant_id,
                    default_session=default_session,
                )
            )
            row["is_satellite_family"] = not bool(row["is_core_family"])
            excluded_reason = self._candidate_excluded_by_runtime_filters(
                family_id=family_id,
                variant_id=variant_id,
            )
            if excluded_reason:
                decision_excluded_by_runtime_filter += 1
                if excluded_reason == "excluded_family_id":
                    self._runtime_excluded_family_count += 1
                elif excluded_reason == "excluded_variant_pattern":
                    self._runtime_excluded_variant_pattern_count += 1
                continue
            candidate_rows.append(row)

        filtered_rows = self._mode_filter(candidate_rows=candidate_rows)
        candidate_count_after_mode_filter = int(len(filtered_rows))
        book_gate_meta: Dict[str, Any] = {
            "enabled": False,
            "selected_book": "",
            "default_book": "",
            "matched_scope": "",
            "matched_bucket": "",
            "reason": "",
            "fallback_to_default": False,
            "candidate_count_excluded": 0,
            "stats_recorded": False,
        }
        if not filtered_rows:
            self._record_book_gate_selection(book_gate_meta)
            self._record_choice_row(
                {
                    "decision_timestamp": decision_ts,
                    "route_decision": LANE_NO_TRADE,
                    "choice_reason": "no_candidates_after_runtime_mode_filter",
                    "runtime_mode": self.runtime_mode,
                    "candidate_count_raw": int(len(candidate_rows)),
                    "candidate_count_after_mode_filter": int(candidate_count_after_mode_filter),
                    "candidate_count_excluded_runtime_filter": int(decision_excluded_by_runtime_filter),
                }
            )
            return self._abstain_result(
                reason="no_candidates_after_runtime_mode_filter",
                feasible_rows=[],
                route_decision=LANE_NO_TRADE,
                route_confidence=0.0,
                route_margin=0.0,
                route_scores={LANE_NO_TRADE: 1.0},
            )

        filtered_rows, book_gate_meta = self._apply_book_gate(
            candidate_rows=filtered_rows,
            context_inputs=ctx,
            default_session=str(default_session or ""),
            current_time=current_time,
        )
        candidate_count_after_book_gate_filter = int(len(filtered_rows))
        candidate_count_excluded_book_gate_filter = int(
            safe_float(book_gate_meta.get("candidate_count_excluded", 0), 0)
        )
        if not filtered_rows:
            self._record_choice_row(
                {
                    "decision_timestamp": decision_ts,
                    "route_decision": LANE_NO_TRADE,
                    "choice_reason": "no_candidates_after_book_gate_filter",
                    "runtime_mode": self.runtime_mode,
                    "candidate_count_raw": int(len(candidate_rows)),
                    "candidate_count_after_mode_filter": int(candidate_count_after_mode_filter),
                    "candidate_count_after_book_gate_filter": int(candidate_count_after_book_gate_filter),
                    "candidate_count_excluded_runtime_filter": int(decision_excluded_by_runtime_filter),
                    "candidate_count_excluded_book_gate_filter": int(candidate_count_excluded_book_gate_filter),
                    "book_gate_enabled": bool(book_gate_meta.get("enabled", False)),
                    "book_gate_selected_book": str(book_gate_meta.get("selected_book", "") or ""),
                    "book_gate_scope": str(book_gate_meta.get("matched_scope", "") or ""),
                    "book_gate_bucket": str(book_gate_meta.get("matched_bucket", "") or ""),
                    "book_gate_reason": str(book_gate_meta.get("reason", "") or ""),
                    "book_gate_fallback_to_default": bool(book_gate_meta.get("fallback_to_default", False)),
                }
            )
            return self._abstain_result(
                reason="no_candidates_after_book_gate_filter",
                feasible_rows=[],
                route_decision=LANE_NO_TRADE,
                route_confidence=0.0,
                route_margin=0.0,
                route_scores={LANE_NO_TRADE: 1.0},
            )

        multibook_decision_switching = bool(book_gate_meta.get("decision_policy_switching", False))
        default_book = str(book_gate_meta.get("default_book", "") or "")
        active_book = str(book_gate_meta.get("selected_book", "") or default_book)
        active_decision_model_cfg = dict(self.direct_decision_model_cfg)
        active_candidate_filter_cfg = dict(self.candidate_variant_filter_cfg)
        if multibook_decision_switching and active_book:
            active_decision_model_cfg = self._book_gate_decision_model_cfg_for_book(active_book)
            active_candidate_filter_cfg = self._book_gate_filter_cfg_for_book(active_book)

        pre_variant_rows = list(filtered_rows)
        active_variant_stats = (
            active_decision_model_cfg.get("variant_stats", {})
            if isinstance(active_decision_model_cfg.get("variant_stats"), dict)
            else {}
        )
        filtered_rows, candidate_variant_filter_meta = self._filter_candidate_rows_with_cfg(
            candidate_rows=pre_variant_rows,
            filter_cfg=active_candidate_filter_cfg,
            variant_stats=active_variant_stats,
            context_label=f"candidate_variant_filter:{active_book or 'default'}",
            annotate_prefix="candidate_variant_filter",
            count_rejections=True,
        )
        candidate_variant_filter_excluded = int(
            candidate_variant_filter_meta.get("excluded", 0) or 0
        )
        if (
            (not filtered_rows)
            and multibook_decision_switching
            and bool(self.book_gate_model_cfg.get("fallback_to_default_on_empty", True))
            and default_book
            and active_book
            and active_book != default_book
        ):
            active_book = str(default_book)
            active_decision_model_cfg = self._book_gate_decision_model_cfg_for_book(active_book)
            active_candidate_filter_cfg = self._book_gate_filter_cfg_for_book(active_book)
            active_variant_stats = (
                active_decision_model_cfg.get("variant_stats", {})
                if isinstance(active_decision_model_cfg.get("variant_stats"), dict)
                else {}
            )
            filtered_rows, candidate_variant_filter_meta = self._filter_candidate_rows_with_cfg(
                candidate_rows=pre_variant_rows,
                filter_cfg=active_candidate_filter_cfg,
                variant_stats=active_variant_stats,
                context_label=f"candidate_variant_filter:{active_book or 'default'}",
                annotate_prefix="candidate_variant_filter",
                count_rejections=True,
            )
            candidate_variant_filter_excluded = int(
                candidate_variant_filter_meta.get("excluded", 0) or 0
            )
            book_gate_meta["selected_book"] = str(active_book)
            book_gate_meta["fallback_to_default"] = True
            book_gate_meta["reason"] = "fallback_to_default_after_variant_quality_filter"

        for row in filtered_rows:
            if not isinstance(row, dict):
                continue
            row["candidate_variant_filter_book"] = str(active_book or "")
            row["book_gate_selected_book"] = str(active_book or "")
            row["book_gate_fallback_to_default"] = bool(
                book_gate_meta.get("fallback_to_default", False)
            )
        self._record_book_gate_selection(book_gate_meta)
        if not filtered_rows:
            self._record_choice_row(
                {
                    "decision_timestamp": decision_ts,
                    "route_decision": LANE_NO_TRADE,
                    "choice_reason": "no_candidates_after_variant_quality_filter",
                    "runtime_mode": self.runtime_mode,
                    "candidate_count_raw": int(len(candidate_rows)),
                    "candidate_count_after_mode_filter": int(candidate_count_after_mode_filter),
                    "candidate_count_after_book_gate_filter": int(candidate_count_after_book_gate_filter),
                    "candidate_count_excluded_runtime_filter": int(decision_excluded_by_runtime_filter),
                    "candidate_count_excluded_book_gate_filter": int(candidate_count_excluded_book_gate_filter),
                    "candidate_count_excluded_variant_quality_filter": int(candidate_variant_filter_excluded),
                    "book_gate_enabled": bool(book_gate_meta.get("enabled", False)),
                    "book_gate_selected_book": str(book_gate_meta.get("selected_book", "") or ""),
                    "book_gate_scope": str(book_gate_meta.get("matched_scope", "") or ""),
                    "book_gate_bucket": str(book_gate_meta.get("matched_bucket", "") or ""),
                    "book_gate_reason": str(book_gate_meta.get("reason", "") or ""),
                    "book_gate_fallback_to_default": bool(book_gate_meta.get("fallback_to_default", False)),
                }
            )
            return self._abstain_result(
                reason="no_candidates_after_variant_quality_filter",
                feasible_rows=[],
                route_decision=LANE_NO_TRADE,
                route_confidence=0.0,
                route_margin=0.0,
                route_scores={LANE_NO_TRADE: 1.0},
            )

        decision_export_rows = self._snapshot_candidate_rows_for_export(filtered_rows)

        decision_side_enabled_any = False
        effective_decision_side_meta: Dict[str, Any] = {}
        effective_decision_side_mode = ""
        effective_decision_side_name = ""
        for model_index, raw_model_cfg in enumerate(self.decision_side_model_cfgs):
            model_cfg = dict(raw_model_cfg) if isinstance(raw_model_cfg, dict) else {}
            if not bool(model_cfg.get("enabled", False)):
                continue
            decision_side_meta = self._evaluate_decision_side_model(
                candidate_rows=filtered_rows,
                default_session=str(default_session or ""),
                current_time=current_time,
                model_cfg=model_cfg,
            )
            decision_side_mode = str(
                model_cfg.get("application_mode", "hard_override") or "hard_override"
            ).strip().lower()
            model_name = str(model_cfg.get("model_name", "") or f"model_{model_index}")
            if bool(decision_side_meta.get("enabled", False)):
                decision_side_enabled_any = True
                effective_decision_side_meta = dict(decision_side_meta)
                effective_decision_side_mode = str(decision_side_mode)
                effective_decision_side_name = str(model_name)
                self._decision_side_eval_count += 1
                for scope_name in decision_side_meta.get("matched_scopes", []):
                    scope_text = str(scope_name or "").strip()
                    if scope_text:
                        self._decision_side_scope_match_counts[f"{model_name}:{scope_text}"] += 1
                predicted_action = str(decision_side_meta.get("predicted_action", "") or "").strip().lower()
                if predicted_action:
                    self._decision_side_predicted_action_counts[predicted_action] += 1
                if predicted_action == "no_trade" and decision_side_mode == "hard_override":
                    self._decision_side_abstain_count += 1
                    self._record_choice_row(
                        {
                            "decision_timestamp": decision_ts,
                            "route_decision": LANE_NO_TRADE,
                            "choice_reason": "decision_side_model:no_trade",
                            "runtime_mode": self.runtime_mode,
                            "candidate_count_raw": int(len(candidate_rows)),
                            "candidate_count_after_mode_filter": int(candidate_count_after_mode_filter),
                            "candidate_count_after_book_gate_filter": int(candidate_count_after_book_gate_filter),
                            "candidate_count_excluded_runtime_filter": int(decision_excluded_by_runtime_filter),
                            "candidate_count_excluded_book_gate_filter": int(candidate_count_excluded_book_gate_filter),
                            "candidate_count_excluded_variant_quality_filter": int(candidate_variant_filter_excluded),
                            "session": str(decision_side_meta.get("session", default_session) or default_session).strip().upper(),
                            "chosen_side": "",
                            "decision_side_model_enabled": True,
                            "decision_side_model_name": str(model_name),
                            "decision_side_model_application_mode": str(decision_side_mode),
                            "decision_side_model_predicted_action": str(predicted_action),
                            "decision_side_model_side_pattern": str(
                                decision_side_meta.get("side_pattern", "") or ""
                            ),
                            "decision_side_model_baseline_side_guess": str(
                                decision_side_meta.get("baseline_side_guess", "") or ""
                            ),
                            "decision_side_model_match_count": int(
                                safe_float(decision_side_meta.get("match_count", 0), 0)
                            ),
                            "decision_side_model_long_score": float(
                                safe_float(decision_side_meta.get("long_score", 0.0), 0.0)
                            ),
                            "decision_side_model_short_score": float(
                                safe_float(decision_side_meta.get("short_score", 0.0), 0.0)
                            ),
                            "decision_side_model_no_trade_score": float(
                                safe_float(decision_side_meta.get("no_trade_score", 0.0), 0.0)
                            ),
                        }
                    )
                    return self._abstain_result(
                        reason="decision_side_model:no_trade",
                        feasible_rows=filtered_rows,
                        decision_export_rows=decision_export_rows,
                        route_decision=LANE_NO_TRADE,
                        route_confidence=0.0,
                        route_margin=0.0,
                        route_scores={LANE_NO_TRADE: 1.0},
                    )
                one_sided_block_no_prediction = bool(
                    model_cfg.get("block_one_sided_on_no_prediction", False)
                )
                side_pattern_text = str(
                    decision_side_meta.get("side_pattern", "") or ""
                ).strip().lower()
                match_count = int(safe_float(decision_side_meta.get("match_count", 0), 0))
                if (
                    decision_side_mode == "hard_override"
                    and one_sided_block_no_prediction
                    and not predicted_action
                    and match_count > 0
                    and side_pattern_text in {"long_only", "short_only"}
                ):
                    self._decision_side_abstain_count += 1
                    self._record_choice_row(
                        {
                            "decision_timestamp": decision_ts,
                            "route_decision": LANE_NO_TRADE,
                            "choice_reason": "decision_side_model:one_sided_no_prediction",
                            "runtime_mode": self.runtime_mode,
                            "candidate_count_raw": int(len(candidate_rows)),
                            "candidate_count_after_mode_filter": int(candidate_count_after_mode_filter),
                            "candidate_count_after_book_gate_filter": int(candidate_count_after_book_gate_filter),
                            "candidate_count_excluded_runtime_filter": int(decision_excluded_by_runtime_filter),
                            "candidate_count_excluded_book_gate_filter": int(candidate_count_excluded_book_gate_filter),
                            "candidate_count_excluded_variant_quality_filter": int(candidate_variant_filter_excluded),
                            "session": str(
                                decision_side_meta.get("session", default_session) or default_session
                            ).strip().upper(),
                            "chosen_side": "",
                            "decision_side_model_enabled": True,
                            "decision_side_model_name": str(model_name),
                            "decision_side_model_application_mode": str(decision_side_mode),
                            "decision_side_model_predicted_action": "",
                            "decision_side_model_side_pattern": str(
                                decision_side_meta.get("side_pattern", "") or ""
                            ),
                            "decision_side_model_baseline_side_guess": str(
                                decision_side_meta.get("baseline_side_guess", "") or ""
                            ),
                            "decision_side_model_match_count": int(match_count),
                            "decision_side_model_long_score": float(
                                safe_float(decision_side_meta.get("long_score", 0.0), 0.0)
                            ),
                            "decision_side_model_short_score": float(
                                safe_float(decision_side_meta.get("short_score", 0.0), 0.0)
                            ),
                            "decision_side_model_no_trade_score": float(
                                safe_float(decision_side_meta.get("no_trade_score", 0.0), 0.0)
                            ),
                        }
                    )
                    return self._abstain_result(
                        reason="decision_side_model:one_sided_no_prediction",
                        feasible_rows=filtered_rows,
                        decision_export_rows=decision_export_rows,
                        route_decision=LANE_NO_TRADE,
                        route_confidence=0.0,
                        route_margin=0.0,
                        route_scores={LANE_NO_TRADE: 1.0},
                    )
                if predicted_action in {"long", "short"} and decision_side_mode == "hard_override":
                    side_filtered_rows = [
                        row
                        for row in filtered_rows
                        if str(row.get("side_considered", "") or "").strip().lower() == predicted_action
                    ]
                    if side_filtered_rows:
                        filtered_rows = side_filtered_rows
                        self._decision_side_override_count += 1
            apply_prior_only_when_predicted = bool(model_cfg.get("apply_prior_only_when_predicted", False))
            for row in filtered_rows:
                if not isinstance(row, dict):
                    continue
                row["decision_side_model_enabled"] = bool(decision_side_enabled_any)
                row["decision_side_model_name"] = str(model_name)
                row["decision_side_model_application_mode"] = str(decision_side_mode)
                if str(decision_side_meta.get("predicted_action", "") or "").strip():
                    row["decision_side_model_predicted_action"] = str(
                        decision_side_meta.get("predicted_action", "") or ""
                    )
                row["decision_side_model_match_count"] = int(
                    safe_float(row.get("decision_side_model_match_count", 0), 0)
                    + safe_float(decision_side_meta.get("match_count", 0), 0)
                )
                row["decision_side_model_long_score"] = float(
                    safe_float(row.get("decision_side_model_long_score", 0.0), 0.0)
                    + safe_float(decision_side_meta.get("long_score", 0.0), 0.0)
                )
                row["decision_side_model_short_score"] = float(
                    safe_float(row.get("decision_side_model_short_score", 0.0), 0.0)
                    + safe_float(decision_side_meta.get("short_score", 0.0), 0.0)
                )
                row["decision_side_model_no_trade_score"] = float(
                    safe_float(row.get("decision_side_model_no_trade_score", 0.0), 0.0)
                    + safe_float(decision_side_meta.get("no_trade_score", 0.0), 0.0)
                )
                if apply_prior_only_when_predicted and not str(
                    decision_side_meta.get("predicted_action", "") or ""
                ).strip():
                    continue
                side_name = str(row.get("side_considered", "") or "").strip().lower()
                long_score = float(safe_float(decision_side_meta.get("long_score", 0.0), 0.0))
                short_score = float(safe_float(decision_side_meta.get("short_score", 0.0), 0.0))
                no_trade_score = max(0.0, float(safe_float(decision_side_meta.get("no_trade_score", 0.0), 0.0)))
                prior_formula = str(
                    model_cfg.get("prior_formula", "relative_minus_no_trade") or "relative_minus_no_trade"
                ).strip().lower()
                long_prior_score = float(long_score - short_score - no_trade_score)
                short_prior_score = float(short_score - long_score - no_trade_score)
                if prior_formula == "relative_side_only":
                    long_prior_score = float(long_score - short_score)
                    short_prior_score = float(short_score - long_score)
                elif prior_formula == "relative_minus_half_no_trade":
                    long_prior_score = float(long_score - short_score - (0.5 * no_trade_score))
                    short_prior_score = float(short_score - long_score - (0.5 * no_trade_score))
                elif prior_formula == "side_minus_no_trade":
                    long_prior_score = float(long_score - no_trade_score)
                    short_prior_score = float(short_score - no_trade_score)
                side_prior_score = 0.0
                if side_name == "long":
                    side_prior_score = float(long_prior_score)
                elif side_name == "short":
                    side_prior_score = float(short_prior_score)
                row["decision_side_model_prior_score"] = float(
                    safe_float(row.get("decision_side_model_prior_score", 0.0), 0.0)
                    + side_prior_score
                )
                prior_weight = max(0.0, safe_float(model_cfg.get("prior_component_weight", 0.0), 0.0))
                prior_scale = max(1e-9, safe_float(model_cfg.get("prior_component_scale", 0.60), 0.60))
                prior_component = float(prior_weight * math.tanh(side_prior_score / prior_scale))
                row["decision_side_model_prior_component_total"] = float(
                    safe_float(row.get("decision_side_model_prior_component_total", 0.0), 0.0)
                    + prior_component
                )
                matched_scopes = (
                    decision_side_meta.get("matched_scopes", [])
                    if isinstance(decision_side_meta.get("matched_scopes", []), list)
                    else []
                )
                prefixed_scopes = [f"{model_name}:{str(scope).strip()}" for scope in matched_scopes if str(scope).strip()]
                existing_scopes = (
                    row.get("decision_side_model_matched_scopes", [])
                    if isinstance(row.get("decision_side_model_matched_scopes", []), list)
                    else []
                )
                row["decision_side_model_matched_scopes"] = list(existing_scopes) + prefixed_scopes
                row["decision_side_model_side_pattern"] = str(
                    decision_side_meta.get("side_pattern", row.get("decision_side_model_side_pattern", "")) or ""
                )
                row["decision_side_model_baseline_side_guess"] = str(
                    decision_side_meta.get(
                        "baseline_side_guess",
                        row.get("decision_side_model_baseline_side_guess", ""),
                    )
                    or ""
                )

        conflict_side_meta = self._evaluate_conflict_side_model(
            candidate_rows=filtered_rows,
            default_session=str(default_session or ""),
            current_time=current_time,
        )
        if bool(conflict_side_meta.get("enabled", False)):
            self._conflict_side_eval_count += 1
            for scope_name in conflict_side_meta.get("matched_scopes", []):
                scope_text = str(scope_name or "").strip()
                if scope_text:
                    self._conflict_side_scope_match_counts[scope_text] += 1
            predicted_side = str(conflict_side_meta.get("predicted_side", "") or "").strip().lower()
            if predicted_side:
                self._conflict_side_predicted_side_counts[predicted_side] += 1
            if predicted_side == "no_trade":
                self._conflict_side_abstain_count += 1
                self._record_choice_row(
                    {
                        "decision_timestamp": decision_ts,
                        "route_decision": LANE_NO_TRADE,
                        "choice_reason": "conflict_side_model:no_trade",
                        "runtime_mode": self.runtime_mode,
                        "candidate_count_raw": int(len(candidate_rows)),
                        "candidate_count_after_mode_filter": int(candidate_count_after_mode_filter),
                        "candidate_count_after_book_gate_filter": int(candidate_count_after_book_gate_filter),
                        "candidate_count_excluded_runtime_filter": int(decision_excluded_by_runtime_filter),
                        "candidate_count_excluded_book_gate_filter": int(candidate_count_excluded_book_gate_filter),
                        "book_gate_enabled": bool(book_gate_meta.get("enabled", False)),
                        "book_gate_selected_book": str(book_gate_meta.get("selected_book", "") or ""),
                        "book_gate_scope": str(book_gate_meta.get("matched_scope", "") or ""),
                        "book_gate_bucket": str(book_gate_meta.get("matched_bucket", "") or ""),
                        "book_gate_reason": str(book_gate_meta.get("reason", "") or ""),
                        "book_gate_fallback_to_default": bool(book_gate_meta.get("fallback_to_default", False)),
                        "session": str(
                            effective_decision_side_meta.get("session", default_session) or default_session
                        ).strip().upper(),
                        "chosen_side": "",
                        "decision_side_model_enabled": bool(decision_side_enabled_any),
                        "decision_side_model_name": str(effective_decision_side_name or ""),
                        "decision_side_model_application_mode": str(
                            effective_decision_side_mode or ""
                        ),
                        "decision_side_model_predicted_action": str(
                            effective_decision_side_meta.get("predicted_action", "") or ""
                        ),
                        "decision_side_model_side_pattern": str(
                            effective_decision_side_meta.get("side_pattern", "") or ""
                        ),
                        "decision_side_model_baseline_side_guess": str(
                            effective_decision_side_meta.get("baseline_side_guess", "") or ""
                        ),
                        "conflict_side_model_enabled": True,
                        "conflict_side_model_predicted_side": str(predicted_side),
                        "conflict_side_model_score": float(
                            safe_float(conflict_side_meta.get("score", 0.0), 0.0)
                        ),
                        "conflict_side_model_threshold": float(
                            safe_float(conflict_side_meta.get("threshold", 0.0), 0.0)
                        ),
                        "conflict_side_model_match_count": int(
                            safe_float(conflict_side_meta.get("match_count", 0), 0)
                        ),
                    }
                )
                return self._abstain_result(
                    reason="conflict_side_model:no_trade",
                    feasible_rows=filtered_rows,
                    decision_export_rows=decision_export_rows,
                    route_decision=LANE_NO_TRADE,
                    route_confidence=0.0,
                    route_margin=0.0,
                    route_scores={LANE_NO_TRADE: 1.0},
                )
            if predicted_side in {"long", "short"}:
                side_filtered_rows = [
                    row
                    for row in filtered_rows
                    if str(row.get("side_considered", "") or "").strip().lower() == predicted_side
                ]
                if side_filtered_rows:
                    filtered_rows = side_filtered_rows
                    self._conflict_side_override_count += 1
        for row in filtered_rows:
            if not isinstance(row, dict):
                continue
            row["conflict_side_model_enabled"] = bool(conflict_side_meta.get("enabled", False))
            row["conflict_side_model_predicted_side"] = str(
                conflict_side_meta.get("predicted_side", "") or ""
            )
            row["conflict_side_model_score"] = float(safe_float(conflict_side_meta.get("score", 0.0), 0.0))
            row["conflict_side_model_threshold"] = float(
                safe_float(conflict_side_meta.get("threshold", 0.0), 0.0)
            )
            row["conflict_side_model_match_count"] = int(
                safe_float(conflict_side_meta.get("match_count", 0), 0)
            )
            row["conflict_side_model_matched_scopes"] = list(
                conflict_side_meta.get("matched_scopes", [])
                if isinstance(conflict_side_meta.get("matched_scopes", []), list)
                else []
            )

        selection_stack = self._select_with_model_stack(
            candidate_rows=filtered_rows,
            default_session=str(default_session or ""),
            decision_model_cfg=active_decision_model_cfg,
            track_counters=True,
        )
        lane_candidates = (
            dict(selection_stack.get("lane_candidates", {}))
            if isinstance(selection_stack.get("lane_candidates", {}), dict)
            else {}
        )
        timeframe_hint = str(selection_stack.get("timeframe_hint", "") or "")
        direct_result = (
            dict(selection_stack.get("direct_result", {}))
            if isinstance(selection_stack.get("direct_result"), dict)
            else None
        )
        baseline_result = (
            dict(selection_stack.get("baseline_result", {}))
            if isinstance(selection_stack.get("baseline_result"), dict)
            else None
        )

        chosen_entry = None
        runner_up_entry = None
        selected_variant_id = ""
        lane_selection_reason = ""
        lane_rows: List[Dict[str, Any]] = []
        if isinstance(direct_result, dict):
            route_decision = str(direct_result.get("route_decision", LANE_NO_TRADE) or LANE_NO_TRADE)
            route_conf = safe_float(direct_result.get("route_confidence", 0.0), 0.0)
            route_margin = safe_float(direct_result.get("route_margin", 0.0), 0.0)
            route_scores = (
                dict(direct_result.get("route_scores", {}))
                if isinstance(direct_result.get("route_scores", {}), dict)
                else {}
            )
            route_reason = str(direct_result.get("route_reason", "") or "")
            lane_result = {
                "lane_candidate_count": int(direct_result.get("lane_candidate_count", 0) or 0),
                "lane_scores": list(direct_result.get("lane_scores", []))
                if isinstance(direct_result.get("lane_scores", []), list)
                else [],
                "selected_variant_id": str(direct_result.get("selected_variant_id", "") or ""),
                "lane_selection_reason": str(
                    direct_result.get("lane_selection_reason", "direct_decision_model_select")
                    or "direct_decision_model_select"
                ),
            }
            chosen_entry = (
                dict(direct_result.get("chosen_entry", {}))
                if isinstance(direct_result.get("chosen_entry"), dict)
                else None
            )
            selected_variant_id = str(direct_result.get("selected_variant_id", "") or "")
            lane_selection_reason = str(
                direct_result.get("lane_selection_reason", "") or ""
            )
            lane_rows = (
                lane_candidates.get(route_decision, [])
                if isinstance(lane_candidates.get(route_decision, []), list)
                else []
            )
        else:
            baseline_selected = (
                baseline_result
                if isinstance(baseline_result, dict)
                else self._select_with_router_lane_stack(
                    lane_candidates=lane_candidates,
                    default_session=str(default_session or ""),
                    timeframe_hint=str(timeframe_hint or ""),
                )
            )
            route_decision = str(
                baseline_selected.get("route_decision", LANE_NO_TRADE) or LANE_NO_TRADE
            )
            route_conf = safe_float(baseline_selected.get("route_confidence", 0.0), 0.0)
            route_margin = safe_float(baseline_selected.get("route_margin", 0.0), 0.0)
            route_scores = (
                dict(baseline_selected.get("route_scores", {}))
                if isinstance(baseline_selected.get("route_scores", {}), dict)
                else {}
            )
            route_reason = str(baseline_selected.get("route_reason", "") or "")
            lane_result = (
                dict(baseline_selected.get("lane_result", {}))
                if isinstance(baseline_selected.get("lane_result", {}), dict)
                else {
                    "lane_candidate_count": 0,
                    "lane_scores": [],
                    "selected_variant_id": "",
                    "lane_selection_reason": "",
                }
            )
            chosen_entry = (
                dict(baseline_selected.get("chosen_entry", {}))
                if isinstance(baseline_selected.get("chosen_entry"), dict)
                else None
            )
            runner_up_entry = (
                dict(baseline_selected.get("runner_up_entry", {}))
                if isinstance(baseline_selected.get("runner_up_entry"), dict)
                else None
            )
            selected_variant_id = str(baseline_selected.get("selected_variant_id", "") or "")
            lane_selection_reason = str(
                baseline_selected.get("lane_selection_reason", "") or ""
            )
            lane_rows = (
                list(baseline_selected.get("lane_rows", []))
                if isinstance(baseline_selected.get("lane_rows", []), list)
                else []
            )

        self._record_router_row(
            {
                "decision_timestamp": decision_ts,
                "route_decision": route_decision,
                "route_confidence": float(route_conf),
                "route_margin": float(route_margin),
                "route_scores": dict(route_scores),
                "route_reason": route_reason,
                "runtime_mode": self.runtime_mode,
                "candidate_count_raw": int(len(candidate_rows)),
                "candidate_count_after_mode_filter": int(len(filtered_rows)),
                "candidate_count_excluded_runtime_filter": int(decision_excluded_by_runtime_filter),
            }
        )

        if route_decision == LANE_NO_TRADE:
            self._record_choice_row(
                {
                    "decision_timestamp": decision_ts,
                    "route_decision": route_decision,
                    "choice_reason": route_reason or "router_no_trade",
                    "runtime_mode": self.runtime_mode,
                    "candidate_count_raw": int(len(candidate_rows)),
                    "candidate_count_after_mode_filter": int(len(filtered_rows)),
                    "candidate_count_after_book_gate_filter": int(candidate_count_after_book_gate_filter),
                    "candidate_count_excluded_runtime_filter": int(decision_excluded_by_runtime_filter),
                    "candidate_count_excluded_book_gate_filter": int(candidate_count_excluded_book_gate_filter),
                    "book_gate_enabled": bool(book_gate_meta.get("enabled", False)),
                    "book_gate_selected_book": str(book_gate_meta.get("selected_book", "") or ""),
                    "book_gate_scope": str(book_gate_meta.get("matched_scope", "") or ""),
                    "book_gate_bucket": str(book_gate_meta.get("matched_bucket", "") or ""),
                    "book_gate_reason": str(book_gate_meta.get("reason", "") or ""),
                    "book_gate_fallback_to_default": bool(book_gate_meta.get("fallback_to_default", False)),
                }
            )
            return self._abstain_result(
                reason=route_reason or "router_no_trade",
                feasible_rows=filtered_rows,
                decision_export_rows=decision_export_rows,
                route_decision=route_decision,
                route_confidence=route_conf,
                route_margin=route_margin,
                route_scores=route_scores,
            )

        if not lane_rows:
            self._record_choice_row(
                {
                    "decision_timestamp": decision_ts,
                    "route_decision": route_decision,
                    "choice_reason": "route_lane_has_no_candidates",
                    "runtime_mode": self.runtime_mode,
                    "candidate_count_raw": int(len(candidate_rows)),
                    "candidate_count_after_mode_filter": int(len(filtered_rows)),
                    "candidate_count_after_book_gate_filter": int(candidate_count_after_book_gate_filter),
                    "candidate_count_excluded_runtime_filter": int(decision_excluded_by_runtime_filter),
                    "candidate_count_excluded_book_gate_filter": int(candidate_count_excluded_book_gate_filter),
                    "book_gate_enabled": bool(book_gate_meta.get("enabled", False)),
                    "book_gate_selected_book": str(book_gate_meta.get("selected_book", "") or ""),
                    "book_gate_scope": str(book_gate_meta.get("matched_scope", "") or ""),
                    "book_gate_bucket": str(book_gate_meta.get("matched_bucket", "") or ""),
                    "book_gate_reason": str(book_gate_meta.get("reason", "") or ""),
                    "book_gate_fallback_to_default": bool(book_gate_meta.get("fallback_to_default", False)),
                }
            )
            return self._abstain_result(
                reason="route_lane_has_no_candidates",
                feasible_rows=filtered_rows,
                decision_export_rows=decision_export_rows,
                route_decision=route_decision,
                route_confidence=route_conf,
                route_margin=route_margin,
                route_scores=route_scores,
            )

        if chosen_entry is None:
            self._record_choice_row(
                {
                    "decision_timestamp": decision_ts,
                    "route_decision": route_decision,
                    "choice_reason": "no_variant_selected_within_lane",
                    "runtime_mode": self.runtime_mode,
                    "candidate_count_raw": int(len(candidate_rows)),
                    "candidate_count_after_mode_filter": int(len(filtered_rows)),
                    "candidate_count_after_book_gate_filter": int(candidate_count_after_book_gate_filter),
                    "candidate_count_excluded_runtime_filter": int(decision_excluded_by_runtime_filter),
                    "candidate_count_excluded_book_gate_filter": int(candidate_count_excluded_book_gate_filter),
                    "book_gate_enabled": bool(book_gate_meta.get("enabled", False)),
                    "book_gate_selected_book": str(book_gate_meta.get("selected_book", "") or ""),
                    "book_gate_scope": str(book_gate_meta.get("matched_scope", "") or ""),
                    "book_gate_bucket": str(book_gate_meta.get("matched_bucket", "") or ""),
                    "book_gate_reason": str(book_gate_meta.get("reason", "") or ""),
                    "book_gate_fallback_to_default": bool(book_gate_meta.get("fallback_to_default", False)),
                }
            )
            return self._abstain_result(
                reason="no_variant_selected_within_lane",
                feasible_rows=filtered_rows,
                decision_export_rows=decision_export_rows,
                route_decision=route_decision,
                route_confidence=route_conf,
                route_margin=route_margin,
                route_scores=route_scores,
            )

        execution_policy_result = self._evaluate_execution_policy(
            chosen_entry=chosen_entry,
            route_confidence=route_conf,
        )
        policy_tier = str(execution_policy_result.get("tier", "disabled") or "disabled")
        policy_reason = str(execution_policy_result.get("reason", "") or "")
        policy_quality_score = safe_float(execution_policy_result.get("quality_score", 0.0), 0.0)
        policy_hard_limit_triggered = bool(execution_policy_result.get("hard_limit_triggered", False))
        policy_hard_limit_reason = str(execution_policy_result.get("hard_limit_reason", "") or "")
        policy_components = (
            dict(execution_policy_result.get("components", {}))
            if isinstance(execution_policy_result.get("components"), dict)
            else {}
        )
        entry_model_enabled = bool(execution_policy_result.get("entry_model_enabled", False))
        entry_model_allow = bool(execution_policy_result.get("entry_model_allow", True))
        entry_model_tier = str(execution_policy_result.get("entry_model_tier", "disabled") or "disabled")
        entry_model_reason = str(
            execution_policy_result.get("entry_model_reason", "calibrated_entry_model_disabled")
            or "calibrated_entry_model_disabled"
        )
        entry_model_score = safe_float(execution_policy_result.get("entry_model_score", 0.0), 0.0)
        entry_model_threshold = safe_float(execution_policy_result.get("entry_model_threshold", 0.0), 0.0)
        entry_model_threshold_base = safe_float(
            execution_policy_result.get("entry_model_threshold_base", 0.0),
            0.0,
        )
        entry_model_threshold_scope_offset = safe_float(
            execution_policy_result.get("entry_model_threshold_scope_offset", 0.0),
            0.0,
        )
        entry_model_scope = str(execution_policy_result.get("entry_model_scope", "disabled") or "disabled")
        entry_model_stats = (
            dict(execution_policy_result.get("entry_model_stats", {}))
            if isinstance(execution_policy_result.get("entry_model_stats"), dict)
            else {}
        )
        entry_model_components = (
            dict(execution_policy_result.get("entry_model_components", {}))
            if isinstance(execution_policy_result.get("entry_model_components"), dict)
            else {}
        )
        entry_model_fallback_scope_guard = (
            dict(execution_policy_result.get("entry_model_fallback_scope_guard", {}))
            if isinstance(execution_policy_result.get("entry_model_fallback_scope_guard"), dict)
            else {}
        )

        policy_allow = bool(execution_policy_result.get("allow", True))
        policy_enforce_veto = bool(
            execution_policy_result.get(
                "enforce_veto",
                self.execution_policy_cfg.get("enforce_veto", True),
            )
        )
        if not policy_allow and policy_enforce_veto:
            self._execution_policy_reject_count += 1
            self._execution_policy_reject_reason_counts[policy_reason or "execution_policy_reject"] += 1
            self._record_choice_row(
                {
                    "decision_timestamp": decision_ts,
                    "route_decision": route_decision,
                    "choice_reason": f"execution_policy:{policy_reason or 'reject'}",
                    "execution_policy_tier": policy_tier,
                    "execution_policy_quality_score": float(policy_quality_score),
                    "execution_policy_reason": policy_reason,
                    "execution_policy_hard_limit_triggered": bool(policy_hard_limit_triggered),
                    "execution_policy_hard_limit_reason": policy_hard_limit_reason,
                    "runtime_mode": self.runtime_mode,
                    "candidate_count_raw": int(len(candidate_rows)),
                    "candidate_count_after_mode_filter": int(len(filtered_rows)),
                    "candidate_count_after_book_gate_filter": int(candidate_count_after_book_gate_filter),
                    "candidate_count_excluded_runtime_filter": int(decision_excluded_by_runtime_filter),
                    "candidate_count_excluded_book_gate_filter": int(candidate_count_excluded_book_gate_filter),
                    "book_gate_enabled": bool(book_gate_meta.get("enabled", False)),
                    "book_gate_selected_book": str(book_gate_meta.get("selected_book", "") or ""),
                    "book_gate_scope": str(book_gate_meta.get("matched_scope", "") or ""),
                    "book_gate_bucket": str(book_gate_meta.get("matched_bucket", "") or ""),
                    "book_gate_reason": str(book_gate_meta.get("reason", "") or ""),
                    "book_gate_fallback_to_default": bool(book_gate_meta.get("fallback_to_default", False)),
                    "session": str(
                        effective_decision_side_meta.get("session", default_session) or default_session
                    ).strip().upper(),
                    "chosen_side": "",
                    "decision_side_model_enabled": bool(decision_side_enabled_any),
                    "decision_side_model_name": str(effective_decision_side_name or ""),
                    "decision_side_model_application_mode": str(
                        effective_decision_side_mode or ""
                    ),
                    "decision_side_model_predicted_action": str(
                        effective_decision_side_meta.get("predicted_action", "") or ""
                    ),
                    "decision_side_model_side_pattern": str(
                        effective_decision_side_meta.get("side_pattern", "") or ""
                    ),
                    "decision_side_model_baseline_side_guess": str(
                        effective_decision_side_meta.get("baseline_side_guess", "") or ""
                    ),
                }
            )
            return self._abstain_result(
                reason=f"execution_policy:{policy_reason or 'reject'}",
                feasible_rows=filtered_rows,
                decision_export_rows=decision_export_rows,
                route_decision=route_decision,
                route_confidence=route_conf,
                route_margin=route_margin,
                route_scores=route_scores,
            )
        if not policy_allow and (not policy_enforce_veto):
            soft_tier = str(self.execution_policy_cfg.get("soft_tier_on_reject", "conservative") or "conservative").strip().lower()
            if soft_tier not in {"conservative", "full", "reject"}:
                soft_tier = "conservative"
            self._execution_policy_soft_pass_count += 1
            self._execution_policy_soft_pass_reason_counts[policy_reason or "execution_policy_reject"] += 1
            policy_tier = str(soft_tier)
            policy_reason = f"{policy_reason or 'execution_policy_reject'}:soft_pass"

        self._execution_policy_tier_counts[policy_tier] += 1
        self._record_execution_policy_row(
            {
                "decision_timestamp": decision_ts,
                "route_decision": route_decision,
                "selected_variant_id": selected_variant_id,
                "tier": policy_tier,
                "reason": policy_reason,
                "quality_score": float(policy_quality_score),
                "allow": bool(policy_allow),
                "enforce_veto": bool(policy_enforce_veto),
                "soft_pass": bool((not policy_allow) and (not policy_enforce_veto)),
                "hard_limit_triggered": bool(policy_hard_limit_triggered),
                "hard_limit_reason": str(policy_hard_limit_reason),
                "components": dict(policy_components),
                "inputs": dict(execution_policy_result.get("inputs", {}) if isinstance(execution_policy_result.get("inputs"), dict) else {}),
                "entry_model_enabled": bool(entry_model_enabled),
                "entry_model_allow": bool(entry_model_allow),
                "entry_model_tier": str(entry_model_tier),
                "entry_model_reason": str(entry_model_reason),
                "entry_model_score": float(entry_model_score),
                "entry_model_threshold": float(entry_model_threshold),
                "entry_model_threshold_base": float(entry_model_threshold_base),
                "entry_model_threshold_scope_offset": float(entry_model_threshold_scope_offset),
                "entry_model_scope": str(entry_model_scope),
                "entry_model_stats": dict(entry_model_stats),
                "entry_model_components": dict(entry_model_components),
                "entry_model_fallback_scope_guard": dict(entry_model_fallback_scope_guard),
            }
        )

        chosen_entry["de3_v4_execution_policy_tier"] = str(policy_tier)
        chosen_entry["de3_v4_execution_quality_score"] = float(policy_quality_score)
        chosen_entry["de3_v4_execution_policy_reason"] = str(policy_reason or "policy_allow")
        chosen_entry["de3_v4_execution_policy_source"] = str(self.execution_policy_cfg.get("source", "defaults"))
        chosen_entry["de3_v4_execution_policy_enforce_veto"] = bool(policy_enforce_veto)
        chosen_entry["de3_v4_execution_policy_soft_pass"] = bool((not policy_allow) and (not policy_enforce_veto))
        chosen_entry["de3_v4_execution_policy_hard_limit_triggered"] = bool(policy_hard_limit_triggered)
        chosen_entry["de3_v4_execution_policy_hard_limit_reason"] = str(policy_hard_limit_reason)
        chosen_entry["de3_v4_execution_policy_components"] = dict(policy_components)
        chosen_entry["de3_v4_entry_model_enabled"] = bool(entry_model_enabled)
        chosen_entry["de3_v4_entry_model_allow"] = bool(entry_model_allow)
        chosen_entry["de3_v4_entry_model_tier"] = str(entry_model_tier)
        chosen_entry["de3_v4_entry_model_reason"] = str(entry_model_reason)
        chosen_entry["de3_v4_entry_model_score"] = float(entry_model_score)
        chosen_entry["de3_v4_entry_model_threshold"] = float(entry_model_threshold)
        chosen_entry["de3_v4_entry_model_threshold_base"] = float(entry_model_threshold_base)
        chosen_entry["de3_v4_entry_model_threshold_scope_offset"] = float(
            entry_model_threshold_scope_offset
        )
        chosen_entry["de3_v4_entry_model_scope"] = str(entry_model_scope)
        chosen_entry["de3_v4_entry_model_stats"] = dict(entry_model_stats)
        chosen_entry["de3_v4_entry_model_components"] = dict(entry_model_components)
        chosen_entry["de3_v4_entry_model_fallback_scope_guard"] = dict(
            entry_model_fallback_scope_guard
        )

        bracket_result = self.bracket_module.select_bracket(
            lane=route_decision,
            variant_id=selected_variant_id,
            chosen_entry=chosen_entry,
            context_inputs=ctx,
        )

        self._record_lane_row(
            {
                "decision_timestamp": decision_ts,
                "selected_lane": route_decision,
                "selected_variant_id": selected_variant_id,
                "lane_candidate_count": int(lane_result.get("lane_candidate_count", 0) or 0),
                "runner_up_variant_id": (
                    str(
                        runner_up_entry.get(
                            "selected_variant_id",
                            runner_up_entry.get("variant_id", ""),
                        )
                    )
                    if isinstance(runner_up_entry, dict)
                    else ""
                ),
                "lane_selection_reason": lane_selection_reason,
                "runtime_mode": self.runtime_mode,
            }
        )
        self._record_bracket_row(
            {
                "decision_timestamp": decision_ts,
                "selected_lane": route_decision,
                "variant_id": selected_variant_id,
                "bracket_mode": str(bracket_result.get("bracket_mode", "canonical")),
                "selected_sl": float(bracket_result.get("selected_sl", 0.0) or 0.0),
                "selected_tp": float(bracket_result.get("selected_tp", 0.0) or 0.0),
                "canonical_default_used": bool(bracket_result.get("canonical_default_used", True)),
                "bracket_reason": str(bracket_result.get("bracket_reason", "")),
            }
        )

        chosen_cand = chosen_entry.get("cand", {}) if isinstance(chosen_entry.get("cand"), dict) else {}
        chosen_family_id = str(
            chosen_entry.get("family_id")
            or build_family_id(
                timeframe=chosen_cand.get("timeframe", ""),
                session=default_session,
                strategy_type=chosen_cand.get("strategy_type", ""),
                threshold=chosen_cand.get("thresh", 0.0),
            )
            or ""
        )
        self._record_choice_row(
            {
                "decision_timestamp": decision_ts,
                "route_decision": route_decision,
                "selected_lane": route_decision,
                "selected_variant_id": selected_variant_id,
                "chosen_family_id": chosen_family_id,
                "session": str(chosen_entry.get("session", default_session) or default_session).strip().upper(),
                "chosen_side": str(
                    chosen_entry.get("side_considered", lane_to_side(route_decision)) or lane_to_side(route_decision)
                ).strip().lower(),
                "choice_reason": (
                    "direct_decision_model"
                    if str(route_reason or "").startswith("direct_decision_model")
                    else "ranked_lane_then_variant"
                ),
                "execution_policy_tier": policy_tier,
                "execution_policy_quality_score": float(policy_quality_score),
                "execution_policy_reason": str(policy_reason or "policy_allow"),
                "execution_policy_hard_limit_triggered": bool(policy_hard_limit_triggered),
                "entry_model_enabled": bool(entry_model_enabled),
                "entry_model_allow": bool(entry_model_allow),
                "entry_model_tier": str(entry_model_tier),
                "entry_model_reason": str(entry_model_reason),
                "runtime_mode": self.runtime_mode,
                "candidate_count_raw": int(len(candidate_rows)),
                "candidate_count_after_mode_filter": int(len(filtered_rows)),
                "candidate_count_after_book_gate_filter": int(candidate_count_after_book_gate_filter),
                "candidate_count_excluded_runtime_filter": int(decision_excluded_by_runtime_filter),
                "candidate_count_excluded_book_gate_filter": int(candidate_count_excluded_book_gate_filter),
                "book_gate_enabled": bool(book_gate_meta.get("enabled", False)),
                "book_gate_selected_book": str(book_gate_meta.get("selected_book", "") or ""),
                "book_gate_scope": str(book_gate_meta.get("matched_scope", "") or ""),
                "book_gate_bucket": str(book_gate_meta.get("matched_bucket", "") or ""),
                "book_gate_reason": str(book_gate_meta.get("reason", "") or ""),
                "book_gate_fallback_to_default": bool(book_gate_meta.get("fallback_to_default", False)),
                "decision_side_model_enabled": bool(decision_side_enabled_any),
                "decision_side_model_name": str(effective_decision_side_name or ""),
                "decision_side_model_application_mode": str(effective_decision_side_mode or ""),
                "decision_side_model_predicted_action": str(
                    effective_decision_side_meta.get("predicted_action", "") or ""
                ),
                "decision_side_model_side_pattern": str(
                    effective_decision_side_meta.get("side_pattern", "") or ""
                ),
                "decision_side_model_baseline_side_guess": str(
                    effective_decision_side_meta.get("baseline_side_guess", "") or ""
                ),
                "conflict_side_model_enabled": bool(conflict_side_meta.get("enabled", False)),
                "conflict_side_model_predicted_side": str(
                    conflict_side_meta.get("predicted_side", "") or ""
                ),
                "conflict_side_model_score": float(safe_float(conflict_side_meta.get("score", 0.0), 0.0)),
                "conflict_side_model_threshold": float(
                    safe_float(conflict_side_meta.get("threshold", 0.0), 0.0)
                ),
                "conflict_side_model_match_count": int(
                    safe_float(conflict_side_meta.get("match_count", 0), 0)
                ),
            }
        )

        return {
            "abstained": False,
            "abstain_reason": "",
            "route_decision": route_decision,
            "route_confidence": float(route_conf),
            "route_margin": float(route_margin),
            "route_scores": route_scores,
            "route_reason": route_reason,
            "selected_lane": route_decision,
            "selected_variant_id": selected_variant_id,
            "lane_candidate_count": int(lane_result.get("lane_candidate_count", 0) or 0),
            "lane_scores": list(lane_result.get("lane_scores", [])),
            "runner_up_within_lane": (
                str(
                    runner_up_entry.get(
                        "selected_variant_id",
                        runner_up_entry.get("variant_id", ""),
                    )
                )
                if isinstance(runner_up_entry, dict)
                else ""
            ),
            "lane_selection_reason": lane_selection_reason,
            "chosen_family_id": chosen_family_id,
            "execution_policy_tier": policy_tier,
            "execution_quality_score": float(policy_quality_score),
            "execution_policy_reason": str(policy_reason or "policy_allow"),
            "chosen_entry": chosen_entry,
            "chosen_family_row": None,
            "chosen_member_local_score": None,
            "local_bracket_adaptation_mode": str(bracket_result.get("bracket_mode", "canonical")),
            "local_bracket_adaptation_enabled": True,
            "local_bracket_override_applied": bool(
                safe_float(bracket_result.get("selected_sl", 0.0), 0.0) > 0.0
                and safe_float(bracket_result.get("selected_tp", 0.0), 0.0) > 0.0
            ),
            "execution_policy_result": dict(execution_policy_result),
            "bracket_result": bracket_result,
            "book_gate_enabled": bool(book_gate_meta.get("enabled", False)),
            "book_gate_selected_book": str(book_gate_meta.get("selected_book", "") or ""),
            "book_gate_scope": str(book_gate_meta.get("matched_scope", "") or ""),
            "book_gate_bucket": str(book_gate_meta.get("matched_bucket", "") or ""),
            "book_gate_reason": str(book_gate_meta.get("reason", "") or ""),
            "book_gate_fallback_to_default": bool(book_gate_meta.get("fallback_to_default", False)),
            "conflict_side_model_enabled": bool(conflict_side_meta.get("enabled", False)),
            "conflict_side_model_predicted_side": str(
                conflict_side_meta.get("predicted_side", "") or ""
            ),
            "conflict_side_model_score": float(safe_float(conflict_side_meta.get("score", 0.0), 0.0)),
            "conflict_side_model_threshold": float(
                safe_float(conflict_side_meta.get("threshold", 0.0), 0.0)
            ),
            "conflict_side_model_match_count": int(
                safe_float(conflict_side_meta.get("match_count", 0), 0)
            ),
            "feasible_family_rows": [],
            "feasible_rows": filtered_rows,
            "decision_export_rows": decision_export_rows,
        }

    def get_runtime_path_counters(self) -> Dict[str, Any]:
        route_counts = Counter(self._route_decision_counts)
        mode_counts = Counter(self._runtime_mode_counts)
        policy_eval_count = int(self._execution_policy_rows_total)
        policy_allowed_count = int(self._execution_policy_allowed_count)
        return {
            "runtime_invocations": int(self._runtime_invocations),
            "route_decision_counts": dict(route_counts),
            "runtime_mode_counts": dict(mode_counts),
            "router_no_trade_count": int(route_counts.get(LANE_NO_TRADE, 0)),
            "lane_selection_count": int(self._lane_rows_total),
            "bracket_decision_count": int(self._bracket_rows_total),
            "trace_max_rows": int(self.trace_max_rows),
            "trace_rows_sampled": {
                "router": int(len(self._router_rows)),
                "lane": int(len(self._lane_rows)),
                "bracket": int(len(self._bracket_rows)),
                "choice": int(len(self._choice_rows)),
                "execution_policy": int(len(self._execution_policy_rows)),
            },
            "runtime_excluded_family_count": int(self._runtime_excluded_family_count),
            "runtime_excluded_variant_pattern_count": int(self._runtime_excluded_variant_pattern_count),
            "candidate_variant_filter_enabled": bool(self.candidate_variant_filter_enabled),
            "candidate_variant_filter_reject_count": int(self._candidate_variant_filter_reject_count),
            "candidate_variant_filter_reject_reason_counts": dict(
                self._candidate_variant_filter_reject_reason_counts
            ),
            "decision_side_model_enabled": bool(self.decision_side_model_enabled),
            "decision_side_models_count": int(len(self.decision_side_model_cfgs)),
            "decision_side_model_names": [
                str(
                    cfg.get("model_name", f"model_{idx}")
                    if isinstance(cfg, dict)
                    else f"model_{idx}"
                )
                for idx, cfg in enumerate(self.decision_side_model_cfgs)
                if isinstance(cfg, dict)
            ],
            "decision_side_eval_count": int(self._decision_side_eval_count),
            "decision_side_override_count": int(self._decision_side_override_count),
            "decision_side_abstain_count": int(self._decision_side_abstain_count),
            "decision_side_predicted_action_counts": dict(self._decision_side_predicted_action_counts),
            "decision_side_scope_match_counts": dict(self._decision_side_scope_match_counts),
            "conflict_side_model_enabled": bool(self.conflict_side_model_enabled),
            "conflict_side_eval_count": int(self._conflict_side_eval_count),
            "conflict_side_override_count": int(self._conflict_side_override_count),
            "conflict_side_abstain_count": int(self._conflict_side_abstain_count),
            "conflict_side_predicted_side_counts": dict(self._conflict_side_predicted_side_counts),
            "conflict_side_scope_match_counts": dict(self._conflict_side_scope_match_counts),
            "book_gate_enabled": bool(self.book_gate_enabled),
            "book_gate_decision_count": int(self._book_gate_decision_count),
            "book_gate_selected_book_counts": dict(self._book_gate_selected_book_counts),
            "book_gate_scope_match_counts": dict(self._book_gate_scope_match_counts),
            "book_gate_context_key_counts": dict(self._book_gate_context_key_counts),
            "book_gate_default_selection_count": int(self._book_gate_default_selection_count),
            "book_gate_empty_fallback_count": int(self._book_gate_empty_fallback_count),
            "execution_policy_enabled": bool(self.execution_policy_enabled),
            "execution_policy_eval_count": int(policy_eval_count),
            "execution_policy_allowed_count": int(policy_allowed_count),
            "execution_policy_reject_count": int(self._execution_policy_reject_count),
            "execution_policy_reject_reason_counts": dict(self._execution_policy_reject_reason_counts),
            "execution_policy_tier_counts": dict(self._execution_policy_tier_counts),
            "execution_policy_soft_pass_count": int(self._execution_policy_soft_pass_count),
            "execution_policy_soft_pass_reason_counts": dict(self._execution_policy_soft_pass_reason_counts),
            "execution_policy_enforce_veto": bool(self.execution_policy_cfg.get("enforce_veto", True)),
            "entry_model_enabled": bool(self.calibrated_entry_model_enabled),
            "entry_model_eval_count": int(self._entry_model_eval_count),
            "entry_model_reject_count": int(self._entry_model_reject_count),
            "entry_model_missing_stats_count": int(self._entry_model_missing_stats_count),
            "entry_model_scope_counts": dict(self._entry_model_scope_counts),
            # Legacy aliases for existing report/diff consumers.
            "execution_filter_enabled": bool(self.execution_policy_enabled),
            "execution_filter_reject_count": int(self._execution_policy_reject_count),
            "execution_filter_reject_reason_counts": dict(self._execution_policy_reject_reason_counts),
        }

    def _sanitized_execution_policy_cfg(self) -> Dict[str, Any]:
        cfg = dict(self.execution_policy_cfg) if isinstance(self.execution_policy_cfg, dict) else {}
        cem = cfg.get("calibrated_entry_model", {}) if isinstance(cfg.get("calibrated_entry_model"), dict) else {}
        if cem:
            cem_clean = dict(cem)
            cem_clean["variant_stats_count"] = int(len(self._entry_model_variant_stats))
            cem_clean["lane_stats_count"] = int(len(self._entry_model_lane_stats))
            cem_clean["global_stats_n_trades"] = int(
                safe_float(
                    (self._entry_model_global_stats if isinstance(self._entry_model_global_stats, dict) else {}).get(
                        "n_trades",
                        0,
                    ),
                    0.0,
                )
            )
            cem_clean.pop("variant_stats", None)
            cem_clean.pop("lane_stats", None)
            cem_clean.pop("global_stats", None)
            cfg["calibrated_entry_model"] = cem_clean
        return cfg

    def get_activation_audit(self) -> Dict[str, Any]:
        meta = self.bundle.get("metadata", {}) if isinstance(self.bundle.get("metadata"), dict) else {}
        training_split = (
            meta.get("training_split", {})
            if isinstance(meta.get("training_split"), dict)
            else {}
        )
        exec_rule_summary = (
            meta.get("execution_rule_summary", {})
            if isinstance(meta.get("execution_rule_summary"), dict)
            else {}
        )
        timestamp_audit = (
            meta.get("timestamp_audit", {})
            if isinstance(meta.get("timestamp_audit"), dict)
            else {}
        )
        return {
            "created_at": dt.datetime.now(NY_TZ).isoformat(),
            "active_de3_version": "v4",
            "bundle_path": str(self.bundle_path),
            "bundle_loaded": bool(self.bundle_loaded),
            "router_enabled": True,
            "lane_selector_enabled": True,
            "bracket_module_enabled": True,
            "runtime_mode": str(self.runtime_mode),
            "core_enabled": bool(self.core_enabled),
            "core_anchor_family_ids": list(self.core_anchor_family_ids),
            "force_anchor_when_eligible": bool(self.force_anchor_when_eligible),
            "excluded_family_ids": sorted(list(self.excluded_family_ids)),
            "excluded_variant_patterns": sorted(list(self.excluded_variant_patterns)),
            "candidate_variant_filter_enabled": bool(self.candidate_variant_filter_enabled),
            "candidate_variant_filter": (
                dict(self.candidate_variant_filter_cfg)
                if isinstance(self.candidate_variant_filter_cfg, dict)
                else {}
            ),
            "decision_side_model_enabled": bool(self.decision_side_model_enabled),
            "decision_side_model": (
                dict(self.decision_side_model_cfg)
                if isinstance(self.decision_side_model_cfg, dict)
                else {}
            ),
            "decision_side_models": [
                dict(cfg) for cfg in self.decision_side_model_cfgs if isinstance(cfg, dict)
            ],
            "conflict_side_model_enabled": bool(self.conflict_side_model_enabled),
            "conflict_side_model": (
                dict(self.conflict_side_model_cfg)
                if isinstance(self.conflict_side_model_cfg, dict)
                else {}
            ),
            "book_gate_enabled": bool(self.book_gate_enabled),
            "book_gate_model": (
                dict(self.book_gate_model_cfg)
                if isinstance(self.book_gate_model_cfg, dict)
                else {}
            ),
            "source_data_path": str(meta.get("source_data_path", "")),
            "source_data_format": str(meta.get("source_data_format", "")),
            "training_split": dict(training_split),
            "leakage_check_passed": bool(
                ((meta.get("leakage_summary") or {}) if isinstance(meta.get("leakage_summary"), dict) else {}).get("leakage_check_passed", False)
            ),
            "split_usage_assertions": {
                "tune_2024_used_for_model_selection": True,
                "oos_2025_held_out_from_training": True,
                "future_2026_excluded_from_training": True,
            },
            "execution_rules_active_for_training_labels": dict(exec_rule_summary),
            "timestamp_interpretation": dict(timestamp_audit),
            "execution_policy_enabled": bool(self.execution_policy_enabled),
            "execution_policy_source": str(self.execution_policy_cfg.get("source", "defaults")),
            "execution_policy_enforce_veto": bool(self.execution_policy_cfg.get("enforce_veto", True)),
            "execution_policy_soft_tier_on_reject": str(
                self.execution_policy_cfg.get("soft_tier_on_reject", "conservative")
            ),
            "calibrated_entry_model_enabled": bool(self.calibrated_entry_model_enabled),
            "calibrated_entry_model_threshold": float(
                safe_float(self.calibrated_entry_model_cfg.get("selected_threshold", 0.0), 0.0)
            ),
            "calibrated_entry_model_threshold_source": str(
                self.calibrated_entry_model_cfg.get("selected_threshold_source", "")
            ),
            "calibrated_entry_model_fit_windows": (
                dict(self.calibrated_entry_model_cfg.get("fit_windows", {}))
                if isinstance(self.calibrated_entry_model_cfg.get("fit_windows"), dict)
                else {}
            ),
            "execution_policy": self._sanitized_execution_policy_cfg(),
            "config_snapshot_relevant": {"DE3_V4": dict(self.cfg)},
        }

    def get_router_summary(self) -> Dict[str, Any]:
        route_counts = Counter(self._route_decision_counts)
        reason_counts = Counter(self._route_reason_counts)
        total = int(self._router_rows_total)
        return {
            "route_count_total": int(total),
            "route_counts": dict(route_counts),
            "route_reason_counts": dict(reason_counts),
            "no_trade_count": int(route_counts.get(LANE_NO_TRADE, 0)),
            "route_confidence_mean": float(self._route_conf_sum / total) if total > 0 else 0.0,
            "route_confidence_min": float(self._route_conf_min) if self._route_conf_min is not None else 0.0,
            "route_confidence_max": float(self._route_conf_max) if self._route_conf_max is not None else 0.0,
            "route_margin_mean": float(self._route_margin_sum / total) if total > 0 else 0.0,
            "route_margin_min": float(self._route_margin_min) if self._route_margin_min is not None else 0.0,
            "route_margin_max": float(self._route_margin_max) if self._route_margin_max is not None else 0.0,
            "trace_max_rows": int(self.trace_max_rows),
            "trace_rows_sampled": int(len(self._router_rows)),
        }

    def get_lane_selection_summary(self) -> Dict[str, Any]:
        lane_counts = Counter(self._lane_selected_counts)
        variant_counts = Counter(self._variant_selected_counts)
        total = int(self._lane_rows_total)
        return {
            "lane_selection_count": int(total),
            "selected_lane_counts": dict(lane_counts),
            "selected_variant_counts": dict(variant_counts),
            "lane_candidate_count_mean": (
                float(self._lane_candidate_count_sum / total)
                if total > 0
                else 0.0
            ),
            "trace_max_rows": int(self.trace_max_rows),
            "trace_rows_sampled": int(len(self._lane_rows)),
        }

    def get_bracket_summary(self) -> Dict[str, Any]:
        mode_counts = Counter(self._bracket_mode_counts)
        total = int(self._bracket_rows_total)
        return {
            "bracket_decision_count": int(total),
            "bracket_mode_counts": dict(mode_counts),
            "selected_sl_mean": float(self._bracket_sl_sum / total) if total > 0 else 0.0,
            "selected_tp_mean": float(self._bracket_tp_sum / total) if total > 0 else 0.0,
            "canonical_default_usage_count": int(self._canonical_default_usage_count),
            "trace_max_rows": int(self.trace_max_rows),
            "trace_rows_sampled": int(len(self._bracket_rows)),
        }

    def get_execution_policy_summary(self) -> Dict[str, Any]:
        rows = list(self._execution_policy_rows)
        allowed_rows = [row for row in rows if bool(row.get("allow", False))]
        rejected_rows = [row for row in rows if not bool(row.get("allow", False))]

        def _avg(values: List[float]) -> float:
            return float(sum(values) / len(values)) if values else 0.0

        def _component_means(sample_rows: List[Dict[str, Any]]) -> Dict[str, float]:
            accum: Dict[str, List[float]] = defaultdict(list)
            for row in sample_rows:
                comps = row.get("components", {}) if isinstance(row.get("components"), dict) else {}
                for key, raw in comps.items():
                    val = self._optional_float(raw)
                    if val is not None:
                        accum[str(key)].append(float(val))
            return {k: _avg(v) for k, v in accum.items()}

        score_all = [safe_float(row.get("quality_score", 0.0), 0.0) for row in rows]
        score_allowed = [safe_float(row.get("quality_score", 0.0), 0.0) for row in allowed_rows]
        score_rejected = [safe_float(row.get("quality_score", 0.0), 0.0) for row in rejected_rows]
        entry_model_scores_all = [safe_float(row.get("entry_model_score", 0.0), 0.0) for row in rows]
        entry_model_scores_allowed = [safe_float(row.get("entry_model_score", 0.0), 0.0) for row in allowed_rows]
        entry_model_scores_rejected = [safe_float(row.get("entry_model_score", 0.0), 0.0) for row in rejected_rows]
        entry_model_scope_counts = Counter(self._entry_model_scope_counts)
        return {
            "active_de3_version": "v4",
            "enabled": bool(self.execution_policy_enabled),
            "source": str(self.execution_policy_cfg.get("source", "defaults")),
            "enforce_veto": bool(self.execution_policy_cfg.get("enforce_veto", True)),
            "soft_tier_on_reject": str(self.execution_policy_cfg.get("soft_tier_on_reject", "conservative")),
            "policy_config": self._sanitized_execution_policy_cfg(),
            "entry_model_enabled": bool(self.calibrated_entry_model_enabled),
            "entry_model_threshold": float(
                safe_float(self.calibrated_entry_model_cfg.get("selected_threshold", 0.0), 0.0)
            ),
            "evaluated_count": int(self._execution_policy_rows_total),
            "allowed_count": int(self._execution_policy_allowed_count),
            "rejected_count": int(
                max(0, self._execution_policy_rows_total - self._execution_policy_allowed_count)
            ),
            "reject_rate": float(
                (self._execution_policy_rows_total - self._execution_policy_allowed_count)
                / self._execution_policy_rows_total
            ) if self._execution_policy_rows_total > 0 else 0.0,
            "tier_counts": dict(self._execution_policy_tier_counts),
            "reject_reason_counts": dict(self._execution_policy_reject_reason_counts),
            "soft_pass_count": int(self._execution_policy_soft_pass_count),
            "soft_pass_reason_counts": dict(self._execution_policy_soft_pass_reason_counts),
            "hard_limit_reject_count": int(
                sum(1 for row in rejected_rows if bool(row.get("hard_limit_triggered", False)))
            ),
            "mean_quality_score_all": _avg(score_all),
            "mean_quality_score_allowed": _avg(score_allowed),
            "mean_quality_score_rejected": _avg(score_rejected),
            "mean_entry_model_score_all": _avg(entry_model_scores_all),
            "mean_entry_model_score_allowed": _avg(entry_model_scores_allowed),
            "mean_entry_model_score_rejected": _avg(entry_model_scores_rejected),
            "entry_model_eval_count": int(self._entry_model_eval_count),
            "entry_model_reject_count": int(self._entry_model_reject_count),
            "entry_model_reject_reason_counts": dict(self._entry_model_reject_reason_counts),
            "entry_model_missing_stats_count": int(self._entry_model_missing_stats_count),
            "entry_model_scope_counts": dict(entry_model_scope_counts),
            "component_means_all": _component_means(rows),
            "component_means_allowed": _component_means(allowed_rows),
            "component_means_rejected": _component_means(rejected_rows),
            "trace_max_rows": int(self.trace_max_rows),
            "trace_rows_sampled": int(len(self._execution_policy_rows)),
        }

    def get_runtime_mode_summary(self) -> Dict[str, Any]:
        chosen_family_counts = Counter(self._chosen_family_counts)
        choice_reason_counts: Counter[str] = Counter()
        for row in self._choice_rows:
            if not isinstance(row, dict):
                continue
            choice_reason = str(row.get("choice_reason", "") or "").strip()
            if choice_reason:
                choice_reason_counts[choice_reason] += 1
        return {
            "active_de3_version": "v4",
            "mode": str(self.runtime_mode),
            "core_enabled": bool(self.core_enabled),
            "core_family_ids_loaded": list(self.core_anchor_family_ids),
            "excluded_family_ids": sorted(list(self.excluded_family_ids)),
            "excluded_variant_patterns": sorted(list(self.excluded_variant_patterns)),
            "candidate_variant_filter_enabled": bool(self.candidate_variant_filter_enabled),
            "candidate_variant_filter_reject_count": int(self._candidate_variant_filter_reject_count),
            "candidate_variant_filter_reject_reason_counts": dict(
                self._candidate_variant_filter_reject_reason_counts
            ),
            "book_gate_enabled": bool(self.book_gate_enabled),
            "book_gate_decision_count": int(self._book_gate_decision_count),
            "book_gate_selected_book_counts": dict(self._book_gate_selected_book_counts),
            "book_gate_scope_match_counts": dict(self._book_gate_scope_match_counts),
            "book_gate_default_selection_count": int(self._book_gate_default_selection_count),
            "book_gate_empty_fallback_count": int(self._book_gate_empty_fallback_count),
            "router_loaded": True,
            "lane_selector_loaded": True,
            "bracket_module_loaded": True,
            "runtime_invocations": int(self._runtime_invocations),
            "chosen_family_counts": dict(chosen_family_counts),
            "chosen_family_unique_count": int(len(chosen_family_counts)),
            "choice_reason_counts": dict(choice_reason_counts),
            "trace_max_rows": int(self.trace_max_rows),
            "trace_rows_sampled": int(len(self._choice_rows)),
            "execution_policy_enabled": bool(self.execution_policy_enabled),
            "execution_policy_reject_count": int(self._execution_policy_reject_count),
            "execution_policy_reject_reason_counts": dict(self._execution_policy_reject_reason_counts),
            "execution_policy_tier_counts": dict(self._execution_policy_tier_counts),
            "execution_policy_soft_pass_count": int(self._execution_policy_soft_pass_count),
            "execution_policy_soft_pass_reason_counts": dict(self._execution_policy_soft_pass_reason_counts),
            "execution_policy_enforce_veto": bool(self.execution_policy_cfg.get("enforce_veto", True)),
            "entry_model_enabled": bool(self.calibrated_entry_model_enabled),
            "entry_model_use_bundle_model": bool(self.calibrated_entry_model_cfg.get("use_bundle_model", False)),
            "entry_model_threshold": float(
                safe_float(self.calibrated_entry_model_cfg.get("selected_threshold", 0.0), 0.0)
            ),
            "entry_model_eval_count": int(self._entry_model_eval_count),
            "entry_model_reject_count": int(self._entry_model_reject_count),
            "entry_model_reject_reason_counts": dict(self._entry_model_reject_reason_counts),
            "entry_model_missing_stats_count": int(self._entry_model_missing_stats_count),
            "entry_model_scope_counts": dict(self._entry_model_scope_counts),
            # Legacy aliases for existing v4 report readers.
            "execution_filters_enabled": bool(self.execution_policy_enabled),
            "execution_filter_reject_count": int(self._execution_policy_reject_count),
            "execution_filter_reject_reason_counts": dict(self._execution_policy_reject_reason_counts),
        }

    def get_decision_side_summary(self) -> Dict[str, Any]:
        def _top_key(counter: Counter[str]) -> str:
            top = counter.most_common(1)
            return str(top[0][0]) if top else ""

        model_rows: List[Dict[str, Any]] = []
        for idx, cfg in enumerate(self.decision_side_model_cfgs):
            if not isinstance(cfg, dict):
                continue
            model_rows.append(
                {
                    "name": str(cfg.get("model_name", f"model_{idx}") or f"model_{idx}"),
                    "enabled": bool(cfg.get("enabled", False)),
                    "application_mode": str(
                        cfg.get("application_mode", "hard_override") or "hard_override"
                    ),
                    "apply_side_patterns": [
                        str(v)
                        for v in (
                            cfg.get("apply_side_patterns", [])
                            if isinstance(cfg.get("apply_side_patterns", []), list)
                            else []
                        )
                        if str(v).strip()
                    ],
                }
            )

        session_predicted_counts: Dict[str, Counter[str]] = defaultdict(Counter)
        session_chosen_counts: Dict[str, Counter[str]] = defaultdict(Counter)
        session_outcome_counts: Dict[str, Counter[str]] = defaultdict(Counter)
        session_choice_reason_counts: Dict[str, Counter[str]] = defaultdict(Counter)
        session_side_pattern_counts: Dict[str, Counter[str]] = defaultdict(Counter)
        predicted_action_counts: Counter[str] = Counter()
        chosen_side_counts: Counter[str] = Counter()
        outcome_counts: Counter[str] = Counter()

        for raw_row in self._choice_rows:
            if not isinstance(raw_row, dict):
                continue
            session = str(raw_row.get("session", "") or "").strip().upper() or "UNKNOWN"
            choice_reason = str(raw_row.get("choice_reason", "") or "").strip()
            if choice_reason:
                session_choice_reason_counts[session][choice_reason] += 1
            predicted_action = str(
                raw_row.get("decision_side_model_predicted_action", "") or ""
            ).strip().lower()
            if predicted_action:
                predicted_action_counts[predicted_action] += 1
                session_predicted_counts[session][predicted_action] += 1
            chosen_side = str(raw_row.get("chosen_side", "") or "").strip().lower()
            if chosen_side in {"long", "short"}:
                chosen_side_counts[chosen_side] += 1
                session_chosen_counts[session][chosen_side] += 1
            side_pattern = str(
                raw_row.get("decision_side_model_side_pattern", "") or ""
            ).strip().lower()
            if side_pattern:
                session_side_pattern_counts[session][side_pattern] += 1

            outcome = "no_prediction_no_trade"
            if predicted_action in {"long", "short"} and chosen_side in {"long", "short"}:
                outcome = "aligned" if predicted_action == chosen_side else "mismatch"
            elif predicted_action == "no_trade":
                if choice_reason == "decision_side_model:no_trade":
                    outcome = "blocked_no_trade"
                elif chosen_side in {"long", "short"}:
                    outcome = "predicted_no_trade_but_traded"
                else:
                    outcome = "predicted_no_trade"
            elif predicted_action in {"long", "short"} and chosen_side not in {"long", "short"}:
                outcome = "predicted_side_but_no_trade"
            elif predicted_action == "" and chosen_side in {"long", "short"}:
                outcome = "no_prediction_traded"

            outcome_counts[outcome] += 1
            session_outcome_counts[session][outcome] += 1

        session_names = sorted(
            set(session_choice_reason_counts)
            | set(session_predicted_counts)
            | set(session_chosen_counts)
            | set(session_outcome_counts)
            | set(session_side_pattern_counts)
        )
        session_rows: List[Dict[str, Any]] = []
        for session in session_names:
            pred_counter = session_predicted_counts.get(session, Counter())
            chosen_counter = session_chosen_counts.get(session, Counter())
            outcome_counter = session_outcome_counts.get(session, Counter())
            reason_counter = session_choice_reason_counts.get(session, Counter())
            pattern_counter = session_side_pattern_counts.get(session, Counter())
            session_rows.append(
                {
                    "session": str(session),
                    "rows": int(sum(reason_counter.values()) or sum(outcome_counter.values())),
                    "predicted_long": int(pred_counter.get("long", 0)),
                    "predicted_short": int(pred_counter.get("short", 0)),
                    "predicted_no_trade": int(pred_counter.get("no_trade", 0)),
                    "chosen_long": int(chosen_counter.get("long", 0)),
                    "chosen_short": int(chosen_counter.get("short", 0)),
                    "aligned": int(outcome_counter.get("aligned", 0)),
                    "mismatch": int(outcome_counter.get("mismatch", 0)),
                    "blocked_no_trade": int(outcome_counter.get("blocked_no_trade", 0)),
                    "predicted_side_but_no_trade": int(
                        outcome_counter.get("predicted_side_but_no_trade", 0)
                    ),
                    "predicted_no_trade_but_traded": int(
                        outcome_counter.get("predicted_no_trade_but_traded", 0)
                    ),
                    "no_prediction_traded": int(outcome_counter.get("no_prediction_traded", 0)),
                    "no_prediction_no_trade": int(
                        outcome_counter.get("no_prediction_no_trade", 0)
                    ),
                    "top_choice_reason": _top_key(reason_counter),
                    "top_side_pattern": _top_key(pattern_counter),
                }
            )

        return {
            "active_de3_version": "v4",
            "decision_side_model_enabled": bool(self.decision_side_model_enabled),
            "decision_side_models": model_rows,
            "eval_count": int(self._decision_side_eval_count),
            "override_count": int(self._decision_side_override_count),
            "abstain_count": int(self._decision_side_abstain_count),
            "predicted_action_counts": dict(predicted_action_counts),
            "chosen_side_counts": dict(chosen_side_counts),
            "outcome_counts": dict(outcome_counts),
            "session_rows": session_rows,
            "trace_max_rows": int(self.trace_max_rows),
            "trace_rows_sampled": int(len(self._choice_rows)),
            "trace_rows_total": int(self._choice_rows_total),
        }

    def get_runtime_status(self) -> Dict[str, Any]:
        lane_inventory = self.bundle.get("lane_inventory", {}) if isinstance(self.bundle.get("lane_inventory"), dict) else {}
        lane_variant_count = int(sum(len(v) for v in lane_inventory.values() if isinstance(v, list)))
        return {
            "bundle_path": str(self.bundle_path),
            "bundle_loaded": bool(self.bundle_loaded),
            "runtime_mode": str(self.runtime_mode),
            "router_enabled": True,
            "lane_selector_enabled": True,
            "bracket_module_enabled": True,
            "direct_decision_enabled": bool(self.direct_decision_enabled),
            "direct_decision_runtime_enabled": bool(self._direct_decision_runtime_enabled),
            "direct_decision_selection_mode": str(self.direct_decision_selection_mode),
            "lane_count": int(len(lane_inventory)),
            "lane_variant_count": int(lane_variant_count),
            "core_anchor_family_ids": list(self.core_anchor_family_ids),
            "excluded_family_ids": sorted(list(self.excluded_family_ids)),
            "excluded_variant_patterns": sorted(list(self.excluded_variant_patterns)),
            "candidate_variant_filter_enabled": bool(self.candidate_variant_filter_enabled),
            "candidate_variant_filter_reject_count": int(self._candidate_variant_filter_reject_count),
            "book_gate_enabled": bool(self.book_gate_enabled),
            "book_gate_decision_count": int(self._book_gate_decision_count),
            "book_gate_selected_book_counts": dict(self._book_gate_selected_book_counts),
            "book_gate_scope_match_counts": dict(self._book_gate_scope_match_counts),
            "execution_policy_enabled": bool(self.execution_policy_enabled),
            "execution_filters_enabled": bool(self.execution_policy_enabled),
            "calibrated_entry_model_enabled": bool(self.calibrated_entry_model_enabled),
            "calibrated_entry_model_use_bundle_model": bool(
                self.calibrated_entry_model_cfg.get("use_bundle_model", False)
            ),
            "calibrated_entry_model_threshold": float(
                safe_float(self.calibrated_entry_model_cfg.get("selected_threshold", 0.0), 0.0)
            ),
            "direct_decision_selected_count": int(self._direct_decision_selected_count),
            "direct_decision_abstain_count": int(self._direct_decision_abstain_count),
            "direct_decision_fallback_count": int(self._direct_decision_fallback_count),
            "direct_decision_baseline_compare_fallback_count": int(
                self._direct_decision_baseline_compare_fallback_count
            ),
        }


def _to_timestamp(value: Any) -> Optional[dt.datetime]:
    try:
        if hasattr(value, "to_pydatetime"):
            return value.to_pydatetime()
        if isinstance(value, dt.datetime):
            return value
    except Exception:
        return None
    return None
