from typing import Any, Dict, List, Optional

from de3_v4_schema import LANE_NO_TRADE, LANE_ORDER, safe_float


class DE3V4Router:
    def __init__(self, router_payload: Optional[Dict[str, Any]], runtime_cfg: Optional[Dict[str, Any]] = None):
        self.payload = dict(router_payload) if isinstance(router_payload, dict) else {}
        self.runtime_cfg = dict(runtime_cfg) if isinstance(runtime_cfg, dict) else {}
        self.lanes = list(self.payload.get("lanes", LANE_ORDER)) or list(LANE_ORDER)
        self.no_trade_label = str(self.payload.get("no_trade_label", LANE_NO_TRADE) or LANE_NO_TRADE)
        self.weights = (
            dict(self.payload.get("route_score_weights", {}))
            if isinstance(self.payload.get("route_score_weights"), dict)
            else {}
        )
        self._session_lane_priors = (
            dict(self.payload.get("lane_priors_by_session", {}))
            if isinstance(self.payload.get("lane_priors_by_session"), dict)
            else {}
        )
        self._timeframe_lane_priors = (
            dict(self.payload.get("lane_priors_by_timeframe", {}))
            if isinstance(self.payload.get("lane_priors_by_timeframe"), dict)
            else {}
        )
        self._global_lane_priors = (
            dict(self.payload.get("lane_priors_global", {}))
            if isinstance(self.payload.get("lane_priors_global"), dict)
            else {}
        )
        self.w_lane_prior = safe_float(self.weights.get("lane_prior", 0.55), 0.55)
        self.w_lane_max_edge = safe_float(self.weights.get("lane_max_edge", 0.30), 0.30)
        self.w_lane_mean_edge = safe_float(self.weights.get("lane_mean_edge", 0.15), 0.15)
        payload_no_trade_bias = safe_float(self.payload.get("no_trade_bias", 0.15), 0.15)
        payload_min_route_conf = safe_float(self.payload.get("min_route_confidence", 0.03), 0.03)
        payload_min_lane_score = safe_float(
            self.payload.get("min_lane_score_to_trade", -0.10),
            -0.10,
        )
        payload_min_score_margin = safe_float(
            self.payload.get("min_score_margin_to_trade", 0.0),
            0.0,
        )
        payload_min_single_lane_margin = safe_float(
            self.payload.get("min_single_lane_internal_margin", 0.0),
            0.0,
        )
        runtime_no_trade_bias = self.runtime_cfg.get("no_trade_bias")
        runtime_min_route_conf = self.runtime_cfg.get("min_route_confidence")
        runtime_min_lane_score = self.runtime_cfg.get("min_lane_score_to_trade")
        runtime_min_score_margin = self.runtime_cfg.get("min_score_margin_to_trade")
        runtime_min_single_lane_margin = self.runtime_cfg.get("min_single_lane_internal_margin")
        self.no_trade_bias = (
            safe_float(runtime_no_trade_bias, payload_no_trade_bias)
            if runtime_no_trade_bias is not None
            else payload_no_trade_bias
        )
        self.min_route_confidence = (
            safe_float(runtime_min_route_conf, payload_min_route_conf)
            if runtime_min_route_conf is not None
            else payload_min_route_conf
        )
        self.min_lane_score_to_trade = (
            safe_float(runtime_min_lane_score, payload_min_lane_score)
            if runtime_min_lane_score is not None
            else payload_min_lane_score
        )
        self.min_score_margin_to_trade = (
            safe_float(runtime_min_score_margin, payload_min_score_margin)
            if runtime_min_score_margin is not None
            else payload_min_score_margin
        )
        self.min_single_lane_internal_margin = (
            safe_float(runtime_min_single_lane_margin, payload_min_single_lane_margin)
            if runtime_min_single_lane_margin is not None
            else payload_min_single_lane_margin
        )
        self.allow_single_lane_trade = bool(
            self.runtime_cfg.get("allow_single_lane_trade", True)
        )
        self.single_lane_confidence_mode = str(
            self.runtime_cfg.get(
                "single_lane_confidence_mode",
                self.payload.get("single_lane_confidence_mode", "fixed"),
            )
            or "fixed"
        ).strip().lower()
        if self.single_lane_confidence_mode not in {"fixed", "internal_margin"}:
            self.single_lane_confidence_mode = "fixed"
        self.single_lane_confidence = safe_float(
            self.runtime_cfg.get("single_lane_confidence", self.min_route_confidence),
            self.min_route_confidence,
        )
        self.single_lane_internal_margin_scale = max(
            1e-6,
            safe_float(
                self.runtime_cfg.get(
                    "single_lane_internal_margin_scale",
                    self.payload.get("single_lane_internal_margin_scale", 0.05),
                ),
                0.05,
            ),
        )
        self.single_lane_confidence_min = safe_float(
            self.runtime_cfg.get(
                "single_lane_confidence_min",
                self.payload.get("single_lane_confidence_min", 0.0),
            ),
            0.0,
        )
        self.single_lane_confidence_max = safe_float(
            self.runtime_cfg.get(
                "single_lane_confidence_max",
                self.payload.get("single_lane_confidence_max", 1.0),
            ),
            1.0,
        )
        if self.single_lane_confidence_max < self.single_lane_confidence_min:
            self.single_lane_confidence_max = self.single_lane_confidence_min

    def _lane_prior(self, lane: str, *, session_name: str, timeframe_hint: str) -> float:
        session_val = 0.0
        if session_name and isinstance(self._session_lane_priors.get(session_name), dict):
            session_val = safe_float(
                (self._session_lane_priors.get(session_name) or {}).get(lane, 0.0),
                0.0,
            )
        tf_val = 0.0
        if timeframe_hint and isinstance(self._timeframe_lane_priors.get(timeframe_hint), dict):
            tf_val = safe_float(
                (self._timeframe_lane_priors.get(timeframe_hint) or {}).get(lane, 0.0),
                0.0,
            )
        global_val = safe_float(self._global_lane_priors.get(lane, 0.0), 0.0)
        return float((0.55 * session_val) + (0.20 * tf_val) + (0.25 * global_val))

    @staticmethod
    def _single_lane_internal_margin(rows: List[Dict[str, Any]]) -> float:
        best = float("-inf")
        second = float("-inf")
        for row in rows:
            if not isinstance(row, dict):
                continue
            # Prefer already-computed selection score; fallback to edge points.
            score_val = safe_float(
                row.get("selection_score", row.get("edge_points", 0.0)),
                0.0,
            )
            score = float(score_val)
            if score >= best:
                second = best
                best = score
            elif score > second:
                second = score
        if best == float("-inf") or second == float("-inf"):
            return 0.0
        return float(max(0.0, best - second))

    def _build_route_result(
        self,
        *,
        route_decision: str,
        route_confidence: float,
        route_margin: float,
        route_scores: Dict[str, float],
        route_components: Dict[str, Dict[str, float]],
        route_reason: str,
    ) -> Dict[str, Any]:
        return {
            "route_decision": str(route_decision),
            "route_confidence": float(route_confidence),
            "route_margin": float(route_margin),
            "route_scores": dict(route_scores),
            "route_components": route_components,
            "route_reason": str(route_reason),
        }

    def route(
        self,
        *,
        lane_candidates: Dict[str, List[Dict[str, Any]]],
        session_name: str,
        timeframe_hint: str,
    ) -> Dict[str, Any]:
        lane_scores: Dict[str, float] = {}
        lane_components: Dict[str, Dict[str, float]] = {}

        for lane in self.lanes:
            rows = lane_candidates.get(lane, []) if isinstance(lane_candidates.get(lane, []), list) else []
            if not rows:
                lane_scores[lane] = float("-inf")
                lane_components[lane] = {
                    "lane_prior": 0.0,
                    "lane_max_edge": 0.0,
                    "lane_mean_edge": 0.0,
                }
                continue
            lane_prior = self._lane_prior(lane, session_name=session_name, timeframe_hint=timeframe_hint)
            edge_sum = 0.0
            edge_count = 0
            lane_max_edge = float("-inf")
            for row in rows:
                edge_val = safe_float((row or {}).get("edge_points", 0.0), 0.0)
                edge_sum += float(edge_val)
                edge_count += 1
                if edge_val > lane_max_edge:
                    lane_max_edge = float(edge_val)
            if lane_max_edge == float("-inf"):
                lane_max_edge = 0.0
            lane_mean_edge = (edge_sum / edge_count) if edge_count > 0 else 0.0
            score = (
                (self.w_lane_prior * lane_prior)
                + (self.w_lane_max_edge * lane_max_edge)
                + (self.w_lane_mean_edge * lane_mean_edge)
            )
            lane_scores[lane] = float(score)
            lane_components[lane] = {
                "lane_prior": float(lane_prior),
                "lane_max_edge": float(lane_max_edge),
                "lane_mean_edge": float(lane_mean_edge),
            }

        ranked = sorted(
            [(lane, score) for lane, score in lane_scores.items() if score > float("-inf")],
            key=lambda x: x[1],
            reverse=True,
        )
        if not ranked:
            return self._build_route_result(
                route_decision=self.no_trade_label,
                route_confidence=0.0,
                route_margin=0.0,
                route_scores={self.no_trade_label: 1.0},
                route_components=lane_components,
                route_reason="no_lane_candidates",
            )

        best_lane, best_score = ranked[0]
        single_lane_only = len(ranked) == 1
        score_margin = 0.0
        if single_lane_only and self.allow_single_lane_trade:
            internal_margin = self._single_lane_internal_margin(
                lane_candidates.get(best_lane, [])
                if isinstance(lane_candidates.get(best_lane, []), list)
                else []
            )
            score_margin = float(max(0.0, internal_margin))
            if self.single_lane_confidence_mode == "internal_margin":
                conf_raw = float(score_margin / self.single_lane_internal_margin_scale)
                confidence = float(
                    max(self.single_lane_confidence_min, min(self.single_lane_confidence_max, conf_raw))
                )
            else:
                confidence = max(
                    float(self.min_route_confidence),
                    float(self.single_lane_confidence),
                )
        else:
            second_score = ranked[1][1] if len(ranked) > 1 else best_score
            score_margin = safe_float(best_score - second_score, 0.0)
            confidence = float(max(0.0, score_margin))

        no_trade_score = float(self.no_trade_bias)
        if best_score < self.min_lane_score_to_trade:
            no_trade_score += float(self.min_lane_score_to_trade - best_score)
        if score_margin < self.min_score_margin_to_trade:
            no_trade_score += float(self.min_score_margin_to_trade - score_margin)

        route_scores: Dict[str, float] = {lane: float(score) for lane, score in lane_scores.items() if score > float("-inf")}
        route_scores[self.no_trade_label] = float(no_trade_score)

        if best_score < self.min_lane_score_to_trade:
            return self._build_route_result(
                route_decision=self.no_trade_label,
                route_confidence=confidence,
                route_margin=score_margin,
                route_scores=route_scores,
                route_components=lane_components,
                route_reason="lane_score_below_trade_floor",
            )
        if single_lane_only and self.allow_single_lane_trade and score_margin < self.min_single_lane_internal_margin:
            return self._build_route_result(
                route_decision=self.no_trade_label,
                route_confidence=confidence,
                route_margin=score_margin,
                route_scores=route_scores,
                route_components=lane_components,
                route_reason="single_lane_internal_margin_below_floor",
            )
        if score_margin < self.min_score_margin_to_trade:
            return self._build_route_result(
                route_decision=self.no_trade_label,
                route_confidence=confidence,
                route_margin=score_margin,
                route_scores=route_scores,
                route_components=lane_components,
                route_reason="score_margin_below_trade_floor",
            )
        if confidence < self.min_route_confidence:
            return self._build_route_result(
                route_decision=self.no_trade_label,
                route_confidence=confidence,
                route_margin=score_margin,
                route_scores=route_scores,
                route_components=lane_components,
                route_reason="low_route_confidence",
            )

        return self._build_route_result(
            route_decision=str(best_lane),
            route_confidence=confidence,
            route_margin=score_margin,
            route_scores=route_scores,
            route_components=lane_components,
            route_reason=(
                "single_lane_route_selection"
                if (single_lane_only and self.allow_single_lane_trade)
                else "ranked_lane_selection"
            ),
        )
