from collections import defaultdict
from heapq import nlargest
from typing import Any, Dict, Iterable, List

from de3_v4_schema import LANE_ORDER, LANE_NO_TRADE, safe_div, safe_float


def _sorted_top_mean(values: Iterable[float], top_k: int = 5) -> float:
    vals = [float(v) for v in values]
    if not vals:
        return 0.0
    k = max(1, min(int(top_k), len(vals)))
    return float(sum(nlargest(k, vals)) / float(k))


def train_de3_v4_router(
    *,
    dataset: Dict[str, Any],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    variants = dataset.get("variants", []) if isinstance(dataset.get("variants"), list) else []
    router_cfg = cfg.get("router", {}) if isinstance(cfg.get("router"), dict) else {}
    weights_cfg = router_cfg.get("route_score_weights", {}) if isinstance(router_cfg.get("route_score_weights"), dict) else {}
    w_lane_prior = safe_float(weights_cfg.get("lane_prior", 0.55), 0.55)
    w_lane_max_edge = safe_float(weights_cfg.get("lane_max_edge", 0.30), 0.30)
    w_lane_mean_edge = safe_float(weights_cfg.get("lane_mean_edge", 0.15), 0.15)
    top_k = max(1, int(safe_float(router_cfg.get("lane_prior_top_k", 5), 5)))

    lane_global_scores: Dict[str, List[float]] = defaultdict(list)
    lane_session_scores: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    lane_tf_scores: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    lane_quality_input: Dict[str, Dict[str, Any]] = {lane: {"variant_count": 0} for lane in LANE_ORDER}

    for row in variants:
        if not isinstance(row, dict):
            continue
        lane = str(row.get("lane", "") or "")
        if lane not in LANE_ORDER:
            continue
        quality_proxy = safe_float(row.get("quality_proxy", 0.0), 0.0)
        support = min(1.0, safe_div(safe_float(row.get("support_trades", 0.0), 0.0), 250.0, 0.0))
        edge_score = safe_float(row.get("avg_pnl", 0.0), 0.0) + 0.65 * (safe_float(row.get("profit_factor", 0.0), 0.0) - 1.0)
        lane_signal = float(0.60 * quality_proxy + 0.30 * edge_score + 0.10 * support)
        lane_global_scores[lane].append(lane_signal)
        lane_quality_input.setdefault(lane, {"variant_count": 0})
        lane_quality_input[lane]["variant_count"] = int(lane_quality_input[lane].get("variant_count", 0) + 1)
        session = str(row.get("session", "") or "")
        timeframe = str(row.get("timeframe", "") or "")
        if session:
            lane_session_scores[session][lane].append(lane_signal)
        if timeframe:
            lane_tf_scores[timeframe][lane].append(lane_signal)

    lane_priors_global: Dict[str, float] = {}
    for lane in LANE_ORDER:
        lane_priors_global[lane] = float(_sorted_top_mean(lane_global_scores.get(lane, []), top_k=top_k))

    lane_priors_by_session: Dict[str, Dict[str, float]] = {}
    for session, lane_map in lane_session_scores.items():
        lane_priors_by_session[session] = {
            lane: float(_sorted_top_mean(lane_map.get(lane, []), top_k=top_k))
            for lane in LANE_ORDER
        }

    lane_priors_by_timeframe: Dict[str, Dict[str, float]] = {}
    for timeframe, lane_map in lane_tf_scores.items():
        lane_priors_by_timeframe[timeframe] = {
            lane: float(_sorted_top_mean(lane_map.get(lane, []), top_k=top_k))
            for lane in LANE_ORDER
        }

    no_trade_bias = safe_float(router_cfg.get("no_trade_bias", 0.15), 0.15)
    min_route_confidence = safe_float(router_cfg.get("min_route_confidence", 0.05), 0.05)
    min_lane_score_to_trade = safe_float(router_cfg.get("min_lane_score_to_trade", -0.10), -0.10)
    min_score_margin_to_trade = safe_float(router_cfg.get("min_score_margin_to_trade", 0.02), 0.02)
    min_single_lane_internal_margin = safe_float(
        router_cfg.get("min_single_lane_internal_margin", 0.02),
        0.02,
    )
    single_lane_confidence_mode = str(
        router_cfg.get("single_lane_confidence_mode", "internal_margin") or "internal_margin"
    ).strip().lower()
    if single_lane_confidence_mode not in {"fixed", "internal_margin"}:
        single_lane_confidence_mode = "internal_margin"
    single_lane_internal_margin_scale = safe_float(
        router_cfg.get("single_lane_internal_margin_scale", 0.05),
        0.05,
    )
    single_lane_confidence = safe_float(router_cfg.get("single_lane_confidence", min_route_confidence), min_route_confidence)

    router_payload = {
        "lanes": list(LANE_ORDER),
        "no_trade_label": LANE_NO_TRADE,
        "route_score_weights": {
            "lane_prior": float(w_lane_prior),
            "lane_max_edge": float(w_lane_max_edge),
            "lane_mean_edge": float(w_lane_mean_edge),
        },
        "lane_priors_global": lane_priors_global,
        "lane_priors_by_session": lane_priors_by_session,
        "lane_priors_by_timeframe": lane_priors_by_timeframe,
        "no_trade_bias": float(no_trade_bias),
        "min_route_confidence": float(min_route_confidence),
        "min_lane_score_to_trade": float(min_lane_score_to_trade),
        "min_score_margin_to_trade": float(min_score_margin_to_trade),
        "min_single_lane_internal_margin": float(min_single_lane_internal_margin),
        "single_lane_confidence_mode": str(single_lane_confidence_mode),
        "single_lane_internal_margin_scale": float(single_lane_internal_margin_scale),
        "single_lane_confidence": float(single_lane_confidence),
    }

    training_report = {
        "router_target_construction": {
            "approach": "lane_level_aggregation",
            "description": (
                "Router targets are lane-level and derived from aggregated lane quality "
                "signals (quality proxy + edge proxy + support). No exact-row winner labels are used."
            ),
            "lane_labels": [LANE_NO_TRADE] + list(LANE_ORDER),
            "no_trade_logic": (
                "No Trade is selected when best lane score is weak or route confidence "
                "is below minimum thresholds."
            ),
        },
        "router_model_summary": {
            "variant_count_used": int(len(variants)),
            "lane_quality_input": lane_quality_input,
            "lane_priors_global": lane_priors_global,
            "session_prior_count": int(len(lane_priors_by_session)),
            "timeframe_prior_count": int(len(lane_priors_by_timeframe)),
            "weights": router_payload.get("route_score_weights", {}),
            "no_trade_bias": float(no_trade_bias),
            "min_route_confidence": float(min_route_confidence),
            "min_lane_score_to_trade": float(min_lane_score_to_trade),
            "min_score_margin_to_trade": float(min_score_margin_to_trade),
            "min_single_lane_internal_margin": float(min_single_lane_internal_margin),
            "single_lane_confidence_mode": str(single_lane_confidence_mode),
            "single_lane_internal_margin_scale": float(single_lane_internal_margin_scale),
            "single_lane_confidence": float(single_lane_confidence),
        },
    }
    return {
        "router_payload": router_payload,
        "router_training_report": training_report,
    }
