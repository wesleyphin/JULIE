from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

from de3_v4_schema import (
    LANE_LONG_REV,
    LANE_ORDER,
    clip,
    safe_div,
    safe_float,
    session_distance_hours,
)


def _norm_tanh(value: float, scale: float = 1.0) -> float:
    if abs(float(scale)) <= 1e-12:
        scale = 1.0
    from math import tanh

    return float(tanh(float(value) / float(scale)))


def _core_anchor_variant(
    variants: List[Dict[str, Any]],
    core_anchor_family_ids: Iterable[str],
) -> Optional[Dict[str, Any]]:
    anchor_ids = {str(v).strip() for v in core_anchor_family_ids if str(v).strip()}
    if not variants:
        return None
    direct = [
        row
        for row in variants
        if str(row.get("family_id", "") or "") in anchor_ids
    ]
    if direct:
        return sorted(
            direct,
            key=lambda r: float(r.get("quality_proxy", 0.0) or 0.0),
            reverse=True,
        )[0]
    long_rev = [row for row in variants if str(row.get("lane", "") or "") == LANE_LONG_REV]
    if long_rev:
        return sorted(
            long_rev,
            key=lambda r: float(r.get("quality_proxy", 0.0) or 0.0),
            reverse=True,
        )[0]
    return sorted(
        variants,
        key=lambda r: float(r.get("quality_proxy", 0.0) or 0.0),
        reverse=True,
    )[0]


def _standalone_viability(row: Dict[str, Any]) -> float:
    metrics_source = str(row.get("performance_metrics_source", "") or "").strip().lower()
    if metrics_source == "fallback_support_only":
        quality_proxy = safe_float(row.get("quality_proxy", 0.0), 0.0)
        support = min(1.0, safe_div(safe_float(row.get("support_trades", 0.0), 0.0), 50000.0, 0.0))
        realized_trade_count = safe_float(row.get("realized_trade_count", 0.0), 0.0)
        realized_net = safe_float(row.get("realized_net_pnl", 0.0), 0.0)
        return float(
            (0.60 * _norm_tanh(quality_proxy, 0.20))
            + (0.25 * support)
            + (0.10 * _norm_tanh(realized_trade_count, 30.0))
            + (0.05 * _norm_tanh(realized_net, 150.0))
        )

    avg_pnl = safe_float(row.get("avg_pnl", 0.0), 0.0)
    pf = safe_float(row.get("profit_factor", 0.0), 0.0)
    win_rate = safe_float(row.get("win_rate", 0.0), 0.0)
    pbr = safe_float(row.get("profitable_block_ratio", 0.0), 0.0)
    worst_pf = safe_float(row.get("worst_block_pf", 0.0), 0.0)
    dd = safe_float(row.get("drawdown_norm", 0.0), 0.0)
    stop_share = safe_float(row.get("stop_like_share", 0.0), 0.0)
    loss_share = safe_float(row.get("loss_share", 0.0), 0.0)
    support = min(1.0, safe_div(safe_float(row.get("support_trades", 0.0), 0.0), 250.0, 0.0))
    score = (
        0.28 * _norm_tanh(avg_pnl, 1.8)
        + 0.24 * _norm_tanh(pf - 1.0, 0.50)
        + 0.12 * _norm_tanh(win_rate - 0.50, 0.15)
        + 0.12 * _norm_tanh(pbr - 0.50, 0.20)
        + 0.10 * _norm_tanh(worst_pf - 1.0, 0.40)
        + 0.08 * support
        - 0.10 * max(0.0, dd - 0.80)
        - 0.06 * max(0.0, stop_share)
        - 0.08 * max(0.0, loss_share - 0.50)
    )
    return float(score)


def _orthogonality_vs_core(row: Dict[str, Any], core: Dict[str, Any]) -> Dict[str, float]:
    lane = str(row.get("lane", "") or "")
    core_lane = str(core.get("lane", "") or "")
    tf = str(row.get("timeframe", "") or "")
    core_tf = str(core.get("timeframe", "") or "")
    sess = str(row.get("session", "") or "")
    core_sess = str(core.get("session", "") or "")
    lane_diff = 1.0 if lane != core_lane else 0.0
    tf_diff = 1.0 if tf != core_tf else 0.0
    session_dist = session_distance_hours(sess, core_sess)
    session_diff = 0.0 if session_dist is None else clip(session_dist / 12.0, 0.0, 1.0)

    overlap_rate = (
        (0.45 if lane == core_lane else 0.0)
        + (0.25 if tf == core_tf else 0.0)
        + (0.25 if sess == core_sess else 0.0)
    )
    overlap_rate = clip(overlap_rate, 0.0, 1.0)
    bad_overlap = clip(
        0.50 * overlap_rate
        + 0.25 * max(0.0, safe_float(row.get("drawdown_norm", 0.0), 0.0)),
        0.0,
        1.0,
    )

    orthogonality = clip((0.45 * lane_diff) + (0.25 * tf_diff) + (0.30 * session_diff), 0.0, 1.0)
    return {
        "orthogonality_component": float(orthogonality),
        "estimated_overlap_with_core": float(overlap_rate),
        "estimated_bad_overlap_with_core": float(bad_overlap),
    }


def _incremental_vs_core(row: Dict[str, Any], core: Dict[str, Any]) -> Dict[str, float]:
    row_metrics_source = str(row.get("performance_metrics_source", "") or "").strip().lower()
    core_metrics_source = str(core.get("performance_metrics_source", "") or "").strip().lower()
    fallback_mode = (row_metrics_source == "fallback_support_only") or (
        core_metrics_source == "fallback_support_only"
    )
    delta_avg = safe_float(row.get("avg_pnl", 0.0), 0.0) - safe_float(core.get("avg_pnl", 0.0), 0.0)
    delta_pf = safe_float(row.get("profit_factor", 0.0), 0.0) - safe_float(core.get("profit_factor", 0.0), 0.0)
    delta_dd = safe_float(core.get("drawdown_norm", 0.0), 0.0) - safe_float(row.get("drawdown_norm", 0.0), 0.0)
    delta_trades = safe_float(row.get("realized_trade_count", 0.0), 0.0) - safe_float(
        core.get("realized_trade_count", 0.0), 0.0
    )
    delta_net = safe_float(row.get("realized_net_pnl", 0.0), 0.0) - safe_float(
        core.get("realized_net_pnl", 0.0), 0.0
    )
    if fallback_mode:
        delta_quality = safe_float(row.get("quality_proxy", 0.0), 0.0) - safe_float(
            core.get("quality_proxy", 0.0), 0.0
        )
        incremental = (
            0.70 * _norm_tanh(delta_quality, 0.20)
            + 0.20 * _norm_tanh(delta_trades, 30.0)
            + 0.10 * _norm_tanh(delta_net, 150.0)
        )
    else:
        incremental = (
            0.35 * _norm_tanh(delta_avg, 1.5)
            + 0.25 * _norm_tanh(delta_pf, 0.5)
            + 0.15 * _norm_tanh(delta_dd, 0.5)
            + 0.15 * _norm_tanh(delta_net, 100.0)
            + 0.10 * _norm_tanh(delta_trades, 20.0)
        )
    return {
        "delta_avg_pnl_vs_core": float(delta_avg),
        "delta_profit_factor_vs_core": float(delta_pf),
        "delta_drawdown_vs_core": float(delta_dd),
        "delta_trade_count_vs_core": float(delta_trades),
        "delta_net_pnl_vs_core": float(delta_net),
        "incremental_component": float(incremental),
    }


def train_de3_v4_lane_selector(
    *,
    dataset: Dict[str, Any],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    variants = dataset.get("variants", []) if isinstance(dataset.get("variants"), list) else []
    core_anchor_family_ids = dataset.get("core_anchor_family_ids", [])
    lane_cfg = cfg.get("lane_selector", {}) if isinstance(cfg.get("lane_selector"), dict) else {}
    sat_cfg = cfg.get("satellites", {}) if isinstance(cfg.get("satellites"), dict) else {}

    max_variants_per_lane = max(1, int(safe_float(lane_cfg.get("max_variants_per_lane", 12), 12)))
    max_retained_satellites = max(0, int(safe_float(sat_cfg.get("max_retained_satellites", 8), 8)))
    min_standalone = safe_float(sat_cfg.get("min_standalone_viability", 0.20), 0.20)
    min_incremental = safe_float(sat_cfg.get("min_incremental_value_over_core", 0.05), 0.05)
    min_standalone_fallback = safe_float(
        sat_cfg.get("min_standalone_viability_fallback", min_standalone),
        min_standalone,
    )
    min_incremental_fallback = safe_float(
        sat_cfg.get("min_incremental_value_over_core_fallback", 0.0),
        0.0,
    )
    require_orthogonality = bool(sat_cfg.get("require_orthogonality", True))
    max_overlap = clip(safe_float(sat_cfg.get("max_overlap_with_core", 0.85), 0.85), 0.0, 1.0)
    max_bad_overlap = clip(safe_float(sat_cfg.get("max_bad_overlap_with_core", 0.60), 0.60), 0.0, 1.0)
    allow_near_core = bool(sat_cfg.get("allow_near_core_variants_if_incremental", True))

    core_variant = _core_anchor_variant(variants, core_anchor_family_ids)
    if core_variant is None:
        core_variant = {}

    rows: List[Dict[str, Any]] = []
    lane_map: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    lane_variant_quality: Dict[str, Dict[str, Any]] = {}
    core_anchor_family_id_set = {
        str(v).strip() for v in core_anchor_family_ids if str(v).strip()
    }
    for row in variants:
        if not isinstance(row, dict):
            continue
        lane = str(row.get("lane", "") or "")
        if lane not in LANE_ORDER:
            continue
        standalone = _standalone_viability(row)
        incremental_meta = _incremental_vs_core(row, core_variant)
        ortho_meta = _orthogonality_vs_core(row, core_variant)
        redundancy_penalty = (
            0.45 * safe_float(ortho_meta.get("estimated_overlap_with_core", 0.0), 0.0)
            + 0.35 * safe_float(ortho_meta.get("estimated_bad_overlap_with_core", 0.0), 0.0)
        )
        satellite_quality = (
            0.50 * standalone
            + 0.35 * safe_float(incremental_meta.get("incremental_component", 0.0), 0.0)
            + 0.25 * safe_float(ortho_meta.get("orthogonality_component", 0.0), 0.0)
            - 0.30 * redundancy_penalty
        )

        row_out = dict(row)
        row_out.update(incremental_meta)
        row_out.update(ortho_meta)
        row_out["standalone_viability_component"] = float(standalone)
        row_out["redundancy_penalty"] = float(redundancy_penalty)
        row_out["satellite_quality_score"] = float(satellite_quality)
        row_out["is_core_variant"] = bool(
            str(row.get("family_id", "") or "") in core_anchor_family_id_set
        )
        rows.append(row_out)
        lane_map[lane].append(row_out)
        lane_variant_quality[str(row.get("variant_id", ""))] = {
            "variant_id": str(row.get("variant_id", "")),
            "family_id": str(row.get("family_id", "")),
            "lane": lane,
            "standalone_viability_component": float(standalone),
            "incremental_component": float(incremental_meta.get("incremental_component", 0.0)),
            "orthogonality_component": float(ortho_meta.get("orthogonality_component", 0.0)),
            "redundancy_penalty": float(redundancy_penalty),
            "satellite_quality_score": float(satellite_quality),
            "quality_proxy": float(safe_float(row.get("quality_proxy", 0.0), 0.0)),
            "performance_metrics_source": str(row.get("performance_metrics_source", "")),
        }

    fallback_metric_rows = [
        row
        for row in rows
        if str(row.get("performance_metrics_source", "") or "").strip().lower() == "fallback_support_only"
    ]
    fallback_mode = bool(rows and (len(fallback_metric_rows) >= int(0.70 * len(rows))))
    effective_min_standalone = float(min_standalone_fallback if fallback_mode else min_standalone)
    effective_min_incremental = float(min_incremental_fallback if fallback_mode else min_incremental)

    lane_inventory: Dict[str, List[str]] = {}
    lane_anchor_variants: Dict[str, str] = {}
    lane_retained_rows: Dict[str, List[Dict[str, Any]]] = {}
    for lane in LANE_ORDER:
        variants_lane = sorted(
            lane_map.get(lane, []),
            key=lambda r: float(r.get("satellite_quality_score", 0.0) or 0.0),
            reverse=True,
        )
        kept = variants_lane[:max_variants_per_lane]
        lane_inventory[lane] = [str(r.get("variant_id", "")) for r in kept if str(r.get("variant_id", "")).strip()]
        lane_retained_rows[lane] = kept
        anchor = None
        if lane == LANE_LONG_REV:
            core_candidates = [r for r in kept if bool(r.get("is_core_variant", False))]
            if core_candidates:
                anchor = core_candidates[0]
        if anchor is None and kept:
            anchor = kept[0]
        lane_anchor_variants[lane] = str(anchor.get("variant_id", "")) if isinstance(anchor, dict) else ""

    # Satellite retention is based on additive value over core, not standalone quality only.
    satellite_rows = [
        row
        for row in rows
        if not bool(row.get("is_core_variant", False))
    ]
    for row in satellite_rows:
        standalone = safe_float(row.get("standalone_viability_component", 0.0), 0.0)
        incremental = safe_float(row.get("incremental_component", 0.0), 0.0)
        overlap = safe_float(row.get("estimated_overlap_with_core", 0.0), 0.0)
        bad_overlap = safe_float(row.get("estimated_bad_overlap_with_core", 0.0), 0.0)
        orth = safe_float(row.get("orthogonality_component", 0.0), 0.0)
        score = safe_float(row.get("satellite_quality_score", 0.0), 0.0)
        near_core = overlap >= 0.90

        passes_standalone = bool(standalone >= effective_min_standalone)
        passes_incremental = bool(incremental >= effective_min_incremental)
        passes_overlap = bool((overlap <= max_overlap) and (bad_overlap <= max_bad_overlap))
        passes_ortho = bool((not require_orthogonality) or (orth >= 0.15))
        if near_core and allow_near_core and passes_incremental:
            passes_overlap = True
            passes_ortho = True

        retained = bool(passes_standalone and passes_incremental and passes_overlap and passes_ortho)
        classification = "suppress_satellite"
        if retained and score >= 0.60:
            classification = "strong_satellite"
        elif retained:
            classification = "keep_satellite"
        elif passes_standalone and passes_incremental:
            classification = "weak_satellite"

        row["satellite_retained"] = bool(retained)
        row["satellite_classification"] = str(classification)
        row["satellite_retention_reason"] = (
            "retained_incremental"
            if retained
            else (
                "fails_incremental_or_overlap"
                if (passes_standalone and (not retained))
                else "fails_standalone"
            )
        )

    retained_satellites = sorted(
        [r for r in satellite_rows if bool(r.get("satellite_retained", False))],
        key=lambda r: float(r.get("satellite_quality_score", 0.0) or 0.0),
        reverse=True,
    )[:max_retained_satellites]
    retained_satellite_ids = {
        str(r.get("variant_id", "")) for r in retained_satellites if str(r.get("variant_id", "")).strip()
    }
    for row in satellite_rows:
        if str(row.get("variant_id", "")) not in retained_satellite_ids:
            if row.get("satellite_classification") in {"strong_satellite", "keep_satellite"}:
                row["satellite_classification"] = "weak_satellite"
            row["satellite_retained"] = False

    lane_training_report = {
        "core_anchor_reference": {
            "anchor_family_ids_configured": [str(v) for v in core_anchor_family_ids],
            "anchor_variant_id": str(core_variant.get("variant_id", "")) if isinstance(core_variant, dict) else "",
            "anchor_family_id": str(core_variant.get("family_id", "")) if isinstance(core_variant, dict) else "",
            "anchor_lane": str(core_variant.get("lane", "")) if isinstance(core_variant, dict) else "",
        },
        "lane_inventory_summary": {
            lane: {
                "retained_variant_count": int(len(lane_inventory.get(lane, []))),
                "anchor_variant_id": str(lane_anchor_variants.get(lane, "")),
            }
            for lane in LANE_ORDER
        },
        "satellite_summary": {
            "candidate_count": int(len(satellite_rows)),
            "retained_count": int(len([r for r in satellite_rows if bool(r.get("satellite_retained", False))])),
            "strong_count": int(len([r for r in satellite_rows if r.get("satellite_classification") == "strong_satellite"])),
            "keep_count": int(len([r for r in satellite_rows if r.get("satellite_classification") == "keep_satellite"])),
            "weak_count": int(len([r for r in satellite_rows if r.get("satellite_classification") == "weak_satellite"])),
            "suppress_count": int(len([r for r in satellite_rows if r.get("satellite_classification") == "suppress_satellite"])),
            "fallback_mode_active": bool(fallback_mode),
            "effective_min_standalone": float(effective_min_standalone),
            "effective_min_incremental": float(effective_min_incremental),
            "fallback_metric_variant_count": int(len(fallback_metric_rows)),
        },
    }

    return {
        "lane_inventory": lane_inventory,
        "lane_anchor_variants": lane_anchor_variants,
        "lane_variant_quality": lane_variant_quality,
        "lane_variants_retained": lane_retained_rows,
        "all_variant_rows_scored": rows,
        "satellite_rows": satellite_rows,
        "lane_training_report": lane_training_report,
    }
