import argparse
import csv
import datetime as dt
import json
import logging
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from config import CONFIG
from de3_v3_family_builder import build_de3_v3_family_inventory
from de3_v3_family_schema import canonical_member_score, safe_float


def _resolve_path(raw_path: Any) -> Path:
    out = Path(str(raw_path or "").strip())
    if not out.is_absolute():
        out = Path(__file__).resolve().parent / out
    return out


def _clip(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, value)))


def _safe_div(numer: float, denom: float, default: float = 0.0) -> float:
    d = float(denom)
    if abs(d) <= 1e-12:
        return float(default)
    return float(numer / d)


def _norm_signed(value: Any, scale: float, clip_abs: float = 2.0) -> float:
    s = float(scale) if abs(float(scale)) > 1e-12 else 1.0
    return _clip(safe_float(value, 0.0) / s, -clip_abs, clip_abs)


def _norm_centered(value: Any, center: float, scale: float, clip_abs: float = 2.0) -> float:
    s = float(scale) if abs(float(scale)) > 1e-12 else 1.0
    return _clip((safe_float(value, center) - float(center)) / s, -clip_abs, clip_abs)


def _default_context_cfg() -> Dict[str, Any]:
    de3_v3_cfg = CONFIG.get("DE3_V3", {}) if isinstance(CONFIG.get("DE3_V3", {}), dict) else {}
    cp_cfg = de3_v3_cfg.get("context_profiles", {}) if isinstance(de3_v3_cfg.get("context_profiles", {}), dict) else {}
    return dict(cp_cfg)


def _refinement_cfg() -> Dict[str, Any]:
    de3_v3_cfg = CONFIG.get("DE3_V3", {}) if isinstance(CONFIG.get("DE3_V3"), dict) else {}
    ref_cfg = de3_v3_cfg.get("refined_universe", {}) if isinstance(de3_v3_cfg.get("refined_universe"), dict) else {}
    fam_scoring = de3_v3_cfg.get("family_scoring", {}) if isinstance(de3_v3_cfg.get("family_scoring"), dict) else {}
    return {
        "enabled": bool(ref_cfg.get("enabled", True)),
        "runtime_use_refined": bool(ref_cfg.get("runtime_use_refined", True)),
        "allow_runtime_raw_universe_override": bool(ref_cfg.get("allow_runtime_raw_universe_override", True)),
        "max_retained_members_per_family": max(1, int(safe_float(ref_cfg.get("max_retained_members_per_family", 2), 2))),
        "max_retained_families": int(max(0, safe_float(ref_cfg.get("max_retained_families", 12), 12))),
        "min_family_quality_strong": safe_float(ref_cfg.get("min_family_quality_strong", 1.10), 1.10),
        "min_family_quality_keep": safe_float(ref_cfg.get("min_family_quality_keep", 0.35), 0.35),
        "min_family_quality_weak": safe_float(ref_cfg.get("min_family_quality_weak", 0.05), 0.05),
        "min_member_quality_anchor": safe_float(ref_cfg.get("min_member_quality_anchor", 0.95), 0.95),
        "min_member_quality_keep": safe_float(ref_cfg.get("min_member_quality_keep", 0.30), 0.30),
        "min_member_quality_weak": safe_float(ref_cfg.get("min_member_quality_weak", 0.00), 0.00),
        "enable_cluster_distinctiveness_filter": bool(ref_cfg.get("enable_cluster_distinctiveness_filter", True)),
        "distinctiveness_margin": max(0.0, safe_float(ref_cfg.get("distinctiveness_margin", 0.30), 0.30)),
        "allow_weak_family_if_universe_too_thin": bool(ref_cfg.get("allow_weak_family_if_universe_too_thin", True)),
        "min_retained_families": max(1, int(safe_float(ref_cfg.get("min_retained_families", 3), 3))),
        "allow_weak_member_for_diversity": bool(ref_cfg.get("allow_weak_member_for_diversity", True)),
        "member_diversity_rr_min_separation": max(0.0, safe_float(ref_cfg.get("member_diversity_rr_min_separation", 0.20), 0.20)),
        "member_diversity_sl_min_separation": max(0.0, safe_float(ref_cfg.get("member_diversity_sl_min_separation", 1.0), 1.0)),
        "require_meaningful_context_support_for_context_weight": bool(
            ref_cfg.get("require_meaningful_context_support_for_context_weight", True)
        ),
        "low_support_context_weight_cap": _clip(
            safe_float(ref_cfg.get("low_support_context_weight_cap", 0.02), 0.02),
            0.0,
            0.25,
        ),
        "active_context_dimensions": list(
            fam_scoring.get("active_context_dimensions", ["volatility_regime", "compression_expansion_regime", "confidence_band"])
            if isinstance(fam_scoring.get("active_context_dimensions"), list)
            else ["volatility_regime", "compression_expansion_regime", "confidence_band"]
        ),
    }


def _core_satellite_cfg() -> Dict[str, Any]:
    de3_v3_cfg = CONFIG.get("DE3_V3", {}) if isinstance(CONFIG.get("DE3_V3"), dict) else {}
    core_cfg = de3_v3_cfg.get("de3v3_core", {}) if isinstance(de3_v3_cfg.get("de3v3_core"), dict) else {}
    sat_cfg = de3_v3_cfg.get("de3v3_satellites", {}) if isinstance(de3_v3_cfg.get("de3v3_satellites"), dict) else {}
    bloat_cfg = de3_v3_cfg.get("bloat_control", {}) if isinstance(de3_v3_cfg.get("bloat_control"), dict) else {}
    core_ids = (
        core_cfg.get("anchor_family_ids", [])
        if isinstance(core_cfg.get("anchor_family_ids"), (list, tuple, set))
        else []
    )
    core_ids_clean = [str(v).strip() for v in core_ids if str(v).strip()]
    if not core_ids_clean:
        core_ids_clean = ["5min|09-12|long|Long_Rev|T6"]
    runtime_mode = str(
        core_cfg.get("default_runtime_mode", core_cfg.get("core_mode", "core_plus_satellites"))
        or "core_plus_satellites"
    ).strip().lower()
    if runtime_mode == "anchor_plus_satellites":
        runtime_mode = "core_plus_satellites"
    if runtime_mode not in {"core_only", "core_plus_satellites", "satellites_only"}:
        runtime_mode = "core_plus_satellites"
    return {
        "core": {
            "enabled": bool(core_cfg.get("enabled", True)),
            "anchor_family_ids": core_ids_clean,
            "default_runtime_mode": runtime_mode,
            "core_mode": str(core_cfg.get("core_mode", "anchor_plus_satellites") or "anchor_plus_satellites"),
            "force_anchor_when_eligible": bool(core_cfg.get("force_anchor_when_eligible", True)),
        },
        "satellites": {
            "enabled": bool(sat_cfg.get("enabled", True)),
            "discovery_enabled": bool(sat_cfg.get("discovery_enabled", True)),
            "min_standalone_viability": safe_float(sat_cfg.get("min_standalone_viability", 0.20), 0.20),
            "min_incremental_value_over_core": safe_float(sat_cfg.get("min_incremental_value_over_core", 0.05), 0.05),
            "max_retained_satellites": max(0, int(safe_float(sat_cfg.get("max_retained_satellites", 6), 6))),
            "require_orthogonality": bool(sat_cfg.get("require_orthogonality", True)),
            "max_overlap_with_core": _clip(safe_float(sat_cfg.get("max_overlap_with_core", 0.80), 0.80), 0.0, 1.0),
            "max_bad_overlap_with_core": _clip(safe_float(sat_cfg.get("max_bad_overlap_with_core", 0.55), 0.55), 0.0, 1.0),
            "allow_near_core_variants_if_incremental": bool(
                sat_cfg.get("allow_near_core_variants_if_incremental", True)
            ),
        },
        "bloat_control": {
            "enable_family_competition_balancing": bool(
                bloat_cfg.get("enable_family_competition_balancing", False)
            ),
            "enable_exploration_bonus": bool(
                bloat_cfg.get("enable_exploration_bonus", False)
            ),
            "enable_dominance_penalty": bool(
                bloat_cfg.get("enable_dominance_penalty", False)
            ),
            "enable_monopoly_canonical_force": bool(
                bloat_cfg.get("enable_monopoly_canonical_force", False)
            ),
            "enable_compatibility_tier_slot_pressure": bool(
                bloat_cfg.get("enable_compatibility_tier_slot_pressure", False)
            ),
        },
    }


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _extract_family_id_from_row(row: Dict[str, Any]) -> str:
    if not isinstance(row, dict):
        return ""
    for key in ("chosen_family_id", "family_id", "family_id_effective"):
        value = _safe_text(row.get(key))
        if value:
            return value
    return ""


def _extract_timestamp_from_row(row: Dict[str, Any]) -> str:
    if not isinstance(row, dict):
        return ""
    for key in ("timestamp", "decision_timestamp", "entry_time", "trade_time", "time"):
        value = _safe_text(row.get(key))
        if value:
            return value
    return ""


def _read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    p = _resolve_path(path)
    if not p.exists():
        return rows
    try:
        with p.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if isinstance(row, dict):
                    rows.append(dict(row))
    except Exception:
        return []
    return rows


def _family_event_maps(
    decisions_csv_path: Path,
    trade_attribution_csv_path: Path,
) -> Dict[str, Any]:
    decisions_rows = _read_csv_rows(decisions_csv_path)
    trade_rows = _read_csv_rows(trade_attribution_csv_path)
    family_decisions: Dict[str, List[str]] = defaultdict(list)
    family_trades: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in decisions_rows:
        if str(row.get("de3_version", "")).strip().lower() != "v3":
            continue
        if str(row.get("family_mode", "")).strip().lower() not in {"1", "true", "yes"}:
            continue
        if str(row.get("chosen", "")).strip().lower() not in {"1", "true", "yes"}:
            continue
        fid = _extract_family_id_from_row(row)
        if not fid:
            continue
        ts = _extract_timestamp_from_row(row)
        family_decisions[fid].append(ts)

    for row in trade_rows:
        fid = _extract_family_id_from_row(row)
        if not fid:
            continue
        pnl = safe_float(row.get("realized_pnl", row.get("pnl", 0.0)), 0.0)
        ts = _extract_timestamp_from_row(row)
        family_trades[fid].append(
            {
                "timestamp": ts,
                "pnl": float(pnl),
            }
        )
    return {
        "family_decisions": dict(family_decisions),
        "family_trades": dict(family_trades),
    }


def _trade_metrics(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    trade_count = int(len(events))
    pnl_values = [safe_float((row or {}).get("pnl", 0.0), 0.0) for row in events]
    net = float(sum(pnl_values))
    gp = float(sum(v for v in pnl_values if v > 0.0))
    gl = float(sum(abs(v) for v in pnl_values if v < 0.0))
    pf = gp / gl if gl > 1e-9 else (999.0 if gp > 0 else 0.0)
    return {
        "trade_count": int(trade_count),
        "net_pnl": float(net),
        "gross_profit": float(gp),
        "gross_loss_abs": float(gl),
        "profit_factor": float(pf),
        "avg_pnl": _safe_div(net, max(1, trade_count), 0.0),
    }


def _normalize_quality(value: float, scale: float = 1.0) -> float:
    s = max(1e-9, float(scale))
    return float(math.tanh(float(value) / s))


def _build_core_satellite_sections(
    *,
    families: List[Dict[str, Any]],
    refined_families: List[Dict[str, Any]],
    family_quality_rows: List[Dict[str, Any]],
    member_rows: List[Dict[str, Any]],
    anchors: Dict[str, Dict[str, Any]],
    decisions_csv_path: Path,
    trade_attribution_csv_path: Path,
) -> Dict[str, Any]:
    cfg = _core_satellite_cfg()
    core_cfg = cfg["core"]
    sat_cfg = cfg["satellites"]
    core_ids = set(core_cfg.get("anchor_family_ids", []))
    family_by_id = {
        str(row.get("family_id", "") or ""): row
        for row in families
        if isinstance(row, dict) and str(row.get("family_id", "")).strip()
    }
    refined_by_id = {
        str(row.get("family_id", "") or ""): row
        for row in refined_families
        if isinstance(row, dict) and str(row.get("family_id", "")).strip()
    }
    quality_by_id = {
        str(row.get("family_id", "") or ""): row
        for row in family_quality_rows
        if isinstance(row, dict) and str(row.get("family_id", "")).strip()
    }
    events = _family_event_maps(decisions_csv_path, trade_attribution_csv_path)
    family_decisions = events.get("family_decisions", {}) if isinstance(events.get("family_decisions"), dict) else {}
    family_trades = events.get("family_trades", {}) if isinstance(events.get("family_trades"), dict) else {}

    core_rows = []
    for fid in sorted(core_ids):
        fam = family_by_id.get(fid, {})
        qual = quality_by_id.get(fid, {})
        core_rows.append(
            {
                "family_id": fid,
                "available_in_inventory": bool(fam),
                "family_quality_score": safe_float(qual.get("family_quality_score", 0.0), 0.0),
                "family_quality_classification": str(qual.get("family_quality_classification", "") or ""),
                "retained_runtime": bool(
                    (refined_by_id.get(fid, {}) or {}).get("family_retained", False)
                ),
            }
        )
    core_member_rows = [
        row
        for row in member_rows
        if str(row.get("family_id", "") or "") in core_ids
    ]
    core_trade_events = []
    core_trade_ts: Set[str] = set()
    core_loss_ts: Set[str] = set()
    for fid in core_ids:
        trades = list(family_trades.get(fid, []))
        core_trade_events.extend(trades)
        for row in trades:
            ts = _safe_text((row or {}).get("timestamp"))
            if ts:
                core_trade_ts.add(ts)
                if safe_float((row or {}).get("pnl", 0.0), 0.0) < 0:
                    core_loss_ts.add(ts)
    core_metrics = _trade_metrics(core_trade_events)
    core_decisions = int(
        sum(len(v) for k, v in family_decisions.items() if str(k) in core_ids)
    )
    core_summary = {
        "core_family_ids": sorted(list(core_ids)),
        "core_family_count": int(len(core_ids)),
        "core_member_count": int(len(core_member_rows)),
        "t6_family_id": "5min|09-12|long|Long_Rev|T6",
        "t6_configured_as_core": bool(
            "5min|09-12|long|Long_Rev|T6" in core_ids
        ),
        "core_decision_count": int(core_decisions),
        "core_trade_metrics": dict(core_metrics),
    }

    satellite_raw = []
    satellite_scored = []
    for row in family_quality_rows:
        fid = str(row.get("family_id", "") or "")
        if not fid or fid in core_ids:
            continue
        satellite_raw.append(row)
        refined_row = refined_by_id.get(fid, {})
        trades = list(family_trades.get(fid, []))
        tmetrics = _trade_metrics(trades)
        sat_ts = {
            _safe_text((t or {}).get("timestamp"))
            for t in trades
            if _safe_text((t or {}).get("timestamp"))
        }
        overlap = int(len(sat_ts & core_trade_ts))
        overlap_rate = _safe_div(overlap, max(1, len(sat_ts)), 0.0)
        bad_overlap = int(len(sat_ts & core_loss_ts))
        bad_overlap_rate = _safe_div(bad_overlap, max(1, len(sat_ts)), 0.0)
        adds_when_core_absent = int(len([ts for ts in sat_ts if ts not in core_trade_ts]))
        add_rate = _safe_div(adds_when_core_absent, max(1, len(sat_ts)), 0.0)
        family_quality = safe_float(row.get("family_quality_score", 0.0), 0.0)
        standalone_component = _normalize_quality(family_quality, 1.0)
        delta_pf = float(
            safe_float(tmetrics.get("profit_factor", 0.0), 0.0)
            - safe_float(core_metrics.get("profit_factor", 0.0), 0.0)
        )
        incremental_component = float(
            (0.45 * _normalize_quality(safe_float(tmetrics.get("net_pnl", 0.0), 0.0), 2000.0))
            + (0.30 * _normalize_quality(delta_pf, 0.8))
            + (0.25 * _normalize_quality(add_rate, 0.4))
        )
        orthogonality_component = float(
            (0.70 * _normalize_quality(add_rate, 0.5))
            + (0.30 * _normalize_quality(1.0 - overlap_rate, 0.5))
        )
        redundancy_penalty = float(
            (0.70 * _normalize_quality(overlap_rate, 0.5))
            + (0.30 * _normalize_quality(bad_overlap_rate, 0.4))
        )
        sat_score = float(
            (0.35 * standalone_component)
            + (0.45 * incremental_component)
            + (0.30 * orthogonality_component)
            - (0.25 * redundancy_penalty)
        )
        near_core_variant = False
        fam_key = refined_row.get("family_key", {}) if isinstance(refined_row.get("family_key"), dict) else {}
        for core_id in core_ids:
            core_key = (
                (family_by_id.get(core_id, {}) or {}).get("family_key", {})
                if isinstance((family_by_id.get(core_id, {}) or {}).get("family_key", {}), dict)
                else {}
            )
            if not core_key:
                continue
            if (
                str(fam_key.get("timeframe", "") or "") == str(core_key.get("timeframe", "") or "")
                and str(fam_key.get("session", "") or "") == str(core_key.get("session", "") or "")
                and str(fam_key.get("side", "") or "") == str(core_key.get("side", "") or "")
                and str(fam_key.get("de3_strategy_type", "") or "") == str(core_key.get("de3_strategy_type", "") or "")
            ):
                near_core_variant = True
                break

        viability_ok = bool(standalone_component >= sat_cfg["min_standalone_viability"])
        incremental_ok = bool(incremental_component >= sat_cfg["min_incremental_value_over_core"])
        orth_ok = True
        orth_reason = ""
        if sat_cfg["require_orthogonality"]:
            if overlap_rate > sat_cfg["max_overlap_with_core"]:
                orth_ok = False
                orth_reason = "overlap_with_core_too_high"
            if bad_overlap_rate > sat_cfg["max_bad_overlap_with_core"]:
                orth_ok = False
                orth_reason = "bad_overlap_with_core_too_high"
        if near_core_variant and (not sat_cfg["allow_near_core_variants_if_incremental"]) and (not incremental_ok):
            orth_ok = False
            orth_reason = "near_core_variant_without_incremental_value"

        if viability_ok and incremental_ok and orth_ok and sat_score >= 0.70:
            cls = "strong_satellite"
            retain = True
            reason = "strong_incremental_value_over_core"
        elif viability_ok and incremental_ok and orth_ok:
            cls = "keep_satellite"
            retain = True
            reason = "incremental_value_over_core"
        elif viability_ok and orth_ok:
            cls = "weak_satellite"
            retain = False
            reason = "standalone_ok_but_incremental_weak"
        else:
            cls = "suppress_satellite"
            retain = False
            reason = orth_reason or "below_viability_or_incremental_threshold"

        satellite_scored.append(
            {
                "family_id": str(fid),
                "standalone_viability_component": float(standalone_component),
                "incremental_portfolio_value_component": float(incremental_component),
                "orthogonality_component": float(orthogonality_component),
                "redundancy_penalty": float(redundancy_penalty),
                "satellite_quality_score": float(sat_score),
                "satellite_classification": str(cls),
                "satellite_retained": bool(retain),
                "retained_reason": str(reason),
                "near_core_variant": bool(near_core_variant),
                "overlap_with_core_trade_rate": float(overlap_rate),
                "overlap_with_core_loser_period_rate": float(bad_overlap_rate),
                "adds_trade_when_core_absent_rate": float(add_rate),
                "standalone_trade_metrics": dict(tmetrics),
                "incremental_vs_core": {
                    "delta_net_pnl_vs_t6": float(safe_float(tmetrics.get("net_pnl", 0.0), 0.0)),
                    "delta_profit_factor_vs_t6": float(delta_pf),
                    "delta_trade_coverage_vs_t6": float(add_rate),
                    "delta_drawdown_vs_t6": float(0.0),
                },
            }
        )

    satellite_scored.sort(
        key=lambda r: float(safe_float(r.get("satellite_quality_score", 0.0), 0.0)),
        reverse=True,
    )
    max_sat = int(sat_cfg["max_retained_satellites"])
    retained_satellite_ids: List[str] = []
    if sat_cfg["enabled"]:
        for row in satellite_scored:
            if row.get("satellite_classification") not in {"strong_satellite", "keep_satellite"}:
                continue
            if max_sat > 0 and len(retained_satellite_ids) >= max_sat:
                break
            retained_satellite_ids.append(str(row.get("family_id", "")))
    retained_satellite_set = set(retained_satellite_ids)
    for row in satellite_scored:
        row["satellite_retained"] = bool(str(row.get("family_id", "")) in retained_satellite_set)
        if row["satellite_retained"] and row.get("satellite_classification") == "weak_satellite":
            row["satellite_classification"] = "keep_satellite"

    suppressed_satellite_ids = [
        str(row.get("family_id", ""))
        for row in satellite_scored
        if not bool(row.get("satellite_retained", False))
    ]
    runtime_mode_state = {
        "mode": str(core_cfg["default_runtime_mode"]),
        "core_enabled": bool(core_cfg["enabled"]),
        "satellites_enabled": bool(sat_cfg["enabled"]),
        "force_anchor_when_eligible": bool(core_cfg["force_anchor_when_eligible"]),
        "core_family_ids": sorted(list(core_ids)),
        "retained_satellite_family_ids": sorted(list(retained_satellite_set)),
        "suppressed_satellite_family_ids": sorted(list(set(suppressed_satellite_ids))),
        "core_family_count": int(len(core_ids)),
        "retained_satellite_family_count": int(len(retained_satellite_set)),
    }
    orth_summary = {
        "max_overlap_with_core": float(sat_cfg["max_overlap_with_core"]),
        "max_bad_overlap_with_core": float(sat_cfg["max_bad_overlap_with_core"]),
        "rows": [
            {
                "family_id": str(row.get("family_id", "")),
                "overlap_with_core_trade_rate": float(row.get("overlap_with_core_trade_rate", 0.0)),
                "overlap_with_core_loser_period_rate": float(row.get("overlap_with_core_loser_period_rate", 0.0)),
                "adds_trade_when_core_absent_rate": float(row.get("adds_trade_when_core_absent_rate", 0.0)),
                "near_core_variant": bool(row.get("near_core_variant", False)),
            }
            for row in satellite_scored
        ],
    }
    portfolio_increment_report = {
        "core_baseline": dict(core_metrics),
        "satellite_increment_tests": [
            {
                "family_id": str(row.get("family_id", "")),
                **dict(row.get("incremental_vs_core", {}) if isinstance(row.get("incremental_vs_core"), dict) else {}),
            }
            for row in satellite_scored
        ],
    }
    return {
        "core_cfg": dict(core_cfg),
        "satellite_cfg": dict(sat_cfg),
        "bloat_control": dict(cfg["bloat_control"]),
        "core_summary": core_summary,
        "core_families": core_rows,
        "core_members": core_member_rows,
        "satellite_candidates_raw": satellite_raw,
        "satellite_candidates_refined": satellite_scored,
        "runtime_core_satellite_state": runtime_mode_state,
        "satellite_quality_summary": {
            "satellite_count": int(len(satellite_scored)),
            "strong_satellite_count": int(sum(1 for r in satellite_scored if r.get("satellite_classification") == "strong_satellite")),
            "keep_satellite_count": int(sum(1 for r in satellite_scored if r.get("satellite_classification") == "keep_satellite")),
            "weak_satellite_count": int(sum(1 for r in satellite_scored if r.get("satellite_classification") == "weak_satellite")),
            "suppress_satellite_count": int(sum(1 for r in satellite_scored if r.get("satellite_classification") == "suppress_satellite")),
            "retained_satellite_count": int(len(retained_satellite_set)),
            "satellites": satellite_scored,
        },
        "portfolio_incremental_tests": portfolio_increment_report,
        "orthogonality_summary": orth_summary,
        "t6_anchor_report": {
            "anchor_family_id": "5min|09-12|long|Long_Rev|T6",
            "configured_as_core_anchor": bool("5min|09-12|long|Long_Rev|T6" in core_ids),
            "anchor_member": dict(anchors.get("5min|09-12|long|Long_Rev|T6", {})),
        },
    }


def _member_alias_key_from_member(member: Dict[str, Any]) -> str:
    timeframe = str(member.get("timeframe", "") or "")
    session = str(member.get("session", "") or "")
    strategy_type = str(member.get("de3_strategy_type", member.get("strategy_type", "")) or "")
    threshold = str(member.get("threshold", "") or "")
    sl_token = f"{safe_float(member.get('sl', 0.0), 0.0):.4f}"
    tp_token = f"{safe_float(member.get('tp', 0.0), 0.0):.4f}"
    return f"{timeframe}|{session}|{strategy_type}|{threshold}|SL{sl_token}|TP{tp_token}".lower()


def _member_alias_key_from_sub_strategy(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    pattern = re.compile(
        r"^(?P<tf>[^_]+)_(?P<session>\d{2}-\d{2})_(?P<stype>.+)_T(?P<thresh>[^_]+)_SL(?P<sl>[^_]+)_TP(?P<tp>[^_]+)$",
        re.IGNORECASE,
    )
    m = pattern.match(text)
    if not m:
        return text.lower()
    tf = str(m.group("tf") or "")
    session = str(m.group("session") or "")
    stype = str(m.group("stype") or "")
    thresh = f"T{str(m.group('thresh') or '')}"
    sl = f"{safe_float(m.group('sl'), 0.0):.4f}"
    tp = f"{safe_float(m.group('tp'), 0.0):.4f}"
    return f"{tf}|{session}|{stype}|{thresh}|SL{sl}|TP{tp}".lower()


def _member_realized_stats(decisions_csv_path: Path, trade_attribution_csv_path: Path) -> Dict[str, Dict[str, Any]]:
    stats: Dict[str, Dict[str, Any]] = {}
    decisions_path = _resolve_path(decisions_csv_path)
    if decisions_path.exists():
        with decisions_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if not str(row.get("de3_version", "")).strip().lower().startswith("v3"):
                    continue
                key = _member_alias_key_from_sub_strategy(row.get("sub_strategy"))
                if not key:
                    continue
                item = stats.setdefault(key, {"chosen_count": 0, "context_supported_count": 0, "fallback_count": 0, "trade_count": 0, "pnl_sum": 0.0, "gp": 0.0, "gl": 0.0})
                item["chosen_count"] += 1
                tier = str(row.get("family_context_support_tier", "low")).lower()
                if tier in {"mid", "strong"}:
                    item["context_supported_count"] += 1
                if str(row.get("family_context_fallback_priors", "true")).lower() in {"1", "true", "yes"}:
                    item["fallback_count"] += 1
    trades_path = _resolve_path(trade_attribution_csv_path)
    if trades_path.exists():
        with trades_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                key = _member_alias_key_from_sub_strategy(row.get("sub_strategy"))
                if not key:
                    continue
                item = stats.setdefault(key, {"chosen_count": 0, "context_supported_count": 0, "fallback_count": 0, "trade_count": 0, "pnl_sum": 0.0, "gp": 0.0, "gl": 0.0})
                pnl = safe_float(row.get("realized_pnl", 0.0), 0.0)
                item["trade_count"] += 1
                item["pnl_sum"] += pnl
                if pnl > 0:
                    item["gp"] += pnl
                elif pnl < 0:
                    item["gl"] += abs(pnl)
    out: Dict[str, Dict[str, Any]] = {}
    for key, raw in stats.items():
        trade_count = int(raw.get("trade_count", 0))
        gp = safe_float(raw.get("gp", 0.0), 0.0)
        gl = safe_float(raw.get("gl", 0.0), 0.0)
        pf = gp / gl if gl > 1e-9 else (999.0 if gp > 0 else 0.0)
        chosen = int(raw.get("chosen_count", 0))
        out[key] = {
            "chosen_count": chosen,
            "trade_count": trade_count,
            "avg_pnl": _safe_div(safe_float(raw.get("pnl_sum", 0.0), 0.0), max(1, trade_count), 0.0),
            "profit_factor": float(pf),
            "context_supported_rate": _safe_div(float(raw.get("context_supported_count", 0)), max(1, chosen), 0.0),
            "fallback_to_prior_rate": _safe_div(float(raw.get("fallback_count", 0)), max(1, chosen), 0.0),
        }
    return out


def _parse_threshold_value(value: Any) -> Optional[float]:
    text = str(value or "").strip()
    if not text:
        return None
    if text.upper().startswith("T"):
        text = text[1:]
    try:
        out = float(text)
        if math.isfinite(out):
            return float(out)
    except Exception:
        return None
    return None


def _family_cluster_id(family_key: Dict[str, Any]) -> str:
    timeframe = str(family_key.get("timeframe", "") or "")
    session = str(family_key.get("session", "") or "")
    side = str(family_key.get("side", "") or "")
    strategy_type = str(family_key.get("de3_strategy_type", "") or "")
    return "|".join([timeframe, session, side, strategy_type])


def _evidence_tier_rank(tier: Any) -> int:
    raw = str(tier or "").strip().lower()
    if raw == "strong":
        return 3
    if raw == "mid":
        return 2
    if raw == "low":
        return 1
    return 0


def _family_quality(family: Dict[str, Any], cfg: Dict[str, Any]) -> Tuple[float, str, Dict[str, Any]]:
    priors = family.get("family_priors") if isinstance(family.get("family_priors"), dict) else {}
    family_key = family.get("family_key") if isinstance(family.get("family_key"), dict) else {}
    state = family.get("family_runtime_state") if isinstance(family.get("family_runtime_state"), dict) else {}
    metrics = state.get("metrics") if isinstance(state.get("metrics"), dict) else {}
    prior_eligible = bool(state.get("prior_eligible", False))
    competition_status = str(state.get("competition_status", "competitive") or "competitive")
    evidence_tier = str(state.get("evidence_support_tier", "none") or "none").strip().lower()

    support_term = _clip(
        _safe_div(
            math.log1p(max(0.0, safe_float(priors.get("total_support_trades", 0.0), 0.0))),
            math.log1p(600.0),
            0.0,
        ),
        0.0,
        1.0,
    )
    best_struct_term = _norm_signed(priors.get("best_member_structural_score", 0.0), 2.0)
    median_struct_term = _norm_signed(priors.get("median_member_structural_score", 0.0), 2.0)
    best_pf_term = _norm_centered(priors.get("best_member_profit_factor", 1.0), 1.0, 0.50)
    best_avg_term = _norm_signed(priors.get("best_member_avg_pnl", 0.0), 1.4)
    best_pbr_term = _norm_centered(priors.get("best_member_profitable_block_ratio", 0.5), 0.5, 0.22)
    best_worst_pf_term = _norm_centered(priors.get("best_member_worst_block_pf", 0.8), 0.8, 0.28)

    worst_block_avg = safe_float(priors.get("best_member_worst_block_avg_pnl", 0.0), 0.0)
    worst_block_penalty = _clip(max(0.0, -worst_block_avg) / 0.8, 0.0, 2.0)
    dd_norm = _clip(_safe_div(safe_float(priors.get("median_drawdown_norm", 0.0), 0.0), 1.1, 0.0), 0.0, 2.0)
    loss_norm = _clip(_safe_div(safe_float(priors.get("median_loss_share", 0.0), 0.0), 0.72, 0.0), 0.0, 2.0)
    stop_norm = _clip(_safe_div(safe_float(priors.get("median_stop_like_share", 0.0), 0.0), 0.62, 0.0), 0.0, 2.0)
    dd_loss_combo_penalty = _clip((0.65 * dd_norm) + (0.35 * loss_norm), 0.0, 2.0)

    realized_trade_count = int(safe_float(metrics.get("executed_trade_count", 0), 0))
    realized_conf = _clip(_safe_div(math.log1p(max(0, realized_trade_count)), math.log1p(120.0), 0.0), 0.0, 1.0)
    realized_pf_term = _norm_centered(metrics.get("realized_profit_factor", 1.0), 1.0, 0.50) * realized_conf
    realized_avg_term = _norm_signed(metrics.get("realized_avg_pnl", 0.0), 1.25) * realized_conf
    realized_stop_penalty = _clip(_safe_div(safe_float(metrics.get("realized_stop_rate", 0.0), 0.0), 0.70, 0.0), 0.0, 2.0) * realized_conf
    realized_gap_penalty = _clip(_safe_div(safe_float(metrics.get("realized_stop_gap_rate", 0.0), 0.0), 0.24, 0.0), 0.0, 2.0) * realized_conf

    context_supported_rate = _clip(safe_float(metrics.get("context_supported_decision_rate", 0.0), 0.0), 0.0, 1.0)
    context_weight = 1.0
    if bool(cfg.get("require_meaningful_context_support_for_context_weight", True)) and _evidence_tier_rank(evidence_tier) <= 1:
        context_weight = min(float(context_weight), safe_float(cfg.get("low_support_context_weight_cap", 0.02), 0.02))
    context_term = float(context_supported_rate * context_weight)
    fallback_penalty = _clip(safe_float(metrics.get("fallback_to_prior_rate", 0.0), 0.0), 0.0, 1.0)

    weak_struct = safe_float(priors.get("median_member_structural_score", 0.0), 0.0) < 0.0
    low_v3_support = realized_trade_count < 6
    weak_no_support_penalty = 1.0 if (weak_struct and low_v3_support) else 0.0

    score = float(
        (0.16 * best_struct_term)
        + (0.13 * median_struct_term)
        + (0.14 * best_pf_term)
        + (0.08 * best_avg_term)
        + (0.08 * best_pbr_term)
        + (0.10 * best_worst_pf_term)
        + (0.07 * support_term)
        + (0.09 * realized_pf_term)
        + (0.06 * realized_avg_term)
        + (0.04 * context_term)
        - (0.10 * worst_block_penalty)
        - (0.10 * dd_loss_combo_penalty)
        - (0.04 * stop_norm)
        - (0.05 * fallback_penalty)
        - (0.05 * realized_stop_penalty)
        - (0.04 * realized_gap_penalty)
        - (0.07 * weak_no_support_penalty)
    )

    if (not prior_eligible) and competition_status == "suppressed":
        cls = "suppress_family"
        class_reason = "suppressed_by_prior_and_state"
    elif score >= safe_float(cfg.get("min_family_quality_strong", 1.0), 1.0):
        cls = "strong_family"
        class_reason = "family_quality_strong"
    elif score >= safe_float(cfg.get("min_family_quality_keep", 0.2), 0.2):
        cls = "keep_family"
        class_reason = "family_quality_keep"
    elif score >= safe_float(cfg.get("min_family_quality_weak", -0.2), -0.2):
        cls = "weak_family"
        class_reason = "family_quality_weak"
    else:
        cls = "suppress_family"
        class_reason = "family_quality_below_weak_threshold"

    meta = {
        "prior_eligible": bool(prior_eligible),
        "competition_status": str(competition_status),
        "evidence_support_tier": str(evidence_tier),
        "cluster_id": _family_cluster_id(family_key),
        "threshold_value": _parse_threshold_value(family_key.get("threshold")) or safe_float(family_key.get("threshold_value", float("nan")), float("nan")),
        "best_member_profit_factor": safe_float(priors.get("best_member_profit_factor", 0.0), 0.0),
        "best_member_avg_pnl": safe_float(priors.get("best_member_avg_pnl", 0.0), 0.0),
        "best_member_worst_block_pf": safe_float(priors.get("best_member_worst_block_pf", 0.0), 0.0),
        "median_structural_score": safe_float(priors.get("median_member_structural_score", 0.0), 0.0),
        "realized_trade_count": int(realized_trade_count),
        "realized_profit_factor": safe_float(metrics.get("realized_profit_factor", 0.0), 0.0),
        "realized_avg_pnl": safe_float(metrics.get("realized_avg_pnl", 0.0), 0.0),
        "context_supported_decision_rate": float(context_supported_rate),
        "fallback_to_prior_rate": float(fallback_penalty),
        "context_weight": float(context_weight),
        "class_reason": str(class_reason),
        "metrics": metrics,
    }
    return score, cls, meta


def _member_quality(member: Dict[str, Any], realized: Dict[str, Any], cfg: Dict[str, Any]) -> Tuple[float, str, Dict[str, Any]]:
    metrics = member.get("metrics") if isinstance(member.get("metrics"), dict) else {}
    realized_trades = int(safe_float(realized.get("trade_count", 0), 0))
    realized_conf = _clip(_safe_div(math.log1p(max(0, realized_trades)), math.log1p(40.0), 0.0), 0.0, 1.0)

    score = float(
        0.24 * _norm_signed(metrics.get("structural_score", canonical_member_score(member)), 2.0)
        + 0.11 * _norm_signed(metrics.get("avg_pnl", 0.0), 1.2)
        + 0.12 * _norm_centered(metrics.get("profit_factor", 1.0), 1.0, 0.5)
        + 0.07 * _norm_centered(metrics.get("win_rate", 0.5), 0.5, 0.25)
        + 0.08 * _norm_centered(metrics.get("profitable_block_ratio", 0.5), 0.5, 0.25)
        + 0.07 * _norm_centered(metrics.get("worst_block_pf", 0.8), 0.8, 0.32)
        + 0.02 * _norm_signed(metrics.get("worst_block_avg_pnl", 0.0), 0.8)
        + 0.10 * _norm_centered(realized.get("profit_factor", 1.0), 1.0, 0.5) * realized_conf
        + 0.07 * _norm_signed(realized.get("avg_pnl", 0.0), 1.1) * realized_conf
        + 0.02 * _clip(safe_float(realized.get("context_supported_rate", 0.0), 0.0), 0.0, 1.0) * realized_conf
        - 0.08 * _clip(_safe_div(safe_float(metrics.get("drawdown_norm", 0.0), 0.0), 1.15, 0.0), 0.0, 2.0)
        - 0.05 * _clip(_safe_div(safe_float(metrics.get("stop_like_share", 0.0), 0.0), 0.62, 0.0), 0.0, 2.0)
        - 0.06 * _clip(_safe_div(safe_float(metrics.get("loss_share", 0.0), 0.0), 0.72, 0.0), 0.0, 2.0)
        - 0.03 * _clip(safe_float(realized.get("fallback_to_prior_rate", 0.0), 0.0), 0.0, 1.0) * realized_conf
    )
    if score >= safe_float(cfg.get("min_member_quality_anchor", 0.85), 0.85):
        cls = "anchor_member"
    elif score >= safe_float(cfg.get("min_member_quality_keep", 0.20), 0.20):
        cls = "keep_member"
    elif score >= safe_float(cfg.get("min_member_quality_weak", -0.10), -0.10):
        cls = "weak_member"
    else:
        cls = "suppress_member"
    info = {
        "profit_factor": safe_float(metrics.get("profit_factor", 0.0), 0.0),
        "avg_pnl": safe_float(metrics.get("avg_pnl", 0.0), 0.0),
        "worst_block_pf": safe_float(metrics.get("worst_block_pf", 0.0), 0.0),
        "drawdown_norm": safe_float(metrics.get("drawdown_norm", 0.0), 0.0),
        "realized_trade_count": int(realized_trades),
        "realized_profit_factor": safe_float(realized.get("profit_factor", 0.0), 0.0),
        "realized_avg_pnl": safe_float(realized.get("avg_pnl", 0.0), 0.0),
    }
    return score, cls, info


def _is_diverse(candidate: Dict[str, Any], selected: List[Dict[str, Any]], cfg: Dict[str, Any]) -> bool:
    if not selected:
        return True
    rr_sep = safe_float(cfg.get("member_diversity_rr_min_separation", 0.20), 0.20)
    sl_sep = safe_float(cfg.get("member_diversity_sl_min_separation", 1.0), 1.0)
    rr = safe_float(candidate.get("rr", 0.0), 0.0)
    sl = safe_float(candidate.get("sl", 0.0), 0.0)
    for row in selected:
        if abs(rr - safe_float(row.get("rr", 0.0), 0.0)) < rr_sep and abs(sl - safe_float(row.get("sl", 0.0), 0.0)) < sl_sep:
            return False
    return True


def _build_refined(families: List[Dict[str, Any]], cfg: Dict[str, Any], realized_by_member: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    family_rows: List[Dict[str, Any]] = []
    for family in families:
        family_id = str(family.get("family_id", "") or "")
        score, cls, meta = _family_quality(family, cfg)
        family_rows.append(
            {
                "family_id": family_id,
                "family_quality_score": float(score),
                "family_quality_classification": str(cls),
                "meta": dict(meta),
                "cluster_id": str(meta.get("cluster_id", "") or ""),
                "suppression_reason": "" if cls != "suppress_family" else str(meta.get("class_reason", "quality_suppressed")),
                "distinctiveness_retained": True,
                "cluster_rank": 0,
                "_family": family,
            }
        )
    family_rows.sort(key=lambda r: float(r.get("family_quality_score", 0.0)), reverse=True)

    cluster_map: Dict[str, List[Dict[str, Any]]] = {}
    for row in family_rows:
        cluster_map.setdefault(str(row.get("cluster_id", "") or ""), []).append(row)

    removed_by_distinctiveness = 0
    distinctiveness_margin = safe_float(cfg.get("distinctiveness_margin", 0.35), 0.35)
    if bool(cfg.get("enable_cluster_distinctiveness_filter", True)):
        for cluster_rows in cluster_map.values():
            cluster_rows.sort(key=lambda r: float(r.get("family_quality_score", 0.0)), reverse=True)
            if not cluster_rows:
                continue
            leader = cluster_rows[0]
            leader["cluster_rank"] = 1
            leader["distinctiveness_retained"] = True
            leader["distinctiveness_reason"] = "cluster_top"
            distinct_thresholds = []
            leader_thresh = _parse_threshold_value((leader.get("meta", {}) or {}).get("threshold_value"))
            if leader_thresh is not None:
                distinct_thresholds.append(float(leader_thresh))
            for rank, row in enumerate(cluster_rows[1:], start=2):
                row["cluster_rank"] = int(rank)
                score_gap = float(leader.get("family_quality_score", 0.0) - row.get("family_quality_score", 0.0))
                row_thresh = _parse_threshold_value((row.get("meta", {}) or {}).get("threshold_value"))
                threshold_distinct = False
                if row_thresh is not None:
                    threshold_distinct = all(abs(float(row_thresh) - t) >= 1.5 for t in distinct_thresholds)
                meta = row.get("meta", {}) if isinstance(row.get("meta"), dict) else {}
                realized_trade_count = int(safe_float(meta.get("realized_trade_count", 0), 0))
                realized_pf = safe_float(meta.get("realized_profit_factor", 0.0), 0.0)
                leader_meta = leader.get("meta", {}) if isinstance(leader.get("meta"), dict) else {}
                leader_realized_pf = safe_float(leader_meta.get("realized_profit_factor", 0.0), 0.0)
                leader_best_worst_pf = safe_float(leader_meta.get("best_member_worst_block_pf", 0.0), 0.0)
                row_best_worst_pf = safe_float(meta.get("best_member_worst_block_pf", 0.0), 0.0)
                row_median_struct = safe_float(meta.get("median_structural_score", 0.0), 0.0)
                leader_median_struct = safe_float(leader_meta.get("median_structural_score", 0.0), 0.0)
                realized_distinct = bool(realized_trade_count >= 10 and realized_pf >= (leader_realized_pf + 0.15))
                robustness_distinct = bool(row_best_worst_pf >= (leader_best_worst_pf + 0.10))
                structural_distinct = bool(row_median_struct >= (leader_median_struct + 0.10))
                distinct = bool(threshold_distinct or realized_distinct or robustness_distinct or structural_distinct)
                if score_gap >= distinctiveness_margin and (not distinct):
                    row["distinctiveness_retained"] = False
                    row["distinctiveness_reason"] = "cluster_dominated"
                    if str(row.get("family_quality_classification", "")) in {"keep_family", "weak_family"}:
                        row["family_quality_classification"] = "suppress_family"
                        row["suppression_reason"] = "cluster_dominated"
                        removed_by_distinctiveness += 1
                else:
                    row["distinctiveness_retained"] = True
                    row["distinctiveness_reason"] = "distinct_or_close"
                    if threshold_distinct and row_thresh is not None:
                        distinct_thresholds.append(float(row_thresh))
    else:
        for cluster_rows in cluster_map.values():
            for rank, row in enumerate(cluster_rows, start=1):
                row["cluster_rank"] = int(rank)
                row["distinctiveness_retained"] = True
                row["distinctiveness_reason"] = "distinctiveness_filter_disabled"

    retained_ids: Set[str] = {
        str(r.get("family_id", ""))
        for r in family_rows
        if (
            str(r.get("family_quality_classification", "")) in {"strong_family", "keep_family"}
            and bool((r.get("meta", {}) if isinstance(r.get("meta"), dict) else {}).get("prior_eligible", False))
        )
    }
    min_retained = int(max(1, safe_float(cfg.get("min_retained_families", 3), 3)))
    allow_weak_for_floor = bool(cfg.get("allow_weak_family_if_universe_too_thin", True))
    cluster_present = {str((r.get("cluster_id") or "")) for r in family_rows if str(r.get("family_id", "")) in retained_ids}

    weak_candidates = [
        r
        for r in family_rows
        if (
            str(r.get("family_quality_classification", "")) == "weak_family"
            and bool((r.get("meta", {}) if isinstance(r.get("meta"), dict) else {}).get("prior_eligible", False))
        )
    ]
    weak_candidates.sort(key=lambda r: float(r.get("family_quality_score", 0.0)), reverse=True)
    weak_admitted_ids: Set[str] = set()
    for row in weak_candidates:
        meta = row.get("meta", {}) if isinstance(row.get("meta"), dict) else {}
        cluster_id = str(row.get("cluster_id", "") or "")
        moderate_inherited_quality = bool(
            safe_float(meta.get("best_member_profit_factor", 0.0), 0.0) >= 1.08
            and safe_float(meta.get("median_structural_score", 0.0), 0.0) >= 0.0
            and safe_float(meta.get("best_member_worst_block_pf", 0.0), 0.0) >= 0.90
        )
        cluster_missing = bool(cluster_id and cluster_id not in cluster_present)
        distinct_keep = bool(row.get("distinctiveness_retained", False))
        row["weak_admission_flags"] = {
            "distinctive": bool(distinct_keep),
            "moderate_inherited_quality": bool(moderate_inherited_quality),
            "cluster_missing": bool(cluster_missing),
            "thin_universe_fill": False,
        }
        if distinct_keep or moderate_inherited_quality or cluster_missing:
            weak_admitted_ids.add(str(row.get("family_id", "")))
            if cluster_missing:
                cluster_present.add(cluster_id)

    if allow_weak_for_floor and len(retained_ids) < min_retained:
        for row in weak_candidates:
            fid = str(row.get("family_id", ""))
            if not fid:
                continue
            if fid not in weak_admitted_ids:
                weak_admitted_ids.add(fid)
                flags = row.get("weak_admission_flags", {}) if isinstance(row.get("weak_admission_flags"), dict) else {}
                flags["thin_universe_fill"] = True
                row["weak_admission_flags"] = flags
            if (len(retained_ids) + len(weak_admitted_ids)) >= min_retained:
                break
    retained_ids.update(weak_admitted_ids)

    max_retained_families = int(max(0, safe_float(cfg.get("max_retained_families", 0), 0)))
    if max_retained_families > 0 and len(retained_ids) > max_retained_families:
        ranked_retained = [r for r in family_rows if str(r.get("family_id", "")) in retained_ids]
        ranked_retained.sort(key=lambda r: float(r.get("family_quality_score", 0.0)), reverse=True)
        keep_ids = {str(r.get("family_id", "")) for r in ranked_retained[:max_retained_families]}
        for row in ranked_retained[max_retained_families:]:
            row["suppression_reason"] = "max_retained_families_cap"
        retained_ids = keep_ids

    removed_by_quality = 0
    for row in family_rows:
        fid = str(row.get("family_id", ""))
        fcls = str(row.get("family_quality_classification", "suppress_family"))
        retained = bool(fid in retained_ids and fcls != "suppress_family")
        row["family_retained"] = retained
        if not retained:
            if not str(row.get("suppression_reason", "")):
                if fcls == "suppress_family":
                    row["suppression_reason"] = "quality_filter"
                    removed_by_quality += 1
                elif not bool((row.get("meta", {}) or {}).get("prior_eligible", False)):
                    row["suppression_reason"] = "prior_ineligible"
                elif fcls == "weak_family":
                    row["suppression_reason"] = "weak_not_admitted"
                else:
                    row["suppression_reason"] = "not_retained"

    member_rows: List[Dict[str, Any]] = []
    refined_families: List[Dict[str, Any]] = []
    anchors: Dict[str, Dict[str, Any]] = {}
    suppressed_members: List[Dict[str, Any]] = []
    max_members = int(max(1, safe_float(cfg.get("max_retained_members_per_family", 3), 3)))
    allow_weak = bool(cfg.get("allow_weak_member_for_diversity", True))

    family_row_by_id = {str(r.get("family_id", "")): r for r in family_rows}
    for family in families:
        family_id = str(family.get("family_id", "") or "")
        frow = family_row_by_id.get(family_id, {})
        family_retained = bool(frow.get("family_retained", False))
        raw_members = list(family.get("members", []) if isinstance(family.get("members"), list) else [])
        ranked: List[Dict[str, Any]] = []
        for member in raw_members:
            member_id = str(member.get("member_id", member.get("strategy_id", "")) or "")
            realized = realized_by_member.get(_member_alias_key_from_member(member), {})
            mscore, base_cls, minfo = _member_quality(member, realized, cfg)
            tp = max(1e-9, safe_float(member.get("tp", 0.0), 0.0))
            sl = max(1e-9, safe_float(member.get("sl", 0.0), 0.0))
            ranked.append(
                {
                    "member_id": member_id,
                    "member": member,
                    "score": float(mscore),
                    "base_cls": str(base_cls),
                    "rr": float(tp / sl),
                    "sl": float(sl),
                    "info": minfo,
                }
            )
        ranked.sort(key=lambda r: float(r["score"]), reverse=True)

        selected: List[Dict[str, Any]] = []
        anchor = ranked[0] if (family_retained and ranked) else None
        if anchor is not None:
            selected.append(anchor)
            for cand in ranked[1:]:
                if len(selected) >= max_members:
                    break
                if cand["base_cls"] == "suppress_member":
                    continue
                if cand["base_cls"] == "weak_member" and not allow_weak:
                    continue
                diverse = _is_diverse(cand, selected, cfg)
                if not diverse:
                    continue
                anchor_info = anchor.get("info", {}) if isinstance(anchor.get("info"), dict) else {}
                cand_info = cand.get("info", {}) if isinstance(cand.get("info"), dict) else {}
                improvement = bool(
                    (safe_float(cand_info.get("profit_factor", 0.0), 0.0) - safe_float(anchor_info.get("profit_factor", 0.0), 0.0) >= 0.08)
                    or (safe_float(cand_info.get("avg_pnl", 0.0), 0.0) - safe_float(anchor_info.get("avg_pnl", 0.0), 0.0) >= 0.08)
                    or (safe_float(cand_info.get("worst_block_pf", 0.0), 0.0) - safe_float(anchor_info.get("worst_block_pf", 0.0), 0.0) >= 0.08)
                    or (safe_float(anchor_info.get("drawdown_norm", 0.0), 0.0) - safe_float(cand_info.get("drawdown_norm", 0.0), 0.0) >= 0.12)
                )
                realized_useful = bool(
                    int(safe_float(cand_info.get("realized_trade_count", 0), 0)) >= 6
                    and safe_float(cand_info.get("realized_profit_factor", 0.0), 0.0) >= 1.0
                )
                quality_keep_floor = safe_float(cfg.get("min_member_quality_keep", 0.30), 0.30)
                if (improvement or realized_useful) and (
                    float(safe_float(cand.get("score", 0.0), 0.0)) >= float(quality_keep_floor) or realized_useful
                ):
                    selected.append(cand)
        selected_ids = {str(s.get("member_id", "")) for s in selected}

        retained_members: List[Dict[str, Any]] = []
        for item in ranked:
            item_info = item.get("info", {}) if isinstance(item.get("info"), dict) else {}
            is_anchor = bool(anchor is not None and item["member_id"] == anchor["member_id"] and family_retained)
            member_retained = bool(family_retained and item["member_id"] in selected_ids)
            if is_anchor:
                final_cls = "anchor_member"
                suppress_reason = ""
            elif member_retained:
                final_cls = "keep_member"
                suppress_reason = ""
            elif not family_retained:
                final_cls = "suppress_member"
                suppress_reason = "family_suppressed"
            elif item["base_cls"] == "suppress_member":
                final_cls = "suppress_member"
                suppress_reason = "member_quality_low"
            else:
                final_cls = "suppress_member"
                suppress_reason = "near_duplicate_low_value"
            row_payload = {
                "family_id": family_id,
                "member_id": item["member_id"],
                "family_retained": bool(family_retained),
                "member_quality_score": float(item["score"]),
                "base_member_quality_class": str(item["base_cls"]),
                "member_quality_classification": str(final_cls),
                "member_retained": bool(member_retained),
                "is_anchor_member": bool(is_anchor),
                "suppression_reason": str(suppress_reason),
                "rr": float(item.get("rr", 0.0)),
                "sl": float(item.get("sl", 0.0)),
                "rr_delta_vs_anchor": float(abs(float(item.get("rr", 0.0)) - float(anchor.get("rr", 0.0)))) if anchor is not None else 0.0,
                "sl_delta_vs_anchor": float(abs(float(item.get("sl", 0.0)) - float(anchor.get("sl", 0.0)))) if anchor is not None else 0.0,
                "profit_factor": safe_float(item_info.get("profit_factor", 0.0), 0.0),
                "avg_pnl": safe_float(item_info.get("avg_pnl", 0.0), 0.0),
                "worst_block_pf": safe_float(item_info.get("worst_block_pf", 0.0), 0.0),
                "drawdown_norm": safe_float(item_info.get("drawdown_norm", 0.0), 0.0),
                "realized_trade_count": int(safe_float(item_info.get("realized_trade_count", 0), 0)),
                "realized_profit_factor": safe_float(item_info.get("realized_profit_factor", 0.0), 0.0),
            }
            member_rows.append(row_payload)
            if member_retained:
                payload = dict(item["member"])
                payload["member_quality_score"] = float(item["score"])
                payload["member_quality_classification"] = final_cls
                payload["member_retained"] = True
                payload["is_anchor_member"] = bool(is_anchor)
                retained_members.append(payload)
            else:
                suppressed_members.append(dict(row_payload))

        canonical = dict(anchor["member"]) if (family_retained and anchor is not None) else {}
        if canonical:
            canonical["member_quality_score"] = float(anchor["score"])
            canonical["member_quality_classification"] = "anchor_member"
            anchors[family_id] = {
                "member_id": str(canonical.get("member_id", "") or canonical.get("strategy_id", "")),
                "sl": safe_float(canonical.get("sl", 0.0), 0.0),
                "tp": safe_float(canonical.get("tp", 0.0), 0.0),
                "anchor_score": float(anchor["score"]),
            }

        fmeta = frow.get("meta", {}) if isinstance(frow.get("meta"), dict) else {}
        refined_families.append(
            {
                "family_id": family_id,
                "family_key": dict(family.get("family_key", {}) if isinstance(family.get("family_key"), dict) else {}),
                "family_cluster_id": str(frow.get("cluster_id", "")),
                "cluster_rank": int(safe_float(frow.get("cluster_rank", 0), 0)),
                "distinctiveness_retained": bool(frow.get("distinctiveness_retained", False)),
                "distinctiveness_reason": str(frow.get("distinctiveness_reason", "")),
                "family_retained": bool(family_retained),
                "family_quality_score": float(frow.get("family_quality_score", 0.0)),
                "family_quality_classification": str(frow.get("family_quality_classification", "suppress_family")),
                "suppression_reason": str(frow.get("suppression_reason", "")),
                "prior_eligible": bool(fmeta.get("prior_eligible", False)),
                "competition_status": str(fmeta.get("competition_status", "competitive")),
                "evidence_support_tier": str(fmeta.get("evidence_support_tier", "none")),
                "context_support_summary": {
                    "context_supported_decision_rate": float(fmeta.get("context_supported_decision_rate", 0.0)),
                    "fallback_to_prior_rate": float(fmeta.get("fallback_to_prior_rate", 0.0)),
                    "context_weight": float(fmeta.get("context_weight", 1.0)),
                },
                "raw_member_count": int(len(raw_members)),
                "retained_member_count": int(len(retained_members)),
                "suppressed_member_count": int(max(0, len(raw_members) - len(retained_members))),
                "retained_member_ids": [str(m.get("member_id", m.get("strategy_id", "")) or "") for m in retained_members],
                "member_count": int(len(retained_members)),
                "member_ids": [str(m.get("member_id", m.get("strategy_id", "")) or "") for m in retained_members],
                "members": retained_members,
                "canonical_representative_member": canonical,
                "family_priors": dict(family.get("family_priors", {}) if isinstance(family.get("family_priors"), dict) else {}),
                "family_context_profiles": dict(family.get("family_context_profiles", {}) if isinstance(family.get("family_context_profiles"), dict) else {}),
                "family_runtime_state": dict(family.get("family_runtime_state", {}) if isinstance(family.get("family_runtime_state"), dict) else {}),
            }
        )

    clusters = {}
    for cid, rows in cluster_map.items():
        ranked_rows = sorted(
            rows,
            key=lambda r: float(safe_float(r.get("family_quality_score", 0.0), 0.0)),
            reverse=True,
        )
        clusters[cid] = {
            "family_count": int(len(rows)),
            "retained_count": int(sum(1 for r in rows if bool(r.get("family_retained", False)))),
            "suppressed_count": int(sum(1 for r in rows if not bool(r.get("family_retained", False)))),
            "families": [
                {
                    "family_id": str(r.get("family_id", "")),
                    "family_quality_score": float(safe_float(r.get("family_quality_score", 0.0), 0.0)),
                    "family_quality_classification": str(r.get("family_quality_classification", "")),
                    "family_retained": bool(r.get("family_retained", False)),
                    "suppression_reason": str(r.get("suppression_reason", "")),
                    "cluster_rank": int(safe_float(r.get("cluster_rank", 0), 0)),
                    "distinctiveness_retained": bool(r.get("distinctiveness_retained", False)),
                    "distinctiveness_reason": str(r.get("distinctiveness_reason", "")),
                }
                for r in ranked_rows
            ],
        }
    return {
        "family_rows": family_rows,
        "member_rows": member_rows,
        "refined_families": refined_families,
        "anchors": anchors,
        "suppressed_members": suppressed_members,
        "cluster_metadata": clusters,
        "removed_by_distinctiveness": int(removed_by_distinctiveness),
        "removed_by_quality": int(removed_by_quality),
    }


def build_de3_v3_bundle_from_inventory(
    *,
    inventory_payload: Dict[str, Any],
    source_v2_path: Path,
    decisions_csv_path: Path,
    trade_attribution_csv_path: Path,
    mode: str,
) -> Dict[str, Any]:
    families = inventory_payload.get("families") if isinstance(inventory_payload.get("families"), list) else []
    ref_cfg = _refinement_cfg()
    realized = _member_realized_stats(decisions_csv_path, trade_attribution_csv_path)
    refined = _build_refined(families, ref_cfg, realized)
    family_rows = list(refined.get("family_rows", []))
    member_rows = list(refined.get("member_rows", []))
    refined_families = list(refined.get("refined_families", []))
    anchors = dict(refined.get("anchors", {}))
    suppressed_members = list(refined.get("suppressed_members", []))
    cluster_metadata = dict(refined.get("cluster_metadata", {}))
    removed_by_distinctiveness = int(safe_float(refined.get("removed_by_distinctiveness", 0), 0))
    removed_by_quality = int(safe_float(refined.get("removed_by_quality", 0), 0))

    family_quality_rows = [{k: v for k, v in row.items() if k != "_family"} for row in family_rows]
    family_quality_rows.sort(key=lambda r: float(r.get("family_quality_score", 0.0)), reverse=True)

    family_quality_counts = {
        "strong_family": int(sum(1 for r in family_quality_rows if r.get("family_quality_classification") == "strong_family")),
        "keep_family": int(sum(1 for r in family_quality_rows if r.get("family_quality_classification") == "keep_family")),
        "weak_family": int(sum(1 for r in family_quality_rows if r.get("family_quality_classification") == "weak_family")),
        "suppress_family": int(sum(1 for r in family_quality_rows if r.get("family_quality_classification") == "suppress_family")),
    }
    member_quality_counts = {
        "anchor_member": int(sum(1 for r in member_rows if r.get("member_quality_classification") == "anchor_member")),
        "keep_member": int(sum(1 for r in member_rows if r.get("member_quality_classification") == "keep_member")),
        "weak_member": int(sum(1 for r in member_rows if r.get("member_quality_classification") == "weak_member")),
        "suppress_member": int(sum(1 for r in member_rows if r.get("member_quality_classification") == "suppress_member")),
    }
    retained_family_count = int(sum(1 for r in refined_families if bool(r.get("family_retained", False))))
    weak_but_retained_family_count = int(
        sum(
            1
            for r in refined_families
            if bool(r.get("family_retained", False))
            and str(r.get("family_quality_classification", "")) == "weak_family"
        )
    )
    raw_member_count = int(sum(int((r or {}).get("raw_member_count", 0) or 0) for r in refined_families))
    retained_member_count = int(sum(int((r or {}).get("retained_member_count", 0) or 0) for r in refined_families))
    retained_runtime_families = [r for r in refined_families if bool(r.get("family_retained", False))]
    suppressed_families = [r for r in refined_families if not bool(r.get("family_retained", False))]
    refinement_summary = {
        "raw_family_count": int(len(families)),
        "retained_family_count": int(retained_family_count),
        "weak_but_retained_family_count": int(weak_but_retained_family_count),
        "suppressed_family_count": int(max(0, len(families) - retained_family_count)),
        "raw_member_count": int(raw_member_count),
        "retained_member_count": int(retained_member_count),
        "suppressed_member_count": int(max(0, raw_member_count - retained_member_count)),
        "anchor_member_count": int(len(anchors)),
        "avg_retained_members_per_family": _safe_div(retained_member_count, max(1, retained_family_count), 0.0),
        "cluster_count": int(len(cluster_metadata)),
        "families_removed_by_distinctiveness_filter": int(removed_by_distinctiveness),
        "families_removed_by_quality_filter": int(removed_by_quality),
        "family_quality_counts": family_quality_counts,
        "member_quality_counts": member_quality_counts,
    }

    runtime_state_build = inventory_payload.get("family_runtime_state_build", {}) if isinstance(inventory_payload.get("family_runtime_state_build"), dict) else {}
    usable_summary = runtime_state_build.get("usable_universe_summary", {}) if isinstance(runtime_state_build.get("usable_universe_summary"), dict) else {}
    runtime_state_meta = runtime_state_build.get("runtime_state_meta", {}) if isinstance(runtime_state_build.get("runtime_state_meta"), dict) else {}
    family_runtime_states = {
        str(family.get("family_id", "")): dict(family.get("family_runtime_state", {}) if isinstance(family.get("family_runtime_state"), dict) else {})
        for family in families
        if str(family.get("family_id", "")).strip()
    }
    core_sat_sections = _build_core_satellite_sections(
        families=families,
        refined_families=refined_families,
        family_quality_rows=family_quality_rows,
        member_rows=member_rows,
        anchors=anchors,
        decisions_csv_path=decisions_csv_path,
        trade_attribution_csv_path=trade_attribution_csv_path,
    )
    top_share = safe_float((usable_summary.get("family_competition_health", {}) or {}).get("top_family_chosen_share", 0.0), 0.0)
    diagnostics_summary = {
        "usable_universe_summary": dict(usable_summary),
        "family_quality_counts": dict(family_quality_counts),
        "member_quality_counts": dict(member_quality_counts),
        "family_competition_health": dict(usable_summary.get("family_competition_health", {}) if isinstance(usable_summary.get("family_competition_health"), dict) else {}),
        "retained_runtime_family_count": int(len(retained_runtime_families)),
        "suppressed_family_count": int(len(suppressed_families)),
        "suppressed_member_count": int(len(suppressed_members)),
        "runtime_mode_summary": dict(core_sat_sections.get("runtime_core_satellite_state", {})),
        "warnings": ["family_choice_is_highly_concentrated"] if top_share >= 0.85 else [],
    }

    return {
        "bundle_version": "de3_v3_bundle_v2",
        "metadata": {
            "schema_version": "de3_v3_bundle_v2",
            "build_timestamp": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(),
            "mode": str(mode),
            "source_inputs": {
                "source_v2_path": str(source_v2_path),
                "source_v2_version": str(inventory_payload.get("source_v2_version", "") or ""),
                "decisions_csv_path": str(decisions_csv_path),
                "trade_attribution_csv_path": str(trade_attribution_csv_path),
            },
            "config_snapshot": {"BACKTEST_DE3_VERSION_OVERRIDE": CONFIG.get("BACKTEST_DE3_VERSION_OVERRIDE"), "DE3_V3": CONFIG.get("DE3_V3", {})},
        },
        "raw_family_inventory": {"family_count": int(len(families)), "families": list(families)},
        "refined_family_inventory": {"family_count": int(len(refined_families)), "families": refined_families},
        "retained_runtime_universe": {"family_count": int(len(retained_runtime_families)), "families": retained_runtime_families},
        "suppressed_families": {
            "family_count": int(len(suppressed_families)),
            "families": [
                {
                    "family_id": str(row.get("family_id", "")),
                    "family_quality_score": float(row.get("family_quality_score", 0.0)),
                    "family_quality_classification": str(row.get("family_quality_classification", "")),
                    "suppression_reason": str(row.get("suppression_reason", "")),
                    "family_cluster_id": str(row.get("family_cluster_id", "")),
                    "cluster_rank": int(safe_float(row.get("cluster_rank", 0), 0)),
                    "distinctiveness_reason": str(row.get("distinctiveness_reason", "")),
                }
                for row in suppressed_families
            ],
        },
        "suppressed_members": {"member_count": int(len(suppressed_members)), "members": suppressed_members},
        "family_cluster_distinctiveness": {"cluster_count": int(len(cluster_metadata)), "clusters": cluster_metadata},
        "core_summary": dict(core_sat_sections.get("core_summary", {})),
        "t6_anchor_report": dict(core_sat_sections.get("t6_anchor_report", {})),
        "core_families": {
            "count": int(len(core_sat_sections.get("core_families", []))),
            "families": list(core_sat_sections.get("core_families", [])),
        },
        "core_members": {
            "count": int(len(core_sat_sections.get("core_members", []))),
            "members": list(core_sat_sections.get("core_members", [])),
        },
        "satellite_candidates_raw": {
            "count": int(len(core_sat_sections.get("satellite_candidates_raw", []))),
            "families": list(core_sat_sections.get("satellite_candidates_raw", [])),
        },
        "satellite_candidates_refined": {
            "count": int(len(core_sat_sections.get("satellite_candidates_refined", []))),
            "families": list(core_sat_sections.get("satellite_candidates_refined", [])),
        },
        "satellite_quality_summary": dict(core_sat_sections.get("satellite_quality_summary", {})),
        "portfolio_incremental_tests": dict(core_sat_sections.get("portfolio_incremental_tests", {})),
        "orthogonality_summary": dict(core_sat_sections.get("orthogonality_summary", {})),
        "runtime_core_satellite_state": dict(core_sat_sections.get("runtime_core_satellite_state", {})),
        # Backward-compatible alias used by runtime/backtest report writers.
        "runtime_mode_summary": dict(core_sat_sections.get("runtime_core_satellite_state", {})),
        "family_quality_summary": {"counts": family_quality_counts, "families": family_quality_rows},
        "member_quality_summary": {"counts": member_quality_counts, "members": member_rows},
        "anchor_members": {"count": int(len(anchors)), "by_family_id": anchors},
        "context_profiles": {
            "enabled": bool(inventory_payload.get("family_context_profiles_enabled", True)),
            "active_dimensions": list((inventory_payload.get("family_context_profile_build", {}) or {}).get("active_dimensions", [])) if isinstance(inventory_payload.get("family_context_profile_build"), dict) else [],
            "build_meta": dict(inventory_payload.get("family_context_profile_build", {}) if isinstance(inventory_payload.get("family_context_profile_build"), dict) else {}),
        },
        "runtime_state_defaults": {
            "runtime_state_meta": dict(runtime_state_meta),
            "usable_universe_summary": dict(usable_summary),
            "family_runtime_states": dict(family_runtime_states),
            "runtime_state_json_path": runtime_state_build.get("runtime_state_json_path"),
            "runtime_use_refined_default": bool(ref_cfg.get("runtime_use_refined", True)),
            "refinement_counts": dict(refinement_summary),
        },
        "refinement_summary": refinement_summary,
        "diagnostics_summary": diagnostics_summary,
        "legacy_family_inventory": inventory_payload,
    }


def _write_aux_reports(bundle: Dict[str, Any], aux_dir: Path) -> Dict[str, str]:
    aux_dir = _resolve_path(aux_dir)
    aux_dir.mkdir(parents=True, exist_ok=True)
    out_paths: Dict[str, str] = {}
    report_specs = [
        ("family_quality_summary", "de3_v3_family_quality_report.json"),
        ("member_quality_summary", "de3_v3_member_quality_report.json"),
        ("refinement_summary", "de3_v3_refinement_summary.json"),
        ("diagnostics_summary", "de3_v3_diagnostics_summary.json"),
        ("core_summary", "de3_v3_core_summary.json"),
        ("t6_anchor_report", "de3_v3_t6_anchor_report.json"),
        ("satellite_quality_summary", "de3_v3_satellite_quality_report.json"),
        ("portfolio_incremental_tests", "de3_v3_portfolio_increment_report.json"),
        ("runtime_core_satellite_state", "de3_v3_runtime_mode_summary.json"),
        ("orthogonality_summary", "de3_v3_orthogonality_summary.json"),
    ]
    for key, filename in report_specs:
        payload = bundle.get(key, {}) if isinstance(bundle.get(key), dict) else {}
        out = aux_dir / filename
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
        out_paths[key] = str(out)

    family_rows = list(
        ((bundle.get("family_quality_summary", {}) if isinstance(bundle.get("family_quality_summary"), dict) else {}).get("families", []))
    )
    member_rows = list(
        ((bundle.get("member_quality_summary", {}) if isinstance(bundle.get("member_quality_summary"), dict) else {}).get("members", []))
    )

    family_refinement_payload = {
        "generated_at": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(),
        "family_count": int(len(family_rows)),
        "families": family_rows,
    }
    member_refinement_payload = {
        "generated_at": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(),
        "member_count": int(len(member_rows)),
        "members": member_rows,
    }
    family_json = aux_dir / "de3_v3_family_refinement_report.json"
    member_json = aux_dir / "de3_v3_member_refinement_report.json"
    family_json.write_text(json.dumps(family_refinement_payload, indent=2, ensure_ascii=True), encoding="utf-8")
    member_json.write_text(json.dumps(member_refinement_payload, indent=2, ensure_ascii=True), encoding="utf-8")
    out_paths["family_refinement_report"] = str(family_json)
    out_paths["member_refinement_report"] = str(member_json)

    family_csv = aux_dir / "de3_v3_family_refinement_report.csv"
    member_csv = aux_dir / "de3_v3_member_refinement_report.csv"
    family_fields = [
        "family_id",
        "family_quality_score",
        "family_quality_classification",
        "family_retained",
        "suppression_reason",
        "cluster_id",
        "cluster_rank",
        "distinctiveness_retained",
        "distinctiveness_reason",
        "prior_eligible",
        "evidence_support_tier",
        "competition_status",
        "best_member_profit_factor",
        "best_member_avg_pnl",
        "best_member_worst_block_pf",
        "median_structural_score",
        "realized_trade_count",
        "realized_profit_factor",
        "realized_avg_pnl",
        "context_supported_decision_rate",
        "fallback_to_prior_rate",
        "context_weight",
    ]
    with family_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=family_fields)
        writer.writeheader()
        for row in family_rows:
            meta = row.get("meta", {}) if isinstance(row.get("meta"), dict) else {}
            writer.writerow(
                {
                    "family_id": str(row.get("family_id", "")),
                    "family_quality_score": safe_float(row.get("family_quality_score", 0.0), 0.0),
                    "family_quality_classification": str(row.get("family_quality_classification", "")),
                    "family_retained": bool(row.get("family_retained", False)),
                    "suppression_reason": str(row.get("suppression_reason", "")),
                    "cluster_id": str(row.get("cluster_id", "")),
                    "cluster_rank": int(safe_float(row.get("cluster_rank", 0), 0)),
                    "distinctiveness_retained": bool(row.get("distinctiveness_retained", False)),
                    "distinctiveness_reason": str(row.get("distinctiveness_reason", "")),
                    "prior_eligible": bool(meta.get("prior_eligible", False)),
                    "evidence_support_tier": str(meta.get("evidence_support_tier", "none")),
                    "competition_status": str(meta.get("competition_status", "competitive")),
                    "best_member_profit_factor": safe_float(meta.get("best_member_profit_factor", 0.0), 0.0),
                    "best_member_avg_pnl": safe_float(meta.get("best_member_avg_pnl", 0.0), 0.0),
                    "best_member_worst_block_pf": safe_float(meta.get("best_member_worst_block_pf", 0.0), 0.0),
                    "median_structural_score": safe_float(meta.get("median_structural_score", 0.0), 0.0),
                    "realized_trade_count": int(safe_float(meta.get("realized_trade_count", 0), 0)),
                    "realized_profit_factor": safe_float(meta.get("realized_profit_factor", 0.0), 0.0),
                    "realized_avg_pnl": safe_float(meta.get("realized_avg_pnl", 0.0), 0.0),
                    "context_supported_decision_rate": safe_float(meta.get("context_supported_decision_rate", 0.0), 0.0),
                    "fallback_to_prior_rate": safe_float(meta.get("fallback_to_prior_rate", 0.0), 0.0),
                    "context_weight": safe_float(meta.get("context_weight", 1.0), 1.0),
                }
            )
    out_paths["family_refinement_report_csv"] = str(family_csv)

    member_fields = [
        "family_id",
        "member_id",
        "member_quality_score",
        "base_member_quality_class",
        "member_quality_classification",
        "member_retained",
        "is_anchor_member",
        "family_retained",
        "suppression_reason",
        "rr",
        "sl",
        "rr_delta_vs_anchor",
        "sl_delta_vs_anchor",
        "profit_factor",
        "avg_pnl",
        "worst_block_pf",
        "drawdown_norm",
        "realized_trade_count",
        "realized_profit_factor",
    ]
    with member_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=member_fields)
        writer.writeheader()
        for row in member_rows:
            writer.writerow(
                {
                    "family_id": str(row.get("family_id", "")),
                    "member_id": str(row.get("member_id", "")),
                    "member_quality_score": safe_float(row.get("member_quality_score", 0.0), 0.0),
                    "base_member_quality_class": str(row.get("base_member_quality_class", "")),
                    "member_quality_classification": str(row.get("member_quality_classification", "")),
                    "member_retained": bool(row.get("member_retained", False)),
                    "is_anchor_member": bool(row.get("is_anchor_member", False)),
                    "family_retained": bool(row.get("family_retained", False)),
                    "suppression_reason": str(row.get("suppression_reason", "")),
                    "rr": safe_float(row.get("rr", 0.0), 0.0),
                    "sl": safe_float(row.get("sl", 0.0), 0.0),
                    "rr_delta_vs_anchor": safe_float(row.get("rr_delta_vs_anchor", 0.0), 0.0),
                    "sl_delta_vs_anchor": safe_float(row.get("sl_delta_vs_anchor", 0.0), 0.0),
                    "profit_factor": safe_float(row.get("profit_factor", 0.0), 0.0),
                    "avg_pnl": safe_float(row.get("avg_pnl", 0.0), 0.0),
                    "worst_block_pf": safe_float(row.get("worst_block_pf", 0.0), 0.0),
                    "drawdown_norm": safe_float(row.get("drawdown_norm", 0.0), 0.0),
                    "realized_trade_count": int(safe_float(row.get("realized_trade_count", 0), 0)),
                    "realized_profit_factor": safe_float(row.get("realized_profit_factor", 0.0), 0.0),
                }
            )
    out_paths["member_refinement_report_csv"] = str(member_csv)
    return out_paths


def build_and_write_de3_v3_bundle(
    *,
    source_v2_path: Path,
    out_bundle_path: Path,
    decisions_csv_path: Path,
    trade_attribution_csv_path: Path,
    out_families_path: Optional[Path] = None,
    write_legacy_family_artifact: bool = True,
    emit_aux_reports: bool = False,
    aux_dir: Optional[Path] = None,
    min_bucket_samples: int = 12,
    strong_bucket_samples: int = 40,
    context_profiles_enabled: bool = True,
    allow_parse_legacy_context_inputs: bool = True,
    mode: str = "full",
) -> Dict[str, Any]:
    mode_norm = str(mode or "full").strip().lower()
    out_bundle = _resolve_path(out_bundle_path)
    if mode_norm == "analyze-only" and out_bundle.exists():
        bundle = json.loads(out_bundle.read_text(encoding="utf-8"))
        if isinstance(bundle, dict):
            bundle.setdefault("metadata", {})
            bundle["metadata"]["analyzed_at"] = dt.datetime.now(dt.timezone.utc).astimezone().isoformat()
            out_bundle.write_text(json.dumps(bundle, indent=2, ensure_ascii=True), encoding="utf-8")
            return bundle

    inventory = build_de3_v3_family_inventory(
        source_v2_path=_resolve_path(source_v2_path),
        decision_csv_path=_resolve_path(decisions_csv_path),
        trade_attribution_csv_path=_resolve_path(trade_attribution_csv_path),
        min_bucket_samples=int(max(1, min_bucket_samples)),
        strong_bucket_samples=int(max(1, strong_bucket_samples)),
        context_profiles_enabled=bool(context_profiles_enabled),
        allow_parse_legacy_context_inputs=bool(allow_parse_legacy_context_inputs),
    )
    if bool(write_legacy_family_artifact):
        target = _resolve_path(out_families_path or "dynamic_engine3_families_v3.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(inventory, indent=2, ensure_ascii=True), encoding="utf-8")
    bundle = build_de3_v3_bundle_from_inventory(
        inventory_payload=inventory,
        source_v2_path=_resolve_path(source_v2_path),
        decisions_csv_path=_resolve_path(decisions_csv_path),
        trade_attribution_csv_path=_resolve_path(trade_attribution_csv_path),
        mode=mode_norm,
    )
    out_bundle.parent.mkdir(parents=True, exist_ok=True)
    out_bundle.write_text(json.dumps(bundle, indent=2, ensure_ascii=True), encoding="utf-8")
    if emit_aux_reports:
        aux = _write_aux_reports(bundle, _resolve_path(aux_dir or "reports/de3_v3_pipeline"))
        bundle.setdefault("metadata", {})
        bundle["metadata"]["aux_reports"] = aux
        out_bundle.write_text(json.dumps(bundle, indent=2, ensure_ascii=True), encoding="utf-8")
    return bundle


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    cp_defaults = _default_context_cfg()
    parser = argparse.ArgumentParser(description="Unified DE3v3 pipeline.")
    parser.add_argument("--source-v2", default="dynamic_engine3_strategies_v2.json")
    parser.add_argument("--decisions-csv", default=cp_defaults.get("decision_csv_path", "reports/de3_decisions.csv"))
    parser.add_argument("--trade-attribution-csv", default=cp_defaults.get("trade_attribution_csv_path", "reports/de3_decisions_trade_attribution.csv"))
    parser.add_argument("--out-bundle", default="dynamic_engine3_v3_bundle.json")
    parser.add_argument("--out-families", default="dynamic_engine3_families_v3.json")
    parser.add_argument("--skip-legacy-family-artifact", action="store_true")
    parser.add_argument("--emit-aux-reports", action="store_true")
    parser.add_argument("--aux-dir", default="reports/de3_v3_pipeline")
    parser.add_argument("--mode", default="full", choices=["full", "raw-only", "refine-only", "analyze-only"])
    parser.add_argument("--min-bucket-samples", type=int, default=int(safe_float(cp_defaults.get("min_bucket_samples", 12), 12)))
    parser.add_argument("--strong-bucket-samples", type=int, default=int(safe_float(cp_defaults.get("strong_bucket_samples", 40), 40)))
    parser.add_argument("--disable-context-profiles", action="store_true")
    parser.add_argument("--disable-legacy-context-parse", action="store_true")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = _parse_args(argv)
    bundle = build_and_write_de3_v3_bundle(
        source_v2_path=_resolve_path(args.source_v2),
        out_bundle_path=_resolve_path(args.out_bundle),
        decisions_csv_path=_resolve_path(args.decisions_csv),
        trade_attribution_csv_path=_resolve_path(args.trade_attribution_csv),
        out_families_path=_resolve_path(args.out_families),
        write_legacy_family_artifact=not bool(args.skip_legacy_family_artifact),
        emit_aux_reports=bool(args.emit_aux_reports),
        aux_dir=_resolve_path(args.aux_dir),
        min_bucket_samples=args.min_bucket_samples,
        strong_bucket_samples=args.strong_bucket_samples,
        context_profiles_enabled=not bool(args.disable_context_profiles),
        allow_parse_legacy_context_inputs=not bool(args.disable_legacy_context_parse),
        mode=str(args.mode or "full"),
    )
    ref = bundle.get("refinement_summary", {}) if isinstance(bundle.get("refinement_summary"), dict) else {}
    logging.info("DE3v3 bundle built: raw_families=%s retained_families=%s raw_members=%s retained_members=%s", ref.get("raw_family_count"), ref.get("retained_family_count"), ref.get("raw_member_count"), ref.get("retained_member_count"))


if __name__ == "__main__":
    main()
