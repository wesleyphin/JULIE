import ast
import json
import math
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

FAMILY_CONTEXT_DIMENSIONS = (
    "volatility_regime",
    "chop_trend_regime",
    "compression_expansion_regime",
    "confidence_band",
    "rvol_liquidity_state",
    "session_substate",
)

ACTIVE_FAMILY_CONTEXT_DIMENSIONS = (
    "volatility_regime",
    "compression_expansion_regime",
    "confidence_band",
)

INACTIVE_FAMILY_CONTEXT_DIMENSIONS = (
    "chop_trend_regime",
    "rvol_liquidity_state",
    "session_substate",
)

FAMILY_CONTEXT_BUCKETS: Dict[str, List[str]] = {
    "volatility_regime": ["low", "mid", "high"],
    "chop_trend_regime": ["chop", "neutral", "trend"],
    "compression_expansion_regime": ["compressed", "neutral", "expanding"],
    "confidence_band": ["low", "mid", "high"],
    "rvol_liquidity_state": ["low", "normal", "high"],
    "session_substate": [],
}

ACTIVE_REQUIRED_CONTEXT_FIELDS = (
    "ctx_volatility_regime",
    "ctx_compression_expansion_regime",
    "ctx_confidence_band",
)

EVIDENCE_SUPPORT_TIERS = ("none", "low", "mid", "strong")
COMPETITION_STATUSES = ("competitive", "competitive_bootstrap", "fallback_only", "suppressed")
COMPETITIVE_COMPETITION_STATUSES = ("competitive", "competitive_bootstrap", "fallback_only")

# Backward-compatible alias used by builder audit/parsing call sites.
REQUIRED_CONTEXT_FIELDS = ACTIVE_REQUIRED_CONTEXT_FIELDS

OPTIONAL_CONTEXT_FIELDS = (
    "ctx_chop_trend_regime",
    "ctx_rvol_liquidity_state",
    "ctx_session_substate",
    "ctx_atr_ratio",
    "ctx_vwap_dist_atr",
    "ctx_price_location",
    "ctx_rvol_ratio",
    "ctx_hour_et",
)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return float(out)


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _fmt_num_token(value: Any, *, precision: int = 6) -> str:
    out = safe_float(value, float("nan"))
    if not math.isfinite(out):
        return ""
    if abs(out - round(out)) < 1e-9:
        return str(int(round(out)))
    return f"{out:.{precision}f}".rstrip("0").rstrip(".")


def _norm_text(value: Any) -> str:
    return str(value or "").strip().lower()


def infer_side(*, signal: Any = None, strategy_type: Any = None) -> str:
    raw_sig = _norm_text(signal)
    if raw_sig in {"long", "short"}:
        return raw_sig
    stype = _norm_text(strategy_type)
    if "long" in stype:
        return "long"
    if "short" in stype:
        return "short"
    return ""


def threshold_bucket(threshold: Any) -> str:
    val = safe_float(threshold, float("nan"))
    if not math.isfinite(val):
        return "TNA"
    return f"T{_fmt_num_token(val, precision=4)}"


def normalize_timeframe(value: Any) -> str:
    return str(value or "").strip().lower()


def normalize_session(value: Any, *, default_session: Optional[str] = None) -> str:
    out = str(value or default_session or "").strip().upper()
    return out


def normalize_strategy_type(value: Any) -> str:
    return str(value or "").strip()


def build_family_key_from_candidate(
    candidate: Dict[str, Any],
    *,
    default_session: Optional[str] = None,
) -> Dict[str, Any]:
    timeframe = normalize_timeframe(candidate.get("timeframe", candidate.get("TF")))
    session = normalize_session(candidate.get("session", candidate.get("Session")), default_session=default_session)
    strategy_type = normalize_strategy_type(candidate.get("strategy_type", candidate.get("Type")))
    thresh_val = safe_float(candidate.get("thresh", candidate.get("Thresh")), float("nan"))
    side = infer_side(signal=candidate.get("signal", candidate.get("Signal")), strategy_type=strategy_type)
    return {
        "timeframe": timeframe,
        "session": session,
        "side": side,
        "de3_strategy_type": strategy_type,
        "threshold": threshold_bucket(thresh_val),
        "threshold_value": thresh_val if math.isfinite(thresh_val) else None,
    }


def family_id_from_key(key: Dict[str, Any]) -> str:
    return "|".join(
        [
            str(key.get("timeframe", "") or ""),
            str(key.get("session", "") or ""),
            str(key.get("side", "") or ""),
            str(key.get("de3_strategy_type", "") or ""),
            str(key.get("threshold", "") or ""),
        ]
    )


def strategy_member_id(candidate: Dict[str, Any], *, default_session: Optional[str] = None) -> str:
    strategy_id = str(candidate.get("strategy_id", candidate.get("id", "")) or "").strip()
    if strategy_id:
        return strategy_id
    key = build_family_key_from_candidate(candidate, default_session=default_session)
    family_id = family_id_from_key(key)
    sl_token = _fmt_num_token(candidate.get("sl", candidate.get("Best_SL")))
    tp_token = _fmt_num_token(candidate.get("tp", candidate.get("Best_TP")))
    return f"{family_id}|SL{sl_token}|TP{tp_token}"


def member_metrics_from_candidate(candidate: Dict[str, Any]) -> Dict[str, float]:
    oos = candidate.get("OOS") if isinstance(candidate.get("OOS"), dict) else {}
    support_trades = safe_int(oos.get("trades", candidate.get("trades", candidate.get("Trades", 0))), 0)
    return {
        "support_trades": float(max(0, support_trades)),
        "structural_score": safe_float(
            candidate.get("StructuralScore", candidate.get("structural_score", candidate.get("de3_v2_rank_score", 0.0))),
            0.0,
        ),
        "avg_pnl": safe_float(oos.get("avg_pnl", candidate.get("avg_pnl", candidate.get("Avg_PnL", 0.0))), 0.0),
        "profit_factor": safe_float(oos.get("profit_factor", candidate.get("profit_factor", 0.0)), 0.0),
        "profitable_block_ratio": safe_float(
            candidate.get("ProfitableBlockRatio", candidate.get("profitable_block_ratio", 0.0)),
            0.0,
        ),
        "worst_block_avg_pnl": safe_float(
            candidate.get("WorstBlockAvgPnL", candidate.get("worst_block_avg_pnl", 0.0)),
            0.0,
        ),
        "worst_block_pf": safe_float(candidate.get("WorstBlockPF", candidate.get("worst_block_pf", 0.0)), 0.0),
        "drawdown_norm": safe_float(oos.get("max_oos_drawdown_norm", 0.0), 0.0),
        "stop_like_share": safe_float(oos.get("stop_like_share", candidate.get("oos_stop_like_share", 0.0)), 0.0),
        "loss_share": safe_float(oos.get("loss_share", candidate.get("oos_loss_share", 0.0)), 0.0),
    }


def empty_profile_bucket_stats() -> Dict[str, Any]:
    return {
        "sample_count": 0,
        "win_rate": 0.0,
        "avg_pnl": 0.0,
        "profit_factor": 0.0,
        "stop_rate": 0.0,
        "stop_gap_rate": 0.0,
        "avg_mae": 0.0,
        "avg_mfe": 0.0,
        "low_support": True,
        "strong_support": False,
    }


def empty_family_context_profiles() -> Dict[str, Dict[str, Dict[str, Any]]]:
    out: Dict[str, Dict[str, Dict[str, Any]]] = {
        "volatility_regime": {},
        "chop_trend_regime": {},
        "compression_expansion_regime": {},
        "confidence_band": {},
        "rvol_liquidity_state": {},
        "session_substate": {},
        "joint_profiles": {},
        "_meta": {
            "schema_version": "v1",
            "has_profile_data": False,
            "min_bucket_samples": 0,
            "strong_bucket_samples": 0,
            "decisions_with_outcomes_used": 0,
            "active_dimensions": list(ACTIVE_FAMILY_CONTEXT_DIMENSIONS),
            "inactive_dimensions": list(INACTIVE_FAMILY_CONTEXT_DIMENSIONS),
            "context_schema_audit": {},
        },
    }
    for dim, buckets in FAMILY_CONTEXT_BUCKETS.items():
        if not buckets:
            continue
        out[dim] = {bucket: empty_profile_bucket_stats() for bucket in buckets}
    return out


def percentile(values: Iterable[float], q: float, default: float = 0.0) -> float:
    arr = np.asarray([safe_float(v, float("nan")) for v in values], dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 0:
        return float(default)
    return float(np.percentile(arr, q))


def mean(values: Iterable[float], default: float = 0.0) -> float:
    arr = np.asarray([safe_float(v, float("nan")) for v in values], dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 0:
        return float(default)
    return float(np.mean(arr))


def max_value(values: Iterable[float], default: float = 0.0) -> float:
    arr = np.asarray([safe_float(v, float("nan")) for v in values], dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 0:
        return float(default)
    return float(np.max(arr))


def min_value(values: Iterable[float], default: float = 0.0) -> float:
    arr = np.asarray([safe_float(v, float("nan")) for v in values], dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 0:
        return float(default)
    return float(np.min(arr))


def canonical_member_score(member: Dict[str, Any]) -> float:
    metrics = member.get("metrics") if isinstance(member.get("metrics"), dict) else {}
    structural = safe_float(metrics.get("structural_score", 0.0), 0.0)
    avg_pnl = safe_float(metrics.get("avg_pnl", 0.0), 0.0)
    profitable_block_ratio = safe_float(metrics.get("profitable_block_ratio", 0.0), 0.0)
    profit_factor = safe_float(metrics.get("profit_factor", 0.0), 0.0)
    worst_block_avg_pnl = safe_float(metrics.get("worst_block_avg_pnl", 0.0), 0.0)
    worst_block_pf = safe_float(metrics.get("worst_block_pf", 0.0), 0.0)
    return float(
        (1.00 * structural)
        + (0.80 * avg_pnl)
        + (0.60 * profitable_block_ratio)
        + (0.45 * profit_factor)
        + (0.30 * worst_block_avg_pnl)
        + (0.15 * worst_block_pf)
    )


def compact_member_definition(candidate: Dict[str, Any], *, default_session: Optional[str] = None) -> Dict[str, Any]:
    key = build_family_key_from_candidate(candidate, default_session=default_session)
    metrics = member_metrics_from_candidate(candidate)
    member_id = strategy_member_id(candidate, default_session=default_session)
    return {
        "member_id": member_id,
        "strategy_id": member_id,
        "timeframe": key["timeframe"],
        "session": key["session"],
        "side": key["side"],
        "de3_strategy_type": key["de3_strategy_type"],
        "threshold": key["threshold"],
        "threshold_value": key["threshold_value"],
        "sl": safe_float(candidate.get("sl", candidate.get("Best_SL")), 0.0),
        "tp": safe_float(candidate.get("tp", candidate.get("Best_TP")), 0.0),
        "metrics": metrics,
    }


def parse_context_inputs_payload(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if value is None:
        return {}
    text = str(value).strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    return {}


def normalize_volatility_bucket(value: Any) -> str:
    raw = _norm_text(value)
    if raw in {"low", "ultra_low"}:
        return "low"
    if raw in {"mid", "normal", "neutral"}:
        return "mid"
    if raw in {"high", "ultra_high"}:
        return "high"
    return "mid"


def normalize_chop_trend_bucket(value: Any) -> str:
    raw = _norm_text(value)
    if raw in {"chop", "choppy", "range"}:
        return "chop"
    if raw in {"trend", "trending"}:
        return "trend"
    return "neutral"


def normalize_compression_bucket(value: Any) -> str:
    raw = _norm_text(value)
    if raw in {"compressed", "compression", "compressing"}:
        return "compressed"
    if raw in {"expanding", "expansion"}:
        return "expanding"
    return "neutral"


def normalize_confidence_bucket(value: Any) -> str:
    raw = _norm_text(value)
    if raw in {"low", "lower", "weak"}:
        return "low"
    if raw in {"high", "upper", "strong"}:
        return "high"
    return "mid"


def normalize_rvol_bucket(value: Any) -> str:
    raw = _norm_text(value)
    if raw in {"low"}:
        return "low"
    if raw in {"high"}:
        return "high"
    if raw in {"normal", "mid", "medium"}:
        return "normal"
    return "normal"


def normalize_session_substate_bucket(value: Any) -> str:
    raw = _norm_text(value)
    return raw


def normalize_context_buckets(raw_context: Dict[str, Any]) -> Dict[str, str]:
    ctx = raw_context if isinstance(raw_context, dict) else {}
    return {
        "volatility_regime": normalize_volatility_bucket(ctx.get("volatility_regime", ctx.get("ctx_volatility_regime"))),
        "chop_trend_regime": normalize_chop_trend_bucket(ctx.get("chop_trend_regime", ctx.get("ctx_chop_trend_regime"))),
        "compression_expansion_regime": normalize_compression_bucket(
            ctx.get("compression_expansion_regime", ctx.get("ctx_compression_expansion_regime"))
        ),
        "confidence_band": normalize_confidence_bucket(ctx.get("confidence_band", ctx.get("ctx_confidence_band"))),
        "rvol_liquidity_state": normalize_rvol_bucket(
            ctx.get("rvol_liquidity_state", ctx.get("ctx_rvol_liquidity_state"))
        ),
        "session_substate": normalize_session_substate_bucket(
            ctx.get("session_substate", ctx.get("ctx_session_substate"))
        ),
    }


def build_active_context_joint_key(buckets: Dict[str, Any]) -> str:
    ctx = buckets if isinstance(buckets, dict) else {}
    return "|".join(
        [
            f"vol={str(ctx.get('volatility_regime', 'mid') or 'mid')}",
            f"comp={str(ctx.get('compression_expansion_regime', 'neutral') or 'neutral')}",
            f"conf={str(ctx.get('confidence_band', 'mid') or 'mid')}",
        ]
    )


def support_tier_from_sample_count(
    sample_count: Any,
    *,
    min_samples: int,
    strong_samples: int,
) -> str:
    n = safe_int(sample_count, 0)
    min_n = max(1, int(min_samples))
    strong_n = max(min_n, int(strong_samples))
    if n >= strong_n:
        return "strong"
    if n >= min_n:
        return "mid"
    return "low"


def evidence_support_tier_from_sample_count(
    sample_count: Any,
    *,
    min_samples: int,
    strong_samples: int,
) -> str:
    n = safe_int(sample_count, 0)
    min_n = max(1, int(min_samples))
    strong_n = max(min_n, int(strong_samples))
    if n <= 0:
        return "none"
    if n >= strong_n:
        return "strong"
    if n >= min_n:
        return "mid"
    return "low"


def support_weight_for_tier(
    tier: Any,
    *,
    strong_weight: float,
    mid_weight: float,
    low_weight: float,
) -> float:
    t = _norm_text(tier)
    if t == "strong":
        return float(strong_weight)
    if t == "mid":
        return float(mid_weight)
    return float(low_weight)


def canonical_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return bool(value)
    if value is None:
        return bool(default)
    raw = _norm_text(value)
    if raw in {"1", "true", "yes", "y", "t"}:
        return True
    if raw in {"0", "false", "no", "n", "f"}:
        return False
    return bool(default)


def canonical_support_tier(
    *,
    support_tier: Any = None,
    support_ratio: Any = None,
    strong_ratio_threshold: float = 0.75,
    mid_ratio_threshold: float = 0.35,
) -> str:
    tier = _norm_text(support_tier)
    if tier in {"strong", "mid", "low"}:
        return tier
    ratio = safe_float(support_ratio, float("nan"))
    if not math.isfinite(ratio):
        return "low"
    if ratio >= float(strong_ratio_threshold):
        return "strong"
    if ratio >= float(mid_ratio_threshold):
        return "mid"
    return "low"


def canonical_evidence_support_tier(
    value: Any = None,
    *,
    sample_count: Any = None,
    min_samples: int = 1,
    strong_samples: int = 10,
) -> str:
    raw = _norm_text(value)
    if raw in EVIDENCE_SUPPORT_TIERS:
        return str(raw)
    if sample_count is None:
        return "none"
    return evidence_support_tier_from_sample_count(
        sample_count,
        min_samples=min_samples,
        strong_samples=strong_samples,
    )


def canonical_competition_status(value: Any, *, default: str = "competitive") -> str:
    raw = _norm_text(value)
    if raw in COMPETITION_STATUSES:
        return str(raw)
    return str(default if default in COMPETITION_STATUSES else "competitive")


def competition_status_is_eligible(value: Any) -> bool:
    status = canonical_competition_status(value, default="suppressed")
    return bool(status in COMPETITIVE_COMPETITION_STATUSES)


def effective_local_support_tier(
    *,
    context_support_tier: Any,
    evidence_support_tier: Any,
) -> str:
    context_tier = canonical_support_tier(support_tier=context_support_tier)
    evidence_tier = canonical_evidence_support_tier(evidence_support_tier)
    evidence_to_context = {
        "none": "low",
        "low": "low",
        "mid": "mid",
        "strong": "strong",
    }
    evidence_ctx_tier = evidence_to_context.get(evidence_tier, "low")
    order = {"low": 0, "mid": 1, "strong": 2}
    return context_tier if order.get(context_tier, 0) <= order.get(evidence_ctx_tier, 0) else evidence_ctx_tier


def canonical_local_bracket_mode(mode: Any, *, support_tier: Any = None) -> str:
    raw = _norm_text(mode)
    if raw in {"full", "conservative", "frozen", "none"}:
        return raw
    tier = canonical_support_tier(support_tier=support_tier)
    if tier == "strong":
        return "full"
    if tier == "mid":
        return "conservative"
    return "frozen"


def canonical_context_usage_snapshot(
    row: Dict[str, Any],
    *,
    strong_ratio_threshold: float = 0.75,
    mid_ratio_threshold: float = 0.35,
) -> Dict[str, Any]:
    payload = row if isinstance(row, dict) else {}
    support_ratio = safe_float(payload.get("family_context_support_ratio"), float("nan"))
    context_tier = canonical_support_tier(
        support_tier=payload.get("family_context_support_tier"),
        support_ratio=support_ratio,
        strong_ratio_threshold=strong_ratio_threshold,
        mid_ratio_threshold=mid_ratio_threshold,
    )
    evidence_raw = payload.get("family_evidence_support_tier")
    evidence_tier = canonical_evidence_support_tier(evidence_raw)
    if evidence_raw is None or str(evidence_raw).strip() == "":
        tier = context_tier
    else:
        tier = effective_local_support_tier(
            context_support_tier=context_tier,
            evidence_support_tier=evidence_tier,
        )
    trusted_val = payload.get("family_context_trusted")
    if trusted_val is None:
        trusted = bool(tier in {"mid", "strong"})
    else:
        trusted = canonical_bool(trusted_val, default=(tier in {"mid", "strong"}))
    fallback_val = payload.get("family_context_fallback_priors")
    if fallback_val is None:
        fallback = not trusted
    else:
        fallback = canonical_bool(fallback_val, default=(not trusted))
    mode = canonical_local_bracket_mode(
        payload.get("local_bracket_adaptation_mode"),
        support_tier=tier,
    )
    mode_enabled = canonical_bool(
        payload.get("local_bracket_adaptation_enabled"),
        default=(mode in {"full", "conservative"}),
    )
    override_applied = canonical_bool(
        payload.get("local_bracket_override_applied"),
        default=False,
    )
    return {
        "context_support_tier": context_tier,
        "evidence_support_tier": evidence_tier,
        "support_tier": tier,
        "trusted_context_used": bool(trusted),
        "fallback_to_priors": bool(fallback),
        "local_bracket_mode": mode,
        "local_bracket_mode_enabled": bool(mode_enabled),
        "local_bracket_override_applied": bool(override_applied),
        "support_ratio": float(support_ratio) if math.isfinite(support_ratio) else None,
    }
