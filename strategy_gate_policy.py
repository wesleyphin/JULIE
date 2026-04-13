from __future__ import annotations

from typing import Dict, Optional, Tuple


def _norm_key(value: Optional[str]) -> str:
    if value is None:
        return ""
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


DEFAULT_PROFILE_MAP: Dict[str, str] = {
    "dynamicengine": "momentum_breakout",
    "dynamicengine1": "momentum_breakout",
    "dynamicengine2": "momentum_breakout",
    "dynamicengine3": "momentum_breakout",
    "impulsebreakout": "momentum_breakout",
    "orb": "momentum_breakout",
    "orbstrategy": "momentum_breakout",
    "confluence": "momentum_breakout",
    "ictmodel": "momentum_breakout",
    "regimeadaptive": "mean_reversion",
    "auctionreversion": "mean_reversion",
    "intradaydip": "mean_reversion",
    "smtanalyzer": "mean_reversion",
    "smtstrategy": "mean_reversion",
    "liquiditysweep": "mean_reversion",
    "vixmeanreversion": "mean_reversion",
    "valueareabreakout": "momentum_breakout",
    "smoothtrendasia": "momentum_breakout",
    "mlphysics": "ml_adaptive",
    "manifoldstrategy": "ml_adaptive",
    "continuation": "momentum_breakout",
}


def resolve_strategy_profile(strategy_label: Optional[str], cfg: Optional[Dict] = None) -> str:
    profile_cfg = ((cfg or {}).get("STRATEGY_GATE_PROFILES", {}) or {})
    default_profile = str(profile_cfg.get("default_profile", "momentum_breakout"))
    raw_map = profile_cfg.get("map", {}) or {}
    merged_map: Dict[str, str] = dict(DEFAULT_PROFILE_MAP)
    for key, value in raw_map.items():
        norm = _norm_key(str(key))
        if not norm:
            continue
        merged_map[norm] = str(value)

    norm_label = _norm_key(strategy_label)
    if not norm_label:
        return default_profile

    if norm_label in merged_map:
        return merged_map[norm_label]
    for key, profile in merged_map.items():
        if key and (norm_label.startswith(key) or key in norm_label):
            return profile
    return default_profile


def evaluate_pre_signal_gate(
    *,
    cfg: Dict,
    session_name: Optional[str],
    strategy_label: Optional[str],
    side: Optional[str],
    asia_viable: Optional[bool],
    asia_reason: Optional[str],
    asia_trend_bias_side: Optional[str],
    is_choppy: bool,
    chop_reason: Optional[str],
    allowed_chop_side: Optional[str],
) -> Tuple[bool, Optional[str], Optional[str], str]:
    profile = resolve_strategy_profile(strategy_label, cfg)
    session = str(session_name or "").upper()
    side_u = str(side or "").upper()
    allowed_side_u = str(allowed_chop_side or "").upper()
    trend_bias_u = str(asia_trend_bias_side or "").upper()

    asia_cfg = cfg.get("ASIA_VIABILITY_GATE", {}) or {}
    asia_mode = str(asia_cfg.get("mode", "global")).lower()
    if session == "ASIA" and asia_mode not in ("off", "disabled", "none"):
        if not bool(asia_viable):
            reason = asia_reason or "Asia gate: no viable condition"
            if asia_mode == "global":
                return False, "AsiaViabilityGate", reason, profile
            blocked_profiles = {
                str(item).strip()
                for item in (asia_cfg.get("block_profiles_when_not_viable", ["momentum_breakout", "ml_adaptive"]) or [])
            }
            if profile in blocked_profiles:
                return (
                    False,
                    "AsiaViabilityGate",
                    f"{reason} [profile={profile}]",
                    profile,
                )
        if (
            bool(asia_cfg.get("enforce_trend_bias", True))
            and trend_bias_u in ("LONG", "SHORT")
            and side_u in ("LONG", "SHORT")
        ):
            enforce_profiles = {
                str(item).strip()
                for item in (asia_cfg.get("enforce_bias_profiles", ["momentum_breakout"]) or [])
            }
            if profile in enforce_profiles and side_u != trend_bias_u:
                return (
                    False,
                    "AsiaTrendBias",
                    f"ASIA trend bias {trend_bias_u} blocks {side_u} [profile={profile}]",
                    profile,
                )

    chop_cfg = cfg.get("DYNAMIC_CHOP_GATE", {}) or {}
    chop_mode = str(chop_cfg.get("mode", "global")).lower()
    if is_choppy and chop_mode not in ("off", "disabled", "none"):
        hard_chop = allowed_side_u not in ("LONG", "SHORT")
        if hard_chop:
            reason = chop_reason or "CHOP hard-stop"
            if chop_mode == "global":
                return False, "DynamicChop", reason, profile
            allow_profiles = {
                str(item).strip()
                for item in (chop_cfg.get("allow_profiles_in_hard_chop", ["mean_reversion"]) or [])
            }
            if profile not in allow_profiles:
                return (
                    False,
                    "DynamicChop",
                    f"{reason} [profile={profile}]",
                    profile,
                )
        elif bool(chop_cfg.get("enforce_range_bias_pre_candidate", False)):
            if side_u in ("LONG", "SHORT") and allowed_side_u in ("LONG", "SHORT") and side_u != allowed_side_u:
                return (
                    False,
                    "ChopRangeBias",
                    f"Opposite HTF Range Bias ({allowed_side_u})",
                    profile,
                )

    return True, None, None, profile
