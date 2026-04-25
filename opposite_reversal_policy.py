from __future__ import annotations

from typing import Any, Optional


def _normalize_side(value: Any) -> Optional[str]:
    side = str(value or "").strip().upper()
    if side in {"LONG", "BUY"}:
        return "LONG"
    if side in {"SHORT", "SELL"}:
        return "SHORT"
    return None


def opposite_reversal_gate_reason(
    signal_payload: Optional[dict[str, Any]],
    active_trades_payload: Optional[list[dict[str, Any]]] = None,
    *,
    cfg: Optional[dict[str, Any]] = None,
) -> Optional[str]:
    if not isinstance(signal_payload, dict):
        return None

    policy_cfg = cfg if isinstance(cfg, dict) else {}
    if not bool(policy_cfg.get("enabled", True)):
        return "Opposite reversal disabled by config"

    allowed_regimes_raw = (
        policy_cfg.get("allowed_vol_regimes")
        or policy_cfg.get("apply_vol_regimes")
        or []
    )
    allowed_regimes = {
        str(value).strip().lower()
        for value in allowed_regimes_raw
        if str(value).strip()
    }
    if allowed_regimes:
        vol_regime = str(signal_payload.get("vol_regime") or "").strip().lower()
        if not vol_regime:
            return "Opposite reversal disabled without vol regime context"
        if vol_regime not in allowed_regimes:
            return f"Opposite reversal disabled in {vol_regime} vol"

    if not bool(policy_cfg.get("block_countertrend_in_trend_day", False)):
        return None

    side = _normalize_side(signal_payload.get("side"))
    if side is None:
        return None

    trend_day_dir = str(signal_payload.get("trend_day_dir") or "").strip().lower()
    if not trend_day_dir:
        for trade in active_trades_payload or []:
            if not isinstance(trade, dict):
                continue
            trend_day_dir = str(trade.get("trend_day_dir") or "").strip().lower()
            if trend_day_dir:
                break

    if trend_day_dir == "up" and side == "SHORT":
        return "Opposite reversal blocked counter-trend on up trend day"
    if trend_day_dir == "down" and side == "LONG":
        return "Opposite reversal blocked counter-trend on down trend day"
    return None


def reversal_confirmation_state_is_confirmed(
    state: Optional[dict[str, Any]],
    *,
    signal_side: Any,
    current_bar_index: int,
    required_confirmations: int,
    window_bars: int,
) -> bool:
    if not isinstance(state, dict):
        return False
    side = _normalize_side(signal_side)
    if side is None:
        return False
    state_side = _normalize_side(state.get("side"))
    if state_side != side:
        return False
    try:
        count = int(state.get("count", 0) or 0)
    except Exception:
        count = 0
    if count < max(1, int(required_confirmations)):
        return False
    try:
        last_bar_index = int(state.get("bar_index"))
    except Exception:
        return False
    return (int(current_bar_index) - last_bar_index) <= max(1, int(window_bars))


def update_multi_family_reversal_consensus_state(
    state: Optional[dict[str, Any]],
    *,
    signal_side: Any,
    signal_family: Optional[str],
    active_families: Optional[list[str] | tuple[str, ...] | set[str]],
    current_bar_index: int,
    window_bars: int,
) -> tuple[bool, dict[str, Any], list[str]]:
    normalized_side = _normalize_side(signal_side)
    normalized_active_families = tuple(
        sorted(
            {
                str(family).strip()
                for family in (active_families or [])
                if str(family).strip()
            }
        )
    )
    reset_state = {
        "side": None,
        "bar_index": None,
        "active_families": (),
        "family_signal_bars": {},
    }
    if normalized_side is None or not normalized_active_families:
        return False, dict(reset_state), list(normalized_active_families)

    if not isinstance(state, dict):
        state = dict(reset_state)

    prior_side = _normalize_side(state.get("side"))
    prior_active_families = tuple(
        sorted(
            {
                str(family).strip()
                for family in (state.get("active_families") or [])
                if str(family).strip()
            }
        )
    )
    raw_prior_bars = state.get("family_signal_bars") or {}
    prior_bars = {}
    if isinstance(raw_prior_bars, dict):
        for family_name, family_bar in raw_prior_bars.items():
            family_key = str(family_name).strip()
            if not family_key:
                continue
            try:
                prior_bars[family_key] = int(family_bar)
            except Exception:
                continue

    if prior_side != normalized_side or prior_active_families != normalized_active_families:
        family_signal_bars: dict[str, int] = {}
    else:
        family_signal_bars = {
            family_name: family_bar
            for family_name, family_bar in prior_bars.items()
            if family_name in normalized_active_families
            and (int(current_bar_index) - int(family_bar)) <= max(1, int(window_bars))
        }

    normalized_signal_family = str(signal_family or "").strip()
    if normalized_signal_family and normalized_signal_family in normalized_active_families:
        family_signal_bars[normalized_signal_family] = int(current_bar_index)

    missing_families = [
        family_name
        for family_name in normalized_active_families
        if family_name not in family_signal_bars
    ]
    next_state = {
        "side": normalized_side,
        "bar_index": int(current_bar_index),
        "active_families": normalized_active_families,
        "family_signal_bars": family_signal_bars,
    }
    return len(missing_families) == 0, next_state, missing_families
