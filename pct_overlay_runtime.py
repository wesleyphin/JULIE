"""Pct-level overlay runtime helpers.

Shared between the live loop (julie001.py) and the canonical backtester
(backtest_mes_et.py) so both pipelines use identical snapshot semantics.

Public API:
- get_pct_level_overlay() / init_pct_level_overlay() / update_pct_level_overlay()
- attach_pct_overlay_snapshot(signal)        # call at signal-birth
- resolve_pct_overlay_snapshot(signal)       # call before order placement
- apply_pct_level_overlay_to_signal(signal)  # legacy one-shot wrapper

Snapshot dict shape (always Python-native types so json.dumps survives):
    signal["pct_overlay_snapshot"] = {
        "size_mult": float,
        "tp_mult": float,
        "veto_reason": str | None,
        "bar_ts": str | None,
        "at_level": bool,
        "engaged": bool,
        # only when at_level=True:
        "level": float | None,
        "tier": str,
        "bias": str,
        "confidence": float,
    }
"""
from __future__ import annotations

import logging
from typing import Optional

from config import CONFIG
from pct_level_overlay import PctLevelOverlay


_PCT_LEVEL_OVERLAY: Optional[PctLevelOverlay] = None


def get_pct_level_overlay() -> Optional[PctLevelOverlay]:
    return _PCT_LEVEL_OVERLAY


def init_pct_level_overlay() -> Optional[PctLevelOverlay]:
    global _PCT_LEVEL_OVERLAY
    cfg_block = {}
    if isinstance(CONFIG, dict):
        cfg_block = CONFIG.get("PCT_LEVEL_OVERLAY") or {}
    if not bool(cfg_block.get("enabled", True)):
        _PCT_LEVEL_OVERLAY = None
        return None
    flag = cfg_block.get("feature_flag", "ENABLE_PCT_LEVEL_OVERLAY")
    if flag and isinstance(CONFIG, dict) and not bool(CONFIG.get(flag, True)):
        _PCT_LEVEL_OVERLAY = None
        return None
    _PCT_LEVEL_OVERLAY = PctLevelOverlay(cfg_block)
    return _PCT_LEVEL_OVERLAY


def update_pct_level_overlay(ts, open_, high, low, close):
    ov = _PCT_LEVEL_OVERLAY
    if ov is None:
        return None
    try:
        return ov.update(ts, float(open_), float(high), float(low), float(close))
    except Exception:
        logging.debug("pct_level_overlay update failed", exc_info=True)
        return None


def attach_pct_overlay_snapshot(signal):
    """Stamp pct overlay state onto a signal at signal-birth time.

    Idempotent: if a snapshot already exists, this is a no-op.
    """
    if not isinstance(signal, dict):
        return
    if "pct_overlay_snapshot" in signal:
        return
    ov = _PCT_LEVEL_OVERLAY
    if ov is None or not ov.enabled:
        signal["pct_overlay_snapshot"] = {
            "size_mult": 1.0,
            "tp_mult": 1.0,
            "veto_reason": None,
            "bar_ts": None,
            "at_level": False,
            "engaged": False,
        }
        return
    state = ov.state
    if not state.at_level:
        signal["pct_overlay_snapshot"] = {
            "size_mult": 1.0,
            "tp_mult": 1.0,
            "veto_reason": None,
            "bar_ts": str(state.ts) if state.ts else None,
            "at_level": False,
            "engaged": False,
        }
        return
    side = str(signal.get("side", "?") or "?")
    size_mult = float(ov.size_multiplier(side))
    mod = ov.tp_trail_modifier(side)
    tp_extend = float(mod.get("tp_extend_pct", 0.0) or 0.0)
    tp_tighten = float(mod.get("trail_tighten_pct", 0.0) or 0.0)
    if tp_extend > 0.0:
        tp_mult = 1.0 + tp_extend
    elif tp_tighten > 0.0:
        tp_mult = max(0.5, 1.0 - tp_tighten)
    else:
        tp_mult = 1.0
    veto, reason = ov.should_veto(side)
    snap = {
        "size_mult": float(size_mult),
        "tp_mult": float(tp_mult),
        "veto_reason": str(reason) if veto else None,
        "bar_ts": str(state.ts) if state.ts else None,
        "at_level": True,
        "engaged": bool(
            abs(size_mult - 1.0) > 1e-3
            or abs(tp_mult - 1.0) > 1e-3
            or veto
        ),
        "level": float(state.nearest_level) if state.nearest_level is not None else None,
        "tier": str(state.tier),
        "bias": str(state.bias),
        "confidence": float(state.confidence),
    }
    signal["pct_overlay_snapshot"] = snap
    if snap["engaged"]:
        logging.info(
            "Pct overlay snapshot: %s %s | level=%+.2f%% bias=%s conf=%.2f | size_mult=%.3f tp_mult=%.3f veto=%s",
            signal.get("strategy", "?"), side,
            state.nearest_level or 0.0, state.bias, float(state.confidence),
            size_mult, tp_mult, snap["veto_reason"] or "-",
        )


def resolve_pct_overlay_snapshot(signal) -> bool:
    """Apply snapshotted overlay decisions to a signal just before order placement.

    Returns False if the snapshot says veto (caller should reject the signal).
    Otherwise applies size_mult and tp_mult in place and returns True.
    """
    if not isinstance(signal, dict):
        return True
    snap = signal.get("pct_overlay_snapshot")
    if not isinstance(snap, dict):
        logging.warning(
            "Pct overlay: missing snapshot at order placement (%s %s) — applying no-op",
            signal.get("strategy", "?"), signal.get("side", "?"),
        )
        return True
    if signal.get("pct_overlay_snapshot_resolved"):
        return True
    signal["pct_overlay_snapshot_resolved"] = True

    if snap.get("veto_reason"):
        signal["pct_overlay_resolved_veto"] = True
        logging.info(
            "Pct overlay VETO (resolved): %s %s | reason=%s",
            signal.get("strategy", "?"), signal.get("side", "?"),
            snap["veto_reason"],
        )
        return False

    base_size = max(1, int(signal.get("size", 1) or 1))
    size_mult = float(snap.get("size_mult", 1.0) or 1.0)
    if abs(size_mult - 1.0) > 1e-3:
        scaled = max(1, int(round(base_size * size_mult)))
        if scaled != base_size:
            signal["pct_overlay_resolved_size_before"] = base_size
            signal["pct_overlay_resolved_size_after"] = scaled
            signal["pct_overlay_resolved_size_mult"] = size_mult
            signal["size"] = scaled
            logging.info(
                "Pct overlay size resolve: %s %s | snapshot mult=%.3f | size %d -> %d",
                signal.get("strategy", "?"), signal.get("side", "?"),
                size_mult, base_size, scaled,
            )

    tp_mult = float(snap.get("tp_mult", 1.0) or 1.0)
    if abs(tp_mult - 1.0) > 1e-3:
        try:
            tp_dist = float(signal.get("tp_dist", 0.0) or 0.0)
            if tp_dist > 0.0:
                new_tp = float(tp_dist * tp_mult)
                signal["pct_overlay_resolved_tp_before"] = tp_dist
                signal["pct_overlay_resolved_tp_after"] = new_tp
                signal["pct_overlay_resolved_tp_mult"] = tp_mult
                signal["tp_dist"] = new_tp
                logging.info(
                    "Pct overlay TP resolve: %s %s | snapshot mult=%.3f | tp %.2f -> %.2f",
                    signal.get("strategy", "?"), signal.get("side", "?"),
                    tp_mult, tp_dist, new_tp,
                )
        except Exception:
            pass

    return True


def apply_pct_level_overlay_to_signal(signal):
    """Legacy one-shot wrapper kept for any callers we haven't migrated.

    New code should call attach_pct_overlay_snapshot at signal birth and
    resolve_pct_overlay_snapshot immediately before order placement.
    """
    if not isinstance(signal, dict):
        return True
    ov = _PCT_LEVEL_OVERLAY
    if ov is None or not ov.enabled:
        return True
    state = ov.state
    if not state.at_level:
        signal["pct_level_overlay_applied"] = False
        return True
    side = str(signal.get("side", "?") or "?")
    size_mult = ov.size_multiplier(side)
    signal["pct_level_overlay_applied"] = True
    signal["pct_level_overlay_tier"] = state.tier
    signal["pct_level_overlay_bias"] = state.bias
    signal["pct_level_overlay_confidence"] = float(state.confidence)
    signal["pct_level_overlay_level"] = state.nearest_level
    signal["pct_level_overlay_size_multiplier"] = float(size_mult)
    base_size = max(1, int(signal.get("size", 1) or 1))
    if size_mult < 0.999 or size_mult > 1.001:
        scaled = max(1, int(round(float(base_size) * float(size_mult))))
        if scaled != base_size:
            signal["pct_level_overlay_size_before"] = int(base_size)
            signal["size"] = int(scaled)
            logging.info(
                "Pct overlay size adjust: %s %s | level=%+.2f%% bias=%s conf=%.2f | size %d -> %d",
                signal.get("strategy", "Unknown"), side,
                state.nearest_level or 0.0, state.bias,
                float(state.confidence), int(base_size), int(scaled),
            )
    mod = ov.tp_trail_modifier(side)
    tp_ext = float(mod.get("tp_extend_pct", 0.0))
    tp_tight = float(mod.get("trail_tighten_pct", 0.0))
    if tp_ext > 0.0 or tp_tight > 0.0:
        try:
            tp_dist = float(signal.get("tp_dist", 0.0) or 0.0)
            if tp_dist > 0.0:
                if tp_ext > 0.0:
                    signal["tp_dist"] = float(tp_dist * (1.0 + tp_ext))
                    signal["pct_level_overlay_tp_extend"] = tp_ext
                elif tp_tight > 0.0:
                    signal["tp_dist"] = float(tp_dist * max(0.5, 1.0 - tp_tight))
                    signal["pct_level_overlay_tp_tighten"] = tp_tight
        except Exception:
            pass
    veto, reason = ov.should_veto(side)
    if veto:
        signal["pct_level_overlay_veto"] = True
        signal["pct_level_overlay_veto_reason"] = reason
        logging.info(
            "Pct overlay VETO: %s %s | level=%+.2f%% reason=%s",
            signal.get("strategy", "Unknown"), side,
            state.nearest_level or 0.0, reason,
        )
        return False
    return True
