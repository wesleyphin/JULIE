"""Runtime regime classifier: rolling close-based whipsaw vs calm-trend detector.

Public API:
- init_regime_classifier() -> Optional[RegimeClassifier]
- get_regime_classifier()   -> Optional[RegimeClassifier]
- update_regime_classifier(ts, close) -> Optional[str]   # returns current regime name
- current_regime() -> str                                # "whipsaw" | "calm_trend" | "neutral" | "warmup"

Enabled by JULIE_REGIME_CLASSIFIER=1. Default: disabled (module is a no-op).

When a regime transition occurs, the classifier mutates CONFIG in place:

    WHIPSAW:
        KALSHI_TRADE_OVERLAY.entry_block_buffer.balanced       = 0.25
        KALSHI_TRADE_OVERLAY.entry_block_buffer.forward_primary = 0.25
        LIVE_OPPOSITE_REVERSAL.required_confirmations          = baseline

    CALM_TREND:
        LIVE_OPPOSITE_REVERSAL.required_confirmations          = 5
        KALSHI_TRADE_OVERLAY.entry_block_buffer.*              = baseline

    NEUTRAL:
        both restored to baselines captured at init time.

Thresholds are calibrated from Apr 2026 Wk1 (WHIPSAW: vol=6.52bp eff=0.039)
and Wk2 (CALM_TREND: vol=2.42bp eff=0.108).
"""
from __future__ import annotations

import logging
import math
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional, Tuple

from config import CONFIG


WINDOW_BARS = 120

# Simplified efficiency-primary thresholds (vol is noisy at daily granularity
# due to overnight gaps; eff cleanly separates regimes).
# Daily Oct 2025 / Apr 2026 analysis:
#   eff < 0.05  -> extreme chop (Oct 21 eff=0.009, Oct 06 eff=0.009, Apr 10 eff=0.003) -> Exp5
#   eff > 0.12  -> strong trend (Oct 10 eff=0.281, Oct 20 eff=0.149, Apr 13 eff=0.209) -> Exp4
EFF_LOW = 0.05
EFF_HIGH = 0.12

WHIPSAW_BUF_VALUE = 0.25
CALM_REV_CONFIRM = 5

TRANSITION_COOLDOWN_BARS = 30

# Per-regime risk caps. Tight on whipsaw (Oct-style chop), loose on calm-trend
# (Apr Wk2-style grind), medium on neutral. JULIE_REGIME_ADAPTIVE_CB=1 enables.
ADAPTIVE_CB_ENABLED = os.environ.get("JULIE_REGIME_ADAPTIVE_CB", "0").strip() == "1"
CB_CAP_WHIPSAW = float(os.environ.get("JULIE_REGIME_CB_WHIPSAW", "250"))
CB_CAP_NEUTRAL = float(os.environ.get("JULIE_REGIME_CB_NEUTRAL", "350"))
CB_CAP_CALM = float(os.environ.get("JULIE_REGIME_CB_CALM", "500"))
CB_CONSEC_WHIPSAW = int(os.environ.get("JULIE_REGIME_CONSEC_WHIPSAW", "4"))
CB_CONSEC_NEUTRAL = int(os.environ.get("JULIE_REGIME_CONSEC_NEUTRAL", "5"))
CB_CONSEC_CALM = int(os.environ.get("JULIE_REGIME_CONSEC_CALM", "7"))

# JULIE_REGIME_WHIPSAW_VETO=1 makes the classifier block entries while
# regime=whipsaw. Default 0 (no veto; only config mutations are applied).
WHIPSAW_VETO_ENABLED = os.environ.get("JULIE_REGIME_WHIPSAW_VETO", "0").strip() == "1"

# Filter D — regime-gated size cap. When regime is whipsaw or calm_trend
# (i.e. NOT neutral), cap incoming signal size to 1 contract. Validated on
# 27-day outrageous set: cuts DD>$350 violations 16→7 while preserving most
# of the +$817 PnL improvement from filter C.
REGIME_SIZE_CAP_ENABLED = os.environ.get("JULIE_REGIME_SIZE_CAP", "0").strip() == "1"
REGIME_SIZE_CAP_VALUE = int(os.environ.get("JULIE_REGIME_SIZE_CAP_VALUE", "1"))

# Filter E — green-day size unlock. On trend regimes where filter D would
# normally cap to 1, recover upside by raising the cap once daily PnL proves
# the direction. Validated: C + D + green_unlock@$200→3 gives +$3,319 PnL
# on 27-day outrageous set (vs +$2,503 shipped C+D) with 8 violations (vs 7).
# Apr 9 rally recovers from $1,039 to $2,347 while chop-day protection holds.
REGIME_GREEN_UNLOCK_THRESHOLD = float(os.environ.get("JULIE_REGIME_GREEN_UNLOCK_PNL", "200"))
REGIME_GREEN_UNLOCK_SIZE = int(os.environ.get("JULIE_REGIME_GREEN_UNLOCK_SIZE", "3"))


@dataclass
class RegimeState:
    regime: str = "warmup"
    vol_bp: float = 0.0
    eff: float = 0.0
    last_transition_bar: int = -10_000
    bar_count: int = 0


class RegimeClassifier:
    def __init__(self) -> None:
        self.enabled = True
        self._closes: Deque[float] = deque(maxlen=WINDOW_BARS)
        self._state = RegimeState()
        # Snapshot baselines once so NEUTRAL can restore them.
        overlay = CONFIG.get("KALSHI_TRADE_OVERLAY") or {}
        buf = overlay.get("entry_block_buffer") or {}
        self._baseline_buf_balanced = float(buf.get("balanced", 0.10))
        self._baseline_buf_fp = float(buf.get("forward_primary", 0.10))
        rev_cfg = CONFIG.get("LIVE_OPPOSITE_REVERSAL") or {}
        self._baseline_rev_confirm = int(rev_cfg.get("required_confirmations", 3))

    @property
    def regime(self) -> str:
        return self._state.regime

    @property
    def state(self) -> RegimeState:
        return self._state

    def update(self, ts, close) -> Optional[str]:
        try:
            c = float(close)
        except Exception:
            return self._state.regime
        if not math.isfinite(c) or c <= 0:
            return self._state.regime
        self._closes.append(c)
        self._state.bar_count += 1
        if len(self._closes) < WINDOW_BARS:
            return self._state.regime  # still warming up
        vol_bp, eff = self._compute_metrics()
        self._state.vol_bp = vol_bp
        self._state.eff = eff
        new_regime = self._classify(vol_bp, eff)
        if new_regime != self._state.regime:
            since = self._state.bar_count - self._state.last_transition_bar
            if self._state.regime != "warmup" and since < TRANSITION_COOLDOWN_BARS:
                return self._state.regime
            self._apply_regime(new_regime, ts, vol_bp, eff)
        return self._state.regime

    def _compute_metrics(self) -> Tuple[float, float]:
        closes = list(self._closes)
        rets = []
        for i in range(1, len(closes)):
            p0 = closes[i - 1]
            if p0 > 0:
                rets.append((closes[i] - p0) / p0)
        if not rets:
            return 0.0, 0.0
        mean = sum(rets) / len(rets)
        var = sum((r - mean) ** 2 for r in rets) / max(1, len(rets) - 1)
        vol_bp = math.sqrt(var) * 10_000.0
        abs_sum = sum(abs(r) for r in rets)
        eff = abs(sum(rets)) / abs_sum if abs_sum > 0 else 0.0
        return vol_bp, eff

    def _classify(self, vol_bp: float, eff: float) -> str:
        # Whipsaw = violent chop: BOTH high vol AND low eff (e.g. tariff-week
        # tape). Pure-eff low with low vol is just a flat tape (not a whipsaw,
        # stays neutral so the bot can still fish).
        # Calm-trend = directional with moderate-to-low vol.
        if vol_bp > 3.5 and eff < EFF_LOW:
            return "whipsaw"
        if eff > EFF_HIGH:
            return "calm_trend"
        return "neutral"

    def _apply_regime(self, new_regime: str, ts, vol_bp: float, eff: float) -> None:
        overlay = CONFIG.setdefault("KALSHI_TRADE_OVERLAY", {})
        buf = overlay.setdefault("entry_block_buffer", {})
        rev_cfg = CONFIG.setdefault("LIVE_OPPOSITE_REVERSAL", {})
        if new_regime == "whipsaw":
            buf["balanced"] = WHIPSAW_BUF_VALUE
            buf["forward_primary"] = WHIPSAW_BUF_VALUE
            rev_cfg["required_confirmations"] = self._baseline_rev_confirm
            cb_cap, cb_consec = CB_CAP_WHIPSAW, CB_CONSEC_WHIPSAW
        elif new_regime == "calm_trend":
            buf["balanced"] = self._baseline_buf_balanced
            buf["forward_primary"] = self._baseline_buf_fp
            rev_cfg["required_confirmations"] = CALM_REV_CONFIRM
            cb_cap, cb_consec = CB_CAP_CALM, CB_CONSEC_CALM
        else:
            buf["balanced"] = self._baseline_buf_balanced
            buf["forward_primary"] = self._baseline_buf_fp
            rev_cfg["required_confirmations"] = self._baseline_rev_confirm
            cb_cap, cb_consec = CB_CAP_NEUTRAL, CB_CONSEC_NEUTRAL

        # Adaptive CB: retune the circuit breaker risk cap to match the current
        # regime. Tight caps on whipsaw (kill chop days fast); loose on calm-
        # trend (let normal drawdowns recover into a winning day).
        if ADAPTIVE_CB_ENABLED:
            try:
                import circuit_breaker as _cb_mod
                cb = getattr(_cb_mod, "_GLOBAL_CB", None)
                if cb is not None:
                    cb.max_daily_loss = float(cb_cap)
                    cb.max_consecutive_losses = int(cb_consec)
            except Exception:
                logging.debug("adaptive CB update failed", exc_info=True)
        prev = self._state.regime
        self._state.regime = new_regime
        self._state.last_transition_bar = self._state.bar_count
        logging.info(
            "Regime transition: %s -> %s | vol=%.2fbp eff=%.3f | buf_bal=%.2f buf_fp=%.2f rev=%d | ts=%s",
            prev, new_regime, vol_bp, eff,
            buf.get("balanced", 0), buf.get("forward_primary", 0),
            rev_cfg.get("required_confirmations", 0), ts,
        )


_CLASSIFIER: Optional[RegimeClassifier] = None


def init_regime_classifier() -> Optional[RegimeClassifier]:
    global _CLASSIFIER
    if os.environ.get("JULIE_REGIME_CLASSIFIER", "0").strip() != "1":
        _CLASSIFIER = None
        return None
    _CLASSIFIER = RegimeClassifier()
    logging.info("Regime classifier enabled (JULIE_REGIME_CLASSIFIER=1).")
    return _CLASSIFIER


def get_regime_classifier() -> Optional[RegimeClassifier]:
    return _CLASSIFIER


def update_regime_classifier(ts, close) -> Optional[str]:
    if _CLASSIFIER is None:
        return None
    try:
        return _CLASSIFIER.update(ts, close)
    except Exception:
        logging.debug("regime classifier update failed", exc_info=True)
        return None


def current_regime() -> str:
    return _CLASSIFIER.regime if _CLASSIFIER is not None else "disabled"


def should_veto_entry() -> tuple[bool, str]:
    """Return (veto, reason). Caller should block the entry if veto is True.

    Active only when JULIE_REGIME_WHIPSAW_VETO=1 and current regime is whipsaw.
    """
    if _CLASSIFIER is None or not WHIPSAW_VETO_ENABLED:
        return False, ""
    if _CLASSIFIER.regime == "whipsaw":
        return True, f"regime=whipsaw vol={_CLASSIFIER.state.vol_bp:.2f}bp eff={_CLASSIFIER.state.eff:.3f}"
    return False, ""


def apply_regime_size_cap(signal: dict) -> bool:
    """Filter D — when regime is whipsaw or calm_trend (non-neutral), cap the
    signal's size field to REGIME_SIZE_CAP_VALUE. Returns True if size was
    modified, False otherwise. Idempotent and mutation is visible to the
    pct_overlay snapshot + downstream consumers.

    Gated by JULIE_REGIME_SIZE_CAP=1. Validated on 27-day outrageous 2025 set:
    chops unaffected, breakouts capped (Apr 9 reduced but net still positive).
    """
    if _CLASSIFIER is None or not REGIME_SIZE_CAP_ENABLED:
        return False
    if not isinstance(signal, dict):
        return False
    regime = _CLASSIFIER.regime
    if regime not in ("whipsaw", "calm_trend"):
        return False
    try:
        base = int(signal.get("size", 1) or 1)
    except Exception:
        base = 1
    cap = max(1, REGIME_SIZE_CAP_VALUE)
    # Filter E: green-day unlock. If today's cum PnL has crossed the threshold,
    # raise the cap so we can press a winning day. Read cum PnL from the
    # circuit_breaker's daily_pnl tracker (shared state, already updated per
    # trade close). Falls through to D's tight cap if CB isn't available or
    # daily PnL hasn't reached the unlock threshold.
    try:
        import circuit_breaker as _cb_mod
        cb = getattr(_cb_mod, "_GLOBAL_CB", None)
        if cb is not None and float(cb.daily_pnl) >= REGIME_GREEN_UNLOCK_THRESHOLD:
            unlocked_cap = max(cap, REGIME_GREEN_UNLOCK_SIZE)
            if unlocked_cap > cap:
                cap = unlocked_cap
                signal["regime_size_cap_unlocked"] = True
    except Exception:
        pass
    if base <= cap:
        return False
    signal["size"] = cap
    signal["regime_size_cap_before"] = base
    signal["regime_size_cap_regime"] = regime
    logging.info(
        "Regime size cap (%s%s): %s %s | size %d -> %d",
        regime, " unlocked" if signal.get("regime_size_cap_unlocked") else "",
        signal.get("strategy", "?"), signal.get("side", "?"), base, cap,
    )
    return True
