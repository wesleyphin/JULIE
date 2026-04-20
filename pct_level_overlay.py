"""Percentage-level overlay.

Tracks price relative to the trading-day session open (18:00 ET -> 17:59 ET
next calendar day) and emits a continuation/pivot bias score whenever the
price approaches one of the ladder levels (+/- 0.25, 0.50, 0.75, 1.00,
1.25, 1.50, 1.75, 2.00, 2.50, 3.00 %).

Design notes (see 15yr ES study results):
  - Continuation bias is structural at every rung (+16..+26pt edge of
    BK% over PV% across 3,876 days).
  - Biggest lift: ATR Q3-Q4 + wide intraday range + NY morning hours.
  - Scout tier (0.25, 0.50) -> sizing nudge only.
  - Primary tier (0.75, 1.00) -> sizing + trail/TP nudge.
  - Exhaustion tier (1.50, 1.75, 2.00+) -> trail/TP nudge only.

The overlay is hot-path friendly: Tier 1 (per-bar) is a single subtract
and abs-min check. Tier 2 is only evaluated when the bar is within the
proximity band of a level.
"""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple
from zoneinfo import ZoneInfo


NY_TZ = ZoneInfo("America/New_York")


DEFAULT_OVERLAY_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "feature_flag": "ENABLE_PCT_LEVEL_OVERLAY",
    "levels_pct": [0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.50, 3.00],
    "proximity_pct": 0.05,
    "breakout_extension_pct": 0.10,
    "pivot_retrace_pct": 0.15,
    "horizon_minutes": 30,
    "atr_window_bars": 30,
    # Q1/Q3 cutoffs for ATR percent per bar, pre-computed from 15yr ES study.
    "atr_q1_pct": 0.020,
    "atr_q3_pct": 0.070,
    # Q1/Q3 cutoffs for intraday range-so-far percent of session open.
    "range_q1_pct": 0.30,
    "range_q3_pct": 1.00,
    # Per-hour continuation edge (BK% - PV%) from the 15yr study, averaged
    # across 0.50 and 0.75 tiers. Positive => continuation favored.
    "hour_edge_table": {
        0: 0.10, 1: -0.05, 2: 0.20, 3: 0.10, 4: 0.05, 5: 0.00,
        6: 0.02, 7: -0.05, 8: 0.28, 9: 0.25, 10: 0.33, 11: 0.30,
        12: 0.10, 13: 0.05, 14: 0.25, 15: 0.30, 16: 0.05, 17: 0.05,
        18: 0.30, 19: 0.15, 20: 0.05, 21: 0.20, 22: 0.10, 23: -0.05,
    },
    # Base edge (BK% - PV%) per absolute level, from 15yr ES study.
    "level_base_edge": {
        0.25: 0.21, 0.50: 0.23, 0.75: 0.24, 1.00: 0.26,
        1.25: 0.23, 1.50: 0.17, 1.75: 0.17, 2.00: 0.16,
        2.50: 0.23, 3.00: 0.11,
    },
    # Scout (sizing-only), primary (sizing + trail/TP), exhaustion (trail/TP).
    "scout_levels": [0.25, 0.50],
    "primary_levels": [0.75, 1.00, 1.25],
    "exhaustion_levels": [1.50, 1.75, 2.00, 2.50, 3.00],
    # Tier 3 action tunables.
    "size_tilt_conf_threshold": 0.30,
    "max_size_tilt_pct": 0.20,
    "min_size_tilt_pct": -0.30,
    "trail_tp_extend_pct": float(os.environ.get("JULIE_PCT_TRAIL_EXT", "0.20")),
    "trail_tp_tighten_pct": float(os.environ.get("JULIE_PCT_TRAIL_TIGHT", "0.30")),
    # Hours where base edge is near zero or negative -> skip Tier 2.
    "dead_hours_et": [1, 7, 23],
}


@dataclass
class PctLevelState:
    """Per-bar state emitted by the overlay."""
    ts: Optional[datetime] = None
    session_open: Optional[float] = None
    price: Optional[float] = None
    pct_from_open: float = 0.0
    nearest_level: Optional[float] = None
    level_distance_pct: float = 0.0
    at_level: bool = False
    tier: str = "neutral"  # scout | primary | exhaustion | neutral
    bias: str = "neutral"  # breakout_lean | pivot_lean | neutral | chop
    confidence: float = 0.0
    atr_pct_30bar: float = 0.0
    range_pct_at_touch: float = 0.0
    atr_bucket: str = "Q2"
    range_bucket: str = "Q2"
    hour_edge: float = 0.0
    last_event: str = ""


class PctLevelOverlay:
    """Hot-path friendly pct-level tracker.

    Call order per bar (live or replay):
        overlay.update(ts, open_, high, low, close)
    Then read overlay.state for sizing/trail/TP decisions.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = dict(DEFAULT_OVERLAY_CONFIG)
        if config:
            cfg.update({k: v for k, v in config.items() if v is not None})
        self.cfg = cfg
        self.enabled = bool(cfg.get("enabled", True))
        self.levels: List[float] = sorted(float(x) for x in cfg["levels_pct"])
        self.prox = float(cfg["proximity_pct"])
        self.bk_ext = float(cfg["breakout_extension_pct"])
        self.pv_ret = float(cfg["pivot_retrace_pct"])
        self.horizon = int(cfg["horizon_minutes"])
        self.atr_window = int(cfg["atr_window_bars"])
        self.atr_q1 = float(cfg["atr_q1_pct"])
        self.atr_q3 = float(cfg["atr_q3_pct"])
        self.range_q1 = float(cfg["range_q1_pct"])
        self.range_q3 = float(cfg["range_q3_pct"])
        self.hour_edge_table = {int(k): float(v) for k, v in cfg["hour_edge_table"].items()}
        self.level_base_edge = {float(k): float(v) for k, v in cfg["level_base_edge"].items()}
        self.scout = set(float(x) for x in cfg["scout_levels"])
        self.primary = set(float(x) for x in cfg["primary_levels"])
        self.exhaustion = set(float(x) for x in cfg["exhaustion_levels"])
        self.size_tilt_conf = float(cfg["size_tilt_conf_threshold"])
        self.max_tilt = float(cfg["max_size_tilt_pct"])
        self.min_tilt = float(cfg["min_size_tilt_pct"])
        self.dead_hours = set(int(x) for x in cfg.get("dead_hours_et", []))

        # Tier 0 state (cold).
        self._current_tday: Optional[datetime] = None
        self._session_open: Optional[float] = None
        self._running_hi: float = float("-inf")
        self._running_lo: float = float("inf")
        self._atr_ring: deque = deque(maxlen=self.atr_window)

        # Tier 3 cooldown: don't re-fire an event at the same level+direction
        # within this window. Debounces back-to-back Tier 2 evaluations at
        # the same pct-level.
        self._last_fire: Dict[Tuple[float, str], datetime] = {}
        self._fire_cooldown = timedelta(minutes=10)

        self.state = PctLevelState()
        self.event_log: List[Dict[str, Any]] = []

    @staticmethod
    def _trade_day_key(ts: datetime) -> datetime.date:
        """18:00 ET -> next calendar date."""
        if ts.hour >= 18:
            return (ts + timedelta(days=1)).date()
        return ts.date()

    def _to_et(self, ts: datetime) -> datetime:
        if ts.tzinfo is None:
            return ts.replace(tzinfo=timezone.utc).astimezone(NY_TZ)
        return ts.astimezone(NY_TZ)

    def update(self, ts: datetime, open_: float, high: float, low: float,
               close: float) -> PctLevelState:
        """Feed one completed bar and return latest state."""
        if not self.enabled:
            return self.state

        ts_et = self._to_et(ts)
        tday = self._trade_day_key(ts_et)

        # Tier 0: new trading day -> reset session anchor.
        if self._current_tday != tday:
            self._current_tday = tday
            self._session_open = open_
            self._running_hi = high
            self._running_lo = low
            self._atr_ring.clear()
            self._atr_ring.append(high - low)
        else:
            self._running_hi = max(self._running_hi, high)
            self._running_lo = min(self._running_lo, low)
            self._atr_ring.append(high - low)

        op = self._session_open
        if op is None or op <= 0:
            return self.state

        pct = (close - op) / op * 100.0

        # Tier 1 hot path: check nearest level.
        nearest = min(self.levels, key=lambda L: min(abs(pct - L), abs(pct + L)))
        # signed level the price is closest to
        dist_plus = abs(pct - nearest)
        dist_minus = abs(pct + nearest)
        signed_lvl = nearest if dist_plus <= dist_minus else -nearest
        level_distance = abs(pct - signed_lvl)

        state = PctLevelState(
            ts=ts_et,
            session_open=op,
            price=close,
            pct_from_open=pct,
            nearest_level=signed_lvl,
            level_distance_pct=level_distance,
            at_level=False,
            tier="neutral",
            bias="neutral",
            confidence=0.0,
        )

        if level_distance > self.prox:
            self.state = state
            return state

        # Tier 2 warm path: compute bias.
        hour_et = ts_et.hour
        if hour_et in self.dead_hours:
            state.at_level = True
            state.tier = self._tier_for(nearest)
            state.bias = "neutral"
            state.last_event = "dead_hour_skip"
            self.state = state
            return state

        # ATR bucket
        atr_pct = 0.0
        if self._atr_ring:
            atr_bars = list(self._atr_ring)[-self.atr_window:]
            if atr_bars:
                atr_pct = (sum(atr_bars) / len(atr_bars)) / op * 100.0
        if atr_pct >= self.atr_q3:
            atr_bucket = "Q4"
        elif atr_pct <= self.atr_q1:
            atr_bucket = "Q1"
        else:
            atr_bucket = "Q2"

        # Range bucket
        range_pct = (self._running_hi - self._running_lo) / op * 100.0
        if range_pct >= self.range_q3:
            range_bucket = "Q4"
        elif range_pct <= self.range_q1:
            range_bucket = "Q1"
        else:
            range_bucket = "Q2"

        # Hour edge
        hour_edge = self.hour_edge_table.get(hour_et, 0.0)
        # Base edge for this level magnitude.
        base_edge = self.level_base_edge.get(nearest, 0.15)

        # Confluence score in roughly [-1, +1].  Positive -> breakout lean.
        atr_w = {"Q1": -0.3, "Q2": 0.0, "Q4": 0.6}.get(atr_bucket, 0.0)
        range_w = {"Q1": -0.3, "Q2": 0.0, "Q4": 0.5}.get(range_bucket, 0.0)
        conf = base_edge + 0.4 * hour_edge + 0.5 * atr_w + 0.4 * range_w
        conf = max(-1.0, min(1.0, conf))

        # Bias label (sign-aware, exhaustion fades on calm+low-range if
        # at big level -> the rare negative-edge zone).
        bias = "neutral"
        if atr_bucket == "Q1" and range_bucket == "Q1" and nearest >= 1.5:
            bias = "pivot_lean"
            conf = -abs(conf)
        elif conf >= self.size_tilt_conf:
            bias = "breakout_lean"
        elif conf <= -self.size_tilt_conf:
            bias = "pivot_lean"
        elif atr_bucket == "Q1" and range_bucket == "Q1":
            bias = "chop"

        state.at_level = True
        state.tier = self._tier_for(nearest)
        state.bias = bias
        state.confidence = conf
        state.atr_pct_30bar = atr_pct
        state.range_pct_at_touch = range_pct
        state.atr_bucket = atr_bucket
        state.range_bucket = range_bucket
        state.hour_edge = hour_edge

        # Log first-touch events for this level+direction (debounced).
        key = (signed_lvl, bias)
        last = self._last_fire.get(key)
        if last is None or (ts_et - last) > self._fire_cooldown:
            self._last_fire[key] = ts_et
            state.last_event = f"fresh_touch_{'pos' if signed_lvl>0 else 'neg'}"
            self.event_log.append({
                "ts": ts_et.isoformat(),
                "signed_lvl": signed_lvl,
                "pct_from_open": pct,
                "bias": bias,
                "confidence": conf,
                "atr_bucket": atr_bucket,
                "range_bucket": range_bucket,
                "hour_et": hour_et,
                "tier": state.tier,
            })
        else:
            state.last_event = "debounced"

        self.state = state
        return state

    def _tier_for(self, level_abs: float) -> str:
        if level_abs in self.scout:
            return "scout"
        if level_abs in self.primary:
            return "primary"
        if level_abs in self.exhaustion:
            return "exhaustion"
        return "neutral"

    # ------------------------------------------------------------------
    # Tier 3 public helpers.  Safe defaults (no tilt) when overlay
    # has no opinion.

    def size_multiplier(self, trade_side: str) -> float:
        """Return a size multiplier (e.g. 1.15 = up 15%, 0.80 = cut 20%)."""
        if not self.enabled or not self.state.at_level:
            return 1.0
        bias = self.state.bias
        tier = self.state.tier
        conf = self.state.confidence
        if bias == "chop":
            return 0.85
        if tier == "exhaustion":
            # Exhaustion tier only modifies trail/TP, not sizing.
            return 1.0
        if bias == "breakout_lean":
            # Agrees with continuation direction of the level.
            lvl = self.state.nearest_level or 0.0
            side_agrees = (lvl > 0 and trade_side.lower() == "long") or (
                lvl < 0 and trade_side.lower() == "short"
            )
            tilt = self.max_tilt * min(1.0, max(0.0, abs(conf)))
            if side_agrees:
                return 1.0 + tilt
            return max(0.7, 1.0 + self.min_tilt * abs(conf))
        if bias == "pivot_lean":
            lvl = self.state.nearest_level or 0.0
            side_counter = (lvl > 0 and trade_side.lower() == "short") or (
                lvl < 0 and trade_side.lower() == "long"
            )
            if side_counter:
                # Counter-trend aligned with pivot expectation.
                return 1.0 + (self.max_tilt * 0.5) * abs(conf)
            return max(0.7, 1.0 + self.min_tilt * abs(conf))
        return 1.0

    def tp_trail_modifier(self, trade_side: str) -> Dict[str, float]:
        """Return dict with 'tp_extend_pct', 'trail_tighten_pct'.

        Multiplier on top of the strategy's base TP/trail decision:
            tp_extend_pct > 0 => allow bigger TP
            trail_tighten_pct > 0 => pull trail closer
        """
        if not self.enabled or not self.state.at_level:
            return {"tp_extend_pct": 0.0, "trail_tighten_pct": 0.0}
        lvl = self.state.nearest_level or 0.0
        tier = self.state.tier
        bias = self.state.bias
        ext = float(self.cfg.get("trail_tp_extend_pct", 0.20))
        tight = float(self.cfg.get("trail_tp_tighten_pct", 0.30))
        side_agrees = (lvl > 0 and trade_side.lower() == "long") or (
            lvl < 0 and trade_side.lower() == "short"
        )
        if bias == "breakout_lean" and side_agrees and tier in ("primary", "exhaustion"):
            # Let the winner run.
            return {"tp_extend_pct": ext, "trail_tighten_pct": 0.0}
        if bias == "pivot_lean" and side_agrees:
            # Take profit sooner, tighten trail -- the level may reject.
            return {"tp_extend_pct": 0.0, "trail_tighten_pct": tight}
        if tier == "exhaustion" and side_agrees:
            # At >=1.50%, 88% of touches are decisive.  Tighten trail to
            # lock gains in case of the ~35-42% pivot outcome.
            return {"tp_extend_pct": 0.0, "trail_tighten_pct": tight * 0.5}
        return {"tp_extend_pct": 0.0, "trail_tighten_pct": 0.0}

    def should_veto(self, trade_side: str) -> Tuple[bool, str]:
        """Hard veto only in the rare reliable-pivot zone."""
        if not self.enabled or not self.state.at_level:
            return False, ""
        lvl = self.state.nearest_level or 0.0
        tier = self.state.tier
        bias = self.state.bias
        side_agrees = (lvl > 0 and trade_side.lower() == "long") or (
            lvl < 0 and trade_side.lower() == "short"
        )
        if tier == "exhaustion" and bias == "pivot_lean" and side_agrees:
            return True, "pct_level_exhaustion_pivot"
        return False, ""

    def snapshot(self) -> Dict[str, Any]:
        s = self.state
        return {
            "ts": s.ts.isoformat() if s.ts else None,
            "session_open": s.session_open,
            "price": s.price,
            "pct_from_open": round(s.pct_from_open, 4),
            "nearest_level": s.nearest_level,
            "level_distance_pct": round(s.level_distance_pct, 4),
            "at_level": s.at_level,
            "tier": s.tier,
            "bias": s.bias,
            "confidence": round(s.confidence, 3),
            "atr_pct_30bar": round(s.atr_pct_30bar, 4),
            "range_pct_at_touch": round(s.range_pct_at_touch, 4),
            "atr_bucket": s.atr_bucket,
            "range_bucket": s.range_bucket,
            "hour_edge": s.hour_edge,
            "last_event": s.last_event,
        }
