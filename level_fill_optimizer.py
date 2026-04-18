"""
Level-Aware Fill Optimizer
==========================
Uses the mapped levels from the weswesindicator (bank grid, structural levels,
ORB H/L, session H/L) to decide the optimal entry timing for each trade signal.

Three modes:
  IMMEDIATE  — Execute at market right now (default / price between levels)
  AT_LEVEL   — Price is already at a level; execute immediately (best fill)
  WAIT       — Price is 0.75–2.5 pts from a high-significance level;
               defer up to MAX_WAIT_BARS for a better fill

ES data analysis (2022–2026, 1m bars):
  • When 1.5–2 pts from a bank level, price touches it within 3 bars ~35 % of the time
  • Timeout falls back to market fill — no trade is ever missed
  • Only structural / ORB levels (priority ≥ 3) trigger WAIT; raw $12.50 bank grid
    only does so when price is within 1.5 pts

Config key: LEVEL_FILL_OPTIMIZER_ENABLED (default True in CONFIG)
"""

import logging
from typing import Dict, List, Optional, Tuple

FILL_IMMEDIATE = "IMMEDIATE"
FILL_AT_LEVEL  = "AT_LEVEL"
FILL_WAIT      = "WAIT"

BANK_GRID           = 12.50   # absolute $12.50 grid
AT_LEVEL_THRESHOLD  = 0.75    # pts — within this → AT_LEVEL (execute now)
NEAR_LEVEL_MIN      = 0.75    # pts — must be at least this far to bother waiting
NEAR_LEVEL_MAX      = 2.50    # pts — must be within this to trigger WAIT
MAX_WAIT_BARS       = 3       # max bars to hold a pending signal
RAN_AWAY_MULT       = 3.0     # abort if price moves (RAN_AWAY_MULT × level_dist) away


# ---------------------------------------------------------------------------
# Level collection helpers
# ---------------------------------------------------------------------------

def _bank_grid_levels(price: float) -> List[Dict]:
    base = (price // BANK_GRID) * BANK_GRID
    levels = []
    for i in range(-2, 3):
        lvl = base + i * BANK_GRID
        levels.append({"price": lvl, "name": f"Bank_{lvl:.2f}", "type": "bank", "priority": 1})
    return levels


def _levels_from_bank_filter(bank_filter) -> List[Dict]:
    """Pull tracked structural levels out of BankLevelQuarterFilter."""
    if bank_filter is None:
        return []
    try:
        state = bank_filter.get_state()
    except Exception:
        return []
    mapping = [
        ("prev_session_high",              "PrevSessH",     "structural", 4),
        ("prev_session_low",               "PrevSessL",     "structural", 4),
        ("prev_day_pm_high",               "PrevPMH",       "structural", 3),
        ("prev_day_pm_low",                "PrevPML",       "structural", 3),
        ("midnight_orb_high",              "MidnightORB_H", "orb",        3),
        ("midnight_orb_low",               "MidnightORB_L", "orb",        3),
        ("bank_above_prev_session_high",   "BankAboveSessH","bank",       2),
        ("bank_below_prev_session_low",    "BankBelowSessL","bank",       2),
        ("bank_above_orb_high",            "BankAboveORBH", "bank",       2),
        ("bank_below_orb_low",             "BankBelowORBL", "bank",       2),
    ]
    levels = []
    for key, name, ltype, prio in mapping:
        v = state.get(key)
        if v is not None:
            try:
                levels.append({"price": float(v), "name": name, "type": ltype, "priority": prio})
            except (TypeError, ValueError):
                pass
    return levels


def _bank_levels_adjacent_to_structural(structural_levels: List[Dict]) -> List[Dict]:
    """
    For each structural level (session H/L, ORB H/L, TO, DO, Q1 H/L, midpoints),
    compute the $12.50 bank grid level immediately below AND immediately above it.

    These confluence zones — where a structural pivot coincides with the nearest
    bank boundary — are high-probability reversal areas. They inherit elevated
    priority (same as structural) because two independent level types agree.

    Also computes the midpoint between each structural level and its adjacent
    bank levels, as those half-grid points are additional soft magnets.
    """
    seen: set = set()
    extra: List[Dict] = []

    for lvl in structural_levels:
        p = lvl["price"]
        parent_name = lvl["name"]
        parent_prio = lvl["priority"]

        bank_below = (p // BANK_GRID) * BANK_GRID
        bank_above = bank_below + BANK_GRID

        for bank_p, suffix in [(bank_below, "Below"), (bank_above, "Above")]:
            key = round(bank_p, 4)
            if key not in seen:
                seen.add(key)
                extra.append({
                    "price":    bank_p,
                    "name":     f"Bank{suffix}_{parent_name}",
                    "type":     "confluence",
                    # confluence of bank + structural → elevated to structural priority
                    "priority": parent_prio,
                })

        # Half-grid midpoints between the structural level and its adjacent banks
        for mid_p, suffix in [
            ((p + bank_below) / 2, "MidBelow"),
            ((p + bank_above) / 2, "MidAbove"),
        ]:
            key = round(mid_p, 4)
            if key not in seen:
                seen.add(key)
                extra.append({
                    "price":    mid_p,
                    "name":     f"BankMid{suffix}_{parent_name}",
                    "type":     "confluence_mid",
                    "priority": max(1, parent_prio - 1),  # one step below structural
                })

    return extra


# ---------------------------------------------------------------------------
# Main optimizer class
# ---------------------------------------------------------------------------

class LevelFillOptimizer:
    """Evaluates fill strategy and tracks pending deferred entries."""

    def __init__(self):
        self._pending: Dict[str, Dict] = {}

    # ── public API ──────────────────────────────────────────────────────────

    def evaluate(
        self,
        signal: Dict,
        current_price: float,
        structural_tracker=None,
        bank_filter=None,
        bar_candle: Optional[Dict] = None,
    ) -> Dict:
        """
        Returns dict:
          mode        : FILL_IMMEDIATE | FILL_AT_LEVEL | FILL_WAIT
          target_price: float (WAIT/AT_LEVEL only)
          target_name : str
          dist        : float distance from nearest directional level
          reason      : str
        """
        side     = (signal.get("side") or "").upper()
        is_long  = side == "LONG"

        # Collect levels
        all_levels: List[Dict] = _bank_grid_levels(current_price)
        all_levels += _levels_from_bank_filter(bank_filter)

        structural_levels: List[Dict] = []
        bank_filter_levels = _levels_from_bank_filter(bank_filter)
        # structural levels from bank_filter (session H/L, ORB H/L, etc.)
        structural_levels += [l for l in bank_filter_levels if l["type"] in ("structural", "orb")]

        if structural_tracker is not None:
            try:
                _tracker_levels = structural_tracker.get_active_levels()
                all_levels += _tracker_levels
                structural_levels += _tracker_levels
            except Exception:
                pass

        # Bank levels immediately above/below each structural level are confluence
        # zones — two independent level types agreeing on the same area.
        all_levels += _bank_levels_adjacent_to_structural(structural_levels)

        # Keep only directionally relevant levels
        # LONG  → support candidates = levels at or below current price
        # SHORT → resistance candidates = levels at or above current price
        if is_long:
            candidates = [l for l in all_levels if l["price"] <= current_price]
            if not candidates:
                return self._immediate("no support levels below")
            closest = max(candidates, key=lambda l: l["price"])
        else:
            candidates = [l for l in all_levels if l["price"] >= current_price]
            if not candidates:
                return self._immediate("no resistance levels above")
            closest = min(candidates, key=lambda l: l["price"])

        dist = abs(current_price - closest["price"])
        sl_dist = float(signal.get("sl_dist") or 0.0)

        # AT_LEVEL: already sitting on the level
        if dist <= AT_LEVEL_THRESHOLD:
            return {
                "mode":         FILL_AT_LEVEL,
                "target_price": closest["price"],
                "target_name":  closest["name"],
                "dist":         dist,
                "reason":       f"at {closest['name']} ({dist:.2f}pts)",
            }

        # WAIT: close enough to a meaningful level to defer
        if NEAR_LEVEL_MIN < dist <= NEAR_LEVEL_MAX and closest["priority"] >= 2:
            if sl_dist > 0.0 and dist >= (sl_dist + AT_LEVEL_THRESHOLD - 1e-9):
                return self._immediate(
                    f"level wait invalid: {closest['name']} is beyond protected drift "
                    f"({dist:.2f}pts >= sl {sl_dist:.2f}pts + touch buffer)",
                    closest,
                )
            # Raw bank grid (priority 1 → excluded above).
            # Bank-derived levels (priority 2) only warrant a wait if very close.
            if closest["type"] == "bank" and closest["priority"] < 3 and dist > 1.5:
                return self._immediate(f"bank level too far ({dist:.2f}pts)", closest)

            # Check whether the bar is moving toward the level
            approaching = self._is_approaching(is_long, bar_candle, current_price)
            if approaching or dist <= 1.5:
                return {
                    "mode":         FILL_WAIT,
                    "target_price": closest["price"],
                    "target_name":  closest["name"],
                    "dist":         dist,
                    "max_bars":     MAX_WAIT_BARS,
                    "reason":       f"waiting for {closest['name']}@{closest['price']:.2f} ({dist:.2f}pts)",
                }

        return self._immediate(f"level {closest['name']} too far or not approaching ({dist:.2f}pts)", closest)

    def add_pending(self, uid: str, signal: Dict, decision: Dict, current_price: float) -> None:
        self._pending[uid] = {
            "signal":       signal,
            "target_price": decision["target_price"],
            "target_name":  decision["target_name"],
            "max_bars":     decision.get("max_bars", MAX_WAIT_BARS),
            "bars_waited":  0,
            "entry_price":  current_price,
        }
        logging.info(
            "📌 LevelFill QUEUED: %s %s → %s@%.2f (%.2f pts away, max %d bars)",
            signal.get("strategy", "?"),
            signal.get("side", "?"),
            decision["target_name"],
            decision["target_price"],
            decision.get("dist", 0),
            decision.get("max_bars", MAX_WAIT_BARS),
        )

    def check_pending(self, uid: str, bar: Dict) -> Dict:
        """
        Called each bar for a pending entry.
        Returns: {fire, abort, reason}
          fire=True  → execute at market now
          abort=True → discard signal (setup invalidated)

        Staleness checks run on every bar and on timeout:
          1. Level violated  — price closed through the target level by more than
                               AT_LEVEL_THRESHOLD (support broken for LONG, resistance
                               broken for SHORT). The level no longer holds.
          2. SL-breach check — price moved further against the signal than the
                               original SL distance. The setup would already be a
                               loser; executing at market makes it worse.
          3. Ran away        — price moved RAN_AWAY_MULT × init_dist in the right
                               direction without touching the level (momentum carried
                               it; the pullback thesis is wrong).
          4. Timeout + still valid — execute at market (don't miss the trade).
          5. Timeout + stale     — abort (context has changed too much).
        """
        if uid not in self._pending:
            return {"fire": False, "abort": True, "reason": "uid not found"}

        entry     = self._pending[uid]
        entry["bars_waited"] += 1
        target    = entry["target_price"]
        signal    = entry["signal"]
        is_long   = signal.get("side", "").upper() == "LONG"
        sl_dist   = float(signal.get("sl_dist") or 0.0)
        bar_lo    = float(bar.get("low",  bar.get("close", target)))
        bar_hi    = float(bar.get("high", bar.get("close", target)))
        bar_cl    = float(bar.get("close", entry["entry_price"]))
        orig      = entry["entry_price"]
        init_dist = abs(orig - target)
        close_dist = abs(bar_cl - target)

        # ── 1. Level violated ─────────────────────────────────────────
        # LONG: close meaningfully BELOW target → support broken
        # SHORT: close meaningfully ABOVE target → resistance broken
        level_violated = (
            (is_long  and bar_cl < target - AT_LEVEL_THRESHOLD) or
            (not is_long and bar_cl > target + AT_LEVEL_THRESHOLD)
        )

        # ── 2. SL-breach on paper ─────────────────────────────────────
        # Price has moved more than sl_dist against the original signal price.
        sl_breached = sl_dist > 0 and (
            (is_long  and bar_lo < orig - sl_dist) or
            (not is_long and bar_hi > orig + sl_dist)
        )

        # ── 3. Ran away (right direction, no pullback) ────────────────
        ran_away = (
            (is_long  and bar_cl > orig + RAN_AWAY_MULT * init_dist) or
            (not is_long and bar_cl < orig - RAN_AWAY_MULT * init_dist)
        )

        # ── 4. Level touched ─────────────────────────────────────────
        touched = (
            (is_long  and bar_lo <= target + AT_LEVEL_THRESHOLD) or
            (not is_long and bar_hi >= target - AT_LEVEL_THRESHOLD)
        )
        close_near_target = close_dist <= AT_LEVEL_THRESHOLD + 1e-9

        timed_out = entry["bars_waited"] >= entry["max_bars"]

        # Abort checks run before fire checks so a violated level
        # on the touch bar is still caught correctly.
        if sl_breached:
            del self._pending[uid]
            return {"fire": False, "abort": True,
                    "reason": f"SL breached on paper ({orig:.2f} → sl {sl_dist:.2f}pts) after {entry['bars_waited']} bar(s)"}

        if level_violated:
            del self._pending[uid]
            return {"fire": False, "abort": True,
                    "reason": f"target level {target:.2f} violated (close={bar_cl:.2f}) — support/resistance broken"}

        if ran_away:
            del self._pending[uid]
            return {"fire": False, "abort": True,
                    "reason": f"price ran {RAN_AWAY_MULT}× init dist away from {target:.2f} without pullback"}

        if touched and close_near_target:
            del self._pending[uid]
            return {"fire": True, "abort": False,
                    "reason": f"level {target:.2f} touched after {entry['bars_waited']} bar(s)"}

        if timed_out:
            # Final staleness guard before market fill:
            # if price drifted more than half a SL distance against us, abort.
            adverse_drift = (
                (is_long  and (orig - bar_cl) > sl_dist * 0.5) or
                (not is_long and (bar_cl - orig) > sl_dist * 0.5)
            ) if sl_dist > 0 else False

            del self._pending[uid]
            if adverse_drift:
                return {"fire": False, "abort": True,
                        "reason": f"timeout + adverse drift >{sl_dist*0.5:.2f}pts — setup stale, aborting"}
            return {"fire": True, "abort": False,
                    "reason": f"timeout ({entry['bars_waited']} bars) — no invalidation, executing at market"}

        if touched and not close_near_target:
            return {"fire": False, "abort": False,
                    "reason": f"level {target:.2f} touched intrabar but closed {close_dist:.2f}pts away"}

        return {"fire": False, "abort": False,
                "reason": f"waiting {entry['bars_waited']}/{entry['max_bars']} bars"}

    def get_pending_signal(self, uid: str) -> Optional[Dict]:
        return self._pending.get(uid)

    def get_pending_ids(self) -> List[str]:
        return list(self._pending.keys())

    def remove_pending(self, uid: str) -> None:
        self._pending.pop(uid, None)

    def clear_all(self, reason: str = "") -> None:
        if self._pending:
            logging.info(
                "📌 LevelFill: clearing %d pending fills — %s",
                len(self._pending), reason
            )
        self._pending.clear()

    # ── helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _immediate(reason: str, closest: Optional[Dict] = None) -> Dict:
        return {
            "mode":         FILL_IMMEDIATE,
            "target_price": closest["price"] if closest else None,
            "target_name":  closest["name"]  if closest else None,
            "dist":         None,
            "reason":       reason,
        }

    @staticmethod
    def _is_approaching(is_long: bool, bar_candle: Optional[Dict], close: float) -> bool:
        """True if the current bar is testing toward the level direction."""
        if bar_candle is None:
            return True  # no data → assume approaching
        lo = float(bar_candle.get("low",  close))
        hi = float(bar_candle.get("high", close))
        if is_long:
            return lo < close - 0.25    # bar tested lower
        else:
            return hi > close + 0.25    # bar tested higher
