"""
Level-Aware Fill Optimizer
==========================
Uses mapped structural and bank levels to decide whether to:
  IMMEDIATE: enter at next market opportunity
  AT_LEVEL:  enter because price is already sitting on a meaningful level
  WAIT:      defer for a short pullback into a nearby level
"""

import logging
from typing import Dict, List, Optional


FILL_IMMEDIATE = "IMMEDIATE"
FILL_AT_LEVEL = "AT_LEVEL"
FILL_WAIT = "WAIT"

BANK_GRID = 12.50
AT_LEVEL_THRESHOLD = 0.75
NEAR_LEVEL_MIN = 0.75
NEAR_LEVEL_MAX = 2.50
MAX_WAIT_BARS = 3
RAN_AWAY_MULT = 3.0


def _bank_grid_levels(price: float) -> List[Dict]:
    base = (price // BANK_GRID) * BANK_GRID
    levels = []
    for offset in range(-2, 3):
        level_price = base + (offset * BANK_GRID)
        levels.append(
            {
                "price": level_price,
                "name": f"Bank_{level_price:.2f}",
                "type": "bank",
                "priority": 1,
            }
        )
    return levels


def _levels_from_bank_filter(bank_filter) -> List[Dict]:
    if bank_filter is None:
        return []
    try:
        state = bank_filter.get_state()
    except Exception:
        return []
    mapping = [
        ("prev_session_high", "PrevSessH", "structural", 4),
        ("prev_session_low", "PrevSessL", "structural", 4),
        ("prev_day_pm_high", "PrevPMH", "structural", 3),
        ("prev_day_pm_low", "PrevPML", "structural", 3),
        ("midnight_orb_high", "MidnightORB_H", "orb", 3),
        ("midnight_orb_low", "MidnightORB_L", "orb", 3),
        ("bank_above_prev_session_high", "BankAboveSessH", "bank", 2),
        ("bank_below_prev_session_low", "BankBelowSessL", "bank", 2),
        ("bank_above_orb_high", "BankAboveORBH", "bank", 2),
        ("bank_below_orb_low", "BankBelowORBL", "bank", 2),
    ]
    levels = []
    for key, name, level_type, priority in mapping:
        value = state.get(key)
        if value is None:
            continue
        try:
            levels.append(
                {
                    "price": float(value),
                    "name": name,
                    "type": level_type,
                    "priority": priority,
                }
            )
        except (TypeError, ValueError):
            continue
    return levels


def _bank_levels_adjacent_to_structural(structural_levels: List[Dict]) -> List[Dict]:
    seen = set()
    extra: List[Dict] = []
    for level in structural_levels:
        price = level["price"]
        parent_name = level["name"]
        parent_priority = level["priority"]
        bank_below = (price // BANK_GRID) * BANK_GRID
        bank_above = bank_below + BANK_GRID

        for bank_price, suffix in ((bank_below, "Below"), (bank_above, "Above")):
            key = round(bank_price, 4)
            if key in seen:
                continue
            seen.add(key)
            extra.append(
                {
                    "price": bank_price,
                    "name": f"Bank{suffix}_{parent_name}",
                    "type": "confluence",
                    "priority": parent_priority,
                }
            )

        for mid_price, suffix in (
            ((price + bank_below) / 2, "MidBelow"),
            ((price + bank_above) / 2, "MidAbove"),
        ):
            key = round(mid_price, 4)
            if key in seen:
                continue
            seen.add(key)
            extra.append(
                {
                    "price": mid_price,
                    "name": f"BankMid{suffix}_{parent_name}",
                    "type": "confluence_mid",
                    "priority": max(1, parent_priority - 1),
                }
            )
    return extra


class LevelFillOptimizer:
    """Evaluates fill strategy and tracks pending deferred entries."""

    def __init__(self):
        self._pending: Dict[str, Dict] = {}

    def evaluate(
        self,
        signal: Dict,
        current_price: float,
        structural_tracker=None,
        bank_filter=None,
        bar_candle: Optional[Dict] = None,
    ) -> Dict:
        side = (signal.get("side") or "").upper()
        is_long = side == "LONG"

        all_levels: List[Dict] = _bank_grid_levels(current_price)
        all_levels += _levels_from_bank_filter(bank_filter)

        structural_levels: List[Dict] = []
        bank_filter_levels = _levels_from_bank_filter(bank_filter)
        structural_levels += [
            level
            for level in bank_filter_levels
            if level["type"] in ("structural", "orb")
        ]

        if structural_tracker is not None:
            try:
                tracker_levels = structural_tracker.get_active_levels()
                all_levels += tracker_levels
                structural_levels += tracker_levels
            except Exception:
                pass

        all_levels += _bank_levels_adjacent_to_structural(structural_levels)

        if is_long:
            candidates = [level for level in all_levels if level["price"] <= current_price]
            if not candidates:
                return self._immediate("no support levels below")
            closest = max(candidates, key=lambda level: level["price"])
        else:
            candidates = [level for level in all_levels if level["price"] >= current_price]
            if not candidates:
                return self._immediate("no resistance levels above")
            closest = min(candidates, key=lambda level: level["price"])

        dist = abs(current_price - closest["price"])
        sl_dist = float(signal.get("sl_dist") or 0.0)

        if dist <= AT_LEVEL_THRESHOLD:
            return {
                "mode": FILL_AT_LEVEL,
                "target_price": closest["price"],
                "target_name": closest["name"],
                "dist": dist,
                "reason": f"at {closest['name']} ({dist:.2f}pts)",
            }

        if NEAR_LEVEL_MIN < dist <= NEAR_LEVEL_MAX and closest["priority"] >= 2:
            if sl_dist > 0.0 and dist >= (sl_dist + AT_LEVEL_THRESHOLD - 1e-9):
                return self._immediate(
                    (
                        f"level wait invalid: {closest['name']} is beyond protected drift "
                        f"({dist:.2f}pts >= sl {sl_dist:.2f}pts + touch buffer)"
                    ),
                    closest,
                )
            if (
                closest["type"] == "bank"
                and closest["priority"] < 3
                and dist > 1.5
            ):
                return self._immediate(
                    f"bank level too far ({dist:.2f}pts)",
                    closest,
                )
            approaching = self._is_approaching(is_long, bar_candle, current_price)
            if approaching or dist <= 1.5:
                return {
                    "mode": FILL_WAIT,
                    "target_price": closest["price"],
                    "target_name": closest["name"],
                    "dist": dist,
                    "max_bars": MAX_WAIT_BARS,
                    "reason": (
                        f"waiting for {closest['name']}@{closest['price']:.2f} "
                        f"({dist:.2f}pts)"
                    ),
                }

        return self._immediate(
            f"level {closest['name']} too far or not approaching ({dist:.2f}pts)",
            closest,
        )

    def add_pending(
        self,
        uid: str,
        signal: Dict,
        decision: Dict,
        current_price: float,
    ) -> None:
        self._pending[uid] = {
            "signal": signal,
            "target_price": decision["target_price"],
            "target_name": decision["target_name"],
            "max_bars": decision.get("max_bars", MAX_WAIT_BARS),
            "bars_waited": 0,
            "entry_price": current_price,
        }
        logging.info(
            "LevelFill queued: %s %s -> %s@%.2f (%.2f pts away, max %d bars)",
            signal.get("strategy", "?"),
            signal.get("side", "?"),
            decision["target_name"],
            decision["target_price"],
            decision.get("dist", 0.0),
            decision.get("max_bars", MAX_WAIT_BARS),
        )

    def check_pending(self, uid: str, bar: Dict) -> Dict:
        if uid not in self._pending:
            return {"fire": False, "abort": True, "reason": "uid not found"}

        entry = self._pending[uid]
        entry["bars_waited"] += 1
        target = entry["target_price"]
        signal = entry["signal"]
        is_long = signal.get("side", "").upper() == "LONG"
        sl_dist = float(signal.get("sl_dist") or 0.0)
        bar_low = float(bar.get("low", bar.get("close", target)))
        bar_high = float(bar.get("high", bar.get("close", target)))
        bar_close = float(bar.get("close", entry["entry_price"]))
        orig = entry["entry_price"]
        init_dist = abs(orig - target)
        close_dist = abs(bar_close - target)

        level_violated = (
            (is_long and bar_close < target - AT_LEVEL_THRESHOLD)
            or (not is_long and bar_close > target + AT_LEVEL_THRESHOLD)
        )
        sl_breached = sl_dist > 0 and (
            (is_long and bar_low < orig - sl_dist)
            or (not is_long and bar_high > orig + sl_dist)
        )
        ran_away = (
            (is_long and bar_close > orig + (RAN_AWAY_MULT * init_dist))
            or (not is_long and bar_close < orig - (RAN_AWAY_MULT * init_dist))
        )
        touched = (
            (is_long and bar_low <= target + AT_LEVEL_THRESHOLD)
            or (not is_long and bar_high >= target - AT_LEVEL_THRESHOLD)
        )
        close_near_target = close_dist <= (AT_LEVEL_THRESHOLD + 1e-9)
        timed_out = entry["bars_waited"] >= entry["max_bars"]

        if sl_breached:
            del self._pending[uid]
            return {
                "fire": False,
                "abort": True,
                "reason": (
                    f"SL breached on paper ({orig:.2f} -> sl {sl_dist:.2f}pts) "
                    f"after {entry['bars_waited']} bar(s)"
                ),
            }
        if level_violated:
            del self._pending[uid]
            return {
                "fire": False,
                "abort": True,
                "reason": (
                    f"target level {target:.2f} violated (close={bar_close:.2f}) - "
                    "support/resistance broken"
                ),
            }
        if ran_away:
            del self._pending[uid]
            return {
                "fire": False,
                "abort": True,
                "reason": (
                    f"price ran {RAN_AWAY_MULT}x init dist away from {target:.2f} "
                    "without pullback"
                ),
            }
        if touched and close_near_target:
            del self._pending[uid]
            return {
                "fire": True,
                "abort": False,
                "reason": f"level {target:.2f} touched after {entry['bars_waited']} bar(s)",
            }
        if timed_out:
            adverse_drift = (
                (
                    is_long and ((orig - bar_close) > (sl_dist * 0.5))
                ) or (
                    (not is_long) and ((bar_close - orig) > (sl_dist * 0.5))
                )
            ) if sl_dist > 0 else False
            del self._pending[uid]
            if adverse_drift:
                return {
                    "fire": False,
                    "abort": True,
                    "reason": (
                        f"timeout + adverse drift >{sl_dist * 0.5:.2f}pts - "
                        "setup stale, aborting"
                    ),
                }
            return {
                "fire": True,
                "abort": False,
                "reason": (
                    f"timeout ({entry['bars_waited']} bars) - no invalidation, "
                    "executing at market"
                ),
            }
        if touched and not close_near_target:
            return {
                "fire": False,
                "abort": False,
                "reason": (
                    f"level {target:.2f} touched intrabar but closed {close_dist:.2f}pts away"
                ),
            }
        return {
            "fire": False,
            "abort": False,
            "reason": f"waiting {entry['bars_waited']}/{entry['max_bars']} bars",
        }

    def get_pending_signal(self, uid: str) -> Optional[Dict]:
        return self._pending.get(uid)

    def get_pending_ids(self) -> List[str]:
        return list(self._pending.keys())

    def remove_pending(self, uid: str) -> None:
        self._pending.pop(uid, None)

    def clear_all(self, reason: str = "") -> None:
        if self._pending:
            logging.info("LevelFill clearing %d pending fills - %s", len(self._pending), reason)
        self._pending.clear()

    @staticmethod
    def _immediate(reason: str, closest: Optional[Dict] = None) -> Dict:
        return {
            "mode": FILL_IMMEDIATE,
            "target_price": closest["price"] if closest else None,
            "target_name": closest["name"] if closest else None,
            "dist": None,
            "reason": reason,
        }

    @staticmethod
    def _is_approaching(is_long: bool, bar_candle: Optional[Dict], close: float) -> bool:
        if bar_candle is None:
            return True
        low = float(bar_candle.get("low", close))
        high = float(bar_candle.get("high", close))
        if is_long:
            return low < (close - 0.25)
        return high > (close + 0.25)
