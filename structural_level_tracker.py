"""
Structural Level Tracker
Tracks key price levels from the weswesindicator:
  - True Open (TO): 90 min after each session start
  - Q1 H/L: High/Low of the first 90 min window (before True Open locks in)
  - Daily Open: 18:00 ET (futures session start)
  - London ORB: 03:00–03:30 AM ET (first 30 min of London session)
  - Morning ORB: 09:30–09:59 AM ET (first 30 min of regular NYSE session)

All times in ET (US/Eastern). Complements BankLevelQuarterFilter which already
tracks prev-session H/L and midnight ORB (00:00–00:15 ET).

Session timing (ET):
  ASIA:   21:00 – 03:00  |  True Open at 22:30 ET
  LONDON: 03:00 – 09:00  |  True Open at 04:30 ET
  NY:     09:00 – 15:00  |  True Open at 10:30 ET
  PM:     15:00 – 21:00  |  True Open at 16:30 ET
"""

import datetime
import logging
from typing import Optional, List, Dict


_TRUE_OPEN_TIMES: Dict[str, tuple] = {
    "ASIA":   (22, 30),
    "LONDON": (4,  30),
    "NY":     (10, 30),
    "PM":     (16, 30),
}

_SESSION_BOUNDS: Dict[str, tuple] = {
    # (start_h, end_h) — hour in ET; Asia wraps midnight
    "ASIA":   (21, 3),
    "LONDON": (3,  9),
    "NY":     (9,  15),
    "PM":     (15, 21),
}


def _session_for_hour(h: int) -> Optional[str]:
    if h >= 21 or h < 3:
        return "ASIA"
    if 3 <= h < 9:
        return "LONDON"
    if 9 <= h < 15:
        return "NY"
    if 15 <= h < 21:
        return "PM"
    return None


class StructuralLevelTracker:
    """Tracks True Open, Q1 H/L, Daily Open, London ORB, and Morning ORB."""

    def __init__(self):
        self.reset()

    def reset(self):
        # True Open (locked 90 min into each session)
        self.true_open: Optional[float] = None
        self.true_open_session: Optional[str] = None
        self._to_captured: bool = False

        # Q1 H/L (rolling high/low before True Open is captured)
        self.q1_high: Optional[float] = None
        self.q1_low: Optional[float] = None
        self._q1_active: bool = False
        self.q1_session: Optional[str] = None

        # Daily Open — first bar of the 18:00 ET futures session
        self.daily_open: Optional[float] = None
        self._daily_open_date: Optional[datetime.date] = None

        # London ORB: 03:00–03:29 AM ET
        self.london_orb_h: Optional[float] = None
        self.london_orb_l: Optional[float] = None
        self._in_london_orb: bool = False

        # Morning ORB: 09:30–09:59 AM ET
        self.morn_orb_h: Optional[float] = None
        self.morn_orb_l: Optional[float] = None
        self._in_morn_orb: bool = False

        self._current_session: Optional[str] = None

    # ------------------------------------------------------------------
    def update(
        self,
        ts: datetime.datetime,
        o: float,
        h: float,
        l: float,
        c: float,
    ) -> None:
        """Feed one completed bar. ts must be timezone-aware (ET preferred)."""
        et_h = ts.hour
        et_m = ts.minute
        session = _session_for_hour(et_h)

        # ── New session boundary ───────────────────────────────────────
        if session != self._current_session:
            self._current_session = session
            self._to_captured = False
            self.true_open = None
            self.true_open_session = None
            self.q1_high = h
            self.q1_low = l
            self._q1_active = True
            self.q1_session = session
            logging.debug("StructuralLevelTracker: new session %s", session)

        # ── Daily Open (18:00 ET = futures session open) ───────────────
        if et_h == 18 and et_m == 0:
            today = ts.date()
            if today != self._daily_open_date:
                self.daily_open = o
                self._daily_open_date = today
                logging.debug("StructuralLevelTracker: DailyOpen=%.2f", o)

        # ── Q1 rolling H/L ────────────────────────────────────────────
        if self._q1_active:
            if self.q1_high is None or h > self.q1_high:
                self.q1_high = h
            if self.q1_low is None or l < self.q1_low:
                self.q1_low = l

        # ── True Open capture ─────────────────────────────────────────
        if session and not self._to_captured:
            th, tm = _TRUE_OPEN_TIMES.get(session, (-1, -1))
            if et_h == th and et_m == tm:
                self.true_open = o
                self.true_open_session = session
                self._to_captured = True
                self._q1_active = False
                logging.debug(
                    "StructuralLevelTracker: TrueOpen=%s %.2f", session, o
                )

        # ── London ORB: 03:00–03:29 AM ET ─────────────────────────────
        in_london = et_h == 3 and et_m < 30
        if in_london and not self._in_london_orb:
            # First bar of London ORB — reset
            self.london_orb_h = h
            self.london_orb_l = l
        elif in_london:
            if self.london_orb_h is None or h > self.london_orb_h:
                self.london_orb_h = h
            if self.london_orb_l is None or l < self.london_orb_l:
                self.london_orb_l = l
        self._in_london_orb = in_london

        # ── Morning ORB: 09:30–09:59 AM ET ───────────────────────────
        in_morn = et_h == 9 and et_m >= 30
        if in_morn and not self._in_morn_orb:
            self.morn_orb_h = h
            self.morn_orb_l = l
        elif in_morn:
            if self.morn_orb_h is None or h > self.morn_orb_h:
                self.morn_orb_h = h
            if self.morn_orb_l is None or l < self.morn_orb_l:
                self.morn_orb_l = l
        self._in_morn_orb = in_morn

    # ------------------------------------------------------------------
    def get_active_levels(self) -> List[Dict]:
        """Return all currently tracked structural levels as dicts."""
        levels = []

        def _add(price, name, ltype, priority):
            if price is not None:
                levels.append(
                    {"price": float(price), "name": name, "type": ltype, "priority": priority}
                )

        _add(self.true_open,  f"TO_{self.true_open_session or 'UNK'}",  "structural", 4)
        _add(self.daily_open, "DailyOpen",                               "structural", 4)
        _add(self.q1_high,    f"Q1H_{self.q1_session or 'UNK'}",        "structural", 3)
        _add(self.q1_low,     f"Q1L_{self.q1_session or 'UNK'}",        "structural", 3)
        _add(self.london_orb_h, "LondonORB_H",                           "orb",        3)
        _add(self.london_orb_l, "LondonORB_L",                           "orb",        3)
        _add(self.morn_orb_h,   "MornORB_H",                             "orb",        4)
        _add(self.morn_orb_l,   "MornORB_L",                             "orb",        4)

        return levels

    # ------------------------------------------------------------------
    def get_state(self) -> dict:
        return {
            "true_open":         self.true_open,
            "true_open_session": self.true_open_session,
            "_to_captured":      self._to_captured,
            "q1_high":           self.q1_high,
            "q1_low":            self.q1_low,
            "_q1_active":        self._q1_active,
            "q1_session":        self.q1_session,
            "daily_open":        self.daily_open,
            "_daily_open_date":  self._daily_open_date.isoformat() if self._daily_open_date else None,
            "london_orb_h":      self.london_orb_h,
            "london_orb_l":      self.london_orb_l,
            "morn_orb_h":        self.morn_orb_h,
            "morn_orb_l":        self.morn_orb_l,
            "_current_session":  self._current_session,
        }

    def load_state(self, state: dict) -> None:
        if not state:
            return
        self.true_open         = state.get("true_open")
        self.true_open_session = state.get("true_open_session")
        self._to_captured      = bool(state.get("_to_captured", False))
        self.q1_high           = state.get("q1_high")
        self.q1_low            = state.get("q1_low")
        self._q1_active        = bool(state.get("_q1_active", False))
        self.q1_session        = state.get("q1_session")
        self.daily_open        = state.get("daily_open")
        do_date                = state.get("_daily_open_date")
        self._daily_open_date  = datetime.date.fromisoformat(do_date) if do_date else None
        self.london_orb_h      = state.get("london_orb_h")
        self.london_orb_l      = state.get("london_orb_l")
        self.morn_orb_h        = state.get("morn_orb_h")
        self.morn_orb_l        = state.get("morn_orb_l")
        self._current_session  = state.get("_current_session")
