"""Time-window cascade circuit breaker.

This is a SECOND, DIFFERENT filter alongside the existing
`DirectionalLossBlocker` (which counts strictly-consecutive losses with
no time bound). This one fires on a **rapid-fire loss cluster within a
sliding time window**, with a cooldown timer.

Rule:
    If within the last `window_minutes` minutes there have been ≥ `count`
    LOSING trades on the same SIDE, block every new same-side entry for
    `cooldown_minutes` after the most recent same-side loss.

Backtest (2025+2026 closed_trades.json, `scripts/backtest_consec_loss_blocker.py`):
    count=2 / window=30min / cool=30min on 5,237 trades across 370 days:
        2026: +$1,133 vs baseline  (PF 1.12 → 1.17, DD ↓$1.5k)
        2025: +$4,222 vs baseline  (PF 1.04 → 1.07, DD ↓$1.1k)
    Blocks ~11% of trades; counterfactual confirms blocked set is net-losing.

Interface mirrors `directional_loss_blocker.DirectionalLossBlocker` so
the julie001.py insertion is a 1-line copy of the existing dir_blocked
check. Default state is OFF — it only does anything when explicitly
activated via env var JULIE_CASCADE_BLOCKER_ACTIVE=1.

Tunables (env vars, read at construction time):
    JULIE_CASCADE_BLOCKER_ACTIVE      — "1" to enable, anything else = OFF
    JULIE_CASCADE_BLOCKER_COUNT       — int, default 2
    JULIE_CASCADE_BLOCKER_WINDOW_MIN  — int, default 30
    JULIE_CASCADE_BLOCKER_COOLDOWN_MIN— int, default 30

State is persistable for bot restarts.
"""
from __future__ import annotations

import datetime as _dt
import logging
import os
from typing import Any, Optional


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    try:
        return int(raw) if raw is not None and raw.strip() else default
    except (ValueError, TypeError):
        return default


def _flag_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return raw in ("1", "true", "True", "TRUE", "yes", "on")


class CascadeLossBlocker:
    """Time-window loss cascade filter. See module docstring."""

    def __init__(
        self,
        *,
        active: Optional[bool] = None,
        count: Optional[int] = None,
        window_minutes: Optional[int] = None,
        cooldown_minutes: Optional[int] = None,
    ) -> None:
        # Resolve params — explicit args win, else env, else defaults
        self.active = (
            active if active is not None
            else _flag_env("JULIE_CASCADE_BLOCKER_ACTIVE", False)
        )
        self.count = count if count is not None else _int_env("JULIE_CASCADE_BLOCKER_COUNT", 2)
        self.window_minutes = (
            window_minutes if window_minutes is not None
            else _int_env("JULIE_CASCADE_BLOCKER_WINDOW_MIN", 30)
        )
        self.cooldown_minutes = (
            cooldown_minutes if cooldown_minutes is not None
            else _int_env("JULIE_CASCADE_BLOCKER_COOLDOWN_MIN", 30)
        )

        # Per-side queues of (timestamp, was_loss_bool). Only recent entries.
        self._long_events: list[tuple[_dt.datetime, bool]] = []
        self._short_events: list[tuple[_dt.datetime, bool]] = []

        logging.info(
            "[CascadeBlocker] active=%s count=%d window=%dmin cooldown=%dmin",
            self.active, self.count, self.window_minutes, self.cooldown_minutes,
        )

    # ─── persistence ─────────────────────────────────────────
    def get_state(self) -> dict:
        return {
            "count": self.count,
            "window_minutes": self.window_minutes,
            "cooldown_minutes": self.cooldown_minutes,
            "active": self.active,
            "long_events": [(t.isoformat(), b) for t, b in self._long_events],
            "short_events": [(t.isoformat(), b) for t, b in self._short_events],
        }

    def load_state(self, state: dict | None) -> None:
        if not state:
            return
        try:
            self._long_events = [
                (_dt.datetime.fromisoformat(t), bool(b))
                for t, b in state.get("long_events", [])
            ]
            self._short_events = [
                (_dt.datetime.fromisoformat(t), bool(b))
                for t, b in state.get("short_events", [])
            ]
        except (ValueError, TypeError) as e:
            logging.warning("[CascadeBlocker] state load error: %s — starting fresh", e)
            self._long_events = []
            self._short_events = []

    # ─── event recording ─────────────────────────────────────
    def record_trade_result(
        self,
        side: str,
        pnl: float,
        current_time: Optional[_dt.datetime] = None,
    ) -> None:
        """Register a closed trade outcome. Always called — the blocker
        only consults this buffer when `should_block_trade` runs."""
        if current_time is None:
            current_time = _dt.datetime.now()
        is_loss = pnl < 0
        bucket = self._long_events if str(side).upper() == "LONG" else self._short_events
        bucket.append((current_time, is_loss))
        # Trim — keep only events inside window + cooldown
        cutoff = current_time - _dt.timedelta(
            minutes=max(self.window_minutes, self.cooldown_minutes) + 5
        )
        bucket[:] = [(t, b) for t, b in bucket if t >= cutoff]

    # ─── entry-time query ────────────────────────────────────
    def should_block_trade(
        self,
        side: str,
        current_time: Optional[_dt.datetime] = None,
    ) -> tuple[bool, str]:
        """Return (blocked, reason). `reason` is an empty string when
        blocked=False. Used exactly like directional_loss_blocker's
        same-named method."""
        if not self.active:
            return False, ""
        if current_time is None:
            current_time = _dt.datetime.now()
        bucket = self._long_events if str(side).upper() == "LONG" else self._short_events
        # Count same-side losses in the window
        window_cutoff = current_time - _dt.timedelta(minutes=self.window_minutes)
        recent_losses = [t for t, is_loss in bucket if is_loss and t >= window_cutoff]
        if len(recent_losses) < self.count:
            return False, ""
        # Cooldown is measured from the MOST RECENT same-side loss
        last_loss = max(recent_losses)
        if current_time - last_loss > _dt.timedelta(minutes=self.cooldown_minutes):
            return False, ""
        mins_remaining = self.cooldown_minutes - (
            (current_time - last_loss).total_seconds() / 60.0
        )
        reason = (
            f"{len(recent_losses)} {side} losses in last {self.window_minutes}min; "
            f"cooldown for {mins_remaining:.1f}min more"
        )
        return True, reason

    # ─── inspection helpers ──────────────────────────────────
    def recent_loss_count(self, side: str, current_time: Optional[_dt.datetime] = None) -> int:
        if current_time is None:
            current_time = _dt.datetime.now()
        bucket = self._long_events if str(side).upper() == "LONG" else self._short_events
        window_cutoff = current_time - _dt.timedelta(minutes=self.window_minutes)
        return sum(1 for t, is_loss in bucket if is_loss and t >= window_cutoff)


class _NoOpCascadeLossBlocker:
    """Null object used when the blocker is configured off — keeps
    julie001.py call sites branch-free."""
    active = False
    count = 0
    window_minutes = 0
    cooldown_minutes = 0

    def record_trade_result(self, *a, **kw) -> None:  # noqa: D401
        return None

    def should_block_trade(self, *a, **kw) -> tuple[bool, str]:
        return False, ""

    def recent_loss_count(self, *a, **kw) -> int:
        return 0

    def get_state(self) -> dict:
        return {}

    def load_state(self, state: Any) -> None:
        return None


__all__ = ["CascadeLossBlocker", "_NoOpCascadeLossBlocker"]
