"""Anti-flip circuit breaker.

Rejects a new signal when the **opposite-side** trade just stopped out
close to the new signal's entry price. The pattern this filter catches
is what cost the bot on 2026-04-23: a SHORT stopped at 7172.50 (loss),
and sixty seconds later DE3 fired a LONG at 7171.75 — the same price
region the SHORT just died defending. That LONG rode into a 64-point
dump and lost. Two trades wasted on the same micro-structure noise.

Rule:
    If the most recent OPPOSITE-SIDE closed trade:
      - was stopped out (PnL < 0 via source='stop' or 'stop_gap'), AND
      - stopped within `window_minutes` of the current signal, AND
      - the current signal's entry is within `max_distance_pts` of the
        prior stop's exit price,
    THEN block the current signal.

Defaults (from backtest sweep over 2025 full year + 2026 Jan–Apr):
    window_minutes = 30
    max_distance_pts = 8.0
Selected as the best-combined-lift configuration that is net positive
on both tapes; see `scripts/backtest_anti_flip_blocker.py` for the
full 25-config grid.

Interface mirrors `DirectionalLossBlocker` / `CascadeLossBlocker` —
drops into the existing entry-path hooks and the close-path record
path with no bespoke plumbing.
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


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    try:
        return float(raw) if raw is not None and raw.strip() else default
    except (ValueError, TypeError):
        return default


def _flag_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return raw in ("1", "true", "True", "TRUE", "yes", "on")


class AntiFlipBlocker:
    """Side-by-side with DirectionalLossBlocker and CascadeLossBlocker;
    each filter catches a different pathological pattern."""

    def __init__(
        self,
        *,
        active: Optional[bool] = None,
        window_minutes: Optional[int] = None,
        max_distance_pts: Optional[float] = None,
    ) -> None:
        self.active = (
            active if active is not None
            else _flag_env("JULIE_ANTI_FLIP_BLOCKER_ACTIVE", False)
        )
        self.window_minutes = (
            window_minutes if window_minutes is not None
            else _int_env("JULIE_ANTI_FLIP_WINDOW_MIN", 30)
        )
        self.max_distance_pts = (
            max_distance_pts if max_distance_pts is not None
            else _float_env("JULIE_ANTI_FLIP_MAX_DIST_PTS", 8.0)
        )

        # Last stop-out per side. None means no stop-out on record.
        self._last_stop_long_ts: Optional[_dt.datetime] = None
        self._last_stop_long_px: Optional[float] = None
        self._last_stop_short_ts: Optional[_dt.datetime] = None
        self._last_stop_short_px: Optional[float] = None

        logging.info(
            "[AntiFlipBlocker] active=%s window=%dmin max_dist=%.1fpt",
            self.active, self.window_minutes, self.max_distance_pts,
        )

    # ─── persistence ─────────────────────────────────────────
    def get_state(self) -> dict:
        return {
            "active": self.active,
            "window_minutes": self.window_minutes,
            "max_distance_pts": self.max_distance_pts,
            "last_stop_long_ts": (
                self._last_stop_long_ts.isoformat()
                if self._last_stop_long_ts else None
            ),
            "last_stop_long_px": self._last_stop_long_px,
            "last_stop_short_ts": (
                self._last_stop_short_ts.isoformat()
                if self._last_stop_short_ts else None
            ),
            "last_stop_short_px": self._last_stop_short_px,
        }

    def load_state(self, state: dict | None) -> None:
        if not state:
            return
        try:
            lt = state.get("last_stop_long_ts")
            self._last_stop_long_ts = _dt.datetime.fromisoformat(lt) if lt else None
            self._last_stop_long_px = (
                float(state["last_stop_long_px"])
                if state.get("last_stop_long_px") is not None else None
            )
            st = state.get("last_stop_short_ts")
            self._last_stop_short_ts = _dt.datetime.fromisoformat(st) if st else None
            self._last_stop_short_px = (
                float(state["last_stop_short_px"])
                if state.get("last_stop_short_px") is not None else None
            )
        except (ValueError, TypeError, KeyError) as e:
            logging.warning(
                "[AntiFlipBlocker] state load error: %s — starting fresh", e
            )
            self._last_stop_long_ts = None
            self._last_stop_long_px = None
            self._last_stop_short_ts = None
            self._last_stop_short_px = None

    # ─── event recording ─────────────────────────────────────
    def record_trade_close(
        self,
        side: str,
        pnl: float,
        exit_price: float,
        *,
        source: str = "",
        sl_price: Optional[float] = None,
        close_time: Optional[_dt.datetime] = None,
    ) -> None:
        """Register a closed trade. The blocker only cares about stop-out
        exits with negative PnL; everything else is ignored.

        A close is classified as a stop-out if the PnL is negative AND
        EITHER of the following is true:
          - source string contains "stop" (matches the backtest-tape
            sources "stop" / "stop_gap" and the live log-prefix
            "confirmed stop fill")
          - sl_price is provided and |exit_price - sl_price| <= 1.5pt
            (covers live paths where `source` is a broker label like
            "trade_search_exact_order" rather than a semantic tag)

        Either one alone is sufficient — the dual test exists because
        the backtest tape tags the source semantically while the live
        path surfaces it as a broker tag.

        Args:
            side: "LONG" or "SHORT" of the trade that just closed.
            pnl: realized PnL (dollars or points, only the SIGN is used).
            exit_price: actual fill price of the exit — used as the
                        reference price for the distance check.
            source: exit source string (optional).
            sl_price: stop price that was on the trade when it closed
                      (optional; used as a fallback detector).
            close_time: when the trade closed. Defaults to now().
        """
        if close_time is None:
            close_time = _dt.datetime.now()
        if pnl >= 0:
            return  # only stop-OUT losses matter
        src_lower = str(source or "").lower()
        looks_like_stop = "stop" in src_lower
        if not looks_like_stop and sl_price is not None:
            try:
                looks_like_stop = abs(float(exit_price) - float(sl_price)) <= 1.5
            except (ValueError, TypeError):
                looks_like_stop = False
        if not looks_like_stop:
            return
        side_upper = str(side).upper()
        if side_upper == "LONG":
            self._last_stop_long_ts = close_time
            self._last_stop_long_px = float(exit_price)
        elif side_upper == "SHORT":
            self._last_stop_short_ts = close_time
            self._last_stop_short_px = float(exit_price)

    # ─── entry-time query ────────────────────────────────────
    def should_block_trade(
        self,
        side: str,
        entry_price: float,
        current_time: Optional[_dt.datetime] = None,
    ) -> tuple[bool, str]:
        """Return (blocked, reason). `reason` is empty when blocked=False.

        Used at the entry path exactly like
        `directional_loss_blocker.should_block_trade(side, time)`,
        with the addition of the entry_price argument for the distance
        check.
        """
        if not self.active:
            return False, ""
        if current_time is None:
            current_time = _dt.datetime.now()
        side_upper = str(side).upper()
        # Check the OPPOSITE side's last stop-out
        if side_upper == "LONG":
            last_ts = self._last_stop_short_ts
            last_px = self._last_stop_short_px
            opp_name = "SHORT"
        elif side_upper == "SHORT":
            last_ts = self._last_stop_long_ts
            last_px = self._last_stop_long_px
            opp_name = "LONG"
        else:
            return False, ""
        if last_ts is None or last_px is None:
            return False, ""
        elapsed_min = (current_time - last_ts).total_seconds() / 60.0
        if elapsed_min < 0 or elapsed_min > self.window_minutes:
            return False, ""
        try:
            ep = float(entry_price)
        except (ValueError, TypeError):
            return False, ""
        dist = abs(ep - last_px)
        if dist > self.max_distance_pts:
            return False, ""
        reason = (
            f"{opp_name} stop {elapsed_min:.1f}min ago @ {last_px:.2f}; "
            f"{side_upper} entry {ep:.2f} Δ={dist:.2f}pt "
            f"(window={self.window_minutes}min, max_dist={self.max_distance_pts:.1f}pt)"
        )
        return True, reason


class _NoOpAntiFlipBlocker:
    """Null object for the off-by-default case."""
    active = False
    window_minutes = 0
    max_distance_pts = 0.0

    def record_trade_close(self, *a, **kw) -> None:
        return None

    def should_block_trade(self, *a, **kw) -> tuple[bool, str]:
        return False, ""

    def get_state(self) -> dict:
        return {}

    def load_state(self, state: Any) -> None:
        return None


__all__ = ["AntiFlipBlocker", "_NoOpAntiFlipBlocker"]
