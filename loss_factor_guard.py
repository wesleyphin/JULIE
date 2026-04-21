"""Loss-Factor Guard — veto entries based on 2025 forensic patterns.

Gated by JULIE_LOSS_FACTOR_GUARD=1. Default off = no behavior change.

Quantified from 2025 big-loss-day forensics (63 days with >$300 loss):
  - LONG bias failure: big-loss days LONG 36% WR (-$23,882) vs winning days 60%.
  - Morning cascade: 42% of big-loss days had first 3 AM trades all stop.
  - Afternoon shutdown: hour 15 ET on losing days = 10% WR, -$776.
  - Stop:take ratio: winning=3x, losing=10x; >5x is the discriminator.
  - Consec-loss median: 4 on big-loss days. Often reaches 6-8 unchecked.

Public API:
  init_guard() -> Optional[LossFactorGuard]
  get_guard() -> Optional[LossFactorGuard]
  should_veto_entry(signal) -> tuple[bool, str]
  notify_trade_closed(trade_dict)
  notify_new_day(day)

Tunable env vars (all with safe defaults):
  JULIE_LFG_LONG_STREAK (default 3)          consecutive LONG stops -> veto LONGs 30 min
  JULIE_LFG_SHORT_STREAK (default 4)         consecutive SHORT stops -> veto SHORTs 30 min
  JULIE_LFG_MORNING_CASCADE (default 3)      first-N AM trades stop -> veto until 11 ET
  JULIE_LFG_AFT_SHUTDOWN_PNL (default -200)  cum PnL threshold at 15:00 for afternoon shutdown
  JULIE_LFG_STOP_TAKE_RATIO (default 5.0)    rolling 5-trade stop:take ratio limit
  JULIE_LFG_VETO_MINUTES (default 30)        duration of side-veto after streak trips
"""
from __future__ import annotations

import logging
import os
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from typing import Deque, Optional, Tuple


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default


LONG_STREAK = _env_int("JULIE_LFG_LONG_STREAK", 3)
SHORT_STREAK = _env_int("JULIE_LFG_SHORT_STREAK", 4)
MORNING_CASCADE_N = _env_int("JULIE_LFG_MORNING_CASCADE", 3)
AFT_SHUTDOWN_PNL = _env_float("JULIE_LFG_AFT_SHUTDOWN_PNL", -200.0)
STOP_TAKE_RATIO_LIMIT = _env_float("JULIE_LFG_STOP_TAKE_RATIO", 5.0)
VETO_MINUTES = _env_int("JULIE_LFG_VETO_MINUTES", 30)
ROLLING_WINDOW = 5

# Reversal-on-strong-trend veto (filter C): when the session is in a
# confirmed trend day AND the signal is a REVERSAL (sub_strategy contains
# "_Rev_") that tries to fade the trend direction, block it. Backtest on 15
# outrageous-breakout 2025 days showed Julie kept firing LONG Rev into
# tariff selling (Apr 4/7/8/10/16 all trend-down days with net -$2,100).
# Minimum trend-day tier that activates the filter (1 = weak, 2 = confirmed,
# 3 = strong). Default tier>=1 so the filter is aggressive; raise to 2 if
# it bleeds too many valid reversals.
TREND_BIAS_MIN_TIER = _env_int("JULIE_LFG_TREND_BIAS_MIN_TIER", 1)


@dataclass
class DailyState:
    day: Optional[date] = None
    long_stop_streak: int = 0
    short_stop_streak: int = 0
    cum_pnl: float = 0.0
    # Last N trade outcomes for rolling ratio: list of source strings
    recent_sources: Deque[str] = field(default_factory=lambda: deque(maxlen=ROLLING_WINDOW))
    # Morning-stop counter (hours 8-10 ET)
    am_trades_done: int = 0
    am_stops: int = 0
    morning_cascade_active: bool = False
    afternoon_shutdown: bool = False
    # Timed side vetos
    long_veto_until: Optional[datetime] = None
    short_veto_until: Optional[datetime] = None
    # Stop:take pause
    pause_all_until: Optional[datetime] = None
    # Latest trend-day signal from the bot's main loop (fed via notify_trend_day)
    trend_day_tier: int = 0
    trend_day_dir: Optional[str] = None


class LossFactorGuard:
    def __init__(self) -> None:
        self.enabled = True
        self.state = DailyState()

    def notify_new_day(self, day) -> None:
        if isinstance(day, datetime):
            day = day.date()
        if self.state.day == day:
            return
        # Preserve externally-managed trend_day state across the reset —
        # the main loop feeds it in independently of new-day events.
        preserved_tier = self.state.trend_day_tier
        preserved_dir = self.state.trend_day_dir
        self.state = DailyState(day=day)
        self.state.trend_day_tier = preserved_tier
        self.state.trend_day_dir = preserved_dir
        logging.info("LossFactorGuard: new day %s — state reset", day)

    def notify_trade_closed(self, trade: dict) -> None:
        """Call after each trade closes. trade dict must have:
        side ("LONG"/"SHORT"), source (exit type), pnl_dollars, exit_time (tz aware dt or iso str), entry_time.
        """
        side = str(trade.get("side", "")).upper()
        source = str(trade.get("source", "")).lower()
        pnl = float(trade.get("pnl_dollars", 0.0) or 0.0)
        self.state.cum_pnl += pnl
        is_stop = source in ("stop", "stop_gap")
        is_take = source in ("take", "take_gap")

        # Streak tracking per side
        if side == "LONG":
            if is_stop:
                self.state.long_stop_streak += 1
            elif pnl > 0:
                self.state.long_stop_streak = 0
        elif side == "SHORT":
            if is_stop:
                self.state.short_stop_streak += 1
            elif pnl > 0:
                self.state.short_stop_streak = 0

        # Morning stop-cascade counter (hours 8-10 ET)
        try:
            entry_hour = _entry_hour_et(trade)
        except Exception:
            entry_hour = None
        if entry_hour is not None and 8 <= entry_hour <= 10:
            self.state.am_trades_done += 1
            if is_stop:
                self.state.am_stops += 1

        self.state.recent_sources.append(source)

        # Trip conditions
        exit_time = _exit_dt_et(trade)
        now_et = exit_time if exit_time is not None else None

        # Long-streak veto
        if self.state.long_stop_streak >= LONG_STREAK and now_et is not None:
            self.state.long_veto_until = now_et + timedelta(minutes=VETO_MINUTES)
            logging.info(
                "LossFactorGuard TRIP long_streak: %d consec LONG stops -> veto LONGs until %s",
                self.state.long_stop_streak, self.state.long_veto_until,
            )
            self.state.long_stop_streak = 0
        if self.state.short_stop_streak >= SHORT_STREAK and now_et is not None:
            self.state.short_veto_until = now_et + timedelta(minutes=VETO_MINUTES)
            logging.info(
                "LossFactorGuard TRIP short_streak: %d consec SHORT stops -> veto SHORTs until %s",
                self.state.short_stop_streak, self.state.short_veto_until,
            )
            self.state.short_stop_streak = 0

        # Morning cascade: first N AM trades all stops
        if (
            not self.state.morning_cascade_active
            and self.state.am_trades_done >= MORNING_CASCADE_N
            and self.state.am_stops >= MORNING_CASCADE_N
        ):
            self.state.morning_cascade_active = True
            logging.info(
                "LossFactorGuard TRIP morning_cascade: first %d AM trades all stopped -> veto until 11:00 ET",
                MORNING_CASCADE_N,
            )

        # Stop:take ratio pause
        if len(self.state.recent_sources) == ROLLING_WINDOW and now_et is not None:
            stops = sum(1 for s in self.state.recent_sources if s in ("stop", "stop_gap"))
            takes = sum(1 for s in self.state.recent_sources if s in ("take", "take_gap"))
            if takes == 0 and stops >= 4:
                self.state.pause_all_until = now_et + timedelta(minutes=VETO_MINUTES)
                logging.info(
                    "LossFactorGuard TRIP stop_take_ratio: %d stops / %d takes in last %d -> pause all until %s",
                    stops, takes, ROLLING_WINDOW, self.state.pause_all_until,
                )
            elif takes > 0 and (stops / takes) >= STOP_TAKE_RATIO_LIMIT:
                self.state.pause_all_until = now_et + timedelta(minutes=VETO_MINUTES)
                logging.info(
                    "LossFactorGuard TRIP stop_take_ratio: %d stops / %d takes = %.1fx -> pause all until %s",
                    stops, takes, stops / takes, self.state.pause_all_until,
                )

    def should_veto_entry(self, signal: dict, current_time_et: Optional[datetime] = None) -> Tuple[bool, str]:
        """Return (veto, reason). signal dict expected to have 'side'."""
        if current_time_et is None:
            return False, ""
        # New-day reset
        self.notify_new_day(current_time_et.date())
        side = str(signal.get("side", "")).upper()
        hour = current_time_et.hour

        # Morning cascade (active until 11:00 ET)
        if self.state.morning_cascade_active and hour < 11:
            return True, f"morning_cascade (am_stops={self.state.am_stops}/{self.state.am_trades_done})"

        # Afternoon shutdown: 15:00 ET onwards if cum PnL <= threshold
        if hour >= 15 and self.state.cum_pnl <= AFT_SHUTDOWN_PNL:
            if not self.state.afternoon_shutdown:
                self.state.afternoon_shutdown = True
                logging.info(
                    "LossFactorGuard TRIP afternoon_shutdown: cum_pnl=$%.0f <= $%.0f at %s -> block new entries",
                    self.state.cum_pnl, AFT_SHUTDOWN_PNL, current_time_et,
                )
            return True, f"afternoon_shutdown (cum_pnl=${self.state.cum_pnl:.0f})"

        # Full pause
        if self.state.pause_all_until is not None and current_time_et < self.state.pause_all_until:
            return True, f"pause_all (until {self.state.pause_all_until.strftime('%H:%M')})"

        # Side vetos
        if side == "LONG" and self.state.long_veto_until is not None and current_time_et < self.state.long_veto_until:
            return True, f"long_bias_veto (until {self.state.long_veto_until.strftime('%H:%M')})"
        if side == "SHORT" and self.state.short_veto_until is not None and current_time_et < self.state.short_veto_until:
            return True, f"short_bias_veto (until {self.state.short_veto_until.strftime('%H:%M')})"

        # Filter C: counter-trend reversal veto. If we're in a confirmed trend
        # day AND the signal is a _Rev_ sub-strategy trying to fade the trend,
        # block it. Momentum signals (_Mom_) aligned with the trend pass.
        if self.state.trend_day_tier >= TREND_BIAS_MIN_TIER and self.state.trend_day_dir:
            sub = str(signal.get("sub_strategy", "") or signal.get("combo_key", "") or "")
            is_reversal = "_Rev_" in sub
            trend = self.state.trend_day_dir.lower()
            if is_reversal:
                fades_trend = (side == "LONG" and trend == "down") or (side == "SHORT" and trend == "up")
                if fades_trend:
                    return True, (
                        f"trend_bias_reversal_veto ({side} Rev vs trend_day={trend} "
                        f"tier={self.state.trend_day_tier})"
                    )

        return False, ""


def _entry_hour_et(trade: dict) -> Optional[int]:
    t = trade.get("entry_time") or trade.get("entry_ts")
    return _hour_from(t)


def _exit_dt_et(trade: dict) -> Optional[datetime]:
    t = trade.get("exit_time") or trade.get("exit_ts") or trade.get("close_time")
    return _dt_et_from(t)


def _dt_et_from(t):
    if t is None:
        return None
    if isinstance(t, datetime):
        dt = t
    else:
        try:
            dt = datetime.fromisoformat(str(t).replace("Z", "+00:00"))
        except Exception:
            return None
    try:
        from zoneinfo import ZoneInfo
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ZoneInfo("America/New_York"))
        return dt.astimezone(ZoneInfo("America/New_York"))
    except Exception:
        return dt


def _hour_from(t) -> Optional[int]:
    dt = _dt_et_from(t)
    return dt.hour if dt is not None else None


_GUARD: Optional[LossFactorGuard] = None


def init_guard() -> Optional[LossFactorGuard]:
    global _GUARD
    if os.environ.get("JULIE_LOSS_FACTOR_GUARD", "0").strip() != "1":
        _GUARD = None
        return None
    _GUARD = LossFactorGuard()
    logging.info(
        "LossFactorGuard enabled: long_streak=%d short_streak=%d morning=%d aft_pnl=$%.0f stop_take=%.1fx veto_min=%d",
        LONG_STREAK, SHORT_STREAK, MORNING_CASCADE_N, AFT_SHUTDOWN_PNL, STOP_TAKE_RATIO_LIMIT, VETO_MINUTES,
    )
    return _GUARD


def get_guard() -> Optional[LossFactorGuard]:
    return _GUARD


def notify_trend_day(tier: int, direction) -> None:
    """Called from the bot's main loop once per bar. Feeds trend-day state
    into the guard so the counter-trend reversal filter can act on it."""
    g = get_guard()
    if g is None:
        return
    try:
        g.state.trend_day_tier = int(tier or 0)
        g.state.trend_day_dir = str(direction).lower() if direction else None
    except Exception:
        pass


def should_veto_entry(signal: dict, current_time_et) -> Tuple[bool, str]:
    if _GUARD is None:
        return False, ""
    try:
        return _GUARD.should_veto_entry(signal, current_time_et)
    except Exception:
        logging.debug("LossFactorGuard veto check failed", exc_info=True)
        return False, ""


def notify_trade_closed(trade: dict) -> None:
    if _GUARD is None:
        return
    try:
        _GUARD.notify_trade_closed(trade)
    except Exception:
        logging.debug("LossFactorGuard close notify failed", exc_info=True)


def notify_new_day(day) -> None:
    if _GUARD is None:
        return
    try:
        _GUARD.notify_new_day(day)
    except Exception:
        logging.debug("LossFactorGuard new-day notify failed", exc_info=True)
