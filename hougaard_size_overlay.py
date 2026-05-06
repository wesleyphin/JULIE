"""Hougaard size overlay — runtime engine for per-trade size boost.

What it does
------------
When a StdevMl trade signal fires at full confidence (prob >= 0.75) and the
proposed direction aligns with today's Hougaard session bias, the position
size is boosted by 1.20x. Otherwise, size is unchanged.

What it does NOT do
-------------------
- Does not change the firing threshold (no gate-opening — that hypothesis was
  falsified, see artifacts/hougaard_overlay/UPDATE_2022_2026.md)
- Does not apply to non-StdevMl strategies (no evidence for those)
- Does not boost more than 1.20x (extrapolation beyond what was tested)
- Does not change SL/TP — only size

Why 1.20x and not regime-conditional
------------------------------------
Backtest evidence (3 windows: VAL 2021-2025, PURE_OOS 2022-2024, OOS 2026):
    bear_aligned_120: +0.1% PnL  (n=88/16/14 boosted — sample too small)
    bull_aligned_120: +3.0% PnL  (n=1060/649/163 — was the negative control)
    any_aligned_120:  +3.1% PnL  (winner, +3% across all 3 windows)
The bear-regime amplification was real in PF terms but economically
irrelevant. The any-aligned 1.2x boost replicated cleanly across all three
independent windows. See size_overlay_summary.txt for full data.

Live API
--------
    from hougaard_size_overlay import HougaardSizeOverlay
    overlay = HougaardSizeOverlay()           # init once at bot startup
    overlay.refresh_if_stale()                # call once per bar (cheap)
    mult = overlay.get_size_multiplier(signal)  # 1.0 or 1.2

Side-effects on signal dict (when overlay would boost):
    signal['hougaard_boosted']         = True
    signal['hougaard_size_multiplier'] = 1.2
    signal['hougaard_bias_dir']        = -1 / 0 / +1
    signal['hougaard_active_scenarios']= "B" | "C+" | "C-" | "B+C+" | ""
    signal['hougaard_session_date']    = "YYYY-MM-DD"

Telemetry log line emitted on boost decision (info level):
    "[hougaard-size] strat=StdevMlStrategy side=LONG conf=0.81 bias_dir=+1 "
    "scenarios=C+ aligned=True boost=1.20x size 1→1"
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

# Reuse the offline batch engine — same trigger logic, just on demand
try:
    from tools.hougaard_context_offline import build_context_table
    _ENGINE_AVAILABLE = True
except Exception:  # pragma: no cover - graceful degradation
    _ENGINE_AVAILABLE = False
    build_context_table = None  # type: ignore


# ─── config ─────────────────────────────────────────────────────────────────

# What strategies this overlay applies to. Only StdevMl has backtest evidence.
SUPPORTED_STRATEGIES = {"StdevMlStrategy"}

# Confidence floor — only boost when ML model is fully confident
CONFIDENCE_FLOOR = 0.75

# Size boost multiplier (do not extrapolate beyond 1.20x without re-validating)
SIZE_BOOST = 1.20

# How long the cached context stays warm before forced refresh (in seconds).
# We re-evaluate every minute anyway via refresh_if_stale, but a hard timeout
# guards against clock drift / day-rollover edge cases.
CACHE_TTL_SEC = 600  # 10 minutes

# Window of context to load — needs ~10 days of prior daily bars for the
# Scenario B/C lookbacks. We load 60 days to be safe.
LOOKBACK_DAYS = 60


# ─── overlay class ──────────────────────────────────────────────────────────

class HougaardSizeOverlay:
    """Runtime engine that emits a per-signal size multiplier."""

    def __init__(self) -> None:
        self._ctx_today: Optional[pd.Series] = None
        self._ctx_date: Optional[date] = None
        self._last_refresh: Optional[datetime] = None
        self._available: bool = _ENGINE_AVAILABLE
        if not self._available:
            logging.warning(
                "[hougaard-size] context engine unavailable; overlay will be no-op."
            )

    # ─── context refresh ────────────────────────────────────────────────────

    def refresh_if_stale(self, now: Optional[datetime] = None) -> None:
        """Cheap idempotent refresh. Safe to call every bar."""
        if not self._available:
            return
        now = now or datetime.now()
        today = now.date()
        # Skip if we already have today's context AND last refresh is within TTL
        if (self._ctx_date == today
                and self._last_refresh is not None
                and (now - self._last_refresh).total_seconds() < CACHE_TTL_SEC):
            return
        try:
            start = (now - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
            end = today.strftime("%Y-%m-%d")
            ctx = build_context_table(start, end)
            # ctx is indexed by tz-naive midnight datetimes; pull today's row
            today_ts = pd.Timestamp(today)
            if today_ts in ctx.index:
                self._ctx_today = ctx.loc[today_ts]
            else:
                # No daily bar yet for today (pre-RTH); use most recent prior
                if len(ctx):
                    self._ctx_today = ctx.iloc[-1]
                    logging.debug(
                        "[hougaard-size] today (%s) not in context; using most "
                        "recent (%s)", today, ctx.index[-1].date(),
                    )
                else:
                    self._ctx_today = None
            self._ctx_date = today
            self._last_refresh = now
            if self._ctx_today is not None:
                logging.info(
                    "[hougaard-size] context refreshed for %s: bias_dir=%+d "
                    "strength=%.2f scenarios=%r bull=%s",
                    today,
                    int(self._ctx_today.get("bias_direction", 0)),
                    float(self._ctx_today.get("bias_strength", 0.0)),
                    self._ctx_today.get("active_scenarios", ""),
                    bool(self._ctx_today.get("bull_regime", True)),
                )
        except Exception as exc:
            logging.warning("[hougaard-size] context refresh failed: %s", exc)

    # ─── multiplier decision ────────────────────────────────────────────────

    def get_size_multiplier(self, signal: dict) -> float:
        """Decide whether to boost this signal's size. Returns 1.0 or SIZE_BOOST.

        Side-effects: stamps signal dict with hougaard_* metadata fields when
        the boost would apply, OR with hougaard_boosted=False otherwise (for
        consistent journal capture).
        """
        # Always set defaults so the journal sees consistent fields
        signal.setdefault("hougaard_boosted", False)
        signal.setdefault("hougaard_size_multiplier", 1.0)
        signal.setdefault("hougaard_bias_dir", 0)
        signal.setdefault("hougaard_active_scenarios", "")
        signal.setdefault("hougaard_session_date",
                          self._ctx_date.strftime("%Y-%m-%d") if self._ctx_date else "")

        if not self._available or self._ctx_today is None:
            return 1.0

        # Only StdevMl has backtest evidence
        strategy = str(signal.get("strategy", ""))
        if strategy not in SUPPORTED_STRATEGIES:
            return 1.0

        # Need a confidence value
        try:
            conf = float(signal.get("confidence", 0.0))
        except (TypeError, ValueError):
            return 1.0
        if conf < CONFIDENCE_FLOOR:
            return 1.0

        # Need a side
        side = str(signal.get("side", "")).upper()
        if side not in ("LONG", "SHORT"):
            return 1.0
        signal_dir = +1 if side == "LONG" else -1

        # Read context
        bias_dir = int(self._ctx_today.get("bias_direction", 0))
        scenarios = str(self._ctx_today.get("active_scenarios", ""))

        # Stamp metadata regardless of boost outcome (for forensic journaling)
        signal["hougaard_bias_dir"] = bias_dir
        signal["hougaard_active_scenarios"] = scenarios

        if bias_dir == 0 or signal_dir != bias_dir:
            return 1.0

        # All conditions met — boost
        signal["hougaard_boosted"] = True
        signal["hougaard_size_multiplier"] = SIZE_BOOST

        logging.info(
            "[hougaard-size] strat=%s side=%s conf=%.2f bias_dir=%+d "
            "scenarios=%r aligned=True boost=%.2fx",
            strategy, side, conf, bias_dir, scenarios, SIZE_BOOST,
        )
        return SIZE_BOOST

    # ─── helper: apply to signal dict in-place ─────────────────────────────

    def apply_to_signal(self, signal: dict) -> None:
        """Convenience: refresh context, compute multiplier, mutate signal[size].

        After this returns, signal['size'] reflects the boosted value (or
        unchanged), and the hougaard_* metadata fields are populated for
        journal capture.

        Integer-rounding caveat: at SIZE_BOOST=1.20x, base_size=1 rounds to 1
        (no actual boost). We log this explicitly so the journal can later
        identify "would-have-boosted" trades that were no-op'd by rounding.
        """
        self.refresh_if_stale()
        mult = self.get_size_multiplier(signal)
        signal["hougaard_boost_applied"] = False  # default; set True only if size changed
        if mult == 1.0:
            return
        original_size = int(signal.get("size", 1) or 1)
        boosted_size = max(1, int(round(original_size * mult)))
        if boosted_size == original_size:
            # Boost intent set, but integer rounding nullifies (e.g., size=1 * 1.2 = 1.2 → 1)
            logging.info(
                "[hougaard-size] would-boost suppressed by int rounding: "
                "size %d * %.2fx = %d (unchanged). Set hougaard_boost_applied=False.",
                original_size, mult, boosted_size,
            )
            signal["hougaard_size_original"] = original_size
            return
        signal["size"] = boosted_size
        signal["hougaard_size_original"] = original_size
        signal["hougaard_boost_applied"] = True
        logging.info(
            "[hougaard-size] applied: size %d -> %d (%.2fx)",
            original_size, boosted_size, mult,
        )


# ─── module-level singleton (for convenience in julie001.py) ───────────────

_GLOBAL_OVERLAY: Optional[HougaardSizeOverlay] = None


def get_overlay() -> HougaardSizeOverlay:
    """Return the process-wide overlay instance, instantiating on first call."""
    global _GLOBAL_OVERLAY
    if _GLOBAL_OVERLAY is None:
        _GLOBAL_OVERLAY = HougaardSizeOverlay()
    return _GLOBAL_OVERLAY


# ─── self-test ──────────────────────────────────────────────────────────────

def _self_test() -> None:
    """Manual smoke test: build context, run a few synthetic signals through."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    overlay = HougaardSizeOverlay()
    overlay.refresh_if_stale()
    print(f"[self-test] context loaded for {overlay._ctx_date}")
    if overlay._ctx_today is None:
        print("[self-test] no context — engine unavailable or no data?")
        return
    print(f"[self-test] today's bias: dir={int(overlay._ctx_today.get('bias_direction', 0))} "
          f"strength={float(overlay._ctx_today.get('bias_strength', 0.0)):.2f} "
          f"scenarios={overlay._ctx_today.get('active_scenarios', '')!r}")

    bias_dir = int(overlay._ctx_today.get("bias_direction", 0))
    aligned_side = "LONG" if bias_dir == +1 else ("SHORT" if bias_dir == -1 else "LONG")
    opposed_side = "SHORT" if aligned_side == "LONG" else "LONG"

    cases = [
        # (label, signal_dict, expected_mult)
        ("StdevMl aligned full-conf",
         {"strategy": "StdevMlStrategy", "side": aligned_side, "confidence": 0.81, "size": 1},
         1.2 if bias_dir != 0 else 1.0),
        ("StdevMl opposed full-conf",
         {"strategy": "StdevMlStrategy", "side": opposed_side, "confidence": 0.81, "size": 1},
         1.0),
        ("StdevMl aligned LOW-conf",
         {"strategy": "StdevMlStrategy", "side": aligned_side, "confidence": 0.65, "size": 1},
         1.0),
        ("DE3 aligned full-conf (other strat — no boost)",
         {"strategy": "DynamicEngine3", "side": aligned_side, "confidence": 0.85, "size": 5},
         1.0),
    ]
    for label, sig, expected in cases:
        mult = overlay.get_size_multiplier(sig)
        ok = "✓" if abs(mult - expected) < 1e-6 else "✗"
        print(f"  {ok} {label:<48s} mult={mult:.2f} (expected {expected:.2f})")


if __name__ == "__main__":
    _self_test()
