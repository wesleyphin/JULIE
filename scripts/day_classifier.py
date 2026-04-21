#!/usr/bin/env python3
"""Day-category classifier that predicts at ~10:30 ET which of:

  breakout    (strong early directional move)
  chop        (big range, low drift)
  flat_calm   (tight range)
  large_trend (moderate emerging drift, might fade)
  moderate    (ordinary tape)

Maps each predicted category to the simulator variant that wins on it:

  breakout    -> V0 (no D cap; ride size-3 through the move)
  chop        -> V1 (cap all; chop kills size-3)
  flat_calm   -> V1 (noise dampener)
  large_trend -> V1 (cap Rev; fades get crushed on continued drift)
  moderate    -> V3 (skip Rev cap on calm_trend; Rev catches pullbacks)

Rules are hand-tuned on 2025 136-day data.  See classify_day_intraday() below.

Usage:
    from day_classifier import classify_day_intraday, variant_for_category
"""
from __future__ import annotations


# Grid-search tuned on 2025 177-day set (scripts/train_day_classifier.py)
# Variant-accuracy 74.6%, full-category accuracy 67.2%.
RANGE_THRESH_BREAKOUT  = 1.20  # % of day open
DRIFT_THRESH_BREAKOUT  = 0.30
RANGE_THRESH_CHOP      = 0.40
DRIFT_THRESH_CHOP_MAX  = 0.20
RANGE_THRESH_FLAT_CALM = 0.10
DRIFT_THRESH_LARGE     = 0.45


def classify_day_intraday(
    range_pct_so_far: float,
    drift_pct_so_far: float,
    eff_so_far: float = 0.0,
    minutes_elapsed: int = 60,
) -> str:
    """Predict full-day category at time-of-call using intraday features.

    range_pct_so_far  — (high_so_far - low_so_far) / open  * 100
    drift_pct_so_far  — (current - open) / open * 100  (signed)
    eff_so_far        — |drift| / range  (0..1)
    minutes_elapsed   — minutes from 09:30 ET (so classifier can scale
                        thresholds by how much of the day has happened)

    All thresholds scale linearly with time-elapsed (because a 0.8% range
    by 10:30 isn't the same as by 15:45).  We normalise to 60-minute
    equivalents so a day that has moved 0.5% by 30 min projects to ~1%.
    """
    if minutes_elapsed <= 0:
        return "moderate"

    # Normalise to 60-minute projection
    scale = 60.0 / float(minutes_elapsed)
    rng = abs(range_pct_so_far) * scale
    drift_signed = drift_pct_so_far * scale
    drift = abs(drift_signed)

    # Rule order matters — check strongest signal first.
    if drift >= DRIFT_THRESH_BREAKOUT and rng >= RANGE_THRESH_BREAKOUT:
        return "breakout"

    if rng >= RANGE_THRESH_CHOP and drift <= DRIFT_THRESH_CHOP_MAX:
        return "chop"

    if rng < RANGE_THRESH_FLAT_CALM:
        return "flat_calm"

    if drift >= DRIFT_THRESH_LARGE and rng < RANGE_THRESH_BREAKOUT:
        return "large_trend"

    return "moderate"


# Per-category winning variant (from sim_subcategorize_days.json analysis)
VARIANT_FOR_CATEGORY = {
    "breakout":    "V0",
    "chop":        "V1",
    "flat_calm":   "V1",
    "large_trend": "V1",
    "moderate":    "V3",
}


def variant_for_category(category: str) -> str:
    return VARIANT_FOR_CATEGORY.get(category, "V1")  # V1 is safe default
