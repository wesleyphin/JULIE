"""Triathlon Engine — per-cell signal tracking with league rankings and medal-driven live feedback.

The engine assigns every signal a "cell" defined by (strategy × regime ×
time-bucket) and scores each cell across three leagues:

    Purity   — win rate on fired trades (hit-rate league)
    Cash     — avg realized $ per fired trade (edge league)
    Velocity — 1 / avg-bars-held for winners (speed league)

Each cell earns a medal (gold / silver / bronze / probation / unrated)
based on its rank across the three leagues. Medals feed back into the
live entry path via size multipliers and priority nudges.

Modules:
    ledger      — sqlite schema + signal/block/outcome recording
    leagues     — per-league metric computation + ranking
    medals      — medal assignment from league ranks + runtime lookup
    counterfactual — forward-walk bar data to score blocked signals
    seed        — bootstrap standings from historical closed_trades
    retrain_hook — purity-drop detection + retrain queue

The runtime hook used by julie001.py lives in
`tools/triathlon_runtime.py` (a thin adapter so the bot doesn't
directly depend on sqlite at import time).
"""
from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
DATA_DIR = REPO_ROOT / "ai_loop_data" / "triathlon"
LEDGER_PATH = DATA_DIR / "ledger.db"

# Time-bucket boundaries in NY hours (24h clock)
TIME_BUCKETS = [
    ("pre_open",    4.0,  9.5),
    ("morning",     9.5, 12.0),
    ("lunch",      12.0, 14.0),
    ("afternoon",  14.0, 16.0),
    ("post_close", 16.0, 17.0),
    ("overnight",  17.0, 28.0),   # wraps through midnight
]

STRATEGIES = ("DynamicEngine3", "AetherFlow", "RegimeAdaptive", "MLPhysics")
REGIMES = ("whipsaw", "calm_trend", "dead_tape", "neutral", "warmup")


def time_bucket_of(hour_24: float) -> str:
    """Map a 24-hour ET hour-of-day (fractional) to a time-bucket name.

    Buckets are closed-open [lo, hi). The overnight bucket includes
    everything from 17:00 to 04:00 next-day (implemented via wrap using
    hour + 24 when hour < 4).
    """
    h = float(hour_24)
    if h < 4.0:
        h += 24.0
    for name, lo, hi in TIME_BUCKETS:
        if lo <= h < hi:
            return name
    return "overnight"


def cell_key(strategy: str, regime: str, time_bucket: str) -> str:
    """Canonical cell-key encoding: `{strategy}|{regime}|{time_bucket}`."""
    return f"{strategy}|{regime}|{time_bucket}"


def cell_key_parts(cell_key_str: str) -> tuple[str, str, str]:
    """Inverse of cell_key(); returns (strategy, regime, time_bucket)."""
    parts = cell_key_str.split("|")
    if len(parts) != 3:
        raise ValueError(f"invalid cell_key: {cell_key_str!r}")
    return parts[0], parts[1], parts[2]
