#!/usr/bin/env python3
"""Reconstruct regime timeline from bar data (for folders whose logs predate
the regime classifier).  Uses the same vol/eff thresholds as
regime_classifier.RegimeClassifier.
"""
from __future__ import annotations

import math
import re
from collections import deque
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

_NY = ZoneInfo("America/New_York")

# Thresholds copied from regime_classifier.py so this stays self-contained
# (no import of the live bot module).
WINDOW_BARS = 120
EFF_LOW = 0.05
EFF_HIGH = 0.12
VOL_HIGH = 3.5  # bp
TRANSITION_COOLDOWN_BARS = 30


def _classify(vol_bp: float, eff: float) -> str:
    if vol_bp > VOL_HIGH and eff < EFF_LOW:
        return "whipsaw"
    if eff > EFF_HIGH:
        return "calm_trend"
    return "neutral"


def _compute_metrics(closes: list[float]) -> tuple[float, float]:
    rets = []
    for i in range(1, len(closes)):
        p0 = closes[i - 1]
        if p0 > 0:
            rets.append((closes[i] - p0) / p0)
    if not rets:
        return 0.0, 0.0
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / max(1, len(rets) - 1)
    vol_bp = math.sqrt(var) * 10_000.0
    abs_sum = sum(abs(r) for r in rets)
    eff = abs(sum(rets)) / abs_sum if abs_sum > 0 else 0.0
    return vol_bp, eff


def reconstruct_from_bars(bars: list[tuple[datetime, float]]) -> list[tuple[datetime, str, float, float]]:
    """Run the classifier forward through the bar stream.  Returns list of
    (ts, regime, vol_bp, eff) for every regime transition, matching the
    format of load_regime_timeline."""
    closes: deque[float] = deque(maxlen=WINDOW_BARS)
    current_regime = "warmup"
    bar_count = 0
    last_transition = -10_000
    events: list[tuple[datetime, str, float, float]] = []
    for ts, price in bars:
        closes.append(price)
        bar_count += 1
        if len(closes) < WINDOW_BARS:
            continue
        vol_bp, eff = _compute_metrics(list(closes))
        new_regime = _classify(vol_bp, eff)
        if new_regime != current_regime:
            if current_regime != "warmup" and (bar_count - last_transition) < TRANSITION_COOLDOWN_BARS:
                continue
            events.append((ts, new_regime, vol_bp, eff))
            current_regime = new_regime
            last_transition = bar_count
    return events


RGX_BAR = re.compile(
    r"Bar: (?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) ET \| Price: (?P<price>[\d.]+)"
)


def load_bars_from_log(log_path: Path) -> list[tuple[datetime, float]]:
    """Load ALL bars (not just NY session) so the 120-bar window can warm up
    with overnight/London data too."""
    bars = []
    with log_path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = RGX_BAR.search(line)
            if not m:
                continue
            # Bar: lines are ET-tagged but printed naive — attach NY tz so
            # downstream comparisons vs tz-aware trade timestamps work.
            ts = datetime.strptime(m.group("ts"), "%Y-%m-%d %H:%M:%S").replace(tzinfo=_NY)
            bars.append((ts, float(m.group("price"))))
    bars.sort(key=lambda x: x[0])
    return bars


def reconstruct_from_log(log_path: Path) -> list[tuple[datetime, str, float, float]]:
    bars = load_bars_from_log(log_path)
    return reconstruct_from_bars(bars)
