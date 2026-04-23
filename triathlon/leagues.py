"""League scoring — compute Purity / Cash / Velocity per cell.

A "cell" is (strategy, regime, time_bucket). For each cell we pull
its signals + outcomes from the ledger and compute three metrics
(all higher = better):

    purity   = win_rate over fired trades
    cash     = avg pnl_dollars per fired trade (per-contract normalized)
    velocity = median-of-winners (1 / bars_held), computed only for wins

Cells with fewer than MIN_SAMPLES fired trades are scored as None
for all three metrics and receive the `unrated` medal. Cells with
enough samples are ranked across peer cells for each league, and the
percentile across leagues drives medal assignment (see medals.py).

A few implementation notes:

  * Per-contract normalization: cash = sum(pnl_dollars) / sum(size).
    Avoids penalizing cells that happen to trade smaller contract sizes
    during dead_tape days when size=1 is forced.
  * Counterfactual signals (blocked trades that were scored via
    forward-walk through the price parquet) CAN be included in the
    cash metric by setting include_counterfactual=True on the caller
    side — default is False so the live PnL column reflects only real
    trades, but the validator can enable it to see what blocks cost.
  * Velocity uses bars_held of winners only; losers always hit SL fast
    so including them biases every cell toward high velocity.
"""
from __future__ import annotations

import logging
import math
import os
import sqlite3
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from . import cell_key, cell_key_parts


MIN_SAMPLES = 20
"""Minimum fired-trade count in a cell for it to be scored. Below
this, the cell is tagged `unrated` and receives default medal
effects (neutral priority / size)."""


# Time-decay scoring (2026-04-23). When a half-life is set, per-trade
# weights decay exponentially with age relative to a reference timestamp:
#
#     weight(trade_ts, reference) = exp(-ln(2) * age_days / half_life_days)
#
# A trade exactly `half_life_days` old contributes 50% as much as a
# trade right at the reference point. Default half-life (set by the
# env var below) is 60 days — roughly a quarter of a calendar year —
# which weights recent behavior more without discarding older data
# entirely.
#
# Disable by setting env JULIE_TRIATHLON_HALFLIFE_DAYS=0 (treats every
# trade as weight 1, identical to the pre-2026-04-23 behavior).
DEFAULT_HALFLIFE_DAYS = float(os.environ.get("JULIE_TRIATHLON_HALFLIFE_DAYS", "60"))


def _trade_weight(
    trade_ts: datetime,
    reference_ts: Optional[datetime],
    half_life_days: float,
) -> float:
    """Exponential time-decay weight. Returns 1.0 when time-decay is off
    (half_life_days <= 0) or reference_ts is missing."""
    if half_life_days <= 0 or reference_ts is None or trade_ts is None:
        return 1.0
    try:
        # Normalize tz: if only one side is tz-aware, strip tz from both
        # for the diff. We just need a relative-days delta, not absolute.
        a = trade_ts
        b = reference_ts
        if a.tzinfo is not None and b.tzinfo is None:
            a = a.replace(tzinfo=None)
        elif b.tzinfo is not None and a.tzinfo is None:
            b = b.replace(tzinfo=None)
        age_days = max(0.0, (b - a).total_seconds() / 86400.0)
    except Exception:
        return 1.0
    return math.exp(-math.log(2) * age_days / half_life_days)


@dataclass
class LeagueScore:
    """All three league scores for one cell, plus sample counts."""
    cell_key: str
    n_signals: int
    n_fired: int
    n_blocked: int
    purity: Optional[float]
    cash: Optional[float]
    velocity: Optional[float]

    def is_rated(self) -> bool:
        return all(v is not None for v in (self.purity, self.cash, self.velocity))


def score_cell(
    conn: sqlite3.Connection,
    strategy: str,
    regime: str,
    time_bucket: str,
    *,
    include_counterfactual: bool = False,
    min_samples: int = MIN_SAMPLES,
    half_life_days: float = DEFAULT_HALFLIFE_DAYS,
    reference_ts: Optional[datetime] = None,
) -> LeagueScore:
    """Compute the three league metrics for one cell.

    Reads from the `signals` + `outcomes` tables, joined.

    Time-decay (added 2026-04-23): when `half_life_days > 0`, every
    trade is weighted by an exponential decay based on how many days
    it sits before `reference_ts` (default: now). A half-life of 60
    days means a trade 60 days old counts half as much as a trade at
    the reference point. Set `half_life_days=0` to restore unweighted
    behavior (every trade weight=1; matches pre-2026-04-23 scoring).

    Counterfactual outcomes (from blocked signals resolved via
    forward-walk) are excluded by default so the cash metric reflects
    only live-realized PnL. Set include_counterfactual=True to fold
    them in.
    """
    ck = cell_key(strategy, regime, time_bucket)

    # Fetch all signals + outcomes for the cell. `outcomes` joined so
    # blocked signals (no outcome) drop out of the live fired-metric
    # computation but still contribute to the blocked-count.
    rows = list(
        conn.execute(
            """
            SELECT
                s.ts,
                s.status,
                s.size,
                o.pnl_dollars,
                o.bars_held,
                o.counterfactual
            FROM signals s
            LEFT JOIN outcomes o ON o.signal_id = s.signal_id
            WHERE s.strategy = ? AND s.regime = ? AND s.time_bucket = ?
            """,
            (strategy, regime, time_bucket),
        )
    )
    n_signals = len(rows)
    n_fired = sum(1 for r in rows if r["status"] == "fired")
    n_blocked = sum(
        1 for r in rows
        if r["status"] in ("blocked", "counterfactual_pending", "counterfactual_resolved")
    )

    # Keep only rows with realized outcomes (real trades by default;
    # also counterfactuals if requested)
    fired_rows = [
        r for r in rows
        if r["pnl_dollars"] is not None
        and (include_counterfactual or (r["counterfactual"] or 0) == 0)
    ]
    if len(fired_rows) < min_samples:
        return LeagueScore(
            cell_key=ck,
            n_signals=n_signals, n_fired=n_fired, n_blocked=n_blocked,
            purity=None, cash=None, velocity=None,
        )

    # Default reference = now (in the trades' timezone is fine since
    # _trade_weight normalizes tz before computing the age delta)
    if reference_ts is None and half_life_days > 0:
        reference_ts = datetime.now(timezone.utc)

    # Per-trade weights — all 1.0 when time-decay is off
    weights: list[float] = []
    for r in fired_rows:
        try:
            ts = datetime.fromisoformat(r["ts"])
        except Exception:
            ts = None
        weights.append(_trade_weight(ts, reference_ts, half_life_days))

    # Effective sample size = sum(weights). When time-decay is off it
    # equals len(fired_rows); when on, stale cells drop below
    # min_samples here and are treated as unrated.
    effective_n = sum(weights)
    if effective_n < min_samples:
        return LeagueScore(
            cell_key=ck,
            n_signals=n_signals, n_fired=n_fired, n_blocked=n_blocked,
            purity=None, cash=None, velocity=None,
        )

    # PURITY — weighted win rate
    sum_w = sum(weights) or 1.0
    sum_w_wins = sum(w for w, r in zip(weights, fired_rows)
                      if (r["pnl_dollars"] or 0) > 0)
    purity = sum_w_wins / sum_w

    # CASH — weighted per-contract avg PnL
    weighted_pnl = sum(w * (r["pnl_dollars"] or 0.0)
                        for w, r in zip(weights, fired_rows))
    weighted_size = sum(w * max(1, r["size"] or 1)
                         for w, r in zip(weights, fired_rows))
    cash = weighted_pnl / max(1.0, weighted_size)

    # VELOCITY — on winners only so losers hitting SL on bar 1 don't
    # bias. When time-decay is off we preserve exact pre-2026-04-23
    # behavior (median of raw bars_held). When on we use the weighted
    # mean of 1/bars_held — same intent, smoother under weights.
    win_entries = [
        (w, int(r["bars_held"]))
        for w, r in zip(weights, fired_rows)
        if (r["pnl_dollars"] or 0) > 0
        and r["bars_held"] is not None and int(r["bars_held"]) > 0
    ]
    if not win_entries:
        velocity = None
    elif half_life_days > 0:
        sum_ww = sum(w for w, _ in win_entries) or 1.0
        weighted_inv = sum(w * (1.0 / b) for w, b in win_entries)
        velocity = weighted_inv / sum_ww
    else:
        velocity = 1.0 / float(statistics.median([b for _, b in win_entries]))

    return LeagueScore(
        cell_key=ck,
        n_signals=n_signals, n_fired=n_fired, n_blocked=n_blocked,
        purity=round(purity, 4),
        cash=round(cash, 2),
        velocity=round(velocity, 4) if velocity is not None else None,
    )


def score_all_cells(
    conn: sqlite3.Connection,
    *,
    include_counterfactual: bool = False,
    min_samples: int = MIN_SAMPLES,
    half_life_days: float = DEFAULT_HALFLIFE_DAYS,
    reference_ts: Optional[datetime] = None,
) -> list[LeagueScore]:
    """Score every (strategy, regime, time_bucket) triple present in
    the signals table. See `score_cell` for the time-decay kwargs."""
    triples = [
        (r["strategy"], r["regime"], r["time_bucket"])
        for r in conn.execute(
            "SELECT DISTINCT strategy, regime, time_bucket FROM signals"
        )
    ]
    return [
        score_cell(
            conn, s, r, t,
            include_counterfactual=include_counterfactual,
            min_samples=min_samples,
            half_life_days=half_life_days,
            reference_ts=reference_ts,
        )
        for s, r, t in triples
    ]


def rank_scores(
    scores: list[LeagueScore],
) -> dict[str, dict[str, Optional[int]]]:
    """Given the full list of cell scores, compute per-cell ranks in
    each league. Ranks are 1-indexed; 1 = best (highest metric).
    Unrated cells get rank=None in all three leagues.

    Returns {cell_key: {purity_rank, cash_rank, velocity_rank}}.
    """
    rated = [s for s in scores if s.is_rated()]
    # Sort descending in each league; ties broken by cell_key for determinism
    purity_sorted = sorted(rated, key=lambda s: (-(s.purity or 0), s.cell_key))
    cash_sorted = sorted(rated, key=lambda s: (-(s.cash or 0), s.cell_key))
    velocity_sorted = sorted(rated, key=lambda s: (-(s.velocity or 0), s.cell_key))

    ranks: dict[str, dict[str, Optional[int]]] = {}
    for idx, s in enumerate(purity_sorted):
        ranks.setdefault(s.cell_key, {})["purity_rank"] = idx + 1
    for idx, s in enumerate(cash_sorted):
        ranks.setdefault(s.cell_key, {})["cash_rank"] = idx + 1
    for idx, s in enumerate(velocity_sorted):
        ranks.setdefault(s.cell_key, {})["velocity_rank"] = idx + 1
    # Fill in unrated cells
    for s in scores:
        if not s.is_rated():
            ranks.setdefault(s.cell_key, {})
            ranks[s.cell_key].setdefault("purity_rank", None)
            ranks[s.cell_key].setdefault("cash_rank", None)
            ranks[s.cell_key].setdefault("velocity_rank", None)
    return ranks


__all__ = [
    "LeagueScore",
    "MIN_SAMPLES",
    "score_cell",
    "score_all_cells",
    "rank_scores",
]
