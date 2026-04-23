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
import sqlite3
import statistics
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from . import cell_key, cell_key_parts


MIN_SAMPLES = 20
"""Minimum fired-trade count in a cell for it to be scored. Below
this, the cell is tagged `unrated` and receives default medal
effects (neutral priority / size)."""


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
) -> LeagueScore:
    """Compute the three league metrics for one cell.

    Reads from the `signals` + `outcomes` tables, joined. Counterfactual
    outcomes (from blocked signals resolved via forward-walk) are
    excluded by default so the cash league reflects only live-realized
    PnL. Set include_counterfactual=True to fold them in.
    """
    ck = cell_key(strategy, regime, time_bucket)

    # Fetch all signals + outcomes for the cell. `outcomes` joined so
    # blocked signals (no outcome) drop out of the live fired-metric
    # computation but still contribute to the blocked-count.
    rows = list(
        conn.execute(
            """
            SELECT
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

    # PURITY — win rate
    wins = [r for r in fired_rows if (r["pnl_dollars"] or 0) > 0]
    purity = len(wins) / len(fired_rows)

    # CASH — per-contract normalized avg PnL
    total_pnl = sum((r["pnl_dollars"] or 0.0) for r in fired_rows)
    total_size = sum(max(1, r["size"] or 1) for r in fired_rows)
    cash = total_pnl / max(1, total_size)

    # VELOCITY — median 1/bars_held over winners (winners only so losers
    # hitting SL on bar 1 don't dominate).
    win_bars = [
        int(r["bars_held"]) for r in wins
        if r["bars_held"] is not None and int(r["bars_held"]) > 0
    ]
    if not win_bars:
        velocity = None
    else:
        velocity = 1.0 / float(statistics.median(win_bars))

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
) -> list[LeagueScore]:
    """Score every (strategy, regime, time_bucket) triple present in
    the signals table."""
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
