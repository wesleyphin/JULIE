"""Medal assignment + live runtime effects.

Each rated cell receives a medal based on its best percentile across
the three leagues:

    gold       top 20% in at least one league           → priority +1,  size × 1.50
    silver     top 50% in at least one league           → priority +0,  size × 1.00
    bronze     top 80% in at least one league           → priority -1,  size × 0.75
    probation  bottom 20% in ALL leagues                → priority -2,  size × 0.50
    unrated    insufficient samples in cell             → priority +0,  size × 1.00

"Priority" is an integer added to the existing DE3 priority score used
by the rescue-queue logic in julie001.py (FAST=2, NORMAL=1, LOOSE=0
are the base values). A gold cell promotes LOOSE → NORMAL; a
probation cell demotes FAST → NORMAL.

"Size multiplier" is applied to the signal's `size` field before the
order is placed, rounded to the nearest integer with a floor of 1.

The "best percentile" rule lets a cell with one standout league
(e.g. a scalp strategy with velocity far ahead of peers) still get
credit even if its cash is middle-of-the-road. Probation requires
being bottom-tier across ALL THREE leagues — cells on probation are
genuinely broken across all dimensions.
"""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from . import cell_key_parts
from .leagues import LeagueScore, rank_scores, score_all_cells


# ─── medal cutoffs ─────────────────────────────────────────────
GOLD_PCTILE = 0.20      # top 20% of rated cells
SILVER_PCTILE = 0.50    # top 50%
BRONZE_PCTILE = 0.80    # top 80%
# Probation: bottom 20% in EVERY league


# ─── runtime effects ───────────────────────────────────────────
#
# Size multipliers were neutralized to 1.0 on 2026-04-23 after an
# out-of-sample validation (`scripts/backtest_triathlon_oos.py`)
# showed the train-period medal ranking did NOT reliably predict
# April-2026 per-cell edge:
#   - silver (+$4.29/tr) and bronze (+$5.12/tr) both beat
#     gold (+$4.17/tr) in the holdout
#   - the +$267 PnL lift on 644 trades was within sample noise
#   - DD got worse by $242
#   - Spearman (train_rank → April_value) was +0.27 to +0.54 —
#     weak to moderate, not strong enough for confident scaling
# Priority deltas are KEPT because (a) closed_trades-only backtest
# can't simulate rescue-queue priority and (b) priority deltas are
# smaller perturbations to live behavior than size multipliers.
# Flip size multipliers back on by editing this dict when a future
# OOS validation shows medal→edge transfer more clearly.
MEDAL_EFFECTS: dict[str, dict[str, float]] = {
    "gold":      {"priority_delta": +1, "size_mult": 1.00},
    "silver":    {"priority_delta":  0, "size_mult": 1.00},
    "bronze":    {"priority_delta": -1, "size_mult": 1.00},
    "probation": {"priority_delta": -2, "size_mult": 1.00},
    "unrated":   {"priority_delta":  0, "size_mult": 1.00},
}


def assign_medal(
    score: LeagueScore,
    ranks: dict[str, int],
    n_rated: int,
) -> str:
    """Given a cell's scores + ranks + total rated-cell count, return
    its medal tier.
    """
    if not score.is_rated():
        return "unrated"
    if n_rated <= 0:
        return "unrated"

    # Compute the cell's best percentile across the three leagues.
    # percentile = rank / n_rated, lower = better (rank 1 = top).
    pct_purity = (ranks.get("purity_rank") or n_rated) / n_rated
    pct_cash = (ranks.get("cash_rank") or n_rated) / n_rated
    pct_velocity = (ranks.get("velocity_rank") or n_rated) / n_rated
    best_pct = min(pct_purity, pct_cash, pct_velocity)
    # Probation requires the cell to be bottom-20% in EVERY league
    # simultaneously. Use min(): if even the BEST league still has
    # the cell in the bottom 20%, the cell is uniformly weak.
    PROBATION_PCTILE = 1.0 - 0.20  # 0.80

    if best_pct >= PROBATION_PCTILE:
        return "probation"
    if best_pct <= GOLD_PCTILE:
        return "gold"
    if best_pct <= SILVER_PCTILE:
        return "silver"
    if best_pct <= BRONZE_PCTILE:
        return "bronze"
    return "silver"  # middle of the pack — default to silver


def medal_effect(medal: str) -> dict[str, float]:
    """Return the runtime effects dict for a medal (default to
    unrated's neutral effects if the medal is unknown)."""
    return MEDAL_EFFECTS.get(medal, MEDAL_EFFECTS["unrated"])


# ─── rescoring ─────────────────────────────────────────────────
def rescore_standings(
    conn: sqlite3.Connection,
    *,
    include_counterfactual: bool = False,
    scored_at: Optional[datetime] = None,
) -> tuple[list[LeagueScore], dict[str, str]]:
    """Recompute every cell's scores, ranks, and medal; append a new
    row per cell into `standings` and replace the `current_medals`
    table with the fresh snapshot.

    Returns (scores, {cell_key: medal}).
    """
    if scored_at is None:
        scored_at = datetime.now(timezone.utc)
    scored_at_iso = scored_at.isoformat()

    scores = score_all_cells(conn, include_counterfactual=include_counterfactual)
    ranks = rank_scores(scores)
    n_rated = sum(1 for s in scores if s.is_rated())

    medals: dict[str, str] = {}
    conn.execute("DELETE FROM current_medals")
    for s in scores:
        r = ranks.get(s.cell_key, {})
        medal = assign_medal(s, r, n_rated)
        medals[s.cell_key] = medal
        conn.execute(
            """INSERT INTO standings
               (cell_key, scored_at, n_signals, n_fired, n_blocked,
                purity, cash, velocity,
                purity_rank, cash_rank, velocity_rank,
                medal)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                s.cell_key, scored_at_iso,
                s.n_signals, s.n_fired, s.n_blocked,
                s.purity, s.cash, s.velocity,
                r.get("purity_rank"), r.get("cash_rank"), r.get("velocity_rank"),
                medal,
            ),
        )
        conn.execute(
            """INSERT OR REPLACE INTO current_medals
               (cell_key, scored_at, medal, n_signals, purity, cash, velocity)
               VALUES (?,?,?,?,?,?,?)""",
            (
                s.cell_key, scored_at_iso, medal,
                s.n_signals, s.purity, s.cash, s.velocity,
            ),
        )

    return scores, medals


# ─── live lookup ───────────────────────────────────────────────
def lookup_medal(
    conn: sqlite3.Connection,
    strategy: str,
    regime: str,
    time_bucket: str,
) -> tuple[str, dict[str, float]]:
    """Fast point-lookup used by the live entry path. Returns
    (medal_name, effects_dict). Falls back to ('unrated', ...) on
    miss (new cells that haven't been scored yet)."""
    from . import cell_key
    ck = cell_key(strategy, regime, time_bucket)
    row = conn.execute(
        "SELECT medal FROM current_medals WHERE cell_key=?", (ck,)
    ).fetchone()
    medal = row["medal"] if row is not None else "unrated"
    return medal, medal_effect(medal)


__all__ = [
    "MEDAL_EFFECTS",
    "GOLD_PCTILE", "SILVER_PCTILE", "BRONZE_PCTILE",
    "assign_medal", "medal_effect",
    "rescore_standings", "lookup_medal",
]
