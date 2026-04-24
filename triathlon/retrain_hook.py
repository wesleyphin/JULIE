"""Retrain-hook — queue Filter G retrains when a strategy's purity drops.

Monitors the standings table across rescoring runs. When a strategy's
aggregate purity (pooled across that strategy's cells) drops by more
than `PURITY_DROP_THRESHOLD` between two consecutive scoring runs,
queue a retrain request in `retrain_queue`.

The hook does NOT actually run the retrain — that's a separate
operator action (or a cron job in the future). The queue exists so
the operator sees which strategies the engine believes need fresh
training data, with a reason and a queued-at timestamp.

This keeps retraining safe and auditable: the engine can identify a
degradation signal, but a human (or a scheduled job with explicit
authorization) makes the decision to actually retrain.
"""
from __future__ import annotations

import logging
import os
import sqlite3
from datetime import datetime, timezone
from typing import Optional

from . import cell_key_parts


PURITY_DROP_THRESHOLD = 0.08
"""Minimum aggregate purity drop (absolute, not relative) between two
consecutive scoring runs to trigger a retrain queue entry. 0.08 means
a strategy that was hitting 60% drops to ≤52%."""

COOLDOWN_DAYS = 3
"""Don't re-queue a retrain for the same strategy within this window."""

# Queue-write kill switch (2026-04-24): when not explicitly '1', the
# `queue_retrains` function still RUNS detection + logs each candidate,
# but SKIPS the INSERT into retrain_queue. This keeps the monitoring
# channel (purity drops get journaled) without committing a retrain
# request to persistent state. `strategies_to_retrain` is untouched.
_QUEUE_WRITES_ENABLED = os.environ.get("JULIE_TRIATHLON_RETRAIN_QUEUE", "0").strip() == "1"


def strategies_to_retrain(conn: sqlite3.Connection) -> list[dict]:
    """Compare latest vs second-latest standings snapshots; return a list of
    per-strategy retrain candidates where aggregate purity dropped >=
    threshold.

    Each candidate: {'strategy', 'prev_purity', 'cur_purity', 'drop',
    'n_signals'}.
    """
    runs = [
        row["scored_at"] for row in conn.execute(
            "SELECT DISTINCT scored_at FROM standings ORDER BY scored_at DESC LIMIT 2"
        )
    ]
    if len(runs) < 2:
        return []
    cur_run, prev_run = runs[0], runs[1]

    def agg(run_at: str) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for row in conn.execute(
            """
            SELECT cell_key, n_fired, purity
            FROM standings
            WHERE scored_at = ? AND purity IS NOT NULL AND n_fired > 0
            """,
            (run_at,),
        ):
            strategy = cell_key_parts(row["cell_key"])[0]
            bucket = out.setdefault(strategy, {"sum_p_n": 0.0, "n": 0})
            bucket["sum_p_n"] += (row["purity"] or 0.0) * (row["n_fired"] or 0)
            bucket["n"] += (row["n_fired"] or 0)
        # Weighted-average purity per strategy
        return {
            s: {
                "purity": (b["sum_p_n"] / b["n"]) if b["n"] > 0 else 0.0,
                "n_fired": b["n"],
            }
            for s, b in out.items()
        }

    cur_agg = agg(cur_run)
    prev_agg = agg(prev_run)

    candidates = []
    for strat, cur in cur_agg.items():
        prev = prev_agg.get(strat)
        if prev is None:
            continue
        drop = prev["purity"] - cur["purity"]
        if drop >= PURITY_DROP_THRESHOLD:
            candidates.append({
                "strategy": strat,
                "prev_purity": round(prev["purity"], 4),
                "cur_purity": round(cur["purity"], 4),
                "drop": round(drop, 4),
                "n_signals": cur["n_fired"],
                "scored_at_prev": prev_run,
                "scored_at_cur": cur_run,
            })
    return candidates


def queue_retrains(
    conn: sqlite3.Connection,
    *,
    now: Optional[datetime] = None,
) -> list[dict]:
    """Detect degraded strategies and append entries to retrain_queue.

    Respects COOLDOWN_DAYS: won't queue a retrain for a strategy
    already in the queue within the cooldown window.

    Returns the list of newly-queued entries (may be empty).
    """
    if now is None:
        now = datetime.now(timezone.utc)
    candidates = strategies_to_retrain(conn)
    if not candidates:
        return []

    queued: list[dict] = []
    for c in candidates:
        strat = c["strategy"]
        # Cooldown check: any queued/running entry for this strategy in last N days?
        from datetime import timedelta
        cutoff = (now - timedelta(days=COOLDOWN_DAYS)).isoformat()
        existing = conn.execute(
            """
            SELECT COUNT(*) AS c FROM retrain_queue
            WHERE status IN ('queued', 'running')
              AND cell_key LIKE ?
              AND queued_at >= ?
            """,
            (f"{strat}|%", cutoff),
        ).fetchone()
        if existing and existing["c"] > 0:
            continue
        reason = (
            f"purity_drop {c['prev_purity']:.3f}→{c['cur_purity']:.3f} "
            f"(Δ={c['drop']:.3f}, n={c['n_signals']})"
        )
        queue_ts = now.isoformat()
        if not _QUEUE_WRITES_ENABLED:
            # Journaling-only mode: log the detection but DON'T write to
            # retrain_queue. Operator can still see the signal via log
            # output or by running `python3 -m triathlon queue-retrains`
            # to list candidates.
            logging.info(
                "[triathlon.retrain-hook] DETECTED degradation (not queued — "
                "JULIE_TRIATHLON_RETRAIN_QUEUE=0): %s | %s", strat, reason,
            )
            queued.append({"strategy": strat, "queue_ts": queue_ts,
                            "reason": reason, "queued_to_db": False})
            continue
        # Queue one entry per cell_key prefix (we use strategy|* as a
        # strategy-wide retrain marker; cells are scoped via the
        # strategy prefix).
        conn.execute(
            """INSERT OR REPLACE INTO retrain_queue
               (cell_key, queued_at, reason, status, completed_at, notes)
               VALUES (?,?,?,?,?,?)""",
            (f"{strat}|_ANY_REGIME|_ANY_BUCKET",
             queue_ts, reason, "queued", None, None),
        )
        queued.append({"strategy": strat, "queue_ts": queue_ts,
                        "reason": reason, "queued_to_db": True})
    return queued


__all__ = [
    "PURITY_DROP_THRESHOLD", "COOLDOWN_DAYS",
    "strategies_to_retrain", "queue_retrains",
]
