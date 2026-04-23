"""Export current Triathlon Engine state to JSON for the dashboard.

Writes `montecarlo/Backtest-Simulator-main/public/triathlon_state.json`
(same directory the existing FilterlessLiveApp fetches from). The
dashboard's Triathlon tab polls this file on a short interval.

Schema:
    {
      "generated_at": "2026-04-23T...",
      "n_cells_total": 46,
      "n_cells_rated": 26,
      "medal_counts": {"gold": 9, "silver": 16, "bronze": 1, "probation": 0, "unrated": 20},
      "cells": [
        {
          "cell_key": "DynamicEngine3|whipsaw|lunch",
          "strategy": "DynamicEngine3",
          "regime": "whipsaw",
          "time_bucket": "lunch",
          "medal": "gold",
          "n_signals": 71,
          "purity": 0.7183,
          "cash": 15.46,
          "velocity": 0.1111,
          "purity_rank": 1,
          "cash_rank": 1,
          "velocity_rank": 1
        },
        ...
      ],
      "recent_signals": [<last 50 live signals — fired + blocked>],
      "retrain_queue_open": <count>
    }
"""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from . import REPO_ROOT, cell_key_parts
from .ledger import open_db


PUBLIC_DIR = REPO_ROOT / "montecarlo" / "Backtest-Simulator-main" / "public"
DEFAULT_OUTPUT_PATH = PUBLIC_DIR / "triathlon_state.json"


def export_state(
    path: Path = DEFAULT_OUTPUT_PATH,
    *,
    conn: Optional[sqlite3.Connection] = None,
    recent_signal_limit: int = 50,
) -> Path:
    """Write the current Triathlon state JSON to `path`. Returns the path.

    Safe to call repeatedly; overwrites the file atomically via a
    tempfile + rename.
    """
    close_conn = False
    if conn is None:
        conn = open_db()
        close_conn = True

    # Full per-cell snapshot joined from the most recent standings row
    cells = []
    medal_counts: dict[str, int] = {}
    rows = conn.execute(
        """
        SELECT
            cm.cell_key, cm.medal, cm.scored_at,
            cm.n_signals, cm.purity, cm.cash, cm.velocity,
            s.n_fired, s.n_blocked,
            s.purity_rank, s.cash_rank, s.velocity_rank
        FROM current_medals cm
        LEFT JOIN standings s
          ON s.cell_key = cm.cell_key AND s.scored_at = cm.scored_at
        """
    ).fetchall()
    for row in rows:
        strategy, regime, time_bucket = cell_key_parts(row["cell_key"])
        cells.append({
            "cell_key": row["cell_key"],
            "strategy": strategy,
            "regime": regime,
            "time_bucket": time_bucket,
            "medal": row["medal"],
            "n_signals": row["n_signals"],
            "n_fired": row["n_fired"],
            "n_blocked": row["n_blocked"],
            "purity": row["purity"],
            "cash": row["cash"],
            "velocity": row["velocity"],
            "purity_rank": row["purity_rank"],
            "cash_rank": row["cash_rank"],
            "velocity_rank": row["velocity_rank"],
            "scored_at": row["scored_at"],
        })
        medal_counts[row["medal"]] = medal_counts.get(row["medal"], 0) + 1
    # Sort cells by medal (gold→probation) then by cash desc
    medal_order = {"gold": 0, "silver": 1, "bronze": 2, "probation": 3, "unrated": 4}
    cells.sort(key=lambda c: (medal_order.get(c["medal"], 9), -(c.get("cash") or 0)))

    # Recent live signals.
    #
    # IMPORTANT: only show signals that either:
    #   (a) have a paired outcome row  → executed trade (closed)
    #   (b) are explicitly tagged `blocked`  → known block
    # Signals in status='fired' with NO outcome are near-certainly
    # SIGNAL-BIRTH recordings that got dropped by downstream filters
    # (single-position-at-a-time, Kalshi, impulse, regime) that aren't
    # instrumented to flip status → 'blocked'. Showing those in the
    # dashboard would imply trades happened that never reached the
    # broker — we filter them out so the Recent Signals list matches
    # what the operator sees on Topstep.
    recent = [
        dict(row)
        for row in conn.execute(
            f"""
            SELECT s.signal_id, s.ts, s.strategy, s.side, s.regime,
                   s.time_bucket, s.status, s.block_filter, s.block_reason,
                   s.entry_price, s.size,
                   o.pnl_dollars, o.exit_source, o.counterfactual
            FROM signals s
            LEFT JOIN outcomes o ON o.signal_id = s.signal_id
            WHERE s.source_tag = 'live'
              AND (o.signal_id IS NOT NULL OR s.status = 'blocked')
            ORDER BY s.ts DESC
            LIMIT ?
            """,
            (recent_signal_limit,),
        )
    ]

    # Separate counts for transparency: TOTAL signals recorded vs how
    # many were genuine executed trades (have outcome row). These
    # numbers disagree because DE3 fires candidate-level signals every
    # bar while a trade is open, all of which the signal-birth hook
    # records as "fired" even though only one became an order.
    counts_live = conn.execute(
        """SELECT
            SUM(CASE WHEN s.source_tag='live' THEN 1 ELSE 0 END)           AS n_live_total,
            SUM(CASE WHEN s.source_tag='live' AND o.signal_id IS NOT NULL
                     THEN 1 ELSE 0 END)                                     AS n_live_executed,
            SUM(CASE WHEN s.source_tag='live' AND s.status='blocked'
                     THEN 1 ELSE 0 END)                                     AS n_live_blocked
           FROM signals s LEFT JOIN outcomes o ON o.signal_id = s.signal_id
        """
    ).fetchone()

    # Retrain queue
    retrain_open = conn.execute(
        "SELECT COUNT(*) AS c FROM retrain_queue WHERE status='queued'"
    ).fetchone()["c"]
    retrain_entries = [
        dict(row)
        for row in conn.execute(
            "SELECT * FROM retrain_queue WHERE status='queued' ORDER BY queued_at DESC LIMIT 20"
        )
    ]

    # Totals
    n_fired_all = conn.execute(
        "SELECT COUNT(*) AS c FROM signals WHERE status='fired'"
    ).fetchone()["c"]
    n_blocked_all = conn.execute(
        "SELECT COUNT(*) AS c FROM signals WHERE status IN ('blocked','counterfactual_pending','counterfactual_resolved')"
    ).fetchone()["c"]

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_cells_total": len(cells),
        "n_cells_rated": sum(1 for c in cells if c["medal"] != "unrated"),
        "medal_counts": medal_counts,
        "totals": {
            "n_fired_all": n_fired_all,
            "n_blocked_all": n_blocked_all,
            "retrain_queue_open": retrain_open,
            # Live-only counts for the dashboard's transparency note.
            # n_live_executed is the count that matches Topstep trade
            # history; n_live_total - n_live_executed is the gap
            # between "DE3 wanted to fire" and "Topstep registered a
            # trade" (mostly single-position-at-a-time suppression).
            "n_live_total": int(counts_live["n_live_total"] or 0),
            "n_live_executed": int(counts_live["n_live_executed"] or 0),
            "n_live_blocked_flagged": int(counts_live["n_live_blocked"] or 0),
        },
        "cells": cells,
        "recent_signals": recent,
        "retrain_queue": retrain_entries,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)  # atomic

    if close_conn:
        conn.close()
    return path


__all__ = ["export_state", "DEFAULT_OUTPUT_PATH", "PUBLIC_DIR"]
