"""SQLite-backed ledger — persistent store for signals, outcomes, and standings.

Four tables:
    signals          — every signal the bot generates (fired or blocked)
    outcomes         — realized or counterfactual PnL for each signal
    standings        — per-cell league scores and medal, one row per scoring run
    retrain_queue    — purity-drop retrain requests (populated by retrain_hook)

The ledger is WAL-mode to tolerate concurrent readers (dashboard,
analyzer) while the bot writes. All writes are autocommitted via
context-manager transactions.

Schema is created lazily on first `open_db()` — no separate migration
step. Adding columns later is safe via `ALTER TABLE ... ADD COLUMN` with
appropriate defaults; existing rows see NULL.

Signal IDs are UUID4 strings so we don't collide across restart
boundaries or across the seed/live boundary.
"""
from __future__ import annotations

import contextlib
import json
import logging
import sqlite3
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

from . import DATA_DIR, LEDGER_PATH


# ─── schema ──────────────────────────────────────────────────
_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS signals (
    signal_id       TEXT PRIMARY KEY,
    ts              TEXT NOT NULL,
    strategy        TEXT NOT NULL,
    sub_strategy    TEXT,
    side            TEXT NOT NULL,
    regime          TEXT NOT NULL,
    time_bucket     TEXT NOT NULL,
    entry_price     REAL NOT NULL,
    tp_dist         REAL,
    sl_dist         REAL,
    size            INTEGER,
    status          TEXT NOT NULL,      -- 'fired' / 'blocked' / 'counterfactual_pending' / 'counterfactual_resolved'
    block_filter    TEXT,               -- null when fired
    block_reason    TEXT,
    source_tag      TEXT NOT NULL       -- 'live' / 'seed_2025' / 'seed_2026' / etc
);
CREATE INDEX IF NOT EXISTS idx_signals_strategy_ts ON signals(strategy, ts);
CREATE INDEX IF NOT EXISTS idx_signals_cell ON signals(strategy, regime, time_bucket);
CREATE INDEX IF NOT EXISTS idx_signals_status ON signals(status);
CREATE INDEX IF NOT EXISTS idx_signals_source_tag ON signals(source_tag);

CREATE TABLE IF NOT EXISTS outcomes (
    signal_id       TEXT PRIMARY KEY,
    pnl_dollars     REAL NOT NULL,
    pnl_points      REAL,
    exit_source     TEXT,               -- 'stop' / 'take' / 'reverse' / 'cf_tp' / 'cf_sl' / 'cf_expired' / ...
    bars_held       INTEGER,
    counterfactual  INTEGER NOT NULL,   -- 0 = real, 1 = simulated for a blocked signal
    FOREIGN KEY (signal_id) REFERENCES signals(signal_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS standings (
    cell_key        TEXT NOT NULL,
    scored_at       TEXT NOT NULL,
    n_signals       INTEGER NOT NULL,
    n_fired         INTEGER NOT NULL,
    n_blocked       INTEGER NOT NULL,
    purity          REAL,
    cash            REAL,
    velocity        REAL,
    purity_rank     INTEGER,
    cash_rank       INTEGER,
    velocity_rank   INTEGER,
    medal           TEXT NOT NULL,      -- 'gold' / 'silver' / 'bronze' / 'probation' / 'unrated'
    PRIMARY KEY (cell_key, scored_at)
);
CREATE INDEX IF NOT EXISTS idx_standings_scored_at ON standings(scored_at);
CREATE INDEX IF NOT EXISTS idx_standings_medal ON standings(medal);

CREATE TABLE IF NOT EXISTS retrain_queue (
    cell_key        TEXT NOT NULL,
    queued_at       TEXT NOT NULL,
    reason          TEXT NOT NULL,
    status          TEXT NOT NULL,      -- 'queued' / 'running' / 'completed' / 'failed' / 'cancelled'
    completed_at    TEXT,
    notes           TEXT,
    PRIMARY KEY (cell_key, queued_at)
);
CREATE INDEX IF NOT EXISTS idx_retrain_status ON retrain_queue(status);

-- Materialized view of "current" medals per cell: the most recent
-- standings row per cell_key. Refreshed by rescore_standings().
CREATE TABLE IF NOT EXISTS current_medals (
    cell_key        TEXT PRIMARY KEY,
    scored_at       TEXT NOT NULL,
    medal           TEXT NOT NULL,
    n_signals       INTEGER NOT NULL,
    purity          REAL,
    cash            REAL,
    velocity        REAL
);
"""


@dataclass
class Signal:
    """One row of the `signals` table."""
    ts: datetime
    strategy: str
    side: str
    regime: str
    time_bucket: str
    entry_price: float
    sub_strategy: Optional[str] = None
    tp_dist: Optional[float] = None
    sl_dist: Optional[float] = None
    size: Optional[int] = None
    status: str = "fired"              # fired / blocked / counterfactual_pending / counterfactual_resolved
    block_filter: Optional[str] = None
    block_reason: Optional[str] = None
    source_tag: str = "live"
    signal_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class Outcome:
    """One row of the `outcomes` table."""
    signal_id: str
    pnl_dollars: float
    pnl_points: Optional[float] = None
    exit_source: Optional[str] = None
    bars_held: Optional[int] = None
    counterfactual: bool = False


# ─── connection management ────────────────────────────────────
def open_db(path: Path = LEDGER_PATH) -> sqlite3.Connection:
    """Open (creating if needed) the ledger DB. WAL mode + foreign keys on."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), isolation_level=None, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.row_factory = sqlite3.Row
    # Create schema
    with conn:
        for stmt in _SCHEMA_SQL.split(";"):
            s = stmt.strip()
            if s:
                conn.execute(s)
    return conn


@contextlib.contextmanager
def connection(path: Path = LEDGER_PATH):
    """Context-managed DB connection with BEGIN/COMMIT around the yielded session."""
    conn = open_db(path)
    try:
        conn.execute("BEGIN")
        yield conn
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


# ─── insertions ─────────────────────────────────────────────
def insert_signal(conn: sqlite3.Connection, signal: Signal) -> str:
    """Insert a signal row; returns the signal_id (generated if missing)."""
    conn.execute(
        """INSERT OR REPLACE INTO signals
           (signal_id, ts, strategy, sub_strategy, side, regime, time_bucket,
            entry_price, tp_dist, sl_dist, size, status, block_filter,
            block_reason, source_tag)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            signal.signal_id,
            signal.ts.isoformat() if isinstance(signal.ts, datetime) else str(signal.ts),
            signal.strategy, signal.sub_strategy, signal.side,
            signal.regime, signal.time_bucket,
            float(signal.entry_price),
            None if signal.tp_dist is None else float(signal.tp_dist),
            None if signal.sl_dist is None else float(signal.sl_dist),
            None if signal.size is None else int(signal.size),
            signal.status, signal.block_filter, signal.block_reason,
            signal.source_tag,
        ),
    )
    return signal.signal_id


def insert_outcome(conn: sqlite3.Connection, outcome: Outcome) -> None:
    """Insert or replace an outcome row."""
    conn.execute(
        """INSERT OR REPLACE INTO outcomes
           (signal_id, pnl_dollars, pnl_points, exit_source, bars_held, counterfactual)
           VALUES (?,?,?,?,?,?)""",
        (
            outcome.signal_id,
            float(outcome.pnl_dollars),
            None if outcome.pnl_points is None else float(outcome.pnl_points),
            outcome.exit_source,
            None if outcome.bars_held is None else int(outcome.bars_held),
            1 if outcome.counterfactual else 0,
        ),
    )


def update_signal_status(conn: sqlite3.Connection, signal_id: str, status: str) -> None:
    conn.execute("UPDATE signals SET status=? WHERE signal_id=?", (status, signal_id))


# ─── helpful queries ──────────────────────────────────────────
def fetch_pending_counterfactuals(
    conn: sqlite3.Connection, limit: int = 500
) -> list[sqlite3.Row]:
    """Signals that were blocked (or counterfactual_pending) but don't yet
    have an outcome row. The counterfactual resolver consumes these.
    """
    return list(
        conn.execute(
            """
            SELECT s.*
            FROM signals s
            LEFT JOIN outcomes o ON o.signal_id = s.signal_id
            WHERE s.status IN ('blocked', 'counterfactual_pending')
              AND o.signal_id IS NULL
            ORDER BY s.ts
            LIMIT ?
            """,
            (limit,),
        )
    )


def fetch_cell_stats(
    conn: sqlite3.Connection,
    strategy: str,
    regime: str,
    time_bucket: str,
    include_counterfactual: bool = True,
) -> dict[str, Any]:
    """Aggregate fired + (optional counterfactual) signals for one cell."""
    cf_clause = "" if include_counterfactual else " AND o.counterfactual = 0"
    row = conn.execute(
        f"""
        SELECT
            COUNT(*) AS n_signals,
            SUM(CASE WHEN s.status='fired' THEN 1 ELSE 0 END) AS n_fired,
            SUM(CASE WHEN s.status='blocked' OR s.status LIKE 'counterfactual%' THEN 1 ELSE 0 END) AS n_blocked
        FROM signals s
        LEFT JOIN outcomes o ON o.signal_id = s.signal_id
        WHERE s.strategy=? AND s.regime=? AND s.time_bucket=?
          {cf_clause}
        """,
        (strategy, regime, time_bucket),
    ).fetchone()
    return dict(row) if row is not None else {}


def load_current_medals(conn: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    """Return {cell_key: {medal, scored_at, n_signals, purity, cash, velocity}}."""
    rows = conn.execute("SELECT * FROM current_medals").fetchall()
    return {row["cell_key"]: dict(row) for row in rows}


def all_signal_cells(conn: sqlite3.Connection) -> list[tuple[str, str, str]]:
    """Every (strategy, regime, time_bucket) triple present in the signals table."""
    return [
        (row["strategy"], row["regime"], row["time_bucket"])
        for row in conn.execute(
            "SELECT DISTINCT strategy, regime, time_bucket FROM signals"
        )
    ]


# ─── CLI convenience ──────────────────────────────────────────
def stats_summary(conn: sqlite3.Connection) -> dict[str, Any]:
    """Top-level counts for quick health-checks."""
    counts: dict[str, Any] = {}
    for status in ("fired", "blocked", "counterfactual_pending", "counterfactual_resolved"):
        row = conn.execute(
            "SELECT COUNT(*) AS c FROM signals WHERE status=?", (status,)
        ).fetchone()
        counts[status] = row["c"] if row else 0
    counts["total_signals"] = sum(counts.values())
    counts["total_outcomes"] = conn.execute(
        "SELECT COUNT(*) AS c FROM outcomes"
    ).fetchone()["c"]
    counts["n_cells_scored"] = conn.execute(
        "SELECT COUNT(*) AS c FROM current_medals"
    ).fetchone()["c"]
    counts["retrain_queue_open"] = conn.execute(
        "SELECT COUNT(*) AS c FROM retrain_queue WHERE status='queued'"
    ).fetchone()["c"]
    return counts


__all__ = [
    "Signal", "Outcome",
    "open_db", "connection",
    "insert_signal", "insert_outcome", "update_signal_status",
    "fetch_pending_counterfactuals", "fetch_cell_stats",
    "load_current_medals", "all_signal_cells",
    "stats_summary",
]
