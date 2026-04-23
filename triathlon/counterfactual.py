"""Counterfactual resolver — score blocked signals via forward-walk through the price parquet.

A blocked signal has an `entry_price`, `tp_dist`, `sl_dist`, and a
timestamp. We simulate what would have happened if the trade had
fired, by walking the bars_df (sourced from
`ai_loop_data/live_prices.parquet`) forward from the signal's
timestamp until we hit:

  1. the TP price (cf_tp, winner)
  2. the SL price (cf_sl, loser)
  3. the lookahead cap (cf_expired, close at last observed price)

The resulting Outcome is tagged `counterfactual=True` so league
scoring can choose whether to fold it in (default: no, keep the cash
metric reflecting live-realized PnL only — but the validator uses it
to estimate what blocks cost).

Resolver is idempotent: any signal with status in
('blocked', 'counterfactual_pending') that lacks an outcome row will
be picked up on the next pass. Runs in batches of up to
`MAX_BATCH_SIZE`; safe to call repeatedly.
"""
from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from . import LEDGER_PATH
from .ledger import Outcome, insert_outcome, update_signal_status, open_db


MAX_BATCH_SIZE = 1000
DEFAULT_LOOKAHEAD_BARS = 60
MES_POINT_VALUE = 5.0


def _load_prices() -> Optional[pd.DataFrame]:
    """Return the live-prices DataFrame if available; None otherwise."""
    try:
        from tools.ai_loop import price_context
        return price_context.load_prices()
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "[triathlon.cf] could not load live_prices parquet: %s", exc,
        )
        return None


def simulate_signal(
    df: pd.DataFrame,
    side: str,
    entry_price: float,
    tp_dist: float,
    sl_dist: float,
    entry_ts: datetime,
    *,
    size: int = 1,
    lookahead_bars: int = DEFAULT_LOOKAHEAD_BARS,
    point_value: float = MES_POINT_VALUE,
) -> Optional[tuple[float, int, str]]:
    """Walk forward from entry_ts, returning (pnl_points, bars_held, exit_source).

    Returns None if there's no price data after entry_ts.
    """
    ts = pd.Timestamp(entry_ts, tz="America/New_York") if entry_ts.tzinfo is None \
        else pd.Timestamp(entry_ts).tz_convert("America/New_York")
    sub = df.loc[df.index > ts].head(lookahead_bars)
    if sub.empty:
        return None
    px = sub["price"].astype(float)
    side_up = str(side).upper()
    if side_up == "LONG":
        sl_price = entry_price - sl_dist
        tp_price = entry_price + tp_dist
    else:
        sl_price = entry_price + sl_dist
        tp_price = entry_price - tp_dist

    for i, (ts_i, price) in enumerate(px.items(), start=1):
        if side_up == "LONG":
            if price <= sl_price:
                return (-sl_dist, i, "cf_sl")
            if price >= tp_price:
                return (tp_dist, i, "cf_tp")
        else:
            if price >= sl_price:
                return (-sl_dist, i, "cf_sl")
            if price <= tp_price:
                return (tp_dist, i, "cf_tp")
    last = float(px.iloc[-1])
    if side_up == "LONG":
        pnl_pts = last - entry_price
    else:
        pnl_pts = entry_price - last
    return (pnl_pts, len(px), "cf_expired")


def resolve_pending(
    conn: Optional[sqlite3.Connection] = None,
    *,
    df: Optional[pd.DataFrame] = None,
    max_batch: int = MAX_BATCH_SIZE,
    lookahead_bars: int = DEFAULT_LOOKAHEAD_BARS,
    point_value: float = MES_POINT_VALUE,
    verbose: bool = False,
) -> dict[str, int]:
    """Walk pending counterfactuals and write outcomes for each.

    Returns {'resolved': N, 'skipped_no_data': N, 'skipped_bad_signal': N}.
    Does NOT score cells; caller runs `rescore_standings` afterward.
    """
    close_conn = False
    if conn is None:
        conn = open_db()
        close_conn = True
    if df is None:
        df = _load_prices()
    if df is None:
        if close_conn: conn.close()
        return {"resolved": 0, "skipped_no_data": 0, "skipped_bad_signal": 0}

    counts = {"resolved": 0, "skipped_no_data": 0, "skipped_bad_signal": 0}

    rows = list(conn.execute(
        """
        SELECT s.signal_id, s.ts, s.side, s.entry_price,
               s.tp_dist, s.sl_dist, s.size, s.status
        FROM signals s
        LEFT JOIN outcomes o ON o.signal_id = s.signal_id
        WHERE s.status IN ('blocked', 'counterfactual_pending')
          AND o.signal_id IS NULL
          AND s.tp_dist IS NOT NULL
          AND s.sl_dist IS NOT NULL
        ORDER BY s.ts
        LIMIT ?
        """,
        (max_batch,),
    ))

    for row in rows:
        try:
            entry_ts = datetime.fromisoformat(row["ts"])
        except (ValueError, TypeError):
            counts["skipped_bad_signal"] += 1
            continue
        if row["tp_dist"] is None or row["sl_dist"] is None or row["entry_price"] is None:
            counts["skipped_bad_signal"] += 1
            continue

        result = simulate_signal(
            df=df, side=row["side"], entry_price=row["entry_price"],
            tp_dist=float(row["tp_dist"]), sl_dist=float(row["sl_dist"]),
            entry_ts=entry_ts,
            size=row["size"] or 1,
            lookahead_bars=lookahead_bars, point_value=point_value,
        )
        if result is None:
            counts["skipped_no_data"] += 1
            continue
        pnl_pts, bars_held, exit_source = result
        size = row["size"] or 1
        pnl_dollars = pnl_pts * point_value * size

        conn.execute("BEGIN")
        try:
            insert_outcome(conn, Outcome(
                signal_id=row["signal_id"],
                pnl_dollars=round(pnl_dollars, 2),
                pnl_points=round(pnl_pts, 4),
                exit_source=exit_source, bars_held=bars_held,
                counterfactual=True,
            ))
            update_signal_status(conn, row["signal_id"], "counterfactual_resolved")
            conn.execute("COMMIT")
            counts["resolved"] += 1
            if verbose and counts["resolved"] % 100 == 0:
                logging.getLogger(__name__).info(
                    "[triathlon.cf] resolved %d blocked signals", counts["resolved"]
                )
        except Exception:
            conn.execute("ROLLBACK")
            raise

    if close_conn:
        conn.close()
    return counts


__all__ = [
    "MAX_BATCH_SIZE",
    "DEFAULT_LOOKAHEAD_BARS",
    "MES_POINT_VALUE",
    "simulate_signal",
    "resolve_pending",
]
