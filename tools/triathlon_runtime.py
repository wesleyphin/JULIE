"""Thin runtime adapter between julie001.py and the Triathlon Engine.

Why a separate adapter module: julie001.py imports this once at startup;
this module lazy-opens the sqlite ledger, caches medals in memory with a
refresh cadence, and fail-closes on any error so the Triathlon Engine
can never break live trading. If the triathlon package can't be imported
or the DB can't be opened, every public function returns a safe
default (medal='unrated', neutral effects, no-op recorders).

Public API consumed by julie001.py:

    is_active() -> bool
        Gates every call site. When False, all functions are cheap
        no-ops.

    lookup_signal_effects(strategy, regime, time_bucket) -> dict
        Return {'medal', 'size_mult', 'priority_delta', 'cell_key'}.
        Used at signal-birth to multiply size + nudge priority.

    record_signal(...)
        Insert a fired or blocked signal into the ledger.

    record_outcome(signal_id, pnl_dollars, ...)
        Insert the realized outcome when the trade closes.

    current_medal_snapshot() -> dict[str, str]
        Full {cell_key: medal} map, cached in memory.

    refresh_medals(force=False)
        Reload the cached medals map from disk. Called automatically on
        a timer; callable manually after a rescore.
"""
from __future__ import annotations

import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Optional


# ─── state ────────────────────────────────────────────────────
_ACTIVE_FLAG_ENV = "JULIE_TRIATHLON_ACTIVE"
_MEDAL_REFRESH_INTERVAL_SEC = 5 * 60    # 5 minutes
_DEFAULT_EFFECTS = {
    "medal": "unrated",
    "size_mult": 1.0,
    "priority_delta": 0,
    "cell_key": None,
}

_cache: dict[str, Any] = {
    "medals_by_cell": {},   # {cell_key: {medal, effects, ...}}
    "last_refresh_ts": 0.0,
    "disabled_reason": None,   # set when is_active() short-circuits to False
    "ledger_conn": None,
}


def _log() -> logging.Logger:
    return logging.getLogger("triathlon_runtime")


def is_active() -> bool:
    """Master gate. Reads env var `JULIE_TRIATHLON_ACTIVE` — anything
    other than '1' / 'true' / 'yes' disables the adapter. Also returns
    False if we previously failed to import / open the ledger
    (`_cache['disabled_reason']` holds the first-fault reason)."""
    if _cache.get("disabled_reason"):
        return False
    raw = os.environ.get(_ACTIVE_FLAG_ENV, "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _disable(reason: str) -> None:
    """Permanent disable — set once when the adapter can't recover."""
    if _cache.get("disabled_reason") is None:
        _log().warning("[triathlon] disabled: %s", reason)
        _cache["disabled_reason"] = reason


def _get_conn():
    """Lazy-open the ledger DB. On error, disables the adapter
    permanently and returns None."""
    if _cache.get("disabled_reason"):
        return None
    if _cache.get("ledger_conn") is not None:
        return _cache["ledger_conn"]
    try:
        from triathlon.ledger import open_db
        conn = open_db()
        _cache["ledger_conn"] = conn
        return conn
    except Exception as exc:
        _disable(f"ledger open failed: {exc!r}")
        return None


def refresh_medals(force: bool = False) -> None:
    """Reload the cached medal map. Throttled to
    `_MEDAL_REFRESH_INTERVAL_SEC` unless `force=True`."""
    if not is_active():
        return
    now = time.time()
    if not force and (now - _cache.get("last_refresh_ts", 0)) < _MEDAL_REFRESH_INTERVAL_SEC:
        return
    conn = _get_conn()
    if conn is None:
        return
    try:
        from triathlon.ledger import load_current_medals
        from triathlon.medals import medal_effect
        raw = load_current_medals(conn)
        # pre-compute effects so the hot path is a dict lookup, not a
        # medal-tier branch
        medals_by_cell: dict[str, dict[str, Any]] = {}
        for cell_key_str, row in raw.items():
            eff = medal_effect(row["medal"])
            medals_by_cell[cell_key_str] = {
                "medal": row["medal"],
                "size_mult": eff["size_mult"],
                "priority_delta": int(eff["priority_delta"]),
                "n_signals": row["n_signals"],
                "purity": row.get("purity"),
                "cash": row.get("cash"),
                "velocity": row.get("velocity"),
            }
        _cache["medals_by_cell"] = medals_by_cell
        _cache["last_refresh_ts"] = now
    except Exception as exc:
        # Transient read failure — don't permanently disable, just log
        _log().debug("[triathlon] medal refresh error: %s", exc)


def lookup_signal_effects(
    strategy: str, regime: str, time_bucket: str
) -> dict[str, Any]:
    """Hot path — called on every signal. Returns effects dict with
    defaults when inactive or cell not yet scored.

    Refreshes the cached medal map lazily on a time-bucket boundary
    (does not block the hot path)."""
    if not is_active():
        return dict(_DEFAULT_EFFECTS)
    try:
        from triathlon import cell_key
        refresh_medals()
        ck = cell_key(strategy, regime, time_bucket)
        entry = _cache["medals_by_cell"].get(ck)
        if entry is None:
            return {**_DEFAULT_EFFECTS, "cell_key": ck}
        return {
            "medal": entry["medal"],
            "size_mult": entry["size_mult"],
            "priority_delta": entry["priority_delta"],
            "cell_key": ck,
        }
    except Exception as exc:
        _log().debug("[triathlon] lookup_signal_effects error: %s", exc)
        return dict(_DEFAULT_EFFECTS)


def record_signal(
    *,
    strategy: str,
    sub_strategy: Optional[str],
    side: str,
    regime: str,
    time_bucket: str,
    entry_price: float,
    tp_dist: Optional[float],
    sl_dist: Optional[float],
    size: Optional[int],
    status: str,                # 'fired' or 'blocked'
    block_filter: Optional[str] = None,
    block_reason: Optional[str] = None,
    ts: Optional[datetime] = None,
) -> Optional[str]:
    """Persist a signal row. Returns the signal_id on success, None
    otherwise. Safe — never raises.
    """
    if not is_active():
        return None
    conn = _get_conn()
    if conn is None:
        return None
    try:
        from triathlon.ledger import Signal, insert_signal
        sig = Signal(
            signal_id=str(uuid.uuid4()),
            ts=ts or datetime.now(timezone.utc),
            strategy=strategy, sub_strategy=sub_strategy,
            side=str(side).upper(),
            regime=regime, time_bucket=time_bucket,
            entry_price=float(entry_price),
            tp_dist=None if tp_dist is None else float(tp_dist),
            sl_dist=None if sl_dist is None else float(sl_dist),
            size=None if size is None else int(size),
            status=status,
            block_filter=block_filter, block_reason=block_reason,
            source_tag="live",
        )
        conn.execute("BEGIN")
        try:
            insert_signal(conn, sig)
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        return sig.signal_id
    except Exception as exc:
        _log().debug("[triathlon] record_signal error: %s", exc)
        return None


def record_outcome(
    signal_id: str,
    *,
    pnl_dollars: float,
    pnl_points: Optional[float] = None,
    exit_source: Optional[str] = None,
    bars_held: Optional[int] = None,
    counterfactual: bool = False,
) -> bool:
    """Persist a realized outcome for a previously-recorded signal.
    Returns True on success. Safe — never raises.
    """
    if not is_active() or not signal_id:
        return False
    conn = _get_conn()
    if conn is None:
        return False
    try:
        from triathlon.ledger import Outcome, insert_outcome
        conn.execute("BEGIN")
        try:
            insert_outcome(conn, Outcome(
                signal_id=signal_id,
                pnl_dollars=float(pnl_dollars),
                pnl_points=None if pnl_points is None else float(pnl_points),
                exit_source=exit_source, bars_held=bars_held,
                counterfactual=counterfactual,
            ))
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        return True
    except Exception as exc:
        _log().debug("[triathlon] record_outcome error: %s", exc)
        return False


def current_medal_snapshot() -> dict[str, str]:
    """Debug helper — full {cell_key: medal} snapshot."""
    refresh_medals()
    return {
        ck: entry["medal"]
        for ck, entry in _cache.get("medals_by_cell", {}).items()
    }


def time_bucket_of(ts_or_hour) -> str:
    """Utility for the julie001.py call site — convert a datetime or
    hour-of-day (float) into the canonical Triathlon time-bucket name."""
    try:
        from triathlon import time_bucket_of as _tb
        if isinstance(ts_or_hour, datetime):
            h = ts_or_hour.hour + ts_or_hour.minute / 60.0
        else:
            h = float(ts_or_hour)
        return _tb(h)
    except Exception:
        return "unknown"


__all__ = [
    "is_active", "refresh_medals",
    "lookup_signal_effects",
    "record_signal", "record_outcome",
    "current_medal_snapshot", "time_bucket_of",
]
