"""Persistent state for the AI-loop. Tracks:
  - last-apply times per param (for cool-down enforcement)
  - cumulative applied-today count (daily cap)
  - the most recent AUTO_APPLIED commit SHA per param (for revert)
  - live-PnL history samples (for drift monitor)
  - frozen state (triggered by stop-loss or kill switch)

Stored as JSON under ai_loop_data/state.json. Every mutation writes
through immediately — no in-memory buffering — so crashes don't lose
audit-critical history.
"""
from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

from .config import STATE_FILE


def _default_state() -> dict:
    return {
        "schema_version": 1,
        "last_applied_at": {},        # {param: iso_ts}
        "last_applied_sha": {},       # {param: git_sha}
        "last_applied_new_value": {}, # {param: value}
        "applies_by_date": {},        # {iso_date: n}
        "live_pnl_samples": [],       # [{ts, date, pnl, drawdown}]
        "frozen": False,
        "freeze_reason": "",
        "freeze_until": None,
    }


def load() -> dict:
    if not STATE_FILE.exists():
        return _default_state()
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return _default_state()


def save(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")


def record_apply(param: str, new_value, sha: str) -> None:
    st = load()
    now = datetime.utcnow().isoformat() + "Z"
    st["last_applied_at"][param] = now
    st["last_applied_sha"][param] = sha
    st["last_applied_new_value"][param] = new_value
    d = date.today().isoformat()
    st["applies_by_date"][d] = st["applies_by_date"].get(d, 0) + 1
    save(st)


def record_pnl_sample(d: date, pnl: float, dd: float) -> None:
    st = load()
    sample = {"date": d.isoformat(), "ts": datetime.utcnow().isoformat() + "Z",
              "pnl": round(pnl, 2), "drawdown": round(dd, 2)}
    st["live_pnl_samples"].append(sample)
    # Cap history
    st["live_pnl_samples"] = st["live_pnl_samples"][-120:]
    save(st)


def set_frozen(reason: str, until_iso: str | None = None) -> None:
    st = load()
    st["frozen"] = True
    st["freeze_reason"] = reason
    st["freeze_until"] = until_iso
    save(st)


def clear_freeze() -> None:
    st = load()
    st["frozen"] = False
    st["freeze_reason"] = ""
    st["freeze_until"] = None
    save(st)
