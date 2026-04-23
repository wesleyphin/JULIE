"""Load the live-price parquet (built by price_parquet_updater.py) and
compute per-day / per-window regime features for the analyzer + journal.

Features we surface:
    open, high, low, close
    range_pts          (high - low)
    trend_pts          (close - open; positive = up day)
    trend_dir          "up" / "down" / "flat"
    realized_vol_pts   stdev of 1-min log returns × √n_bars (simple estimate)
    n_bars             number of bars the day has on record
    sessions: dict     keyed by session name → sub-dict of same stats

`load_prices()` is cached across calls within a single process.
"""
from __future__ import annotations

import math
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from .config import DATA_DIR

PRICE_PARQUET = DATA_DIR / "live_prices.parquet"


# ─── sessions (NY hours, match backtest_journal.py) ─────────
_SESSIONS = [
    ("pre_open",   4.0,  9.5),
    ("morning",    9.5, 12.0),
    ("lunch",     12.0, 14.0),
    ("afternoon", 14.0, 16.0),
    ("post_close",16.0, 17.0),
    ("overnight", 17.0, 28.0),   # wraps past midnight
]


def _session_of(ts: pd.Timestamp) -> str:
    h = ts.hour + ts.minute / 60.0
    for name, lo, hi in _SESSIONS:
        if lo <= h < hi:
            return name
    return "overnight"


@lru_cache(maxsize=1)
def load_prices() -> pd.DataFrame | None:
    """Return the whole live-price DataFrame, or None if the parquet hasn't
    been built yet. Cached per-process."""
    if not PRICE_PARQUET.exists():
        return None
    try:
        df = pd.read_parquet(PRICE_PARQUET)
    except Exception:
        return None
    if df.empty:
        return None
    # Normalize to NY tz
    if df.index.tz is None:
        df.index = df.index.tz_localize("America/New_York",
                                        ambiguous="NaT",
                                        nonexistent="shift_forward")
    else:
        df.index = df.index.tz_convert("America/New_York")
    df = df.sort_index()
    return df


def clear_cache() -> None:
    load_prices.cache_clear()


def _stats_for(sub: pd.DataFrame) -> dict:
    if sub.empty:
        return {"n_bars": 0}
    px = sub["price"].astype(float)
    open_px = float(px.iloc[0])
    close_px = float(px.iloc[-1])
    hi = float(px.max())
    lo = float(px.min())
    range_pts = hi - lo
    trend_pts = close_px - open_px
    trend_dir = "up" if trend_pts > 0.5 else "down" if trend_pts < -0.5 else "flat"
    # realized vol as stdev of 1-bar absolute moves (pts) * sqrt(n_bars)
    diffs = px.diff().dropna().abs()
    rv_pts = float(diffs.std()) if len(diffs) > 1 else 0.0
    return {
        "n_bars": int(len(sub)),
        "open": round(open_px, 2),
        "high": round(hi, 2),
        "low": round(lo, 2),
        "close": round(close_px, 2),
        "range_pts": round(range_pts, 2),
        "trend_pts": round(trend_pts, 2),
        "trend_dir": trend_dir,
        "bar_vol_pts": round(rv_pts, 3),
    }


def day_context(d: date | str, df: pd.DataFrame | None = None) -> dict | None:
    """Per-day stats + per-session breakdown. Returns None if no bars for d."""
    df = df if df is not None else load_prices()
    if df is None:
        return None
    if isinstance(d, str):
        d = datetime.strptime(d, "%Y-%m-%d").date()
    start = pd.Timestamp(d, tz="America/New_York")
    end = start + pd.Timedelta(days=1)
    sub = df.loc[(df.index >= start) & (df.index < end)]
    if sub.empty:
        return None
    out = {"date": d.isoformat(), **_stats_for(sub)}
    # per session
    sessions = {}
    for name, _, _ in _SESSIONS:
        ssub = sub[sub.index.map(_session_of) == name]
        if not ssub.empty:
            sessions[name] = _stats_for(ssub)
    out["sessions"] = sessions
    return out


def window_context(
    start: date | str,
    end: date | str,
    df: pd.DataFrame | None = None,
) -> dict | None:
    """Stats across a [start, end] inclusive date window."""
    df = df if df is not None else load_prices()
    if df is None:
        return None
    if isinstance(start, str): start = datetime.strptime(start, "%Y-%m-%d").date()
    if isinstance(end, str):   end   = datetime.strptime(end,   "%Y-%m-%d").date()
    lo = pd.Timestamp(start, tz="America/New_York")
    hi = pd.Timestamp(end, tz="America/New_York") + pd.Timedelta(days=1)
    sub = df.loc[(df.index >= lo) & (df.index < hi)]
    if sub.empty:
        return None
    # per-day roll-up
    by_day: dict[str, dict] = {}
    for d, g in sub.groupby(sub.index.date):
        by_day[d.isoformat()] = _stats_for(g)
    agg = {
        "window": (start.isoformat(), end.isoformat()),
        "n_days": len(by_day),
        **_stats_for(sub),
        "by_day": by_day,
    }
    return agg


def annotate_daily_journal(journal_stats: dict) -> dict:
    """Given a daily journal's top-level dict (has `date`), attach a
    `price_context` field in place and return it.

    The daily journal.py currently doesn't call this — leave the hook
    here so we can drop `journal_stats = annotate_daily_journal(js)`
    into the write-journal flow later without another pass."""
    if "date" not in journal_stats:
        return journal_stats
    ctx = day_context(journal_stats["date"])
    if ctx is not None:
        journal_stats["price_context"] = ctx
    return journal_stats


def annotate_backtest_stats(stats: dict, per_day_cap: int = 150) -> dict:
    """Given a backtest-consensus `stats` dict, add `price_context` with
    per-day roll-ups for every day the trade-tape references (up to
    `per_day_cap`) plus an aggregate summary. Mutates + returns stats.
    """
    df = load_prices()
    if df is None:
        stats["price_context"] = {"available": False, "reason": "parquet missing"}
        return stats
    # pull the set of days from the backtest stats.
    # Prefer best/worst days we already kept, fall back to date_range.
    days: list[str] = []
    for k in ("best_days", "worst_days"):
        for d, _ in stats.get(k, []):
            if d not in days:
                days.append(d)
    if "date_range" in stats and stats["date_range"]:
        first, last = stats["date_range"]
        # add every day in range if cheap
        try:
            d0 = datetime.strptime(first, "%Y-%m-%d").date()
            d1 = datetime.strptime(last, "%Y-%m-%d").date()
            cur = d0
            while cur <= d1 and len(days) < per_day_cap:
                ds = cur.isoformat()
                if ds not in days:
                    days.append(ds)
                cur = cur + (pd.Timedelta(days=1).to_pytimedelta())
        except Exception:
            pass

    per_day: dict[str, dict] = {}
    for d in days[:per_day_cap]:
        ctx = day_context(d, df=df)
        if ctx is not None:
            per_day[d] = ctx

    # aggregate — wins-vs-losses split if stats supplies day_pnls
    day_pnls = {}
    for d, p in stats.get("best_days", []) + stats.get("worst_days", []):
        day_pnls[d] = p

    def _summarize(days_subset):
        ranges = [per_day[d]["range_pts"] for d in days_subset if d in per_day]
        vols = [per_day[d]["bar_vol_pts"] for d in days_subset if d in per_day]
        return {
            "n": len(ranges),
            "avg_range_pts": round(sum(ranges)/len(ranges), 2) if ranges else None,
            "avg_bar_vol_pts": round(sum(vols)/len(vols), 3) if vols else None,
        }

    best_side = _summarize([d for d, p in day_pnls.items() if p > 0])
    worst_side = _summarize([d for d, p in day_pnls.items() if p < 0])

    stats["price_context"] = {
        "available": True,
        "n_days_with_prices": len(per_day),
        "best_days_summary": best_side,
        "worst_days_summary": worst_side,
        "per_day": per_day,
    }
    return stats


__all__ = [
    "load_prices",
    "clear_cache",
    "day_context",
    "window_context",
    "annotate_daily_journal",
    "annotate_backtest_stats",
]
