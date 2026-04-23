"""Seed the ledger from historical closed_trades.json files.

Historical closed_trades don't carry the regime label the live bot
attaches at signal birth. We compute regime at signal time by
replaying the price series from the relevant parquet through the
regime classifier — same rolling-close / vol-eff calculator that runs
live. That way historical cells map onto the same taxonomy the live
bot uses.

Entry points:

    seed_from_trade_files(paths, source_tag, ...) — ingest N closed_trades.json
    seed_2025_and_2026_full() — preset that ingests the same source
        bundle the backtest-consensus journals use

All inserts are idempotent: seeding the same file twice won't
double-insert (signal IDs are deterministic-ish via a hash on entry
time + strategy + side + entry price).
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from . import (
    LEDGER_PATH, REPO_ROOT,
    STRATEGIES, REGIMES, TIME_BUCKETS,
    time_bucket_of,
)
from .ledger import Outcome, Signal, insert_outcome, insert_signal, open_db


# ─── regime replay ────────────────────────────────────────────
try:
    from regime_classifier import (
        RegimeClassifier, WINDOW_BARS as REGIME_WINDOW_BARS,
        DEAD_TAPE_VOL_BP, EFF_LOW, EFF_HIGH,
    )
except Exception:
    RegimeClassifier = None
    REGIME_WINDOW_BARS = 120
    DEAD_TAPE_VOL_BP = 1.5
    EFF_LOW = 0.05
    EFF_HIGH = 0.12


def _classify_from_scratch(closes: list[float]) -> str:
    """Standalone regime classifier mirroring regime_classifier.py's
    _classify() logic. Used when the live RegimeClassifier class isn't
    available (circular-import safeguard).
    """
    if len(closes) < REGIME_WINDOW_BARS:
        return "warmup"
    recent = closes[-REGIME_WINDOW_BARS:]
    rets = []
    for i in range(1, len(recent)):
        p0 = recent[i - 1]
        if p0 > 0:
            rets.append((recent[i] - p0) / p0)
    if not rets:
        return "neutral"
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / max(1, len(rets) - 1)
    vol_bp = math.sqrt(var) * 10_000.0
    abs_sum = sum(abs(r) for r in rets)
    eff = abs(sum(rets)) / abs_sum if abs_sum > 0 else 0.0
    if vol_bp < DEAD_TAPE_VOL_BP:
        return "dead_tape"
    if vol_bp > 3.5 and eff < EFF_LOW:
        return "whipsaw"
    if eff > EFF_HIGH:
        return "calm_trend"
    return "neutral"


def _build_regime_lookup(df, ts_iter) -> dict[datetime, str]:
    """For each target timestamp, look up the regime computed from the
    most recent REGIME_WINDOW_BARS prior closes in `df`.

    `df` is the live_prices parquet. `ts_iter` is an iterable of
    datetime objects (tz-aware NY) we need regime labels for.

    Returns {ts: regime_name}. Timestamps not in df range get "warmup".
    """
    if df is None or df.empty:
        return {ts: "warmup" for ts in ts_iter}
    # Work from sorted unique timestamps for linear scanning
    targets = sorted(set(ts_iter))
    out: dict[datetime, str] = {}
    import bisect
    df_index = df.index
    closes = df["price"].astype(float).tolist()
    for ts in targets:
        # Locate the largest index ≤ ts
        # df_index is tz-aware NY; ts may be naive (assume NY) or aware
        import pandas as pd
        ts_ny = pd.Timestamp(ts)
        if ts_ny.tzinfo is None:
            ts_ny = ts_ny.tz_localize("America/New_York", ambiguous="NaT", nonexistent="shift_forward")
        else:
            ts_ny = ts_ny.tz_convert("America/New_York")
        pos = df_index.searchsorted(ts_ny, side="right") - 1
        if pos < 0:
            out[ts] = "warmup"
            continue
        window_end = pos + 1
        window_start = max(0, window_end - REGIME_WINDOW_BARS)
        window = closes[window_start:window_end]
        out[ts] = _classify_from_scratch(window)
    return out


# ─── seed ingestion ───────────────────────────────────────────
def _stable_signal_id(trade: dict, source_tag: str) -> str:
    """Deterministic signal_id so re-seeding the same file doesn't
    produce duplicate rows."""
    h = hashlib.sha1(
        f"{source_tag}|{trade.get('entry_time')}|{trade.get('strategy')}|"
        f"{trade.get('side')}|{trade.get('entry_price')}|{trade.get('sub_strategy') or trade.get('combo_key')}"
        .encode("utf-8")
    ).hexdigest()
    return f"seed_{h[:24]}"


def _infer_sl_tp_dist(trade: dict) -> tuple[Optional[float], Optional[float]]:
    """Derive tp_dist / sl_dist from the closed_trade row. closed_trades
    don't carry these directly, but DE3 sub-strategy names encode them:
    `5min_09-12_Long_Rev_T5_SL10_TP25` → sl=10, tp=25.
    """
    sub = str(trade.get("sub_strategy") or trade.get("combo_key") or "")
    sl = tp = None
    # Match `_SL<number>` and `_TP<number>` substrings
    import re
    m_sl = re.search(r"_SL([\d.]+)", sub)
    m_tp = re.search(r"_TP([\d.]+)", sub)
    if m_sl:
        try: sl = float(m_sl.group(1))
        except ValueError: sl = None
    if m_tp:
        try: tp = float(m_tp.group(1))
        except ValueError: tp = None
    return sl, tp


def seed_from_trade_files(
    paths: list[Path],
    source_tag: str,
    *,
    conn: Optional[sqlite3.Connection] = None,
    verbose: bool = True,
) -> dict[str, int]:
    """Ingest one or more closed_trades.json files into the ledger.

    Deduplicates via deterministic signal_id hashing — safe to call
    repeatedly. Returns {'inserted_signals': N, 'inserted_outcomes': N,
    'skipped': N, 'source_files': N}.
    """
    close_conn = False
    if conn is None:
        conn = open_db()
        close_conn = True

    # Load live_prices parquet once to classify regime per signal
    prices_df = None
    try:
        from tools.ai_loop import price_context
        prices_df = price_context.load_prices()
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "[triathlon.seed] prices parquet unavailable (%s) — regime=neutral fallback", exc,
        )

    # Pass 1: collect every trade timestamp so we can batch-lookup regimes
    all_trades: list[tuple[Path, dict]] = []
    for p in paths:
        if not p.exists():
            if verbose:
                logging.getLogger(__name__).warning("[triathlon.seed] missing: %s", p)
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            logging.getLogger(__name__).warning("[triathlon.seed] %s parse error: %s", p, exc)
            continue
        for t in data:
            all_trades.append((p, t))

    entry_times = set()
    for _, t in all_trades:
        try:
            entry_times.add(datetime.fromisoformat(t["entry_time"]))
        except Exception:
            pass

    regime_lookup = _build_regime_lookup(prices_df, entry_times) if prices_df is not None else {}

    counts = {
        "inserted_signals": 0, "inserted_outcomes": 0,
        "skipped": 0, "source_files": len({str(p) for p, _ in all_trades}),
    }

    conn.execute("BEGIN")
    try:
        for src_path, trade in all_trades:
            try:
                entry_ts = datetime.fromisoformat(trade["entry_time"])
            except Exception:
                counts["skipped"] += 1
                continue
            strategy = str(trade.get("strategy") or "").strip() or "Unknown"
            side = str(trade.get("side") or "").upper()
            if side not in ("LONG", "SHORT"):
                counts["skipped"] += 1
                continue

            # Regime from price replay; fallback to neutral
            regime = regime_lookup.get(entry_ts, "neutral")

            # Time bucket from NY hour
            hour = entry_ts.hour + entry_ts.minute / 60.0
            tb = time_bucket_of(hour)

            sl_dist, tp_dist = _infer_sl_tp_dist(trade)
            size = int(trade.get("size") or 1)

            sig = Signal(
                signal_id=_stable_signal_id(trade, source_tag),
                ts=entry_ts,
                strategy=strategy,
                sub_strategy=trade.get("sub_strategy") or trade.get("combo_key"),
                side=side,
                regime=regime,
                time_bucket=tb,
                entry_price=float(trade.get("entry_price") or 0.0),
                tp_dist=tp_dist,
                sl_dist=sl_dist,
                size=size,
                status="fired",
                source_tag=source_tag,
            )
            insert_signal(conn, sig)
            counts["inserted_signals"] += 1

            # Outcome
            try:
                pnl_dollars = float(trade.get("pnl_dollars") or 0.0)
            except Exception:
                pnl_dollars = 0.0
            try:
                pnl_points = float(trade.get("pnl_points") or 0.0)
            except Exception:
                pnl_points = None

            # bars_held: approximate from exit_time - entry_time in minutes
            bars_held = None
            exit_time_str = trade.get("exit_time")
            if exit_time_str:
                try:
                    exit_ts = datetime.fromisoformat(exit_time_str)
                    delta = (exit_ts - entry_ts).total_seconds() / 60.0
                    if delta >= 0:
                        bars_held = max(1, int(round(delta)))
                except Exception:
                    pass

            exit_source = str(trade.get("source") or "").lower() or None
            insert_outcome(conn, Outcome(
                signal_id=sig.signal_id,
                pnl_dollars=pnl_dollars,
                pnl_points=pnl_points,
                exit_source=exit_source,
                bars_held=bars_held,
                counterfactual=False,
            ))
            counts["inserted_outcomes"] += 1
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise

    if close_conn:
        conn.close()
    return counts


# ─── preset bundles ───────────────────────────────────────────
def seed_2025_and_2026_full(*, conn: Optional[sqlite3.Connection] = None) -> dict:
    """Run the canonical seed: 2025 monthly replays + 2026 Jan-Apr bundle.

    Mirrors the source list used by the backtest-consensus journals.
    """
    root = REPO_ROOT
    paths_2025 = [
        root / f"backtest_reports/full_live_replay/2025_{m:02d}/closed_trades.json"
        for m in range(1, 13)
    ] + [root / "backtest_reports/full_live_replay/outrageous_apr/closed_trades.json"]
    paths_2026 = [
        root / "backtest_reports/full_live_replay/2026_jan_apr/closed_trades.json",
        root / "backtest_reports/full_live_replay/2026_04_ml_stacks/closed_trades.json",
        root / "backtest_reports/replay_apr2026_p1/live_loop_MES_20260421_061829/closed_trades.json",
        root / "backtest_reports/af_fast_replay/2026_01/closed_trades.json",
        root / "backtest_reports/af_fast_replay/2026_02/closed_trades.json",
        root / "backtest_reports/af_fast_replay/2026_03/closed_trades.json",
        root / "backtest_reports/af_fast_replay/2026_04/closed_trades.json",
        root / "backtest_reports/pivot_week_4_19_21_ml/live_loop_MES_20260422_013828/closed_trades.json",
    ]
    summary = {"2025": {}, "2026": {}}
    summary["2025"] = seed_from_trade_files(paths_2025, source_tag="seed_2025", conn=conn)
    summary["2026"] = seed_from_trade_files(paths_2026, source_tag="seed_2026", conn=conn)
    return summary


__all__ = [
    "seed_from_trade_files", "seed_2025_and_2026_full",
]
