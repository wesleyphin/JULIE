#!/usr/bin/env python3
"""Backfill Triathlon ledger from the live bot log (topstep_live_bot.log).

Source of truth:
  - Entries         : [TRADE_PLACED] events (strategy, side, entry, tp, sl, order_id)
  - Regime context  : nearest preceding `regime=<name>` from any filter/gate line
  - Real exits      : "📊 Trade closed ..." events (only ~5/day — reverse closes)
  - Synthetic exits : forward-walk ai_loop_data/live_prices.parquet with TP/SL
                      brackets for every TRADE_PLACED that lacks a real exit

The dashboard's `recent_signals` filter shows signals with paired outcomes
OR explicit blocks, so each ingested entry needs an outcome row. Signals
that land AFTER the parquet's last bar (e.g. 04/23 afternoon, right now)
will carry no outcome — those trades will just be absent from the dashboard
list until the parquet refreshes or a real exit is logged.

Usage:
    python3 scripts/backfill_from_bot_log.py               # 04/22 + 04/23
    python3 scripts/backfill_from_bot_log.py 2026-04-22    # one date
"""
from __future__ import annotations

import re
import sqlite3
import sys
import uuid
from collections import namedtuple
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from triathlon import time_bucket_of, REGIMES  # noqa: E402
from triathlon.counterfactual import simulate_signal, _load_prices  # noqa: E402

# Only canonical triathlon regimes are accepted; volatility-regime strings
# (normal/high/low/ultra_low) share the `regime=` prefix in filter lines
# and must be ignored or they pollute the cell bucketing.
VALID_REGIMES = set(REGIMES)

LOG_PATH = ROOT / "topstep_live_bot.log"
LEDGER = ROOT / "ai_loop_data" / "triathlon" / "ledger.db"
PT = ZoneInfo("America/Los_Angeles")  # bot logs in local PT
NY = ZoneInfo("America/New_York")
MES_PT_VALUE = 5.0

# ---- parsers ---------------------------------------------------------------

RE_HEADER = re.compile(r"^(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}),(\d{3}) ")

RE_TRADE_PLACED = re.compile(
    r"\[TRADE_PLACED\] .*?Order placed: (?P<side>LONG|SHORT) @ (?P<entry>[-\d.]+) "
    r"\| order_id=(?P<oid>[a-f0-9-]+) "
    r"\| strategy=(?P<strategy>\S+) "
    r"\| side=(?:LONG|SHORT) "
    r"\| entry=(?P<entry2>[-\d.]+) "
    r"\| tp=(?P<tp>[-\d.]+) "
    r"\| sl=(?P<sl>[-\d.]+)"
)

RE_STRATEGY_SIGNAL = re.compile(
    r"\[STRATEGY_SIGNAL\] .*?strategy=(?P<strategy>\S+) "
    r"\| side=(?P<side>LONG|SHORT).*?"
    r"(?:sub_strategy=(?P<sub>\S+))?"
)

RE_TRADE_CLOSED_REAL = re.compile(
    r"📊 Trade closed(?: \([^)]+\))?: (?P<strat>[^:|\s]+)(?::(?P<sub>\S+))? "
    r"(?P<side>LONG|SHORT) \| Entry: (?P<entry>[-\d.]+) "
    r"\| Exit: (?P<exit>[-\d.]+) "
    r"\| PnL: (?P<pts>-?[\d.]+) pts \(\$(?P<pnl>-?[\d.]+)\) "
    r"\| source=(?P<src>\S+) .*?size=(?P<size>\d+)"
)

RE_REGIME = re.compile(r"regime=(?P<regime>[a-z_]+)")

# SENDING ORDER gives size (separate line from TRADE_PLACED)
RE_SENDING_ORDER = re.compile(
    r"SENDING ORDER: (?P<side>LONG|SHORT) @ ~(?P<entry>[-\d.]+)"
)
RE_SIZE_LINE = re.compile(r"Size: (?P<size>\d+) contracts")

Entry = namedtuple(
    "Entry",
    "ts strategy sub side entry tp_dist sl_dist size order_id regime time_bucket",
)
RealExit = namedtuple("RealExit", "ts strategy side entry exit pnl_dollars pnl_points src size")


# ---- log walker ------------------------------------------------------------


def parse_log(log_path: Path, dates: set[str]) -> tuple[list[Entry], list[RealExit]]:
    """Walk the log once, extract TRADE_PLACED + real close events for the
    given set of YYYY-MM-DD log-date strings (bot-local PT).

    Regime is looked up as the most-recent `regime=<name>` seen before the
    TRADE_PLACED event. Size is pulled from the preceding "Size: N contracts"
    line that accompanies every entry."""
    entries: list[Entry] = []
    real_exits: list[RealExit] = []
    last_regime = "unknown"
    pending_size = None  # size read from the preceding SENDING ORDER block
    with log_path.open(errors="ignore") as fh:
        for raw in fh:
            hm = RE_HEADER.match(raw)
            if not hm:
                continue
            date_str, time_str, ms = hm.group(1), hm.group(2), hm.group(3)
            if date_str not in dates:
                # still track regime across day boundary? not needed; skip
                continue
            # PT local → build aware timestamp
            ts_pt = datetime.strptime(
                f"{date_str} {time_str}.{ms}", "%Y-%m-%d %H:%M:%S.%f"
            ).replace(tzinfo=PT)
            line = raw.rstrip("\n")

            # rolling regime context (only accept canonical triathlon regimes)
            m = RE_REGIME.search(line)
            if m and m.group("regime") in VALID_REGIMES:
                last_regime = m.group("regime")

            # size hint from "Size: N contracts"
            m = RE_SIZE_LINE.search(line)
            if m:
                pending_size = int(m.group("size"))
                continue

            m = RE_TRADE_PLACED.search(line)
            if m:
                entry = float(m.group("entry"))
                tp = float(m.group("tp"))
                sl = float(m.group("sl"))
                side = m.group("side")
                # tp_dist / sl_dist are always positive distances
                tp_dist = abs(tp - entry)
                sl_dist = abs(sl - entry)
                ny_hour = ts_pt.astimezone(NY).hour + ts_pt.astimezone(NY).minute / 60.0
                tb = time_bucket_of(ny_hour)
                entries.append(Entry(
                    ts=ts_pt,
                    strategy=m.group("strategy"),
                    sub=None,
                    side=side,
                    entry=entry,
                    tp_dist=tp_dist,
                    sl_dist=sl_dist,
                    size=pending_size or 1,
                    order_id=m.group("oid"),
                    regime=last_regime,
                    time_bucket=tb,
                ))
                pending_size = None
                continue

            m = RE_TRADE_CLOSED_REAL.search(line)
            if m:
                real_exits.append(RealExit(
                    ts=ts_pt,
                    strategy=m.group("strat"),
                    side=m.group("side"),
                    entry=float(m.group("entry")),
                    exit=float(m.group("exit")),
                    pnl_dollars=float(m.group("pnl")),
                    pnl_points=float(m.group("pts")),
                    src=m.group("src"),
                    size=int(m.group("size")),
                ))
    return entries, real_exits


# ---- matching + insert -----------------------------------------------------


def pair_real_exit(entry: Entry, real_exits: list[RealExit]) -> RealExit | None:
    """Return the earliest RealExit after this entry matching strategy/side
    and entry-price tolerance. Consumes the match so it isn't reused."""
    tol = 0.5  # half-point match tolerance on entry price
    for i, r in enumerate(real_exits):
        if r.ts <= entry.ts:
            continue
        if r.strategy != entry.strategy:
            continue
        if r.side != entry.side:
            continue
        if abs(r.entry - entry.entry) > tol:
            continue
        return real_exits.pop(i)
    return None


def bars_held(entry_ts: datetime, exit_ts: datetime) -> int:
    return max(1, int((exit_ts - entry_ts).total_seconds() // 60))


def backfill(entries: list[Entry], real_exits: list[RealExit]) -> dict[str, int]:
    """Insert signals + outcomes. Uses real exit where available, else
    forward-walks the parquet."""
    conn = sqlite3.connect(LEDGER)
    conn.row_factory = sqlite3.Row
    # dedupe: our backfill tags block_reason='botlog_backfill:...'
    existing = {
        (r["ts"], r["side"], r["entry_price"])
        for r in conn.execute(
            "SELECT ts, side, entry_price FROM signals WHERE block_reason LIKE 'botlog_backfill:%'"
        )
    }
    prices = _load_prices()
    if prices is not None:
        prices = prices.sort_index()

    stats = {"inserted_real": 0, "inserted_cf": 0, "inserted_pending": 0,
             "skipped_dupe": 0}

    for e in entries:
        entry_iso = e.ts.isoformat()
        key = (entry_iso, e.side, round(e.entry, 4))
        if key in existing:
            stats["skipped_dupe"] += 1
            continue

        sid = str(uuid.uuid4())
        # find real exit
        real = pair_real_exit(e, real_exits)
        cf = None
        if real is None and prices is not None:
            cf = simulate_signal(
                prices, e.side, e.entry, e.tp_dist, e.sl_dist, e.ts,
                size=e.size,
            )
        # decide outcome classification
        if real is not None:
            pnl_dollars = real.pnl_dollars
            pnl_points = real.pnl_points
            exit_source = real.src
            bh = bars_held(e.ts, real.ts)
            note = f"real_exit:{real.src}"
            stats["inserted_real"] += 1
        elif cf is not None:
            pnl_pts, bh, exit_source = cf
            pnl_points = pnl_pts
            pnl_dollars = pnl_pts * MES_PT_VALUE * e.size
            note = f"cf:{exit_source}"
            stats["inserted_cf"] += 1
        else:
            note = "no_exit_data"
            stats["inserted_pending"] += 1

        conn.execute(
            """INSERT INTO signals
               (signal_id, ts, strategy, sub_strategy, side, regime,
                time_bucket, entry_price, tp_dist, sl_dist, size, status,
                block_filter, block_reason, source_tag)
               VALUES (?, ?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, 'fired',
                       NULL, ?, 'live')""",
            (sid, entry_iso, e.strategy, e.side, e.regime, e.time_bucket,
             e.entry, e.tp_dist, e.sl_dist, e.size,
             f"botlog_backfill:{note}"),
        )
        if real is not None or cf is not None:
            conn.execute(
                """INSERT INTO outcomes
                   (signal_id, pnl_dollars, pnl_points, exit_source,
                    bars_held, counterfactual)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (sid, round(pnl_dollars, 2), round(pnl_points, 2),
                 exit_source, bh, 1 if real is None else 0),
            )
    conn.commit()
    conn.close()
    return stats


# ---- main ------------------------------------------------------------------


def main() -> None:
    if len(sys.argv) > 1:
        dates = set(sys.argv[1:])
    else:
        dates = {"2026-04-22", "2026-04-23"}

    entries, real_exits = parse_log(LOG_PATH, dates)
    print(f"=== bot-log backfill for {sorted(dates)} ===")
    print(f"TRADE_PLACED events parsed : {len(entries)}")
    print(f"Real 'Trade closed' events : {len(real_exits)}")

    # strategy breakdown
    from collections import Counter
    strat_count = Counter(e.strategy for e in entries)
    regime_count = Counter(e.regime for e in entries)
    bucket_count = Counter(e.time_bucket for e in entries)
    print(f"  strategies       : {dict(strat_count)}")
    print(f"  regime snapshots : {dict(regime_count)}")
    print(f"  time_buckets     : {dict(bucket_count)}")

    stats = backfill(entries, real_exits)
    print()
    print(f"inserted (real exit)         : {stats['inserted_real']}")
    print(f"inserted (counterfactual)    : {stats['inserted_cf']}")
    print(f"inserted (no outcome yet)    : {stats['inserted_pending']}")
    print(f"skipped (already backfilled) : {stats['skipped_dupe']}")


if __name__ == "__main__":
    main()
