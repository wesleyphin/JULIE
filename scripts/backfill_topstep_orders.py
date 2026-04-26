#!/usr/bin/env python3
"""Backfill Topstep orders_export.csv into the Triathlon ledger.

Problem: the signal-birth hook wasn't wired when the bot was trading
04/22 and early 04/23, so the Triathlon dashboard's recent-signals list
looks empty compared to what the operator sees on Topstep. Fix: parse
the Topstep order export, pair Openings with their Closings, and insert
synthetic (signal + outcome) rows into the ledger tagged as live.

We tag these synthetic rows with source_tag='live' (so they flow through
the existing export.py filter that shows "signals with a paired outcome")
and regime='backfill' / strategy='DynamicEngine3' (DE3 is the only live
strategy in this session). Time-bucket comes from the entry's NY hour.

Usage:
    python3 scripts/backfill_topstep_orders.py /Users/wes/Downloads/orders_export.csv
"""
from __future__ import annotations

import csv
import sqlite3
import sys
import uuid
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from triathlon import time_bucket_of, cell_key  # noqa: E402

LEDGER = ROOT / "ai_loop_data" / "triathlon" / "ledger.db"
NY = ZoneInfo("America/New_York")
MES_PT_VALUE = 5.0  # $5 per point per contract


def parse_ts(s: str) -> datetime:
    # Topstep format: "04/23/2026 00:01:14 -07:00"
    return datetime.strptime(s.strip(), "%m/%d/%Y %H:%M:%S %z")


def load_filled(csv_path: Path) -> list[dict]:
    rows = []
    with csv_path.open() as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            if r.get("Status") != "Filled":
                continue
            rows.append(r)
    return rows


def walk_positions(rows: list[dict]) -> list[dict]:
    """Given all Filled rows, pair Openings → Closings to reconstruct
    completed trades. Single-position bot: one Opening establishes the
    position; subsequent Closings (of opposite side) close it."""
    # Sort by filled time
    rows_sorted = sorted(rows, key=lambda r: parse_ts(r["FilledAt"]))
    trades = []
    open_pos = None  # dict | None
    for r in rows_sorted:
        disp = r.get("PositionDisposition", "")
        size = int(r.get("Size") or 0)
        side = r.get("Side", "")  # Bid = buy, Ask = sell
        price = float(r.get("ExecutePrice") or 0.0)
        ts = parse_ts(r["FilledAt"])
        # Opening leg
        if disp == "Opening":
            if open_pos is not None:
                # Flip without an explicit close — treat the new Opening
                # as if the prior was flat-closed at this leg's price.
                # (Rare; happens on reverse orders.)
                prior = open_pos
                exit_px = price
                pts = (exit_px - prior["entry_px"]) * (
                    1 if prior["side"] == "LONG" else -1
                )
                trades.append({
                    "entry_ts":   prior["entry_ts"],
                    "exit_ts":    ts,
                    "side":       prior["side"],
                    "size":       prior["size"],
                    "entry_px":   prior["entry_px"],
                    "exit_px":    exit_px,
                    "pnl_points": pts,
                    "pnl_dollars": pts * MES_PT_VALUE * prior["size"],
                    "exit_source": "flip",
                })
            long_side = (side == "Bid")
            open_pos = {
                "entry_ts":   ts,
                "side":       "LONG" if long_side else "SHORT",
                "size":       size,
                "entry_px":   price,
            }
        # Closing leg
        elif disp == "Closing":
            if open_pos is None:
                continue  # orphan closer (cancelled remnant) — skip
            exit_px = price
            pts = (exit_px - open_pos["entry_px"]) * (
                1 if open_pos["side"] == "LONG" else -1
            )
            # Bracket closer type: Stop = stop-loss, Limit = take-profit,
            # Market (ClosePosition) = manual flatten
            order_type = r.get("Type", "")
            creation = r.get("CreationDisposition", "")
            if order_type == "Stop":
                exit_source = "stop"
            elif order_type == "Limit":
                exit_source = "take"
            elif creation == "ClosePosition":
                exit_source = "close_trade_leg"
            else:
                exit_source = order_type.lower() or "close"
            trades.append({
                "entry_ts":    open_pos["entry_ts"],
                "exit_ts":     ts,
                "side":        open_pos["side"],
                "size":        open_pos["size"],
                "entry_px":    open_pos["entry_px"],
                "exit_px":     exit_px,
                "pnl_points":  pts,
                "pnl_dollars": pts * MES_PT_VALUE * open_pos["size"],
                "exit_source": exit_source,
            })
            open_pos = None
    return trades


def bars_held(entry_ts: datetime, exit_ts: datetime) -> int:
    """Minutes held (bot runs 1-min bars)."""
    return max(1, int((exit_ts - entry_ts).total_seconds() // 60))


def insert_trades(trades: list[dict]) -> tuple[int, int]:
    """Insert (signal, outcome) pairs. Returns (inserted, skipped_dupes)."""
    conn = sqlite3.connect(LEDGER)
    conn.row_factory = sqlite3.Row
    # Dedupe key: (entry_ts rounded to second, side, size, entry_px).
    # Use block_reason to stamp the topstep order-linkage marker so we
    # don't double-ingest on rerun.
    existing = {
        (r["ts"], r["side"], r["size"], round(r["entry_price"] or 0, 4))
        for r in conn.execute(
            "SELECT ts, side, size, entry_price FROM signals WHERE block_reason LIKE 'topstep_backfill:%'"
        )
    }
    ins = 0
    skip = 0
    for t in trades:
        entry_iso = t["entry_ts"].isoformat()
        key = (entry_iso, t["side"], t["size"], round(t["entry_px"], 4))
        if key in existing:
            skip += 1
            continue
        sid = str(uuid.uuid4())
        ny_hour = t["entry_ts"].astimezone(NY).hour + \
                  t["entry_ts"].astimezone(NY).minute / 60.0
        tb = time_bucket_of(ny_hour)
        regime = "backfill"          # distinct bucket; doesn't pollute real cells
        strategy = "DynamicEngine3"  # only live strat
        conn.execute(
            """INSERT INTO signals
               (signal_id, ts, strategy, sub_strategy, side, regime,
                time_bucket, entry_price, tp_dist, sl_dist, size, status,
                block_filter, block_reason, source_tag)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?, 'fired',
                       NULL, ?, 'live')""",
            (sid, entry_iso, strategy, None, t["side"], regime,
             tb, t["entry_px"], t["size"],
             f"topstep_backfill:{t['exit_source']}"),
        )
        conn.execute(
            """INSERT INTO outcomes
               (signal_id, pnl_dollars, pnl_points, exit_source,
                bars_held, counterfactual)
               VALUES (?, ?, ?, ?, ?, 0)""",
            (sid, round(t["pnl_dollars"], 2), round(t["pnl_points"], 2),
             t["exit_source"], bars_held(t["entry_ts"], t["exit_ts"])),
        )
        ins += 1
    conn.commit()
    conn.close()
    return ins, skip


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: backfill_topstep_orders.py <orders_export.csv>")
        sys.exit(2)
    csv_path = Path(sys.argv[1])
    if not csv_path.exists():
        print(f"not found: {csv_path}")
        sys.exit(2)

    filled = load_filled(csv_path)
    trades = walk_positions(filled)
    ins, skip = insert_trades(trades)

    # Per-day stats
    wins = sum(1 for t in trades if t["pnl_dollars"] > 0)
    losses = sum(1 for t in trades if t["pnl_dollars"] < 0)
    flats = sum(1 for t in trades if t["pnl_dollars"] == 0)
    net = sum(t["pnl_dollars"] for t in trades)

    by_day: dict[str, list[dict]] = {}
    for t in trades:
        d = t["entry_ts"].astimezone(NY).date().isoformat()
        by_day.setdefault(d, []).append(t)

    print(f"=== Topstep CSV backfill — {csv_path.name} ===")
    print(f"filled rows parsed      : {len(filled)}")
    print(f"reconstructed trades    : {len(trades)}  "
          f"(W {wins} / L {losses} / flat {flats})")
    print(f"aggregate PnL (est.)    : ${net:+,.2f}")
    print(f"inserted into ledger    : {ins}")
    print(f"skipped (already present): {skip}")
    print()
    print("per-day breakdown:")
    for d in sorted(by_day):
        day = by_day[d]
        w = sum(1 for t in day if t["pnl_dollars"] > 0)
        lo = sum(1 for t in day if t["pnl_dollars"] < 0)
        pnl = sum(t["pnl_dollars"] for t in day)
        print(f"  {d}  n={len(day):3d}  W={w:3d}  L={lo:3d}  "
              f"pnl=${pnl:+9,.2f}")


if __name__ == "__main__":
    main()
