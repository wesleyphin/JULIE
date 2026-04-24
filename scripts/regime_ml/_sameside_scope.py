#!/usr/bin/env python3
"""Scoping pass: parse one replay log, extract same-side suppression events,
report per-event context availability (position state, signal price, market
timestamp), and count minute-unique events.

Purpose: before committing to a full training pipeline, verify we can
extract ALL needed fields reliably.
"""
from __future__ import annotations
import re, sys
from pathlib import Path
from collections import Counter, deque
from datetime import datetime, timedelta

ROOT = Path("/Users/wes/Downloads/JULIE001")
LOG = ROOT / "backtest_reports/full_live_replay/2025_04/topstep_live_bot.log"

RE_HEADER = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3})")
RE_BAR = re.compile(r"Bar: (?P<mts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) ET \| Price: (?P<px>[-\d.]+)")
RE_CANDIDATE = re.compile(
    r"📊 CANDIDATE.*?(?P<strategy>DynamicEngine3|RegimeAdaptive|AetherFlow)\S*\s+(?P<side>LONG|SHORT) @ (?P<price>[-\d.]+)"
)
RE_SAMESIDE = re.compile(
    r"Ignoring same-side signal while (?P<pos>LONG|SHORT) position is already active: (?P<strategy>\S+)"
)
RE_TRADE_PLACED = re.compile(
    r"\[TRADE_PLACED\].*?Order placed: (?P<side>LONG|SHORT) @ (?P<entry>[-\d.]+)"
    r".*?strategy=(?P<strategy>\S+).*?entry=(?P<e>[-\d.]+).*?tp=(?P<tp>[-\d.]+).*?sl=(?P<sl>[-\d.]+)"
)
RE_TRADE_CLOSED = re.compile(
    r"\[TRADE_CLOSED\] 💸 Trade closed|Trade closed \(reverse\)|Position Sync.*FLAT"
)


def main():
    print(f"parsing {LOG} (size {LOG.stat().st_size // 1024 // 1024} MB)")
    state_active = False          # True when a position is open
    state_side = None
    state_entry = None
    state_entry_mts = None        # market timestamp of entry
    state_tp = None
    state_sl = None
    last_candidate = None         # (log_ts, strategy, side, price)
    last_bar_mts = None
    last_bar_price = None

    raw_events = []
    events = []                   # one dict per same-side event

    with LOG.open(errors="ignore") as fh:
        for line in fh:
            hm = RE_HEADER.match(line)
            if not hm:
                continue
            log_ts = hm.group(1)

            bm = RE_BAR.search(line)
            if bm:
                last_bar_mts = bm.group("mts")
                last_bar_price = float(bm.group("px"))
                continue

            cm = RE_CANDIDATE.search(line)
            if cm:
                last_candidate = {
                    "log_ts": log_ts,
                    "strategy": cm.group("strategy"),
                    "side": cm.group("side"),
                    "price": float(cm.group("price")),
                }
                continue

            tp = RE_TRADE_PLACED.search(line)
            if tp:
                state_active = True
                state_side = tp.group("side")
                state_entry = float(tp.group("entry"))
                state_entry_mts = last_bar_mts
                state_tp = float(tp.group("tp"))
                state_sl = float(tp.group("sl"))
                continue

            if RE_TRADE_CLOSED.search(line):
                state_active = False
                state_side = None
                state_entry = None
                continue

            sm = RE_SAMESIDE.search(line)
            if sm:
                raw_events.append(log_ts)
                if not state_active:
                    continue   # can't label without position context
                # Link to preceding candidate
                sig = last_candidate
                events.append({
                    "log_ts": log_ts,
                    "market_ts": last_bar_mts,
                    "position_side": state_side,
                    "position_entry": state_entry,
                    "position_entry_ts": state_entry_mts,
                    "position_tp": state_tp,
                    "position_sl": state_sl,
                    "signal_side": sm.group("pos"),
                    "signal_strategy": sm.group("strategy"),
                    "signal_price": sig["price"] if sig and sig["side"] == sm.group("pos") else None,
                    "signal_has_price": sig is not None and sig["side"] == sm.group("pos"),
                    "latest_bar_price": last_bar_price,
                })

    print(f"\nraw same-side events:   {len(raw_events)}")
    print(f"events with position ctx: {len(events)}")
    if events:
        with_price = sum(1 for e in events if e["signal_has_price"])
        with_market_ts = sum(1 for e in events if e["market_ts"] is not None)
        with_entry = sum(1 for e in events if e["position_entry"] is not None)
        with_bar_px = sum(1 for e in events if e["latest_bar_price"] is not None)
        print(f"  signal price available: {with_price} ({with_price/len(events)*100:.1f}%)")
        print(f"  market_ts available:    {with_market_ts} ({with_market_ts/len(events)*100:.1f}%)")
        print(f"  position entry:         {with_entry} ({with_entry/len(events)*100:.1f}%)")
        print(f"  latest bar price:       {with_bar_px} ({with_bar_px/len(events)*100:.1f}%)")

    # Dedupe at (market minute, strategy, side, price)
    uniq = {}
    for e in events:
        if e["market_ts"] is None:
            continue
        minute = e["market_ts"][:16]
        key = (minute, e["signal_strategy"], e["signal_side"])
        if key not in uniq:
            uniq[key] = e
    print(f"\nminute-unique events (w/ market_ts): {len(uniq)}")

    if uniq:
        print("\n-- sample event ---")
        ex = list(uniq.values())[0]
        for k, v in ex.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
