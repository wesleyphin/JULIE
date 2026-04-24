#!/usr/bin/env python3
"""Count minute-unique same-side suppression events across all available logs.

For each log:
    - Parse 'Bar: <mts> ET' to get current market minute
    - Parse 'Ignoring same-side signal' to log raw events
    - Dedupe at (market_minute, strategy, signal_side)
Tally per-log. Total sample = sum.

Decision: if total minute-unique events < 2000, honest kill (per user's gate).
"""
from __future__ import annotations
import re, sys
from pathlib import Path
from collections import defaultdict

ROOT = Path("/Users/wes/Downloads/JULIE001")

RE_HEADER = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
RE_BAR = re.compile(r"Bar: (?P<mts>\d{4}-\d{2}-\d{2} \d{2}:\d{2})")
RE_SAMESIDE = re.compile(r"Ignoring same-side signal while (?P<pos>LONG|SHORT) position is already active: (?P<strategy>\S+)")


def count_log(path: Path) -> dict:
    latest_bar_minute = None
    raw_n = 0
    uniq = set()
    with path.open(errors="ignore") as fh:
        for line in fh:
            if not RE_HEADER.match(line):
                continue
            bm = RE_BAR.search(line)
            if bm:
                latest_bar_minute = bm.group("mts")
                continue
            sm = RE_SAMESIDE.search(line)
            if sm:
                raw_n += 1
                if latest_bar_minute is not None:
                    uniq.add((latest_bar_minute, sm.group("strategy"), sm.group("pos")))
    return {"raw": raw_n, "uniq": len(uniq)}


def main():
    logs = []
    # Canonical monthly replay logs
    rdir = ROOT / "backtest_reports/full_live_replay"
    for m in ("2025_01 2025_02 2025_03 2025_04 2025_05 2025_06 2025_07 "
              "2025_08 2025_09 2025_10 2025_11 2025_12 2026_01 2026_02 "
              "2026_03 2026_04").split():
        p = rdir / m / "topstep_live_bot.log"
        if p.exists(): logs.append(p)
    # Current live log
    live = ROOT / "topstep_live_bot.log"
    if live.exists(): logs.append(live)

    print(f"{'log':<70} {'raw':>7} {'uniq':>7}")
    print("-" * 90)
    total_raw, total_uniq = 0, 0
    per_log = []
    for p in logs:
        if p.stat().st_size < 50_000: continue
        r = count_log(p)
        per_log.append((p, r))
        rel = p.relative_to(ROOT)
        print(f"  {str(rel):<68} {r['raw']:>7} {r['uniq']:>7}")
        total_raw += r["raw"]
        total_uniq += r["uniq"]
    print("-" * 90)
    print(f"  {'TOTAL':<68} {total_raw:>7} {total_uniq:>7}")
    print()
    print(f"=== sample-size decision ===")
    print(f"  minute-unique events: {total_uniq}")
    gate = 2000
    if total_uniq < gate:
        print(f"  BELOW gate of {gate} → HONEST KILL per user policy")
        return 1
    print(f"  ABOVE gate of {gate} → proceed with training pipeline")
    return 0


if __name__ == "__main__":
    sys.exit(main())
