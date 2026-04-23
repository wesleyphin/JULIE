"""Triathlon CLI — seed / rescore / resolve / status / queue.

Examples:
    python3 -m triathlon seed
    python3 -m triathlon rescore
    python3 -m triathlon resolve-cf --batch 500
    python3 -m triathlon status
    python3 -m triathlon medals --top 10
    python3 -m triathlon queue-retrains
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime

from . import LEDGER_PATH, cell_key_parts
from .counterfactual import resolve_pending as resolve_counterfactuals
from .export import export_state, DEFAULT_OUTPUT_PATH as EXPORT_PATH
from .ledger import load_current_medals, open_db, stats_summary
from .medals import rescore_standings, MEDAL_EFFECTS
from .retrain_hook import queue_retrains, strategies_to_retrain
from .seed import seed_2025_and_2026_full


def cmd_seed(args):
    print(f"[triathlon.seed] ingesting 2025 + 2026 historical trades...")
    conn = open_db()
    summary = seed_2025_and_2026_full(conn=conn)
    conn.close()
    for year, counts in summary.items():
        print(f"  {year}: {counts}")


def cmd_rescore(args):
    print(f"[triathlon.rescore] scoring every cell...")
    conn = open_db()
    scores, medals = rescore_standings(
        conn, include_counterfactual=args.include_counterfactual
    )
    # Auto-export the dashboard state after every rescore so the MC
    # dashboard's Triathlon tab stays fresh.
    try:
        out_path = export_state(conn=conn)
        print(f"  exported state -> {out_path}")
    except Exception as exc:
        print(f"  export failed: {exc}")
    conn.close()
    rated = [s for s in scores if s.is_rated()]
    print(f"  total cells: {len(scores)}")
    print(f"  rated (>= min_samples): {len(rated)}")
    by_medal: dict[str, int] = {}
    for m in medals.values():
        by_medal[m] = by_medal.get(m, 0) + 1
    for medal in ("gold", "silver", "bronze", "probation", "unrated"):
        count = by_medal.get(medal, 0)
        eff = MEDAL_EFFECTS[medal]
        print(f"  {medal:<10} {count:>3}  (prio {eff['priority_delta']:+d}, "
              f"size ×{eff['size_mult']:.2f})")


def cmd_export(args):
    print(f"[triathlon.export] writing dashboard state JSON...")
    conn = open_db()
    out = export_state(conn=conn)
    conn.close()
    print(f"  wrote {out}")


def cmd_resolve_cf(args):
    print(f"[triathlon.resolve-cf] forward-walking blocked signals...")
    conn = open_db()
    out = resolve_counterfactuals(conn, max_batch=args.batch, verbose=args.verbose)
    conn.close()
    print(f"  {out}")


def cmd_status(args):
    conn = open_db()
    summary = stats_summary(conn)
    print(f"[triathlon.status] {LEDGER_PATH}")
    for k, v in summary.items():
        print(f"  {k:<30} {v}")
    conn.close()


def cmd_medals(args):
    conn = open_db()
    medals = load_current_medals(conn)
    by_medal: dict[str, list] = {"gold": [], "silver": [], "bronze": [],
                                   "probation": [], "unrated": []}
    for ck, info in medals.items():
        by_medal.setdefault(info["medal"], []).append((ck, info))
    for medal in ("gold", "silver", "bronze", "probation"):
        rows = by_medal.get(medal, [])
        rows.sort(key=lambda x: -(x[1].get("cash") or 0))
        print(f"\n=== {medal.upper()} ({len(rows)}) ===")
        for ck, info in rows[: args.top]:
            strat, regime, tb = cell_key_parts(ck)
            print(f"  {strat:<16} {regime:<12} {tb:<12} "
                  f"n={info['n_signals']:>4}  purity={info.get('purity')}  "
                  f"cash=${(info.get('cash') or 0):+.2f}  "
                  f"velocity={info.get('velocity')}")
    conn.close()


def cmd_queue_retrains(args):
    conn = open_db()
    cands = strategies_to_retrain(conn)
    print(f"[triathlon.queue-retrains] candidates: {len(cands)}")
    for c in cands:
        print(f"  {c}")
    if not args.dry_run:
        queued = queue_retrains(conn)
        print(f"[triathlon.queue-retrains] queued: {len(queued)}")
        for q in queued:
            print(f"  {q}")
    conn.close()


def main():
    parser = argparse.ArgumentParser(prog="python3 -m triathlon")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("seed", help="Seed ledger from 2025 + 2026 historical trades")
    sp.set_defaults(fn=cmd_seed)

    sp = sub.add_parser("rescore", help="Rescore all cells + update medals")
    sp.add_argument("--include-counterfactual", action="store_true",
                    help="Fold counterfactual outcomes into cash league")
    sp.set_defaults(fn=cmd_rescore)

    sp = sub.add_parser("resolve-cf", help="Forward-walk blocked signals to produce counterfactual outcomes")
    sp.add_argument("--batch", type=int, default=500)
    sp.add_argument("--verbose", action="store_true")
    sp.set_defaults(fn=cmd_resolve_cf)

    sp = sub.add_parser("status", help="Print ledger health summary")
    sp.set_defaults(fn=cmd_status)

    sp = sub.add_parser("medals", help="Show current medal assignments")
    sp.add_argument("--top", type=int, default=10, help="Max cells per medal tier to show")
    sp.set_defaults(fn=cmd_medals)

    sp = sub.add_parser("queue-retrains", help="Detect purity drops + queue retrain entries")
    sp.add_argument("--dry-run", action="store_true")
    sp.set_defaults(fn=cmd_queue_retrains)

    sp = sub.add_parser("export", help="Write dashboard state JSON (montecarlo public/)")
    sp.set_defaults(fn=cmd_export)

    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
