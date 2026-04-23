"""Orchestrator — runs all 5 AI-loop layers in sequence.

Intended to be cron'd nightly at EOD:
    0 2 * * *  cd /Users/wes/Downloads/JULIE001 && python3 -m tools.ai_loop.run_daily >> ai_loop_data/run_daily.log 2>&1

Manual invocation:
    python3 -m tools.ai_loop.run_daily --dry-run    # observe, don't apply
    python3 -m tools.ai_loop.run_daily              # full loop, apply green-lit changes
    python3 -m tools.ai_loop.run_daily --date 2026-04-22
    python3 -m tools.ai_loop.run_daily --skip-apply   # run through layer 3 only

Exit status:
    0 = ok (may or may not have applied changes)
    1 = error in any layer
    2 = frozen (kill switch or stop-loss)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import date, datetime
from pathlib import Path

from . import journal, analyzer, validator, applier, monitor
from . import price_parquet_updater, price_context
from .config import is_frozen, KILL_SWITCH_ENV, DATA_DIR
from . import state


def main() -> int:
    ap = argparse.ArgumentParser(description="AI-loop nightly orchestrator")
    ap.add_argument("--date", default=None,
                    help="Date to process (YYYY-MM-DD); default = today")
    ap.add_argument("--dry-run", action="store_true",
                    help="Go through all layers but don't auto-apply any change.")
    ap.add_argument("--skip-apply", action="store_true",
                    help="Stop after validator (don't run applier or monitor).")
    ap.add_argument("--skip-monitor", action="store_true",
                    help="Don't run the drift monitor.")
    ap.add_argument(
        "--backtest-journal",
        action="append",
        default=None,
        help=(
            "Backtest consensus label(s) to feed the analyzer as priors. "
            "Repeatable. Default: ['2026_full']. Pass '' to disable."
        ),
    )
    args = ap.parse_args()
    # argparse "append" + "default" would double-add; supply default manually
    if args.backtest_journal is None:
        prior_labels = ["2026_full"]
    else:
        prior_labels = [lbl for lbl in args.backtest_journal if lbl]

    d = (datetime.strptime(args.date, "%Y-%m-%d").date() if args.date
         else date.today())
    t0 = datetime.utcnow()
    print(f"═══ AI-loop run — date={d} dry_run={args.dry_run} ═══")
    print(f"  frozen?          {is_frozen()}")
    print(f"  state.frozen?    {state.load().get('frozen', False)}")
    print(f"  kill_switch env  {os.environ.get(KILL_SWITCH_ENV, '(unset)')}")
    print(f"  data_dir         {DATA_DIR}")
    print()

    rc = 0
    try:
        # ─── Layer 0: Price parquet append (logs → parquet) ─
        print("Layer 0 · price-parquet updater")
        try:
            pu = price_parquet_updater.update_from_logs(verbose=False)
            print(f"  added {pu['new_rows']:,} new bars  "
                  f"(total on disk: {pu['total_rows']:,})")
            # reset cache so downstream rules pick up the new rows
            price_context.clear_cache()
        except Exception as exc:
            print(f"  ! updater error (non-fatal): {type(exc).__name__}: {exc}")

        # ─── Layer 1: Journal ─────────────────────────────
        print("\nLayer 1 · journal")
        md, js = journal.write_journal(d)
        print(f"  wrote {md.name}")
        print(f"  wrote {js.name}")
        # Sample today's PnL into state history
        monitor.sample_today_pnl_from_journal()

        # ─── Layer 2: Analyzer ────────────────────────────
        print("\nLayer 2 · analyzer")
        if prior_labels:
            print(f"  priors: {prior_labels}")
        rec_path = analyzer.write_recommendations(d, prior_labels=prior_labels or None)
        rec_data = json.loads(rec_path.read_text())
        n_recs = len(rec_data["recommendations"])
        print(f"  {n_recs} recommendation(s) generated")
        for r in rec_data["recommendations"]:
            if "error" in r:
                print(f"    ! {r['id']} errored: {r['error']}")
            elif r.get("kind") == "advisory":
                findings = r.get("findings")
                if findings:
                    detail = f"{len(findings)} structural findings"
                elif r.get("direction"):
                    detail = f"price-regime {r['direction']} signal"
                else:
                    detail = r.get("rule_id", "advisory")
                print(f"    ⚑ {r['id']}: ADVISORY ({r.get('prior_label') or r.get('date','?')}) — {detail}")
            else:
                print(f"    · {r['id']}  {r['param']}  {r.get('current')}→{r.get('proposed')}")
        if n_recs == 0:
            print("  (no recommendations — layers 3/4 will no-op)")

        # ─── Layer 3: Validator ───────────────────────────
        print("\nLayer 3 · validator")
        val_path = validator.validate_all(d)
        val_data = json.loads(val_path.read_text())
        greens = sum(1 for r in val_data["recommendations"]
                     if r.get("validation", {}).get("status") == "green")
        rejects = sum(1 for r in val_data["recommendations"]
                      if r.get("validation", {}).get("status") == "reject")
        errors = sum(1 for r in val_data["recommendations"]
                     if r.get("validation", {}).get("status") == "error")
        print(f"  green={greens}  reject={rejects}  error={errors}")

        if args.skip_apply:
            print("\n[skip-apply] stopping after validator as requested.")
            return 0

        # ─── Layer 4: Applier ─────────────────────────────
        print("\nLayer 4 · applier")
        applied_path = applier.apply_all(d, dry_run=args.dry_run)
        ap_data = json.loads(applied_path.read_text())
        for r in ap_data["results"]:
            param = r.get("param") or f"(advisory:{r.get('rule_id','?')})"
            print(f"  {r['status']:10s}  {param:45s}  {r['reason']}")

        # ─── Layer 5: Monitor ─────────────────────────────
        if not args.skip_monitor:
            print("\nLayer 5 · monitor (drift check + auto-revert)")
            m = monitor.run_monitor(auto_revert=not args.dry_run)
            print(f"  alerts={len(m['alerts'])}  reverts={len(m['reverts'])}")

    except Exception as exc:
        print(f"\n!!! ORCHESTRATOR ERROR: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        rc = 1

    dt = (datetime.utcnow() - t0).total_seconds()
    print(f"\n═══ done in {dt:.1f}s — rc={rc} ═══")
    return rc


if __name__ == "__main__":
    sys.exit(main())
