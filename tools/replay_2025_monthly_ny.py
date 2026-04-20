#!/usr/bin/env python3
"""Run full-v4 NY-only replay across 2025 in monthly chunks.

Wraps run_full_live_replay_parquet.py with JULIE_REPLAY_NY_ONLY=1 and
the full v4 env stack, writing results to backtest_reports/2025_ny_<tag>/.

Usage:
  python3 tools/replay_2025_monthly_ny.py [--months 1,2,3] [--tag v1]

Env overrides passed to each sub-process:
  JULIE_KALSHI_MIN_TP_MULT, JULIE_KALSHI_MAX_TP_MULT, JULIE_REVERSAL_CONFIRM,
  JULIE_KALSHI_BLOCK_BUF_BALANCED, JULIE_KALSHI_BLOCK_BUF_FP,
  JULIE_PCT_TRAIL_TIGHT, JULIE_PCT_TRAIL_EXT
"""
from __future__ import annotations

import argparse
import calendar
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BARS_PARQUET = "/Users/wes/Downloads/es_master_outrights.parquet"
REPORT_DIR = ROOT / "backtest_reports"

MONTH_TO_SYMBOL: Dict[int, str] = {
    1: "MESH5", 2: "MESH5", 3: "MESM5", 4: "MESM5",
    5: "MESM5", 6: "MESU5", 7: "MESU5", 8: "MESU5",
    9: "MESZ5", 10: "MESZ5", 11: "MESZ5", 12: "MESH6",
}

# Env vars to forward from parent process to sub-processes
_FORWARD_ENV_VARS = [
    "JULIE_KALSHI_MIN_TP_MULT",
    "JULIE_KALSHI_MAX_TP_MULT",
    "JULIE_REVERSAL_CONFIRM",
    "JULIE_REVERSAL_WINDOW",
    "JULIE_KALSHI_BLOCK_BUF_BALANCED",
    "JULIE_KALSHI_BLOCK_BUF_FP",
    "JULIE_PCT_TRAIL_TIGHT",
    "JULIE_PCT_TRAIL_EXT",
]


def _month_bounds(year: int, month: int) -> Tuple[str, str]:
    last = calendar.monthrange(year, month)[1]
    return f"{year:04d}-{month:02d}-01 00:00", f"{year:04d}-{month:02d}-{last:02d} 23:59"


def _run_one(*, year: int, month: int, account_id: int, bars_parquet: str,
             initial_balance: float, tag: str) -> Dict[str, Any]:
    start, end = _month_bounds(year, month)
    run_name = f"{year:04d}_{month:02d}_ny_{tag}"
    log_path = REPORT_DIR / f"{run_name}.stdout.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-u",
        str(ROOT / "tools" / "run_full_live_replay_parquet.py"),
        "--start", start, "--end", end,
        "--bars-parquet", str(bars_parquet),
        "--target-symbol", MONTH_TO_SYMBOL[month],
        "--run-name", run_name,
        "--account-id", str(account_id),
        "--initial-balance", f"{initial_balance:.2f}",
        "--lookback-minutes", "20000",
    ]
    env = dict(os.environ)
    env["JULIE_REPLAY_NY_ONLY"] = "1"
    env["JULIE_CB"] = "1"
    env["JULIE_DLB"] = "1"
    env["JULIE_DD_SCALE"] = "1"
    env["JULIE_KALSHI_CONTINUATION_TP"] = "1"
    t0 = time.time()
    with log_path.open("w") as fh:
        res = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                             cwd=str(ROOT), env=env)
    elapsed = time.time() - t0
    summary_path = REPORT_DIR / "full_live_replay" / run_name / "live_replay_summary.json"
    if not summary_path.exists():
        summary_path = REPORT_DIR / run_name / "live_replay_summary.json"
    summary: Dict[str, Any] = {}
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
        except Exception:
            summary = {"error": "summary_parse_failed"}
    return {
        "year": year, "month": month, "tag": tag,
        "returncode": res.returncode, "elapsed_s": round(elapsed, 1),
        "log": str(log_path), "summary": summary,
    }


def _discover_account_id() -> int:
    env = os.environ.get("REPLAY_ACCOUNT_ID", "").strip()
    if env:
        return int(env)
    state = ROOT / "bot_state.json"
    if state.exists():
        try:
            data = json.loads(state.read_text())
            acct = data.get("live_drawdown", {}).get("account_id")
            if acct is not None:
                return int(acct)
        except Exception:
            pass
    raise RuntimeError("Set REPLAY_ACCOUNT_ID or ensure bot_state.json has live_drawdown.account_id")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--initial-balance", type=float, default=50_000.0)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("REPLAY_WORKERS", "4")))
    parser.add_argument("--months", default=os.environ.get("REPLAY_MONTHS", ""),
                        help="CSV of months (default: all 12)")
    parser.add_argument("--bars-parquet", default=os.environ.get("REPLAY_BARS_PARQUET", DEFAULT_BARS_PARQUET))
    parser.add_argument("--tag", default="v1", help="Run-name tag (e.g. v1, notp, rev5)")
    args = parser.parse_args()

    account_id = _discover_account_id()
    months: List[int] = [int(x) for x in args.months.split(",") if x.strip()] if args.months.strip() else list(range(1, 13))
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Launching {len(months)} monthly NY-only replays (workers={args.workers} tag={args.tag} acct={account_id})", flush=True)
    t0 = time.time()
    results: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(_run_one, year=args.year, month=m, account_id=account_id,
                      bars_parquet=args.bars_parquet, initial_balance=args.initial_balance,
                      tag=args.tag): m
            for m in months
        }
        for fut in as_completed(futures):
            m = futures[fut]
            try:
                res = fut.result()
            except Exception as exc:
                res = {"year": args.year, "month": m, "tag": args.tag, "error": str(exc)}
            results.append(res)
            s = res.get("summary", {})
            pnl = s.get("net_pnl")
            wr = s.get("winrate")
            dd = s.get("max_drawdown")
            trades = s.get("closed_trades")
            rc = res.get("returncode", "?")
            elapsed = time.time() - t0
            status = "✅" if (pnl or 0) > 0 and (wr or 0) > 50 and (dd or 9999) <= 1400 else "❌"
            print(
                f"  {status} [{len(results)}/{len(months)}] month={m:02d} rc={rc} "
                f"pnl=${pnl:.0f} wr={wr:.1f}% dd=${dd:.0f} trades={trades} "
                f"elapsed={elapsed:.0f}s",
                flush=True,
            )

    results.sort(key=lambda r: r.get("month", 0))
    totals = {"net_pnl": 0.0, "closed_trades": 0, "wins": 0, "losses": 0}
    all_pass = True
    for r in results:
        s = r.get("summary", {}) or {}
        for k in totals:
            v = s.get(k)
            if isinstance(v, (int, float)):
                totals[k] += v
        pnl = s.get("net_pnl", 0) or 0
        wr = s.get("winrate", 0) or 0
        dd = s.get("max_drawdown", 9999) or 9999
        if pnl <= 0 or wr <= 50 or dd > 1400:
            all_pass = False

    totals["winrate"] = totals["wins"] / totals["closed_trades"] if totals["closed_trades"] else 0.0
    out = {"year": args.year, "tag": args.tag, "workers": args.workers, "account_id": account_id,
           "elapsed_s": round(time.time() - t0, 1), "totals": totals, "months": results}
    summary_path = REPORT_DIR / f"2025_ny_{args.tag}_summary.json"
    summary_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {summary_path}", flush=True)
    print(
        f"Totals: pnl=${totals['net_pnl']:.0f} trades={totals['closed_trades']} "
        f"wr={totals['winrate']:.2%} {'ALL PASS ✅' if all_pass else 'SOME FAIL ❌'}",
        flush=True,
    )
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
