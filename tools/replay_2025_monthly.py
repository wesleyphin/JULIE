#!/usr/bin/env python3
"""Run the full live-loop replay across all of 2025 in 12 monthly chunks in parallel.

For each month we spawn a subprocess calling
`tools/run_full_live_replay.py --bars-parquet ... --run-name 2025_MM`.
Results are written to `backtest_reports/full_live_replay/2025_MM/` and then
aggregated into `backtest_reports/full_live_replay/2025_summary.json`.

Why 12:
- 2025 = ~252 trading days; serial replay would take hours
- 12 months ≈ 20-22 trading days each — CPU-balanced chunks
- 4-way parallelism keeps memory under control while still cutting wall time

Env overrides:
- REPLAY_WORKERS: max concurrent subprocesses (default 4)
- REPLAY_ACCOUNT_ID: account id (default from bot_state.json)
- REPLAY_BARS_PARQUET: override ES parquet path
- REPLAY_MONTHS: CSV of months (e.g. "3,4,5") to run only a subset
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
REPORT_DIR = ROOT / "backtest_reports" / "full_live_replay"

# MES front-month symbol dominant in each month of 2025 (per daily volume).
# The replay loader picks per-day max-volume symbol inside the ES parquet, so
# this label is mostly cosmetic — it populates CONFIG["TARGET_SYMBOL"].
MONTH_TO_SYMBOL: Dict[int, str] = {
    1: "MESH5",
    2: "MESH5",
    3: "MESM5",
    4: "MESM5",
    5: "MESM5",
    6: "MESU5",
    7: "MESU5",
    8: "MESU5",
    9: "MESZ5",
    10: "MESZ5",
    11: "MESZ5",
    12: "MESH6",
}


def _month_bounds(year: int, month: int) -> Tuple[str, str]:
    last = calendar.monthrange(year, month)[1]
    return f"{year:04d}-{month:02d}-01 00:00", f"{year:04d}-{month:02d}-{last:02d} 23:59"


def _run_one(
    *,
    year: int,
    month: int,
    account_id: int,
    bars_parquet: str,
    initial_balance: float,
) -> Dict[str, Any]:
    start, end = _month_bounds(year, month)
    run_name = f"{year:04d}_{month:02d}"
    log_path = REPORT_DIR / f"{run_name}.stdout.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-u",
        str(ROOT / "tools" / "run_full_live_replay.py"),
        "--start", start,
        "--end", end,
        "--bars-parquet", str(bars_parquet),
        "--target-symbol", MONTH_TO_SYMBOL[month],
        "--run-name", run_name,
        "--account-id", str(account_id),
        "--initial-balance", f"{initial_balance:.2f}",
        "--lookback-minutes", "20000",
    ]
    t0 = time.time()
    with log_path.open("w") as fh:
        res = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(ROOT))
    elapsed = time.time() - t0
    summary_path = REPORT_DIR / run_name / "live_replay_summary.json"
    summary: Dict[str, Any] = {}
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
        except Exception:
            summary = {"error": "summary_parse_failed"}
    return {
        "year": year,
        "month": month,
        "returncode": res.returncode,
        "elapsed_s": round(elapsed, 1),
        "log_path": str(log_path),
        "summary": summary,
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
    parser.add_argument(
        "--months",
        default=os.environ.get("REPLAY_MONTHS", ""),
        help="CSV of months to run (default: all 12)",
    )
    parser.add_argument(
        "--bars-parquet",
        default=os.environ.get("REPLAY_BARS_PARQUET", DEFAULT_BARS_PARQUET),
    )
    args = parser.parse_args()

    account_id = _discover_account_id()
    months: List[int]
    if args.months.strip():
        months = [int(x) for x in args.months.split(",") if x.strip()]
    else:
        months = list(range(1, 13))
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    aggregate_path = REPORT_DIR / f"{args.year}_summary.json"
    results: List[Dict[str, Any]] = []

    print(
        f"Launching {len(months)} monthly replays with {args.workers} workers "
        f"(account={account_id}, parquet={args.bars_parquet})",
        flush=True,
    )
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(
                _run_one,
                year=args.year,
                month=m,
                account_id=account_id,
                bars_parquet=args.bars_parquet,
                initial_balance=args.initial_balance,
            ): m
            for m in months
        }
        for fut in as_completed(futures):
            m = futures[fut]
            try:
                res = fut.result()
            except Exception as exc:
                res = {"year": args.year, "month": m, "error": str(exc)}
            results.append(res)
            pnl = res.get("summary", {}).get("net_pnl")
            trades = res.get("summary", {}).get("closed_trades")
            rc = res.get("returncode", "?")
            elapsed_total = time.time() - t0
            print(
                f"  [{len(results)}/{len(months)}] month={m:02d} rc={rc} "
                f"elapsed_total={elapsed_total:.0f}s pnl={pnl} trades={trades}",
                flush=True,
            )

    results.sort(key=lambda r: r.get("month", 0))
    totals = {
        "net_pnl": 0.0,
        "closed_trades": 0,
        "wins": 0,
        "losses": 0,
    }
    for r in results:
        s = r.get("summary", {}) or {}
        for k in totals:
            v = s.get(k)
            if isinstance(v, (int, float)):
                totals[k] += v
    totals["winrate"] = (
        totals["wins"] / totals["closed_trades"] if totals["closed_trades"] else 0.0
    )
    out = {
        "year": args.year,
        "workers": args.workers,
        "account_id": account_id,
        "bars_parquet": args.bars_parquet,
        "elapsed_s": round(time.time() - t0, 1),
        "totals": totals,
        "months": results,
    }
    aggregate_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"Wrote {aggregate_path}", flush=True)
    print(
        f"Totals: pnl=${totals['net_pnl']:.2f}  trades={totals['closed_trades']} "
        f"wins={totals['wins']} losses={totals['losses']} "
        f"winrate={totals['winrate']:.2%}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
