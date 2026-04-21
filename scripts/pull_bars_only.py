#!/usr/bin/env python3
"""Pull ProjectX 1-minute bars for a date range — no strategies, no Kalshi.

Writes bars into a synthetic topstep_live_bot.log in the given report dir so
downstream code (extract_bars, classifier) can read it unchanged.

Usage:
    python3 scripts/pull_bars_only.py \\
        --start "2026-03-01 00:00" --end "2026-03-31 23:59" \\
        --report-dir backtest_reports/bars_mar2026
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import pandas as pd
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from client import ProjectXClient
from config import CONFIG, determine_current_contract_symbol
from bot_state import load_bot_state

NY = ZoneInfo("America/New_York")
UTC = dt.timezone.utc


def parse_dt(s: str) -> dt.datetime:
    d = dt.datetime.strptime(s, "%Y-%m-%d %H:%M")
    return d.replace(tzinfo=NY)


def resolve_account_id(client: ProjectXClient):
    state = load_bot_state(ROOT / "bot_state.json")
    if isinstance(state, dict):
        dd = state.get("live_drawdown", {})
        if isinstance(dd, dict) and dd.get("account_id"):
            client.account_id = int(dd["account_id"])
            return
    cfg_acct = CONFIG.get("ACCOUNT_ID")
    if cfg_acct:
        client.account_id = int(cfg_acct)


def fetch_window(client: ProjectXClient, start: dt.datetime, end: dt.datetime, contract_id: str):
    """Pull 1-min bars from [start, end) using /api/History/retrieveBars."""
    url = f"{client.base_url}/api/History/retrieveBars"
    payload = {
        "accountId": client.account_id,
        "contractId": contract_id,
        "live": False,
        "limit": 20000,
        "startTime": start.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "endTime": end.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "unit": 2,  # 2 == minute
        "unitNumber": 1,
        "includePartialBar": False,
    }
    resp = client.session.post(url, json=payload)
    client._track_general_request()
    resp.raise_for_status()
    data = resp.json()
    bars = data.get("bars", [])
    return bars


def bars_to_log_lines(bars: list[dict]) -> list[str]:
    lines = []
    for b in bars:
        # Bars come as UTC ISO strings; convert to NY time for the log format
        ts_utc = dt.datetime.fromisoformat(b["t"].replace("Z", "+00:00"))
        ts_ny = ts_utc.astimezone(NY)
        stamp = ts_ny.strftime("%Y-%m-%d %H:%M:%S")
        price = float(b["c"])
        lines.append(
            f"{stamp},000 [INFO] Bar: {stamp} ET | Price: {price:.2f}\n"
        )
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="ET start, e.g. '2026-03-01 00:00'")
    ap.add_argument("--end", required=True, help="ET end, e.g. '2026-03-31 23:59'")
    ap.add_argument("--contract-root", default="MES")
    ap.add_argument("--report-dir", required=True)
    args = ap.parse_args()

    start = parse_dt(args.start)
    end = parse_dt(args.end)
    print(f"[pull] bars {start} -> {end}")

    report_dir = ROOT / args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)

    client = ProjectXClient(contract_root=args.contract_root)
    client.login()
    resolve_account_id(client)
    CONFIG["CONTRACT_ROOT"] = args.contract_root
    CONFIG["TARGET_SYMBOL"] = determine_current_contract_symbol(args.contract_root)
    contract_id = client.fetch_contracts()
    if not contract_id:
        print("ERROR: could not resolve contract")
        sys.exit(1)
    print(f"[pull] contract_id={contract_id}")

    # Pull in chunks (20k bars = ~14 days)
    all_bars = []
    cursor = start
    while cursor < end:
        window_end = min(cursor + dt.timedelta(days=14), end)
        print(f"[pull] window {cursor} -> {window_end}")
        try:
            bars = fetch_window(client, cursor, window_end, contract_id)
        except Exception as e:
            print(f"  error: {e}")
            break
        print(f"  got {len(bars)} bars")
        all_bars.extend(bars)
        if len(bars) < 20_000:
            cursor = window_end
        else:
            # Advance by last bar's timestamp
            last_ts = dt.datetime.fromisoformat(bars[-1]["t"].replace("Z", "+00:00"))
            cursor = last_ts.astimezone(NY) + dt.timedelta(minutes=1)

    print(f"[pull] total bars: {len(all_bars)}")
    # Dedupe + sort
    seen = set()
    uniq = []
    for b in all_bars:
        key = b["t"]
        if key in seen:
            continue
        seen.add(key)
        uniq.append(b)
    uniq.sort(key=lambda x: x["t"])
    print(f"[pull] unique bars: {len(uniq)}")

    # Write synthetic log
    log_path = report_dir / "topstep_live_bot.log"
    with log_path.open("w", encoding="utf-8") as fh:
        fh.writelines(bars_to_log_lines(uniq))
    print(f"[write] {log_path}  ({log_path.stat().st_size:,} bytes)")

    # Also an empty closed_trades.json so folder looks like a full replay dir
    (report_dir / "closed_trades.json").write_text("[]", encoding="utf-8")


if __name__ == "__main__":
    main()
