#!/usr/bin/env python3
"""Tick chart poller — pulls 15-second OHLC bars from Topstep REST every 15
seconds and writes them to artifacts/tick_chart_ohlc.json for the live
dashboard chart.

Runs as a SIDECAR (separate process). No bot changes required. Safe to
start/stop at any time — the bot is unaffected.

Usage:
    python3 tools/tick_chart_poller.py                  # default: poll every 15s
    JULIE_TICK_POLL_INTERVAL=10 python3 tools/tick_chart_poller.py  # custom interval

Output:
    artifacts/tick_chart_ohlc.json  — list of {t, o, h, l, c, v} for the
                                       last ~60 minutes of 15-sec bars.

Stop:
    Ctrl-C, or `pkill -f tick_chart_poller`
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo

# Make the JULIE root importable so we can reuse the existing ProjectX client.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from client import ProjectXClient
from config import CONFIG

NY_TZ = ZoneInfo("America/New_York")
OUTPUT_PATH = ROOT / "artifacts" / "tick_chart_ohlc.json"
POLL_INTERVAL_S = int(os.environ.get("JULIE_TICK_POLL_INTERVAL", "15"))
HISTORY_MINUTES = int(os.environ.get("JULIE_TICK_POLL_LOOKBACK_MIN", "60"))
MAX_BARS = (HISTORY_MINUTES * 60) // 15 + 4  # rolling buffer
SHUTDOWN = False


def _on_shutdown(*_args):
    global SHUTDOWN
    SHUTDOWN = True


def _format_time(dt: datetime.datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    return dt.astimezone(NY_TZ).isoformat()


def _fetch_15sec_bars(client: ProjectXClient, lookback_minutes: int) -> list[dict] | None:
    """POST /api/History/retrieveBars with unit=1 (Seconds), unitNumber=15.
    Returns parsed bars list (newest last) or None on error."""
    if client.contract_id is None:
        logging.error("contract_id is None — call fetch_contracts() first")
        return None
    end_time = datetime.datetime.now(datetime.timezone.utc)
    start_time = end_time - datetime.timedelta(minutes=lookback_minutes)
    url = f"{client.base_url}/api/History/retrieveBars"
    payload = {
        "accountId": client.account_id,
        "contractId": client.contract_id,
        "live": False,
        "limit": MAX_BARS,
        "startTime": start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "endTime": end_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "unit": 1,
        "unitNumber": 15,
    }
    headers = {"Cache-Control": "no-cache, no-store, must-revalidate"}
    try:
        resp = client.session.post(url, json=payload, headers=headers, timeout=10)
    except Exception as exc:
        logging.warning("poll request error: %s", exc)
        return None
    if resp.status_code == 429:
        logging.warning("rate limited (429); skipping this poll")
        return None
    if resp.status_code == 401:
        logging.warning("auth expired (401); attempting re-login")
        try:
            client.login()
        except Exception as exc:
            logging.warning("re-login failed: %s", exc)
        return None
    if resp.status_code != 200:
        logging.warning("unexpected HTTP %d: %s", resp.status_code, resp.text[:200])
        return None
    try:
        data = resp.json() or {}
    except Exception as exc:
        logging.warning("JSON parse error: %s", exc)
        return None
    raw = data.get("bars") or data.get("data") or []
    if not isinstance(raw, list):
        logging.warning("unexpected payload shape: %r", type(raw))
        return None
    bars: list[dict] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        t_raw = entry.get("t") or entry.get("timestamp") or entry.get("time")
        if t_raw is None:
            continue
        try:
            t_dt = datetime.datetime.fromisoformat(str(t_raw).replace("Z", "+00:00"))
        except Exception:
            continue
        try:
            bar = {
                "t": _format_time(t_dt),
                "o": float(entry.get("o") if entry.get("o") is not None else entry.get("open", 0.0)),
                "h": float(entry.get("h") if entry.get("h") is not None else entry.get("high", 0.0)),
                "l": float(entry.get("l") if entry.get("l") is not None else entry.get("low", 0.0)),
                "c": float(entry.get("c") if entry.get("c") is not None else entry.get("close", 0.0)),
                "v": float(entry.get("v") if entry.get("v") is not None else entry.get("volume", 0.0)),
            }
        except Exception:
            continue
        bars.append(bar)
    bars.sort(key=lambda b: b["t"])
    return bars


def _write_json_atomic(path: Path, payload: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    text = json.dumps(payload, separators=(",", ":"))
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [tick_chart_poller] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    signal.signal(signal.SIGINT, _on_shutdown)
    signal.signal(signal.SIGTERM, _on_shutdown)

    logging.info("starting; output=%s interval=%ds lookback=%dmin",
                 OUTPUT_PATH, POLL_INTERVAL_S, HISTORY_MINUTES)

    # Resolve account ID from env BEFORE the client tries to prompt.
    env_account_id = os.environ.get("JULIE_ACCOUNT_ID")
    if env_account_id:
        try:
            CONFIG["ACCOUNT_ID"] = int(env_account_id)
        except (TypeError, ValueError):
            CONFIG["ACCOUNT_ID"] = env_account_id

    client = ProjectXClient()
    try:
        client.login()
    except Exception as exc:
        logging.error("login failed: %s", exc)
        return 2
    if client.account_id is None:
        if not CONFIG.get("ACCOUNT_ID"):
            logging.error(
                "no JULIE_ACCOUNT_ID env var set and no CONFIG['ACCOUNT_ID']; "
                "set JULIE_ACCOUNT_ID=<account_id> before running this poller"
            )
            return 3
        try:
            client.fetch_accounts()
        except Exception as exc:
            logging.error("fetch_accounts failed: %s", exc)
            return 3
    if client.contract_id is None:
        try:
            client.fetch_contracts()
        except Exception as exc:
            logging.error("fetch_contracts failed: %s", exc)
            return 4
    logging.info("authenticated; account=%s contract=%s", client.account_id, client.contract_id)

    last_log_count = -1
    while not SHUTDOWN:
        try:
            bars = _fetch_15sec_bars(client, HISTORY_MINUTES)
            if bars is not None:
                _write_json_atomic(OUTPUT_PATH, bars)
                if len(bars) != last_log_count:
                    logging.info("wrote %d bars (last close=%s)",
                                 len(bars),
                                 bars[-1]["c"] if bars else None)
                    last_log_count = len(bars)
        except Exception as exc:
            logging.warning("poll iteration error: %s", exc)
        # Sleep in small chunks so SIGINT is responsive
        for _ in range(POLL_INTERVAL_S):
            if SHUTDOWN:
                break
            time.sleep(1)
    logging.info("shutdown complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
