"""Extract per-minute price ticks from bot logs and (idempotently)
append them to a parquet store the analyzer can read.

Input line format (live + replay logs both use this):
    2026-04-15 21:43:29,475 [INFO] Bar: 2026-04-15 21:43:00 ET | Price: 7075.75

Also supports the heartbeat line as a secondary source:
    2026-04-15 21:43:28,144 [INFO] 💓 Heartbeat #1: 21:43:28 | ... | Price: 7075.75

Output: `ai_loop_data/live_prices.parquet`
    index: bar_ts (tz-aware America/New_York)
    cols:  price (float), source (str), log_name (str)

Design:
  - Load existing parquet if present; remember max bar_ts seen.
  - Walk the specified logs (default: topstep_live_bot.log + every
    replay dir under backtest_reports/full_live_replay/*/topstep_live_bot.log)
  - Keep only bars strictly newer than the max seen OR bars from logs we
    haven't seen before (tracked via a small manifest of hashed log paths).
  - Merge, dedupe by bar_ts (prefer live over replay when both available),
    and write back.

Usage:
    python3 -m tools.ai_loop.price_parquet_updater              # default scan
    python3 -m tools.ai_loop.price_parquet_updater --dry-run
    python3 -m tools.ai_loop.price_parquet_updater --log foo.log --log bar.log
    python3 -m tools.ai_loop.price_parquet_updater --rebuild    # ignore manifest

The analyzer + backtest_journal consume this via `price_context.load_prices()`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import DATA_DIR, ROOT

NY_TZ = "America/New_York"

PRICE_PARQUET = DATA_DIR / "live_prices.parquet"
MANIFEST_PATH = DATA_DIR / "live_prices_manifest.json"

# Live / replay logs:  "2026-04-15 21:43:29,... [INFO] Bar: 2026-04-15 21:43:00 ET | Price: 7075.75"
_RE_BAR = re.compile(
    r"\[INFO\]\s+Bar:\s+(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+ET\s+\|\s+"
    r"Price:\s+([0-9]+\.?[0-9]*)"
)
# Heartbeat (fallback — gives second-granularity ticks when bars are missing)
_RE_HB = re.compile(
    r"(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}),\d+\s+\[INFO\]\s+💓 Heartbeat.*"
    r"Price:\s+([0-9]+\.?[0-9]*)"
)


def _load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        try:
            return json.loads(MANIFEST_PATH.read_text())
        except Exception:
            pass
    return {"logs": {}}  # log_path_abs -> {size, mtime, bars_added, first_ts, last_ts}


def _save_manifest(m: dict) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(m, indent=2, default=str))


def _log_fingerprint(p: Path) -> tuple[int, float]:
    """(size, mtime) — if a log grew since last run we re-scan the tail."""
    st = p.stat()
    return st.st_size, st.st_mtime


def _default_logs() -> list[Path]:
    """Pick up live + every replay log we know about."""
    candidates: list[Path] = []
    live = ROOT / "topstep_live_bot.log"
    if live.exists():
        candidates.append(live)
    # replay dirs
    for p in (ROOT / "backtest_reports").glob("**/topstep_live_bot.log"):
        candidates.append(p)
    return candidates


def _iter_bar_rows(log_path: Path) -> Iterable[tuple[datetime, float, str]]:
    """Yield (bar_ts, price, 'bar') for every parseable Bar line, and
    (ts, price, 'hb') for fallback heartbeat lines on seconds the bar
    stream missed."""
    with log_path.open("r", errors="replace") as f:
        for line in f:
            if m := _RE_BAR.search(line):
                try:
                    ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
                    yield ts, float(m.group(2)), "bar"
                except (ValueError, TypeError):
                    continue
            elif m := _RE_HB.search(line):
                try:
                    ts = datetime.strptime(
                        f"{m.group(1)} {m.group(2)}", "%Y-%m-%d %H:%M:%S"
                    )
                    # Round heartbeat to the minute so it dedupes against bars
                    ts = ts.replace(second=0)
                    yield ts, float(m.group(3)), "hb"
                except (ValueError, TypeError):
                    continue


def _existing_max_ts() -> pd.Timestamp | None:
    if not PRICE_PARQUET.exists():
        return None
    try:
        df = pd.read_parquet(PRICE_PARQUET)
        if df.empty:
            return None
        return df.index.max()
    except Exception:
        return None


def _to_ny_index(ts_series: pd.Series) -> pd.DatetimeIndex:
    """Log timestamps are naive but all stamped 'ET'. Localize to NY tz."""
    idx = pd.to_datetime(ts_series)
    if idx.dt.tz is None:
        idx = idx.dt.tz_localize(NY_TZ, ambiguous="NaT", nonexistent="shift_forward")
    return pd.DatetimeIndex(idx)


def update_from_logs(
    logs: list[Path] | None = None,
    *,
    dry_run: bool = False,
    rebuild: bool = False,
    verbose: bool = True,
) -> dict:
    """Scan logs and (unless dry_run) persist an updated parquet.

    Returns a summary dict for the caller.
    """
    logs = logs if logs is not None else _default_logs()
    manifest = {"logs": {}} if rebuild else _load_manifest()

    all_rows: list[tuple[datetime, float, str, str]] = []
    per_log: list[dict] = []

    for log_path in logs:
        if not log_path.exists():
            continue
        key = str(log_path.resolve())
        size, mtime = _log_fingerprint(log_path)
        entry = manifest["logs"].get(key, {})
        # Skip if this log hasn't grown since we last scanned it.
        # (A file can only grow or be truncated; we don't handle truncation
        # beyond just re-scanning it from scratch.)
        if (not rebuild and entry.get("size") == size
                and entry.get("mtime") == mtime):
            if verbose:
                print(f"  [skip]   {log_path.name} — unchanged since last scan")
            continue

        rows = list(_iter_bar_rows(log_path))
        n_rows = len(rows)
        if verbose:
            print(f"  [scan]   {log_path.name} — {n_rows:,} price rows")
        log_name = log_path.parent.name if log_path.parent.name != "JULIE001" else "live"
        for ts, price, source in rows:
            all_rows.append((ts, price, source, log_name))

        if rows:
            manifest["logs"][key] = {
                "size": size,
                "mtime": mtime,
                "bars_added": n_rows,
                "first_ts": str(rows[0][0]),
                "last_ts": str(rows[-1][0]),
                "scanned_at": datetime.utcnow().isoformat() + "Z",
            }
            per_log.append({
                "log": log_name, "path": key, "rows": n_rows,
                "first": str(rows[0][0]), "last": str(rows[-1][0]),
            })

    if not all_rows:
        if verbose:
            print("  (no new rows to add)")
        return {
            "new_rows": 0, "total_rows": 0,
            "per_log": per_log, "parquet_path": str(PRICE_PARQUET),
        }

    new_df = pd.DataFrame(all_rows, columns=["bar_ts", "price", "source", "log_name"])
    new_df["bar_ts"] = _to_ny_index(new_df["bar_ts"])
    new_df = new_df.set_index("bar_ts").sort_index()

    # Merge with existing
    if PRICE_PARQUET.exists() and not rebuild:
        try:
            existing = pd.read_parquet(PRICE_PARQUET)
            combined = pd.concat([existing, new_df])
        except Exception:
            combined = new_df
    else:
        combined = new_df

    # Dedupe: prefer "bar" over "hb", and last-seen within source on ties.
    combined = combined.reset_index()
    combined["source_rank"] = combined["source"].map({"bar": 0, "hb": 1}).fillna(2)
    combined = combined.sort_values(["bar_ts", "source_rank"])
    combined = combined.drop_duplicates(subset=["bar_ts"], keep="first")
    combined = combined.drop(columns=["source_rank"])
    combined = combined.set_index("bar_ts").sort_index()

    if dry_run:
        if verbose:
            print(f"  [dry-run] would write {len(combined):,} rows "
                  f"(added {len(new_df):,} new) to {PRICE_PARQUET}")
        return {
            "new_rows": len(new_df),
            "total_rows": len(combined),
            "per_log": per_log,
            "parquet_path": str(PRICE_PARQUET),
            "dry_run": True,
        }

    PRICE_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(PRICE_PARQUET)
    _save_manifest(manifest)
    if verbose:
        print(f"  [write]   {PRICE_PARQUET}  rows={len(combined):,} "
              f"(added {len(new_df):,})")
    return {
        "new_rows": len(new_df),
        "total_rows": len(combined),
        "per_log": per_log,
        "parquet_path": str(PRICE_PARQUET),
        "date_range": (str(combined.index.min()), str(combined.index.max())),
    }


def main():
    ap = argparse.ArgumentParser(description="Append bot-log price bars to parquet")
    ap.add_argument("--log", action="append", default=None,
                    help="Override logs to scan (repeatable).")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rebuild", action="store_true",
                    help="Ignore manifest and re-scan every log.")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    logs = [Path(p) for p in args.log] if args.log else None
    r = update_from_logs(
        logs, dry_run=args.dry_run, rebuild=args.rebuild, verbose=not args.quiet
    )
    print(f"\n[price-parquet] new_rows={r['new_rows']:,}  total={r['total_rows']:,}")
    if not args.dry_run and r.get("date_range"):
        print(f"[price-parquet] range {r['date_range'][0]} → {r['date_range'][1]}")


if __name__ == "__main__":
    main()
