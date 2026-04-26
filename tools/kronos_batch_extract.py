"""Batch-extract Kronos forecast features for the v11 corpus.

For each candidate row:
  - Build 150-bar context from es_master_outrights.parquet (filtered to the
    walk_contract symbol, last 150 bars before candidate ts).
  - Send to a long-running Kronos daemon subprocess.
  - Parse response; collect 5 features.

Writes checkpoint to artifacts/v11_corpus_with_kronos_features.parquet
every CHECKPOINT_EVERY rows. Resumes from checkpoint if re-run.

Usage:
    .kronos_venv/bin/python3 tools/kronos_batch_extract.py [--max-rows N]
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
CORPUS = REPO / "artifacts" / "v11_corpus_with_bar_paths.parquet"
MASTER = REPO / "es_master_outrights.parquet"
OUT = REPO / "artifacts" / "v11_corpus_with_kronos_features.parquet"
LOG = REPO / "artifacts" / "kronos_batch_extract.log"

CONTEXT_BARS = 150
DAEMON_PATH = REPO / "tools" / "kronos_predict_features.py"
PYTHON = REPO / ".kronos_venv" / "bin" / "python3"
CHECKPOINT_EVERY = 50
DAEMON_RESTART_LIMIT = 5


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def spawn_daemon() -> subprocess.Popen:
    proc = subprocess.Popen(
        [str(PYTHON), str(DAEMON_PATH), "--daemon"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1, cwd=str(REPO),
    )
    # Wait for "loading" then "ready"
    loading = proc.stdout.readline().strip()
    log(f"daemon loading: {loading}")
    ready = proc.stdout.readline().strip()
    log(f"daemon ready:   {ready}")
    if "ready" not in ready:
        raise RuntimeError(f"daemon did not become ready: {ready}")
    return proc


def shutdown_daemon(proc: subprocess.Popen) -> None:
    try:
        proc.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
        proc.stdin.flush()
        proc.wait(timeout=10)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def load_master_by_symbol() -> Dict[str, pd.DataFrame]:
    log(f"loading master bars: {MASTER}")
    m = pd.read_parquet(MASTER)
    # Restrict to 2025-01-01+ for memory
    m = m[m.index >= pd.Timestamp("2025-01-01", tz="US/Eastern")]
    out: Dict[str, pd.DataFrame] = {}
    for sym in m["symbol"].unique():
        sub = m[m["symbol"] == sym][["open", "high", "low", "close", "volume"]].copy()
        sub.index = pd.to_datetime(sub.index, utc=True)
        out[str(sym)] = sub.sort_index()
    log(f"master split: {len(out)} symbols")
    for s, d in out.items():
        log(f"  {s}: {len(d)} bars  {d.index[0]} -> {d.index[-1]}")
    return out


def build_bars_payload(master_sym: pd.DataFrame, candidate_ts_utc: pd.Timestamp) -> Optional[List[Dict]]:
    """Return list of last CONTEXT_BARS bars BEFORE candidate_ts_utc."""
    sub = master_sym[master_sym.index < candidate_ts_utc]
    if len(sub) < 100:  # daemon's MIN_BARS
        return None
    sub = sub.iloc[-CONTEXT_BARS:]
    return [
        {
            "ts": ts.isoformat(),
            "open": float(row.open),
            "high": float(row.high),
            "low": float(row.low),
            "close": float(row.close),
            "volume": float(row.volume),
        }
        for ts, row in sub.iterrows()
    ]


def main():
    args = sys.argv[1:]
    max_rows: Optional[int] = None
    if "--max-rows" in args:
        i = args.index("--max-rows")
        max_rows = int(args[i + 1])

    log("=" * 80)
    log("Kronos batch extraction for v11 corpus")
    log("=" * 80)

    corpus = pd.read_parquet(CORPUS).sort_values("ts").reset_index(drop=True)
    log(f"corpus rows: {len(corpus)}")

    # Resume from checkpoint if present
    if OUT.exists():
        prior = pd.read_parquet(OUT)
        already_done = set(prior["row_idx"].tolist())
        log(f"checkpoint found: {len(prior)} rows already complete")
    else:
        already_done = set()

    # Pre-split master by symbol
    master = load_master_by_symbol()

    if max_rows is not None:
        corpus = corpus.head(max_rows).copy()
        log(f"--max-rows={max_rows} active")

    # Convert candidate ts to UTC for matching the master index
    corpus["ts_utc"] = pd.to_datetime(corpus["ts"]).dt.tz_convert("UTC")

    # Spawn daemon
    daemon = spawn_daemon()
    daemon_restarts = 0

    results: List[Dict] = []
    if OUT.exists():
        # Carry forward the existing checkpoint rows
        prior = pd.read_parquet(OUT)
        for _, r in prior.iterrows():
            results.append(r.to_dict())

    t_batch_start = time.time()
    n_processed = 0

    for idx, row in corpus.iterrows():
        if idx in already_done:
            continue
        sym = row["walk_contract"]
        if sym not in master:
            results.append({
                "row_idx": int(idx),
                "ts": row["ts"],
                "walk_contract": sym,
                "kronos_failed": f"unknown_symbol:{sym}",
                "kronos_max_high_above": None,
                "kronos_min_low_below": None,
                "kronos_pred_atr_30bar": None,
                "kronos_dir_move": None,
                "kronos_close_vs_entry": None,
                "kronos_inf_time_s": None,
            })
            n_processed += 1
            continue

        bars = build_bars_payload(master[sym], row["ts_utc"])
        if bars is None:
            results.append({
                "row_idx": int(idx),
                "ts": row["ts"],
                "walk_contract": sym,
                "kronos_failed": "insufficient_bars",
                "kronos_max_high_above": None,
                "kronos_min_low_below": None,
                "kronos_pred_atr_30bar": None,
                "kronos_dir_move": None,
                "kronos_close_vs_entry": None,
                "kronos_inf_time_s": None,
            })
            n_processed += 1
            continue

        req = {"bars": bars, "entry_price": float(row["entry_price"])}
        try:
            daemon.stdin.write(json.dumps(req) + "\n")
            daemon.stdin.flush()
            line = daemon.stdout.readline().strip()
        except (BrokenPipeError, OSError) as e:
            log(f"daemon write/read failed at row {idx}: {e}")
            line = ""

        if not line:
            # daemon dead, restart
            log(f"empty response at row {idx}, restarting daemon (restart {daemon_restarts + 1}/{DAEMON_RESTART_LIMIT})")
            try:
                daemon.kill()
            except Exception:
                pass
            daemon_restarts += 1
            if daemon_restarts > DAEMON_RESTART_LIMIT:
                log(f"daemon restart limit exceeded. saving checkpoint and exiting.")
                break
            daemon = spawn_daemon()
            # Retry the same row once
            try:
                daemon.stdin.write(json.dumps(req) + "\n")
                daemon.stdin.flush()
                line = daemon.stdout.readline().strip()
            except Exception as e:
                log(f"daemon retry failed at row {idx}: {e}")
                line = ""

        try:
            resp = json.loads(line) if line else {"kronos_failed": "empty_line"}
        except Exception as e:
            resp = {"kronos_failed": f"bad_json:{str(e)[:100]}"}

        rec = {
            "row_idx": int(idx),
            "ts": row["ts"],
            "walk_contract": sym,
            "kronos_failed": resp.get("kronos_failed"),
            "kronos_max_high_above": resp.get("kronos_max_high_above"),
            "kronos_min_low_below": resp.get("kronos_min_low_below"),
            "kronos_pred_atr_30bar": resp.get("kronos_pred_atr_30bar"),
            "kronos_dir_move": resp.get("kronos_dir_move"),
            "kronos_close_vs_entry": resp.get("kronos_close_vs_entry"),
            "kronos_inf_time_s": resp.get("kronos_inf_time_s"),
        }
        results.append(rec)
        n_processed += 1

        if n_processed % CHECKPOINT_EVERY == 0:
            df_ckpt = pd.DataFrame(results)
            df_ckpt.to_parquet(OUT, index=False)
            elapsed = time.time() - t_batch_start
            rate = n_processed / max(1.0, elapsed)
            remaining = len(corpus) - len(set(r["row_idx"] for r in results))
            eta_s = remaining / max(0.001, rate)
            log(
                f"checkpoint @ {len(results)} ({n_processed} this run) — "
                f"rate={rate:.2f}/s eta={eta_s/60:.1f}min "
                f"failed={sum(1 for r in results if r.get('kronos_failed'))}"
            )

    shutdown_daemon(daemon)

    # Final checkpoint
    df_final = pd.DataFrame(results)
    df_final.to_parquet(OUT, index=False)
    log("=" * 80)
    log(f"COMPLETE: {len(df_final)} total rows in checkpoint")
    failed = df_final[df_final["kronos_failed"].notna()]
    log(f"failed:   {len(failed)} rows")
    if len(failed) > 0:
        log(f"failure reasons: {failed['kronos_failed'].value_counts().to_dict()}")
    succeeded = df_final[df_final["kronos_failed"].isna()]
    if len(succeeded) > 0:
        log(f"avg inf time: {succeeded['kronos_inf_time_s'].mean():.3f}s")
        log(f"feature stats:")
        for c in ["kronos_max_high_above", "kronos_min_low_below",
                  "kronos_pred_atr_30bar", "kronos_dir_move", "kronos_close_vs_entry"]:
            log(f"  {c}: mean={succeeded[c].mean():.3f} std={succeeded[c].std():.3f}")


if __name__ == "__main__":
    main()
