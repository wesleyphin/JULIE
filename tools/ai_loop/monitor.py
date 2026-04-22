"""Layer 5 — Live-vs-backtest drift monitor + auto-revert.

After changes have been AUTO_APPLIED, we expect live PnL to track the
backtest forecast. If live under-delivers significantly over several
sessions, the monitor:
  1. Flips state to frozen
  2. Reverts the most recent AUTO_APPLIED commit via `git revert`
  3. Writes an audit-log entry

Runs nightly as part of the orchestrator, AFTER the journal writer
(so we have fresh live numbers to compare against).

Drift detection (simple version):
  - For each AUTO_APPLIED change in the last 30 days:
      - Expected: candidate_pnl from validator (scaled to per-session)
      - Actual:   realized PnL in the live sessions since the change
      - If actual < expected − LIVE_VS_BACKTEST_SIGMA × σ for LIVE_VS_BACKTEST_WINDOW
        consecutive sessions → trip revert
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import date, datetime, timedelta
from pathlib import Path
from statistics import mean, pstdev

from . import state
from .config import (
    ROOT, AUDIT_LOG, JOURNALS_DIR, APPLIED_DIR,
    LIVE_VS_BACKTEST_SIGMA, LIVE_VS_BACKTEST_WINDOW,
)


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(ROOT), *args],
                          capture_output=True, text=True, check=True)


def _audit(record: dict) -> None:
    AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with AUDIT_LOG.open("a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def _load_recent_live_pnl(days: int = 10) -> list[dict]:
    """Pull the last N days of realized daily PnL from journals."""
    out = []
    d = date.today()
    for i in range(days * 2):  # walk back up to 2× for weekends
        p = JOURNALS_DIR / f"{d.isoformat()}.json"
        if p.exists():
            try:
                data = json.loads(p.read_text())
                out.append({"date": d.isoformat(),
                            "pnl": data["summary"]["total_pnl"],
                            "dd": data["summary"].get("max_drawdown", 0)})
            except Exception: pass
        d = d - timedelta(days=1)
        if len(out) >= days: break
    return list(reversed(out))


def _recent_applied_changes(days: int = 30) -> list[dict]:
    """Return AUTO_APPLIED results from the past N days."""
    out = []
    d = date.today()
    for i in range(days + 1):
        p = APPLIED_DIR / f"{d.isoformat()}.json"
        if p.exists():
            try:
                data = json.loads(p.read_text())
                for r in data.get("results", []):
                    if r.get("status") == "applied":
                        r["applied_date"] = d.isoformat()
                        out.append(r)
            except Exception: pass
        d = d - timedelta(days=1)
    return out


def _post_change_pnl(change_date: str) -> list[dict]:
    """Return daily PnL samples AFTER the given change date."""
    cutoff = datetime.fromisoformat(change_date).date()
    return [s for s in _load_recent_live_pnl(30)
            if datetime.fromisoformat(s["date"]).date() > cutoff]


def check_drift() -> list[dict]:
    """Return list of alerts (empty if no drift)."""
    alerts = []
    changes = _recent_applied_changes(30)
    for ch in changes:
        post = _post_change_pnl(ch["applied_date"])
        if len(post) < LIVE_VS_BACKTEST_WINDOW:
            continue
        pnls = [s["pnl"] for s in post[-LIVE_VS_BACKTEST_WINDOW:]]
        live_mean = mean(pnls)
        live_std = pstdev(pnls) if len(pnls) > 1 else abs(live_mean) * 0.5 + 50
        # Derive expected from validator's candidate PnL (approximate
        # per-session = total / n_days; baseline data varies).
        # For v0: if live_mean is > 2 sigma below 0 (sustained losses
        # since change), flag. Better calibration comes after we have
        # a few real applies to measure.
        if live_mean + LIVE_VS_BACKTEST_SIGMA * live_std < 0:
            alerts.append({
                "change_date": ch["applied_date"],
                "param": ch["param"],
                "commit": ch["commit_sha"],
                "live_pnl_mean_post": round(live_mean, 2),
                "live_pnl_std_post": round(live_std, 2),
                "reason": f"live PnL {live_mean:.2f} ± {live_std:.2f} across "
                          f"{LIVE_VS_BACKTEST_WINDOW} sessions after change — "
                          f"sustained negative drift",
            })
    return alerts


def revert_commit(sha: str, reason: str) -> bool:
    try:
        msg = f"[AUTO_REVERTED] revert {sha[:8]}\n\nTriggered by monitor: {reason}"
        _git("revert", "--no-edit", sha)
        # Amend the revert commit to add our reason
        _git("commit", "--amend", "-m", msg)
        return True
    except subprocess.CalledProcessError as exc:
        _audit({"event": "revert_failed", "sha": sha,
                "err": exc.stderr or exc.stdout})
        return False


def run_monitor(auto_revert: bool = True) -> dict:
    alerts = check_drift()
    revert_results = []
    if alerts and auto_revert:
        # Revert the MOST RECENT applied change per alert
        for a in alerts:
            ok = revert_commit(a["commit"], a["reason"])
            revert_results.append({**a, "reverted": ok})
            if ok:
                state.set_frozen(f"drift revert: {a['param']} ({a['commit'][:8]})")
    _audit({"event": "monitor_run",
            "ts": datetime.utcnow().isoformat() + "Z",
            "alerts": alerts, "reverts": revert_results})
    return {"alerts": alerts, "reverts": revert_results}


def sample_today_pnl_from_journal() -> None:
    """Append today's realized PnL to the state PnL history."""
    today = date.today()
    p = JOURNALS_DIR / f"{today.isoformat()}.json"
    if not p.exists():
        return
    try:
        data = json.loads(p.read_text())
        state.record_pnl_sample(today, data["summary"]["total_pnl"],
                                 data["summary"].get("max_drawdown", 0))
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-auto-revert", action="store_true",
                    help="Report alerts only; don't revert commits.")
    ap.add_argument("--sample-pnl", action="store_true",
                    help="Append today's PnL sample from the day's journal.")
    args = ap.parse_args()
    if args.sample_pnl:
        sample_today_pnl_from_journal()
        print("[monitor] sampled today's PnL")
    result = run_monitor(auto_revert=not args.no_auto_revert)
    print(f"[monitor] alerts: {len(result['alerts'])}")
    for a in result["alerts"]:
        print(f"  ! {a['change_date']}  {a['param']}  {a['reason']}")
    for r in result["reverts"]:
        s = "reverted" if r.get("reverted") else "revert-failed"
        print(f"  {s}: {r['param']} ({r['commit'][:8]})")


if __name__ == "__main__":
    main()
