"""Layer 4 — Auto-apply (THE MONEY PART).

Takes green-lit recommendations from validator and turns them into
actual config changes on disk. Every change is a single git commit
tagged `[AUTO_APPLIED]` so you can grep history and revert.

Safety rails — ALL must pass for a change to apply:
  1. Global kill switch `JULIE_FREEZE_AUTO_CONFIG=1` is not set
  2. State is not "frozen" (set by monitor or stop-loss)
  3. Param is on the whitelist (config.AUTO_ADJUSTABLE_PARAMS)
  4. Proposed value passes config.validate_param_delta
  5. Validator's status == "green"
  6. Param cool-down has elapsed (config.COOLDOWN_DAYS)
  7. Daily apply cap (config.MAX_APPLIES_PER_DAY) not exceeded
  8. For `high_risk` params, validator must show at least 2 consecutive
     green evaluations across different days (not implemented yet —
     treated as always-skip for now)

On success:
    - Config file patched (launcher env default OR joblib payload)
    - git commit with detailed rationale
    - state.json updated (cooldown + daily count)
    - audit.jsonl append-only record
    - applied/YYYY-MM-DD.json gets the applied-rec
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import date, datetime, timedelta
from pathlib import Path

import joblib

from . import state
from .config import (
    AUTO_ADJUSTABLE_PARAMS, APPLIED_DIR, AUDIT_LOG, ROOT, VALIDATED_DIR,
    COOLDOWN_DAYS, MAX_APPLIES_PER_DAY, STOP_LOSS_48H_DOLLARS,
    is_frozen, validate_param_delta,
)


def _audit(record: dict) -> None:
    AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with AUDIT_LOG.open("a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a git command in the repo root."""
    return subprocess.run(["git", "-C", str(ROOT), *args],
                          capture_output=True, text=True, check=check)


def _cooldown_ok(param: str) -> tuple[bool, str]:
    st = state.load()
    last = st["last_applied_at"].get(param)
    if not last:
        return True, "no prior apply"
    try:
        last_dt = datetime.fromisoformat(last.replace("Z", ""))
    except Exception:
        return True, "unparseable timestamp — treating as never applied"
    age = (datetime.utcnow() - last_dt).days
    if age < COOLDOWN_DAYS:
        return False, f"cooldown: applied {age}d ago, need {COOLDOWN_DAYS}d"
    return True, f"last apply {age}d ago"


def _daily_cap_ok() -> tuple[bool, str]:
    st = state.load()
    today = date.today().isoformat()
    n = st["applies_by_date"].get(today, 0)
    if n >= MAX_APPLIES_PER_DAY:
        return False, f"daily cap reached: {n}/{MAX_APPLIES_PER_DAY}"
    return True, f"{n}/{MAX_APPLIES_PER_DAY} applied today"


def _stop_loss_ok() -> tuple[bool, str]:
    """If live trading has bled > threshold in last 48h, freeze."""
    st = state.load()
    samples = [s for s in st["live_pnl_samples"]
               if (datetime.utcnow() - datetime.fromisoformat(s["ts"].replace("Z", ""))).days < 2]
    if not samples:
        return True, "no recent samples"
    cum_pnl = sum(s["pnl"] for s in samples)
    cum_dd = max(s["drawdown"] for s in samples)
    if cum_pnl < -STOP_LOSS_48H_DOLLARS or cum_dd > STOP_LOSS_48H_DOLLARS:
        return False, f"stop-loss trip: 48h pnl ${cum_pnl:.2f}, max_dd ${cum_dd:.2f}"
    return True, f"48h pnl ${cum_pnl:.2f}, max_dd ${cum_dd:.2f}"


def _patch_env_var(param: str, new_value) -> Path:
    """Rewrite the matching `os.environ.setdefault("PARAM", "VAL")` line
    in launch_filterless_live.py. Returns the path that was modified."""
    p = ROOT / "launch_filterless_live.py"
    text = p.read_text()
    # Match both "1" and literal 1 forms
    pat = re.compile(
        rf'(os\.environ\.setdefault\(\s*"{re.escape(param)}"\s*,\s*)"([^"]*)"(\s*\))'
    )
    new_str = str(int(new_value)) if isinstance(new_value, bool) or (
        isinstance(new_value, (int, float)) and float(new_value).is_integer()
    ) else str(new_value)
    replaced, n = pat.subn(rf'\g<1>"{new_str}"\g<3>', text)
    if n == 0:
        raise RuntimeError(f"could not find launcher line for env '{param}'")
    p.write_text(replaced)
    return p


def _patch_joblib(param_key: str, new_value) -> Path:
    """Modify a joblib payload in place. Key format is 'relative/path:payload_key'."""
    rel, pkey = param_key.split(":", 1)
    full = ROOT / rel
    payload = joblib.load(full)
    payload[pkey] = new_value
    joblib.dump(payload, full)
    return full


def apply_one(rec: dict, dry_run: bool = False) -> dict:
    """Apply ONE green-lit recommendation. Returns result dict."""
    param = rec["param"]
    spec = AUTO_ADJUSTABLE_PARAMS.get(param)
    result = {
        "param": param, "current": rec["current"], "proposed": rec["proposed"],
        "rule_id": rec.get("rule_id"),
        "attempted_at": datetime.utcnow().isoformat() + "Z",
        "status": "pending",
        "reason": "",
        "commit_sha": None,
        "dry_run": dry_run,
    }

    # Kill switches
    # JULIE_AILOOP_APPLY (2026-04-24): default-OFF kill switch.
    # Distinct from JULIE_FREEZE_AUTO_CONFIG (which is opt-IN freeze) —
    # this one is opt-IN unfreeze. Even if JULIE_FREEZE_AUTO_CONFIG is
    # unset, the applier refuses to write unless JULIE_AILOOP_APPLY=1.
    # Journaling (analyzer, validator, audit log) is untouched — this
    # kill switch only blocks the write+commit+state mutation.
    import os as _os
    if _os.environ.get("JULIE_AILOOP_APPLY", "0").strip() != "1":
        result["status"] = "skipped"
        result["reason"] = "JULIE_AILOOP_APPLY=0 — auto-apply disabled (journaling-only mode)"
        _audit(result)
        return result
    if is_frozen():
        result["status"] = "skipped"
        result["reason"] = f"JULIE_FREEZE_AUTO_CONFIG is set"
        _audit(result)
        return result
    st = state.load()
    if st.get("frozen"):
        result["status"] = "skipped"
        result["reason"] = f"state frozen: {st.get('freeze_reason', '')}"
        _audit(result)
        return result

    # Validator must have green-lit
    val = rec.get("validation", {})
    if val.get("status") != "green":
        result["status"] = "skipped"
        result["reason"] = f"validator status = {val.get('status')}: {val.get('reason')}"
        _audit(result)
        return result

    # Spec + delta gate
    ok, why = validate_param_delta(param, float(rec["current"]), float(rec["proposed"]))
    if not ok:
        result["status"] = "skipped"; result["reason"] = f"delta gate: {why}"
        _audit(result); return result

    # High-risk requires extra confirmation (not implemented; deny for now)
    if spec.get("high_risk"):
        result["status"] = "skipped"
        result["reason"] = "high_risk param — manual apply required, auto-skipping"
        _audit(result); return result

    # Cool-down
    ok, why = _cooldown_ok(param)
    if not ok:
        result["status"] = "skipped"; result["reason"] = why
        _audit(result); return result

    # Daily cap
    ok, why = _daily_cap_ok()
    if not ok:
        result["status"] = "skipped"; result["reason"] = why
        _audit(result); return result

    # Stop-loss
    ok, why = _stop_loss_ok()
    if not ok:
        state.set_frozen(why)
        result["status"] = "frozen"; result["reason"] = f"48h stop-loss tripped: {why}"
        _audit(result); return result

    # DRY-RUN: report but don't touch files
    if dry_run:
        result["status"] = "would-apply"
        result["reason"] = "all gates passed (dry-run)"
        _audit(result); return result

    # ─── Apply ───
    try:
        if spec["target"] == "env":
            patched = _patch_env_var(param, rec["proposed"])
            patched_files = [patched]
        elif spec["target"] == "joblib":
            patched = _patch_joblib(spec["key"], rec["proposed"])
            patched_files = [patched]
            for also in spec.get("also_update", []):
                patched_files.append(_patch_joblib(also, rec["proposed"]))
        else:
            result["status"] = "error"
            result["reason"] = f"unknown target type: {spec['target']}"
            _audit(result); return result

        # git commit
        for p in patched_files:
            _git("add", str(p.relative_to(ROOT)))
        msg = (
            f"[AUTO_APPLIED] {param}: {rec['current']} → {rec['proposed']}\n\n"
            f"Rule: {rec.get('rule_id', 'n/a')}\n"
            f"Basis:\n{json.dumps(rec.get('basis', {}), indent=2)}\n\n"
            f"Validator numbers:\n{json.dumps(rec.get('validation', {}), indent=2)}\n\n"
            f"Revert with: git revert HEAD"
        )
        _git("commit", "-m", msg)
        sha = _git("rev-parse", "HEAD").stdout.strip()
        state.record_apply(param, rec["proposed"], sha)
        result["status"] = "applied"; result["commit_sha"] = sha
        result["reason"] = "ok"
    except subprocess.CalledProcessError as exc:
        result["status"] = "error"
        result["reason"] = f"git error: {exc.stderr or exc.stdout}"
    except Exception as exc:
        result["status"] = "error"
        result["reason"] = f"{type(exc).__name__}: {exc}"
    _audit(result)
    return result


def apply_all(target_date: date, dry_run: bool = False) -> Path:
    val_path = VALIDATED_DIR / f"{target_date.isoformat()}.json"
    if not val_path.exists():
        raise FileNotFoundError(f"no validated recs for {target_date}")
    data = json.loads(val_path.read_text())
    results = []
    for rec in data.get("recommendations", []):
        results.append(apply_one(rec, dry_run=dry_run))
    out = APPLIED_DIR / f"{target_date.isoformat()}.json"
    out.write_text(json.dumps({
        "date": target_date.isoformat(),
        "dry_run": dry_run,
        "results": results,
    }, indent=2, default=str), encoding="utf-8")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what WOULD apply without touching anything.")
    args = ap.parse_args()
    d = datetime.strptime(args.date, "%Y-%m-%d").date() if args.date else date.today()
    out = apply_all(d, dry_run=args.dry_run)
    print(f"[applier] wrote {out}")
    data = json.loads(out.read_text())
    for r in data["results"]:
        print(f"  {r['status']:10s}  {r['param']:45s}  {r['reason']}")


if __name__ == "__main__":
    main()
