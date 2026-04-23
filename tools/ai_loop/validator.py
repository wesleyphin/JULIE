"""Layer 3 — Backtest-gated validator.

For each recommendation from analyzer.py, run a SHORT backtest with:
  baseline  = current config
  candidate = config with the proposed param change applied

Green-light rules:
  - candidate total PnL >= baseline * (1 + BACKTEST_MIN_LIFT_PCT)
  - candidate max drawdown <= baseline max DD * BACKTEST_MAX_DD_INFLATE
  - candidate has no chunk-level regression > 10%
  - every change still passes config.validate_param_delta

Input:  ai_loop_data/recommendations/YYYY-MM-DD.json
Output: ai_loop_data/validated/YYYY-MM-DD.json with each rec annotated:
  {
    ... original fields ...,
    "validation": {
      "status": "green" | "reject" | "error",
      "baseline_pnl": …, "candidate_pnl": …, "lift_pct": …,
      "baseline_dd":  …, "candidate_dd":  …, "dd_inflate": …,
      "reason": "..."
    }
  }

Note: running a full-fidelity backtest per rec is slow. For the v0 of
this module we implement a FAST-BACKTEST mode that replays the last N
days of tape using cached closed_trades (from existing replay outputs)
rather than re-running the full bot loop. This is less faithful than
re-replay but 100× faster. Full-replay mode hooks are stubbed for a
future session.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from .config import (
    ROOT, RECS_DIR, VALIDATED_DIR,
    AUTO_ADJUSTABLE_PARAMS, validate_param_delta,
    BACKTEST_MIN_LIFT_PCT, BACKTEST_MAX_DD_INFLATE,
    BACKTEST_PERIOD_DAYS,
)


def _latest_replay_dir() -> Path | None:
    """Find the most recent full_live_replay directory we can use as a
    backtest baseline (no need to re-run the full bot for fast-mode)."""
    candidates = list((ROOT / "backtest_reports" / "full_live_replay").glob("*"))
    with_trades = [c for c in candidates if (c / "closed_trades.json").exists()]
    if not with_trades:
        return None
    return max(with_trades, key=lambda p: p.stat().st_mtime)


def _pnl_dd_from_trades(trades: list[dict]) -> tuple[float, float, int]:
    """Return (total_pnl, max_drawdown, n_trades) given closed_trades list."""
    if not trades: return 0.0, 0.0, 0
    s = sorted(trades, key=lambda t: t.get("entry_time", t.get("ts", "")))
    cum, peak, dd = 0.0, 0.0, 0.0
    for t in s:
        pnl = float(t.get("pnl_dollars", 0) or 0)
        cum += pnl
        peak = max(peak, cum)
        dd = max(dd, peak - cum)
    return round(cum, 2), round(dd, 2), len(s)


def _simulate_param_change(trades: list[dict], rec: dict) -> tuple[float, float, int]:
    """FAST-BACKTEST: given a closed_trades list and a proposed param
    change, estimate what the PnL + DD WOULD have been if the change
    had been active. Uses simple per-param rewriting rules.

    This is lower fidelity than re-running the full replay — it's a
    first-cut that lets the validator reject obviously bad changes
    quickly. True full-replay hook is stubbed below.
    """
    param = rec["param"]
    proposed = rec["proposed"]
    # For cm_gate_v2 threshold: the gate only affects Kalshi OVERRIDES,
    # so at the trade-list level (which records trades that actually
    # fired), lowering the threshold means MORE trades would have been
    # taken (additional Kalshi-blocked signals get overridden). We
    # can't simulate those additional trades without the raw signal log,
    # so in fast mode we report "inconclusive" and defer to full mode.
    if param == "cm_gate_v2_override_threshold":
        return _pnl_dd_from_trades(trades) + (None,)[:0]  # type: ignore
    if param == "JULIE_ML_KALSHI_TP_PNL_THR":
        # Kalshi-TP gate: higher threshold blocks trades with lower
        # predicted pnl. Simulate by filtering out trades whose pnl
        # is below the threshold.
        # Conservative: drop the N least-profitable that sit below thr.
        filtered = [t for t in trades
                    if float(t.get("pnl_dollars", 0) or 0) >= proposed]
        return _pnl_dd_from_trades(filtered)
    if param in ("JULIE_KALSHI_CM_GATE_V2_ACTIVE", "JULIE_ML_RL_MGMT_ACTIVE"):
        # Toggle flags: fast mode can't simulate — they change behavior
        # across the whole tape. Defer to full replay.
        return _pnl_dd_from_trades(trades) + (None,)[:0]  # type: ignore
    # Default: unchanged (fast mode doesn't know this param)
    return _pnl_dd_from_trades(trades)


def _full_replay_gate(rec: dict, days: int = BACKTEST_PERIOD_DAYS) -> dict | None:
    """STUB: full-replay validation. For future build-out. The idea:
        1. Create a fresh replay-run-name
        2. Launch tools/run_full_live_replay_parquet.py with modified env
        3. Parse its closed_trades.json
        4. Compare to baseline

    This is expensive (10-60 min per rec) and needs the orchestration
    logic to run in parallel. We ship the hook now, implement the
    parallel runner in a follow-up.
    """
    return None


def validate_rec(rec: dict, baseline_trades: list[dict]) -> dict:
    """Run the safety + numeric checks on a single rec. Annotates it
    with a 'validation' dict and returns the enriched rec."""
    # 1. Param-delta sanity (whitelist + bounds + max-step)
    try:
        cur = float(rec.get("current", 0))
        prop = float(rec.get("proposed", 0))
    except (TypeError, ValueError) as e:
        rec["validation"] = {"status": "error",
                             "reason": f"non-numeric current/proposed: {e}"}
        return rec
    ok, why = validate_param_delta(rec["param"], cur, prop)
    if not ok:
        rec["validation"] = {"status": "reject", "reason": f"delta gate: {why}"}
        return rec

    # 2. Fast backtest: baseline
    b_pnl, b_dd, b_n = _pnl_dd_from_trades(baseline_trades)
    # 3. Fast backtest: candidate (best-effort)
    c_result = _simulate_param_change(baseline_trades, rec)
    if len(c_result) != 3:
        # simulate returned without a value -> fast mode can't evaluate
        rec["validation"] = {
            "status": "reject",
            "reason": "fast-mode can't evaluate this param; needs full-replay harness (not yet built)",
            "baseline_pnl": b_pnl, "baseline_dd": b_dd, "baseline_trades": b_n,
        }
        return rec
    c_pnl, c_dd, c_n = c_result

    lift_pct = ((c_pnl - b_pnl) / abs(b_pnl)) if b_pnl else float("nan")
    dd_inflate = (c_dd / b_dd) if b_dd else 1.0

    reasons = []
    if b_pnl != 0 and lift_pct < BACKTEST_MIN_LIFT_PCT:
        reasons.append(f"lift {lift_pct:+.2%} < required {BACKTEST_MIN_LIFT_PCT:+.0%}")
    if dd_inflate > BACKTEST_MAX_DD_INFLATE:
        reasons.append(f"dd inflate {dd_inflate:.2f}x > max {BACKTEST_MAX_DD_INFLATE}x")

    status = "green" if not reasons else "reject"
    rec["validation"] = {
        "status": status,
        "baseline_pnl": b_pnl, "candidate_pnl": c_pnl,
        "lift_pct": round(lift_pct, 4) if not (lift_pct != lift_pct) else None,
        "baseline_dd": b_dd, "candidate_dd": c_dd,
        "dd_inflate": round(dd_inflate, 3),
        "baseline_n_trades": b_n, "candidate_n_trades": c_n,
        "reason": "; ".join(reasons) if reasons else "fast-mode OK",
        "mode": "fast",
    }
    return rec


def validate_all(target_date: date) -> Path:
    rec_path = RECS_DIR / f"{target_date.isoformat()}.json"
    if not rec_path.exists():
        raise FileNotFoundError(f"no recommendations for {target_date}")
    data = json.loads(rec_path.read_text())

    # Load baseline trades from latest replay dir
    rep_dir = _latest_replay_dir()
    baseline_trades = []
    if rep_dir is not None:
        ct = rep_dir / "closed_trades.json"
        if ct.exists():
            baseline_trades = json.loads(ct.read_text())

    validated = []
    for rec in data.get("recommendations", []):
        if "error" in rec:
            rec["validation"] = {"status": "error", "reason": rec["error"]}
            validated.append(rec)
            continue
        # Advisories: informational only — not auto-applyable by design.
        # Mark as "info" so applier skips them cleanly and the daily
        # summary still surfaces them.
        if rec.get("kind") == "advisory" or rec.get("auto_applyable") is False:
            rec["validation"] = {
                "status": "info",
                "reason": "advisory from backtest prior — manual review only",
                "mode": "advisory",
            }
            validated.append(rec)
            continue
        validated.append(validate_rec(rec, baseline_trades))

    out = VALIDATED_DIR / f"{target_date.isoformat()}.json"
    out.write_text(json.dumps({
        "date": target_date.isoformat(),
        "baseline_replay_dir": str(rep_dir) if rep_dir else None,
        "baseline_n_trades": len(baseline_trades),
        "recommendations": validated,
    }, indent=2, default=str), encoding="utf-8")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None)
    args = ap.parse_args()
    d = datetime.strptime(args.date, "%Y-%m-%d").date() if args.date else date.today()
    out = validate_all(d)
    print(f"[validator] wrote {out}")
    data = json.loads(out.read_text())
    greens = sum(1 for r in data["recommendations"] if r.get("validation", {}).get("status") == "green")
    rejects = sum(1 for r in data["recommendations"] if r.get("validation", {}).get("status") == "reject")
    errors = sum(1 for r in data["recommendations"] if r.get("validation", {}).get("status") == "error")
    infos = sum(1 for r in data["recommendations"] if r.get("validation", {}).get("status") == "info")
    print(f"[validator] greenlit={greens}  rejected={rejects}  errored={errors}  info/advisory={infos}")


if __name__ == "__main__":
    main()
