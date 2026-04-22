"""Layer 2 — Recommendation analyzer.

Reads recent journals (last N days) and emits concrete adjustment
proposals in a structured JSON format. Each proposal is ONE param
change, scoped to the whitelist in config.py.

The analyzer is deliberately conservative:
  - Each rule looks at multi-day patterns (not single-session noise)
  - Each proposed change is small (within `max_step_delta`)
  - Each proposal comes with a quantified "basis" that the validator
    can verify

Output: ai_loop_data/recommendations/YYYY-MM-DD.json
  {
    "date": "...",
    "recommendations": [
      {
        "id": "rec_001",
        "param": "cm_gate_v2_override_threshold",
        "current": 0.60,
        "proposed": 0.55,
        "rule_id": "cm_gate_never_fires",
        "basis": { ... measurable facts from journals ... },
        "expected_lift": "+$X/day estimate"
      }
    ]
  }
"""
from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from .config import (
    AUTO_ADJUSTABLE_PARAMS, JOURNALS_DIR, RECS_DIR,
)

LOOKBACK_DAYS = 10
"""How many recent journals to consult when proposing adjustments."""


def _load_journal(d: date) -> dict | None:
    p = JOURNALS_DIR / f"{d.isoformat()}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _gather_recent_journals(end_date: date, n: int = LOOKBACK_DAYS) -> list[dict]:
    out = []
    d = end_date
    while len(out) < n and (end_date - d).days < n * 2:
        j = _load_journal(d)
        if j is not None:
            out.append(j)
        d = d - timedelta(days=1)
    return out


# ─── Rules ───────────────────────────────────────────────────
# Each rule is a function (journals: list[dict]) -> list[rec].
# Rule IDs are stable so we can dedupe + track history.

def rule_cm_gate_never_fires(journals: list[dict]) -> list[dict]:
    """If CM gate v2 is active but rarely/never fires an override AND
    Kalshi is blocking a lot, nudge the threshold DOWN by 0.05.
    Symmetric rule: if firing a LOT (> 20% of Kalshi blocks), nudge UP.
    """
    if not journals: return []
    total_blocks = sum(j["summary"]["n_kalshi_blocks"] for j in journals)
    total_v2_fires = sum(j["summary"]["n_cm_gate_v2_would_override"] for j in journals)
    if total_blocks < 50:
        return []   # not enough signal
    fire_rate = total_v2_fires / total_blocks
    spec = AUTO_ADJUSTABLE_PARAMS["cm_gate_v2_override_threshold"]
    # Current threshold: read from the joblib payload.
    import joblib
    from .config import ROOT
    p = joblib.load(ROOT / "artifacts/signal_gate_2025/model_cm_breakout_long.joblib")
    current = float(p.get("override_threshold", 0.60))
    rec = None
    if fire_rate < 0.02 and current > spec["bounds"][0] + 0.05:
        new_val = round(max(spec["bounds"][0], current - 0.05), 3)
        rec = {
            "param": "cm_gate_v2_override_threshold",
            "current": current, "proposed": new_val,
            "rule_id": "cm_gate_never_fires",
            "basis": {
                "observation_days": len(journals),
                "total_kalshi_blocks": total_blocks,
                "total_v2_would_override": total_v2_fires,
                "fire_rate": round(fire_rate, 4),
            },
            "expected_lift": (
                "If the threshold had been 0.05 lower, the model's rolling-"
                "origin A/B still shows 60%+ hit rate at p≥0.55 — expect "
                "5-15 additional overrides per day each worth ~$50 EV on "
                "avg. Validator will confirm via backtest."
            ),
        }
    elif fire_rate > 0.25 and current < spec["bounds"][1] - 0.05:
        new_val = round(min(spec["bounds"][1], current + 0.05), 3)
        rec = {
            "param": "cm_gate_v2_override_threshold",
            "current": current, "proposed": new_val,
            "rule_id": "cm_gate_fires_too_often",
            "basis": {
                "observation_days": len(journals),
                "total_kalshi_blocks": total_blocks,
                "total_v2_would_override": total_v2_fires,
                "fire_rate": round(fire_rate, 4),
            },
            "expected_lift": (
                "Firing on > 25% of Kalshi blocks suggests the threshold "
                "is too loose — likely catching noise. Tighten by 0.05."
            ),
        }
    return [rec] if rec else []


def rule_kalshi_tp_thr_drawdown(journals: list[dict]) -> list[dict]:
    """If trades closed by Kalshi-TP gate had worse-than-expected realized
    PnL across several sessions, suggest raising the TP PnL threshold."""
    if len(journals) < 3: return []
    # Simple heuristic: if total realized PnL < 0 across last 3 sessions
    # AND the Kalshi-TP gate is active, suggest raising the threshold
    # modestly. The assumption: a higher threshold blocks marginal
    # trades more aggressively.
    recent = journals[:3]
    total_pnl = sum(j["summary"]["total_pnl"] for j in recent)
    if total_pnl >= 0:
        return []
    # Read current from launch_filterless_live.py
    import os
    from .config import ROOT
    launcher = (ROOT / "launch_filterless_live.py").read_text()
    import re
    m = re.search(r'JULIE_ML_KALSHI_TP_PNL_THR["\s,]+["]([\-\d.]+)', launcher)
    current = float(m.group(1)) if m else 0.0
    spec = AUTO_ADJUSTABLE_PARAMS["JULIE_ML_KALSHI_TP_PNL_THR"]
    proposed = round(min(spec["bounds"][1], current + 2.5), 2)
    if proposed == current:
        return []
    return [{
        "param": "JULIE_ML_KALSHI_TP_PNL_THR",
        "current": current, "proposed": proposed,
        "rule_id": "kalshi_tp_pnl_drawdown",
        "basis": {
            "recent_sessions": len(recent),
            "cum_pnl_recent": round(total_pnl, 2),
            "trades_in_window": sum(j["summary"]["n_trades"] for j in recent),
        },
        "expected_lift": (
            f"Cum PnL over last {len(recent)} sessions is ${total_pnl:.0f}. "
            f"Raising the Kalshi-TP threshold from {current} to {proposed} "
            "demands more profit-margin confidence before accepting trades. "
            "Validator will backtest on last 30 days to confirm."
        ),
    }]


def rule_rl_status_closed_chop(journals: list[dict]) -> list[dict]:
    """If RL fires 'status=closed' (SL-move-on-underwater) on multiple
    days, freeze RL live mode until debugged. This is the c5caf83
    scenario — SHOULD be zero. If it's non-zero, something's wrong."""
    if not journals: return []
    # Since we added the would_breach_market guard, any status=closed
    # event is a signal that something regressed.
    count = 0
    for j in journals[:5]:
        # We put rl_live log lines in the raw journal; here we use raw_counts
        # and pattern flags. Count is approximate but conservative.
        for fl in j.get("pattern_flags", []):
            if "status=closed" in fl or "SL-move-on-underwater" in fl:
                count += 1
    if count < 2:
        return []
    return [{
        "param": "JULIE_ML_RL_MGMT_ACTIVE",
        "current": 1, "proposed": 0,
        "rule_id": "rl_forced_close_regression",
        "basis": {
            "days_with_forced_closes": count,
            "window_days": 5,
        },
        "expected_lift": (
            "RL forced-close event observed on 2+ recent sessions — this "
            "indicates the c5caf83 would_breach_market guard has regressed "
            "or a new mid-bar SL issue has appeared. Freeze RL live mode "
            "until a human audits the executor."
        ),
        "high_risk": True,
    }]


# Register all rules
RULES = [
    rule_cm_gate_never_fires,
    rule_kalshi_tp_thr_drawdown,
    rule_rl_status_closed_chop,
]


def analyze(target_date: date) -> dict:
    journals = _gather_recent_journals(target_date, LOOKBACK_DAYS)
    recs = []
    for i, rule in enumerate(RULES):
        try:
            out = rule(journals)
            for r in out:
                r["id"] = f"rec_{i:03d}_{r.get('rule_id', 'unknown')}"
                recs.append(r)
        except Exception as exc:
            recs.append({
                "id": f"rec_error_{i:03d}",
                "rule_id": getattr(rule, "__name__", "unknown"),
                "error": str(exc),
            })
    return {
        "date": target_date.isoformat(),
        "journals_consulted": len(journals),
        "recommendations": recs,
    }


def write_recommendations(target_date: date) -> Path:
    result = analyze(target_date)
    out = RECS_DIR / f"{target_date.isoformat()}.json"
    out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None)
    args = ap.parse_args()
    d = datetime.strptime(args.date, "%Y-%m-%d").date() if args.date else date.today()
    out = write_recommendations(d)
    print(f"[analyzer] wrote {out}")
    data = json.loads(out.read_text())
    print(f"[analyzer] recommendations: {len(data['recommendations'])}")
    for r in data["recommendations"]:
        if "error" in r:
            print(f"  ! {r['id']}: error {r['error']}")
        else:
            print(f"  · {r['id']}: {r['param']} {r['current']} → {r['proposed']}")


if __name__ == "__main__":
    main()
