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
from . import price_context

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


def _load_backtest_priors(labels: list[str]) -> list[dict]:
    """Load one or more backtest-consensus JSONs produced by
    `tools.ai_loop.backtest_journal`. Each entry has shape
        {"label": ..., "sources": [...], "stats": { ... }}
    and is surfaced to every rule under the `priors` kwarg so rules
    can use them as a long-tape sanity check alongside the
    short-horizon daily journals.
    """
    out = []
    for label in labels:
        p = JOURNALS_DIR / f"backtest_{label}.json"
        if not p.exists():
            # also accept raw filename
            alt = JOURNALS_DIR / label
            if alt.exists():
                p = alt
            else:
                raise SystemExit(
                    f"backtest prior not found: {p} "
                    f"(run `python3 -m tools.ai_loop.backtest_journal --label {label} ...` first)"
                )
        out.append(json.loads(p.read_text()))
    return out


# ─── Rules ───────────────────────────────────────────────────
# Each rule is a function (journals: list[dict]) -> list[rec].
# Rule IDs are stable so we can dedupe + track history.

def rule_cm_gate_never_fires(journals: list[dict], priors: list[dict] | None = None) -> list[dict]:
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


def rule_kalshi_tp_thr_drawdown(journals: list[dict], priors: list[dict] | None = None) -> list[dict]:
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


def rule_rl_status_closed_chop(journals: list[dict], priors: list[dict] | None = None) -> list[dict]:
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


def rule_structural_advisory_from_priors(journals: list[dict], priors: list[dict] | None = None) -> list[dict]:
    """Prior-only rule: surface STRUCTURAL findings from the long-tape
    backtest consensus. These do NOT propose auto-applies (the params
    involved — side disable, session window, sub-strategy retirement
    — are not in the auto-adjust whitelist). They're emitted as
    `kind=advisory` entries so the analyzer's output has a single
    place where the validator / human sees the big-picture signal
    the daily journals can't see.

    Fires when priors contain clear structural drag: negative-expectancy
    side, negative-expectancy session, negative-expectancy sub-strategy
    (≥20 trades), or profit factor < 1.30.
    """
    if not priors:
        return []
    advisories: list[dict] = []
    for prior in priors:
        label = prior.get("label", "?")
        s = prior.get("stats", {})
        if not s or s.get("n_trades", 0) < 100:
            continue
        findings = []
        # 1. Side asymmetry
        sides = s.get("by_side", {})
        for side, st in sides.items():
            if st["n"] >= 50 and st["pnl"] < -200:
                findings.append({
                    "kind": "side_drag",
                    "side": side,
                    "n": st["n"], "pnl": st["pnl"], "wr": st["wr"],
                    "suggestion": f"Consider disabling {side} entries or adding a directional bias filter",
                })
        # 2. Family drag
        fams = s.get("by_family", {})
        for fam, st in fams.items():
            if st["n"] >= 40 and st["pnl"] < -500:
                findings.append({
                    "kind": "family_drag",
                    "family": fam,
                    "n": st["n"], "pnl": st["pnl"], "wr": st["wr"],
                    "suggestion": f"Retire or tighten the {fam} sub-strategy family",
                })
        # 3. Session drag
        sess = s.get("by_session", {})
        for name, st in sess.items():
            if st["n"] >= 50 and st["pnl"] < -500:
                findings.append({
                    "kind": "session_drag",
                    "session": name,
                    "n": st["n"], "pnl": st["pnl"], "wr": st["wr"],
                    "suggestion": f"Add a {name}-hours trade-window veto",
                })
        # 4. Worst sub-strategies with sample size
        subs = s.get("by_sub_strategy", {})
        for sub, st in sorted(subs.items(), key=lambda kv: kv[1]["pnl"])[:5]:
            if st["n"] >= 20 and st["pnl"] < -400:
                findings.append({
                    "kind": "sub_strategy_drag",
                    "sub_strategy": sub,
                    "n": st["n"], "pnl": st["pnl"], "wr": st["wr"], "avg": st["avg"],
                    "suggestion": f"Candidate for retirement — negative EV over {st['n']} trades",
                })
        # 5. Profit factor warning
        pf = s.get("profit_factor", 0.0)
        if pf and pf < 1.30 and s.get("n_trades", 0) >= 100:
            findings.append({
                "kind": "low_profit_factor",
                "profit_factor": pf,
                "suggestion": (
                    f"PF={pf:.2f} < 1.30 — threshold nudges won't save it. "
                    "Need structural change (retire worst sub-strategies / "
                    "add session or side filter)."
                ),
            })
        # 6. Cascade-day density → circuit breaker justification
        flags = s.get("pattern_flags", [])
        for fl in flags:
            if "cascade" in fl.lower():
                findings.append({
                    "kind": "cascade_density",
                    "flag": fl,
                    "suggestion": (
                        "Justifies a mechanical consecutive-loss circuit "
                        "breaker (add JULIE_CONSEC_LOSS_BLOCKER_ACTIVE to whitelist "
                        "and wire a 2-loss-in-10min hard-stop)."
                    ),
                })
                break

        if findings:
            advisories.append({
                "kind": "advisory",
                "auto_applyable": False,
                "rule_id": "structural_advisory_from_priors",
                "param": None,
                "current": None, "proposed": None,
                "prior_label": label,
                "prior_window": s.get("date_range"),
                "prior_n_trades": s.get("n_trades"),
                "prior_net_pnl": s.get("total_pnl"),
                "prior_profit_factor": s.get("profit_factor"),
                "findings": findings,
                "expected_lift": (
                    f"Long-tape ({label}) flagged {len(findings)} structural "
                    "drag patterns. Manual review required — these changes "
                    "are outside the auto-adjust whitelist."
                ),
            })
    return advisories


def rule_price_regime_correlates_losses(
    journals: list[dict],
    priors: list[dict] | None = None,
    prices=None,
) -> list[dict]:
    """Prior + price-parquet rule: for each backtest prior, look at
    BEST-day vs WORST-day price regime (avg intraday range + bar vol).
    If worst days have systematically higher range/vol than best days,
    that's evidence a pre-trade vol filter would help — emit an
    advisory suggesting a volatility-based trade-window veto.
    """
    if not priors:
        return []
    advisories: list[dict] = []
    for prior in priors:
        label = prior.get("label", "?")
        pc = (prior.get("stats") or {}).get("price_context") or {}
        if not pc.get("available"):
            continue
        bs = pc.get("best_days_summary") or {}
        ws = pc.get("worst_days_summary") or {}
        if not (bs.get("avg_range_pts") and ws.get("avg_range_pts")):
            continue
        range_delta = ws["avg_range_pts"] - bs["avg_range_pts"]
        vol_delta = (ws.get("avg_bar_vol_pts") or 0) - (bs.get("avg_bar_vol_pts") or 0)
        # Two directions worth surfacing:
        #   A) worst days ≥ 15pt WIDER or ≥0.25pt higher bar-vol than best
        #      → high-vol skip filter
        #   B) worst days ≥ 15pt NARROWER or ≥0.25pt lower bar-vol than best
        #      → low-vol skip filter (quiet-tape = no edge)
        direction = None
        if range_delta >= 15.0 or vol_delta >= 0.25:
            direction = "high_vol"
        elif range_delta <= -15.0 or vol_delta <= -0.25:
            direction = "low_vol"
        if direction is None:
            continue
        if direction == "high_vol":
            lift = (
                f"In the {label} tape, losing days had {range_delta:+.1f}pt wider "
                f"intraday range and {vol_delta:+.3f}pt higher bar-to-bar vol than "
                "winning days. A HIGH-vol skip filter (skip new entries when "
                "trailing 30-min bar-vol > Xpt) is worth backtesting."
            )
        else:
            lift = (
                f"In the {label} tape, losing days had {-range_delta:.1f}pt NARROWER "
                f"intraday range and {-vol_delta:.3f}pt lower bar-to-bar vol than "
                "winning days — the bot needs movement to work. A LOW-vol skip "
                "filter (skip new entries when trailing 30-min range < Ypt) "
                "is the natural hedge. Now that the price parquet is wired, "
                "this is directly backtestable."
            )
        advisories.append({
            "kind": "advisory",
            "auto_applyable": False,
            "rule_id": "price_regime_correlates_losses",
            "param": None, "current": None, "proposed": None,
            "prior_label": label,
            "direction": direction,
            "best_days_avg_range_pts": bs.get("avg_range_pts"),
            "worst_days_avg_range_pts": ws.get("avg_range_pts"),
            "range_delta_pts": round(range_delta, 2),
            "best_days_avg_bar_vol_pts": bs.get("avg_bar_vol_pts"),
            "worst_days_avg_bar_vol_pts": ws.get("avg_bar_vol_pts"),
            "vol_delta_pts": round(vol_delta, 3),
            "expected_lift": lift,
        })
    return advisories


def rule_today_price_regime_flag(
    journals: list[dict],
    priors: list[dict] | None = None,
    prices=None,
) -> list[dict]:
    """If today's intraday range or realized vol is in the top decile of
    the 2025+2026 tape AND today's journal has cascade-day signature
    (≥3-loss streak), emit an advisory to freeze auto-applies for the
    night — we shouldn't let a single-day outlier drive config changes.
    """
    if not journals:
        return []
    today = journals[0]
    d = today.get("date")
    if not d or prices is None:
        return []
    try:
        ctx = price_context.day_context(d, df=prices)
    except Exception:
        return []
    if ctx is None:
        return []
    # Heuristic thresholds keyed to 2025+2026 distribution:
    #   range > 80pt is high; range > 120pt is extreme
    #   bar_vol > 2.5 is high
    range_pts = ctx.get("range_pts", 0)
    bar_vol = ctx.get("bar_vol_pts", 0)
    is_extreme = range_pts > 120 or bar_vol > 3.0
    is_high = range_pts > 80 or bar_vol > 2.0
    if not (is_high or is_extreme):
        return []
    # Check for cascade-day signature in today's journal flags
    flags = today.get("pattern_flags", [])
    cascade_like = any(("consecutive losses" in fl.lower()) for fl in flags)
    if not (is_extreme or cascade_like):
        return []
    return [{
        "kind": "advisory",
        "auto_applyable": False,
        "rule_id": "today_price_regime_flag",
        "param": None, "current": None, "proposed": None,
        "date": d,
        "range_pts": range_pts,
        "bar_vol_pts": bar_vol,
        "trend_dir": ctx.get("trend_dir"),
        "cascade_signature": cascade_like,
        "expected_lift": (
            f"Today {d} had {range_pts:.1f}pt range / bar-vol {bar_vol:.2f}pt "
            f"({'EXTREME' if is_extreme else 'elevated'}"
            f"{' + cascade signature' if cascade_like else ''}). "
            "Recommend freezing auto-applies for 24h so a volatile outlier "
            "doesn't drive a config change. Set "
            f"{'JULIE_FREEZE_AUTO_CONFIG=1 for this cycle' if is_extreme else 'manual review'}."
        ),
    }]


# Register all rules
RULES = [
    rule_cm_gate_never_fires,
    rule_kalshi_tp_thr_drawdown,
    rule_rl_status_closed_chop,
    rule_structural_advisory_from_priors,
    rule_price_regime_correlates_losses,
    rule_today_price_regime_flag,
]


def analyze(target_date: date, prior_labels: list[str] | None = None) -> dict:
    journals = _gather_recent_journals(target_date, LOOKBACK_DAYS)
    priors = _load_backtest_priors(prior_labels) if prior_labels else []
    # Load price parquet once per run — rules that need it take kwarg `prices`
    prices = price_context.load_prices()
    recs = []
    for i, rule in enumerate(RULES):
        try:
            # Some older rules only accept (journals, priors). Use kwargs so
            # adding `prices=` doesn't break them.
            try:
                out = rule(journals, priors=priors, prices=prices)
            except TypeError:
                out = rule(journals, priors=priors)
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
        "priors_consulted": [p.get("label") for p in priors],
        "prices_loaded": None if prices is None else len(prices),
        "recommendations": recs,
    }


def write_recommendations(target_date: date, prior_labels: list[str] | None = None) -> Path:
    result = analyze(target_date, prior_labels=prior_labels)
    out = RECS_DIR / f"{target_date.isoformat()}.json"
    out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None)
    ap.add_argument(
        "--backtest-journal",
        action="append",
        default=None,
        help="Backtest consensus label (filename minus .json). Repeatable. "
             "Surfaces long-tape priors to every rule.",
    )
    args = ap.parse_args()
    d = datetime.strptime(args.date, "%Y-%m-%d").date() if args.date else date.today()
    out = write_recommendations(d, prior_labels=args.backtest_journal)
    print(f"[analyzer] wrote {out}")
    data = json.loads(out.read_text())
    print(f"[analyzer] journals consulted: {data['journals_consulted']}")
    print(f"[analyzer] priors consulted: {data.get('priors_consulted') or 'none'}")
    print(f"[analyzer] recommendations: {len(data['recommendations'])}")
    for r in data["recommendations"]:
        if "error" in r:
            print(f"  ! {r['id']}: error {r['error']}")
        elif r.get("kind") == "advisory":
            findings = r.get("findings")
            if findings:
                print(f"  ⚑ {r['id']}: ADVISORY from {r.get('prior_label')} — "
                      f"{len(findings)} structural findings")
                for f in findings:
                    print(f"     · [{f['kind']}] {f.get('suggestion','')}")
            elif r.get("direction"):
                lbl = r.get("prior_label") or r.get("date", "?")
                print(f"  ⚑ {r['id']}: ADVISORY ({lbl}) — "
                      f"price-regime {r['direction']} signal "
                      f"(Δrange {r.get('range_delta_pts','?')}pt · "
                      f"Δvol {r.get('vol_delta_pts','?')}pt)")
            else:
                print(f"  ⚑ {r['id']}: ADVISORY — {r.get('rule_id','?')}")
        else:
            print(f"  · {r['id']}: {r['param']} {r['current']} → {r['proposed']}")


if __name__ == "__main__":
    main()
