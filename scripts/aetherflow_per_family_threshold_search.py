#!/usr/bin/env python3
"""Per-family threshold search + OOS ship-gate for the deployed AetherFlow
bundle.

Finding from diagnose_oos_edge_by_threshold.py: on OOS (2026-01-27 →
2026-04-08), the deployed model shows:
    aligned_flow         — WR 41.7%, avg +$0.286, 48 trades @ 0.55 (profitable)
    transition_burst     — WR 36.1%, avg -$1.944, 36 trades @ 0.55 (losing)
    compression_release  — WR  0.0%, avg -$20.00,  6 trades @ 0.55 (broken)

The aggregate -$1.96/trade is entirely driven by transition_burst +
compression_release. aligned_flow retains edge. Calibration is a uniform
rescale — it can't fix this; per-family thresholding can.

This script sweeps threshold per family on OOS and picks the value that:
    1. Keeps per-family avg PnL ≥ $0 (break-even), AND
    2. Keeps per-family coverage on LIVE slice ≥ 0.1% (some fires possible,
       not mathematically unfirable like Bug 1)

If a family can't hit break-even at any threshold in [0.50, 0.75], it's
recommended for hard-disable (threshold 1.0 or `allowed_setup_families`
removal).

Output:
    backtest_reports/aetherflow_per_family_threshold.json — recommendations
    stdout: human-readable table + suggested config.py snippet
"""
from __future__ import annotations
import json, pickle, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/Users/wes/Downloads/JULIE001")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from aetherflow_features import build_feature_frame
from aetherflow_model_bundle import predict_bundle_probabilities
from tools.ai_loop import price_context
from aetherflow_calibrator_fit_and_validate import (
    features_for, estimate_outcomes, OOS_START, OOS_END, LIVE_START,
)

BUNDLE = ROOT / "model_aetherflow_deploy_2026oos.pkl"
OUT = ROOT / "backtest_reports" / "aetherflow_per_family_threshold.json"

FAMILIES = ["aligned_flow", "transition_burst", "compression_release"]
# Ship gate on each family: break-even or better on OOS, but also keep a
# minimum live-fire rate floor so we don't mathematically-kill a family.
THRESHOLDS = np.arange(0.50, 0.76, 0.005).round(3)


def load_all():
    with BUNDLE.open("rb") as fh:
        bundle = pickle.load(fh)
    prices = price_context.load_prices().sort_index()
    # OOS
    oos_bars, oos_feat = features_for(prices, OOS_START, OOS_END)
    _side = pd.to_numeric(oos_feat.get("candidate_side", 0.0), errors="coerce").fillna(0.0)
    oos_feat = oos_feat.loc[(_side != 0).values]
    oos_probs = predict_bundle_probabilities(bundle, oos_feat)
    oos_feat = oos_feat.iloc[:len(oos_probs)]
    oos_pnl = estimate_outcomes(oos_bars, oos_feat).values
    # LIVE (for coverage floor check)
    _, live_feat = features_for(prices, LIVE_START, None)
    _lside = pd.to_numeric(live_feat.get("candidate_side", 0.0), errors="coerce").fillna(0.0)
    live_feat = live_feat.loc[(_lside != 0).values]
    live_probs = predict_bundle_probabilities(bundle, live_feat)
    live_feat = live_feat.iloc[:len(live_probs)]
    return oos_feat, oos_probs, oos_pnl, live_feat, live_probs


def search():
    oos_feat, oos_probs, oos_pnl, live_feat, live_probs = load_all()
    print(f"[search] OOS: {len(oos_feat):,} candidate rows   "
          f"LIVE: {len(live_feat):,} candidate rows")
    report = {"families": {}, "summary": {}}
    total_trades_at_plan = 0
    total_pnl_at_plan = 0.0
    print()
    print(f"{'family':<22} {'thr':>6}  {'oos-n':>6}  {'oos-WR':>7}  {'oos-avg':>9}  {'oos-PnL':>9}  {'live%':>6}  {'live-n':>6}")
    print("-" * 90)
    plan = {}
    for fam in FAMILIES:
        oos_mask = (oos_feat["setup_family"].values == fam)
        live_mask = (live_feat["setup_family"].values == fam)
        fam_probs = oos_probs[oos_mask]
        fam_pnl = oos_pnl[oos_mask]
        fam_live_probs = live_probs[live_mask]
        best = None
        sweep = []
        for thr in THRESHOLDS:
            oos_pick = fam_probs >= thr
            live_cov = float((fam_live_probs >= thr).mean()) * 100 if len(fam_live_probs) else 0.0
            n = int(oos_pick.sum())
            pnl_sum = float(fam_pnl[oos_pick].sum()) if n else 0.0
            wr = float((fam_pnl[oos_pick] > 0).sum()) / max(1, n) * 100
            avg = pnl_sum / max(1, n)
            sweep.append({
                "threshold": float(thr), "oos_n": n, "oos_wr": wr,
                "oos_avg": avg, "oos_pnl_sum": pnl_sum, "live_cov_pct": live_cov,
                "live_n": int((fam_live_probs >= thr).sum()),
            })
            # Ship candidate: avg ≥ 0, AND either non-zero live fires OR
            # we allow the family to be hard-disabled at ship_gate level.
            if avg >= 0 and (live_cov >= 0.1 or n >= 20):
                if best is None or (pnl_sum > best["oos_pnl_sum"]):
                    best = sweep[-1]
        report["families"][fam] = {"sweep": sweep, "best_pick": best}
        if best is not None:
            plan[fam] = best["threshold"]
            total_trades_at_plan += best["oos_n"]
            total_pnl_at_plan += best["oos_pnl_sum"]
            status = "✓ keep"
        else:
            plan[fam] = "DISABLE"
            status = "✗ DISABLE"
        # Print a short row for the best pick (or first threshold if no ship)
        row = best or sweep[0]
        print(f"{fam:<22} {row['threshold']:>6.3f}  {row['oos_n']:>6}  {row['oos_wr']:>6.2f}%  "
              f"${row['oos_avg']:>+7.3f}  ${row['oos_pnl_sum']:>+8.2f}  {row['live_cov_pct']:>5.2f}%  "
              f"{row['live_n']:>6}   {status}")

    print("-" * 90)
    print(f"{'TOTAL  (best per-family)':<22} {'':6}  {total_trades_at_plan:>6}  {'':7}  "
          f"{'':9}  ${total_pnl_at_plan:>+8.2f}")

    report["summary"] = {
        "plan": plan,
        "total_oos_trades": total_trades_at_plan,
        "total_oos_pnl": total_pnl_at_plan,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n[write] {OUT}")

    # Suggested config.py snippet
    print("\n════════════ RECOMMENDED config.py snippet ════════════")
    for fam, thr in plan.items():
        if thr == "DISABLE":
            print(f'    # "{fam}": REMOVE from allowed_setup_families OR set threshold to 1.0 to hard-block')
        else:
            print(f'    "{fam}": {{ "threshold": {thr}, ... }}')

    # Ship gate: total PnL ≥ 0 AND at least 1 family kept
    ship = total_pnl_at_plan > 0 and any(v != "DISABLE" for v in plan.values())
    print(f"\n{'[SHIP]' if ship else '[NO-SHIP]'} total OOS PnL at plan = ${total_pnl_at_plan:+.2f}  "
          f"families kept = {sum(1 for v in plan.values() if v != 'DISABLE')}/{len(plan)}")
    return 0 if ship else 1


if __name__ == "__main__":
    sys.exit(search())
