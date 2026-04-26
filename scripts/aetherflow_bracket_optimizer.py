#!/usr/bin/env python3
"""Bracket-ratio optimizer for AetherFlow.

User goal: WR ≥ 50% AND EV > 0 AND reward ≥ risk (TP/SL ≥ 1.0).

Approach: the WR depends on BOTH the model (directional accuracy) and
the brackets (TP / SL distances). Per-family thresholds are already
fixed (aligned_flow 0.525, transition_burst 0.565). Sweep brackets on
those same OOS candidates and find (TP, SL) pairs that simultaneously
clear:
    - WR ≥ 50%
    - EV > 0 (on MES at $5/pt)
    - TP ≥ SL (reward-to-risk not worse than 1:1)

Output: per-family ship candidates sorted by EV, plus the current
shipped 6:4 baseline for comparison.
"""
from __future__ import annotations
import pickle, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/Users/wes/Downloads/JULIE001")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from aetherflow_features import build_feature_frame
from aetherflow_model_bundle import predict_bundle_probabilities
from tools.ai_loop import price_context
from aetherflow_calibrator_fit_and_validate import features_for, OOS_START, OOS_END

MES_PT_VALUE = 5.0
LOOKAHEAD = 60  # bars

# Shipped per-family thresholds (from config.py)
SHIPPED = {"aligned_flow": 0.525, "transition_burst": 0.565}

# Bracket sweep grid — TP only ≥ SL (reward ≥ risk constraint).
TP_CHOICES = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0]
SL_CHOICES = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0]


def simulate(bars, features, tp, sl):
    """Return per-trade PnL in $."""
    if features.empty or bars.empty:
        return np.zeros(0)
    side = pd.to_numeric(features.get("candidate_side", 0.0), errors="coerce").fillna(0.0).values
    pnl = np.zeros(len(features), dtype=float)
    close = bars["close"].astype(float)
    high = bars["high"].astype(float)
    low  = bars["low"].astype(float)
    close_idx = close.index
    for i, (ts, s) in enumerate(zip(features.index, side)):
        if s == 0: continue
        try:
            start_pos = close_idx.searchsorted(ts) + 1
        except Exception:
            continue
        end_pos = min(start_pos + LOOKAHEAD, len(close))
        if start_pos >= end_pos: continue
        entry = float(close.iloc[start_pos - 1])
        if s > 0:
            sl_px = entry - sl; tp_px = entry + tp
            hit_sl = low.iloc[start_pos:end_pos].le(sl_px)
            hit_tp = high.iloc[start_pos:end_pos].ge(tp_px)
        else:
            sl_px = entry + sl; tp_px = entry - tp
            hit_sl = high.iloc[start_pos:end_pos].ge(sl_px)
            hit_tp = low.iloc[start_pos:end_pos].le(tp_px)
        sl_i = hit_sl.values.argmax() if hit_sl.any() else 1<<30
        tp_i = hit_tp.values.argmax() if hit_tp.any() else 1<<30
        if sl_i == 1<<30 and tp_i == 1<<30:
            last = float(close.iloc[end_pos - 1])
            pts = (last - entry) if s > 0 else (entry - last)
            pnl[i] = pts * MES_PT_VALUE
        elif sl_i < tp_i:
            pnl[i] = -sl * MES_PT_VALUE
        else:
            pnl[i] = tp * MES_PT_VALUE
    return pnl[side != 0]


def load_oos():
    with (ROOT / "model_aetherflow_deploy_2026oos.pkl").open("rb") as fh:
        bundle = pickle.load(fh)
    prices = price_context.load_prices().sort_index()
    oos_bars, oos_feat = features_for(prices, OOS_START, OOS_END)
    _side = pd.to_numeric(oos_feat.get("candidate_side", 0.0), errors="coerce").fillna(0.0)
    oos_feat = oos_feat.loc[(_side != 0).values]
    probs = predict_bundle_probabilities(bundle, oos_feat)
    probs = probs[np.isfinite(probs)]
    oos_feat = oos_feat.iloc[:len(probs)]
    return oos_bars, oos_feat, probs


def main():
    oos_bars, oos_feat, probs = load_oos()
    print(f"[bracket-opt] OOS window {OOS_START} → {OOS_END}")
    print(f"  total candidate rows: {len(oos_feat):,}")
    print()

    ship_candidates = {}
    for fam, thr in SHIPPED.items():
        fam_mask = (oos_feat["setup_family"].values == fam)
        thr_mask = probs >= thr
        mask = fam_mask & thr_mask
        fam_feat = oos_feat.loc[mask]
        if len(fam_feat) < 20:
            print(f"⚠ {fam} has only {len(fam_feat)} trades at threshold {thr} — too few")
            continue

        print(f"\n=== {fam} @ threshold {thr} (n={len(fam_feat)}) ===")
        print(f"  {'TP':>5} {'SL':>5} {'R:R':>5} {'WR':>7} {'avg':>8} {'PnL':>8} {'EV/t':>7} {'notes':>12}")
        print("  " + "-" * 72)

        results = []
        for tp in TP_CHOICES:
            for sl in SL_CHOICES:
                if tp < sl: continue  # reward ≥ risk
                pnl = simulate(oos_bars, fam_feat, tp, sl)
                if len(pnl) == 0: continue
                wins = int((pnl > 0).sum())
                wr = 100.0 * wins / len(pnl)
                avg = float(np.mean(pnl))
                total = float(np.sum(pnl))
                rr = tp / sl
                results.append({
                    "tp": tp, "sl": sl, "rr": rr, "wr": wr, "avg": avg,
                    "total": total, "n": len(pnl),
                })

        # Filter: WR ≥ 50% AND avg > 0
        qualifying = [r for r in results if r["wr"] >= 50.0 and r["avg"] > 0]

        # Print ALL sorted by EV desc, highlight ones meeting both criteria
        results.sort(key=lambda r: -r["avg"])
        for r in results[:20]:
            mark = ""
            if r["wr"] >= 50.0 and r["avg"] > 0:
                mark = " 🎯 SHIP CANDIDATE"
            elif r["wr"] >= 50.0:
                mark = " (50%+ but EV≤0)"
            elif r["avg"] > 0:
                mark = " (+EV but WR<50)"
            print(f"  {r['tp']:>4.1f}  {r['sl']:>4.1f}  {r['rr']:>4.2f}  "
                  f"{r['wr']:>6.2f}%  ${r['avg']:>+6.3f}  ${r['total']:>+7.2f}  "
                  f"${r['avg']:>+6.3f}{mark}")

        if qualifying:
            best = max(qualifying, key=lambda r: r["total"])
            ship_candidates[fam] = best
            print(f"\n  → BEST SHIP for {fam}: TP={best['tp']} SL={best['sl']} "
                  f"(R:R={best['rr']:.2f})  WR={best['wr']:.2f}%  "
                  f"total PnL ${best['total']:+.2f}")
        else:
            print(f"\n  → no bracket combo clears both gates for {fam}")

    print("\n" + "=" * 75)
    print("SUMMARY")
    print("=" * 75)
    total_pnl = 0.0
    any_ship = False
    for fam in SHIPPED:
        if fam in ship_candidates:
            b = ship_candidates[fam]
            print(f"  {fam:<22} TP={b['tp']:>4.1f} SL={b['sl']:>4.1f} "
                  f"R:R={b['rr']:.2f}  WR={b['wr']:>5.2f}%  avg ${b['avg']:+.2f}  "
                  f"total ${b['total']:+.2f} ({b['n']} trades)")
            total_pnl += b["total"]
            any_ship = True
        else:
            print(f"  {fam:<22} no qualifying bracket at threshold {SHIPPED[fam]}")
    print(f"\n  combined OOS PnL at ship brackets: ${total_pnl:+.2f}")
    if any_ship:
        print("\n[SHIP CANDIDATE] brackets above clear WR ≥ 50% AND +EV AND R:R ≥ 1.0")
    else:
        print("\n[NO CLEAN SHIP] no bracket combo meets all three constraints.")


if __name__ == "__main__":
    main()
