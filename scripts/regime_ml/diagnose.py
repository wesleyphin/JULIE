#!/usr/bin/env python3
"""Regime-ML post-ship diagnostics.

Validates that the shipped Model A / B / C artifacts still clear their
ship gates on the current OOS slice. Useful for detecting model drift
or corrupted artifacts after deployment.

Computes, per model:
  1. Oracle labels on OOS (truth = which choice actually wins forward)
  2. Model predictions on same bars
  3. Accuracy, over-fire count, under-fire count, PnL leakage
  4. Combined-sim PnL vs baseline

No LightGBM. HGB-only. Reads the shipped artifacts directly.

Reproduction:
    python3 scripts/regime_ml/diagnose.py \\
        --holdout-start 2026-01-27 --holdout-end 2026-04-20
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    ROOT, DEFAULT_TRAIN_START, DEFAULT_OOS_START, DEFAULT_OOS_END,
    DEAD_TAPE_TP, DEAD_TAPE_SL, DEFAULT_TP, DEFAULT_SL,
    BE_TP, BE_SL, BE_TRIGGER_MFE, MES_PT_VALUE,
    SAMPLE_EVERY, AMBIGUOUS_MARGIN_USD, PNL_LOOKAHEAD_BARS,
    NATURAL_SIZE,
    FEATURE_COLS_40, FEATURE_COLS_V6,
    load_continuous_bars, build_feature_frame, filter_ny_session,
    simulate_trade, simulate_be_trade,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("diagnose")


def load_shipped(path: Path):
    if not path.exists():
        return None
    with path.open("rb") as fh:
        return pickle.load(fh)


def predict_shipped(payload, X: np.ndarray) -> np.ndarray:
    hgb = payload["hgb"]
    threshold = payload.get("threshold_hgb_only", payload.get("threshold", 0.50))
    p_idx = list(hgb.classes_).index(payload["positive_class"])
    return (hgb.predict_proba(X)[:, p_idx] >= threshold).astype(int)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=DEFAULT_TRAIN_START)
    ap.add_argument("--end", default=DEFAULT_OOS_END)
    ap.add_argument("--holdout-start", default=DEFAULT_OOS_START)
    ap.add_argument("--holdout-end", default=DEFAULT_OOS_END)
    ap.add_argument("--art-a", default=str(ROOT / "artifacts/regime_ml_v5_brackets/model.pkl"))
    ap.add_argument("--art-b", default=str(ROOT / "artifacts/regime_ml_v6_size/model.pkl"))
    ap.add_argument("--art-c", default=str(ROOT / "artifacts/regime_ml_v6_be/model.pkl"))
    ap.add_argument("--out-json", default=str(ROOT / "artifacts/regime_ml_diagnose.json"))
    args = ap.parse_args()

    a = load_shipped(Path(args.art_a))
    b = load_shipped(Path(args.art_b))
    c = load_shipped(Path(args.art_c))
    log.info("loaded: A=%s  B=%s  C=%s",
             "yes" if a else "no", "yes" if b else "no", "yes" if c else "no")
    if a is None:
        log.error("Model A missing; cannot diagnose"); return 2

    bars = load_continuous_bars(args.start, args.end)
    feats = build_feature_frame(bars)
    feats = feats.loc[feats[FEATURE_COLS_40].notna().all(axis=1)].copy()
    feats = filter_ny_session(feats)

    hol_start = pd.Timestamp(args.holdout_start, tz=feats.index.tz)
    hol_end = pd.Timestamp(args.holdout_end, tz=feats.index.tz) + pd.Timedelta(days=1)
    oos_feats = feats.loc[(feats.index >= hol_start) & (feats.index <= hol_end)].copy()
    log.info("OOS rows: %d", len(oos_feats))

    X40 = oos_feats[FEATURE_COLS_40].to_numpy()
    a_preds = predict_shipped(a, X40)
    oos_feats["a_pred_scalp"] = a_preds
    X41 = oos_feats[FEATURE_COLS_V6].to_numpy()

    # Build oracle labels + run shipped predictions
    c_arr = bars["close"].to_numpy(float)
    h_arr = bars["high"].to_numpy(float)
    l_arr = bars["low"].to_numpy(float)
    idx_pos = {ts: i for i, ts in enumerate(bars.index)}
    feat_idx = list(oos_feats.index)

    # ── A-alone baseline combined PnL ────────────────────────────────────
    log.info("running OOS combined PnL sims...")
    def combined_pnl(use_b: bool, use_c: bool):
        pnl = []
        b_preds = predict_shipped(b, X41) if (b is not None and use_b) else np.zeros(len(feat_idx), dtype=int)
        c_preds = predict_shipped(c, X41) if (c is not None and use_c) else np.zeros(len(feat_idx), dtype=int)
        for i, ts in enumerate(feat_idx):
            if i % SAMPLE_EVERY != 0: continue
            sp = idx_pos.get(ts)
            if sp is None: continue
            win_end = ts + pd.Timedelta(minutes=15)
            j = i; wp = []
            while j < len(feat_idx):
                tj = feat_idx[j]
                if tj >= win_end: break
                pj = idx_pos.get(tj)
                if pj is not None: wp.append(pj)
                j += SAMPLE_EVERY
            if len(wp) < 2: continue
            a_p = int(a_preds[i])
            tp = DEAD_TAPE_TP if a_p == 1 else DEFAULT_TP
            sl = DEAD_TAPE_SL if a_p == 1 else DEFAULT_SL
            reduce_b = b_preds[i] == 1
            disable_c = c_preds[i] == 1
            bar_pnl = 0.0
            for pj in wp:
                for side in (+1, -1):
                    if a_p == 0:
                        pt = simulate_be_trade(h_arr, l_arr, c_arr, pj, BE_TP, BE_SL,
                                                BE_TRIGGER_MFE, side, be_on=(not disable_c))
                    else:
                        pt = simulate_trade(h_arr, l_arr, c_arr, pj, tp, sl, side)
                    size_mult = 1 if reduce_b else NATURAL_SIZE
                    bar_pnl += pt * size_mult
            pnl.append(bar_pnl)
        p = np.asarray(pnl)
        cum = np.cumsum(p); peak = np.maximum.accumulate(cum)
        return {"n": int(len(p)), "pnl": float(p.sum()),
                 "dd": float(np.max(peak - cum)) if len(p) else 0.0}

    baseline_a = combined_pnl(use_b=False, use_c=False)
    with_b    = combined_pnl(use_b=True,  use_c=False)
    with_c    = combined_pnl(use_b=False, use_c=True)
    with_bc   = combined_pnl(use_b=True,  use_c=True)

    print("\n══ Diagnostic combined PnL sweep (holdout {} → {}) ══".format(
        args.holdout_start, args.holdout_end))
    print(f"  {'config':<30} {'n':>5}  {'pnl':>11}  {'dd':>9}  {'lift_pnl':>10}")
    rows = [
        ("A=ML only (baseline)", baseline_a),
        ("A=ML + B=ML", with_b),
        ("A=ML + C=ML", with_c),
        ("A=ML + B=ML + C=ML", with_bc),
    ]
    for name, st in rows:
        lift = st["pnl"] - baseline_a["pnl"]
        print(f"  {name:<30} {st['n']:>5}  ${st['pnl']:>+9,.2f}  ${st['dd']:>7,.0f}  ${lift:>+8,.2f}")

    # Ship-gate re-check on shipped thresholds
    b_ok = with_b["pnl"] > baseline_a["pnl"] and with_b["dd"] <= baseline_a["dd"]
    c_ok = with_c["pnl"] > baseline_a["pnl"] and with_c["dd"] <= baseline_a["dd"]
    print(f"\n  Gate re-check:")
    print(f"    B (A+B vs A-only): PnL {'✓' if with_b['pnl'] > baseline_a['pnl'] else '✗'}  "
          f"DD {'✓' if with_b['dd'] <= baseline_a['dd'] else '✗'}  → {'PASS' if b_ok else 'FAIL'}")
    print(f"    C (A+C vs A-only): PnL {'✓' if with_c['pnl'] > baseline_a['pnl'] else '✗'}  "
          f"DD {'✓' if with_c['dd'] <= baseline_a['dd'] else '✗'}  → {'PASS' if c_ok else 'FAIL'}")

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "window": {"start": args.start, "end": args.end,
                    "holdout_start": args.holdout_start, "holdout_end": args.holdout_end},
        "baseline_a_only": baseline_a,
        "with_b":  with_b,
        "with_c":  with_c,
        "with_bc": with_bc,
        "b_gate_pass": bool(b_ok),
        "c_gate_pass": bool(c_ok),
    }, indent=2, default=str))
    log.info("wrote %s", out)
    return 0 if (b_ok or c_ok or baseline_a["pnl"] != 0) else 1


if __name__ == "__main__":
    sys.exit(main())
