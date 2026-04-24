#!/usr/bin/env python3
"""v5 post-mortem: why did B and C fail?

For each model, compute oracle labels on the OOS slice and compare to v5 ML
predictions. Surface the specific failure mode:
  - B: over-fires on bars that were actually profitable at natural size
  - C: can't distinguish BE-helps from BE-hurts in the noise of unreached-BE trades

Outputs diagnostic rows + summary to artifacts/regime_ml_v5_postmortem.json.
"""
from __future__ import annotations
import pickle, sys, json
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path("/Users/wes/Downloads/JULIE001")
sys.path.insert(0, str(ROOT / "scripts"))

from ml_regime_v5_three_models import (
    load_and_featurize, build_size_labels, build_be_labels,
    simulate_be_trade, FEATURE_COLS, TRAIN_END, OOS_START, OOS_END,
    NATURAL_SIZE, PNL_LOOKAHEAD_BARS, MES_PT_VALUE,
    BE_TP, BE_SL, BE_TRIGGER_MFE, SAMPLE_EVERY,
)
from ml_regime_classifier_v4 import (
    simulate_trade, DEFAULT_TP, DEFAULT_SL,
    DEAD_TAPE_VOL_BP,
)

import lightgbm as lgb


def main():
    bars_all, feats_all = load_and_featurize()
    labeled_b = build_size_labels(bars_all, feats_all)
    labeled_c = build_be_labels(bars_all, feats_all)

    # === MODEL B POSTMORTEM ===
    print("\n═══ Model B post-mortem ═══")
    tr_cut = pd.Timestamp(TRAIN_END, tz=labeled_b.index.tz)
    oos_s  = pd.Timestamp(OOS_START, tz=labeled_b.index.tz)
    tr_b = labeled_b.loc[labeled_b.index <= tr_cut]
    oos_b = labeled_b.loc[labeled_b.index >= oos_s]

    # Oracle label vs v5 label
    pnl_nat = oos_b["pnl_natural"].to_numpy()
    pnl_red = oos_b["pnl_reduced"].to_numpy()
    v5_label = oos_b["label_b"].to_numpy()
    # Oracle = which choice ACTUALLY produces higher PnL (no arbitrary threshold)
    oracle = np.where(pnl_red >= pnl_nat, "reduce", "natural")

    # When v5 label says reduce but oracle says natural, model "B" is over-firing
    disagreement = v5_label != oracle
    over_fire = (v5_label == "reduce") & (oracle == "natural")
    under_fire = (v5_label == "natural") & (oracle == "reduce")
    print(f"  OOS rows: {len(oos_b)}")
    print(f"  v5 label dist:     {dict(Counter(v5_label))}")
    print(f"  oracle label dist: {dict(Counter(oracle))}")
    print(f"  disagreement:      {disagreement.sum()} ({disagreement.sum()/len(oos_b)*100:.1f}%)")
    print(f"  over-fires  (v5 reduce, oracle natural): {over_fire.sum()} "
          f"({over_fire.sum()/len(oos_b)*100:.1f}%)")
    print(f"  under-fires (v5 natural, oracle reduce): {under_fire.sum()}")

    # PnL cost of over-fires: natural minus reduced (what we gave up by reducing)
    cost_over = float((pnl_nat[over_fire] - pnl_red[over_fire]).sum())
    gain_under = float((pnl_red[under_fire] - pnl_nat[under_fire]).sum())
    print(f"  over-fire PnL cost (give-up): ${cost_over:,.2f}")
    print(f"  under-fire PnL gain missed:  ${gain_under:,.2f}")

    # Fix test: what's the AMBIGUOUS_MARGIN impact? Drop bars where
    # |pnl_red - pnl_nat| < 30 (low-signal bars)
    margin = pnl_red - pnl_nat
    clear_signal = np.abs(margin) >= 30.0
    print(f"  bars with |margin| >= $30: {clear_signal.sum()} "
          f"({clear_signal.sum()/len(oos_b)*100:.1f}%)")
    # On clear-signal bars only, how much edge is the oracle?
    oracle_clear_pnl = np.where(oracle[clear_signal] == "reduce",
                                  pnl_red[clear_signal], pnl_nat[clear_signal])
    natural_clear_pnl = pnl_nat[clear_signal]
    print(f"  clear-signal bars: oracle PnL=${oracle_clear_pnl.sum():+.2f}  "
          f"natural PnL=${natural_clear_pnl.sum():+.2f}  "
          f"oracle lift=${oracle_clear_pnl.sum() - natural_clear_pnl.sum():+.2f}")

    # === MODEL C POSTMORTEM ===
    print("\n═══ Model C post-mortem ═══")
    tr_c = labeled_c.loc[labeled_c.index <= tr_cut]
    oos_c = labeled_c.loc[labeled_c.index >= oos_s]
    pnl_off = oos_c["pnl_be_off"].to_numpy()
    pnl_on  = oos_c["pnl_be_on"].to_numpy()
    v5_label_c = oos_c["label_c"].to_numpy()
    oracle_c = np.where(pnl_off >= pnl_on, "disable", "keep")

    print(f"  OOS rows: {len(oos_c)}")
    print(f"  v5 label dist:     {dict(Counter(v5_label_c))}")
    print(f"  oracle label dist: {dict(Counter(oracle_c))}")

    # Check conditional: how many OOS rows had forward trades that REACHED BE trigger?
    # Re-simulate: for each bar, count trades in 15-min window that hit MFE >= 5
    c = bars_all["close"].to_numpy(float)
    h = bars_all["high"].to_numpy(float)
    l = bars_all["low"].to_numpy(float)
    idx_pos = {ts: i for i, ts in enumerate(bars_all.index)}
    reached_be = np.zeros(len(oos_c), dtype=int)
    total_trades = np.zeros(len(oos_c), dtype=int)
    for ri, ts in enumerate(oos_c.index):
        start_pos = idx_pos.get(ts)
        if start_pos is None: continue
        # For each of 3 trades in the 15-min window, check if MFE >= 5
        for bar_offset in (0, SAMPLE_EVERY, 2 * SAMPLE_EVERY):
            pj = start_pos + bar_offset
            if pj + PNL_LOOKAHEAD_BARS >= len(c): continue
            for side in (+1, -1):
                total_trades[ri] += 1
                entry = c[pj]
                if side > 0:
                    mfe = float(h[pj+1 : pj+1+PNL_LOOKAHEAD_BARS].max() - entry)
                else:
                    mfe = float(entry - l[pj+1 : pj+1+PNL_LOOKAHEAD_BARS].min())
                if mfe >= BE_TRIGGER_MFE:
                    reached_be[ri] += 1

    be_reach_rate = reached_be.sum() / max(1, total_trades.sum())
    print(f"  Fraction of trades that reach BE trigger (MFE>=5): "
          f"{reached_be.sum()}/{total_trades.sum()} = {be_reach_rate*100:.1f}%")
    # Conditional label: only bars where ≥1 trade in window reached BE
    cond_mask = reached_be >= 1
    print(f"  OOS bars with ≥1 BE-reaching trade: {cond_mask.sum()} "
          f"({cond_mask.sum()/len(oos_c)*100:.1f}%)")
    # How discriminative is BE-off vs BE-on on those conditional bars?
    cond_diff = (pnl_off - pnl_on)[cond_mask]
    print(f"  On conditional bars: BE-off vs BE-on PnL diff  "
          f"mean=${cond_diff.mean():+.2f}  std=${cond_diff.std():.2f}")
    non_cond_diff = (pnl_off - pnl_on)[~cond_mask]
    print(f"  On NON-conditional bars: PnL diff mean=${non_cond_diff.mean():+.2f}  "
          f"std=${non_cond_diff.std():.2f}  "
          f"(should be ~0 — BE doesn't matter when trigger not reached)")

    # Label quality: on conditional bars, is BE-off/BE-on meaningfully signed?
    cond_clear = np.abs(cond_diff) >= 15.0  # clear signal threshold
    print(f"  Clear-signal conditional bars (|diff|>=$15): {cond_clear.sum()} "
          f"({cond_clear.sum()/max(1,cond_mask.sum())*100:.1f}% of conditionals)")

    # Write summary
    out = ROOT / "artifacts" / "regime_ml_v5_postmortem.json"
    summary = {
        "model_b": {
            "oos_rows": int(len(oos_b)),
            "disagreement_pct": float(disagreement.mean() * 100),
            "over_fire_pct":  float(over_fire.mean() * 100),
            "under_fire_pct": float(under_fire.mean() * 100),
            "over_fire_cost_usd": cost_over,
            "under_fire_gain_missed_usd": gain_under,
            "clear_signal_bars": int(clear_signal.sum()),
            "clear_signal_pct": float(clear_signal.mean() * 100),
        },
        "model_c": {
            "oos_rows": int(len(oos_c)),
            "be_reach_rate_pct": float(be_reach_rate * 100),
            "oos_bars_with_be_reaching_trade": int(cond_mask.sum()),
            "conditional_pct": float(cond_mask.mean() * 100),
            "cond_diff_mean": float(cond_diff.mean()),
            "cond_diff_std": float(cond_diff.std()),
            "clear_signal_conditional_pct": float(cond_clear.mean() * 100)
                                             if cond_mask.sum() > 0 else 0.0,
        },
    }
    out.write_text(json.dumps(summary, indent=2))
    print(f"\n[write] {out}")


if __name__ == "__main__":
    main()
