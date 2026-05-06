"""Regime-stratified Tier A — does Hougaard alignment predict trade quality
in bear/high-vol regimes specifically?

Headline Tier A on VAL showed aligned PF (5.79) < neutral PF (6.57) — alignment
doesn't help on average. But the original context analysis showed Scenario B's
edge concentrated in bear/high-vol regimes. So slice Tier A by regime to test
if the conditional edge survives at the model-level.

Also adds the cleanest control comparison: for each base_thr + delta config,
compare overlay marginal trades to a "naive lower threshold by delta" control
(no Hougaard at all). If overlay-marginal PF > naive-marginal PF, the bias
adds incremental edge over just relaxing the threshold.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path('/Users/wes/Downloads/JULIE001')
sys.path.insert(0, str(ROOT / 'tools'))
from stdev_ml_iter11_atr import PERIODS, PT_USD
from hougaard_context_offline import build_context_table
from hougaard_overlay_backtest import (
    run_inference_on_period, simulate_trades, MODEL_DIR, TP_MULT, SL_MULT, OUT,
)

warnings.filterwarnings("ignore")


def regime_stratified_tier_a(preds, ctx, base_thr=0.75, label='VAL'):
    """For each regime cell × bias-bucket, compute trade metrics."""
    sim = simulate_trades(preds, base_thr, 0.0, ctx)
    fired = sim[sim['fired']].copy()
    fired['bucket'] = np.where(
        (fired['bias_dir'] != 0) & (fired['signal_dir'] == fired['bias_dir']), 'aligned',
        np.where(
            (fired['bias_dir'] != 0) & (fired['signal_dir'] != fired['bias_dir']), 'opposed',
            'neutral'))

    def bucket_metrics(sub, regime, bucket):
        n = len(sub)
        if n == 0:
            return {'period': label, 'regime': regime, 'bucket': bucket, 'n': 0}
        wins = int(sub['win'].sum())
        gw = float(sub.loc[sub['win'], 'pnl'].sum())
        gl = float(-sub.loc[~sub['win'], 'pnl'].sum())
        pf = gw / gl if gl > 0 else (float('inf') if gw > 0 else 0)
        return {
            'period': label, 'regime': regime, 'bucket': bucket,
            'n': n, 'wr': wins / n,
            'pnl': float(sub['pnl'].sum()),
            'avg': float(sub['pnl'].sum() / n),
            'pf': pf,
        }

    rows = []
    for regime_label, regime_mask in [
        ('all',         pd.Series(True, index=fired.index)),
        ('bull',        fired['bull_regime']),
        ('bear',        ~fired['bull_regime']),
        ('high_vol',    fired['high_vol']),
        ('low_vol',     ~fired['high_vol']),
        ('bear_or_hv',  ~fired['bull_regime'] | fired['high_vol']),
        ('bull_lv',     fired['bull_regime'] & ~fired['high_vol']),
    ]:
        regime_subset = fired[regime_mask]
        for bucket in ['aligned', 'opposed', 'neutral']:
            sub = regime_subset[regime_subset['bucket'] == bucket]
            rows.append(bucket_metrics(sub, regime_label, bucket))
    return pd.DataFrame(rows)


def control_comparison(preds, ctx, base_thr=0.75, deltas=(-0.05, -0.08, -0.12), label='VAL'):
    """Compare overlay-marginal trades to a naive 'just lower the threshold'
    control. The cleanest test of incremental edge from the bias."""
    rows = []
    base_sim = simulate_trades(preds, base_thr, 0.0, ctx)
    for delta in deltas:
        # Overlay: aligned-only threshold tilt
        overlay_sim = simulate_trades(preds, base_thr, delta, ctx)
        overlay_marg = overlay_sim[overlay_sim['fired'] & ~base_sim['fired']]

        # Control: just drop the threshold by `delta` for everyone (no bias)
        control_thr = base_thr + delta  # delta is negative
        control_sim = simulate_trades(preds, control_thr, 0.0, ctx)
        # Marginal = trades that fire at control but not at base
        control_marg = control_sim[control_sim['fired'] & ~base_sim['fired']]

        # Within control_marg, split into "aligned" and "rest" — this is the
        # apples-to-apples comparison: among bars in [base_thr+delta, base_thr),
        # do the Hougaard-aligned ones win more than the non-aligned?
        control_marg = control_marg.copy()
        control_marg['bucket'] = np.where(
            (control_marg['bias_dir'] != 0) & (control_marg['signal_dir'] == control_marg['bias_dir']), 'aligned',
            np.where(
                (control_marg['bias_dir'] != 0) & (control_marg['signal_dir'] != control_marg['bias_dir']), 'opposed',
                'neutral'))

        for bucket in ['all', 'aligned', 'opposed', 'neutral']:
            sub = control_marg if bucket == 'all' else control_marg[control_marg['bucket'] == bucket]
            n = len(sub)
            if n == 0:
                rows.append({'period': label, 'base_thr': base_thr, 'delta': delta,
                              'bucket': bucket, 'n': 0})
                continue
            wins = int(sub['win'].sum())
            gw = float(sub.loc[sub['win'], 'pnl'].sum())
            gl = float(-sub.loc[~sub['win'], 'pnl'].sum())
            pf = gw / gl if gl > 0 else (float('inf') if gw > 0 else 0)
            rows.append({
                'period': label, 'base_thr': base_thr, 'delta': delta,
                'bucket': bucket, 'n': n,
                'wr': wins / n,
                'pnl': float(sub['pnl'].sum()),
                'avg': float(sub['pnl'].sum() / n),
                'pf': pf,
            })
    return pd.DataFrame(rows)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--val-start', default=PERIODS['VAL_biden'][0])
    ap.add_argument('--val-end',   default=PERIODS['VAL_biden'][1])
    ap.add_argument('--oos-start', default=PERIODS['OOS_2026'][0])
    ap.add_argument('--oos-end',   default=PERIODS['OOS_2026'][1])
    ap.add_argument('--val-label', default='VAL')
    ap.add_argument('--oos-label', default='OOS')
    ap.add_argument('--out-suffix', default='')
    args = ap.parse_args()
    sfx = args.out_suffix

    t0 = time.time()
    print('[regime-drill] loading models...')
    model_l = joblib.load(MODEL_DIR / f'iter12_{TP_MULT}_{SL_MULT}_L.pkl')
    model_s = joblib.load(MODEL_DIR / f'iter12_{TP_MULT}_{SL_MULT}_S.pkl')

    print('[regime-drill] building context...')
    ctx_start = min(args.val_start, args.oos_start, '2017-01-01')
    ctx_end = max(args.val_end, args.oos_end, '2026-05-06')
    ctx = build_context_table(ctx_start, ctx_end)

    val = run_inference_on_period(args.val_label, args.val_start, args.val_end, model_l, model_s)
    oos = run_inference_on_period(args.oos_label, args.oos_start, args.oos_end, model_l, model_s)

    print(f'\n[regime-drill] Tier A by regime — {args.val_label}...')
    a_val = regime_stratified_tier_a(val, ctx, 0.75, args.val_label)
    print(a_val.to_string(index=False))

    print(f'\n[regime-drill] Tier A by regime — {args.oos_label}...')
    a_oos = regime_stratified_tier_a(oos, ctx, 0.75, args.oos_label)
    print(a_oos.to_string(index=False))

    a_full = pd.concat([a_val, a_oos], ignore_index=True)
    a_full.to_csv(OUT / f'tier_a_regime_drilldown{sfx}.csv', index=False)

    # ── headline insight: aligned vs neutral PF lift by regime ──
    print('\n[regime-drill] Aligned PF − Neutral PF (the edge by regime):')
    for label, df in [(args.val_label, a_val), (args.oos_label, a_oos)]:
        print(f'\n  {label}:')
        for regime in ['all', 'bull', 'bear', 'high_vol', 'low_vol', 'bear_or_hv', 'bull_lv']:
            ali = df[(df['regime'] == regime) & (df['bucket'] == 'aligned')]
            neu = df[(df['regime'] == regime) & (df['bucket'] == 'neutral')]
            if len(ali) == 0 or len(neu) == 0:
                continue
            n_ali = ali.iloc[0].get('n', 0); n_neu = neu.iloc[0].get('n', 0)
            if n_ali == 0 or n_neu == 0:
                continue
            pf_ali = ali.iloc[0].get('pf', 0); pf_neu = neu.iloc[0].get('pf', 0)
            wr_ali = ali.iloc[0].get('wr', 0); wr_neu = neu.iloc[0].get('wr', 0)
            avg_ali = ali.iloc[0].get('avg', 0); avg_neu = neu.iloc[0].get('avg', 0)
            print(f'    {regime:<10s}: '
                  f'ali n={n_ali:>4d} PF={pf_ali:>5.2f} WR={wr_ali*100:>4.1f}% avg=${avg_ali:>+5.0f}  |  '
                  f'neu n={n_neu:>4d} PF={pf_neu:>5.2f} WR={wr_neu*100:>4.1f}% avg=${avg_neu:>+5.0f}  |  '
                  f'PF Δ {pf_ali - pf_neu:+.2f} avg Δ ${avg_ali - avg_neu:+.0f}')

    # ── control comparison: overlay-marginal vs naive-lowered-threshold ──
    print('\n[regime-drill] Control: overlay-marginal vs naive lower-threshold...')
    print('  KEY: among trades in [base+delta, base), do aligned ones outperform neutral?')
    print('  If aligned > neutral by PF — bias has incremental edge over just lowering thr.')
    c_val = control_comparison(val, ctx, 0.75, (-0.05, -0.08, -0.12), args.val_label)
    c_oos = control_comparison(oos, ctx, 0.75, (-0.05, -0.08, -0.12), args.oos_label)
    print(f'\n{args.val_label}:')
    print(c_val.to_string(index=False))
    print(f'\n{args.oos_label}:')
    print(c_oos.to_string(index=False))

    pd.concat([c_val, c_oos], ignore_index=True).to_csv(
        OUT / f'control_comparison{sfx}.csv', index=False)

    # Final write-up
    lines = ['=' * 90, 'REGIME-STRATIFIED TIER A — does alignment edge concentrate in bear/HV?',
             '=' * 90, '\nAligned PF − Neutral PF by regime (positive = bias predictive):']
    for label, df in [(args.val_label, a_val), (args.oos_label, a_oos)]:
        lines.append(f'\n{label}:')
        for regime in ['all', 'bull', 'bear', 'high_vol', 'low_vol', 'bear_or_hv', 'bull_lv']:
            ali = df[(df['regime'] == regime) & (df['bucket'] == 'aligned')]
            neu = df[(df['regime'] == regime) & (df['bucket'] == 'neutral')]
            if len(ali) == 0 or len(neu) == 0:
                continue
            n_a = ali.iloc[0].get('n', 0); n_n = neu.iloc[0].get('n', 0)
            if n_a == 0 or n_n == 0:
                continue
            pf_a = ali.iloc[0].get('pf', 0); pf_n = neu.iloc[0].get('pf', 0)
            avg_a = ali.iloc[0].get('avg', 0); avg_n = neu.iloc[0].get('avg', 0)
            lines.append(f'  {regime:<10s}: '
                        f'ali n={n_a:>4d} PF={pf_a:>5.2f}  '
                        f'neu n={n_n:>4d} PF={pf_n:>5.2f}  '
                        f'Δ PF {pf_a - pf_n:+.2f}, Δ avg ${avg_a - avg_n:+.0f}')
    lines.extend(['\n', '=' * 90,
                  'CONTROL: among bars in [base+delta, base), aligned vs neutral',
                  'If aligned PF > neutral PF: bias has incremental edge.', '=' * 90,
                  f'\n{args.val_label}:', c_val.to_string(index=False),
                  f'\n{args.oos_label}:', c_oos.to_string(index=False)])
    (OUT / f'regime_drilldown_summary{sfx}.txt').write_text('\n'.join(lines))

    print(f'\n[regime-drill] wall: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
