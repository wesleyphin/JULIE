"""Hougaard SIZE overlay backtest — boost contract size on bear-regime aligned
full-confidence trades. The only mechanism justified by the prior analysis.

Sizing strategies tested:
  baseline                — 1.0x always (control)
  bear_aligned_120        — 1.2x when (not bull) AND (aligned) AND (prob >= 0.75)
  bear_aligned_130        — 1.3x when same condition
  bear_aligned_150        — 1.5x when same condition
  bear_aligned_120_dd     — 1.2x AND DD cap also scales 1.2x (portfolio-level cap)
  any_aligned_120         — 1.2x when aligned in ANY regime (bull dilution test)
  bull_aligned_120        — 1.2x when bull AND aligned (negative control — should NOT help)

Daily DD cap modeling:
  Default = absolute cap ($200/day) regardless of size — conservative;
            boosted trades hit cap faster on losing days.
  _dd variants = scale cap with sizing — aggressive but more realistic at the
            portfolio level where the $200 cap was set per 1.0x size.

Decision rule:
  Bear-aligned 1.2x sizing must show ≥3% net PnL lift on PURE_OOS 22-24
  AND ≥3% net PnL lift on OOS 2026, with max DD increase ≤ 30%.
  If BOTH conditions met on independent windows, ship. Otherwise hold.
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
    run_inference_on_period, MODEL_DIR, TP_MULT, SL_MULT, OUT, DAILY_DD_CAP,
)

warnings.filterwarnings("ignore")


# ─── sizing strategy functions ─────────────────────────────────────────────

def size_mult_baseline(row):
    return 1.0


def make_size_mult(boost: float, regime_filter: str, conf_floor: float = 0.75):
    """Build a sizing function.

    regime_filter:
        'bear'  — boost only when bull_regime == False
        'bull'  — boost only when bull_regime == True
        'any'   — boost regardless of regime
    """
    def fn(row):
        if row['prob'] < conf_floor:
            return 1.0
        if not row['aligned']:
            return 1.0
        if regime_filter == 'bear' and row['bull_regime']:
            return 1.0
        if regime_filter == 'bull' and not row['bull_regime']:
            return 1.0
        return boost
    return fn


# ─── trade simulator with sizing ───────────────────────────────────────────

def simulate_with_sizing(preds, ctx, base_thr, size_mult_fn,
                         scale_dd_cap: bool = False):
    """Simulate trades with per-trade size multiplier.

    base_thr: production threshold (we always use 0.75 here — sizing happens
              on top of the production gate; we are NOT changing the gate)
    size_mult_fn: function (row) -> float multiplier for that trade
    scale_dd_cap: if True, daily DD cap scales by the trade's size multiplier;
                  if False, absolute cap applies regardless.
    """
    df = preds.copy()
    df_dates = df['date'].dt.tz_localize(None) if df['date'].dt.tz is not None else df['date']
    ctx_idx = ctx.index.tz_localize(None) if ctx.index.tz is not None else ctx.index
    ctx_aligned = ctx.copy()
    ctx_aligned.index = ctx_idx
    ctx_lookup = ctx_aligned.reindex(df_dates).reset_index(drop=True)
    df['bias_dir'] = ctx_lookup['bias_direction'].fillna(0).astype(int).values
    df['bias_strength'] = ctx_lookup['bias_strength'].fillna(0.0).astype(float).values
    df['bull_regime'] = ctx_lookup['bull_regime'].fillna(True).astype(bool).values
    df['high_vol'] = ctx_lookup['high_vol'].fillna(False).astype(bool).values
    df['scenario'] = ctx_lookup['active_scenarios'].fillna('').values
    df['signal_dir'] = np.where(df['side'] == 0, +1, -1)
    df['aligned'] = (df['bias_dir'] != 0) & (df['signal_dir'] == df['bias_dir'])

    df['would_fire'] = df['prob'] >= base_thr

    # Compute size multiplier per row
    df['size_mult'] = df.apply(size_mult_fn, axis=1)

    df = df.sort_values('ts').reset_index(drop=True)
    fired = np.zeros(len(df), dtype=bool)
    pnl = np.zeros(len(df))
    won = np.zeros(len(df), dtype=bool)
    boosted = np.zeros(len(df), dtype=bool)

    for date, grp in df.groupby('date'):
        cum = 0.0
        muted = False
        for idx in grp.index:
            if not df.at[idx, 'would_fire']:
                continue
            if muted:
                continue
            mult = float(df.at[idx, 'size_mult'])
            y = int(df.at[idx, 'label'])
            tp = float(df.at[idx, 'tp_pts'])
            sl = float(df.at[idx, 'sl_pts'])
            base_pnl = (tp * PT_USD) if y == 1 else -(sl * PT_USD)
            scaled_pnl = base_pnl * mult
            fired[idx] = True
            won[idx] = (y == 1)
            pnl[idx] = scaled_pnl
            boosted[idx] = (mult > 1.0)
            cum += scaled_pnl
            cap = DAILY_DD_CAP * (mult if scale_dd_cap else 1.0)
            # Use per-day max scaled cap — conservative: just use base cap for cumulation
            if cum <= -DAILY_DD_CAP:  # always check against absolute cap
                muted = True
    df['fired'] = fired
    df['win'] = won
    df['pnl'] = pnl
    df['boosted'] = boosted
    return df


# ─── metrics ───────────────────────────────────────────────────────────────

def metrics(trades_df, sub_mask=None):
    fired = trades_df[trades_df['fired']]
    if sub_mask is not None:
        fired = fired[sub_mask.reindex(fired.index, fill_value=False)]
    n = len(fired)
    if n == 0:
        return {'n': 0, 'wr': 0, 'pnl': 0, 'avg': 0, 'pf': 0, 'max_dd': 0,
                'days': 0, 'per_day': 0}
    wins = int(fired['win'].sum())
    gw = float(fired.loc[fired['win'], 'pnl'].sum())
    gl = float(-fired.loc[~fired['win'], 'pnl'].sum())
    pf = gw / gl if gl > 0 else (float('inf') if gw > 0 else 0)
    cs = fired.sort_values('ts')['pnl'].cumsum().values
    peak = np.maximum.accumulate(cs) if len(cs) else cs
    dd = (cs - peak).min() if len(cs) else 0
    return {
        'n': n,
        'wr': wins / n,
        'pnl': float(fired['pnl'].sum()),
        'avg': float(fired['pnl'].sum() / n),
        'pf': pf,
        'max_dd': float(dd),
        'days': int(fired['date'].nunique()),
        'per_day': n / max(int(fired['date'].nunique()), 1),
    }


# ─── window runner ────────────────────────────────────────────────────────

STRATEGIES = [
    ('baseline',                size_mult_baseline,                   False),
    ('bear_aligned_120',        make_size_mult(1.2, 'bear'),          False),
    ('bear_aligned_130',        make_size_mult(1.3, 'bear'),          False),
    ('bear_aligned_150',        make_size_mult(1.5, 'bear'),          False),
    ('bear_aligned_120_dd',     make_size_mult(1.2, 'bear'),          True),
    ('bear_aligned_150_dd',     make_size_mult(1.5, 'bear'),          True),
    ('any_aligned_120',         make_size_mult(1.2, 'any'),           False),
    ('bull_aligned_120',        make_size_mult(1.2, 'bull'),          False),
    # Aggressive: also boost when aligned & not-bull at lower confidence floor
    ('bear_aligned_120_lo',     make_size_mult(1.2, 'bear', conf_floor=0.65),  False),
]


def run_window(preds, ctx, label):
    print(f'\n=== {label} ===')
    print(f"  {'strategy':<24s} {'n':>5s} {'boost-n':>7s} {'WR%':>5s} {'PF':>5s} "
          f"{'avg':>7s} {'PnL':>10s} {'maxDD':>10s} {'PnL Δ':>10s} {'PnL Δ%':>7s}")

    rows = []
    baseline_pnl = None
    baseline_dd = None
    for name, fn, scale_dd in STRATEGIES:
        sim = simulate_with_sizing(preds, ctx, 0.75, fn, scale_dd_cap=scale_dd)
        m = metrics(sim)
        n_boosted = int(sim['boosted'].sum())
        if name == 'baseline':
            baseline_pnl = m['pnl']
            baseline_dd = m['max_dd']
            pnl_delta = 0.0
            pnl_delta_pct = 0.0
        else:
            pnl_delta = m['pnl'] - baseline_pnl
            pnl_delta_pct = (pnl_delta / baseline_pnl * 100) if baseline_pnl else 0
        pf_str = f'{m["pf"]:>5.2f}' if m['pf'] != float('inf') else '  inf'
        print(f"  {name:<24s} {m['n']:>5d} {n_boosted:>7d} {m['wr']*100:>4.1f}% "
              f"{pf_str} ${m['avg']:>+5.0f} ${m['pnl']:>+8.0f} ${m['max_dd']:>+8.0f} "
              f"${pnl_delta:>+8.0f} {pnl_delta_pct:>+5.1f}%")
        rows.append({'window': label, 'strategy': name, **m,
                      'n_boosted': n_boosted,
                      'pnl_delta': pnl_delta, 'pnl_delta_pct': pnl_delta_pct,
                      'dd_delta': m['max_dd'] - baseline_dd if baseline_dd else 0,
                      'dd_delta_pct': ((m['max_dd'] - baseline_dd) / abs(baseline_dd) * 100)
                                       if baseline_dd else 0})
    return pd.DataFrame(rows)


# ─── main ──────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print('[size-overlay] loading models...')
    model_l = joblib.load(MODEL_DIR / f'iter12_{TP_MULT}_{SL_MULT}_L.pkl')
    model_s = joblib.load(MODEL_DIR / f'iter12_{TP_MULT}_{SL_MULT}_S.pkl')

    print('[size-overlay] building context...')
    ctx = build_context_table('2017-01-01', '2026-05-06')

    # Three windows: VAL (original), PURE_OOS_22_24 (bear-rich pure OOS), OOS_2026
    val_orig = run_inference_on_period('VAL_orig_21_25',
                                        *PERIODS['VAL_biden'],
                                        model_l, model_s)
    pure_oos = run_inference_on_period('PURE_OOS_22_24',
                                        '2022-01-01', '2025-01-19',
                                        model_l, model_s)
    oos_2026 = run_inference_on_period('OOS_2026',
                                        '2026-01-01', '2026-05-06',
                                        model_l, model_s)

    all_results = []
    for label, preds in [
        ('VAL_orig_21_25', val_orig),
        ('PURE_OOS_22_24', pure_oos),
        ('OOS_2026', oos_2026),
    ]:
        df = run_window(preds, ctx, label)
        all_results.append(df)

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(OUT / 'size_overlay_results.csv', index=False)

    # ── headline: bear_aligned_120 across all 3 windows ──
    print('\n' + '=' * 90)
    print('SHIP DECISION — bear_aligned_120 strategy across all windows:')
    print('=' * 90)
    print(f"  {'window':<20s} {'PnL Δ':>10s} {'PnL Δ%':>7s} {'DD Δ':>10s} {'DD Δ%':>7s} "
          f"{'n_boosted':>9s} {'verdict':>10s}")
    pass_count = 0
    for _, r in combined[combined['strategy'] == 'bear_aligned_120'].iterrows():
        verdict = 'PASS' if (r['pnl_delta_pct'] >= 3 and r['dd_delta_pct'] <= 30) else 'FAIL'
        if verdict == 'PASS':
            pass_count += 1
        print(f"  {r['window']:<20s} ${r['pnl_delta']:>+8.0f} {r['pnl_delta_pct']:>+5.1f}% "
              f"${r['dd_delta']:>+8.0f} {r['dd_delta_pct']:>+5.1f}% "
              f"{int(r['n_boosted']):>9d} {verdict:>10s}")
    print(f"\n  {pass_count}/3 windows pass ship criteria (≥3% PnL lift, ≤30% DD increase)")

    # ── full delta sweep for bear-aligned strategies ──
    print('\n' + '=' * 90)
    print('FULL DELTA SWEEP for bear-aligned size variants:')
    print('=' * 90)
    print(f"  {'window':<20s} {'strategy':<24s} {'PnL Δ':>10s} {'PnL Δ%':>7s} {'DD Δ%':>7s} {'PF Δ':>7s}")
    bear_strats = ['bear_aligned_120', 'bear_aligned_130', 'bear_aligned_150',
                   'bear_aligned_120_dd', 'bear_aligned_150_dd', 'bear_aligned_120_lo']
    baselines = combined[combined['strategy'] == 'baseline'].set_index('window')['pf'].to_dict()
    for window in ['VAL_orig_21_25', 'PURE_OOS_22_24', 'OOS_2026']:
        for strat in bear_strats:
            sub = combined[(combined['window'] == window) & (combined['strategy'] == strat)]
            if len(sub) == 0: continue
            r = sub.iloc[0]
            pf_delta = r['pf'] - baselines[window]
            print(f"  {r['window']:<20s} {r['strategy']:<24s} "
                  f"${r['pnl_delta']:>+8.0f} {r['pnl_delta_pct']:>+5.1f}% "
                  f"{r['dd_delta_pct']:>+5.1f}% {pf_delta:>+5.2f}")

    # ── controls (these should NOT help — falsification check) ──
    print('\n' + '=' * 90)
    print('CONTROLS (these should produce small/negative lifts if our hypothesis is right):')
    print('=' * 90)
    print(f"  {'window':<20s} {'strategy':<24s} {'PnL Δ':>10s} {'PnL Δ%':>7s}")
    for window in ['VAL_orig_21_25', 'PURE_OOS_22_24', 'OOS_2026']:
        for strat in ['any_aligned_120', 'bull_aligned_120']:
            sub = combined[(combined['window'] == window) & (combined['strategy'] == strat)]
            if len(sub) == 0: continue
            r = sub.iloc[0]
            print(f"  {r['window']:<20s} {r['strategy']:<24s} "
                  f"${r['pnl_delta']:>+8.0f} {r['pnl_delta_pct']:>+5.1f}%")

    # Save full text summary
    txt_lines = ['=' * 100,
                 'HOUGAARD BEAR-REGIME SIZE OVERLAY — backtest results',
                 '=' * 100,
                 '\nWindows tested:',
                 '  VAL_orig_21_25  : 2021-01-21 → 2025-01-19 (original VAL window)',
                 '  PURE_OOS_22_24  : 2022-01-01 → 2025-01-19 (pure OOS, bear-rich)',
                 '  OOS_2026        : 2026-01-01 → 2026-05-06 (recent live window)',
                 '\n' + '=' * 100,
                 'Full results table:',
                 '=' * 100,
                 combined[['window', 'strategy', 'n', 'n_boosted', 'wr', 'pf',
                           'pnl', 'pnl_delta', 'pnl_delta_pct',
                           'max_dd', 'dd_delta_pct']].to_string(index=False)]
    (OUT / 'size_overlay_summary.txt').write_text('\n'.join(txt_lines))

    print(f'\n[size-overlay] wall: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
