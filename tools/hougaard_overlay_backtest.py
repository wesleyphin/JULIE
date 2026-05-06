"""Hougaard overlay backtest — does conditional bias add P&L to StdevMl iter12?

Pipeline:
  1. Load existing iter12 pickle models (LONG + SHORT, TP/SL = 1.5/1.0).
  2. Replay over VAL_biden + OOS_2026 with the existing label-walk logic.
     (Reuses tools/stdev_ml_iter12_both_sides.py:build_period for features +
      labels; iter11_atr.compute_features for feature engineering.)
  3. For each bar, get p_long, p_short, pick the higher-confidence side.
     Each bar produces a (ts, side, prob, label, tp_pts, sl_pts) record.
  4. Join with Hougaard context table by date.
  5. **Tier A — stratification**: at the production threshold (0.75), split
     baseline trades into aligned / opposed / neutral buckets and compare
     PF, WR, avg PnL.
  6. **Tier B — overlay sweep**: sweep (base_threshold × delta). For each
     (base, delta) pair, compute baseline trades (no overlay) and overlay
     trades (threshold tilted by `delta * strength * amplifier` when bias
     is aligned with the bar's predicted side). Apply daily DD cap = $200
     (production setting). Report:
       - Total trades + trades/day
       - WR, PF, net PnL, max DD
       - "Marginal trades only" (the M trades the overlay added) — the real
         test. If marginal-only PF < 1.0, the overlay is dilutive even if
         total PnL is up.
       - Per-regime, per-scenario breakdowns of the marginal set.

Outputs to artifacts/hougaard_overlay/:
  - tier_a_stratification.csv    (per-bucket metrics at base=0.75)
  - tier_a_summary.txt
  - tier_b_summary.txt           (delta sweep table)
  - tier_b_marginal_trades_only.csv (per-config marginal sub-set metrics)
  - tier_b_per_regime.csv
  - calibration_diagram.txt      (reliability of iter12 by prob bin)

Decision rule (from prior analysis):
  Tier A: aligned PF ≥ neutral PF + 0.10 → bias has predictive content.
  Tier B: marginal-only PF ≥ 1.10 AND OOS marginal-only PnL > 0 → ship.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path('/Users/wes/Downloads/JULIE001')
sys.path.insert(0, str(ROOT / 'tools'))

from stdev_ml_iter11_atr import (
    compute_features, load_period, PERIODS, WIN_HOURS, H_MIN_NEW, PT_USD,
)
from stdev_ml_iter12_both_sides import build_period
from hougaard_context_offline import build_context_table

warnings.filterwarnings("ignore")

OUT = ROOT / 'artifacts' / 'hougaard_overlay'
OUT.mkdir(parents=True, exist_ok=True)
MODEL_DIR = ROOT / 'artifacts' / 'stdev_ml_hr11_12' / 'iter12_both_sides'
TP_MULT, SL_MULT = 1.5, 1.0
DAILY_DD_CAP = 200  # USD per day (production setting)


# ─── inference ──────────────────────────────────────────────────────────────

def run_inference_on_period(name, start, end, model_l, model_s):
    """Generate prediction records for one period.
    Returns DataFrame with one row per bar in WIN window: ts, p_long, p_short,
    side (winner), prob (winner), label (1=TP first, 0=SL first), tp_pts, sl_pts."""
    print(f'\n[infer-{name}] {start}→{end}')
    t0 = time.time()
    df = build_period(name, start, end, TP_MULT, SL_MULT)
    if df is None or df.empty:
        return None

    # build_period returns 2 rows per bar (LONG + SHORT). Split.
    long_rows = df[df['_orig_side'] == 0]
    short_rows = df[df['_orig_side'] == 1]

    feat_l = model_l['features']
    feat_s = model_s['features']

    def prep(rows, feats):
        X = rows[feats].astype(np.float32)
        valid = (X.isna().sum(axis=1) < len(feats) * 0.5)
        X_clean = X[valid].replace([np.inf, -np.inf], np.nan).fillna(0).clip(-1e6, 1e6)
        return X_clean, valid

    Xl, vl = prep(long_rows, feat_l)
    Xs, vs = prep(short_rows, feat_s)

    p_l = model_l['hgb'].predict_proba(Xl.values)[:, 1]
    p_s = model_s['hgb'].predict_proba(Xs.values)[:, 1]

    long_clean = long_rows[vl].copy()
    short_clean = short_rows[vs].copy()
    long_clean['prob'] = p_l
    short_clean['prob'] = p_s
    long_clean['side'] = 0  # LONG
    short_clean['side'] = 1  # SHORT

    # Outer-join on timestamp; some bars may have one side only
    L = long_clean[['prob', 'label', 'tp_pts', 'sl_pts']].rename(columns={
        'prob': 'p_long', 'label': 'y_long', 'tp_pts': 'tp_long', 'sl_pts': 'sl_long'})
    S = short_clean[['prob', 'label', 'tp_pts', 'sl_pts']].rename(columns={
        'prob': 'p_short', 'label': 'y_short', 'tp_pts': 'tp_short', 'sl_pts': 'sl_short'})
    merged = L.join(S, how='outer')
    merged['ts'] = merged.index
    # Fill missing side with NaN; if a side is missing, only the other can fire
    # Determine winner side
    merged['p_long'] = merged['p_long'].fillna(0.0)
    merged['p_short'] = merged['p_short'].fillna(0.0)
    merged['side'] = np.where(merged['p_long'] >= merged['p_short'], 0, 1)
    merged['prob'] = np.where(merged['side'] == 0, merged['p_long'], merged['p_short'])
    merged['label'] = np.where(merged['side'] == 0, merged['y_long'], merged['y_short'])
    merged['tp_pts'] = np.where(merged['side'] == 0, merged['tp_long'], merged['tp_short'])
    merged['sl_pts'] = np.where(merged['side'] == 0, merged['sl_long'], merged['sl_short'])
    merged = merged.dropna(subset=['label', 'tp_pts', 'sl_pts'])
    merged['date'] = merged.index.normalize()
    print(f'[infer-{name}] {len(merged):,} bars predicted, dt={int(time.time()-t0)}s')
    return merged.reset_index(drop=True)


# ─── trade simulation with optional overlay ────────────────────────────────

def simulate_trades(preds, base_thr, delta, ctx, side_aware=True):
    """Apply firing rule + daily DD cap, return per-bar trade dataframe.

    If delta == 0: pure baseline (always use base_thr).
    Else: when Hougaard bias aligned with bar.side AND active, use
        effective_thr = base_thr + delta * strength * amplifier_b.
    `delta` should be NEGATIVE to lower the threshold (loosen the gate).

    Returns df with columns: ts, date, side, prob, fired, win, pnl,
                              bias_dir, bias_strength, aligned, regime, scenario.
    """
    df = preds.copy()
    # Map context fields by date
    df_dates = df['date'].dt.tz_localize(None) if df['date'].dt.tz is not None else df['date']
    ctx_idx = ctx.index.tz_localize(None) if ctx.index.tz is not None else ctx.index
    ctx_aligned = ctx.copy()
    ctx_aligned.index = ctx_idx
    ctx_lookup = ctx_aligned.reindex(df_dates).reset_index(drop=True)
    df['bias_dir'] = ctx_lookup['bias_direction'].fillna(0).astype(int).values
    df['bias_strength'] = ctx_lookup['bias_strength'].fillna(0.0).astype(float).values
    df['regime_amplifier_b'] = ctx_lookup['regime_amplifier_b'].fillna(1.0).astype(float).values
    df['bull_regime'] = ctx_lookup['bull_regime'].fillna(True).astype(bool).values
    df['high_vol'] = ctx_lookup['high_vol'].fillna(False).astype(bool).values
    df['scenario'] = ctx_lookup['active_scenarios'].fillna('').values
    # side: 0=LONG (signal_dir=+1), 1=SHORT (signal_dir=-1)
    df['signal_dir'] = np.where(df['side'] == 0, +1, -1)
    df['aligned'] = (df['bias_dir'] != 0) & (df['signal_dir'] == df['bias_dir'])

    # Effective threshold per bar
    if delta == 0:
        df['eff_thr'] = base_thr
    elif side_aware:
        # Tilt only when aligned
        tilt = delta * df['bias_strength'] * df['regime_amplifier_b']
        df['eff_thr'] = np.where(df['aligned'], base_thr + tilt, base_thr)
    else:
        # Symmetric: tilt down if aligned, up if opposed
        opposed = (df['bias_dir'] != 0) & (df['signal_dir'] != df['bias_dir'])
        tilt_a = delta * df['bias_strength'] * df['regime_amplifier_b']
        tilt_o = -delta * df['bias_strength'] * df['regime_amplifier_b']
        df['eff_thr'] = base_thr + np.where(df['aligned'], tilt_a,
                                              np.where(opposed, tilt_o, 0.0))

    df['would_fire'] = df['prob'] >= df['eff_thr']

    # Apply daily DD cap (mute remaining fires once cum pnl ≤ -cap)
    df = df.sort_values('ts').reset_index(drop=True)
    fired = np.zeros(len(df), dtype=bool)
    pnl = np.zeros(len(df))
    won = np.zeros(len(df), dtype=bool)
    for date, grp in df.groupby('date'):
        cum = 0.0
        muted = False
        for idx in grp.index:
            if not df.at[idx, 'would_fire']:
                continue
            if muted:
                continue
            y = int(df.at[idx, 'label'])
            tp = float(df.at[idx, 'tp_pts'])
            sl = float(df.at[idx, 'sl_pts'])
            p = (tp * PT_USD) if y == 1 else -(sl * PT_USD)
            fired[idx] = True
            won[idx] = (y == 1)
            pnl[idx] = p
            cum += p
            if cum <= -DAILY_DD_CAP:
                muted = True
    df['fired'] = fired
    df['win'] = won
    df['pnl'] = pnl
    return df


# ─── metrics ───────────────────────────────────────────────────────────────

def metrics(trades_df):
    """Compute metrics on a fired-trades subset."""
    fired = trades_df[trades_df['fired']]
    n = len(fired)
    if n == 0:
        return {'n': 0, 'wr': 0, 'pnl': 0, 'avg': 0, 'pf': 0, 'max_dd': 0,
                'days': 0, 'per_day': 0, 'wins': 0, 'losses': 0,
                'gross_win': 0, 'gross_loss': 0}
    wins = int(fired['win'].sum())
    losses = n - wins
    gw = float(fired.loc[fired['win'], 'pnl'].sum())
    gl = float(-fired.loc[~fired['win'], 'pnl'].sum())  # positive number
    pf = gw / gl if gl > 0 else (float('inf') if gw > 0 else 0)
    pnl_total = float(fired['pnl'].sum())
    days = int(fired['date'].nunique())
    # cumulative max DD
    cs = fired.sort_values('ts')['pnl'].cumsum().values
    peak = np.maximum.accumulate(cs)
    dd = (cs - peak).min() if len(cs) else 0
    return {
        'n': n,
        'wr': wins / n,
        'pnl': pnl_total,
        'avg': pnl_total / n,
        'pf': pf,
        'max_dd': float(dd),
        'days': days,
        'per_day': n / max(days, 1),
        'wins': wins,
        'losses': losses,
        'gross_win': gw,
        'gross_loss': gl,
    }


# ─── Tier A — stratification ───────────────────────────────────────────────

def tier_a_stratification(preds, ctx, base_thr=0.75, label='VAL'):
    """At the production threshold, split fired trades by Hougaard alignment."""
    sim = simulate_trades(preds, base_thr, 0.0, ctx)
    fired = sim[sim['fired']].copy()
    fired['bucket'] = np.where(
        (fired['bias_dir'] != 0) & (fired['signal_dir'] == fired['bias_dir']), 'aligned',
        np.where(
            (fired['bias_dir'] != 0) & (fired['signal_dir'] != fired['bias_dir']), 'opposed',
            'neutral'))
    rows = []
    for bucket in ['aligned', 'opposed', 'neutral', 'all']:
        sub = fired if bucket == 'all' else fired[fired['bucket'] == bucket]
        m = metrics(pd.concat([sub, sim[~sim['fired']]], ignore_index=True))  # passes sub-fired only
        # cleaner: rebuild on subset
        if len(sub) == 0:
            rows.append({'period': label, 'bucket': bucket, 'n': 0,
                         'wr': 0, 'pnl': 0, 'avg': 0, 'pf': 0,
                         'max_dd': 0, 'per_day': 0})
            continue
        wins = int(sub['win'].sum()); n = len(sub)
        gw = float(sub.loc[sub['win'], 'pnl'].sum())
        gl = float(-sub.loc[~sub['win'], 'pnl'].sum())
        pf = gw / gl if gl > 0 else (float('inf') if gw > 0 else 0)
        cs = sub.sort_values('ts')['pnl'].cumsum().values
        peak = np.maximum.accumulate(cs) if len(cs) else cs
        dd = (cs - peak).min() if len(cs) else 0
        rows.append({
            'period': label, 'bucket': bucket, 'n': n,
            'wr': wins / n, 'pnl': sub['pnl'].sum(),
            'avg': sub['pnl'].sum() / n,
            'pf': pf, 'max_dd': float(dd),
            'days': int(sub['date'].nunique()),
            'per_day': n / max(int(sub['date'].nunique()), 1),
        })
    return pd.DataFrame(rows)


# ─── Tier B — overlay sweep ────────────────────────────────────────────────

def tier_b_sweep(preds, ctx, label='VAL'):
    """Sweep (base_thr, delta) — compare baseline vs overlay."""
    base_thrs = [0.65, 0.70, 0.75]
    deltas = [-0.02, -0.05, -0.08, -0.12]
    rows = []
    marginal_trades_all = []
    for base in base_thrs:
        # Baseline (no overlay) for this base threshold
        baseline = simulate_trades(preds, base, 0.0, ctx)
        m_base = metrics(baseline)
        rows.append({**m_base, 'period': label, 'base_thr': base, 'delta': 0.00,
                      'kind': 'baseline'})
        for delta in deltas:
            overlay = simulate_trades(preds, base, delta, ctx)
            m_overlay = metrics(overlay)
            rows.append({**m_overlay, 'period': label, 'base_thr': base,
                          'delta': delta, 'kind': 'overlay'})
            # Marginal subset = trades that fired in overlay but NOT in baseline
            marginal_mask = overlay['fired'] & ~baseline['fired']
            marginal = overlay[marginal_mask].copy()
            if len(marginal) > 0:
                m_marginal = {
                    'n': len(marginal),
                    'wr': float(marginal['win'].mean()),
                    'pnl': float(marginal['pnl'].sum()),
                    'avg': float(marginal['pnl'].mean()),
                    'gross_win': float(marginal.loc[marginal['win'], 'pnl'].sum()),
                    'gross_loss': float(-marginal.loc[~marginal['win'], 'pnl'].sum()),
                }
                m_marginal['pf'] = (m_marginal['gross_win'] / m_marginal['gross_loss']
                                    if m_marginal['gross_loss'] > 0
                                    else (float('inf') if m_marginal['gross_win'] > 0 else 0))
                rows.append({**m_marginal, 'period': label, 'base_thr': base,
                              'delta': delta, 'kind': 'marginal',
                              'max_dd': 0, 'days': int(marginal['date'].nunique()),
                              'per_day': len(marginal) / max(int(marginal['date'].nunique()), 1),
                              'wins': int(marginal['win'].sum()),
                              'losses': len(marginal) - int(marginal['win'].sum())})
                marginal['period'] = label
                marginal['base_thr'] = base
                marginal['delta'] = delta
                marginal_trades_all.append(marginal)
            else:
                rows.append({'n': 0, 'period': label, 'base_thr': base,
                              'delta': delta, 'kind': 'marginal'})
    return pd.DataFrame(rows), marginal_trades_all


# ─── reliability diagram (calibration check) ───────────────────────────────

def calibration_table(preds, n_bins=10):
    """For each prob bin, compute actual TP-first rate. Tells us if 0.75
    actually corresponds to ~75% win rate, etc."""
    df = preds.copy()
    bins = np.linspace(0.30, 1.00, n_bins + 1)
    df['bin'] = pd.cut(df['prob'], bins=bins, include_lowest=True)
    rows = []
    for b, grp in df.groupby('bin', observed=True):
        if len(grp) < 20:
            continue
        rows.append({
            'bin': str(b),
            'n': len(grp),
            'mean_prob': float(grp['prob'].mean()),
            'actual_win_rate': float(grp['label'].mean()),
        })
    return pd.DataFrame(rows)


# ─── main ──────────────────────────────────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--val-start', default=PERIODS['VAL_biden'][0])
    ap.add_argument('--val-end',   default=PERIODS['VAL_biden'][1])
    ap.add_argument('--oos-start', default=PERIODS['OOS_2026'][0])
    ap.add_argument('--oos-end',   default=PERIODS['OOS_2026'][1])
    ap.add_argument('--val-label', default='VAL')
    ap.add_argument('--oos-label', default='OOS')
    ap.add_argument('--out-suffix', default='', help='suffix appended to output filenames')
    args = ap.parse_args()

    t0 = time.time()
    print(f'[main] loading iter12 models (TP/SL = {TP_MULT}/{SL_MULT})...')
    model_l = joblib.load(MODEL_DIR / f'iter12_{TP_MULT}_{SL_MULT}_L.pkl')
    model_s = joblib.load(MODEL_DIR / f'iter12_{TP_MULT}_{SL_MULT}_S.pkl')
    print(f'[main]   LONG  features: {len(model_l["features"])}, AUC val={model_l["auc_val"]:.3f}, OOS={model_l["auc_oos"]:.3f}')
    print(f'[main]   SHORT features: {len(model_s["features"])}, AUC val={model_s["auc_val"]:.3f}, OOS={model_s["auc_oos"]:.3f}')

    # Build Hougaard context for the entire backtest range
    print(f'\n[main] building Hougaard context table...')
    ctx_start = min(args.val_start, args.oos_start, '2017-01-01')
    ctx_end = max(args.val_end, args.oos_end, '2026-05-06')
    ctx = build_context_table(ctx_start, ctx_end)
    print(f'[main]   context: {len(ctx):,} sessions ({ctx_start}→{ctx_end})')

    # Run inference on VAL + OOS
    val = run_inference_on_period(args.val_label, args.val_start, args.val_end, model_l, model_s)
    oos = run_inference_on_period(args.oos_label, args.oos_start, args.oos_end, model_l, model_s)

    # ── Calibration ─────────────────────────────────────────────────────
    print(f'\n[main] calibration on VAL...')
    cal_val = calibration_table(val)
    print(cal_val.to_string(index=False))
    cal_oos = calibration_table(oos)
    print(f'\n[main] calibration on OOS...')
    print(cal_oos.to_string(index=False))
    sfx = args.out_suffix
    cal_lines = ['=' * 80, 'RELIABILITY DIAGRAM — iter12 σ-ML 1.5/1.0', '=' * 80,
                 f'\n{args.val_label} ({args.val_start}→{args.val_end}):',
                 cal_val.to_string(index=False),
                 f'\n{args.oos_label} ({args.oos_start}→{args.oos_end}):',
                 cal_oos.to_string(index=False)]
    (OUT / f'calibration_diagram{sfx}.txt').write_text('\n'.join(cal_lines))

    # ── Tier A ──────────────────────────────────────────────────────────
    print(f'\n[main] Tier A — stratification at base=0.75...')
    a_val = tier_a_stratification(val, ctx, 0.75, args.val_label)
    a_oos = tier_a_stratification(oos, ctx, 0.75, args.oos_label)
    a_full = pd.concat([a_val, a_oos], ignore_index=True)
    a_full.to_csv(OUT / f'tier_a_stratification{sfx}.csv', index=False)
    print(f'\n{args.val_label}:'); print(a_val.to_string(index=False))
    print(f'\n{args.oos_label}:'); print(a_oos.to_string(index=False))

    a_summary_lines = ['=' * 80, 'TIER A — STRATIFICATION (base_thr = 0.75)',
                       '=' * 80,
                       '\nDecision rule: aligned PF >= neutral PF + 0.10  ?',
                       f'\n{args.val_label} ({args.val_start}→{args.val_end}):',
                       a_val.to_string(index=False),
                       f'\n{args.oos_label} ({args.oos_start}→{args.oos_end}):',
                       a_oos.to_string(index=False)]
    (OUT / f'tier_a_summary{sfx}.txt').write_text('\n'.join(a_summary_lines))

    # ── Tier B ──────────────────────────────────────────────────────────
    print(f'\n[main] Tier B — overlay sweep on {args.val_label}...')
    b_val, marg_val = tier_b_sweep(val, ctx, args.val_label)
    print(f'[main] Tier B — overlay sweep on {args.oos_label}...')
    b_oos, marg_oos = tier_b_sweep(oos, ctx, args.oos_label)
    b_full = pd.concat([b_val, b_oos], ignore_index=True)
    b_full.to_csv(OUT / f'tier_b_sweep{sfx}.csv', index=False)

    # Build clean comparison table
    lines = ['=' * 100, 'TIER B — OVERLAY SWEEP', '=' * 100]
    for label, b_df in [(f'{args.val_label} ({args.val_start}→{args.val_end})', b_val),
                          (f'{args.oos_label} ({args.oos_start}→{args.oos_end})', b_oos)]:
        lines.append(f'\n=== {label} ===')
        lines.append(f"\n  {'kind':<10s} {'base':>5s} {'delta':>6s} {'n':>5s} {'/d':>5s} "
                     f"{'WR%':>5s} {'PF':>5s} {'avg':>7s} {'PnL':>10s} {'maxDD':>10s}")
        for _, r in b_df.iterrows():
            n = int(r.get('n', 0)) if pd.notna(r.get('n', 0)) else 0
            if n == 0:
                lines.append(f"  {r['kind']:<10s} {r['base_thr']:>5.2f} {r['delta']:>+6.2f} "
                             f"{n:>5d} (no trades)")
                continue
            wr = r.get('wr', 0)
            pf = r.get('pf', 0)
            pf_str = f'{pf:>5.2f}' if pf != float('inf') else '  inf'
            dd = r.get('max_dd', 0) or 0
            per_day = r.get('per_day', 0) or 0
            avg = r.get('avg', 0) or 0
            pnl = r.get('pnl', 0) or 0
            lines.append(f"  {r['kind']:<10s} {r['base_thr']:>5.2f} {r['delta']:>+6.2f} "
                         f"{n:>5d} {per_day:>4.2f} {wr*100:>4.1f}% "
                         f"{pf_str} ${avg:>+5.0f} ${pnl:>+8.0f} ${dd:>+8.0f}")

    # Marginal-trades-only summary by config
    lines.append('\n' + '=' * 100)
    lines.append('MARGINAL-ONLY (trades the overlay added, NOT in baseline)')
    lines.append('=' * 100)
    for label, b_df in [('VAL', b_val), ('OOS', b_oos)]:
        lines.append(f'\n=== {label} ===')
        lines.append(f"  {'base':>5s} {'delta':>6s} {'n':>5s} {'WR%':>5s} {'PF':>5s} "
                     f"{'avg':>7s} {'PnL':>10s}")
        marg_only = b_df[b_df['kind'] == 'marginal']
        for _, r in marg_only.iterrows():
            n = int(r.get('n', 0)) if pd.notna(r.get('n', 0)) else 0
            if n == 0:
                lines.append(f"  {r['base_thr']:>5.2f} {r['delta']:>+6.2f} {n:>5d} (none)")
                continue
            wr = r.get('wr', 0); pf = r.get('pf', 0)
            pf_str = f'{pf:>5.2f}' if pf != float('inf') else '  inf'
            avg = r.get('avg', 0); pnl = r.get('pnl', 0)
            lines.append(f"  {r['base_thr']:>5.2f} {r['delta']:>+6.2f} {n:>5d} "
                         f"{wr*100:>4.1f}% {pf_str} ${avg:>+5.0f} ${pnl:>+8.0f}")

    txt = '\n'.join(lines)
    print('\n' + txt)
    (OUT / f'tier_b_summary{sfx}.txt').write_text(txt)

    # Save marginal trades for inspection
    if marg_val:
        m_all = pd.concat(marg_val, ignore_index=True)
        m_all.to_csv(OUT / f'tier_b_marginal_trades_{args.val_label}{sfx}.csv', index=False)
    if marg_oos:
        m_all = pd.concat(marg_oos, ignore_index=True)
        m_all.to_csv(OUT / f'tier_b_marginal_trades_{args.oos_label}{sfx}.csv', index=False)

    # Per-regime breakdown for the OOS marginal at delta=-0.05 base=0.75
    if marg_oos:
        m_all = pd.concat(marg_oos, ignore_index=True)
        target = m_all[(m_all['base_thr'] == 0.75) & (m_all['delta'] == -0.05)]
        if len(target):
            regime_rows = []
            for cond, name in [
                (target['bull_regime'], 'bull_regime'),
                (~target['bull_regime'], 'bear_regime'),
                (target['high_vol'], 'high_vol'),
                (~target['high_vol'], 'low_vol'),
                (target['scenario'].str.contains('B', na=False), 'scenario_B'),
                (target['scenario'].str.contains('C', na=False), 'scenario_C'),
            ]:
                sub = target[cond]
                if len(sub) == 0:
                    regime_rows.append({'cond': name, 'n': 0})
                    continue
                regime_rows.append({
                    'cond': name, 'n': len(sub),
                    'wr': float(sub['win'].mean()),
                    'pnl': float(sub['pnl'].sum()),
                    'avg': float(sub['pnl'].mean()),
                })
            pd.DataFrame(regime_rows).to_csv(OUT / 'tier_b_per_regime.csv', index=False)
            print('\n[main] OOS marginal (base=0.75, delta=-0.05) per-regime:')
            print(pd.DataFrame(regime_rows).to_string(index=False))

    print(f'\n[main] wall: {time.time()-t0:.0f}s')
    print(f'[main] outputs in {OUT}/')


if __name__ == '__main__':
    main()
