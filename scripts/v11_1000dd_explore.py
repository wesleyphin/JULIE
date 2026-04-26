"""V11 DD <= $1,000 gate exploration.

Task 1: per-head threshold compression sweep.
Task 2: per-strategy isolation.
Task 3: combined v11 stacks (Pareto).
Task 4: outputs.

Read-only on existing data. No retraining.
"""
import json
import numpy as np
import pandas as pd
from itertools import product

CORPUS = 'artifacts/v11_training_corpus.parquet'

# Gates (relaxed)
G1_DD_MAX = 1000.0
G2_PNL_BASELINE = -2886.25  # holdout baseline
G2_TRADES_MAX = 560
G3_N_OOS_MIN = 50
G4_WR_MIN = 0.55


def compute_dd_pnl(pnl_series: pd.Series) -> tuple[float, float, int, float]:
    """Return (pnl_total, dd, n, wr) on chronological order. Quoted formula."""
    s = pnl_series.reset_index(drop=True)
    n = len(s)
    if n == 0:
        return 0.0, 0.0, 0, 0.0
    equity = s.cumsum()
    # DD formula: max((equity.cummax() - equity).max(), 0)
    dd = max(float((equity.cummax() - equity).max()), 0.0)
    pnl_total = float(s.sum())
    wr = float((s > 0).mean())
    return pnl_total, dd, n, wr


def gate_eval(n: int, pnl: float, dd: float, wr: float) -> dict:
    return {
        'G1_dd_le_1000': dd <= G1_DD_MAX,
        'G2_pnl_ok': (pnl >= G2_PNL_BASELINE) and (n <= G2_TRADES_MAX),
        'G3_n_oos': n >= G3_N_OOS_MIN,
        'G4_wr_ge_55': wr >= G4_WR_MIN,
    }


def gate_str(gates: dict) -> str:
    return ''.join(['Y' if gates[k] else 'N' for k in ['G1_dd_le_1000', 'G2_pnl_ok', 'G3_n_oos', 'G4_wr_ge_55']])


def block_mask_for_head(holdout: pd.DataFrame, head: str, thr: float) -> pd.Series:
    """Return boolean Series: True = BLOCK (drop). Applies head-specific applicability."""
    proba_col = {
        'kalshi_de3': 'kalshi_proba',
        'lfo_de3': 'lfo_proba',
        'pct_de3': 'pct_proba',
        'filterg_de3': 'fg_proba',
        'pivot_de3': 'pivot_proba',
    }[head]
    proba = holdout[proba_col]
    block = proba >= thr
    # Applicability gates
    if head == 'kalshi_de3':
        # Per Phase 2: only DE3 rows in kalshi window
        applicable = (holdout['family'] != 'regimeadaptive') & (holdout['in_kalshi_window'] == True)
        block = block & applicable
    else:
        # All other heads: applies to DE3 only? Per spec: "apply to all DE3 rows"
        # But fg/lfo/pct/pivot heads were trained on DE3 in Phase 2.
        applicable = (holdout['family'] != 'regimeadaptive')
        block = block & applicable
    return block


def main():
    df = pd.read_parquet(CORPUS)
    df['ts'] = pd.to_datetime(df['ts'])
    df['ts_naive'] = df['ts'].dt.tz_localize(None)
    holdout = df[(df['ts_naive'] >= '2026-01-01') & (df['allowed_by_friend_rule'] == True)].copy()
    holdout = holdout.sort_values('ts').reset_index(drop=True)
    print(f'Holdout: {len(holdout)} rows')

    # Baseline
    pnl_b, dd_b, n_b, wr_b = compute_dd_pnl(holdout['net_pnl_after_haircut'])
    print(f'BASELINE: n={n_b}, WR={wr_b*100:.2f}%, PnL={pnl_b:.2f}, DD={dd_b:.2f}')

    baseline = {'n': n_b, 'wr': wr_b, 'pnl': pnl_b, 'dd': dd_b}

    # ===== Task 1: per-head threshold sweep =====
    heads = ['kalshi_de3', 'lfo_de3', 'pct_de3', 'filterg_de3', 'pivot_de3']
    thrs = np.round(np.arange(0.10, 0.4001, 0.025), 4)
    rows = []
    for head in heads:
        for thr in thrs:
            block = block_mask_for_head(holdout, head, float(thr))
            kept = holdout.loc[~block].sort_values('ts').reset_index(drop=True)
            pnl, dd, n, wr = compute_dd_pnl(kept['net_pnl_after_haircut'])
            gates = gate_eval(n, pnl, dd, wr)
            rows.append({
                'head': head,
                'thr': float(thr),
                'n_kept': n,
                'n_blocked': int(block.sum()),
                'WR': wr,
                'PnL': pnl,
                'DD': dd,
                'G1': gates['G1_dd_le_1000'],
                'G2': gates['G2_pnl_ok'],
                'G3': gates['G3_n_oos'],
                'G4': gates['G4_wr_ge_55'],
                'gates_str': gate_str(gates),
            })
    task1_df = pd.DataFrame(rows)
    task1_df.to_parquet('artifacts/v11_threshold_compression_sweep.parquet')
    print(f'Task1 sweep rows: {len(task1_df)}')

    # ===== Task 2: per-strategy isolation =====
    isolations = {}

    def isolate(name, mask):
        sub = holdout.loc[mask].sort_values('ts').reset_index(drop=True)
        pnl, dd, n, wr = compute_dd_pnl(sub['net_pnl_after_haircut'])
        gates = gate_eval(n, pnl, dd, wr)
        isolations[name] = {
            'n': n, 'WR': wr, 'PnL': pnl, 'DD': dd,
            'gates': gates, 'gates_str': gate_str(gates),
        }

    isolate('DE3_only', holdout['family'] != 'regimeadaptive')
    isolate('RA_only', holdout['family'] == 'regimeadaptive')
    # Cross-family pairs: trades that occurred close in time? Per spec hard to define.
    # Interpret as: holdout is already friend-rule filtered, so "DE3+RA pairs" = full holdout.
    isolate('full_holdout', pd.Series(True, index=holdout.index))

    # DE3-only + best Kalshi head decision
    de3_mask = holdout['family'] != 'regimeadaptive'
    # Find best Kalshi thr in Task1 by Pareto: min DD with PnL >= baseline, then highest PnL
    k_sub = task1_df[task1_df['head'] == 'kalshi_de3'].copy()
    k_pareto = k_sub[(k_sub['DD'] <= G1_DD_MAX)]
    if len(k_pareto):
        # restrict to PnL >= baseline
        k_pass = k_pareto[k_pareto['PnL'] >= G2_PNL_BASELINE]
        k_best = k_pass.sort_values(['PnL'], ascending=False).iloc[0] if len(k_pass) else k_pareto.sort_values('DD').iloc[0]
    else:
        k_best = k_sub.sort_values('DD').iloc[0]
    print(f'best kalshi: thr={k_best["thr"]} DD={k_best["DD"]:.0f} PnL={k_best["PnL"]:.0f}')

    # ===== Task 3: combined stacks =====
    # Build BLOCK series cache for each head/thr
    def head_block(head, thr):
        return block_mask_for_head(holdout, head, thr)

    def stack_eval(block_combined: pd.Series):
        kept = holdout.loc[~block_combined].sort_values('ts').reset_index(drop=True)
        pnl, dd, n, wr = compute_dd_pnl(kept['net_pnl_after_haircut'])
        gates = gate_eval(n, pnl, dd, wr)
        return n, wr, pnl, dd, gates

    fg_thrs = [0.20, 0.30, 0.40, 0.50, 0.60]
    k_thrs = [0.20, 0.30, 0.40, 0.50]
    lfo_thrs = [0.20, 0.30, 0.40, 0.50]
    pct_thrs = [0.20, 0.30, 0.40, 0.50]

    stack_rows = []
    # Stack A: FG + Kalshi
    for fg, k in product(fg_thrs, k_thrs):
        block = head_block('filterg_de3', fg) | head_block('kalshi_de3', k)
        n, wr, pnl, dd, g = stack_eval(block)
        stack_rows.append({
            'stack': 'A_FG_K', 'fg_thr': fg, 'k_thr': k, 'lfo_thr': None, 'pct_thr': None,
            'n_kept': n, 'WR': wr, 'PnL': pnl, 'DD': dd,
            'G1': g['G1_dd_le_1000'], 'G2': g['G2_pnl_ok'], 'G3': g['G3_n_oos'], 'G4': g['G4_wr_ge_55'],
            'gates_str': gate_str(g),
        })
    # Stack B: FG + LFO
    for fg, lfo in product(fg_thrs, lfo_thrs):
        block = head_block('filterg_de3', fg) | head_block('lfo_de3', lfo)
        n, wr, pnl, dd, g = stack_eval(block)
        stack_rows.append({
            'stack': 'B_FG_LFO', 'fg_thr': fg, 'k_thr': None, 'lfo_thr': lfo, 'pct_thr': None,
            'n_kept': n, 'WR': wr, 'PnL': pnl, 'DD': dd,
            'G1': g['G1_dd_le_1000'], 'G2': g['G2_pnl_ok'], 'G3': g['G3_n_oos'], 'G4': g['G4_wr_ge_55'],
            'gates_str': gate_str(g),
        })
    # Stack C: FG + Kalshi + LFO
    for fg, k, lfo in product(fg_thrs, k_thrs, lfo_thrs):
        block = head_block('filterg_de3', fg) | head_block('kalshi_de3', k) | head_block('lfo_de3', lfo)
        n, wr, pnl, dd, g = stack_eval(block)
        stack_rows.append({
            'stack': 'C_FG_K_LFO', 'fg_thr': fg, 'k_thr': k, 'lfo_thr': lfo, 'pct_thr': None,
            'n_kept': n, 'WR': wr, 'PnL': pnl, 'DD': dd,
            'G1': g['G1_dd_le_1000'], 'G2': g['G2_pnl_ok'], 'G3': g['G3_n_oos'], 'G4': g['G4_wr_ge_55'],
            'gates_str': gate_str(g),
        })
    # Stack D: all 4 heads
    for fg, k, lfo, pct in product(fg_thrs, k_thrs, lfo_thrs, pct_thrs):
        block = head_block('filterg_de3', fg) | head_block('kalshi_de3', k) | head_block('lfo_de3', lfo) | head_block('pct_de3', pct)
        n, wr, pnl, dd, g = stack_eval(block)
        stack_rows.append({
            'stack': 'D_FG_K_LFO_PCT', 'fg_thr': fg, 'k_thr': k, 'lfo_thr': lfo, 'pct_thr': pct,
            'n_kept': n, 'WR': wr, 'PnL': pnl, 'DD': dd,
            'G1': g['G1_dd_le_1000'], 'G2': g['G2_pnl_ok'], 'G3': g['G3_n_oos'], 'G4': g['G4_wr_ge_55'],
            'gates_str': gate_str(g),
        })

    stacks_df = pd.DataFrame(stack_rows)
    stacks_df.to_parquet('artifacts/v11_combined_stacks_1000dd.parquet')
    print(f'Total stack combos: {len(stacks_df)}')

    # Per-strategy isolation extras: best head applied to DE3 only
    # Best Kalshi config (any thr) on DE3-only
    best_k_thr = k_best['thr']
    block_k = block_mask_for_head(holdout, 'kalshi_de3', float(best_k_thr))
    de3_block_k = (~de3_mask) | block_k  # drop RA + drop kalshi-blocks
    sub = holdout.loc[~de3_block_k].sort_values('ts').reset_index(drop=True)
    pnl, dd, n, wr = compute_dd_pnl(sub['net_pnl_after_haircut'])
    gates = gate_eval(n, pnl, dd, wr)
    isolations[f'DE3_only+kalshi@{best_k_thr:.3f}'] = {
        'n': n, 'WR': wr, 'PnL': pnl, 'DD': dd, 'gates': gates, 'gates_str': gate_str(gates),
    }

    # Best LFO
    lfo_sub = task1_df[task1_df['head'] == 'lfo_de3']
    lfo_pass = lfo_sub[(lfo_sub['DD'] <= G1_DD_MAX) & (lfo_sub['PnL'] >= G2_PNL_BASELINE)]
    if len(lfo_pass):
        lfo_best_thr = lfo_pass.sort_values('PnL', ascending=False).iloc[0]['thr']
    else:
        lfo_best_thr = lfo_sub.sort_values('DD').iloc[0]['thr']
    block_lfo = block_mask_for_head(holdout, 'lfo_de3', float(lfo_best_thr))
    de3_block_lfo = (~de3_mask) | block_lfo
    sub = holdout.loc[~de3_block_lfo].sort_values('ts').reset_index(drop=True)
    pnl, dd, n, wr = compute_dd_pnl(sub['net_pnl_after_haircut'])
    gates = gate_eval(n, pnl, dd, wr)
    isolations[f'DE3_only+lfo@{lfo_best_thr:.3f}'] = {
        'n': n, 'WR': wr, 'PnL': pnl, 'DD': dd, 'gates': gates, 'gates_str': gate_str(gates),
    }

    # Best FG (filterg)
    fg_sub = task1_df[task1_df['head'] == 'filterg_de3']
    fg_pass = fg_sub[(fg_sub['DD'] <= G1_DD_MAX) & (fg_sub['PnL'] >= G2_PNL_BASELINE)]
    if len(fg_pass):
        fg_best_thr = fg_pass.sort_values('PnL', ascending=False).iloc[0]['thr']
    else:
        fg_best_thr = fg_sub.sort_values('DD').iloc[0]['thr']
    block_fg = block_mask_for_head(holdout, 'filterg_de3', float(fg_best_thr))
    de3_block_fg = (~de3_mask) | block_fg
    sub = holdout.loc[~de3_block_fg].sort_values('ts').reset_index(drop=True)
    pnl, dd, n, wr = compute_dd_pnl(sub['net_pnl_after_haircut'])
    gates = gate_eval(n, pnl, dd, wr)
    isolations[f'DE3_only+fg@{fg_best_thr:.3f}'] = {
        'n': n, 'WR': wr, 'PnL': pnl, 'DD': dd, 'gates': gates, 'gates_str': gate_str(gates),
    }

    with open('artifacts/v11_per_strategy_isolation.json', 'w') as f:
        json.dump(isolations, f, indent=2, default=str)
    print(f'Isolations: {len(isolations)} variants')

    # ===== Task 4: top-line summary =====
    summary = {'baseline': baseline, 'best_per_stack': {}, 'best_per_head': {}, 'global_best_passing': None}
    # Best per head (lowest DD <= 1000 with PnL >= baseline; else lowest DD)
    for head in heads:
        sub = task1_df[task1_df['head'] == head]
        passing = sub[(sub['DD'] <= G1_DD_MAX) & (sub['PnL'] >= G2_PNL_BASELINE) &
                      (sub['n_kept'] >= G3_N_OOS_MIN) & (sub['WR'] >= G4_WR_MIN)]
        if len(passing):
            row = passing.sort_values(['DD', 'PnL'], ascending=[True, False]).iloc[0]
            summary['best_per_head'][head] = {
                'thr': float(row['thr']), 'n_kept': int(row['n_kept']),
                'WR': float(row['WR']), 'PnL': float(row['PnL']), 'DD': float(row['DD']),
                'all_4_pass': True, 'gates_str': row['gates_str'],
            }
        else:
            # closest: smallest DD with WR >= 55 (G4 binding)
            relax = sub.sort_values(['G1', 'G4', 'PnL'], ascending=[False, False, False]).iloc[0]
            summary['best_per_head'][head] = {
                'thr': float(relax['thr']), 'n_kept': int(relax['n_kept']),
                'WR': float(relax['WR']), 'PnL': float(relax['PnL']), 'DD': float(relax['DD']),
                'all_4_pass': False, 'gates_str': relax['gates_str'],
            }

    # Best per stack
    for stack_name in ['A_FG_K', 'B_FG_LFO', 'C_FG_K_LFO', 'D_FG_K_LFO_PCT']:
        sub = stacks_df[stacks_df['stack'] == stack_name]
        passing = sub[(sub['DD'] <= G1_DD_MAX) & (sub['PnL'] >= G2_PNL_BASELINE) &
                      (sub['n_kept'] >= G3_N_OOS_MIN) & (sub['WR'] >= G4_WR_MIN)]
        if len(passing):
            row = passing.sort_values(['PnL', 'DD'], ascending=[False, True]).iloc[0]
            summary['best_per_stack'][stack_name] = {
                **{k: (float(row[k]) if isinstance(row[k], (int, float, np.floating)) and not pd.isna(row[k]) else None)
                   for k in ['fg_thr', 'k_thr', 'lfo_thr', 'pct_thr']},
                'n_kept': int(row['n_kept']),
                'WR': float(row['WR']), 'PnL': float(row['PnL']), 'DD': float(row['DD']),
                'all_4_pass': True, 'gates_str': row['gates_str'],
            }
        else:
            # closest by # gates passing then PnL
            sub2 = sub.copy()
            sub2['n_pass'] = sub2[['G1', 'G2', 'G3', 'G4']].sum(axis=1)
            relax = sub2.sort_values(['n_pass', 'PnL'], ascending=[False, False]).iloc[0]
            summary['best_per_stack'][stack_name] = {
                **{k: (float(relax[k]) if isinstance(relax[k], (int, float, np.floating)) and not pd.isna(relax[k]) else None)
                   for k in ['fg_thr', 'k_thr', 'lfo_thr', 'pct_thr']},
                'n_kept': int(relax['n_kept']),
                'WR': float(relax['WR']), 'PnL': float(relax['PnL']), 'DD': float(relax['DD']),
                'all_4_pass': False, 'gates_str': relax['gates_str'],
            }

    # Global best — any config across heads/stacks passing all 4 gates
    all_pass_rows = []
    for head in heads:
        sub = task1_df[task1_df['head'] == head]
        p = sub[(sub['DD'] <= G1_DD_MAX) & (sub['PnL'] >= G2_PNL_BASELINE) &
                (sub['n_kept'] >= G3_N_OOS_MIN) & (sub['WR'] >= G4_WR_MIN)]
        for _, r in p.iterrows():
            all_pass_rows.append({'source': f'head:{head}', 'thr': r['thr'], 'n_kept': r['n_kept'],
                                  'WR': r['WR'], 'PnL': r['PnL'], 'DD': r['DD']})
    p2 = stacks_df[(stacks_df['DD'] <= G1_DD_MAX) & (stacks_df['PnL'] >= G2_PNL_BASELINE) &
                   (stacks_df['n_kept'] >= G3_N_OOS_MIN) & (stacks_df['WR'] >= G4_WR_MIN)]
    for _, r in p2.iterrows():
        all_pass_rows.append({
            'source': f'stack:{r["stack"]}', 'fg_thr': r['fg_thr'], 'k_thr': r['k_thr'],
            'lfo_thr': r['lfo_thr'], 'pct_thr': r['pct_thr'],
            'n_kept': int(r['n_kept']), 'WR': float(r['WR']), 'PnL': float(r['PnL']), 'DD': float(r['DD']),
        })

    summary['n_configs_passing_all_4_gates'] = len(all_pass_rows)
    if all_pass_rows:
        all_pass_df = pd.DataFrame(all_pass_rows).sort_values('PnL', ascending=False)
        summary['top_passing_configs'] = all_pass_df.head(10).to_dict(orient='records')
    else:
        summary['top_passing_configs'] = []

    with open('artifacts/v11_1000dd_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f'Configs passing all 4 gates: {len(all_pass_rows)}')

    # quick echo top sweeps for sanity
    print('\nTop 5 per-head DD-passing rows (sorted by DD asc):')
    print(task1_df[task1_df['DD'] <= G1_DD_MAX].sort_values('DD').head(15).to_string(index=False))


if __name__ == '__main__':
    main()
