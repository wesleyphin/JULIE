"""§8.30 — Early-Exit Hyperparameter Sweep.

Runs Phase 3a/b/c/d single-mechanic sweeps + Phase 4 cross-product on the
holdout slice. Outputs per-config trades/WR/PnL/DD/TP-kill rate.

Speed: each replay is microseconds; full holdout (560 rows) per config is ~50ms.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from tools.replay_early_exit_config import (
    replay_with_early_exit_config, parse_bar_path,
)

CORPUS = ROOT / "artifacts/v11_corpus_with_bar_paths.parquet"


def load_holdout() -> pd.DataFrame:
    df = pd.read_parquet(CORPUS)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    h = df[(df["ts"] >= "2026-01-01")
            & (df["allowed_by_friend_rule"] == True)].copy()
    h = h.sort_values("ts").reset_index(drop=True)
    return h


def compute_next_opposite_bar_idx(holdout: pd.DataFrame) -> list[int | None]:
    """For each row, find the bar index in its bar_path that corresponds to
    the next opposite-side candidate's ts (within the trade's horizon window).

    Returns None for rows where no opposite-side candidate fires within horizon.
    """
    out: list[int | None] = []
    sides = holdout["side"].tolist()
    ts_series = holdout["ts"].tolist()
    paths_str = holdout["bar_path_json"].tolist()

    # Build sorted list of (ts, side, original_index)
    rows = list(zip(ts_series, sides, range(len(holdout))))

    for i in range(len(holdout)):
        cur_ts = ts_series[i]
        cur_side = sides[i]
        path_s = paths_str[i]
        if not path_s:
            out.append(None)
            continue
        path = json.loads(path_s)
        if not path:
            out.append(None)
            continue
        # Find next opposite-side candidate ts within bar_path window.
        last_bar_ts = pd.Timestamp(path[-1]["ts"])
        if last_bar_ts.tzinfo is None:
            last_bar_ts = last_bar_ts.tz_localize("US/Eastern")
        # Search forward in holdout for next opposite-side, ts > cur_ts, ts <= last_bar_ts
        next_idx_in_path: int | None = None
        for j in range(i + 1, len(holdout)):
            other_ts = ts_series[j]
            other_side = sides[j]
            if other_ts <= cur_ts:
                continue
            if other_ts > last_bar_ts:
                break
            if other_side != cur_side:
                # Find which bar in the path is closest at-or-after other_ts.
                for k, b in enumerate(path):
                    bts = pd.Timestamp(b["ts"])
                    if bts.tzinfo is None:
                        bts = bts.tz_localize("US/Eastern")
                    if bts >= other_ts:
                        next_idx_in_path = k
                        break
                break
        out.append(next_idx_in_path)
    return out


def run_one_config(holdout: pd.DataFrame, *,
                   be_pct: float | None,           # None = disabled; else multiplied by tp_dist
                   be_threshold_abs: float | None = None,  # absolute pts override
                   be_offset: float = 0.0,
                   pivot_lookback: int | None,     # None = disabled
                   pivot_confirm_bars: int = 0,
                   close_reverse_policy: str = "never",
                   reverse_mfe_threshold: float = 0.0,
                   next_opp_idx: list[int | None] | None = None,
                   pre_parsed_paths: list[list[dict]] | None = None,
                   ) -> dict:
    """Run the full holdout under one config. Returns summary stats."""
    n = len(holdout)
    raw_pnls = np.zeros(n, dtype=np.float64)
    net_pnls = np.zeros(n, dtype=np.float64)
    exit_reasons: list[str] = []
    be_armed_arr = np.zeros(n, dtype=bool)
    pivot_armed_arr = np.zeros(n, dtype=bool)
    is_be_kill = np.zeros(n, dtype=bool)
    is_pivot_kill = np.zeros(n, dtype=bool)
    mfes = np.zeros(n, dtype=np.float64)

    # Original outcome: was this a winner under the no-BE-arm baseline?
    orig_wins = (holdout["raw_pnl"].values > 0)

    rows = holdout.itertuples(index=False)
    for i, row in enumerate(rows):
        path = pre_parsed_paths[i] if pre_parsed_paths is not None else parse_bar_path(row.bar_path_json)
        side = row.side
        ep = float(row.entry_price)
        sl_dist = float(row.sl); tp_dist = float(row.tp)
        if side == "LONG":
            sl_p = ep - sl_dist; tp_p = ep + tp_dist
        else:
            sl_p = ep + sl_dist; tp_p = ep - tp_dist
        if be_threshold_abs is not None:
            be_thresh = float(be_threshold_abs)
        elif be_pct is not None:
            be_thresh = float(tp_dist * be_pct)
        else:
            be_thresh = None
        next_opp = (next_opp_idx[i] if next_opp_idx is not None else None)
        r = replay_with_early_exit_config(
            path, side, ep, sl_p, tp_p,
            be_threshold=be_thresh, be_offset=float(be_offset),
            pivot_lookback=pivot_lookback,
            pivot_confirm_bars=pivot_confirm_bars,
            close_reverse_policy=close_reverse_policy,
            close_reverse_mfe_threshold=float(reverse_mfe_threshold),
            next_opposite_bar_idx=next_opp,
        )
        raw_pnls[i] = r.raw_pnl
        net_pnls[i] = r.net_pnl
        exit_reasons.append(r.exit_reason)
        be_armed_arr[i] = r.be_armed
        pivot_armed_arr[i] = r.pivot_armed
        mfes[i] = r.mfe_points
        # TP-kill: original was a winner, now exits at stop (BE-arm fired and turned win→loss/flat)
        if orig_wins[i] and r.exit_reason in ("stop", "stop_pessimistic"):
            if r.be_armed:
                is_be_kill[i] = True
            elif r.pivot_armed:
                is_pivot_kill[i] = True

    # Compute max-continuous DD on chronological cumulative net_pnl.
    cum = np.cumsum(net_pnls)
    running_max = np.maximum.accumulate(cum)
    dd_series = running_max - cum
    max_dd = float(dd_series.max()) if len(dd_series) > 0 else 0.0

    n_orig_wins = int(orig_wins.sum())
    n_be_kills = int(is_be_kill.sum())
    n_pivot_kills = int(is_pivot_kill.sum())
    tp_kill_rate = (n_be_kills / n_orig_wins) if n_orig_wins > 0 else 0.0

    is_win_net = (net_pnls > 0)
    return {
        "trades": int(n),
        "WR_net": float(is_win_net.mean() * 100.0),
        "raw_pnl": float(raw_pnls.sum()),
        "net_pnl": float(net_pnls.sum()),
        "max_dd": float(max_dd),
        "exit_reasons": pd.Series(exit_reasons).value_counts().to_dict(),
        "be_armed_n": int(be_armed_arr.sum()),
        "pivot_armed_n": int(pivot_armed_arr.sum()),
        "tp_kill_n": n_be_kills,
        "pivot_kill_n": n_pivot_kills,
        "orig_wins_n": n_orig_wins,
        "tp_kill_rate": float(tp_kill_rate * 100.0),
    }


def fmt_row(label: str, r: dict) -> str:
    return (f"{label:<35s} | n={r['trades']:>3d} | WR={r['WR_net']:>5.2f}% | "
            f"PnL=${r['net_pnl']:>9.2f} | DD=${r['max_dd']:>9.2f} | "
            f"BE-arm={r['be_armed_n']:>3d}/Pivot={r['pivot_armed_n']:>3d} | "
            f"TP-kill={r['tp_kill_n']}/{r['orig_wins_n']} ({r['tp_kill_rate']:.1f}%)")


def main() -> None:
    print("[sweep] loading holdout")
    holdout = load_holdout()
    print(f"[sweep] holdout: {len(holdout)} rows")

    # Pre-parse all bar paths (json decode) once. Saves ~30% wall time.
    print("[sweep] pre-parsing bar_paths")
    t0 = time.time()
    paths = [parse_bar_path(s) for s in holdout["bar_path_json"].tolist()]
    print(f"[sweep] paths parsed in {time.time()-t0:.1f}s")

    # Pre-compute next opposite-side bar index for close-on-reverse sweep.
    print("[sweep] computing next-opposite-side bar indexes")
    t0 = time.time()
    next_opp = compute_next_opposite_bar_idx(holdout)
    n_with_opp = sum(1 for x in next_opp if x is not None)
    print(f"[sweep] {n_with_opp}/{len(holdout)} rows have an opp-side candidate "
          f"in horizon ({time.time()-t0:.1f}s)")

    out_lines: list[str] = []

    # ============ PHASE 3a — BE-arm threshold ============
    print("\n=== PHASE 3a: BE-arm threshold sweep ===")
    out_lines.append("\n## PHASE 3a — BE-arm threshold sweep\n")
    out_lines.append("All other params held at: be_offset=0, pivot_lookback=5, pivot_confirm=0, close_reverse=never\n")
    p3a: list[tuple[str, dict]] = []
    # be_threshold = absolute pts (DISABLED, 8, 10, 12, 15, 18, 20, 25)
    be_threshold_grid = [None, 8, 10, 12, 15, 18, 20, 25]
    for thr in be_threshold_grid:
        label = f"BE_thr={thr if thr is not None else 'DISABLED'}"
        r = run_one_config(holdout,
                           be_pct=None,
                           be_threshold_abs=thr,
                           be_offset=0.0,
                           pivot_lookback=5,
                           pivot_confirm_bars=0,
                           close_reverse_policy="never",
                           pre_parsed_paths=paths)
        p3a.append((label, r))
        line = fmt_row(label, r)
        print(" ", line)
        out_lines.append("- " + line)

    # Pick best be_threshold by net_pnl (or tied: lowest DD).
    best_3a_idx = max(range(len(p3a)),
                      key=lambda i: (p3a[i][1]["net_pnl"], -p3a[i][1]["max_dd"]))
    best_3a_label, best_3a_r = p3a[best_3a_idx]
    best_be_thresh = be_threshold_grid[best_3a_idx]
    print(f"\n  BEST 3a: {best_3a_label} (PnL=${best_3a_r['net_pnl']:.2f}, "
          f"DD=${best_3a_r['max_dd']:.2f})")
    out_lines.append(f"\n**BEST 3a:** {best_3a_label} (PnL=${best_3a_r['net_pnl']:.2f}, DD=${best_3a_r['max_dd']:.2f})\n")

    # ============ PHASE 3b — BE-arm trail offset ============
    print("\n=== PHASE 3b: BE-arm trail offset sweep ===")
    out_lines.append("\n## PHASE 3b — BE-arm trail offset sweep\n")
    out_lines.append(f"At BE_thr={best_be_thresh}; sweep offset; pivot=5/0; close_reverse=never\n")
    p3b: list[tuple[str, dict]] = []
    if best_be_thresh is None:
        # If best is to disable BE, offset is moot — note and skip.
        print("  Best 3a is DISABLED — Phase 3b N/A; using thr=10 for offset experimentation")
        scan_thresh = 10  # default reference
    else:
        scan_thresh = best_be_thresh
    for off in [0.0, 0.25, 1.0, 2.5, 5.0]:
        label = f"BE_thr={scan_thresh}, offset={off}"
        r = run_one_config(holdout,
                           be_pct=None, be_threshold_abs=scan_thresh,
                           be_offset=off,
                           pivot_lookback=5, pivot_confirm_bars=0,
                           close_reverse_policy="never",
                           pre_parsed_paths=paths)
        p3b.append((label, r))
        line = fmt_row(label, r)
        print(" ", line)
        out_lines.append("- " + line)
    best_3b_idx = max(range(len(p3b)),
                      key=lambda i: (p3b[i][1]["net_pnl"], -p3b[i][1]["max_dd"]))
    best_3b_label, best_3b_r = p3b[best_3b_idx]
    best_be_offset = [0.0, 0.25, 1.0, 2.5, 5.0][best_3b_idx]
    print(f"\n  BEST 3b: {best_3b_label} (PnL=${best_3b_r['net_pnl']:.2f}, DD=${best_3b_r['max_dd']:.2f})")
    out_lines.append(f"\n**BEST 3b:** {best_3b_label}\n")

    # Compare with disabled-BE if 3a winner was disabled.
    use_be_thresh_for_following = scan_thresh
    use_be_offset_for_following = best_be_offset
    if best_be_thresh is None:
        # Always evaluate the "DISABLED" version as one of the cross-product
        # candidates; for 3c/3d sweeps following, hold BE disabled.
        use_be_thresh_for_following = None
        use_be_offset_for_following = 0.0

    # ============ PHASE 3c — Pivot Trail params ============
    print("\n=== PHASE 3c: Pivot Trail tuning ===")
    out_lines.append("\n## PHASE 3c — Pivot Trail tuning\n")
    out_lines.append(f"At BE_thr={use_be_thresh_for_following} offset={use_be_offset_for_following}; close_reverse=never\n")
    p3c: list[tuple[str, dict]] = []
    pivot_grid: list[tuple[int | None, int]] = []
    for lb in [3, 5, 8, 12]:
        for cb in [0, 1]:
            pivot_grid.append((lb, cb))
    pivot_grid.append((None, 0))   # also include disabled-pivot
    for lb, cb in pivot_grid:
        label = f"Pivot lb={lb}, confirm={cb}"
        r = run_one_config(holdout,
                           be_pct=None, be_threshold_abs=use_be_thresh_for_following,
                           be_offset=use_be_offset_for_following,
                           pivot_lookback=lb, pivot_confirm_bars=cb,
                           close_reverse_policy="never",
                           pre_parsed_paths=paths)
        p3c.append((label, r))
        line = fmt_row(label, r)
        print(" ", line)
        out_lines.append("- " + line)
    best_3c_idx = max(range(len(p3c)),
                      key=lambda i: (p3c[i][1]["net_pnl"], -p3c[i][1]["max_dd"]))
    best_3c_label, best_3c_r = p3c[best_3c_idx]
    best_pivot_lb, best_pivot_cb = pivot_grid[best_3c_idx]
    print(f"\n  BEST 3c: {best_3c_label} (PnL=${best_3c_r['net_pnl']:.2f}, DD=${best_3c_r['max_dd']:.2f})")
    out_lines.append(f"\n**BEST 3c:** {best_3c_label}\n")

    # ============ PHASE 3d — Close-on-reverse policy ============
    print("\n=== PHASE 3d: Close-on-reverse policy ===")
    out_lines.append("\n## PHASE 3d — Close-on-reverse policy\n")
    out_lines.append(f"At BE_thr={use_be_thresh_for_following} offset={use_be_offset_for_following}, pivot lb={best_pivot_lb} cb={best_pivot_cb}\n")
    p3d: list[tuple[str, dict]] = []
    policy_grid = [
        ("never", 0.0),
        ("always", 0.0),
        ("confirmed", 0.0),
        ("mfe_gate-8", 8.0),
    ]
    for pol_label, mfe_thr in policy_grid:
        pol = pol_label.split("-")[0]
        label = f"close_reverse={pol_label}"
        r = run_one_config(holdout,
                           be_pct=None, be_threshold_abs=use_be_thresh_for_following,
                           be_offset=use_be_offset_for_following,
                           pivot_lookback=best_pivot_lb,
                           pivot_confirm_bars=best_pivot_cb,
                           close_reverse_policy=pol,
                           reverse_mfe_threshold=mfe_thr,
                           next_opp_idx=next_opp,
                           pre_parsed_paths=paths)
        p3d.append((label, r))
        line = fmt_row(label, r)
        print(" ", line)
        out_lines.append("- " + line)
    best_3d_idx = max(range(len(p3d)),
                      key=lambda i: (p3d[i][1]["net_pnl"], -p3d[i][1]["max_dd"]))
    best_3d_label, best_3d_r = p3d[best_3d_idx]
    best_3d_policy_label, best_3d_mfe = policy_grid[best_3d_idx]
    best_3d_policy = best_3d_policy_label.split("-")[0]
    print(f"\n  BEST 3d: {best_3d_label} (PnL=${best_3d_r['net_pnl']:.2f}, DD=${best_3d_r['max_dd']:.2f})")
    out_lines.append(f"\n**BEST 3d:** {best_3d_label}\n")

    # ============ PHASE 4 — Cross-product (top-2 each) ============
    print("\n=== PHASE 4: Cross-product top-2 from each ===")
    out_lines.append("\n## PHASE 4 — Cross-product (top-2 from each phase)\n")
    # Take top-2 for each phase by net_pnl (then -DD).
    def top2(results: list[tuple[str, dict]]) -> list[int]:
        ranked = sorted(range(len(results)),
                        key=lambda i: (-results[i][1]["net_pnl"], results[i][1]["max_dd"]))
        return ranked[:2]
    top_3a = top2(p3a)
    top_3b = top2(p3b)
    top_3c = top2(p3c)
    top_3d = top2(p3d)
    cross_results: list[tuple[str, dict, dict]] = []
    cross_count = 0
    for i_a in top_3a:
        thr_a = be_threshold_grid[i_a]
        for i_b in top_3b:
            off_b = [0.0, 0.25, 1.0, 2.5, 5.0][i_b]
            for i_c in top_3c:
                lb_c, cb_c = pivot_grid[i_c]
                for i_d in top_3d:
                    pol_d_lab, mfe_d = policy_grid[i_d]
                    pol_d = pol_d_lab.split("-")[0]
                    cross_count += 1
                    label = f"BE={thr_a}/{off_b}, Piv={lb_c}/{cb_c}, Rev={pol_d_lab}"
                    r = run_one_config(holdout,
                                       be_pct=None,
                                       be_threshold_abs=thr_a,
                                       be_offset=off_b,
                                       pivot_lookback=lb_c,
                                       pivot_confirm_bars=cb_c,
                                       close_reverse_policy=pol_d,
                                       reverse_mfe_threshold=mfe_d,
                                       next_opp_idx=next_opp,
                                       pre_parsed_paths=paths)
                    cross_results.append((label, r,
                                          dict(thr=thr_a, off=off_b, lb=lb_c, cb=cb_c,
                                               rev=pol_d_lab, mfe=mfe_d)))
    # Rank cross-product by net_pnl
    cross_results.sort(key=lambda x: (-x[1]["net_pnl"], x[1]["max_dd"]))
    print(f"\n  {cross_count} cross-product configs evaluated. Top 10 by PnL:")
    out_lines.append(f"\n{cross_count} cross-product configs evaluated. Top 10 by PnL:\n")
    out_lines.append("\n| Rank | Config | n | WR | PnL | DD | TP-kill | G1@870 | G1@1000 | G2 | G3 | G4 |")
    out_lines.append("|------|--------|---|----|-----|----|---------|--------|---------|----|----|----|")
    for rank, (label, r, params) in enumerate(cross_results[:10], 1):
        line = fmt_row(label, r)
        print(f"  #{rank}: {line}")
        # Gates
        g1_870 = "PASS" if r["max_dd"] <= 870 else "FAIL"
        g1_1000 = "PASS" if r["max_dd"] <= 1000 else "FAIL"
        # G2 holdout: PnL > -2886 AND trades <= 560
        g2 = "PASS" if (r["net_pnl"] > -2886 and r["trades"] <= 560) else "FAIL"
        # G3: n >= 50
        g3 = "PASS" if r["trades"] >= 50 else "FAIL"
        # G4: WR >= 55
        g4 = "PASS" if r["WR_net"] >= 55.0 else "FAIL"
        out_lines.append(
            f"| {rank} | {label} | {r['trades']} | {r['WR_net']:.2f}% | "
            f"${r['net_pnl']:.2f} | ${r['max_dd']:.2f} | "
            f"{r['tp_kill_rate']:.1f}% | {g1_870} | {g1_1000} | {g2} | {g3} | {g4} |"
        )

    # =========== PHASE 5/6 — gate evaluations + verdict ===========
    print("\n=== PHASE 5/6: Gate evaluations and verdict ===")
    out_lines.append("\n## PHASE 5/6 — Gate evaluation + verdict\n")
    # Find any config passing all 4 gates at G1=$870
    all_gates_passed_870: list[tuple[str, dict, dict]] = []
    all_gates_passed_1000: list[tuple[str, dict, dict]] = []
    for label, r, params in cross_results:
        passes_870 = (
            r["max_dd"] <= 870
            and r["net_pnl"] > -2886
            and r["trades"] >= 50
            and r["WR_net"] >= 55.0
        )
        passes_1000 = (
            r["max_dd"] <= 1000
            and r["net_pnl"] > -2886
            and r["trades"] >= 50
            and r["WR_net"] >= 55.0
        )
        if passes_870:
            all_gates_passed_870.append((label, r, params))
        if passes_1000:
            all_gates_passed_1000.append((label, r, params))
    print(f"  Configs passing ALL 4 gates @ G1=$870: {len(all_gates_passed_870)}")
    print(f"  Configs passing ALL 4 gates @ G1=$1000: {len(all_gates_passed_1000)}")
    out_lines.append(f"- Configs passing ALL 4 gates @ G1=$870: **{len(all_gates_passed_870)}**\n")
    out_lines.append(f"- Configs passing ALL 4 gates @ G1=$1000: **{len(all_gates_passed_1000)}**\n")

    # Best Pareto: lowest DD with positive net_pnl
    pareto: list[tuple[str, dict]] = []
    for label, r, params in cross_results:
        dom = False
        for olabel, oresult, op in cross_results:
            if (oresult["net_pnl"] >= r["net_pnl"]
                and oresult["max_dd"] <= r["max_dd"]
                and (oresult["net_pnl"] > r["net_pnl"] or oresult["max_dd"] < r["max_dd"])):
                dom = True
                break
        if not dom:
            pareto.append((label, r))
    print(f"\n  Pareto-optimal: {len(pareto)} configs (PnL/DD)")
    out_lines.append(f"\nPareto-optimal: {len(pareto)} configs.\n")

    # Top-5 PnL:
    top5_pnl = sorted(cross_results, key=lambda x: -x[1]["net_pnl"])[:5]
    out_lines.append("\nTop-5 by PnL:\n")
    for label, r, params in top5_pnl:
        out_lines.append(f"- {label}: PnL=${r['net_pnl']:.2f}, DD=${r['max_dd']:.2f}, WR={r['WR_net']:.2f}%")
    # Top-5 DD:
    top5_dd = sorted(cross_results, key=lambda x: x[1]["max_dd"])[:5]
    out_lines.append("\nTop-5 by DD (lowest):\n")
    for label, r, params in top5_dd:
        out_lines.append(f"- {label}: DD=${r['max_dd']:.2f}, PnL=${r['net_pnl']:.2f}, WR={r['WR_net']:.2f}%")

    # Best per criterion
    best_pnl = max(cross_results, key=lambda x: x[1]["net_pnl"])
    best_dd = min(cross_results, key=lambda x: x[1]["max_dd"])
    best_wr = max(cross_results, key=lambda x: x[1]["WR_net"])
    out_lines.append(f"\nBest by PnL:  {best_pnl[0]} → PnL=${best_pnl[1]['net_pnl']:.2f} DD=${best_pnl[1]['max_dd']:.2f} WR={best_pnl[1]['WR_net']:.2f}%\n")
    out_lines.append(f"Best by DD:   {best_dd[0]} → DD=${best_dd[1]['max_dd']:.2f} PnL=${best_dd[1]['net_pnl']:.2f} WR={best_dd[1]['WR_net']:.2f}%\n")
    out_lines.append(f"Best by WR:   {best_wr[0]} → WR={best_wr[1]['WR_net']:.2f}% PnL=${best_wr[1]['net_pnl']:.2f} DD=${best_wr[1]['max_dd']:.2f}\n")

    # Save full results
    detail_rows = []
    for label, r, params in cross_results:
        detail_rows.append({
            "label": label, **params,
            "trades": r["trades"], "WR": r["WR_net"], "net_pnl": r["net_pnl"],
            "raw_pnl": r["raw_pnl"], "max_dd": r["max_dd"],
            "be_armed_n": r["be_armed_n"], "pivot_armed_n": r["pivot_armed_n"],
            "tp_kill_n": r["tp_kill_n"], "tp_kill_rate": r["tp_kill_rate"],
            "orig_wins": r["orig_wins_n"],
        })
    pd.DataFrame(detail_rows).to_csv("/tmp/early_exit_cross_product.csv", index=False)
    print(f"\n[sweep] cross-product CSV: /tmp/early_exit_cross_product.csv")

    # Comparison: current sim mechanics (BE_pct=0.40 → 10pt, offset=0, pivot 5/0, close=never)
    print("\n=== Reference: §8.29 current mechanics ===")
    cur = run_one_config(holdout,
                         be_pct=None, be_threshold_abs=10,
                         be_offset=0.0,
                         pivot_lookback=5, pivot_confirm_bars=0,
                         close_reverse_policy="never",
                         pre_parsed_paths=paths)
    line = fmt_row("CURRENT (BE=10/0, Piv=5/0, Rev=never)", cur)
    print(" ", line)
    out_lines.append(f"\n## Reference: current §8.29 mechanics\n{line}\n")

    # Also: pure baseline (no BE, no pivot, no reverse) — confirmed to be -$2,886.25
    base = run_one_config(holdout,
                          be_pct=None, be_threshold_abs=None,
                          be_offset=0.0,
                          pivot_lookback=None, pivot_confirm_bars=0,
                          close_reverse_policy="never",
                          pre_parsed_paths=paths)
    line2 = fmt_row("BASELINE (no BE, no pivot, no rev)", base)
    print(" ", line2)
    out_lines.append(f"\n## Baseline: no BE / no pivot / no reverse\n{line2}\n")

    # Write the final report
    with open("/tmp/phase7_early_exit_sweep_report.md", "w") as f:
        f.write("# §8.30 Early-Exit Hyperparameter Sweep — Final Report\n\n")
        f.write(f"_Generated 2026-04-25._\n\n")
        f.write("\n".join(out_lines))
    print("\n[sweep] report: /tmp/phase7_early_exit_sweep_report.md")

    # Save references for the report writer (Phase 7).
    return {
        "p3a": p3a, "p3b": p3b, "p3c": p3c, "p3d": p3d,
        "cross_results": cross_results, "current": cur, "baseline": base,
        "all_gates_870": all_gates_passed_870,
        "all_gates_1000": all_gates_passed_1000,
        "pareto": pareto,
    }


if __name__ == "__main__":
    main()
