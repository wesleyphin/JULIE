"""3-way comparison of per-cell layer states applied to the v11 corrected corpus.

Configs:
  A) BASELINE_DORMANT  — per-cell mult forced to 1.0 (the actual 14-month live state
     because of the case-mismatch bug; fixed in §8.33).
  B) V1_PER_CELL_ACTIVE — per-cell layer ON, reading filterg_threshold_overrides.json
     (v1 calibration against the broken simulator).
  C) V2_PER_CELL_ACTIVE — per-cell layer ON, reading filterg_threshold_overrides_v2.json
     (v11-corpus recalibration — current default).

For each config, replay every candidate through Filter G's veto rule and accumulate
realized PnL on rows that survive. Use the same regime classification, session-
adaptive, and effective-threshold floor as signal_gate_2025.py.

NO LIVE state changes; reads parquets and JSONs only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

REPO = Path(__file__).resolve().parent.parent

# Mirror signal_gate_2025.py constants exactly (as of 2026-04-26 commit).
BASE_THR = 0.35
EFF_THR_FLOOR = 0.25
REGIME_MULT = {"whipsaw": 0.60, "calm_trend": 1.05, "neutral": 1.0, "warmup": 1.0,
               "dead_tape": 1.0}
SESS_LENIENT_PNL, SESS_LENIENT_MULT = 100.0, 1.25
SESS_AGGRESSIVE_PNL, SESS_AGGRESSIVE_MULT = -200.0, 0.80

# regime_classifier.py thresholds
EFF_LOW, EFF_HIGH = 0.05, 0.12
DEAD_TAPE_VOL_BP = 1.5

STRATEGY_NAME_NORMALIZE = {
    "de3": "DynamicEngine3",
    "ra": "RegimeAdaptive",
    "af": "AetherFlow",
    "mlphysics": "MLPhysics",
    "DynamicEngine3": "DynamicEngine3",
    "RegimeAdaptive": "RegimeAdaptive",
    "AetherFlow": "AetherFlow",
    "MLPhysics": "MLPhysics",
}


def classify_regime(vol_bp: float, eff: float) -> str:
    if vol_bp is None or pd.isna(vol_bp) or vol_bp < DEAD_TAPE_VOL_BP:
        return "dead_tape"
    if vol_bp > 3.5 and (eff is not None and not pd.isna(eff) and eff < EFF_LOW):
        return "whipsaw"
    if eff is not None and not pd.isna(eff) and eff > EFF_HIGH:
        return "calm_trend"
    return "neutral"


def time_bucket(et_hour: int) -> str:
    h = float(et_hour)
    if h < 4.0:
        h += 24.0
    if 4.0 <= h < 9.5:   return "pre_open"
    if 9.5 <= h < 12.0:  return "morning"
    if 12.0 <= h < 14.0: return "lunch"
    if 14.0 <= h < 16.0: return "afternoon"
    if 16.0 <= h < 17.0: return "post_close"
    return "overnight"


def session_mult(cum_day_pnl: float) -> float:
    if cum_day_pnl >= SESS_LENIENT_PNL:
        return SESS_LENIENT_MULT
    if cum_day_pnl <= SESS_AGGRESSIVE_PNL:
        return SESS_AGGRESSIVE_MULT
    return 1.0


def parse_per_cell_payload(payload: Dict) -> Dict[str, float]:
    """Same parser as the production loader (signal_gate_2025.py)."""
    out: Dict[str, float] = {}
    rt = payload.get("runtime_multipliers") or {}
    if isinstance(rt, dict) and rt:
        for k, v in rt.items():
            try:
                out[str(k)] = float(v)
            except (ValueError, TypeError):
                continue
        return out
    cells = payload.get("cells") or {}
    for k, meta in cells.items():
        if isinstance(meta, dict) and "mult" in meta:
            try:
                out[str(k)] = float(meta["mult"])
            except (ValueError, TypeError):
                continue
        elif isinstance(meta, (int, float)):
            try:
                out[str(k)] = float(meta)
            except (ValueError, TypeError):
                continue
    return out


def load_per_cell(path: Path) -> Dict[str, float]:
    if not path.exists():
        return {}
    return parse_per_cell_payload(json.loads(path.read_text()))


def normalize_strategy(s: str) -> str:
    if not s:
        return s
    return STRATEGY_NAME_NORMALIZE.get(s, STRATEGY_NAME_NORMALIZE.get(s.lower(), s))


def per_cell_mult(table: Dict[str, float], strategy: str, regime: str, et_hour: int) -> float:
    """1.0 when table is empty (DORMANT)."""
    if not table:
        return 1.0
    strat = normalize_strategy(strategy)
    tb = time_bucket(et_hour)
    key = f"{strat}|{(regime or '').lower()}|{tb}"
    return float(table.get(key, 1.0))


def effective_threshold(regime: str, cum_day_pnl: float, pc_mult: float) -> float:
    rm = REGIME_MULT.get(regime, 1.0)
    sm = session_mult(cum_day_pnl)
    eff = BASE_THR * rm * sm * pc_mult
    return max(EFF_THR_FLOOR, eff)


def replay_config(df: pd.DataFrame, table: Dict[str, float], label: str) -> pd.DataFrame:
    """Process rows in time order, decide veto, simulate fire/skip with per-strategy
    cum_day_pnl tracking. Returns annotated df."""
    out = df.copy().reset_index(drop=True)
    out["regime_label"] = [
        classify_regime(v, e)
        for v, e in zip(out["bf_regime_vol_bp"], out["bf_regime_eff"])
    ]
    out["et_hour"] = pd.to_datetime(out["ts"]).dt.tz_convert("America/New_York").dt.hour
    out["time_bucket"] = out["et_hour"].apply(time_bucket)
    out["pc_mult"] = [
        per_cell_mult(table, s, r, h)
        for s, r, h in zip(out["strategy"], out["regime_label"], out["et_hour"])
    ]

    # iterate in time order, track cum_day_pnl per strategy per day (ET)
    out = out.sort_values("ts").reset_index(drop=True)
    out["et_date"] = pd.to_datetime(out["ts"]).dt.tz_convert("America/New_York").dt.date

    fired_flags: List[bool] = []
    eff_thrs: List[float] = []
    cum_pnls: List[float] = []
    sm_list: List[float] = []

    daily_state: Dict[Tuple, float] = {}  # (et_date, strategy) -> cum_day_pnl
    for i, row in out.iterrows():
        key = (row["et_date"], row["strategy"])
        cum = daily_state.get(key, 0.0)
        sm = session_mult(cum)
        rm = REGIME_MULT.get(row["regime_label"], 1.0)
        eff_thr = max(EFF_THR_FLOOR, BASE_THR * rm * sm * row["pc_mult"])
        vetoed = float(row["fg_proba"]) >= eff_thr
        fired = not vetoed
        if fired:
            cum += float(row["net_pnl_after_haircut"])
            daily_state[key] = cum
        fired_flags.append(fired)
        eff_thrs.append(eff_thr)
        cum_pnls.append(cum)
        sm_list.append(sm)

    out[f"{label}_fired"] = fired_flags
    out[f"{label}_eff_thr"] = eff_thrs
    out[f"{label}_cum_day_pnl_after"] = cum_pnls
    out[f"{label}_session_mult"] = sm_list
    return out


def summarize(df: pd.DataFrame, label: str) -> Dict:
    fired_col = f"{label}_fired"
    fired = df[df[fired_col]]
    n = len(fired)
    pnl = float(fired["net_pnl_after_haircut"].sum())
    wins = int((fired["net_pnl_after_haircut"] > 0).sum())
    losses = int((fired["net_pnl_after_haircut"] < 0).sum())
    flat = int((fired["net_pnl_after_haircut"] == 0).sum())
    wr = wins / n if n else 0.0

    # max DD on equity curve in time order
    eq = fired.sort_values("ts")["net_pnl_after_haircut"].cumsum()
    peak = eq.cummax()
    dd_series = eq - peak
    max_dd = float(dd_series.min()) if len(dd_series) else 0.0

    avg_per_trade = pnl / n if n else 0.0

    return {
        "label": label,
        "trades": n,
        "wins": wins,
        "losses": losses,
        "flat": flat,
        "wr": wr,
        "pnl": pnl,
        "avg_per_trade": avg_per_trade,
        "max_dd": max_dd,
        "blocked": len(df) - n,
        "block_rate": (len(df) - n) / len(df) if len(df) else 0.0,
    }


def per_month(df: pd.DataFrame, label: str) -> pd.DataFrame:
    fired = df[df[f"{label}_fired"]].copy()
    fired["month"] = pd.to_datetime(fired["ts"]).dt.tz_convert("America/New_York").dt.strftime("%Y-%m")
    g = fired.groupby("month").agg(
        trades=("ts", "count"),
        wr=("net_pnl_after_haircut", lambda s: (s > 0).sum() / len(s) if len(s) else 0.0),
        pnl=("net_pnl_after_haircut", "sum"),
    )
    return g


def by_strategy(df: pd.DataFrame, label: str) -> pd.DataFrame:
    fired = df[df[f"{label}_fired"]]
    g = fired.groupby("strategy").agg(
        trades=("ts", "count"),
        wr=("net_pnl_after_haircut", lambda s: (s > 0).sum() / len(s) if len(s) else 0.0),
        pnl=("net_pnl_after_haircut", "sum"),
    )
    return g


def main():
    corpus = pd.read_parquet(REPO / "artifacts" / "v11_corpus_with_bar_paths.parquet")
    print(f"corpus: {len(corpus)} rows, "
          f"{pd.to_datetime(corpus['ts']).min()} → {pd.to_datetime(corpus['ts']).max()}")
    print(f"corpus filterless PnL (sum of net_pnl_after_haircut): "
          f"${corpus['net_pnl_after_haircut'].sum():,.2f}")
    print()

    v1_path = REPO / "ai_loop_data" / "triathlon" / "filterg_threshold_overrides.json"
    v2_path = REPO / "ai_loop_data" / "triathlon" / "filterg_threshold_overrides_v2.json"
    v1_table = load_per_cell(v1_path)
    v2_table = load_per_cell(v2_path)
    print(f"v1 table: {len(v1_table)} cells from {v1_path.name}")
    print(f"v2 table: {len(v2_table)} cells from {v2_path.name}")
    print()

    df_a = replay_config(corpus, table={}, label="A_DORMANT")
    df_b = replay_config(df_a, table=v1_table, label="B_V1")
    df_c = replay_config(df_b, table=v2_table, label="C_V2")
    full = df_c

    summaries = [
        summarize(full, "A_DORMANT"),
        summarize(full, "B_V1"),
        summarize(full, "C_V2"),
    ]
    print("=" * 100)
    print(f"{'config':<22} {'trades':>7} {'WR':>7} {'net PnL':>11} "
          f"{'avg/trade':>10} {'max DD':>10} {'blocked':>8} {'block%':>7}")
    print("-" * 100)
    for s in summaries:
        print(f"{s['label']:<22} {s['trades']:>7d} {s['wr']*100:>6.2f}% "
              f"${s['pnl']:>10,.2f} ${s['avg_per_trade']:>9.2f} "
              f"${s['max_dd']:>9,.2f} {s['blocked']:>8d} {s['block_rate']*100:>6.2f}%")
    print("=" * 100)

    # Deltas
    a, b, c = summaries
    print(f"\nDELTA C_V2 vs A_DORMANT (the meaningful answer — what per-cell activation buys vs status quo):")
    print(f"  trades: {c['trades'] - a['trades']:+d}  ({(c['trades']-a['trades'])/max(1,a['trades'])*100:+.2f}%)")
    print(f"  WR:     {(c['wr']-a['wr'])*100:+.2f}pp")
    print(f"  PnL:    ${c['pnl'] - a['pnl']:+,.2f}")
    print(f"  DD:     ${c['max_dd'] - a['max_dd']:+,.2f}  (more negative = worse)")
    print(f"\nDELTA C_V2 vs B_V1 (corpus-correction effect alone — same activation, different calibration):")
    print(f"  trades: {c['trades'] - b['trades']:+d}")
    print(f"  WR:     {(c['wr']-b['wr'])*100:+.2f}pp")
    print(f"  PnL:    ${c['pnl'] - b['pnl']:+,.2f}")
    print(f"  DD:     ${c['max_dd'] - b['max_dd']:+,.2f}")

    # Per-month for V2 vs DORMANT
    print("\n" + "=" * 100)
    print("PER-MONTH BREAKDOWN (C_V2 vs A_DORMANT)")
    print("=" * 100)
    pm_a = per_month(full, "A_DORMANT")
    pm_c = per_month(full, "C_V2")
    months = sorted(set(pm_a.index) | set(pm_c.index))
    print(f"{'month':<10} {'A trades':>9} {'A pnl':>10} {'A wr':>7} | "
          f"{'C trades':>9} {'C pnl':>10} {'C wr':>7} | "
          f"{'Δtrd':>6} {'Δpnl':>10}")
    print("-" * 100)
    for m in months:
        a_r = pm_a.loc[m] if m in pm_a.index else None
        c_r = pm_c.loc[m] if m in pm_c.index else None
        a_t = int(a_r["trades"]) if a_r is not None else 0
        a_p = float(a_r["pnl"]) if a_r is not None else 0.0
        a_w = float(a_r["wr"]) if a_r is not None else 0.0
        c_t = int(c_r["trades"]) if c_r is not None else 0
        c_p = float(c_r["pnl"]) if c_r is not None else 0.0
        c_w = float(c_r["wr"]) if c_r is not None else 0.0
        print(f"{m:<10} {a_t:>9d} ${a_p:>9,.2f} {a_w*100:>6.1f}% | "
              f"{c_t:>9d} ${c_p:>9,.2f} {c_w*100:>6.1f}% | "
              f"{c_t-a_t:>+6d} ${c_p-a_p:>+9,.2f}")

    # By-strategy
    print("\n" + "=" * 100)
    print("BY-STRATEGY BREAKDOWN")
    print("=" * 100)
    for label in ["A_DORMANT", "B_V1", "C_V2"]:
        bs = by_strategy(full, label)
        print(f"\n{label}:")
        for strat, row in bs.iterrows():
            print(f"  {strat:<20} trades={int(row['trades']):>5d} "
                  f"wr={row['wr']*100:>6.2f}% pnl=${float(row['pnl']):>10,.2f}")

    # Save annotated df for journal
    out_csv = REPO / "artifacts" / "percell_3way_results.csv"
    cols = ["ts", "strategy", "regime_label", "time_bucket", "pc_mult",
            "fg_proba", "net_pnl_after_haircut",
            "A_DORMANT_fired", "A_DORMANT_eff_thr",
            "B_V1_fired", "B_V1_eff_thr",
            "C_V2_fired", "C_V2_eff_thr"]
    full[cols].to_csv(out_csv, index=False)
    print(f"\nannotated rows -> {out_csv}")

    # Also save summary JSON
    out_json = REPO / "artifacts" / "percell_3way_summary.json"
    out_json.write_text(json.dumps({
        "configs": summaries,
        "deltas": {
            "C_V2_vs_A_DORMANT": {
                "trades": c["trades"] - a["trades"],
                "wr_pp": (c["wr"] - a["wr"]) * 100,
                "pnl": c["pnl"] - a["pnl"],
                "dd": c["max_dd"] - a["max_dd"],
            },
            "C_V2_vs_B_V1": {
                "trades": c["trades"] - b["trades"],
                "wr_pp": (c["wr"] - b["wr"]) * 100,
                "pnl": c["pnl"] - b["pnl"],
                "dd": c["max_dd"] - b["max_dd"],
            },
        },
        "per_month_A": {m: {"trades": int(pm_a.loc[m]["trades"]),
                            "pnl": float(pm_a.loc[m]["pnl"]),
                            "wr": float(pm_a.loc[m]["wr"])} for m in pm_a.index},
        "per_month_C": {m: {"trades": int(pm_c.loc[m]["trades"]),
                            "pnl": float(pm_c.loc[m]["pnl"]),
                            "wr": float(pm_c.loc[m]["wr"])} for m in pm_c.index},
    }, indent=2))
    print(f"summary JSON -> {out_json}")


if __name__ == "__main__":
    main()
