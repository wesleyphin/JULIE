"""V11 realistic-sim recomputation.

Adds three structural early-exit corrections that the v2/v11 walk-forward
sim does NOT model, then re-evaluates the closest-to-passing configs from
/tmp/v11_1000dd_exploration_report.md.

  1. BE-arm correction: if exit_reason=='stop' AND mfe_points >= tp_dist*0.40,
     replace raw_pnl with 0.0 (BE flat). Crude — assumes the live BE-arm
     would have moved the stop to entry once MFE crossed the trigger.
  2. SL=6pt re-walk: only for DE3 rows where original SL distance is ~10pt;
     uses sl6_* columns from the rebuilt corpus.
  3. Daily circuit breaker: at the equity-curve level, stop trading once
     cumulative day P&L drops below cb_threshold (in dollars).

Outputs:
  artifacts/v11_realistic_sim_landscape.parquet  -- per-config recomputation
  artifacts/v11_realistic_sim_summary.json       -- top-line
"""
from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "artifacts/v11_training_corpus_with_mfe.parquet"
LAND_OUT = ROOT / "artifacts/v11_realistic_sim_landscape.parquet"
SUM_OUT = ROOT / "artifacts/v11_realistic_sim_summary.json"

HAIRCUT = 7.50

# Gates (from v11_1000dd_explore.py)
G1_DD_MAX = 1000.0
G2_PNL_BASELINE = -2886.25
G2_TRADES_MAX = 560
G3_N_OOS_MIN = 50
G4_WR_MIN = 0.55

CB_THRS = [-200.0, -300.0, -500.0]
BE_TRIGGER_PCT = 0.40  # julie001.py config: trade_management.break_even.trigger_pct


def compute_dd_pnl(pnl_series: pd.Series) -> tuple[float, float, int, float]:
    s = pnl_series.reset_index(drop=True)
    n = len(s)
    if n == 0:
        return 0.0, 0.0, 0, 0.0
    equity = s.cumsum()
    dd = max(float((equity.cummax() - equity).max()), 0.0)
    pnl_total = float(s.sum())
    wr = float((s > 0).mean())
    return pnl_total, dd, n, wr


def gate_eval(n: int, pnl: float, dd: float, wr: float) -> dict:
    return {
        "G1_dd_le_1000": dd <= G1_DD_MAX,
        "G2_pnl_ok": (pnl >= G2_PNL_BASELINE) and (n <= G2_TRADES_MAX),
        "G3_n_oos": n >= G3_N_OOS_MIN,
        "G4_wr_ge_55": wr >= G4_WR_MIN,
    }


def gate_str(g: dict) -> str:
    return "".join(["Y" if g[k] else "N" for k in ["G1_dd_le_1000", "G2_pnl_ok", "G3_n_oos", "G4_wr_ge_55"]])


def apply_be_arm(df: pd.DataFrame, *,
                 reason_col: str = "exit_reason",
                 raw_col: str = "raw_pnl",
                 mfe_col: str = "mfe_points",
                 sl_dist_col: str = "sl",
                 tp_dist_col: str = "tp",
                 ) -> pd.Series:
    """Return a NEW raw_pnl Series with BE-arm correction applied.

    BE-arm fires when: exit_reason in {'stop','stop_pessimistic'} AND mfe >= tp*0.40.
    Outcome: raw_pnl replaced with 0.0 (BE flat — exit at entry, no SL loss).
    Net P&L (after the $7.50 haircut) becomes -$7.50 for these rows, mirroring
    the live bot's BE exit (touched stop at entry => P&L=0 minus fee).
    """
    raw = df[raw_col].astype(float).copy()
    is_stop = df[reason_col].isin(["stop", "stop_pessimistic"])
    threshold = df[tp_dist_col].astype(float) * BE_TRIGGER_PCT
    fired = is_stop & (df[mfe_col].astype(float) >= threshold)
    raw[fired] = 0.0
    return raw, fired


def apply_sl6(df: pd.DataFrame) -> pd.Series:
    """Return a NEW raw_pnl Series using SL=6pt for eligible rows (DE3 with
    original sl=10pt). For non-eligible rows, fall back to the original raw_pnl.
    """
    raw = df["raw_pnl"].astype(float).copy()
    eligible = df["sl6_eligible"].astype(bool) & df["sl6_raw_pnl"].notna()
    raw[eligible] = df.loc[eligible, "sl6_raw_pnl"].astype(float)
    return raw


def apply_be_arm_on_pnl(raw: pd.Series, df: pd.DataFrame, *,
                        reason_col: str, mfe_col: str, tp_col: str = "tp",
                        ) -> pd.Series:
    """Variant: apply BE-arm to a pre-computed raw_pnl Series.

    Used to chain SL=6pt + BE-arm: SL=6 produces a (possibly different) exit
    reason; we then apply BE-arm based on that variant's mfe.
    """
    out = raw.astype(float).copy()
    is_stop = df[reason_col].isin(["stop", "stop_pessimistic"])
    threshold = df[tp_col].astype(float) * BE_TRIGGER_PCT
    fired = is_stop & (df[mfe_col].astype(float) >= threshold)
    out[fired] = 0.0
    return out


def apply_daily_cb(trades_df: pd.DataFrame, cb_threshold: float, *,
                   ts_col: str = "ts", net_col: str = "net_pnl") -> pd.DataFrame:
    """Daily circuit breaker: skip trades on a day once cumulative day P&L
    drops below cb_threshold."""
    if cb_threshold is None:
        return trades_df
    df = trades_df.sort_values(ts_col).reset_index(drop=True).copy()
    if len(df) == 0:
        return df
    df["_date"] = pd.to_datetime(df[ts_col]).dt.tz_convert("US/Eastern").dt.date
    keep_mask = []
    for date, group in df.groupby("_date", sort=False):
        cum = 0.0
        cb_tripped = False
        for _, row in group.iterrows():
            if cb_tripped:
                keep_mask.append(False)
                continue
            keep_mask.append(True)
            cum += float(row[net_col])
            if cum <= cb_threshold:
                cb_tripped = True
    df["_keep"] = keep_mask
    out = df[df["_keep"]].drop(columns=["_date", "_keep"]).reset_index(drop=True)
    return out


def block_mask_for_head(holdout: pd.DataFrame, head: str, thr: float) -> pd.Series:
    proba_col = {
        "kalshi_de3": "kalshi_proba",
        "lfo_de3": "lfo_proba",
        "pct_de3": "pct_proba",
        "filterg_de3": "fg_proba",
        "pivot_de3": "pivot_proba",
    }[head]
    proba = holdout[proba_col]
    block = proba >= thr
    if head == "kalshi_de3":
        applicable = (holdout["family"] != "regimeadaptive") & (holdout["in_kalshi_window"] == True)
        block = block & applicable
    else:
        applicable = (holdout["family"] != "regimeadaptive")
        block = block & applicable
    return block


def evaluate(df: pd.DataFrame, net_col: str, *, ts_col: str = "ts") -> dict:
    df = df.sort_values(ts_col).reset_index(drop=True)
    pnl, dd, n, wr = compute_dd_pnl(df[net_col])
    g = gate_eval(n, pnl, dd, wr)
    return {
        "n": n, "WR": wr, "PnL": pnl, "DD": dd,
        "gates_str": gate_str(g),
        "all_4_pass": all(g.values()),
    }


def build_variants(holdout: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return dict variant_name -> dataframe with `net_pnl` column.

    Variants:
      base: original net_pnl_after_haircut (no add-ons)
      bea : BE-arm only
      sl6 : SL=6pt only
      bea_sl6: BE-arm + SL=6pt
    Daily CB is applied DOWNSTREAM (per gate eval).
    """
    out = {}

    # 1. base
    base = holdout.copy()
    base["net_pnl"] = base["net_pnl_after_haircut"].astype(float)
    out["base"] = base

    # 2. BE-arm
    bea_raw, _bea_fired = apply_be_arm(holdout)
    bea = holdout.copy()
    # Net = raw - HAIRCUT (no fee waiver — we charge the fee even on BE)
    bea["net_pnl"] = bea_raw - HAIRCUT
    out["bea"] = bea

    # 3. SL=6pt
    sl6_raw = apply_sl6(holdout)
    sl6 = holdout.copy()
    sl6["net_pnl"] = sl6_raw - HAIRCUT
    out["sl6"] = sl6

    # 4. SL=6pt + BE-arm (BE-arm uses sl6 exit_reason / sl6_mfe where eligible,
    #    falls back to original for non-eligible RA rows)
    sl6_bea = holdout.copy()
    # First take sl6 raw_pnl, then apply BE-arm using sl6 exit reason
    raw = apply_sl6(holdout)
    # Compose composite reason/mfe columns
    eligible = holdout["sl6_eligible"].astype(bool) & holdout["sl6_exit_reason"].notna()
    composite_reason = holdout["exit_reason"].copy().astype(object)
    composite_reason[eligible] = holdout.loc[eligible, "sl6_exit_reason"].values
    composite_mfe = holdout["mfe_points"].astype(float).copy()
    sl6_mfe_vals = holdout.loc[eligible, "sl6_mfe_points"].astype(float)
    composite_mfe[eligible] = sl6_mfe_vals
    tmp = holdout.copy()
    tmp["_reason"] = composite_reason
    tmp["_mfe"] = composite_mfe
    raw2 = apply_be_arm_on_pnl(raw, tmp, reason_col="_reason", mfe_col="_mfe")
    sl6_bea["net_pnl"] = raw2 - HAIRCUT
    out["bea_sl6"] = sl6_bea

    return out


def build_subset(holdout_with_net: pd.DataFrame, source: str, params: dict) -> pd.DataFrame:
    """Apply a config (head threshold, stack, or all) to holdout and return kept subset."""
    h = holdout_with_net
    if source == "baseline":
        return h.copy()
    if source.startswith("head:"):
        head = source.split(":", 1)[1]
        thr = float(params["thr"])
        block = block_mask_for_head(h, head, thr)
        return h.loc[~block].copy()
    if source.startswith("stack:"):
        stack = source.split(":", 1)[1]
        block = pd.Series(False, index=h.index)
        if stack == "B_FG_LFO":
            block |= block_mask_for_head(h, "filterg_de3", float(params["fg_thr"]))
            block |= block_mask_for_head(h, "lfo_de3", float(params["lfo_thr"]))
        elif stack == "C_FG_K_LFO":
            block |= block_mask_for_head(h, "filterg_de3", float(params["fg_thr"]))
            block |= block_mask_for_head(h, "kalshi_de3", float(params["k_thr"]))
            block |= block_mask_for_head(h, "lfo_de3", float(params["lfo_thr"]))
        else:
            raise ValueError(f"unknown stack {stack}")
        return h.loc[~block].copy()
    raise ValueError(f"unknown source {source}")


def main():
    df = pd.read_parquet(CORPUS)
    df["ts"] = pd.to_datetime(df["ts"])
    if df["ts"].dt.tz is None:
        df["ts"] = df["ts"].dt.tz_localize("US/Eastern")
    else:
        df["ts"] = df["ts"].dt.tz_convert("US/Eastern")
    df["ts_naive"] = df["ts"].dt.tz_localize(None)

    holdout = df[(df["ts_naive"] >= "2026-01-01") & (df["allowed_by_friend_rule"] == True)].copy()
    holdout = holdout.sort_values("ts").reset_index(drop=True)
    print(f"holdout rows: {len(holdout)}")

    variants = build_variants(holdout)
    print(f"sim variants: {list(variants.keys())}")

    # Stage 3: per-add-on impact on FULL holdout (no head filtering)
    stage3 = []
    for vname, vdf in variants.items():
        for cb in [None] + CB_THRS:
            cb_name = "none" if cb is None else f"{int(cb)}"
            sub = vdf.copy().sort_values("ts").reset_index(drop=True)
            if cb is not None:
                sub = apply_daily_cb(sub, cb, net_col="net_pnl")
            r = evaluate(sub, "net_pnl")
            stage3.append({
                "sim_variant": vname,
                "cb_thr": cb_name,
                "n": r["n"], "WR": r["WR"], "PnL": r["PnL"], "DD": r["DD"],
                "gates_str": r["gates_str"],
                "all_4_pass": r["all_4_pass"],
            })
    stage3_df = pd.DataFrame(stage3)
    print("\nStage 3 — per-add-on impact on full holdout:")
    print(stage3_df.to_string(index=False))

    # Stage 4: closest configs under realistic sim
    # Top-5 Pareto from /tmp/v11_1000dd_exploration_report.md:
    #   1) pivot_de3 thr 0.40
    #   2) Stack C: fg=0.50, lfo=0.40 (representative; report says 0.5-0.6)
    #   3) Stack B: fg=0.60, lfo=0.50
    #   4) filterg_de3 thr 0.10 (saturated)
    #   5) Stack C: fg=0.50, lfo=0.30
    # Plus the "baseline" full-holdout pseudo-config for reference.
    pareto_configs = [
        {"name": "pivot_de3@0.40", "source": "head:pivot_de3", "params": {"thr": 0.40}},
        {"name": "stackC_fg0.50_k0.20_lfo0.40", "source": "stack:C_FG_K_LFO", "params": {"fg_thr": 0.50, "k_thr": 0.20, "lfo_thr": 0.40}},
        {"name": "stackB_fg0.60_lfo0.50", "source": "stack:B_FG_LFO", "params": {"fg_thr": 0.60, "lfo_thr": 0.50}},
        {"name": "filterg_de3@0.10", "source": "head:filterg_de3", "params": {"thr": 0.10}},
        {"name": "stackC_fg0.50_k0.20_lfo0.30", "source": "stack:C_FG_K_LFO", "params": {"fg_thr": 0.50, "k_thr": 0.20, "lfo_thr": 0.30}},
        {"name": "kalshi_de3@0.40", "source": "head:kalshi_de3", "params": {"thr": 0.40}},
        {"name": "lfo_de3@0.10", "source": "head:lfo_de3", "params": {"thr": 0.10}},
        {"name": "baseline", "source": "baseline", "params": {}},
    ]

    landscape = []
    for cfg in pareto_configs:
        for vname, vdf in variants.items():
            sub = build_subset(vdf, cfg["source"], cfg["params"])
            for cb in [None] + CB_THRS:
                cb_name = "none" if cb is None else f"{int(cb)}"
                sub2 = sub.copy().sort_values("ts").reset_index(drop=True)
                if cb is not None:
                    sub2 = apply_daily_cb(sub2, cb, net_col="net_pnl")
                r = evaluate(sub2, "net_pnl")
                landscape.append({
                    "config": cfg["name"],
                    "source": cfg["source"],
                    "params": json.dumps(cfg["params"]),
                    "sim_variant": vname,
                    "cb_thr": cb_name,
                    "n": r["n"], "WR": r["WR"], "PnL": r["PnL"], "DD": r["DD"],
                    "gates_str": r["gates_str"],
                    "all_4_pass": r["all_4_pass"],
                })
    land_df = pd.DataFrame(landscape)
    land_df.to_parquet(LAND_OUT)
    print(f"\nlandscape rows: {len(land_df)}; written to {LAND_OUT}")

    n_pass = land_df["all_4_pass"].sum()
    print(f"configs passing all 4 gates: {n_pass}")
    if n_pass:
        print("\nTOP PASSING CONFIGS:")
        print(land_df[land_df["all_4_pass"]].sort_values(["WR", "PnL"], ascending=[False, False]).to_string(index=False))

    # Closest configs (relax G4 if no full passes)
    print("\nClosest configs by gates_str then WR desc (top 20):")
    n_pass_arr = land_df["gates_str"].map(lambda s: s.count("Y"))
    land_df["_n_pass"] = n_pass_arr
    print(land_df.sort_values(["_n_pass", "WR", "PnL"], ascending=[False, False, False]).head(20)[
        ["config", "sim_variant", "cb_thr", "n", "WR", "PnL", "DD", "gates_str"]
    ].to_string(index=False))

    # Top-line summary
    summary = {
        "holdout_rows": int(len(holdout)),
        "sim_variants": list(variants.keys()),
        "cb_thresholds": CB_THRS,
        "n_configs_evaluated": int(len(land_df)),
        "n_configs_passing_all_4_gates": int(n_pass),
        "stage3_full_holdout_per_addon": stage3_df.to_dict(orient="records"),
    }
    if n_pass:
        passing = land_df[land_df["all_4_pass"]].sort_values(["WR", "PnL"], ascending=[False, False])
        summary["top_passing_configs"] = passing.head(20).drop(columns=["_n_pass"]).to_dict(orient="records")
    else:
        summary["top_passing_configs"] = []
        # Closest by # gates, then WR
        closest = land_df.sort_values(["_n_pass", "WR", "PnL"], ascending=[False, False, False]).head(15)
        summary["closest_configs"] = closest.drop(columns=["_n_pass"]).to_dict(orient="records")

    # BE-arm impact stats
    bea_raw, fired = apply_be_arm(holdout)
    summary["be_arm_stats"] = {
        "n_holdout": int(len(holdout)),
        "n_be_arm_fired": int(fired.sum()),
        "pct_be_arm_fired": float(fired.mean() * 100.0),
        "loss_avoided_dollars": float((holdout.loc[fired, "raw_pnl"].astype(float).sum()) * -1.0),  # raw was negative; we replaced with 0
    }
    # SL6 impact stats
    sl6_raw = apply_sl6(holdout)
    summary["sl6_stats"] = {
        "n_eligible": int(holdout["sl6_eligible"].sum()),
        "delta_pnl_eligible": float((sl6_raw - holdout["raw_pnl"].astype(float)).sum()),
    }

    with open(SUM_OUT, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nsummary written to {SUM_OUT}")


if __name__ == "__main__":
    main()
