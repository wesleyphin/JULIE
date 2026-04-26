"""V11 Phase 6 — gate landscape under FULL integrated stepped-SL sim.

Reads artifacts/v11_corpus_full_stepped.parquet and applies the same Pareto
configs from /tmp/v11_realistic_sim_report.md, evaluating gate passage with
the FULL integrated sim (BE-arm + Pivot Trail INSIDE the bar walk).

Outputs:
  artifacts/v11_full_stepped_landscape.parquet
  artifacts/v11_full_stepped_summary.json
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "artifacts/v11_corpus_full_stepped.parquet"
LAND_OUT = ROOT / "artifacts/v11_full_stepped_landscape.parquet"
SUM_OUT = ROOT / "artifacts/v11_full_stepped_summary.json"

HAIRCUT = 7.50

# Gates (G1 has two cuts evaluated: $870 strict and $1000 relaxed)
G1_DD_STRICT = 870.0
G1_DD_RELAXED = 1000.0
G2_PNL_BASELINE = -2886.25
G2_TRADES_MAX = 560
G3_N_OOS_MIN = 50
G4_WR_MIN = 0.55

CB_THRS = [None, -200.0, -300.0, -500.0]


def compute_dd_pnl(s: pd.Series):
    s = s.reset_index(drop=True)
    if len(s) == 0:
        return 0.0, 0.0, 0, 0.0
    equity = s.cumsum()
    dd = max(float((equity.cummax() - equity).max()), 0.0)
    return float(s.sum()), dd, int(len(s)), float((s > 0).mean())


def gate_eval(n, pnl, dd, wr, g1_cut):
    return {
        f"G1_dd_le_{int(g1_cut)}": dd <= g1_cut,
        "G2_pnl_ok": (pnl >= G2_PNL_BASELINE) and (n <= G2_TRADES_MAX),
        "G3_n_oos": n >= G3_N_OOS_MIN,
        "G4_wr_ge_55": wr >= G4_WR_MIN,
    }


def gate_str(g):
    keys = sorted(g.keys(), key=lambda k: (0 if k.startswith("G1") else 1,
                                           1 if k.startswith("G2") else 2,
                                           2 if k.startswith("G3") else 3,
                                           3 if k.startswith("G4") else 4, k))
    return "".join("Y" if g[k] else "N" for k in keys)


def block_mask_for_head(holdout, head, thr):
    proba_col = {
        "kalshi_de3": "kalshi_proba",
        "lfo_de3": "lfo_proba",
        "pct_de3": "pct_proba",
        "filterg_de3": "fg_proba",
        "pivot_de3": "pivot_proba",
    }[head]
    block = holdout[proba_col] >= thr
    if head == "kalshi_de3":
        applicable = (holdout["family"] != "regimeadaptive") & (holdout["in_kalshi_window"] == True)
        block = block & applicable
    else:
        applicable = (holdout["family"] != "regimeadaptive")
        block = block & applicable
    return block


def apply_daily_cb(df, cb, *, ts_col="ts", net_col="net_pnl"):
    if cb is None:
        return df
    d = df.sort_values(ts_col).reset_index(drop=True).copy()
    if not len(d):
        return d
    d["_date"] = pd.to_datetime(d[ts_col]).dt.tz_convert("US/Eastern").dt.date
    keep = []
    for date, g in d.groupby("_date", sort=False):
        cum = 0.0
        tripped = False
        for _, r in g.iterrows():
            if tripped:
                keep.append(False)
                continue
            keep.append(True)
            cum += float(r[net_col])
            if cum <= cb:
                tripped = True
    d["_keep"] = keep
    return d[d["_keep"]].drop(columns=["_date", "_keep"]).reset_index(drop=True)


def build_subset(holdout, source, params):
    if source == "baseline":
        return holdout.copy()
    if source.startswith("head:"):
        head = source.split(":", 1)[1]
        return holdout.loc[~block_mask_for_head(holdout, head, float(params["thr"]))].copy()
    if source.startswith("stack:"):
        stack = source.split(":", 1)[1]
        block = pd.Series(False, index=holdout.index)
        if stack == "B_FG_LFO":
            block |= block_mask_for_head(holdout, "filterg_de3", float(params["fg_thr"]))
            block |= block_mask_for_head(holdout, "lfo_de3", float(params["lfo_thr"]))
        elif stack == "C_FG_K_LFO":
            block |= block_mask_for_head(holdout, "filterg_de3", float(params["fg_thr"]))
            block |= block_mask_for_head(holdout, "kalshi_de3", float(params["k_thr"]))
            block |= block_mask_for_head(holdout, "lfo_de3", float(params["lfo_thr"]))
        else:
            raise ValueError(f"unknown stack {stack}")
        return holdout.loc[~block].copy()
    raise ValueError(source)


def evaluate(df, net_col):
    df = df.sort_values("ts").reset_index(drop=True)
    pnl, dd, n, wr = compute_dd_pnl(df[net_col])
    g_strict = gate_eval(n, pnl, dd, wr, G1_DD_STRICT)
    g_relax = gate_eval(n, pnl, dd, wr, G1_DD_RELAXED)
    return {
        "n": n, "WR": wr, "PnL": pnl, "DD": dd,
        "gates_strict": gate_str(g_strict),
        "gates_relaxed": gate_str(g_relax),
        "all_4_strict": all(g_strict.values()),
        "all_4_relaxed": all(g_relax.values()),
    }


def main():
    df = pd.read_parquet(CORPUS)
    df["ts"] = pd.to_datetime(df["ts"])
    if df["ts"].dt.tz is None:
        df["ts"] = df["ts"].dt.tz_localize("US/Eastern")
    else:
        df["ts"] = df["ts"].dt.tz_convert("US/Eastern")
    df["ts_naive"] = df["ts"].dt.tz_localize(None)

    holdout = df[
        (df["ts_naive"] >= "2026-01-01")
        & (df["allowed_by_friend_rule"] == True)
    ].copy().sort_values("ts").reset_index(drop=True)
    print(f"holdout rows: {len(holdout)}")

    # Use net_pnl_full_stepped as the PnL field
    holdout = holdout.assign(net_pnl=holdout["net_pnl_full_stepped"].astype(float))

    pareto_configs = [
        {"name": "baseline", "source": "baseline", "params": {}},
        {"name": "kalshi_de3@0.40", "source": "head:kalshi_de3", "params": {"thr": 0.40}},
        {"name": "kalshi_de3@0.30", "source": "head:kalshi_de3", "params": {"thr": 0.30}},
        {"name": "kalshi_de3@0.20", "source": "head:kalshi_de3", "params": {"thr": 0.20}},
        {"name": "lfo_de3@0.10", "source": "head:lfo_de3", "params": {"thr": 0.10}},
        {"name": "lfo_de3@0.20", "source": "head:lfo_de3", "params": {"thr": 0.20}},
        {"name": "pct_de3@0.50", "source": "head:pct_de3", "params": {"thr": 0.50}},
        {"name": "pct_de3@0.40", "source": "head:pct_de3", "params": {"thr": 0.40}},
        {"name": "pivot_de3@0.40", "source": "head:pivot_de3", "params": {"thr": 0.40}},
        {"name": "pivot_de3@0.30", "source": "head:pivot_de3", "params": {"thr": 0.30}},
        {"name": "filterg_de3@0.10", "source": "head:filterg_de3", "params": {"thr": 0.10}},
        {"name": "filterg_de3@0.20", "source": "head:filterg_de3", "params": {"thr": 0.20}},
        {"name": "stackB_fg0.60_lfo0.50", "source": "stack:B_FG_LFO", "params": {"fg_thr": 0.60, "lfo_thr": 0.50}},
        {"name": "stackC_fg0.50_k0.20_lfo0.40", "source": "stack:C_FG_K_LFO", "params": {"fg_thr": 0.50, "k_thr": 0.20, "lfo_thr": 0.40}},
        {"name": "stackC_fg0.50_k0.20_lfo0.30", "source": "stack:C_FG_K_LFO", "params": {"fg_thr": 0.50, "k_thr": 0.20, "lfo_thr": 0.30}},
    ]

    landscape = []
    for cfg in pareto_configs:
        sub = build_subset(holdout, cfg["source"], cfg["params"])
        for cb in CB_THRS:
            cb_name = "none" if cb is None else f"{int(cb)}"
            sub2 = sub.copy().sort_values("ts").reset_index(drop=True)
            if cb is not None:
                sub2 = apply_daily_cb(sub2, cb, net_col="net_pnl")
            r = evaluate(sub2, "net_pnl")
            landscape.append({
                "config": cfg["name"],
                "source": cfg["source"],
                "params": json.dumps(cfg["params"]),
                "cb_thr": cb_name,
                "n": r["n"], "WR": r["WR"], "PnL": r["PnL"], "DD": r["DD"],
                "gates_strict": r["gates_strict"],
                "gates_relaxed": r["gates_relaxed"],
                "all_4_strict": r["all_4_strict"],
                "all_4_relaxed": r["all_4_relaxed"],
            })

    land_df = pd.DataFrame(landscape)
    land_df.to_parquet(LAND_OUT)
    print(f"\nlandscape rows: {len(land_df)}; written to {LAND_OUT}")

    n_strict = int(land_df["all_4_strict"].sum())
    n_relax = int(land_df["all_4_relaxed"].sum())
    print(f"\nconfigs passing all 4 gates @ G1=$870 strict: {n_strict}")
    print(f"configs passing all 4 gates @ G1=$1000 relaxed: {n_relax}")
    if n_strict:
        print("\nTOP STRICT PASSING CONFIGS:")
        print(land_df[land_df["all_4_strict"]].sort_values(["WR", "PnL"], ascending=[False, False]).to_string(index=False))
    if n_relax:
        print("\nTOP RELAXED PASSING CONFIGS:")
        print(land_df[land_df["all_4_relaxed"]].sort_values(["WR", "PnL"], ascending=[False, False]).to_string(index=False))

    # Closest configs
    print("\nClosest configs (top by WR desc, then DD asc, then PnL desc):")
    print(
        land_df.sort_values(["WR", "DD", "PnL"], ascending=[False, True, False])
        .head(20)[["config", "cb_thr", "n", "WR", "PnL", "DD",
                   "gates_strict", "gates_relaxed"]]
        .to_string(index=False)
    )

    # By DD
    print("\nLowest DD configs:")
    print(
        land_df.sort_values(["DD", "WR"], ascending=[True, False])
        .head(15)[["config", "cb_thr", "n", "WR", "PnL", "DD",
                   "gates_strict", "gates_relaxed"]]
        .to_string(index=False)
    )

    summary = {
        "holdout_rows": int(len(holdout)),
        "n_configs_evaluated": int(len(land_df)),
        "n_passing_all_4_strict_g870": n_strict,
        "n_passing_all_4_relaxed_g1000": n_relax,
        "be_arm_fired_holdout": int(holdout["be_armed"].sum()),
        "pivot_armed_holdout": int(holdout["pivot_armed"].sum()),
        "both_fired_holdout": int(((holdout["be_armed"]) & (holdout["pivot_armed"])).sum()),
    }
    if n_strict or n_relax:
        passing = land_df[
            land_df["all_4_strict"] | land_df["all_4_relaxed"]
        ].sort_values(["WR", "PnL"], ascending=[False, False])
        summary["top_passing_configs"] = passing.head(20).to_dict(orient="records")
    else:
        summary["top_passing_configs"] = []
        # closest
        closest = land_df.sort_values(["WR", "DD", "PnL"], ascending=[False, True, False]).head(15)
        summary["closest_configs"] = closest.to_dict(orient="records")

    with open(SUM_OUT, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nsummary written to {SUM_OUT}")


if __name__ == "__main__":
    main()
