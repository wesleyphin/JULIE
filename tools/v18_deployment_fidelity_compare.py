"""V18 + Recipe B 5-way deployment-fidelity-FAST comparison on the v11 corpus.

Method = friend's speed (cached features + vectorized PnL/DD) + deployment
realism (single-position, NY-only, friend's same-side rule already in
`allowed_by_friend_rule`, downstream filter stack, size-aware PnL).

Configs:
  A: filterless = friend's same-side rule + NY-only + single-position
  B: production proxy = A + Filter G base 0.35 (the dominant gate per §8.25)
  C: V15 stacked meta @ 0.65 (size=1)
  D: V18 stacked meta @ 0.60 (size=1, no Recipe B)
  E: V18 + Recipe B tiered sizing (DEPLOYED STATE)

For each: trades, WR, net PnL (size-aware for E), avg/trade, max DD on
TIME-ORDERED size-aware cumulative equity peak-to-trough, by-side, by-strategy.

Reads:
  artifacts/v12_training_corpus.parquet  (3,438 candidates with corrected sim PnL)
  artifacts/v11_corpus_with_kronos_features.parquet  (per-row Kronos features)
  artifacts/regime_ml_kalshi_v12/de3/model.joblib  (k12 proba)
  artifacts/regime_ml_meta_v15/de3/model.joblib    (V15)
  artifacts/regime_ml_v18_de3/de3/model.joblib     (V18)

Writes:
  artifacts/v18_5way_deployment_fidelity_summary.json
  artifacts/v18_5way_deployment_fidelity_per_month.csv
  artifacts/v18_5way_deployment_fidelity_results.csv
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
V12_CORPUS = REPO / "artifacts" / "v12_training_corpus.parquet"
KRONOS_FEATS = REPO / "artifacts" / "v11_corpus_with_kronos_features.parquet"
KALSHI_V12 = REPO / "artifacts" / "regime_ml_kalshi_v12" / "de3" / "model.joblib"
V15_DE3 = REPO / "artifacts" / "regime_ml_meta_v15" / "de3" / "model.joblib"
V18_DE3 = REPO / "artifacts" / "regime_ml_v18_de3" / "de3" / "model.joblib"

OUT_SUMMARY = REPO / "artifacts" / "v18_5way_deployment_fidelity_summary.json"
OUT_PER_MONTH = REPO / "artifacts" / "v18_5way_deployment_fidelity_per_month.csv"
OUT_RESULTS = REPO / "artifacts" / "v18_5way_deployment_fidelity_results.csv"


def safe_predict(payload: Dict, X_df: pd.DataFrame) -> np.ndarray:
    feats = payload["features"]
    Xs = X_df[feats].copy()
    for c in Xs.columns:
        if Xs[c].isna().any():
            med = Xs[c].median()
            Xs[c] = Xs[c].fillna(med if not pd.isna(med) else 0.0)
    return payload["model"].predict_proba(Xs.values)[:, 1]


def vectorized_single_position(ts_utc: np.ndarray, exit_ts_utc: np.ndarray,
                                fired_mask: np.ndarray) -> np.ndarray:
    """Walk in time order; among rows where fired_mask is True, only keep
    rows whose entry ts is >= prior fired trade's exit ts.

    All ts arrays must already be in nanosecond UTC int64 form, ts-sorted.
    Returns boolean mask of same shape as fired_mask.
    """
    out = np.zeros_like(fired_mask, dtype=bool)
    busy_until = np.iinfo(np.int64).min
    for i in range(len(ts_utc)):
        if not fired_mask[i]:
            continue
        if ts_utc[i] >= busy_until:
            out[i] = True
            et = exit_ts_utc[i]
            busy_until = et if et > 0 else ts_utc[i]
    return out


def recipe_b_size(proba: float) -> int:
    if proba >= 0.85:
        return 10
    if proba >= 0.65:
        return 4
    if proba >= 0.60:
        return 1
    return 0


def evaluate_config(df: pd.DataFrame, gate_mask: np.ndarray,
                    sizes: np.ndarray, label: str) -> Dict:
    """Apply gate_mask + single-position; compute size-aware metrics with
    time-ordered DD."""
    ts_utc = df["_ts_utc_ns"].values
    exit_utc = df["_exit_ts_utc_ns"].values
    fire_after_sp = vectorized_single_position(ts_utc, exit_utc, gate_mask)
    sub = df[fire_after_sp].copy()
    sub_sizes = sizes[fire_after_sp]
    sub["size"] = sub_sizes
    sub["sized_pnl"] = sub["net_pnl_after_haircut"].values * sub_sizes
    n = len(sub)
    if n == 0:
        return {"label": label, "trades": 0, "wins": 0, "losses": 0, "wr": 0.0,
                "pnl": 0.0, "avg_per_trade": 0.0, "max_dd": 0.0,
                "by_side": {}, "by_strategy": {}, "fired_idx": [], "fired_df": sub}
    wins = int((sub["sized_pnl"] > 0).sum())
    losses = int((sub["sized_pnl"] < 0).sum())
    pnl = float(sub["sized_pnl"].sum())
    wr = wins / n
    avg = pnl / n
    # Time-ordered DD on size-aware equity (sub is already time-sorted because
    # df was sorted by ts before evaluate_config was called)
    eq = np.cumsum(sub["sized_pnl"].values)
    peak = np.maximum.accumulate(eq)
    dd = float((eq - peak).min()) if len(eq) else 0.0

    by_side = {}
    for side in sub["side"].unique():
        m = sub["side"] == side
        ns = int(m.sum())
        by_side[side] = {
            "n": ns,
            "wr": float(((m) & (sub["sized_pnl"] > 0)).sum() / max(1, ns)),
            "pnl": float(sub.loc[m, "sized_pnl"].sum()),
        }
    by_strat = {}
    for strat in sub["strategy"].unique():
        m = sub["strategy"] == strat
        ns = int(m.sum())
        by_strat[strat] = {
            "n": ns,
            "wr": float(((m) & (sub["sized_pnl"] > 0)).sum() / max(1, ns)),
            "pnl": float(sub.loc[m, "sized_pnl"].sum()),
        }
    return {
        "label": label,
        "trades": n,
        "wins": wins,
        "losses": losses,
        "wr": wr,
        "pnl": pnl,
        "avg_per_trade": avg,
        "max_dd": dd,
        "by_side": by_side,
        "by_strategy": by_strat,
        "fired_df": sub,
    }


def per_month(fired_df: pd.DataFrame) -> pd.DataFrame:
    f = fired_df.copy()
    f["month"] = pd.to_datetime(f["ts"]).dt.tz_convert("America/New_York").dt.strftime("%Y-%m")
    g = f.groupby("month").agg(
        trades=("ts", "count"),
        wins=("sized_pnl", lambda s: int((s > 0).sum())),
        wr=("sized_pnl", lambda s: float((s > 0).sum() / max(1, len(s)))),
        pnl=("sized_pnl", "sum"),
        max_size=("size", "max"),
    )
    # within-month DD on time-ordered cumulative
    dd_list = []
    for m in g.index:
        eq = np.cumsum(f[f["month"] == m].sort_values("ts")["sized_pnl"].values)
        if len(eq) == 0:
            dd_list.append(0.0)
            continue
        peak = np.maximum.accumulate(eq)
        dd_list.append(float((eq - peak).min()))
    g["dd"] = dd_list
    return g


def main():
    t0 = time.time()
    print("=" * 90)
    print("V18 + Recipe B 5-way DEPLOYMENT-FIDELITY-FAST comparison")
    print("=" * 90)

    # 1) load corpus
    df = pd.read_parquet(V12_CORPUS).sort_values("ts").reset_index(drop=True)
    print(f"corpus: {len(df)} rows  {df['ts'].min()} → {df['ts'].max()}")
    print(f"corpus filterless raw PnL: ${df['net_pnl_after_haircut'].sum():,.2f}")

    # 2) merge Kronos features
    kron = pd.read_parquet(KRONOS_FEATS)
    n_kron_failed = int(kron["kronos_failed"].notna().sum())
    print(f"kronos features cache: {len(kron)} rows  failed={n_kron_failed}")
    if "row_idx" in kron.columns and len(kron) == len(df):
        kron = kron.sort_values("row_idx").reset_index(drop=True)
        for c in ["kronos_max_high_above", "kronos_min_low_below",
                  "kronos_pred_atr_30bar", "kronos_dir_move", "kronos_close_vs_entry",
                  "kronos_failed"]:
            df[c] = kron[c].values
    else:
        # ts-merge fallback
        df["_ts_dt"] = pd.to_datetime(df["ts"])
        kron["_ts_dt"] = pd.to_datetime(kron["ts"])
        df = df.merge(kron[["_ts_dt", "kronos_max_high_above", "kronos_min_low_below",
                             "kronos_pred_atr_30bar", "kronos_dir_move",
                             "kronos_close_vs_entry", "kronos_failed"]],
                       on="_ts_dt", how="left")
        df = df.drop(columns=["_ts_dt"])
        df = df.sort_values("ts").reset_index(drop=True)

    # impute kronos features for failed rows with col-medians
    for c in ["kronos_max_high_above", "kronos_min_low_below", "kronos_pred_atr_30bar",
              "kronos_dir_move", "kronos_close_vs_entry"]:
        if df[c].isna().any():
            med = df[c].median()
            df[c] = df[c].fillna(med if not pd.isna(med) else 0.0)

    # 3) compute kalshi_v12_proba (V15/V18 require it)
    print("\nComputing kalshi_v12_proba ...")
    k12_payload = joblib.load(KALSHI_V12)
    df["kalshi_v12_proba"] = safe_predict(k12_payload, df)

    print("Computing v15_proba ...")
    v15_payload = joblib.load(V15_DE3)
    df["v15_proba"] = safe_predict(v15_payload, df)

    print("Computing v18_proba ...")
    v18_payload = joblib.load(V18_DE3)
    df["v18_proba"] = safe_predict(v18_payload, df)

    print(f"  v15: mean={df['v15_proba'].mean():.3f}  >=0.65={(df['v15_proba']>=0.65).sum()}")
    print(f"  v18: mean={df['v18_proba'].mean():.3f}  >=0.60={(df['v18_proba']>=0.60).sum()}  "
          f">=0.65={(df['v18_proba']>=0.65).sum()}  >=0.85={(df['v18_proba']>=0.85).sum()}")

    # 4) Build deployment-realistic constraint masks
    et = pd.to_datetime(df["ts"]).dt.tz_convert("America/New_York")
    ny_mask = ((et.dt.hour >= 8) & (et.dt.hour < 16)).values
    fa_mask = df["allowed_by_friend_rule"].astype(bool).values

    # ts in UTC ns for vectorized single-position
    df["_ts_utc_ns"] = pd.to_datetime(df["ts"]).dt.tz_convert("UTC").astype("int64")
    df["_exit_ts_utc_ns"] = pd.to_datetime(df["exit_ts"]).dt.tz_convert("UTC").astype("int64")

    base_mask = ny_mask & fa_mask
    print(f"\nfriend ∩ NY-only base mask: {base_mask.sum()} of {len(df)}")

    one_size = np.ones(len(df), dtype=int)

    # 5) Run the 5 configs
    configs = []

    # A: filterless
    a = evaluate_config(df, base_mask, one_size, "A_filterless")
    configs.append(a)

    # B: production proxy = filterless + Filter G base 0.35
    fg_pass = (df["fg_proba"].values < 0.35)
    b_mask = base_mask & fg_pass
    b = evaluate_config(df, b_mask, one_size, "B_filterG_base")
    configs.append(b)

    # C: V15 stacked @ 0.65 size=1
    c_pass = (df["v15_proba"].values >= 0.65)
    c_mask = base_mask & c_pass
    c = evaluate_config(df, c_mask, one_size, "C_V15_065")
    configs.append(c)

    # D: V18 @ 0.60 size=1
    d_pass = (df["v18_proba"].values >= 0.60)
    d_mask = base_mask & d_pass
    d = evaluate_config(df, d_mask, one_size, "D_V18_060_flat")
    configs.append(d)

    # E: V18 + Recipe B tiered sizing
    e_sizes = np.array([recipe_b_size(p) for p in df["v18_proba"].values], dtype=int)
    e_mask = base_mask & (e_sizes > 0)
    e = evaluate_config(df, e_mask, e_sizes, "E_V18_RecipeB_DEPLOYED")
    configs.append(e)

    # 6) Print headline
    print("\n" + "=" * 130)
    print(f"{'config':<28} {'trades':>7} {'WR':>7} {'PnL':>13} {'avg/trade':>11} "
          f"{'max DD':>12} {'wins':>6} {'losses':>7}")
    print("-" * 130)
    for s in configs:
        print(f"{s['label']:<28} {s['trades']:>7d} {s['wr']*100:>6.2f}% "
              f"${s['pnl']:>12,.2f} ${s['avg_per_trade']:>10.2f} "
              f"${s['max_dd']:>11,.2f} {s['wins']:>6d} {s['losses']:>7d}")
    print("=" * 130)

    # 7) Δ vs B (production)
    print(f"\nΔ vs B (production proxy):")
    for s in (a, c, d, e):
        dt = s["trades"] - b["trades"]
        dpnl = s["pnl"] - b["pnl"]
        ddd = s["max_dd"] - b["max_dd"]
        dwr = (s["wr"] - b["wr"]) * 100
        print(f"  {s['label']:<28} Δtrades={dt:+5d}  Δpnl=${dpnl:+10,.2f}  "
              f"ΔWR={dwr:+5.2f}pp  ΔDD=${ddd:+9,.2f}")

    # 8) Recipe B tier breakdown for E
    print("\n" + "=" * 90)
    print("Recipe B tier breakdown (config E)")
    print("=" * 90)
    fired_e = e["fired_df"].copy()
    # the row index of fired_e maps back to df row index
    fired_e["proba"] = df.loc[fired_e.index, "v18_proba"].values
    tier_rows = []
    for tier_name, lo, hi, sz in [("tier10 (>=0.85)", 0.85, 1.01, 10),
                                  ("tier4  (0.65-0.85)", 0.65, 0.85, 4),
                                  ("tier1  (0.60-0.65)", 0.60, 0.65, 1)]:
        mask = (fired_e["proba"] >= lo) & (fired_e["proba"] < hi)
        sub = fired_e[mask]
        n = len(sub)
        if n == 0:
            print(f"  {tier_name:<22} size={sz:<3} n=0")
            tier_rows.append({"tier": tier_name, "size": sz, "n": 0, "wr": 0.0,
                              "pnl": 0.0, "avg": 0.0})
            continue
        wins = int((sub["sized_pnl"] > 0).sum())
        wr = wins / n
        pnl_total = float(sub["sized_pnl"].sum())
        avg = pnl_total / n
        # tier-specific DD (time-ordered cumsum within tier)
        sub_sorted = sub.sort_values("ts")
        eq_t = np.cumsum(sub_sorted["sized_pnl"].values)
        peak_t = np.maximum.accumulate(eq_t)
        dd_t = float((eq_t - peak_t).min()) if len(eq_t) else 0.0
        print(f"  {tier_name:<22} size={sz:<3} n={n:<4} wr={wr*100:>5.2f}%  "
              f"pnl=${pnl_total:>10,.2f}  avg=${avg:>8.2f}  tier-DD=${dd_t:>9,.2f}")
        tier_rows.append({"tier": tier_name, "size": sz, "n": n, "wr": wr,
                          "pnl": pnl_total, "avg": avg, "dd": dd_t})

    # 9) Per-month (E)
    print("\n" + "=" * 90)
    print("Per-month (config E)")
    print("=" * 90)
    pm_e = per_month(e["fired_df"])
    print(pm_e.to_string())

    # 10) DD breach flags
    print("\nDD breach checks (config E per-month):")
    for m, row in pm_e.iterrows():
        flags = []
        if row["dd"] < -870:
            flags.append("DD_>_870")
        if row["dd"] < -1000:
            flags.append("DD_>_1000")
        if row["max_size"] >= 10:
            flags.append("size10_active")
        if flags:
            print(f"  {m}: {flags}  (DD=${row['dd']:.2f})")

    # 11) 14-month projection vs $32.5k/8mo
    print("\n" + "=" * 90)
    months = 14
    e_pnl_per_mo = e["pnl"] / months
    user_proj_per_mo = 32500 / 8
    e_8mo_proj = e_pnl_per_mo * 8
    print(f"E PnL: ${e['pnl']:,.2f} over {months} months → ${e_pnl_per_mo:,.2f}/mo")
    print(f"User projection: $32,500 over 8 months = ${user_proj_per_mo:,.2f}/mo")
    print(f"E scaled to 8 months: ${e_8mo_proj:,.2f}")
    print(f"Ratio of E-8mo to user projection: {e_8mo_proj/32500*100:.1f}%")

    # 12) Save artifacts
    OUT_SUMMARY.write_text(json.dumps({
        "configs": [{k: v for k, v in s.items() if k != "fired_df"} for s in configs],
        "deltas_vs_B": {
            s["label"]: {
                "trades": s["trades"] - b["trades"],
                "pnl": s["pnl"] - b["pnl"],
                "wr_pp": (s["wr"] - b["wr"]) * 100,
                "dd": s["max_dd"] - b["max_dd"],
            } for s in (a, c, d, e)
        },
        "recipe_b_tiers": tier_rows,
        "monthly_projection": {
            "E_pnl_total_14mo": e["pnl"],
            "E_per_month": e_pnl_per_mo,
            "E_scaled_to_8mo": e_8mo_proj,
            "user_projection_8mo": 32500.0,
            "ratio_E_to_user_proj": e_8mo_proj / 32500.0,
        },
        "kronos_imputed_count": n_kron_failed,
        "v18_proba_distribution": {
            "mean": float(df["v18_proba"].mean()),
            "std": float(df["v18_proba"].std()),
            "ge_060": int((df["v18_proba"] >= 0.60).sum()),
            "ge_065": int((df["v18_proba"] >= 0.65).sum()),
            "ge_085": int((df["v18_proba"] >= 0.85).sum()),
        },
        "runtime_seconds": time.time() - t0,
    }, indent=2, default=str))
    print(f"\nsummary -> {OUT_SUMMARY}")
    pm_e.to_csv(OUT_PER_MONTH)
    print(f"per-month -> {OUT_PER_MONTH}")

    # save the annotated decisions for E (for audit)
    out_cols = ["ts", "strategy", "side", "v15_proba", "v18_proba", "fg_proba",
                "kalshi_proba", "kalshi_v12_proba", "lfo_proba", "pct_proba",
                "pivot_proba", "net_pnl_after_haircut",
                "kronos_max_high_above", "kronos_min_low_below",
                "kronos_pred_atr_30bar", "kronos_dir_move", "kronos_close_vs_entry"]
    df[out_cols].to_csv(OUT_RESULTS, index=False)
    print(f"results -> {OUT_RESULTS}")
    print(f"\ntotal eval runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
