"""5-way V18 + Recipe B comparison on the v11 corpus with REAL Kronos features.

Runs after tools/kronos_batch_extract.py has populated
artifacts/v11_corpus_with_kronos_features.parquet (5 features per row).

Configs:
  A: filterless = friend rule + single-position (the canonical baseline)
  B: v11 ml_full_ny actual ≈ A + Filter G base 0.35 (production gate; the
     dominant blocker per §8.25-§8.26 — Kalshi/LFO/PCT/Pivot are downstream
     filters but FG carries the load)
  C: V15 stacked meta @ threshold 0.65 (size=1)
  D: V18 (V15's 6 base probas + 5 Kronos features) @ threshold 0.60 (size=1)
  E: V18 + Recipe B tiered sizing — THE DEPLOYED STATE
     (proba >=0.85 → size=10, 0.65–0.85 → size=4, 0.60–0.65 → size=1)

For each: trades, WR, net PnL (size-aware for E), avg/trade, max DD,
by-side, by-strategy, per-month split.

Uses:
  - artifacts/v12_training_corpus.parquet for the 6 base + k12 features +
    upstream-precomputed kalshi_v12_proba target features
  - artifacts/regime_ml_kalshi_v12/de3/model.joblib for kalshi_v12_proba
  - artifacts/regime_ml_meta_v15/de3/model.joblib for V15 stacked
  - artifacts/regime_ml_v18_de3/de3/model.joblib for V18 stacked
  - artifacts/v11_corpus_with_kronos_features.parquet for real Kronos features

Runs friend's same-side rule + single-position simulation against precomputed
net_pnl_after_haircut (from the corrected simulator_trade_through.py).

Writes:
  - artifacts/v18_5way_real_kronos_summary.json
  - artifacts/v18_5way_real_kronos_per_month.csv
  - artifacts/v18_5way_real_kronos_results.csv
"""
from __future__ import annotations

import json
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

OUT_SUMMARY = REPO / "artifacts" / "v18_5way_real_kronos_summary.json"
OUT_PER_MONTH = REPO / "artifacts" / "v18_5way_real_kronos_per_month.csv"
OUT_RESULTS = REPO / "artifacts" / "v18_5way_real_kronos_results.csv"


def safe_predict(payload: Dict, X_df: pd.DataFrame) -> np.ndarray:
    feats = payload["features"]
    Xs = X_df[feats].copy()
    for c in Xs.columns:
        if Xs[c].isna().any():
            med = Xs[c].median()
            Xs[c] = Xs[c].fillna(med if not pd.isna(med) else 0.0)
    return payload["model"].predict_proba(Xs.values)[:, 1]


def apply_single_position(df: pd.DataFrame, fired_mask: pd.Series) -> pd.Series:
    """Walk in time order; respect single-position constraint."""
    out = pd.Series(False, index=df.index)
    busy_until = pd.Timestamp.min.tz_localize("UTC")
    for idx, row in df.iterrows():
        if not bool(fired_mask.loc[idx]):
            continue
        t = pd.to_datetime(row["ts"])
        if t.tz is None:
            t = t.tz_localize("UTC")
        else:
            t = t.tz_convert("UTC")
        if t >= busy_until:
            out.loc[idx] = True
            et = pd.to_datetime(row["exit_ts"])
            if pd.isna(et):
                busy_until = t
            else:
                busy_until = et.tz_localize("UTC") if et.tz is None else et.tz_convert("UTC")
    return out


def recipe_b_size(proba: float) -> int:
    if proba >= 0.85:
        return 10
    if proba >= 0.65:
        return 4
    if proba >= 0.60:
        return 1
    return 0


def evaluate(df: pd.DataFrame, fired_mask: pd.Series, sizes: pd.Series, label: str) -> Dict:
    sp = apply_single_position(df, fired_mask)
    f = df[sp].copy()
    f["size"] = sizes[sp].values
    f["sized_pnl"] = f["net_pnl_after_haircut"] * f["size"]
    n = len(f)
    if n == 0:
        return {"label": label, "trades": 0, "wins": 0, "losses": 0, "wr": 0.0,
                "pnl": 0.0, "avg_per_trade": 0.0, "max_dd": 0.0,
                "by_side": {}, "by_strategy": {},
                "fired_idx": []}
    wins = int((f["sized_pnl"] > 0).sum())
    losses = int((f["sized_pnl"] < 0).sum())
    pnl = float(f["sized_pnl"].sum())
    wr = wins / n
    avg = pnl / n
    eq = f.sort_values("ts")["sized_pnl"].cumsum()
    dd = float((eq - eq.cummax()).min()) if len(eq) else 0.0
    by_side = {
        side: {
            "n": int((f["side"] == side).sum()),
            "wr": float(((f["side"] == side) & (f["sized_pnl"] > 0)).sum() /
                        max(1, (f["side"] == side).sum())),
            "pnl": float(f.loc[f["side"] == side, "sized_pnl"].sum()),
        } for side in f["side"].unique()
    }
    by_strat = {
        s: {
            "n": int((f["strategy"] == s).sum()),
            "wr": float(((f["strategy"] == s) & (f["sized_pnl"] > 0)).sum() /
                        max(1, (f["strategy"] == s).sum())),
            "pnl": float(f.loc[f["strategy"] == s, "sized_pnl"].sum()),
        } for s in f["strategy"].unique()
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
        "fired_df": f,
    }


def per_month(fired_df: pd.DataFrame) -> pd.DataFrame:
    f = fired_df.copy()
    f["month"] = pd.to_datetime(f["ts"]).dt.tz_convert("America/New_York").dt.strftime("%Y-%m")
    g = f.groupby("month").agg(
        trades=("ts", "count"),
        wins=("sized_pnl", lambda s: (s > 0).sum()),
        wr=("sized_pnl", lambda s: (s > 0).sum() / max(1, len(s))),
        pnl=("sized_pnl", "sum"),
    )
    return g


def main():
    print("=" * 90)
    print("V18 + Recipe B 5-way comparison (REAL Kronos features)")
    print("=" * 90)

    # 1) load
    df = pd.read_parquet(V12_CORPUS).sort_values("ts").reset_index(drop=True)
    print(f"corpus: {len(df)} rows  {df['ts'].min()} → {df['ts'].max()}")

    kron = pd.read_parquet(KRONOS_FEATS)
    print(f"kronos features: {len(kron)} rows  failures={kron['kronos_failed'].notna().sum()}")

    # Merge kronos by row_idx (kronos was generated against ts-sorted v11 corpus
    # which has the same row order as ts-sorted v12 corpus since both have 3438
    # rows and same `ts` values).
    # Verify alignment via ts merge.
    if "row_idx" in kron.columns and len(kron) == len(df):
        kron_sorted = kron.sort_values("row_idx").reset_index(drop=True)
        # keep just the kronos cols
        kron_cols = ["kronos_max_high_above", "kronos_min_low_below",
                     "kronos_pred_atr_30bar", "kronos_dir_move", "kronos_close_vs_entry",
                     "kronos_failed"]
        for c in kron_cols:
            df[c] = kron_sorted[c].values
    else:
        # Fall back to ts-merge
        kron["ts"] = pd.to_datetime(kron["ts"])
        df["ts_merge"] = pd.to_datetime(df["ts"])
        df = df.merge(kron[["ts"] + ["kronos_max_high_above", "kronos_min_low_below",
                                       "kronos_pred_atr_30bar", "kronos_dir_move",
                                       "kronos_close_vs_entry", "kronos_failed"]],
                       left_on="ts_merge", right_on="ts", how="left", suffixes=("", "_k"))
        df = df.drop(columns=["ts_merge"])

    # Impute kronos features for failed rows with col-medians (the V18 model
    # treats absent kronos as "neutral" via fallback in production)
    for c in ["kronos_max_high_above", "kronos_min_low_below", "kronos_pred_atr_30bar",
              "kronos_dir_move", "kronos_close_vs_entry"]:
        if df[c].isna().any():
            med = df[c].median()
            df[c] = df[c].fillna(med if not pd.isna(med) else 0.0)
            print(f"  imputed {c} with median {med:.3f}")

    # 2) compute kalshi_v12_proba (V15/V18 require it; not in corpus)
    print("computing kalshi_v12_proba ...")
    k12_payload = joblib.load(KALSHI_V12)
    df["kalshi_v12_proba"] = safe_predict(k12_payload, df)

    # 3) compute V15 + V18 probas
    print("computing v15_proba ...")
    v15_payload = joblib.load(V15_DE3)
    df["v15_proba"] = safe_predict(v15_payload, df)

    print("computing v18_proba ...")
    v18_payload = joblib.load(V18_DE3)
    df["v18_proba"] = safe_predict(v18_payload, df)

    print(f"  v15 mean={df['v15_proba'].mean():.3f}  >=0.65: {(df['v15_proba']>=0.65).sum()} >=0.85: {(df['v15_proba']>=0.85).sum()}")
    print(f"  v18 mean={df['v18_proba'].mean():.3f}  >=0.60: {(df['v18_proba']>=0.60).sum()} >=0.65: {(df['v18_proba']>=0.65).sum()} >=0.85: {(df['v18_proba']>=0.85).sum()}")

    # 4) NY-only filter (08:00-16:00 ET); friend rule already encoded in `allowed_by_friend_rule`
    et = pd.to_datetime(df["ts"]).dt.tz_convert("America/New_York")
    ny_mask = (et.dt.hour >= 8) & (et.dt.hour < 16)
    fa_mask = df["allowed_by_friend_rule"].astype(bool) & ny_mask
    print(f"NY-only ∩ friend-allowed: {fa_mask.sum()} of {len(df)}")

    # 5) Run 5 configs
    one_size = pd.Series([1] * len(df), index=df.index)

    # A: filterless
    a = evaluate(df, fa_mask, one_size, "A_filterless")

    # B: production proxy = friend ∩ NY ∩ (FG base 0.35: fire if fg_proba < 0.35)
    fg_pass = df["fg_proba"] < 0.35
    b = evaluate(df, fa_mask & fg_pass, one_size, "B_filterG_base")

    # C: V15 stacked @ 0.65
    c_pass = df["v15_proba"] >= 0.65
    c = evaluate(df, fa_mask & c_pass, one_size, "C_V15_065")

    # D: V18 @ 0.60, size=1
    d_pass = df["v18_proba"] >= 0.60
    d = evaluate(df, fa_mask & d_pass, one_size, "D_V18_060_flat")

    # E: V18 + Recipe B tiered sizing
    e_sizes = df["v18_proba"].apply(recipe_b_size).astype(int)
    e_mask = fa_mask & (e_sizes > 0)
    e = evaluate(df, e_mask, e_sizes, "E_V18_RecipeB_DEPLOYED")

    # 6) Print headline
    summaries = [a, b, c, d, e]
    print("\n" + "=" * 120)
    print(f"{'config':<28} {'trades':>7} {'WR':>7} {'PnL':>13} {'avg/trade':>11} {'max DD':>12} {'wins':>6} {'losses':>7}")
    print("-" * 120)
    for s in summaries:
        print(f"{s['label']:<28} {s['trades']:>7d} {s['wr']*100:>6.2f}% "
              f"${s['pnl']:>12,.2f} ${s['avg_per_trade']:>10.2f} ${s['max_dd']:>11,.2f} "
              f"{s['wins']:>6d} {s['losses']:>7d}")
    print("=" * 120)

    # 7) Δ vs B (production)
    print(f"\nΔ vs B (production proxy):")
    for s in (a, c, d, e):
        dt = s["trades"] - b["trades"]
        dpnl = s["pnl"] - b["pnl"]
        ddd = s["max_dd"] - b["max_dd"]
        dwr = (s["wr"] - b["wr"]) * 100
        print(f"  {s['label']:<28} Δtrades={dt:+5d}  Δpnl=${dpnl:+10,.2f}  ΔWR={dwr:+5.2f}pp  ΔDD=${ddd:+9,.2f}")

    # 8) Recipe B tier breakdown for E
    print("\n" + "=" * 80)
    print("Recipe B tier breakdown (config E)")
    print("=" * 80)
    fired_e = e["fired_df"].copy()
    fired_e["proba"] = df.loc[fired_e.index, "v18_proba"].values
    tier_rows = []
    for tier_name, lo, hi, sz in [("tier10 (>=0.85)", 0.85, 1.01, 10),
                                  ("tier4 (0.65-0.85)", 0.65, 0.85, 4),
                                  ("tier1 (0.60-0.65)", 0.60, 0.65, 1)]:
        mask = (fired_e["proba"] >= lo) & (fired_e["proba"] < hi)
        sub = fired_e[mask]
        n = len(sub)
        if n == 0:
            print(f"  {tier_name:<22} size={sz:<3} n=0")
            tier_rows.append({"tier": tier_name, "size": sz, "n": 0,
                              "wr": 0.0, "pnl": 0.0, "avg": 0.0})
            continue
        wins = int((sub["sized_pnl"] > 0).sum())
        wr = wins / n
        pnl_total = float(sub["sized_pnl"].sum())
        avg = pnl_total / n
        print(f"  {tier_name:<22} size={sz:<3} n={n:<4} wr={wr*100:>5.2f}%  "
              f"pnl=${pnl_total:>10,.2f}  avg/trade=${avg:>8.2f}")
        tier_rows.append({"tier": tier_name, "size": sz, "n": n, "wr": wr,
                          "pnl": pnl_total, "avg": avg})

    # 9) Per-month for E
    print("\n" + "=" * 90)
    print("Per-month (config E only)")
    print("=" * 90)
    pm_e = per_month(e["fired_df"])
    print(pm_e.to_string())

    # 10) 14-month projection vs $32.5k/8mo
    print("\n" + "=" * 90)
    months = 14
    e_pnl_per_mo = e["pnl"] / months
    user_proj_per_mo = 32500 / 8
    print(f"E PnL: ${e['pnl']:,.2f} over {months} months = ${e_pnl_per_mo:.2f}/mo")
    print(f"User projection: $32,500 over 8 months = ${user_proj_per_mo:.2f}/mo")
    print(f"Gap: simulated {e_pnl_per_mo/user_proj_per_mo*100:+.1f}% of user projection")

    # 11) Save artifacts
    OUT_SUMMARY.write_text(json.dumps({
        "configs": [{k: v for k, v in s.items() if k != "fired_df"} for s in summaries],
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
            "E_pnl_total": e["pnl"],
            "E_per_month": e_pnl_per_mo,
            "user_projected_per_month": user_proj_per_mo,
            "ratio_to_user_projection": e_pnl_per_mo / user_proj_per_mo,
        },
        "v18_proba_distribution": {
            "mean": float(df["v18_proba"].mean()),
            "std": float(df["v18_proba"].std()),
            "ge_060": int((df["v18_proba"] >= 0.60).sum()),
            "ge_065": int((df["v18_proba"] >= 0.65).sum()),
            "ge_085": int((df["v18_proba"] >= 0.85).sum()),
        },
        "kronos_imputed_count": int(kron['kronos_failed'].notna().sum()),
    }, indent=2, default=str))
    print(f"\nsummary -> {OUT_SUMMARY}")

    pm_e.to_csv(OUT_PER_MONTH)
    print(f"per-month -> {OUT_PER_MONTH}")

    # save annotated decisions for E
    out_cols = ["ts", "strategy", "side", "v15_proba", "v18_proba", "fg_proba",
                "kalshi_proba", "kalshi_v12_proba", "lfo_proba", "pct_proba",
                "pivot_proba", "net_pnl_after_haircut",
                "kronos_max_high_above", "kronos_min_low_below",
                "kronos_pred_atr_30bar", "kronos_dir_move", "kronos_close_vs_entry"]
    df[out_cols].to_csv(OUT_RESULTS, index=False)
    print(f"results -> {OUT_RESULTS}")


if __name__ == "__main__":
    main()
