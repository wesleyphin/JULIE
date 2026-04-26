"""V18 + Recipe B 5-way DEPLOYMENT-FIDELITY-FAST comparison.

Final harness used for §8.33.10 journal entry. Uses friend's pre-cached
Kronos features (the same ones V18 was trained against) and applies
deployment-realistic constraints (NY-only, single-position, time-ordered
DD, size-aware PnL) on top of friend's vectorized eval pattern.

Both 14-month (in-sample + holdout) and Jan-Apr 2026 holdout-only views
are emitted so the user can see in-sample inflation vs. honest forward
projection. The headline = HOLDOUT.

Reads:
  artifacts/v12_training_corpus.parquet
  artifacts/v18_kronos_features.parquet  (friend's 1,599 DE3+friend cache)
  artifacts/regime_ml_kalshi_v12/de3/model.joblib
  artifacts/regime_ml_meta_v15/de3/model.joblib
  artifacts/regime_ml_v18_de3/de3/model.joblib

Writes:
  artifacts/v18_5way_holdout_summary.json
  artifacts/v18_5way_holdout_per_month.csv

Methodology:
  - Friend's selection: DE3 + allowed_by_friend_rule=True (1,599 of 3,438 corpus rows)
  - Friend's split: train ts<2026-01-01, holdout ts>=2026-01-01
  - Kronos features: cached (NeoQuasar/Kronos-small, daemon mode)
  - kalshi_v12_proba: re-fired from K12 model
  - V15/V18 proba: batched predict_proba on cached features
  - Deployment realism: NY-only [08:00, 16:00) ET, single-position, time-ordered cum DD
  - Recipe B sizing: proba>=0.85→10, 0.65-0.85→4, 0.60-0.65→1
  - $7.50/trade haircut (already applied in net_pnl_after_haircut from corrected sim)

Caveats explicitly NOT modeled:
  - Regime ML (v5_brackets, v6_size, v6_be) — features not cached + bracket changes need re-sim
  - SameSide ML — features not cached + concurrency violates single-position model
  - Downstream Kalshi/LFO/PCT/Pivot overlays — V18 IS the meta over their probas
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
KRON_CACHE = REPO / "artifacts" / "v18_kronos_features.parquet"
KALSHI_V12 = REPO / "artifacts" / "regime_ml_kalshi_v12" / "de3" / "model.joblib"
V15_DE3 = REPO / "artifacts" / "regime_ml_meta_v15" / "de3" / "model.joblib"
V18_DE3 = REPO / "artifacts" / "regime_ml_v18_de3" / "de3" / "model.joblib"

OUT_SUMMARY = REPO / "artifacts" / "v18_5way_holdout_summary.json"
OUT_PER_MONTH = REPO / "artifacts" / "v18_5way_holdout_per_month.csv"

KRONOS_FEATS = ["kronos_max_high_above", "kronos_min_low_below",
                "kronos_pred_atr_30bar", "kronos_dir_move", "kronos_close_vs_entry"]


def safe_predict(payload, X):
    feats = payload["features"]
    Xs = X[feats].astype(float).fillna(0.0).values
    return payload["model"].predict_proba(Xs)[:, 1]


def single_position(ts_arr, exit_arr, fired_mask):
    out = np.zeros_like(fired_mask, dtype=bool)
    busy = -(2**62)
    for i in range(len(ts_arr)):
        if not fired_mask[i]:
            continue
        if ts_arr[i] >= busy:
            out[i] = True
            busy = exit_arr[i] if exit_arr[i] > ts_arr[i] else ts_arr[i]
    return out


def tier_size(p):
    if p >= 0.85: return 10
    if p >= 0.65: return 4
    if p >= 0.60: return 1
    return 0


def evaluate(label: str, df, ts_ns, exit_ns, gate_mask, sizes):
    fire = single_position(ts_ns, exit_ns, gate_mask)
    sub = df[fire].copy()
    sub["size"] = sizes[fire]
    sub["sized_pnl"] = sub["net_pnl_after_haircut"].values * sizes[fire]
    n = len(sub)
    if n == 0:
        return {"label": label, "trades": 0, "wr": 0, "pnl": 0, "dd": 0,
                "avg": 0, "wins": 0, "losses": 0, "sub": sub}
    wins = int((sub["sized_pnl"] > 0).sum())
    losses = int((sub["sized_pnl"] < 0).sum())
    pnl = float(sub["sized_pnl"].sum())
    eq = np.cumsum(sub["sized_pnl"].values)
    dd = float((eq - np.maximum.accumulate(eq)).min())
    by_strat = {}
    for s in sub["strategy"].unique():
        msk = sub["strategy"] == s
        ns = int(msk.sum())
        by_strat[str(s)] = {
            "n": ns,
            "wr": float(((msk) & (sub["sized_pnl"] > 0)).sum() / max(1, ns)),
            "pnl": float(sub.loc[msk, "sized_pnl"].sum()),
        }
    by_side = {}
    for sd in sub["side"].unique():
        msk = sub["side"] == sd
        ns = int(msk.sum())
        by_side[str(sd)] = {
            "n": ns,
            "wr": float(((msk) & (sub["sized_pnl"] > 0)).sum() / max(1, ns)),
            "pnl": float(sub.loc[msk, "sized_pnl"].sum()),
        }
    return {
        "label": label, "trades": n, "wins": wins, "losses": losses,
        "wr": wins/n, "pnl": pnl, "avg": pnl/n, "dd": dd,
        "by_strategy": by_strat, "by_side": by_side, "sub": sub,
    }


def main():
    t0 = time.time()
    print("=" * 90)
    print("V18 + Recipe B 5-way deployment-fidelity-fast comparison")
    print("=" * 90)

    corpus = pd.read_parquet(V12_CORPUS).sort_values("ts").reset_index(drop=True)
    kron = pd.read_parquet(KRON_CACHE)
    print(f"corpus: {len(corpus)} rows, kronos cache: {len(kron)} rows (DE3+friend)")

    # Friend's selection
    de3 = corpus[(corpus["family"] == "de3") &
                  (corpus["allowed_by_friend_rule"] == True)].copy()
    de3 = de3.reset_index(drop=True)
    de3["row_idx"] = de3.index
    m = de3.merge(kron[["row_idx"] + KRONOS_FEATS], on="row_idx", how="inner")
    m = m.sort_values("ts").reset_index(drop=True)
    print(f"DE3+friend with kronos: {len(m)} rows")

    # Models
    print("\nfitting / loading meta-learner inputs ...")
    k12 = joblib.load(KALSHI_V12)
    m["kalshi_v12_proba"] = safe_predict(k12, m)
    v15 = joblib.load(V15_DE3)
    m["v15_proba"] = safe_predict(v15, m)
    v18 = joblib.load(V18_DE3)
    m["v18_proba"] = safe_predict(v18, m)
    print(f"  v18 distribution: mean={m['v18_proba'].mean():.3f}  "
          f">=0.60={(m['v18_proba']>=0.60).sum()}  "
          f">=0.65={(m['v18_proba']>=0.65).sum()}  "
          f">=0.85={(m['v18_proba']>=0.85).sum()}")

    # Constraints
    et = pd.to_datetime(m["ts"]).dt.tz_convert("America/New_York")
    ny_mask = ((et.dt.hour >= 8) & (et.dt.hour < 16)).values
    ts = pd.to_datetime(m["ts"])
    ts_ns = ts.dt.tz_convert("UTC").dt.tz_localize(None).astype("datetime64[ns]").astype("int64").values
    exit_ns = pd.to_datetime(m["exit_ts"]).dt.tz_convert("UTC").dt.tz_localize(None).astype("datetime64[ns]").astype("int64").values

    holdout_mask = (ts >= pd.Timestamp("2026-01-01", tz="US/Eastern")).values
    months_holdout = (ts[holdout_mask].max() - ts[holdout_mask].min()).total_seconds() / (86400 * 30.44)

    one = np.ones(len(m), dtype=int)
    sizes_e = np.array([tier_size(p) for p in m["v18_proba"].values], dtype=int)

    results = {"in_sample_warning": "Configs C/D/E include train data Mar 2025-Dec 2025 — V18 has seen these labels",
               "holdout_months": months_holdout, "holdout_rows": int(holdout_mask.sum()),
               "v15_in_corpus": True, "v18_in_corpus": True,
               "kronos_imputed": int(kron["kronos_failed"].notna().sum()) if "kronos_failed" in kron.columns else 0,
               "windows": {}}

    for window_label, mask in [("FULL_14mo_in_sample_inflated", ny_mask),
                                 ("HOLDOUT_2026_Q1_honest", ny_mask & holdout_mask)]:
        print(f"\n=== {window_label} ===")
        a = evaluate("A_filterless", m, ts_ns, exit_ns, mask, one)
        b = evaluate("B_FilterG_base_0.35", m, ts_ns, exit_ns,
                     mask & (m["fg_proba"].values < 0.35), one)
        c = evaluate("C_V15_thr_0.65", m, ts_ns, exit_ns,
                     mask & (m["v15_proba"].values >= 0.65), one)
        d = evaluate("D_V18_thr_0.60_flat", m, ts_ns, exit_ns,
                     mask & (m["v18_proba"].values >= 0.60), one)
        e = evaluate("E_V18_RecipeB_DEPLOYED", m, ts_ns, exit_ns,
                     mask & (sizes_e > 0), sizes_e)

        configs = [a, b, c, d, e]
        print(f"{'config':<28} {'trades':>7} {'WR':>7} {'PnL':>13} {'avg':>9} {'DD':>11}")
        print("-"*88)
        for s in configs:
            print(f"{s['label']:<28} {s['trades']:>7d} {s['wr']*100:>6.2f}% "
                  f"${s['pnl']:>12,.2f} ${s['avg']:>8.2f} ${s['dd']:>10,.2f}")

        # Recipe B tier breakdown for E
        fired_e = e["sub"].copy()
        fired_e["proba"] = m.loc[fired_e.index, "v18_proba"].values
        tier_breakdown = []
        for tname, lo, hi, sz in [("tier10>=0.85", 0.85, 1.01, 10),
                                  ("tier4(0.65-0.85)", 0.65, 0.85, 4),
                                  ("tier1(0.60-0.65)", 0.60, 0.65, 1)]:
            mm = (fired_e["proba"] >= lo) & (fired_e["proba"] < hi)
            sub = fired_e[mm]
            n = len(sub)
            if n == 0:
                tier_breakdown.append({"tier": tname, "size": sz, "n": 0,
                                       "wr": 0, "pnl": 0, "dd": 0})
                continue
            wins = int((sub["sized_pnl"] > 0).sum())
            eq_t = np.cumsum(sub.sort_values("ts")["sized_pnl"].values)
            dd_t = float((eq_t - np.maximum.accumulate(eq_t)).min())
            tier_breakdown.append({
                "tier": tname, "size": sz, "n": n, "wr": wins/n,
                "pnl": float(sub["sized_pnl"].sum()), "dd": dd_t,
                "avg": float(sub["sized_pnl"].sum() / n),
            })

        print(f"  Recipe B tiers (E):")
        for t in tier_breakdown:
            print(f"    {t['tier']:<22} size={t['size']:>2} n={t['n']:>4} "
                  f"wr={t['wr']*100:>5.1f}% pnl=${t['pnl']:>10,.2f} dd=${t['dd']:>9,.2f}")

        # per-month
        fired_e["month"] = pd.to_datetime(fired_e["ts"]).dt.tz_convert(
            "America/New_York").dt.strftime("%Y-%m")
        g = fired_e.groupby("month").agg(
            trades=("ts", "count"),
            pnl=("sized_pnl", "sum"),
            wr=("sized_pnl", lambda s: float((s > 0).sum() / max(1, len(s)))),
            max_size=("size", "max"),
        )
        dd_per_mo = []
        for mo in g.index:
            eq = np.cumsum(fired_e[fired_e["month"] == mo].sort_values("ts")["sized_pnl"].values)
            dd_per_mo.append(float((eq - np.maximum.accumulate(eq)).min()))
        g["dd"] = dd_per_mo
        if window_label.startswith("HOLDOUT"):
            print(f"\n  E per-month (holdout):")
            print(g.to_string().replace("\n", "\n  "))
            g.to_csv(OUT_PER_MONTH)

        results["windows"][window_label] = {
            "configs": [{k: v for k, v in s.items() if k != "sub"} for s in configs],
            "deltas_vs_B": {
                s["label"]: {
                    "trades": s["trades"] - b["trades"],
                    "pnl": s["pnl"] - b["pnl"],
                    "wr_pp": (s["wr"] - b["wr"]) * 100,
                    "dd": s["dd"] - b["dd"],
                } for s in (a, c, d, e)
            },
            "recipe_b_tiers": tier_breakdown,
        }

    # Friend's eval pattern (no single-pos, no NY) on holdout for delta
    print("\n=== friend's eval pattern on HOLDOUT (no single-pos, no NY) ===")
    holdout_idx = np.where(holdout_mask)[0]
    sub_friend_idx = holdout_idx[sizes_e[holdout_idx] > 0]
    sub_friend = m.loc[sub_friend_idx].copy()
    sub_friend["size"] = sizes_e[sub_friend_idx]
    sub_friend["sized_pnl"] = (sub_friend["net_pnl_after_haircut"].values *
                                sizes_e[sub_friend_idx])
    n_f = len(sub_friend)
    wins_f = int((sub_friend["sized_pnl"] > 0).sum())
    pnl_f = float(sub_friend["sized_pnl"].sum())
    eq_f = np.cumsum(sub_friend.sort_values("ts")["sized_pnl"].values)
    dd_f = float((eq_f - np.maximum.accumulate(eq_f)).min())
    print(f"  trades={n_f}  WR={wins_f/n_f*100:.2f}%  PnL=${pnl_f:,.2f}  DD=${dd_f:,.2f}")
    print(f"  scaled to 8mo: ${pnl_f / months_holdout * 8:,.2f}")

    e_holdout = results["windows"]["HOLDOUT_2026_Q1_honest"]["configs"][-1]
    print(f"\nΔ deployment-fidelity (E) vs friend's eval pattern:")
    print(f"  Δtrades = {e_holdout['trades'] - n_f:+d}")
    print(f"  ΔPnL    = ${e_holdout['pnl'] - pnl_f:+,.2f}")
    print(f"  ΔDD     = ${e_holdout['dd'] - dd_f:+,.2f}")
    constraint_cost_pct = (pnl_f - e_holdout["pnl"]) / max(1.0, pnl_f) * 100
    print(f"  Constraint cost: ${pnl_f - e_holdout['pnl']:.2f} "
          f"({constraint_cost_pct:.2f}% of friend's projection)")

    results["friend_eval_pattern_holdout"] = {
        "trades": n_f, "wr": wins_f/n_f, "pnl": pnl_f, "dd": dd_f,
        "scaled_8mo": pnl_f / months_holdout * 8,
    }
    results["constraint_cost"] = {
        "delta_pnl": e_holdout["pnl"] - pnl_f,
        "delta_dd": e_holdout["dd"] - dd_f,
        "delta_trades": e_holdout["trades"] - n_f,
        "pct_of_friend_projection": constraint_cost_pct,
    }
    results["projection"] = {
        "E_holdout_pnl": e_holdout["pnl"],
        "E_per_month": e_holdout["pnl"] / months_holdout,
        "E_scaled_to_8mo": e_holdout["pnl"] / months_holdout * 8,
        "user_projection_8mo": 32500,
        "ratio_to_user_projection": e_holdout["pnl"] / months_holdout * 8 / 32500,
    }

    OUT_SUMMARY.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nsummary -> {OUT_SUMMARY}")
    print(f"per-month -> {OUT_PER_MONTH}")
    print(f"runtime: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
