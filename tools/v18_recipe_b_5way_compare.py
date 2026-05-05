"""5-way comparison of V18 + Recipe B against baselines on the v11 corrected corpus.

Configs:
  A: filterless (friend rule + single-position only)
  B: v11 ml_full_ny actual = Filter G base 0.35 (size=1) — the dominant production gate
  C: V15 stacked meta @ threshold 0.65 (size=1)
  D: V18 substitute @ threshold 0.60 (size=1)  -- KRONOS FEATURES MISSING IN CORPUS
  E: V18 substitute @ threshold 0.60 + Recipe B tiered sizing
     (proba >=0.85 size=10, 0.65-0.85 size=4, 0.60-0.65 size=1)

CAVEAT (printed prominently in output): the v11/v12 corpora do not contain
the 5 kronos_* features required by V18 inference. V18 features are
generated at runtime by the Kronos daemon subprocess (NeoQuasar/Kronos-small)
and not cached in either parquet. We use V15 proba in place of V18 proba
for D and E. This is a faithful proxy for the V15 base layer of V18 only —
it omits Kronos's marginal contribution to per-trade ranking and tier
assignment. Recipe B tier behavior in E is therefore an APPROXIMATION
of the deployed config.

NO live changes. Reads parquets and joblibs. Writes results to artifacts/.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent

V15_DE3_MODEL = REPO / "artifacts/regime_ml_meta_v15/de3/model.joblib"
KALSHI_V12_MODEL = REPO / "artifacts/regime_ml_kalshi_v12/de3/model.joblib"
V12_CORPUS = REPO / "artifacts/v12_training_corpus.parquet"


def compute_kalshi_v12_proba(df: pd.DataFrame) -> np.ndarray:
    payload = joblib.load(KALSHI_V12_MODEL)
    model = payload["model"]
    feats = payload["features"]
    X = df[feats].copy()
    # impute simple median for any nulls (k12 features have ~317 nulls when
    # k12_window inactive; default to neutral medians).
    for c in X.columns:
        if X[c].isna().any():
            med = X[c].median()
            X[c] = X[c].fillna(med if not pd.isna(med) else 0.0)
    return model.predict_proba(X.values)[:, 1]


def compute_v15_proba(df: pd.DataFrame) -> np.ndarray:
    payload = joblib.load(V15_DE3_MODEL)
    model = payload["model"]
    feats = payload["features"]
    X = df[feats].copy()
    for c in X.columns:
        if X[c].isna().any():
            med = X[c].median()
            X[c] = X[c].fillna(med if not pd.isna(med) else 0.0)
    return model.predict_proba(X.values)[:, 1]


def apply_single_position(df: pd.DataFrame, fired_mask: pd.Series) -> pd.Series:
    """Walk in time order. Among candidate rows where fired_mask is True,
    only keep rows whose entry ts is >= the previous fired trade's exit ts."""
    out = pd.Series([False] * len(df), index=df.index)
    busy_until = pd.Timestamp.min.tz_localize("UTC")
    for idx, row in df.iterrows():
        if not fired_mask.loc[idx]:
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
                if et.tz is None:
                    et = et.tz_localize("UTC")
                else:
                    et = et.tz_convert("UTC")
                busy_until = et
    return out


def recipe_b_size(proba: float) -> int:
    """tier-10: proba >=0.85; tier-4: 0.65-0.85; tier-1: 0.60-0.65; else 0."""
    if proba >= 0.85:
        return 10
    if proba >= 0.65:
        return 4
    if proba >= 0.60:
        return 1
    return 0


def run_config(df: pd.DataFrame, fired_mask: pd.Series, sizes: pd.Series, label: str) -> Dict:
    """Apply single-position then compute size-aware metrics."""
    sp = apply_single_position(df, fired_mask)
    fired = df[sp].copy()
    fired["size"] = sizes[sp].values
    fired["sized_pnl"] = fired["net_pnl_after_haircut"] * fired["size"]

    n = len(fired)
    if n == 0:
        return {
            "label": label, "trades": 0, "wins": 0, "losses": 0, "wr": 0.0,
            "pnl": 0.0, "avg_per_trade": 0.0, "max_dd": 0.0,
            "by_side": {}, "by_strategy": {}, "fired_idx": [],
        }
    wins = int((fired["sized_pnl"] > 0).sum())
    losses = int((fired["sized_pnl"] < 0).sum())
    pnl = float(fired["sized_pnl"].sum())
    wr = wins / n
    avg = pnl / n

    eq = fired.sort_values("ts")["sized_pnl"].cumsum()
    dd = float((eq - eq.cummax()).min())

    by_side = {
        side: {
            "n": int((fired["side"] == side).sum()),
            "wr": float(((fired["side"] == side) & (fired["sized_pnl"] > 0)).sum()
                        / max(1, (fired["side"] == side).sum())),
            "pnl": float(fired.loc[fired["side"] == side, "sized_pnl"].sum()),
        }
        for side in fired["side"].unique()
    }
    by_strat = {
        s: {
            "n": int((fired["strategy"] == s).sum()),
            "wr": float(((fired["strategy"] == s) & (fired["sized_pnl"] > 0)).sum()
                        / max(1, (fired["strategy"] == s).sum())),
            "pnl": float(fired.loc[fired["strategy"] == s, "sized_pnl"].sum()),
        }
        for s in fired["strategy"].unique()
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
        "fired_df": fired,
    }


def main():
    print("=" * 90)
    print("V18 + Recipe B 5-way comparison on v11 corrected corpus")
    print("=" * 90)
    print()

    df = pd.read_parquet(V12_CORPUS).sort_values("ts").reset_index(drop=True)
    print(f"corpus: {len(df)} rows, {df['ts'].min()} → {df['ts'].max()}")
    print(f"corpus filterless PnL (ALL rows): ${df['net_pnl_after_haircut'].sum():,.2f}")
    print()

    print("=== Feature availability check ===")
    kronos_cols = ["kronos_max_high_above", "kronos_min_low_below",
                   "kronos_pred_atr_30bar", "kronos_dir_move", "kronos_close_vs_entry",
                   "kronos_pred_favorable"]
    missing_kronos = [c for c in kronos_cols if c not in df.columns]
    print(f"  Kronos features in corpus: {len(kronos_cols) - len(missing_kronos)}/{len(kronos_cols)}")
    print(f"  MISSING: {missing_kronos}")
    print(f"  → Configs D and E will use V15 proba in place of V18 proba (proxy).")
    print()

    print("=== Computing kalshi_v12_proba (not cached in corpus) ===")
    df["kalshi_v12_proba"] = compute_kalshi_v12_proba(df)
    print(f"  kalshi_v12_proba: mean={df['kalshi_v12_proba'].mean():.3f} "
          f"std={df['kalshi_v12_proba'].std():.3f}")

    print("=== Computing V15 stacked proba ===")
    df["v15_proba"] = compute_v15_proba(df)
    print(f"  V15 proba: mean={df['v15_proba'].mean():.3f} "
          f"std={df['v15_proba'].std():.3f}")
    print(f"  V15 proba >=0.65: {(df['v15_proba']>=0.65).sum()} rows")
    print(f"  V15 proba >=0.60: {(df['v15_proba']>=0.60).sum()} rows")
    print(f"  V15 proba >=0.85: {(df['v15_proba']>=0.85).sum()} rows")
    print()

    # NY-only filter (08:00-16:00 ET) — matches the live ny session window.
    et = pd.to_datetime(df["ts"]).dt.tz_convert("America/New_York")
    ny_mask = (et.dt.hour >= 8) & (et.dt.hour < 16)
    print(f"NY-only filter (08:00-16:00 ET): {ny_mask.sum()} of {len(df)} rows pass")

    # Restrict to friend-allowed AND NY-only for all configs.
    fa_mask = df["allowed_by_friend_rule"].astype(bool) & ny_mask
    print(f"friend-allowed AND NY-only: {fa_mask.sum()} of {len(df)}")
    print()

    # --- Config A: filterless = friend-allowed + single-position only
    a = run_config(df, fa_mask, pd.Series([1] * len(df), index=df.index), "A_filterless")

    # --- Config B: v11 ml_full_ny actual proxy = friend + Filter G base 0.35 (block where fg >= 0.35)
    fg_pass = df["fg_proba"] < 0.35
    b_mask = fa_mask & fg_pass
    b = run_config(df, b_mask, pd.Series([1] * len(df), index=df.index), "B_filterG_base")

    # --- Config C: V15 stacked @ 0.65, size=1
    c_pass = df["v15_proba"] >= 0.65
    c_mask = fa_mask & c_pass
    c = run_config(df, c_mask, pd.Series([1] * len(df), index=df.index), "C_V15_065")

    # --- Config D: V18 substitute @ 0.60, size=1
    d_pass = df["v15_proba"] >= 0.60
    d_mask = fa_mask & d_pass
    d = run_config(df, d_mask, pd.Series([1] * len(df), index=df.index), "D_V18sub_060")

    # --- Config E: V18 substitute @ 0.60 + Recipe B sizing
    e_sizes = df["v15_proba"].apply(recipe_b_size).astype(int)
    e_mask = fa_mask & (e_sizes > 0)
    e = run_config(df, e_mask, e_sizes, "E_V18sub_RecipeB")

    print("=" * 110)
    print(f"{'config':<22} {'trades':>7} {'WR':>7} {'PnL':>13} {'avg/trade':>11} {'max DD':>12} {'wins':>6} {'losses':>7}")
    print("-" * 110)
    for s in (a, b, c, d, e):
        print(f"{s['label']:<22} {s['trades']:>7d} {s['wr']*100:>6.2f}% "
              f"${s['pnl']:>12,.2f} ${s['avg_per_trade']:>10.2f} "
              f"${s['max_dd']:>11,.2f} {s['wins']:>6d} {s['losses']:>7d}")
    print("=" * 110)
    print()

    print("=== Δ vs B (production proxy) ===")
    for s in (a, c, d, e):
        dt = s["trades"] - b["trades"]
        dpnl = s["pnl"] - b["pnl"]
        ddd = s["max_dd"] - b["max_dd"]
        dwr = (s["wr"] - b["wr"]) * 100
        print(f"  {s['label']:<22} Δtrades={dt:+5d}  Δpnl=${dpnl:+10,.2f}  ΔWR={dwr:+5.2f}pp  ΔDD=${ddd:+9,.2f}")
    print()

    # --- Recipe B tier breakdown for E
    print("=" * 90)
    print("Recipe B tier breakdown (config E)")
    print("=" * 90)
    fired_e = e["fired_df"].copy()
    fired_e["proba"] = df.loc[fired_e.index, "v15_proba"].values
    tiers = []
    for tier_name, lo, hi, sz in [("tier10 (>=0.85)", 0.85, 1.01, 10),
                                  ("tier4 (0.65-0.85)", 0.65, 0.85, 4),
                                  ("tier1 (0.60-0.65)", 0.60, 0.65, 1)]:
        mask = (fired_e["proba"] >= lo) & (fired_e["proba"] < hi)
        sub = fired_e[mask]
        n = len(sub)
        if n == 0:
            tiers.append((tier_name, sz, 0, 0.0, 0.0, 0.0))
            print(f"  {tier_name:<22} size={sz:<3} n=0")
            continue
        wins = int((sub["sized_pnl"] > 0).sum())
        wr = wins / n
        pnl_total = float(sub["sized_pnl"].sum())
        avg = pnl_total / n
        tiers.append((tier_name, sz, n, wr, pnl_total, avg))
        print(f"  {tier_name:<22} size={sz:<3} n={n:<4} wr={wr*100:>5.2f}% "
              f"pnl=${pnl_total:>10,.2f} avg/trade=${avg:>8.2f}")
    print()

    # --- 14-month lift estimate
    months = 14
    e_pnl = e["pnl"]
    b_pnl = b["pnl"]
    lift = e_pnl - b_pnl
    print(f"14-mo lift estimate (E vs B): ${lift:,.2f} = ${lift/months:.2f}/mo")
    print(f"User's projection: $32.5k/8mo = $4,062/mo. Computed proxy lift = ${lift/months:.2f}/mo.")
    print()

    # --- Save summary
    out = {
        "configs": [{k: v for k, v in s.items() if k != "fired_df"} for s in (a, b, c, d, e)],
        "deltas_vs_B": {
            s["label"]: {
                "trades": s["trades"] - b["trades"],
                "pnl": s["pnl"] - b["pnl"],
                "wr_pp": (s["wr"] - b["wr"]) * 100,
                "dd": s["max_dd"] - b["max_dd"],
            } for s in (a, c, d, e)
        },
        "recipe_b_tiers": [
            {"tier": t[0], "size": t[1], "n": t[2], "wr": t[3], "pnl": t[4], "avg": t[5]}
            for t in tiers
        ],
        "caveats": {
            "kronos_features_missing": missing_kronos,
            "kalshi_v12_proba_recomputed": True,
            "D_E_proxy": "V15 proba substituted for V18 proba (V18 needs Kronos features at inference time which are not cached in the corpus)",
        },
    }
    out_path = REPO / "artifacts" / "v18_recipe_b_5way_summary.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"summary -> {out_path}")


if __name__ == "__main__":
    main()
