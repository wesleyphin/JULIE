"""Phase 3e — Combined V9 Stack v2.

Re-runs Section 8.23 (combined v9 stack simulation) using:
  - Corrected mistake corpus  : artifacts/full_overlay_stack_simulation_14mo_v2.parquet
  - Corrected filterless base : artifacts/filterless_reconstruction_14mo_v2.parquet
  - V9_v2 heads (FG/K/LFO/PCT) trained inline using the same prep as Phase 3d
    (tools/run_v9_retrain_v2.py)
  - Holdout window            : Jan-Apr 2026 (ts >= 2026-01-01)
  - Friend-rule single-position simulator (raw_pnl_walk - $7.50 haircut already in
    raw_net column; friend-rule applied here)

Sweeps a 5x5x5x5 = 625-config grid over per-head BLOCK thresholds:
  thresholds = {0.40, 0.50, 0.60, 0.70, 0.80} per head.

Per combo:
  candidate is BLOCKED if ANY of FG/K/LFO/PCT predicts BLOCK at its threshold
  (sequential pipeline ordering only matters for training scope; for application
  we union the BLOCK decisions because that is the strictest combined gate).

Also evaluates the 19 named-config sweep from §8.23 (or as close as we can
reproduce — the original config table is shown in the journal).

Outputs:
  artifacts/combined_v9_v2_simulation_holdout.parquet  (best-Pareto config rows)
  artifacts/combined_v9_stack_v2_summary.json          (full summary)
"""
from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "artifacts" / "full_overlay_stack_simulation_14mo_v2.parquet"
BASELINE = ROOT / "artifacts" / "filterless_reconstruction_14mo_v2.parquet"
ARTIFACTS = ROOT / "artifacts"
HOLDOUT_CUTOFF = pd.Timestamp("2026-01-01", tz="US/Eastern")
HAIRCUT = 7.50
BIG_LOSS_THRESHOLD = -50.0

GATE1_DD_MAX = 870.0
GATE3_NOOS_MIN = 50
GATE4_WR_MIN = 0.55

GRID = [0.40, 0.50, 0.60, 0.70, 0.80]


# ---------------------------------------------------------------------------
# Data loading (mirrors run_v9_retrain_v2.py)
# ---------------------------------------------------------------------------

def load_corpus():
    df = pd.read_parquet(CORPUS)
    df = df[df["family"] == "de3"].copy()
    df["ts"] = pd.to_datetime(df["ts"])
    df["is_big_loss"] = (df["pnl_baseline"] <= BIG_LOSS_THRESHOLD).astype(int)
    df["raw_net"] = df["raw_pnl_walk"] - HAIRCUT
    return df.sort_values("ts").reset_index(drop=True)


def load_holdout_baseline():
    bl = pd.read_parquet(BASELINE)
    holdout = bl[bl["ts"] >= HOLDOUT_CUTOFF].copy()
    taken = holdout[holdout["exit_reason"] != "skipped_friend_rule"].copy()
    taken = taken.sort_values("ts").reset_index(drop=True)
    pnl = float(taken["net_pnl_after_haircut"].sum())
    wr = float((taken["net_pnl_after_haircut"] > 0).mean())
    cum = taken["net_pnl_after_haircut"].cumsum()
    dd = float((cum - cum.cummax()).min())
    return {
        "trades": int(len(taken)),
        "pnl": pnl,
        "wr": wr,
        "dd": dd,
    }


# ---------------------------------------------------------------------------
# Feature builders (mirrors run_v9_retrain_v2.py)
# ---------------------------------------------------------------------------

def base_features(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    f["price"] = df["price"].values
    f["sl_pts"] = df["sl"].values
    f["tp_pts"] = df["tp"].values
    f["rr"] = df["tp"].values / df["sl"].values
    f["side_long"] = (df["side"].str.upper() == "LONG").astype(int)
    ts = df["ts"]
    f["hour"] = ts.dt.hour.values
    f["minute"] = ts.dt.minute.values
    f["dow"] = ts.dt.dayofweek.values
    f["month"] = ts.dt.month.values
    f["day"] = ts.dt.day.values
    f["minutes_since_open"] = ((ts.dt.hour - 9) * 60 + ts.dt.minute - 30).values
    f["pct_size_mult"] = df["pct_size_mult"].values
    f["pct_tp_mult"] = df["pct_tp_mult"].values
    return f


def features_for_head(df: pd.DataFrame, head: str) -> pd.DataFrame:
    f = base_features(df)
    if head in ("kalshi", "lfo", "pct"):
        f["fg_proba"] = df["fg_proba"].fillna(0.5).values
    if head in ("lfo", "pct"):
        kp = df["k_proba"].copy()
        f["k_proba"] = kp.fillna(0.5).values
        f["k_has_market"] = kp.notna().astype(int).values
    if head == "pct":
        f["lfo_proba"] = df["lfo_proba"].fillna(0.5).values
    if head != "pct":
        for cat in ("BREAKOUT_LEAN", "PIVOT_LEAN", "NOT_AT_LEVEL", "NEUTRAL"):
            f[f"pct_dec_{cat}"] = (df["pct_decision"] == cat).astype(int).values
    return f


# ---------------------------------------------------------------------------
# Friend-rule simulator over surviving rows
# ---------------------------------------------------------------------------

def friend_rule_pnl(taken_signals: pd.DataFrame) -> tuple[int, float, float, float]:
    if len(taken_signals) == 0:
        return 0, 0.0, 0.0, 0.0
    ts_sorted = taken_signals.sort_values("ts").reset_index(drop=True)
    accepted_idx = []
    last_exit = None
    for i, row in ts_sorted.iterrows():
        if last_exit is None or row["ts"] >= last_exit:
            accepted_idx.append(i)
            last_exit = row["exit_ts"]
    accepted = ts_sorted.iloc[accepted_idx].copy()
    n = len(accepted)
    if n == 0:
        return 0, 0.0, 0.0, 0.0
    pnl = float(accepted["raw_net"].sum())
    wr = float((accepted["raw_net"] > 0).mean())
    cum = accepted["raw_net"].cumsum()
    dd = float((cum - cum.cummax()).min())
    return n, pnl, wr, dd


def evaluate_gates(row: dict, hb: dict) -> dict:
    g1 = abs(row["dd"]) <= GATE1_DD_MAX
    g2_pnl = row["pnl"] >= hb["pnl"]
    g2_trades = row["trades"] <= hb["trades"]
    g2 = g2_pnl and g2_trades
    g3 = row["n_oos_pre_friend"] >= GATE3_NOOS_MIN
    g4 = row["wr"] >= GATE4_WR_MIN
    binding = []
    if not g1: binding.append(f"G1(DD>{GATE1_DD_MAX:.0f})")
    if not g2: binding.append(f"G2(PnL<{hb['pnl']:.0f}|trades>{hb['trades']})")
    if not g3: binding.append(f"G3(nOOS<{GATE3_NOOS_MIN})")
    if not g4: binding.append(f"G4(WR<{GATE4_WR_MIN:.2f})")
    return {
        "G1": g1, "G2": g2, "G3": g3, "G4": g4,
        "all_pass": g1 and g2 and g3 and g4,
        "binding": "+".join(binding) if binding else "ALL",
    }


# ---------------------------------------------------------------------------
# Train one head -> return calibrated test probas (sized to len(test))
# ---------------------------------------------------------------------------

def train_head_proba(
    head: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[np.ndarray, float | None]:
    """Train HGB head on full train (no upstream block intersection — we want
    each head's proba on every test row, then we union BLOCK decisions in the
    combined sweep)."""
    Xtr = features_for_head(train, head).values
    ytr = train["is_big_loss"].values
    Xte = features_for_head(test, head).values
    yte = test["is_big_loss"].values

    model = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.05,
        max_depth=4,
        min_samples_leaf=20,
        l2_regularization=0.1,
        random_state=42,
    )
    model.fit(Xtr, ytr)
    proba = model.predict_proba(Xte)[:, 1]
    auc = None
    try:
        if len(np.unique(yte)) >= 2:
            auc = float(roc_auc_score(yte, proba))
    except Exception:
        pass
    return proba, auc


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def main():
    print(f"[load] corpus {CORPUS.name}")
    de3 = load_corpus()
    print(f"  DE3 rows: {len(de3)}, big_loss rate: {de3['is_big_loss'].mean():.3f}")

    train = de3[de3["ts"] < HOLDOUT_CUTOFF].copy().reset_index(drop=True)
    test = de3[de3["ts"] >= HOLDOUT_CUTOFF].copy().reset_index(drop=True)
    print(f"  Train: {len(train)} rows ({train['is_big_loss'].sum()} pos)")
    print(f"  Test : {len(test)} rows ({test['is_big_loss'].sum()} pos)")

    hb = load_holdout_baseline()
    print(f"  Holdout baseline: {hb}")

    # Step 1 — Train all 4 heads on full train, get test probas
    head_probas: dict[str, np.ndarray] = {}
    head_aucs: dict[str, float | None] = {}
    for head in ("filter_g", "kalshi", "lfo", "pct"):
        print(f"[train] head={head}")
        proba, auc = train_head_proba(head, train, test)
        head_probas[head] = proba
        head_aucs[head] = auc
        print(f"  test_auc={auc}")

    # Step 2 — Sweep 5^4 = 625 threshold grid, evaluate each combo
    print(f"[sweep] grid: {GRID} per head -> {len(GRID)**4} combos")
    rows = []
    test_indexed = test.reset_index(drop=True)
    for fg, kk, lf, pp in product(GRID, GRID, GRID, GRID):
        block = (
            (head_probas["filter_g"] >= fg)
            | (head_probas["kalshi"] >= kk)
            | (head_probas["lfo"] >= lf)
            | (head_probas["pct"] >= pp)
        )
        survivors_mask = ~block
        n_oos_pre_friend = int(survivors_mask.sum())
        if n_oos_pre_friend == 0:
            rows.append({
                "fg": fg, "k": kk, "lfo": lf, "pct": pp,
                "trades": 0, "wr": 0.0, "pnl": 0.0, "dd": 0.0,
                "n_oos_pre_friend": 0,
            })
            continue
        survivors = test_indexed[survivors_mask]
        n, pnl, wr, dd = friend_rule_pnl(survivors)
        rows.append({
            "fg": fg, "k": kk, "lfo": lf, "pct": pp,
            "trades": n, "wr": wr, "pnl": pnl, "dd": dd,
            "n_oos_pre_friend": n_oos_pre_friend,
        })
    df = pd.DataFrame(rows)
    df["abs_dd"] = df["dd"].abs()

    gate_rows = []
    for _, r in df.iterrows():
        g = evaluate_gates(r.to_dict(), hb)
        gate_rows.append(g)
    gate_df = pd.DataFrame(gate_rows)
    df = pd.concat([df.reset_index(drop=True), gate_df.reset_index(drop=True)], axis=1)
    df["passing"] = df[["G1", "G2", "G3", "G4"]].sum(axis=1)
    df["gates"] = df.apply(
        lambda r: f"{int(r['G1'])}{int(r['G2'])}{int(r['G3'])}{int(r['G4'])}", axis=1
    )

    # ----- Named-configs (subset of §8.23) -----
    named = {
        "all_loose":              {"fg": 0.40, "k": 0.40, "lfo": 0.40, "pct": 0.40},
        "all_strict":             {"fg": 0.70, "k": 0.70, "lfo": 0.70, "pct": 0.70},
        "all_very_strict":        {"fg": 0.80, "k": 0.80, "lfo": 0.80, "pct": 0.80},
        "loose_pct_strict_rest":  {"fg": 0.70, "k": 0.70, "lfo": 0.70, "pct": 0.40},
        "tight_filter_g_loose_rest": {"fg": 0.80, "k": 0.50, "lfo": 0.50, "pct": 0.50},
        "tight_kalshi_loose_rest":   {"fg": 0.50, "k": 0.80, "lfo": 0.50, "pct": 0.50},
        "tight_lfo_loose_rest":      {"fg": 0.50, "k": 0.50, "lfo": 0.80, "pct": 0.50},
        "tight_pct_loose_rest":      {"fg": 0.50, "k": 0.50, "lfo": 0.50, "pct": 0.80},
        "balanced_60":            {"fg": 0.60, "k": 0.60, "lfo": 0.60, "pct": 0.60},
        "balanced_50":            {"fg": 0.50, "k": 0.50, "lfo": 0.50, "pct": 0.50},
        "fg_lead_rest_relaxed":   {"fg": 0.40, "k": 0.70, "lfo": 0.70, "pct": 0.70},
        "k_lead_rest_relaxed":    {"fg": 0.70, "k": 0.40, "lfo": 0.70, "pct": 0.70},
        "lfo_lead_rest_relaxed":  {"fg": 0.70, "k": 0.70, "lfo": 0.40, "pct": 0.70},
        "pct_lead_rest_relaxed":  {"fg": 0.70, "k": 0.70, "lfo": 0.70, "pct": 0.40},
        "two_loose_two_strict":   {"fg": 0.40, "k": 0.40, "lfo": 0.70, "pct": 0.70},
        "kpct_loose_fglfo_strict":{"fg": 0.70, "k": 0.40, "lfo": 0.70, "pct": 0.40},
        "fglfo_loose_kpct_strict":{"fg": 0.40, "k": 0.70, "lfo": 0.40, "pct": 0.70},
        "all_60_lead_pct40":      {"fg": 0.60, "k": 0.60, "lfo": 0.60, "pct": 0.40},
        "very_tight_pct_loose_rest":{"fg": 0.40, "k": 0.40, "lfo": 0.40, "pct": 0.80},
    }
    named_results = []
    for name, cfg in named.items():
        block = (
            (head_probas["filter_g"] >= cfg["fg"])
            | (head_probas["kalshi"] >= cfg["k"])
            | (head_probas["lfo"] >= cfg["lfo"])
            | (head_probas["pct"] >= cfg["pct"])
        )
        survivors = test_indexed[~block]
        n, pnl, wr, dd = friend_rule_pnl(survivors)
        n_oos = int((~block).sum())
        rec = {
            "name": name,
            **cfg,
            "trades": n, "wr": wr, "pnl": pnl, "dd": dd,
            "n_oos_pre_friend": n_oos,
        }
        rec.update(evaluate_gates(rec, hb))
        named_results.append(rec)
    named_df = pd.DataFrame(named_results)

    # ----- Top configs -----
    top_min_dd = df.sort_values(["abs_dd", "pnl"], ascending=[True, False]).head(20).copy()
    top_max_pnl = df.sort_values(["pnl", "abs_dd"], ascending=[False, True]).head(20).copy()

    # Pareto frontier (min DD subject to PnL >= baseline)
    pareto_eligible = df[df["pnl"] >= hb["pnl"]].copy()
    pareto_top = pareto_eligible.sort_values("abs_dd", ascending=True).head(20).copy()
    if len(pareto_top) > 0:
        pareto_winner = pareto_top.iloc[0].to_dict()
    else:
        pareto_winner = None

    # Closest-to-passing
    df["closest_score"] = df["passing"] * 1000 - df["abs_dd"] / 100 + df["pnl"] / 1000
    closest = df.sort_values(["passing", "pnl"], ascending=[False, False]).iloc[0].to_dict()

    any_pass = bool((df["all_pass"]).any())

    # ----- Save best Pareto config rows (per-trade detail) for parquet -----
    if pareto_winner is not None:
        cfg = {k: pareto_winner[k] for k in ("fg", "k", "lfo", "pct")}
    else:
        cfg = {"fg": closest["fg"], "k": closest["k"], "lfo": closest["lfo"], "pct": closest["pct"]}

    block = (
        (head_probas["filter_g"] >= cfg["fg"])
        | (head_probas["kalshi"] >= cfg["k"])
        | (head_probas["lfo"] >= cfg["lfo"])
        | (head_probas["pct"] >= cfg["pct"])
    )
    survivors = test_indexed[~block].copy()
    survivors = survivors.sort_values("ts").reset_index(drop=True)
    accepted_idx, last_exit = [], None
    for i, row in survivors.iterrows():
        if last_exit is None or row["ts"] >= last_exit:
            accepted_idx.append(i)
            last_exit = row["exit_ts"]
    accepted = survivors.iloc[accepted_idx].copy()
    accepted["config_fg"] = cfg["fg"]
    accepted["config_k"] = cfg["k"]
    accepted["config_lfo"] = cfg["lfo"]
    accepted["config_pct"] = cfg["pct"]
    parquet_out = ARTIFACTS / "combined_v9_v2_simulation_holdout.parquet"
    accepted.to_parquet(parquet_out, index=False)
    print(f"[write] {parquet_out} ({len(accepted)} accepted rows)")

    # ----- Summary JSON -----
    def _row_brief(r):
        if r is None:
            return None
        return {
            "fg": float(r["fg"]), "k": float(r["k"]),
            "lfo": float(r["lfo"]), "pct": float(r["pct"]),
            "trades": int(r["trades"]),
            "wr": float(r["wr"]),
            "pnl": float(r["pnl"]),
            "dd": float(r["dd"]),
            "n_oos_pre_friend": int(r["n_oos_pre_friend"]),
            "G1": bool(r["G1"]), "G2": bool(r["G2"]),
            "G3": bool(r["G3"]), "G4": bool(r["G4"]),
            "all_pass": bool(r["all_pass"]),
            "binding": r.get("binding", ""),
            "passing": int(r.get("passing", 0)),
        }

    summary = {
        "method": "Phase 3e combined v9 stack v2: union of BLOCK decisions across FG/K/LFO/PCT.",
        "corpus": str(CORPUS.relative_to(ROOT)),
        "baseline": str(BASELINE.relative_to(ROOT)),
        "holdout_baseline": hb,
        "head_test_aucs": head_aucs,
        "grid": GRID,
        "n_combos": int(len(df)),
        "any_combo_passes_all_gates": any_pass,
        "best_pareto_winner": _row_brief(pareto_winner) if pareto_winner is not None else None,
        "closest_to_passing": _row_brief(closest),
        "top10_min_dd": [_row_brief(r) for _, r in top_min_dd.head(10).iterrows()],
        "top10_max_pnl": [_row_brief(r) for _, r in top_max_pnl.head(10).iterrows()],
        "top10_pareto": [_row_brief(r) for _, r in pareto_top.head(10).iterrows()],
        "named_configs": [
            {
                **{k: (float(v) if k in ("fg", "k", "lfo", "pct", "wr", "pnl", "dd") else v)
                   for k, v in r.items()},
            }
            for r in named_df.to_dict(orient="records")
        ],
        "gates": {
            "G1_dd_max": GATE1_DD_MAX,
            "G2_baseline_pnl": hb["pnl"],
            "G2_baseline_trades": hb["trades"],
            "G3_nOOS_min": GATE3_NOOS_MIN,
            "G4_wr_min": GATE4_WR_MIN,
        },
        # Comparators
        "compare_v9v2_best_per_head_kalshi_045": {
            "trades": 481, "wr": 0.437, "pnl": -2151, "dd": -3360,
        },
        "compare_filterless_holdout": hb,
    }
    summary_path = ARTIFACTS / "combined_v9_stack_v2_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[write] {summary_path}")

    # ----- Console table -----
    print("\n=== Top 10 Pareto frontier (min DD with PnL >= baseline) ===")
    for _, r in pareto_top.head(10).iterrows():
        print(f"  fg={r['fg']:.2f} k={r['k']:.2f} lfo={r['lfo']:.2f} pct={r['pct']:.2f}"
              f"  trades={int(r['trades']):4d} wr={r['wr']:.3f}"
              f"  pnl=${r['pnl']:7.0f} dd=${r['dd']:7.0f} gates={r['gates']}")
    print("\n=== Top 5 by lowest |DD| (any PnL) ===")
    for _, r in top_min_dd.head(5).iterrows():
        print(f"  fg={r['fg']:.2f} k={r['k']:.2f} lfo={r['lfo']:.2f} pct={r['pct']:.2f}"
              f"  trades={int(r['trades']):4d} wr={r['wr']:.3f}"
              f"  pnl=${r['pnl']:7.0f} dd=${r['dd']:7.0f} gates={r['gates']}")
    print("\n=== Top 5 by highest PnL ===")
    for _, r in top_max_pnl.head(5).iterrows():
        print(f"  fg={r['fg']:.2f} k={r['k']:.2f} lfo={r['lfo']:.2f} pct={r['pct']:.2f}"
              f"  trades={int(r['trades']):4d} wr={r['wr']:.3f}"
              f"  pnl=${r['pnl']:7.0f} dd=${r['dd']:7.0f} gates={r['gates']}")
    print(f"\nany combo passes all 4 gates? {any_pass}")
    print(f"closest-to-passing: passing={closest['passing']}/4 binding={closest['binding']}")


if __name__ == "__main__":
    main()
