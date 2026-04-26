"""V9 per-overlay retrain v2 — corrected mistake corpus + corrected baseline.

Re-runs Section 8.22 (the v9 per-overlay retrain) using:
  - Corrected mistake corpus  : artifacts/full_overlay_stack_simulation_14mo_v2.parquet
  - Corrected filterless base : artifacts/filterless_reconstruction_14mo_v2.parquet
  - Holdout window            : Jan-Apr 2026 (ts >= 2026-01-01)

Produces 4 retrained heads (filter_g, kalshi, lfo, pct), DE3 only.
HGB only (no LightGBM — OMP-crash discipline). Sequential gate: each head's
SCOPE is the candidate stream that survived the prior overlay decisions.

Walk-forward "simulation" is via the corpus column ``raw_pnl_walk`` (already
generated with the fixed simulator) minus a $7.50/trade haircut, then
single-position friend-rule applied over the surviving stream.

Outputs per head under artifacts/regime_ml_<head>_v9_v2/de3/:
  - model.joblib (only if ANY threshold passes all 4 gates AND it is the chosen one)
  - thresholds.json (best thr config)
  - metrics.json (status, sweep table, binding gate)

Plus artifacts/v9_retrain_v2_summary.json.

Per-head SCOPE for sequential application (consistent with §8.22):
  filter_g: ALL DE3 candidates
  kalshi  : DE3 candidates that survive filter_g (final-decision != BLOCK)
  lfo     : DE3 candidates that survive both filter_g and kalshi (not blocked)
  pct     : DE3 candidates that survive filter_g + kalshi + lfo (lfo IMMEDIATE)

Each head outputs a probability ``p_wrong`` (probability the trade is a big
loss). Decision: ``p_wrong >= thr -> BLOCK`` (drop the candidate). Surviving
candidates flow on; downstream heads' gates apply to whoever survived.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import joblib

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "artifacts" / "full_overlay_stack_simulation_14mo_v2.parquet"
BASELINE = ROOT / "artifacts" / "filterless_reconstruction_14mo_v2.parquet"
ARTIFACTS = ROOT / "artifacts"
HOLDOUT_CUTOFF = pd.Timestamp("2026-01-01", tz="US/Eastern")
HAIRCUT = 7.50
BIG_LOSS_THRESHOLD = -50.0  # pnl_baseline <= -50 ==> is_big_loss=1

# v9 strict gates (against CORRECTED holdout baseline)
GATE1_DD_MAX = 870.0          # HARD risk cap
GATE3_NOOS_MIN = 50           # Min OOS sample
GATE4_WR_MIN = 0.55           # Win rate floor


def load_corpus():
    df = pd.read_parquet(CORPUS)
    df = df[df["family"] == "de3"].copy()
    df["ts"] = pd.to_datetime(df["ts"])
    df["is_big_loss"] = (df["pnl_baseline"] <= BIG_LOSS_THRESHOLD).astype(int)
    df["raw_net"] = df["raw_pnl_walk"] - HAIRCUT  # per-signal net if taken
    return df.sort_values("ts").reset_index(drop=True)


def load_holdout_baseline():
    """Holdout filterless baseline: trades, PnL, WR, DD."""
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


def base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Light feature set — context + brackets + prior overlay probas."""
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
    # PCT context (always present)
    f["pct_size_mult"] = df["pct_size_mult"].values
    f["pct_tp_mult"] = df["pct_tp_mult"].values
    return f


def features_for_head(df: pd.DataFrame, head: str) -> pd.DataFrame:
    """Sequential feature set: each head sees prior overlay probas / decisions
    in addition to base features."""
    f = base_features(df)
    if head in ("kalshi", "lfo", "pct"):
        f["fg_proba"] = df["fg_proba"].fillna(0.5).values
    if head in ("lfo", "pct"):
        kp = df["k_proba"].copy()
        f["k_proba"] = kp.fillna(0.5).values  # NA = no kalshi market = neutral 0.5
        f["k_has_market"] = kp.notna().astype(int).values
    if head == "pct":
        f["lfo_proba"] = df["lfo_proba"].fillna(0.5).values
    # PCT decision is a categorical context for any head AFTER pct exists,
    # but for pct itself we exclude its own decision.
    if head != "pct":
        # one-hot pct_decision (might be informative for other heads)
        for cat in ("BREAKOUT_LEAN", "PIVOT_LEAN", "NOT_AT_LEVEL", "NEUTRAL"):
            f[f"pct_dec_{cat}"] = (df["pct_decision"] == cat).astype(int).values
    return f


def scope_mask(df: pd.DataFrame, head: str) -> pd.Series:
    """SCOPE per head — the candidates this head sees during the sequential pipeline.
    For training we use SCOPE on the train slice; for OOS we use SCOPE on the test slice
    intersected with the survivors of upstream heads' BLOCK decisions."""
    if head == "filter_g":
        return pd.Series(True, index=df.index)
    if head == "kalshi":
        return pd.Series(True, index=df.index)
    if head == "lfo":
        return pd.Series(True, index=df.index)
    if head == "pct":
        return pd.Series(True, index=df.index)
    raise ValueError(head)


def friend_rule_pnl(taken_signals: pd.DataFrame) -> tuple[int, float, float, float]:
    """Apply single-position friend-rule over taken_signals (must include exit_ts).
    Returns (n_taken, total_net_pnl, win_rate, max_dd)."""
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


def simulate_threshold_sweep(
    train: pd.DataFrame,
    test: pd.DataFrame,
    head: str,
    proba_test: np.ndarray,
    upstream_block_mask_test: pd.Series,
) -> pd.DataFrame:
    """Run threshold sweep 0.40–0.85 step 0.05.

    For each threshold:
      1. The head BLOCKS test signals where ``proba >= thr`` (within its scope).
      2. Surviving signals = (NOT blocked by upstream) AND (NOT blocked by this head).
      3. Apply friend-rule on surviving signals' raw_net (each survivor IS a taken signal).
      4. Compute trades / WR / PnL / DD.

    Note: surviving signals are simulated AS IF taken (raw_pnl_walk is per-signal
    independent walk; friend-rule decides if the engine actually puts on the trade).
    """
    rows = []
    test = test.reset_index(drop=True)
    upstream = upstream_block_mask_test.reset_index(drop=True)
    proba_arr = pd.Series(proba_test).reset_index(drop=True).values
    for thr in np.arange(0.40, 0.85 + 1e-9, 0.05):
        thr = round(float(thr), 2)
        head_block = proba_arr >= thr
        # Final survivors = NOT upstream-blocked AND NOT head-blocked
        survivors = (~upstream.values) & (~head_block)
        candidate = test[survivors].copy()
        n_oos = int(survivors.sum())
        if n_oos == 0:
            rows.append({
                "thr": thr, "trades": 0, "wr": 0.0, "pnl": 0.0, "dd": 0.0,
                "n_oos_pre_friend": 0,
            })
            continue
        n, pnl, wr, dd = friend_rule_pnl(candidate)
        rows.append({
            "thr": thr,
            "trades": n,
            "wr": wr,
            "pnl": pnl,
            "dd": dd,
            "n_oos_pre_friend": n_oos,
        })
    return pd.DataFrame(rows)


def evaluate_gates(row: dict, hb: dict) -> dict:
    """Apply the 4 ship gates against the holdout baseline."""
    g1 = abs(row["dd"]) <= GATE1_DD_MAX
    g2_pnl = row["pnl"] >= hb["pnl"]
    g2_trades = row["trades"] <= hb["trades"]
    g2 = g2_pnl and g2_trades
    g3 = row["n_oos_pre_friend"] >= GATE3_NOOS_MIN
    g4 = row["wr"] >= GATE4_WR_MIN
    binding = []
    if not g1:
        binding.append(f"G1 DD>{GATE1_DD_MAX}")
    if not g2:
        binding.append(f"G2 PnL<{hb['pnl']:.0f} or trades>{hb['trades']}")
    if not g3:
        binding.append(f"G3 nOOS<{GATE3_NOOS_MIN}")
    if not g4:
        binding.append(f"G4 WR<{GATE4_WR_MIN}")
    return {
        "G1": g1, "G2": g2, "G3": g3, "G4": g4,
        "all_pass": g1 and g2 and g3 and g4,
        "binding_gates": binding,
    }


def train_and_eval_head(
    head: str,
    train_corpus: pd.DataFrame,
    test_corpus: pd.DataFrame,
    hb: dict,
    upstream_block_mask_train: pd.Series,
    upstream_block_mask_test: pd.Series,
    out_dir: Path,
):
    """Train HGB head on train_corpus[scope], predict on test_corpus[scope],
    threshold-sweep with gate semantics, write artifacts."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Training scope: train rows where upstream did NOT block
    train_scope_mask = ~upstream_block_mask_train.values
    test_scope_mask = ~upstream_block_mask_test.values

    train_sub = train_corpus[train_scope_mask].copy()
    test_sub = test_corpus[test_scope_mask].copy()

    Xtr = features_for_head(train_sub, head).values
    ytr = train_sub["is_big_loss"].values
    Xte = features_for_head(test_sub, head).values
    yte = test_sub["is_big_loss"].values

    if len(np.unique(ytr)) < 2 or ytr.sum() < 5:
        # Cannot train
        metrics = {
            "head": head,
            "status": "KILL",
            "test_auc": None,
            "reason": f"insufficient training labels (n_pos={int(ytr.sum())})",
            "holdout_baseline": hb,
            "n_train": int(len(train_sub)),
            "n_test": int(len(test_sub)),
        }
        with open(out_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        return metrics

    model = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.05,
        max_depth=4,
        min_samples_leaf=20,
        l2_regularization=0.1,
        random_state=42,
    )
    model.fit(Xtr, ytr)

    proba_full_test = np.full(len(test_corpus), 0.0, dtype=float)
    proba_full_test[test_scope_mask] = model.predict_proba(Xte)[:, 1]

    # If sample is in scope, we have a real proba. If NOT in scope (upstream blocked),
    # the candidate is already blocked — proba doesn't matter (head gate not applied).
    test_auc = None
    try:
        if len(np.unique(yte)) >= 2:
            test_auc = float(roc_auc_score(yte, proba_full_test[test_scope_mask]))
    except Exception:
        pass

    sweep = simulate_threshold_sweep(
        train_corpus, test_corpus, head, proba_full_test, upstream_block_mask_test
    )
    # Add gate evaluation
    gate_rows = []
    for _, r in sweep.iterrows():
        g = evaluate_gates(r.to_dict(), hb)
        gate_rows.append({**r.to_dict(), **g})
    gate_df = pd.DataFrame(gate_rows)

    # Find best (passes all 4 gates → highest PnL among passers; else closest)
    passers = gate_df[gate_df["all_pass"]]
    if len(passers) > 0:
        best = passers.sort_values(["pnl", "dd"], ascending=[False, True]).iloc[0]
        status = "SHIP"
    else:
        # Closest: minimize sum of soft penalties (for diagnostic)
        # We report 3 closest configs: by min DD, by max PnL, and by best gate-count
        gate_df["passing"] = gate_df[["G1", "G2", "G3", "G4"]].sum(axis=1)
        best = gate_df.sort_values(["passing", "pnl", "dd"], ascending=[False, False, True]).iloc[0]
        status = "KILL"

    closest_min_dd = gate_df.loc[gate_df["dd"].abs().idxmin()]
    closest_max_pnl = gate_df.loc[gate_df["pnl"].idxmax()]

    # Save sweep
    gate_df.to_csv(out_dir / "sweep.csv", index=False)

    metrics = {
        "head": head,
        "status": status,
        "test_auc": test_auc,
        "n_train": int(len(train_sub)),
        "n_train_pos": int(ytr.sum()),
        "n_test": int(len(test_sub)),
        "n_test_pos": int(yte.sum()),
        "best": {
            "thr": float(best["thr"]),
            "trades": int(best["trades"]),
            "wr": float(best["wr"]),
            "pnl": float(best["pnl"]),
            "dd": float(best["dd"]),
            "n_oos_pre_friend": int(best["n_oos_pre_friend"]),
            "gates": {
                "G1": bool(best["G1"]), "G2": bool(best["G2"]),
                "G3": bool(best["G3"]), "G4": bool(best["G4"]),
            },
            "binding_gates": best["binding_gates"] if isinstance(best["binding_gates"], list) else list(best["binding_gates"]),
        },
        "closest_min_dd": {
            "thr": float(closest_min_dd["thr"]),
            "trades": int(closest_min_dd["trades"]),
            "wr": float(closest_min_dd["wr"]),
            "pnl": float(closest_min_dd["pnl"]),
            "dd": float(closest_min_dd["dd"]),
        },
        "closest_max_pnl": {
            "thr": float(closest_max_pnl["thr"]),
            "trades": int(closest_max_pnl["trades"]),
            "wr": float(closest_max_pnl["wr"]),
            "pnl": float(closest_max_pnl["pnl"]),
            "dd": float(closest_max_pnl["dd"]),
        },
        "holdout_baseline": hb,
        "gates": {
            "G1_dd_max": GATE1_DD_MAX,
            "G2_baseline_pnl": hb["pnl"],
            "G2_baseline_trades": hb["trades"],
            "G3_nOOS_min": GATE3_NOOS_MIN,
            "G4_wr_min": GATE4_WR_MIN,
        },
        "sweep": gate_df.drop(columns=["binding_gates"]).to_dict(orient="records"),
    }
    if status == "KILL":
        metrics["reason"] = "no threshold passes all 4 gates"
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    # Save thresholds
    with open(out_dir / "thresholds.json", "w") as f:
        json.dump({
            "best_thr": float(best["thr"]),
            "status": status,
            "block_rule": "p_wrong >= thr ==> BLOCK",
        }, f, indent=2)

    # Save model only if SHIP
    if status == "SHIP":
        joblib.dump(model, out_dir / "model.joblib")

    # ALSO return updated upstream_block_mask for downstream heads if SHIP
    # (We use best thresholds → produce block decisions for the rest of the pipeline.)
    test_block_mask = upstream_block_mask_test.copy()
    train_block_mask = upstream_block_mask_train.copy()
    if status == "SHIP":
        # Apply best threshold's block decisions to test
        test_block_mask = upstream_block_mask_test | (proba_full_test >= float(best["thr"]))
        # And to train (using in-sample probas — needed only for downstream training scope)
        train_proba = np.full(len(train_corpus), 0.0, dtype=float)
        train_proba[train_scope_mask] = model.predict_proba(Xtr)[:, 1]
        train_block_mask = upstream_block_mask_train | (train_proba >= float(best["thr"]))
    # If KILL, the head produces NO block — downstream sees same upstream mask.

    return metrics, train_block_mask, test_block_mask, model


def main():
    print(f"Loading corpus: {CORPUS}")
    de3 = load_corpus()
    print(f"  DE3 rows: {len(de3)}, big_loss rate: {de3['is_big_loss'].mean():.3f}")

    train = de3[de3["ts"] < HOLDOUT_CUTOFF].copy().reset_index(drop=True)
    test = de3[de3["ts"] >= HOLDOUT_CUTOFF].copy().reset_index(drop=True)
    print(f"  Train: {len(train)} rows ({train['is_big_loss'].sum()} pos)")
    print(f"  Test:  {len(test)} rows ({test['is_big_loss'].sum()} pos)")

    hb = load_holdout_baseline()
    print(f"  Holdout baseline: {hb}")

    summary = {"holdout_baseline": hb, "heads": {}}

    # Sequential pipeline state
    train_block = pd.Series(False, index=range(len(train)))
    test_block = pd.Series(False, index=range(len(test)))

    for head in ["filter_g", "kalshi", "lfo", "pct"]:
        out_dir = ARTIFACTS / f"regime_ml_{'filterg' if head=='filter_g' else head}_v9_v2" / "de3"
        print(f"\n=== Training head: {head} -> {out_dir} ===")
        result = train_and_eval_head(
            head, train, test, hb,
            train_block, test_block,
            out_dir,
        )
        if isinstance(result, dict):
            metrics = result
            # Block masks unchanged
        else:
            metrics, train_block, test_block, _model = result
        summary["heads"][head] = {
            "status": metrics["status"],
            "best": metrics.get("best"),
            "closest_min_dd": metrics.get("closest_min_dd"),
            "closest_max_pnl": metrics.get("closest_max_pnl"),
            "n_train": metrics.get("n_train"),
            "n_test": metrics.get("n_test"),
            "test_auc": metrics.get("test_auc"),
        }
        print(f"  -> {metrics['status']} (test_auc={metrics.get('test_auc')})")
        if "best" in metrics and metrics["best"]:
            b = metrics["best"]
            print(f"     best: thr={b['thr']} trades={b['trades']} wr={b['wr']:.3f} pnl=${b['pnl']:.0f} dd=${b['dd']:.0f}")

    summary_path = ARTIFACTS / "v9_retrain_v2_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary written: {summary_path}")


if __name__ == "__main__":
    main()
