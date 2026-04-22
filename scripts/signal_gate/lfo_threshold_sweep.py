"""Sweep LFO veto_threshold on realized holdout PnL.

v2 LFO's output is `p_wait_better` — probability that waiting for a
better fill beats immediate. Current production threshold = 0.40 (carried
over from v1). With v2's different score distribution, the optimal
cutoff may have moved.

This script:
  1. Loads LFO training data (which has `imm_pnl_dol` / `wait_pnl_dol`
     — real dollar outcomes per signal, already baked into the parquet).
  2. Uses 6 rolling-origin windows (same protocol as the A/B harness).
  3. In each window, trains v2 on the train chunk, scores the test
     chunk, and for each threshold ∈ {0.30, ..., 0.60} computes
     realized holdout PnL:
         per signal: if p >= thr  →  realized = wait_pnl_dol
                     else         →  realized = imm_pnl_dol
         total = sum across test rows
  4. Aggregates across windows and picks the threshold that maximizes
     mean realized PnL.

Updates `veto_threshold` on the canonical + _v2 payloads if the
optimizer finds a better value. Writes
`artifacts/signal_gate_2025/lfo_threshold_sweep_results.json`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.signal_gate.retrain_with_encoder import (
    compute_encoder_embeddings, compute_cross_market_features,
)

NY = ZoneInfo("America/New_York")
ARTIFACTS = ROOT / "artifacts" / "signal_gate_2025"

NUMERIC = [
    "de3_entry_ret1_atr", "de3_entry_body_pos1", "de3_entry_body1_ratio",
    "de3_entry_lower_wick_ratio", "de3_entry_upper_wick_ratio",
    "de3_entry_range10_atr", "de3_entry_vol1_rel20", "de3_entry_atr14",
    "dist_to_bank_below", "dist_to_bank_above", "dist_to_bank_in_dir",
    "bar_range_pts", "bar_close_pct_body", "sl_dist_pts", "tp_dist_pts",
    "atr_ratio_to_sl",
]
CATEGORICAL = ["side", "session", "mkt_regime"]
THRESHOLDS = [0.30, 0.32, 0.34, 0.36, 0.38, 0.40, 0.42, 0.44, 0.46, 0.48,
              0.50, 0.52, 0.54, 0.56, 0.58, 0.60]
CLF_KWARGS = dict(n_estimators=250, max_depth=4, learning_rate=0.05,
                  min_samples_leaf=50, random_state=42)


def _encode_cat(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    parts = []
    for c in cols:
        known = sorted(df[c].astype(str).unique().tolist())
        parts.append(pd.DataFrame(
            {f"{c}__{v}": (df[c].astype(str) == v).astype(int) for v in known},
            index=df.index,
        ))
    return pd.concat(parts, axis=1) if parts else pd.DataFrame(index=df.index)


def build_v2_matrix(ds: pd.DataFrame):
    ds = ds.dropna(subset=NUMERIC + CATEGORICAL + ["label_wait_better"]).reset_index(drop=True)
    print("[sweep] computing encoder embeddings...")
    emb = compute_encoder_embeddings(ds, ts_col="entry_time")
    print("[sweep] computing cross-market features...")
    cm_df = compute_cross_market_features(ds, ts_col="entry_time")
    parts = [ds[NUMERIC].astype(np.float32)]
    parts.append(_encode_cat(ds, CATEGORICAL))
    parts.append(cm_df)
    parts.append(pd.DataFrame(emb, columns=[f"enc_{i:02d}" for i in range(emb.shape[1])],
                              index=ds.index).astype(np.float32))
    X = pd.concat(parts, axis=1).fillna(0.0)
    return X, ds


def realized_pnl(p, thr, imm_dol, wait_dol):
    """Total dollar PnL if we WAIT when p >= thr else IMMEDIATE."""
    take_wait = (np.asarray(p) >= thr).astype(bool)
    realized = np.where(take_wait, wait_dol, imm_dol)
    return float(realized.sum())


def main():
    ds_all = pd.read_parquet(ARTIFACTS / "lfo_training_data.parquet")
    ds_all = ds_all.sort_values("entry_time").reset_index(drop=True)
    print(f"[sweep] {len(ds_all)} training rows")

    X, ds = build_v2_matrix(ds_all.copy())
    y = ds["label_wait_better"].astype(int).values
    imm_dol = ds["imm_pnl_dol"].astype(float).values
    wait_dol = ds["wait_pnl_dol"].astype(float).values
    n = len(ds)
    print(f"[sweep] feature matrix: {X.shape}")

    # Rolling-origin — same 6 windows as the A/B
    per_chunk_pnl = {t: [] for t in THRESHOLDS}
    baseline_rule_pnl = []   # "always IMMEDIATE" vs "always WAIT" vs "oracle"
    oracle_pnl = []
    always_imm_pnl = []
    always_wait_pnl = []
    chunk_summaries = []

    for t in [0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
        tr_end = int(n * t)
        te_end = min(n, int(n * (t + 0.10)))
        if te_end - tr_end < 50: continue
        X_tr, X_te = X.iloc[:tr_end], X.iloc[tr_end:te_end]
        y_tr = y[:tr_end]
        imm_te, wait_te = imm_dol[tr_end:te_end], wait_dol[tr_end:te_end]

        if len(set(y_tr)) < 2:
            continue
        clf = GradientBoostingClassifier(**CLF_KWARGS).fit(X_tr, y_tr)
        p = clf.predict_proba(X_te)[:, list(clf.classes_).index(1)]

        row = {"train_frac": t, "test_frac": t+0.10, "n_test": len(p),
               "oracle_pnl": float(np.maximum(imm_te, wait_te).sum()),
               "always_imm_pnl": float(imm_te.sum()),
               "always_wait_pnl": float(wait_te.sum())}
        oracle_pnl.append(row["oracle_pnl"])
        always_imm_pnl.append(row["always_imm_pnl"])
        always_wait_pnl.append(row["always_wait_pnl"])
        for thr in THRESHOLDS:
            pnl = realized_pnl(p, thr, imm_te, wait_te)
            per_chunk_pnl[thr].append(pnl)
            row[f"pnl_thr{thr:.2f}"] = pnl
        chunk_summaries.append(row)

    # Aggregate
    print(f"\n{'thr':>6}  {'mean_pnl':>10}  {'sum_pnl':>10}  {'wins_vs_0.40':>14}")
    thr_04_pnl = per_chunk_pnl[0.40]
    best_thr = 0.40
    best_mean = float("-inf")
    summary = {}
    for thr in THRESHOLDS:
        pnls = per_chunk_pnl[thr]
        mean_p = float(np.mean(pnls))
        sum_p = float(np.sum(pnls))
        wins = sum(1 for a, b in zip(pnls, thr_04_pnl) if a > b)
        marker = ""
        if mean_p > best_mean:
            best_mean = mean_p
            best_thr = thr
            marker = "  ← best"
        summary[f"thr{thr:.2f}"] = {"mean_pnl": mean_p, "sum_pnl": sum_p,
                                     "wins_vs_0.40": wins, "n_chunks": len(pnls)}
        print(f"{thr:>6.2f}  {mean_p:>+10.2f}  {sum_p:>+10.2f}  {wins:>14d}{marker}")

    print(f"\n{'reference':>6}  {np.mean(oracle_pnl):>+10.2f}  {np.sum(oracle_pnl):>+10.2f}    oracle (upper bound)")
    print(f"{'    ':>6}  {np.mean(always_imm_pnl):>+10.2f}  {np.sum(always_imm_pnl):>+10.2f}    always IMMEDIATE")
    print(f"{'    ':>6}  {np.mean(always_wait_pnl):>+10.2f}  {np.sum(always_wait_pnl):>+10.2f}    always WAIT")

    print(f"\nBest threshold: {best_thr:.2f}  (mean chunk PnL = ${best_mean:+.2f})")
    print(f"Current threshold: 0.40     (mean chunk PnL = ${np.mean(thr_04_pnl):+.2f})")
    delta = best_mean - np.mean(thr_04_pnl)
    print(f"Delta if we promote best:    ${delta:+.2f}/chunk  (×{len(thr_04_pnl)} chunks = ${delta*len(thr_04_pnl):+.2f})")

    results = {
        "chunk_summaries": chunk_summaries,
        "threshold_summary": summary,
        "oracle_pnl_mean": float(np.mean(oracle_pnl)),
        "always_imm_pnl_mean": float(np.mean(always_imm_pnl)),
        "always_wait_pnl_mean": float(np.mean(always_wait_pnl)),
        "best_threshold": best_thr,
        "best_mean_pnl": best_mean,
        "current_threshold": 0.40,
        "current_mean_pnl": float(np.mean(thr_04_pnl)),
        "delta_per_chunk": delta,
    }
    out = ARTIFACTS / "lfo_threshold_sweep_results.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n[write] {out}")

    # Decide whether to promote new threshold
    # Require the new threshold to beat 0.40 on mean AND in at least 4/6 chunks
    thr_best_pnls = per_chunk_pnl[best_thr]
    chunk_wins = sum(1 for a, b in zip(thr_best_pnls, thr_04_pnl) if a > b)
    if best_thr != 0.40 and chunk_wins >= 4:
        print(f"\n[promote] threshold {best_thr:.2f} wins {chunk_wins}/{len(thr_best_pnls)} chunks — updating payloads")
        for fn in ("model_lfo.joblib", "model_lfo_v2.joblib"):
            p = joblib.load(ARTIFACTS / fn)
            p["veto_threshold_prev"] = p.get("veto_threshold", 0.40)
            p["veto_threshold"] = best_thr
            p["veto_threshold_optimized_on"] = "lfo_threshold_sweep_results.json (6 rolling-origin chunks, realized PnL)"
            joblib.dump(p, ARTIFACTS / fn)
            print(f"  [write] {fn}  veto_threshold: {p['veto_threshold_prev']:.2f} → {best_thr:.2f}")
    else:
        print(f"\n[keep] current threshold 0.40 retained (best={best_thr:.2f} only wins {chunk_wins}/{len(thr_best_pnls)} chunks)")


if __name__ == "__main__":
    main()
