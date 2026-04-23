"""Retrain the DE3 ML-G big-loss gate with RECENT-LOSS context features.

Motivated by 2026-04-22 17:06–17:14 live cascade:
  DE3 fired Long_Rev on 3 consecutive 10-pt drops; G-gate passed all 3
  with P(big_loss) = 0.18, 0.16, 0.18 (all under 0.35 threshold);
  bot ate 3× -$150 losses.

The core miss: G's features describe the CURRENT bar's shape + regime
but don't see "the bot just took a same-side loss 4 minutes ago". In
cascade regimes, the most reliable signal of "this LONG will be a big
loss" is "the last LONG was a big loss 5 minutes ago."

v2 adds:
    n_recent_losses_60min              any-side losses in last hour
    n_recent_losses_same_side_60min    same-side losses only
    n_consec_losses_same_side          immediately preceding trades
    mins_since_last_same_side_loss     time decay of recent pain
    cum_pnl_last_60min                 running dollar PnL 60-min trail
    had_consec_loss_flag               binary: last 2 same-side both lost

Output:
    artifacts/signal_gate_2025/model_de3_v3_recent_loss.joblib

Rolling-origin 6-chunk A/B vs the existing v1 gate. Decision: keep v1
unless v2 wins ≥4/6 chunks AND mean ΔAUC ≥ +0.02.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts" / "signal_gate_2025"

# Original v1 features + schema ---------------------------------------
NUMERIC_V1 = [
    "de3_entry_ret1_atr", "de3_entry_body_pos1", "de3_entry_body1_ratio",
    "de3_entry_lower_wick_ratio", "de3_entry_upper_wick_ratio",
    "de3_entry_upper1_ratio", "de3_entry_close_pos1",
    "de3_entry_flips5", "de3_entry_down3", "de3_entry_range10_atr",
    "de3_entry_dist_low5_atr", "de3_entry_dist_high5_atr",
    "de3_entry_vol1_rel20", "de3_entry_atr14",
    "de3_entry_velocity_30", "de3_entry_dist_low30_atr",
    "de3_entry_dist_high30_atr", "de3_entry_ret30_atr",
]
CATEGORICAL = ["side", "regime", "session"]
ORDINAL = ["et_hour"]

# v3 adds -------------------------------------------------------------
RECENT_LOSS_FEATS = [
    "n_recent_losses_60min",
    "n_recent_losses_same_side_60min",
    "n_consec_losses_same_side",
    "mins_since_last_same_side_loss",
    "cum_pnl_last_60min",
    "had_consec_loss_flag",
]

CLF_KWARGS = dict(
    n_estimators=200, max_depth=3, learning_rate=0.05,
    min_samples_leaf=30, random_state=42,
)


def _encode_cat(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    parts = []
    for c in cols:
        known = sorted(df[c].astype(str).unique().tolist())
        parts.append(pd.DataFrame(
            {f"{c}__{v}": (df[c].astype(str) == v).astype(int) for v in known},
            index=df.index,
        ))
    return pd.concat(parts, axis=1) if parts else pd.DataFrame(index=df.index)


def compute_recent_loss_features(df: pd.DataFrame) -> pd.DataFrame:
    """For each row, look back at chronologically-earlier rows and compute
    recent-loss context. The df must be sorted by entry_time ascending."""
    if df.empty:
        return df.assign(**{f: 0 for f in RECENT_LOSS_FEATS})
    df = df.sort_values("entry_time").reset_index(drop=True)
    ts = pd.to_datetime(df["entry_time"], utc=True, format="mixed").dt.tz_convert("America/New_York")
    sides = df["side"].astype(str).values
    losses = (df["pnl_dollars"] < 0).astype(int).values
    pnl = df["pnl_dollars"].astype(float).values
    big = df["big_loss"].astype(int).values

    out = {f: np.zeros(len(df)) for f in RECENT_LOSS_FEATS}
    for i in range(len(df)):
        t_now = ts.iloc[i]
        side_now = sides[i]
        t_60 = t_now - pd.Timedelta(minutes=60)
        mask60 = (ts > t_60) & (ts < t_now)      # strictly earlier
        mask60_np = mask60.to_numpy()
        # Any-side losses in last 60 min
        out["n_recent_losses_60min"][i] = int(losses[mask60_np].sum())
        # Same-side only
        same_side_mask = mask60_np & (sides == side_now)
        out["n_recent_losses_same_side_60min"][i] = int(
            losses[same_side_mask].sum()
        )
        # Cumulative PnL 60-min
        out["cum_pnl_last_60min"][i] = float(pnl[mask60_np].sum())
        # Consecutive same-side losses immediately preceding (scan backward)
        consec = 0
        for j in range(i - 1, -1, -1):
            if sides[j] != side_now:
                break
            if (t_now - ts.iloc[j]).total_seconds() > 2 * 60 * 60:
                break  # stop at 2-hour gap
            if losses[j] == 1:
                consec += 1
            else:
                break
        out["n_consec_losses_same_side"][i] = consec
        # Minutes since last same-side loss (0 if none in last 24h)
        out["mins_since_last_same_side_loss"][i] = 9999.0
        for j in range(i - 1, -1, -1):
            if sides[j] == side_now and losses[j] == 1:
                out["mins_since_last_same_side_loss"][i] = float(
                    (t_now - ts.iloc[j]).total_seconds() / 60.0
                )
                break
            if (t_now - ts.iloc[j]).total_seconds() > 24 * 60 * 60:
                break
        out["had_consec_loss_flag"][i] = int(consec >= 2)
    for f in RECENT_LOSS_FEATS:
        df[f] = out[f]
    return df


def build_matrices(ds_raw: pd.DataFrame, *, include_recent_loss: bool):
    """Build X, y. ds_raw must already have the recent-loss columns filled."""
    ds = ds_raw.dropna(
        subset=NUMERIC_V1 + CATEGORICAL + ORDINAL + ["big_loss"]
    ).reset_index(drop=True)
    y = ds["big_loss"].astype(int).values
    parts = [ds[NUMERIC_V1].astype(np.float32)]
    parts.append(_encode_cat(ds, CATEGORICAL))
    parts.append(ds[ORDINAL].astype(float))
    if include_recent_loss:
        parts.append(ds[RECENT_LOSS_FEATS].astype(np.float32))
    X = pd.concat(parts, axis=1).fillna(0.0)
    return X, y, ds


def rolling_ab(X_v1, X_v3, y):
    n = len(y)
    rows = []
    for t in [0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
        tr_end = int(n * t)
        te_end = min(n, int(n * (t + 0.10)))
        if te_end - tr_end < 50: continue
        y_tr, y_te = y[:tr_end], y[tr_end:te_end]
        if len(set(y_tr)) < 2 or len(set(y_te)) < 2: continue
        clf1 = GradientBoostingClassifier(**CLF_KWARGS).fit(X_v1.iloc[:tr_end], y_tr)
        clf3 = GradientBoostingClassifier(**CLF_KWARGS).fit(X_v3.iloc[:tr_end], y_tr)
        p1 = clf1.predict_proba(X_v1.iloc[tr_end:te_end])[:, list(clf1.classes_).index(1)]
        p3 = clf3.predict_proba(X_v3.iloc[tr_end:te_end])[:, list(clf3.classes_).index(1)]
        a1 = float(roc_auc_score(y_te, p1))
        a3 = float(roc_auc_score(y_te, p3))
        rows.append({"t": t, "auc_v1": a1, "auc_v3": a3, "delta": a3 - a1,
                     "n_train": tr_end, "n_test": te_end - tr_end})
        print(f"  {int(t*100):>3}% → {int((t+0.1)*100):>3}%  "
              f"v1={a1:.3f}  v3={a3:.3f}  Δ={a3-a1:+.3f}  n_te={te_end-tr_end}")
    return rows


def main():
    print("[load] training_rows_v2.parquet")
    raw = pd.read_parquet(ARTIFACTS / "training_rows_v2.parquet")
    # DE3 only (the 32 RA rows are too few)
    raw = raw[raw["strategy"] == "DynamicEngine3"].copy()
    print(f"  {len(raw)} DE3 rows  big_loss rate {raw['big_loss'].mean():.3f}")

    print("\n[feature-gen] computing recent-loss features...")
    ds = compute_recent_loss_features(raw)
    print("  new feature summary:")
    for f in RECENT_LOSS_FEATS:
        print(f"    {f:<36}  mean={ds[f].mean():>8.3f}  max={ds[f].max():>7.1f}")
    # Correlation of each new feature with big_loss (sanity check)
    print("\n  Pearson corr with big_loss:")
    for f in RECENT_LOSS_FEATS:
        corr = ds[f].corr(ds["big_loss"])
        print(f"    {f:<36}  r={corr:+.3f}")

    print("\n[build] v1 matrix (baseline)")
    X_v1, y, _ = build_matrices(ds, include_recent_loss=False)
    print(f"  X_v1 shape: {X_v1.shape}")
    print("\n[build] v3 matrix (with recent-loss)")
    X_v3, _, ds_clean = build_matrices(ds, include_recent_loss=True)
    print(f"  X_v3 shape: {X_v3.shape}  (+{len(RECENT_LOSS_FEATS)} recent-loss features)")

    print("\n=== rolling-origin A/B (6 chunks) ===")
    ab = rolling_ab(X_v1, X_v3, y)
    import statistics as s
    mean_v1 = s.mean(r["auc_v1"] for r in ab)
    mean_v3 = s.mean(r["auc_v3"] for r in ab)
    wins = sum(1 for r in ab if r["delta"] > 0)
    print(f"\nMEAN AUC  v1={mean_v1:.3f}  v3={mean_v3:.3f}  Δ={mean_v3-mean_v1:+.3f}")
    print(f"v3 beats v1 in AUC: {wins}/{len(ab)} chunks")

    # Fit final model on ALL data
    print("\n[fit] final v3 model on all data")
    clf_full = GradientBoostingClassifier(**CLF_KWARGS).fit(X_v3, y)
    cat_maps = {c: sorted(ds_clean[c].astype(str).unique().tolist()) for c in CATEGORICAL}
    full_numeric = NUMERIC_V1 + RECENT_LOSS_FEATS
    out_path = ARTIFACTS / "model_de3_v3_recent_loss.joblib"
    joblib.dump({
        "model": clf_full,
        "model_kind": "GBT_d3_per_strategy_v3_recent_loss",
        "target": "big_loss",
        "veto_threshold": 0.35,
        "feature_names": list(X_v3.columns),
        "numeric_features": full_numeric,
        "categorical_features": CATEGORICAL,
        "categorical_maps": cat_maps,
        "ordinal_features": ORDINAL,
        "uses_recent_loss": True,
        "recent_loss_keys": RECENT_LOSS_FEATS,
        "cv_auc_mean": mean_v3,
        "training_rows": len(ds_clean),
        "training_date_utc": pd.Timestamp.utcnow().isoformat(),
    }, out_path)
    print(f"[write] {out_path}")

    # Decide + print promotion recommendation
    promote = (wins >= 4) and (mean_v3 - mean_v1 >= 0.02)
    print()
    if promote:
        print(f"✅ PROMOTE: wins {wins}/6, ΔAUC {mean_v3-mean_v1:+.3f} ≥ +0.02")
        print(f"   Run: cp artifacts/signal_gate_2025/model_de3.joblib artifacts/signal_gate_2025/model_de3_v1_pre_recentloss.joblib")
        print(f"   Run: cp artifacts/signal_gate_2025/model_de3_v3_recent_loss.joblib artifacts/signal_gate_2025/model_de3.joblib")
    else:
        print(f"❌ KEEP v1: wins {wins}/6, ΔAUC {mean_v3-mean_v1:+.3f} (threshold 4/6 + +0.02)")
        print(f"   v3 saved as model_de3_v3_recent_loss.joblib for reference")

    results_path = ARTIFACTS / "de3_gate_recent_loss_ab_results.json"
    results_path.write_text(json.dumps({
        "chunks": ab, "mean_v1_auc": mean_v1, "mean_v3_auc": mean_v3,
        "delta": mean_v3 - mean_v1, "wins": wins, "promote": promote,
    }, indent=2))
    print(f"[write] {results_path}")


if __name__ == "__main__":
    main()
