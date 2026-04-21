#!/usr/bin/env python3
"""Train 2025 signal-gate on full 2025 data, test OOS on April 2026.

Uses the SAME feature extractor as build_training_data.py but points at the
April 2026 replay folders for the held-out set. Reports per-threshold P&L
delta on the OOS set.
"""
from __future__ import annotations

import sys
import json
import re
from pathlib import Path
from datetime import datetime
from bisect import bisect_right
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report

ROOT = Path("/Users/wes/Downloads/JULIE001")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "signal_gate"))

from build_de3_chosen_shape_dataset import _compute_feature_frame, ENTRY_SHAPE_COLUMNS  # noqa: E402
from reconstruct_regime import reconstruct_from_log  # noqa: E402
from build_training_data import bars_df_from_log, session_bucket  # noqa: E402
from train_gate import (  # noqa: E402
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, ORDINAL_FEATURES, assemble_X,
)

NY = ZoneInfo("America/New_York")


def extract_trade_rows(folder: Path) -> pd.DataFrame:
    """Extract feature rows for each trade in folder/closed_trades.json."""
    ct = folder / "closed_trades.json"
    log = folder / "topstep_live_bot.log"
    if not ct.exists() or not log.exists():
        return pd.DataFrame()
    trades = json.loads(ct.read_text(encoding="utf-8"))
    bars = bars_df_from_log(log)
    if bars.empty:
        return pd.DataFrame()
    feats = _compute_feature_frame(bars)
    regime_events = reconstruct_from_log(log)
    regime_keys = [e[0] for e in regime_events]
    rows = []
    for t in trades:
        try:
            et = datetime.fromisoformat(t["entry_time"]).astimezone(NY)
        except Exception:
            continue
        try:
            idx = feats.index.searchsorted(et)
        except Exception:
            continue
        if idx <= 0 or idx > len(feats):
            continue
        feat_row = feats.iloc[idx - 1]
        if feat_row.isna().all():
            continue
        ri = bisect_right(regime_keys, et) - 1
        regime = regime_events[ri][1] if ri >= 0 else "warmup"
        pnl = float(t.get("pnl_dollars", 0.0) or 0.0)
        row = {
            "day": et.date().isoformat(),
            "entry_time": et.isoformat(),
            "et_hour": et.hour,
            "session": session_bucket(et.hour),
            "side": str(t.get("side", "")).upper(),
            "size": int(t.get("size", 1) or 1),
            "entry_price": float(t.get("entry_price", 0.0) or 0.0),
            "pnl_dollars": pnl,
            "win": 1 if pnl > 0 else 0,
            "big_loss": 1 if pnl <= -100 else 0,
            "regime": regime,
        }
        for col in ENTRY_SHAPE_COLUMNS:
            row[col] = float(feat_row.get(col, float("nan")))
        rows.append(row)
    return pd.DataFrame(rows)


def build_oos_set() -> pd.DataFrame:
    """Combine April 2026 replay folders into one OOS DataFrame."""
    frames = []
    # Big replay (Apr 1-17)
    p1 = ROOT / "backtest_reports" / "replay_apr2026_p1"
    for loop in sorted(p1.glob("live_loop_MES_*")):
        if (loop / "closed_trades.json").exists():
            frames.append(extract_trade_rows(loop))
            break
    # Apr 19-20
    w = ROOT / "backtest_reports" / "replay_apr20" / "baseline_warm"
    for loop in sorted(w.glob("live_loop_MES_*")):
        if (loop / "closed_trades.json").exists():
            df = extract_trade_rows(loop)
            # Filter to Apr 19/20 only (folder spans 18:00 Apr 19 → 16:00 Apr 20)
            df = df[df["day"].str.startswith("2026-04")]
            frames.append(df)
            break
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def evaluate_thresholds(df: pd.DataFrame, y_proba: np.ndarray, target_side: str):
    """target_side = 'win' (veto if P(win) < t) or 'loss' (veto if P(loss) > t)."""
    base_pnl = df["pnl_dollars"].sum()
    base_dd_viols = _dd_violations(df, veto_mask=np.zeros(len(df), dtype=bool))
    rows = []
    thresholds = np.arange(0.20, 0.80, 0.025)
    for t in thresholds:
        if target_side == "win":
            veto = y_proba < t
        else:
            veto = y_proba > t
        kept_pnl = df.loc[~veto, "pnl_dollars"].sum()
        vetoed_pnl = df.loc[veto, "pnl_dollars"].sum()
        vetoed_wins = int(((df["pnl_dollars"] > 0) & veto).sum())
        vetoed_losses = int(((df["pnl_dollars"] < 0) & veto).sum())
        # Recompute DD violations with veto applied
        kept_dd_viols = _dd_violations(df, veto_mask=veto)
        rows.append({
            "thresh": round(float(t), 3),
            "vetoed_n": int(veto.sum()),
            "vetoed_wins": vetoed_wins,
            "vetoed_losses": vetoed_losses,
            "vetoed_pnl": round(float(vetoed_pnl), 2),
            "kept_pnl": round(float(kept_pnl), 2),
            "delta": round(float(kept_pnl - base_pnl), 2),
            "base_dd_viols": base_dd_viols,
            "dd_viols": kept_dd_viols,
        })
    return rows


def _dd_violations(df: pd.DataFrame, veto_mask: np.ndarray, dd_limit: float = 350.0) -> int:
    """Count days where the trailing DD (peak P&L - current P&L) exceeds $350."""
    by_day = df.assign(veto=veto_mask).sort_values("entry_time").groupby("day")
    viols = 0
    for _, day_df in by_day:
        cum = 0.0
        peak = 0.0
        max_dd = 0.0
        for _, row in day_df.iterrows():
            if row["veto"]:
                continue
            cum += float(row["pnl_dollars"])
            peak = max(peak, cum)
            max_dd = max(max_dd, peak - cum)
        if max_dd > dd_limit:
            viols += 1
    return viols


def main():
    print("[train] loading 2025 training rows...")
    df_train = pd.read_parquet(ROOT / "artifacts" / "signal_gate_2025" / "training_rows.parquet")
    required = NUMERIC_FEATURES + CATEGORICAL_FEATURES + ORDINAL_FEATURES
    df_train = df_train.dropna(subset=[c for c in required if c in df_train.columns]).reset_index(drop=True)
    print(f"  {len(df_train)} rows, {(df_train['pnl_dollars']>0).mean():.1%} win rate, "
          f"base P&L ${df_train['pnl_dollars'].sum():+.2f}")

    print("[oos] extracting April 2026 rows...")
    df_oos = build_oos_set()
    df_oos = df_oos.dropna(subset=[c for c in required if c in df_oos.columns]).reset_index(drop=True)
    print(f"  {len(df_oos)} rows, {(df_oos['pnl_dollars']>0).mean():.1%} win rate, "
          f"base P&L ${df_oos['pnl_dollars'].sum():+.2f}")

    X_train, cat_maps = assemble_X(df_train)
    X_oos, _ = assemble_X(df_oos, categorical_maps=cat_maps)
    # Align columns (in case OOS has an unseen category level)
    for col in X_train.columns:
        if col not in X_oos.columns:
            X_oos[col] = 0
    X_oos = X_oos[X_train.columns]

    print("\n" + "=" * 78)
    print("OOS EVAL — trained on 2025, tested on April 2026")
    print("=" * 78)

    # Three configs
    configs = [
        ("GBT_d3_win",    GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, min_samples_leaf=20, random_state=42), "win"),
        ("GBT_d5_win",    GradientBoostingClassifier(n_estimators=150, max_depth=5, learning_rate=0.03, min_samples_leaf=30, random_state=42), "win"),
        ("GBT_d3_big",    GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, min_samples_leaf=20, random_state=42), "big_loss"),
        ("GBT_d5_big",    GradientBoostingClassifier(n_estimators=150, max_depth=5, learning_rate=0.03, min_samples_leaf=30, random_state=42), "big_loss"),
        ("RF_d6_win",     RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=10, random_state=42), "win"),
        ("RF_d6_big",     RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=10, random_state=42), "big_loss"),
    ]
    for name, clf, target in configs:
        y_train = df_train["win" if target == "win" else "big_loss"].astype(int).values
        y_oos = df_oos["win" if target == "win" else "big_loss"].astype(int).values
        clf.fit(X_train, y_train)
        p_oos = clf.predict_proba(X_oos)[:, 1]
        auc = roc_auc_score(y_oos, p_oos) if len(set(y_oos)) == 2 else float("nan")
        target_side = "win" if target == "win" else "loss"
        results = evaluate_thresholds(df_oos, p_oos, target_side)
        # Best positive-delta threshold that vetoes < 50% of trades
        good = [r for r in results if r["vetoed_n"] / max(1, len(df_oos)) < 0.50 and r["delta"] > 0]
        best = max(good, key=lambda r: r["delta"]) if good else max(results, key=lambda r: r["delta"])
        print(f"\n--- {name} (target={target}) ---")
        print(f"  OOS AUC: {auc:.3f}")
        print(f"  best @ veto<{50}%: thresh={best['thresh']}, "
              f"vetoed={best['vetoed_n']} ({best['vetoed_wins']}W/{best['vetoed_losses']}L), "
              f"vetoed_pnl=${best['vetoed_pnl']:+.2f}, "
              f"delta=${best['delta']:+.2f}, "
              f"DD viols: {best['base_dd_viols']}→{best['dd_viols']}")
        # Show threshold curve (subsample)
        print(f"  threshold curve (every 5th):")
        print(f"    {'thresh':>7} {'vet%':>6} {'W/L':>8} {'kept_$':>11} {'delta':>10} {'DD':>5}")
        for r in results[::5]:
            pct = 100 * r["vetoed_n"] / max(1, len(df_oos))
            wl = f"{r['vetoed_wins']}/{r['vetoed_losses']}"
            print(f"    {r['thresh']:>7.3f} {pct:>5.1f}% {wl:>8} {r['kept_pnl']:>+11.2f} {r['delta']:>+10.2f} {r['dd_viols']:>5}")


if __name__ == "__main__":
    main()
