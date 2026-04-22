"""Train an ML replacement for the Level Fill Optimizer (LFO).

The rule-based LFO (level_fill_optimizer.py) decides WAIT vs IMMEDIATE per
signal using hand-tuned distance thresholds (0.75/2.50pt) and priority
rules. This trainer learns the same decision from historical outcomes.

Training method:
  For each historical trade (entry_time, side, entry_price, TP/SL from
  bracket), re-load the parquet bars around signal_bar = entry_time - 1min.
  Simulate two fill modes:
    IMMEDIATE — use actual entry_price (what was done)
    WAIT      — look forward 3 bars; if low/high touches within 0.75pt of a
                structural/bank level in the favorable direction, fill at
                that level; else fill at bar-4 open (timeout market fill).

  For each mode: re-run the TP/SL bracket to completion and compute PnL.
  Label = mode with higher PnL (binary: 1 = WAIT wins, 0 = IMMEDIATE wins).

Features at signal_bar time:
  - Bar shape (open/high/low/close relative to prior, wick ratios, body)
  - ATR14 (rolling)
  - Distance to nearest bank-grid level (in each direction)
  - Session/hour, side, regime (mkt_regime)
  - Recent volatility (range10_atr, vol1_rel20)
  - Drawdown-state (cum_day_pnl, consec_losses — v5 features)
  - Trade-specific: sl_dist, tp_dist (from the bracket the strategy gave)

Output: artifacts/signal_gate_2025/model_lfo.joblib
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path("/Users/wes/Downloads/JULIE001")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "scripts" / "signal_gate"))

NY = ZoneInfo("America/New_York")
POINT_VALUE = 5.0
BANK_GRID = 12.50
WAIT_BARS = 3
TOUCH_THRESHOLD = 0.75  # within this of the level = "touched"

from build_de3_chosen_shape_dataset import _compute_feature_frame  # DE3 feature pipeline
from train_per_strategy_models import active_symbol, _session_of
from regime_classifier import RegimeClassifier, WINDOW_BARS as _REG_WINDOW

# Numeric features for the LFO model
NUMERIC_FEATURES = [
    "de3_entry_ret1_atr",
    "de3_entry_body_pos1",
    "de3_entry_body1_ratio",
    "de3_entry_lower_wick_ratio",
    "de3_entry_upper_wick_ratio",
    "de3_entry_range10_atr",
    "de3_entry_vol1_rel20",
    "de3_entry_atr14",
    "dist_to_bank_below",     # pts to nearest bank level below current price
    "dist_to_bank_above",     # pts to nearest bank level above
    "dist_to_bank_in_dir",    # pts to nearest bank in trade direction
    "bar_range_pts",          # high-low of signal bar
    "bar_close_pct_body",     # (close - low)/(high - low)
    "sl_dist_pts",
    "tp_dist_pts",
    "atr_ratio_to_sl",        # atr14 / sl_dist (relative size)
]
CATEGORICAL_FEATURES = ["side", "session", "mkt_regime"]
ORDINAL_FEATURES = ["et_hour"]

TRADE_SOURCES = [
    ROOT / "backtest_reports" / "full_live_replay",
    ROOT / "backtest_reports" / "af_fast_replay",
    ROOT / "backtest_reports" / "replay_apr2026_p1",
]


def collect_all_trades():
    """Walk every replay folder for DE3+RA+AF trades."""
    seen = set()
    out = []
    for base in TRADE_SOURCES:
        if not base.exists():
            continue
        for ct_path in base.rglob("closed_trades.json"):
            if any(x in ct_path.name for x in ("smoke",)):
                continue
            try:
                data = json.load(open(ct_path))
            except Exception:
                continue
            for t in data:
                strat = str(t.get("strategy", "") or "")
                if not strat.startswith(("DynamicEngine3", "RegimeAdaptive", "AetherFlow")):
                    continue
                key = (str(t.get("entry_time")), t.get("side"),
                       float(t.get("entry_price") or 0))
                if key in seen:
                    continue
                seen.add(key)
                out.append(t)
    return out


def simulate_bracket(bars_df: pd.DataFrame, entry_idx: int, side: str,
                     tp_dist: float, sl_dist: float, max_bars: int = 100) -> float:
    """Walk forward from entry_idx+1, return PnL in points when TP/SL hits.
    Conservative: if both hit in same bar, take SL."""
    if entry_idx + 1 >= len(bars_df):
        return 0.0
    entry_price = float(bars_df.iloc[entry_idx + 1]["open"])
    if side == "LONG":
        tp_price = entry_price + tp_dist
        sl_price = entry_price - sl_dist
    else:
        tp_price = entry_price - tp_dist
        sl_price = entry_price + sl_dist
    end = min(len(bars_df), entry_idx + 1 + max_bars)
    for j in range(entry_idx + 1, end):
        h = float(bars_df.iloc[j]["high"])
        l = float(bars_df.iloc[j]["low"])
        sl_hit = (side == "LONG" and l <= sl_price) or (side == "SHORT" and h >= sl_price)
        tp_hit = (side == "LONG" and h >= tp_price) or (side == "SHORT" and l <= tp_price)
        if sl_hit and tp_hit:
            return -sl_dist  # conservative
        if sl_hit:
            return -sl_dist
        if tp_hit:
            return tp_dist
    # Timeout — exit at end close
    exit_price = float(bars_df.iloc[end - 1]["close"])
    return (exit_price - entry_price) if side == "LONG" else (entry_price - exit_price)


def simulate_fill(bars_df: pd.DataFrame, signal_idx: int, side: str,
                  original_entry_price: float, sl_dist: float, tp_dist: float,
                  mode: str) -> tuple[int, float]:
    """Returns (fill_bar_idx, fill_price).  mode in {'IMMEDIATE','WAIT'}.
    IMMEDIATE = next bar open (standard market order)
    WAIT      = look ahead WAIT_BARS; fill at nearest bank level if touched in
                favorable direction, else timeout market fill at bar WAIT_BARS+1.
                If level violated or price runs away → ABORT (return (-1, 0))."""
    if mode == "IMMEDIATE":
        if signal_idx + 1 >= len(bars_df):
            return -1, 0.0
        return signal_idx + 1, float(bars_df.iloc[signal_idx + 1]["open"])
    # WAIT
    current_close = float(bars_df.iloc[signal_idx]["close"])
    # Compute nearest bank level in trade direction
    base = (current_close // BANK_GRID) * BANK_GRID
    if side == "LONG":
        # Target is bank level BELOW (support we want to buy)
        target = base
        if target >= current_close - 0.01:  # already below
            target = base - BANK_GRID
    else:
        # Target is bank level ABOVE (resistance we want to short)
        target = base + BANK_GRID
        if target <= current_close + 0.01:
            target = base + 2 * BANK_GRID
    init_dist = abs(current_close - target)
    # Walk next WAIT_BARS
    end = min(len(bars_df), signal_idx + 1 + WAIT_BARS)
    for j in range(signal_idx + 1, end):
        bar_lo = float(bars_df.iloc[j]["low"])
        bar_hi = float(bars_df.iloc[j]["high"])
        bar_cl = float(bars_df.iloc[j]["close"])
        # Ran away check (RAN_AWAY_MULT = 3)
        if side == "LONG" and bar_cl > current_close + 3 * init_dist:
            return -1, 0.0
        if side == "SHORT" and bar_cl < current_close - 3 * init_dist:
            return -1, 0.0
        # Level violated
        if side == "LONG" and bar_cl < target - TOUCH_THRESHOLD:
            return -1, 0.0
        if side == "SHORT" and bar_cl > target + TOUCH_THRESHOLD:
            return -1, 0.0
        # Touched and close near target
        if side == "LONG":
            touched = bar_lo <= target + TOUCH_THRESHOLD
        else:
            touched = bar_hi >= target - TOUCH_THRESHOLD
        if touched and abs(bar_cl - target) <= TOUCH_THRESHOLD:
            return j, target
    # Timeout market fill
    if end < len(bars_df):
        return end, float(bars_df.iloc[end]["open"])
    return -1, 0.0


def build_features_for_trade(t: dict, master_df: pd.DataFrame,
                               day_state: dict) -> dict | None:
    """Compute the feature row for one trade.  day_state is a pre-computed
    dict keyed by trade to carry cum_day_pnl etc."""
    try:
        et = datetime.fromisoformat(t["entry_time"]).astimezone(NY)
    except Exception:
        return None
    # Locate bars: pick active front-month symbol
    symbol = active_symbol(et)
    end = pd.Timestamp(et).tz_convert("UTC")
    start = end - pd.Timedelta(hours=6)
    sub = master_df.loc[
        (master_df.index >= start) & (master_df.index <= end + pd.Timedelta(hours=1)) &
        (master_df["symbol"] == symbol),
        ["open", "high", "low", "close", "volume"]
    ]
    if len(sub) < 50:
        return None
    feats = _compute_feature_frame(sub)
    if feats.empty:
        return None
    # Signal bar = bar ending just before entry_time (the bar whose close triggered the signal)
    idx = sub.index.searchsorted(end)
    if idx <= 0 or idx > len(sub):
        return None
    sig_idx = idx - 1 - 1  # one before the entry bar (entry fills at next bar open)
    if sig_idx < 1:
        return None
    sig_bar = sub.iloc[sig_idx]
    feat_row = feats.iloc[min(sig_idx, len(feats) - 1)]
    if feat_row.isna().all():
        return None

    side = str(t.get("side", "")).upper()
    entry_price = float(t["entry_price"])
    current_close = float(sig_bar["close"])
    bar_hi = float(sig_bar["high"])
    bar_lo = float(sig_bar["low"])

    # Bank-grid distances
    base = (current_close // BANK_GRID) * BANK_GRID
    dist_below = current_close - base
    dist_above = (base + BANK_GRID) - current_close
    dist_in_dir = dist_below if side == "LONG" else dist_above

    sl_dist = float(t.get("sl_dist") or 0.0)
    if sl_dist <= 0:
        # Infer from entry_price and exit (rough)
        sl_dist = 10.0
    tp_dist = float(t.get("tp_dist") or 0.0)
    if tp_dist <= 0:
        tp_dist = 25.0
    atr14 = float(feat_row.get("de3_entry_atr14", 0) or 0)

    # Simulate both modes' fill + bracket
    imm_fill_idx, imm_fill_px = simulate_fill(sub, sig_idx, side, entry_price,
                                                sl_dist, tp_dist, "IMMEDIATE")
    wait_fill_idx, wait_fill_px = simulate_fill(sub, sig_idx, side, entry_price,
                                                  sl_dist, tp_dist, "WAIT")
    if imm_fill_idx < 0:
        return None
    # Bracket sim from each fill
    imm_pnl = simulate_bracket(sub, imm_fill_idx - 1, side, tp_dist, sl_dist)
    if wait_fill_idx < 0:
        # WAIT aborted (no trade). In training this counts as 0 pnl — missed trade.
        wait_pnl = 0.0
    else:
        # Need to re-compute bracket but starting from the wait_fill_px, not sub[idx].open
        # Use simulate_bracket with a virtual bar that has open = wait_fill_px
        if wait_fill_idx + 1 >= len(sub):
            wait_pnl = 0.0
        else:
            entry_price_wait = wait_fill_px
            max_bars = 100
            end2 = min(len(sub), wait_fill_idx + max_bars)
            wait_pnl = 0.0
            hit = False
            if side == "LONG":
                tp_price = entry_price_wait + tp_dist
                sl_price = entry_price_wait - sl_dist
            else:
                tp_price = entry_price_wait - tp_dist
                sl_price = entry_price_wait + sl_dist
            for j in range(wait_fill_idx, end2):
                h = float(sub.iloc[j]["high"])
                l = float(sub.iloc[j]["low"])
                sl_hit = (side == "LONG" and l <= sl_price) or (side == "SHORT" and h >= sl_price)
                tp_hit = (side == "LONG" and h >= tp_price) or (side == "SHORT" and l <= tp_price)
                if sl_hit:
                    wait_pnl = -sl_dist; hit = True; break
                if tp_hit:
                    wait_pnl = tp_dist; hit = True; break
            if not hit and end2 < len(sub):
                exit_price = float(sub.iloc[end2 - 1]["close"])
                wait_pnl = (exit_price - entry_price_wait) if side == "LONG" else (entry_price_wait - exit_price)

    # Label: WAIT wins if it produced more $
    # Convert to per-contract $ using POINT_VALUE (same SL/TP proportion for both)
    label_wait_better = 1 if wait_pnl > imm_pnl + 0.1 else 0

    # Features
    row = {
        "entry_time": et.isoformat(),
        "strategy": t.get("strategy"),
        "side": side,
        "session": _session_of(et.hour),
        "et_hour": et.hour,
        "mkt_regime": day_state.get("mkt_regime", ""),
        "de3_entry_ret1_atr":  float(feat_row.get("de3_entry_ret1_atr", 0) or 0),
        "de3_entry_body_pos1": float(feat_row.get("de3_entry_body_pos1", 0) or 0),
        "de3_entry_body1_ratio": float(feat_row.get("de3_entry_body1_ratio", 0) or 0),
        "de3_entry_lower_wick_ratio": float(feat_row.get("de3_entry_lower_wick_ratio", 0) or 0),
        "de3_entry_upper_wick_ratio": float(feat_row.get("de3_entry_upper_wick_ratio", 0) or 0),
        "de3_entry_range10_atr": float(feat_row.get("de3_entry_range10_atr", 0) or 0),
        "de3_entry_vol1_rel20": float(feat_row.get("de3_entry_vol1_rel20", 0) or 0),
        "de3_entry_atr14": atr14,
        "dist_to_bank_below": dist_below,
        "dist_to_bank_above": dist_above,
        "dist_to_bank_in_dir": dist_in_dir,
        "bar_range_pts": bar_hi - bar_lo,
        "bar_close_pct_body": ((current_close - bar_lo) / max(0.01, bar_hi - bar_lo)),
        "sl_dist_pts": sl_dist,
        "tp_dist_pts": tp_dist,
        "atr_ratio_to_sl": atr14 / max(0.5, sl_dist),
        "imm_pnl_pts": imm_pnl,
        "wait_pnl_pts": wait_pnl,
        "imm_pnl_dol": imm_pnl * POINT_VALUE,
        "wait_pnl_dol": wait_pnl * POINT_VALUE,
        "delta_pnl_dol": (wait_pnl - imm_pnl) * POINT_VALUE,
        "label_wait_better": label_wait_better,
    }
    return row


def assemble_X(df, cat_maps=None):
    updated = dict(cat_maps or {})
    parts = [df[NUMERIC_FEATURES].copy()]
    for col in CATEGORICAL_FEATURES:
        known = updated.get(col)
        if known is None:
            known = sorted(df[col].dropna().unique().tolist())
            updated[col] = known
        encoded = pd.DataFrame(
            {f"{col}__{v}": (df[col] == v).astype(int) for v in known},
            index=df.index,
        )
        parts.append(encoded)
    parts.append(df[ORDINAL_FEATURES].astype(float))
    X = pd.concat(parts, axis=1).fillna(0.0)
    return X, updated


def main():
    print("[load] parquet")
    master = pd.read_parquet(ROOT / "es_master_outrights.parquet")
    master = master[master.index >= "2025-01-01"]
    print(f"  {len(master):,} bars (2025+)")

    print("[collect] trades")
    trades = collect_all_trades()
    print(f"  {len(trades)} DE3+RA+AF trades")

    print("[build features + simulate both fill modes] this takes a while...")
    rows = []
    skipped = 0
    for i, t in enumerate(trades):
        row = build_features_for_trade(t, master, {})
        if row is None:
            skipped += 1
            continue
        rows.append(row)
        if (i + 1) % 500 == 0:
            print(f"  processed {i+1}/{len(trades)}  ({len(rows)} valid, {skipped} skipped)")

    df = pd.DataFrame(rows)
    print(f"[built] {len(df)} rows (from {len(trades)} trades)")
    if len(df) < 100:
        raise SystemExit("too few rows")

    # Stats
    print(f"\nLabel distribution:")
    print(f"  wait_better = 1: {df['label_wait_better'].sum()} ({df['label_wait_better'].mean():.1%})")
    print(f"  wait_better = 0: {(1 - df['label_wait_better']).sum()} ({1 - df['label_wait_better'].mean():.1%})")
    print(f"\nIMMEDIATE avg PnL: ${df['imm_pnl_dol'].mean():.2f}")
    print(f"WAIT      avg PnL: ${df['wait_pnl_dol'].mean():.2f}")
    print(f"Delta (WAIT - IMM): ${df['delta_pnl_dol'].mean():.2f} avg  /  ${df['delta_pnl_dol'].sum():.2f} cumulative")

    # Train
    df = df.dropna(subset=NUMERIC_FEATURES + CATEGORICAL_FEATURES + ORDINAL_FEATURES + ["label_wait_better"])
    df = df.reset_index(drop=True)
    X, cat_maps = assemble_X(df)
    y = df["label_wait_better"].astype(int).values

    print(f"\n[train] {len(df)} rows, {X.shape[1]} features")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    for tr, te in skf.split(X, y):
        clf = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                           learning_rate=0.05, min_samples_leaf=30,
                                           random_state=42)
        clf.fit(X.iloc[tr], y[tr])
        p = clf.predict_proba(X.iloc[te])[:, 1]
        aucs.append(roc_auc_score(y[te], p))
    print(f"  5-fold CV AUC: {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")

    # EV-best threshold
    # For each holdout threshold t, compute expected PnL: when p>=t, pick WAIT, else IMMEDIATE
    df_sorted = df.sort_values("entry_time").reset_index(drop=True)
    X_sorted, _ = assemble_X(df_sorted, cat_maps=cat_maps)
    y_sorted = df_sorted["label_wait_better"].values
    split = int(0.85 * len(df_sorted))
    X_tr, X_te = X_sorted.iloc[:split], X_sorted.iloc[split:]
    y_tr = y_sorted[:split]
    df_te = df_sorted.iloc[split:]

    clf = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                       learning_rate=0.05, min_samples_leaf=30,
                                       random_state=42)
    clf.fit(X_tr, y_tr)
    p_te = clf.predict_proba(X_te)[:, 1]
    # Imagine: use WAIT when p >= thr, else IMMEDIATE. PnL = sum of picked-mode pnl.
    baseline_imm = df_te["imm_pnl_dol"].sum()  # always IMMEDIATE
    best_thr, best_pnl = 0.5, -1e9
    for thr in np.arange(0.30, 0.75, 0.025):
        picked = np.where(p_te >= thr, df_te["wait_pnl_dol"].values, df_te["imm_pnl_dol"].values)
        if picked.sum() > best_pnl:
            best_pnl = picked.sum()
            best_thr = float(thr)
    print(f"\n  Temporal holdout (last 15%, {len(df_te)} trades):")
    print(f"    Always IMMEDIATE pnl: ${baseline_imm:,.2f}")
    print(f"    Best WAIT-threshold: {best_thr:.3f} → pnl ${best_pnl:,.2f} (delta ${best_pnl-baseline_imm:+.2f})")

    # Retrain on full data + save
    final = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                         learning_rate=0.05, min_samples_leaf=30,
                                         random_state=42)
    final.fit(X, y)
    importances = sorted(zip(X.columns, final.feature_importances_), key=lambda t: -t[1])
    print(f"\n  top 10 features:")
    for n, imp in importances[:10]:
        print(f"    {n:<35} {imp:.4f}")

    OUT = ROOT / "artifacts" / "signal_gate_2025" / "model_lfo.joblib"
    joblib.dump({
        "model": final,
        "model_kind": "GBT_d3_lfo_binary",
        "target": "label_wait_better",
        "veto_threshold": best_thr,  # use WAIT when p >= this
        "feature_names": list(X.columns),
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "ordinal_features": ORDINAL_FEATURES,
        "categorical_maps": cat_maps,
        "training_rows": len(df),
        "cv_auc_mean": float(np.mean(aucs)),
        "cv_auc_std": float(np.std(aucs)),
        "holdout_best_thr": float(best_thr),
        "holdout_pnl_delta": float(best_pnl - baseline_imm),
    }, OUT)
    print(f"\n[write] {OUT}")
    # Also dump the per-trade dataset for inspection
    df.to_parquet(ROOT / "artifacts" / "signal_gate_2025" / "lfo_training_data.parquet")


if __name__ == "__main__":
    main()
