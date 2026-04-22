"""Train an ML replacement for the Percentage-Level Overlay bias classifier.

The rule-based PctLevelOverlay (pct_level_overlay.py) assigns a bias
("breakout_lean" | "pivot_lean" | "neutral" | "chop") whenever price touches
a %-from-session-open level, using a hand-tuned confluence score built from
ATR bucket, range bucket, hour-edge table, and per-level base edge.

This trainer learns the SAME bias classification from historical outcomes.

Training method:
  Walk the parquet minute-bars chronologically. Maintain the overlay's
  session-level running state. On each bar where overlay detects a fresh
  level touch (state.at_level=True, event="fresh_touch_*"), record:
    - Features: every state field at touch time + raw bar stats + rolling
      atr/range/hour features
    - Label: look forward horizon_minutes; did the price
        BREAKOUT  — extended by breakout_extension_pct (0.10%) past the
                    level in the direction dictated by sign of the level?
        PIVOT     — retraced by pivot_retrace_pct (0.15%) back before
                    breakout was hit?
        NEUTRAL   — neither condition fired within horizon_minutes

Label encoding (multi-class): 0=neutral, 1=breakout, 2=pivot

Output: artifacts/signal_gate_2025/model_pct_overlay.joblib
"""
from __future__ import annotations

import json
import sys
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
from pct_level_overlay import PctLevelOverlay, DEFAULT_OVERLAY_CONFIG

NY = ZoneInfo("America/New_York")

# Outcome horizon + thresholds come from the overlay's own defaults
OVERLAY_CFG = dict(DEFAULT_OVERLAY_CONFIG)
HORIZON_MINUTES = int(OVERLAY_CFG.get("horizon_minutes", 60))
BREAKOUT_PCT = float(OVERLAY_CFG.get("breakout_extension_pct", 0.10))
PIVOT_PCT = float(OVERLAY_CFG.get("pivot_retrace_pct", 0.15))

# Label encoding (binary — neutrals empirically never fire within 60min horizon)
LBL_PIVOT = 0    # price retraces back from level
LBL_BREAKOUT = 1 # price extends past level
LBL_NEUTRAL = 0  # unreachable; alias to PIVOT for safety

NUMERIC_FEATURES = [
    "pct_from_open",      # signed % from session open
    "signed_level",       # -3.00 to +3.00
    "abs_level",          # |signed_level|
    "level_distance_pct", # |pct - signed_level|
    "atr_pct_30bar",
    "range_pct_at_touch",
    "hour_edge",
    "minutes_since_open", # elapsed in trading day
    "dist_to_running_hi_pct",  # (price - running_hi) / price * 100
    "dist_to_running_lo_pct",
    "rule_confidence",     # the overlay's hand-tuned score (leak the teacher)
]
CATEGORICAL_FEATURES = ["tier", "atr_bucket", "range_bucket", "hour_bucket", "direction"]
ORDINAL_FEATURES: list[str] = []  # none — all encoded

PARQUET = ROOT / "es_master_outrights.parquet"


def hour_bucket(h: int) -> str:
    if 18 <= h or h < 3: return "ASIA"
    if 3 <= h < 7: return "LONDON"
    if 7 <= h < 9: return "NY_PRE"
    if 9 <= h < 12: return "NY_AM"
    if 12 <= h < 16: return "NY_PM"
    return "POST"


def load_bars(start_year: int = 2020) -> pd.DataFrame:
    """Load parquet, take front-month per day, sort."""
    df = pd.read_parquet(PARQUET)
    df = df[df.index.year >= start_year]
    # Pick the most-common symbol per day as front-month
    df = df.sort_index()
    # Deduplicate — keep the highest-volume symbol for each timestamp
    if "symbol" in df.columns and "volume" in df.columns:
        df = df.sort_values("volume", ascending=False).groupby(df.index).first().sort_index()
    elif "symbol" in df.columns:
        df = df[~df.index.duplicated(keep="first")]
    return df[["open", "high", "low", "close", "volume"]]


def generate_training_samples(bars_df: pd.DataFrame):
    """Walk bars, run overlay, capture touch events + future outcomes."""
    overlay = PctLevelOverlay(config=OVERLAY_CFG)
    # Keep last ts -> state dict for lookback
    samples = []
    # Keep index for future-lookup
    idx_arr = bars_df.index
    closes = bars_df["close"].values
    highs = bars_df["high"].values
    lows = bars_df["low"].values
    opens = bars_df["open"].values
    n = len(bars_df)

    last_tday = None
    session_open_val = None
    session_open_idx = None

    for i, ts in enumerate(idx_arr):
        o, h, l, c = opens[i], highs[i], lows[i], closes[i]
        ts_py = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
        state = overlay.update(ts_py, float(o), float(h), float(l), float(c))
        # Track session open for "minutes_since_open" feature
        if state.session_open is not None:
            tday = overlay._current_tday
            if tday != last_tday:
                last_tday = tday
                session_open_idx = i
            mins_since_open = (i - (session_open_idx or 0))
        else:
            mins_since_open = 0

        # Record training sample only on fresh touches
        if not state.at_level:
            continue
        if not (state.last_event or "").startswith("fresh_touch_"):
            continue

        # Future outcome (look forward up to HORIZON_MINUTES bars)
        # The relevant level is at state.signed_level (as a % from session_open)
        # Convert to price: level_price = session_open * (1 + signed_level/100)
        if state.session_open is None or state.nearest_level is None:
            continue
        op = float(state.session_open)
        signed_lvl = float(state.nearest_level)
        level_price = op * (1.0 + signed_lvl / 100.0)
        # Breakout threshold (extension past level in same direction as sign)
        # For positive level: breakout if price goes BREAKOUT_PCT higher
        # For negative level: breakout if price goes BREAKOUT_PCT lower
        if signed_lvl >= 0:
            breakout_target = op * (1.0 + (signed_lvl + BREAKOUT_PCT) / 100.0)
            pivot_target    = op * (1.0 + (signed_lvl - PIVOT_PCT) / 100.0)
            breakout_dir = 1  # up
        else:
            breakout_target = op * (1.0 + (signed_lvl - BREAKOUT_PCT) / 100.0)
            pivot_target    = op * (1.0 + (signed_lvl + PIVOT_PCT) / 100.0)
            breakout_dir = -1  # down

        # Walk forward up to HORIZON_MINUTES bars to find first outcome
        end = min(n, i + 1 + HORIZON_MINUTES)
        label = LBL_NEUTRAL
        for j in range(i + 1, end):
            hj, lj = highs[j], lows[j]
            # Break out?
            if breakout_dir == 1 and hj >= breakout_target:
                label = LBL_BREAKOUT; break
            if breakout_dir == -1 and lj <= breakout_target:
                label = LBL_BREAKOUT; break
            # Pivot?
            if breakout_dir == 1 and lj <= pivot_target:
                label = LBL_PIVOT; break
            if breakout_dir == -1 and hj >= pivot_target:
                label = LBL_PIVOT; break

        dist_to_hi_pct = (float(c) - overlay._running_hi) / max(1.0, float(c)) * 100.0
        dist_to_lo_pct = (float(c) - overlay._running_lo) / max(1.0, float(c)) * 100.0

        et = ts_py.astimezone(NY) if ts_py.tzinfo else ts_py
        samples.append({
            "ts": ts_py.isoformat(),
            "pct_from_open": state.pct_from_open,
            "signed_level": signed_lvl,
            "abs_level": abs(signed_lvl),
            "level_distance_pct": state.level_distance_pct,
            "atr_pct_30bar": state.atr_pct_30bar,
            "range_pct_at_touch": state.range_pct_at_touch,
            "hour_edge": state.hour_edge,
            "minutes_since_open": mins_since_open,
            "dist_to_running_hi_pct": dist_to_hi_pct,
            "dist_to_running_lo_pct": dist_to_lo_pct,
            "rule_confidence": state.confidence,
            "tier": state.tier,
            "atr_bucket": state.atr_bucket,
            "range_bucket": state.range_bucket,
            "hour_bucket": hour_bucket(et.hour if hasattr(et, "hour") else 0),
            "direction": "up" if signed_lvl >= 0 else "down",
            "rule_bias": state.bias,
            "label": label,
        })
    return samples


def assemble_X(df, cat_maps=None):
    updated = dict(cat_maps or {})
    parts = [df[NUMERIC_FEATURES].copy()]
    for col in CATEGORICAL_FEATURES:
        known = updated.get(col)
        if known is None:
            known = sorted(df[col].dropna().unique().tolist())
            updated[col] = known
        encoded = pd.DataFrame(
            {f"{col}__{v}": (df[col] == v).astype(int) for v in known}, index=df.index
        )
        parts.append(encoded)
    X = pd.concat(parts, axis=1).fillna(0.0)
    return X, updated


def main():
    print("[load] parquet")
    df = load_bars(start_year=2020)
    print(f"  {len(df):,} bars (2020+)")

    print("[walk bars + overlay] this is the slow step (single pass)")
    samples = generate_training_samples(df)
    print(f"[samples] {len(samples)} level-touch events")

    ds = pd.DataFrame(samples)
    print(f"\nLabel distribution:")
    from collections import Counter
    c = Counter(ds["label"].tolist())
    for lbl, name in [(0, "neutral"), (1, "breakout"), (2, "pivot")]:
        print(f"  {lbl} ({name:<10}) {c.get(lbl, 0):>6}  ({c.get(lbl, 0)/len(ds):.1%})")
    print(f"\nRule classifier distribution:")
    print(ds["rule_bias"].value_counts().to_dict())
    # Agreement between rule and outcome
    ds["rule_lbl"] = ds["rule_bias"].map(
        {"breakout_lean": 1, "pivot_lean": 2, "neutral": 0, "chop": 0}
    )
    rule_acc = (ds["rule_lbl"] == ds["label"]).mean()
    print(f"  rule accuracy vs outcome: {rule_acc:.1%}")

    # Train
    ds = ds.dropna(subset=NUMERIC_FEATURES + CATEGORICAL_FEATURES).reset_index(drop=True)
    # Ensure binary: if any LBL_NEUTRAL=0 survived, merge with pivot (already 0)
    X, cat_maps = assemble_X(ds)
    y = ds["label"].astype(int).values
    print(f"\n[train] {len(ds)} rows, {X.shape[1]} features, binary classification")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_aucs = []
    for tr, te in skf.split(X, y):
        clf = GradientBoostingClassifier(
            n_estimators=250, max_depth=4, learning_rate=0.05,
            min_samples_leaf=50, random_state=42,
        )
        clf.fit(X.iloc[tr], y[tr])
        p_bo = clf.predict_proba(X.iloc[te])[:, list(clf.classes_).index(LBL_BREAKOUT)]
        auc = roc_auc_score(y[te], p_bo)  # y is 0/1, p_bo is P(breakout)
        fold_aucs.append(auc)
    print(f"  5-fold CV AUC (breakout vs pivot): {np.mean(fold_aucs):.3f} ± {np.std(fold_aucs):.3f}")

    # Temporal holdout
    ds_sorted = ds.sort_values("ts").reset_index(drop=True)
    X_sorted, _ = assemble_X(ds_sorted, cat_maps=cat_maps)
    y_sorted = ds_sorted["label"].astype(int).values
    split = int(0.85 * len(ds_sorted))
    X_tr, X_te = X_sorted.iloc[:split], X_sorted.iloc[split:]
    y_tr, y_te = y_sorted[:split], y_sorted[split:]
    clf = GradientBoostingClassifier(
        n_estimators=250, max_depth=4, learning_rate=0.05,
        min_samples_leaf=50, random_state=42,
    )
    clf.fit(X_tr, y_tr)
    p_te = clf.predict_proba(X_te)
    # Map column index back to class label via clf.classes_
    pred_te = clf.classes_[p_te.argmax(axis=1)]
    acc_te = (pred_te == y_te).mean()
    # Rule acc on same tail (comparing rule's bias label to actual outcome)
    # Map rule_lbl as stored: {breakout_lean:1, pivot_lean:2, neutral:0, chop:0}
    # But we're now binary 0=pivot 1=breakout. So: rule says breakout_lean→1, else→0 (pivot)
    rule_bias = ds_sorted.iloc[split:]["rule_bias"].values
    rule_pred = np.array([1 if b == "breakout_lean" else 0 for b in rule_bias])
    rule_acc_tail = (rule_pred == y_te).mean()
    # AUC on holdout too
    p_bo = clf.predict_proba(X_te)[:, list(clf.classes_).index(LBL_BREAKOUT)]
    ho_auc = roc_auc_score(y_te, p_bo)
    print(f"  Temporal-tail (last 15%, {len(y_te)} events):")
    print(f"    ML accuracy:   {acc_te:.1%}")
    print(f"    Rule accuracy: {rule_acc_tail:.1%}")
    print(f"    ML holdout AUC: {ho_auc:.3f}")
    print(f"    Improvement:   {(acc_te-rule_acc_tail)*100:+.1f} pp accuracy")

    # Retrain full + save
    final = GradientBoostingClassifier(
        n_estimators=250, max_depth=4, learning_rate=0.05,
        min_samples_leaf=50, random_state=42,
    )
    final.fit(X, y)
    imps = sorted(zip(X.columns, final.feature_importances_), key=lambda t: -t[1])
    print(f"\n  top 10 features:")
    for n_, imp in imps[:10]:
        print(f"    {n_:<35} {imp:.4f}")

    OUT = ROOT / "artifacts" / "signal_gate_2025" / "model_pct_overlay.joblib"
    joblib.dump({
        "model": final,
        "model_kind": "GBT_d4_pct_overlay_3class",
        "classes": {0: "neutral", 1: "breakout", 2: "pivot"},
        "feature_names": list(X.columns),
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "categorical_maps": cat_maps,
        "training_rows": len(ds),
        "cv_auc_mean": float(np.mean(fold_aucs)),
        "cv_auc_std": float(np.std(fold_aucs)),
        "holdout_acc": float(acc_te),
        "rule_baseline_acc": float(rule_acc_tail),
        "horizon_minutes": HORIZON_MINUTES,
        "breakout_pct": BREAKOUT_PCT,
        "pivot_pct": PIVOT_PCT,
    }, OUT)
    ds.to_parquet(ROOT / "artifacts" / "signal_gate_2025" / "pct_overlay_training_data.parquet")
    print(f"\n[write] {OUT}")


if __name__ == "__main__":
    main()
