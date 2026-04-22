"""Train an ML model that predicts whether a confirmed swing pivot will HOLD.

The live bot has a rule-based pivot trail:
  - _detect_pivot_high / _detect_pivot_low scan the last 5 bars; the center
    bar is confirmed as a swing pivot if it's the window extreme.
  - _compute_pivot_trail_sl then ratchets the SL to one bank behind (Reading B)
    or at the pivot bank (Reading C), gated only on "min profit pts >= 12.5".

The rule has no concept of whether the pivot is likely to hold. This trainer
learns from historical bars: given a pivot just confirmed, will price respect
it (stay above Reading-C anchor for a pivot_high, below for a pivot_low)
for the next HORIZON_BARS?

Label (binary):
  1 = "held"    — forward window never traded through anchor - BUFFER (LONG)
                 or anchor + BUFFER (SHORT)
  0 = "broke"   — forward window violated the anchor within horizon

At inference we plug this into ml_overlay_shadow.score_pivot_trail(...) so the
bot can (shadow) compare the rule's "just ratchet" vs ML's "only ratchet when
P(hold) >= threshold".

Output: artifacts/signal_gate_2025/model_pivot_trail.joblib
"""
from __future__ import annotations

import math
import sys
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

NY = ZoneInfo("America/New_York")

# --- Live-loop constants (must match julie001.py) ---
PIVOT_LOOKBACK = 5            # _PIVOT_TRAIL_LOOKBACK
BANK_STEP = 12.5              # _BANK_FILL_STEP (12.5 pts per bank level)
BUFFER = 0.25                 # _PIVOT_TRAIL_BUFFER (1 tick clearance)

# --- Training knobs ---
HORIZON_BARS = 20             # 20 min forward window (typical trade duration)
MIN_YEAR = 2020
MAX_SAMPLES = 200_000         # downsample for tractable training (stratified)

# Features
NUMERIC_FEATURES = [
    # Pivot bar structure
    "pivot_range_pts",        # bar high - bar low
    "pivot_body_pts",         # |close - open|
    "upper_wick_pct",         # (high - max(o,c)) / range
    "lower_wick_pct",
    "pivot_height_pts",       # pivot extreme vs window mean
    # Tape context
    "atr14_pts",
    "range_30bar_pts",
    "trend_20bar_pct",        # (close - close_20) / close * 100
    "dist_to_20bar_hi_pct",
    "dist_to_20bar_lo_pct",
    # Bank-level context (how close the pivot is to grid)
    "dist_pivot_to_bank_pts",
    "anchor_distance_from_entry_pts",  # set to 0 when walking cold (trade-state neutral)
    # Momentum leading in
    "vel_5bar_pts_per_min",
    "vel_20bar_pts_per_min",
    # Reading-B vs Reading-C gap (structural)
    "reading_b_buffer_pts",
]
CATEGORICAL_FEATURES = [
    "pivot_type",             # "HIGH" / "LOW"
    "session",                # ASIA/LONDON/NY_PRE/NY_AM/NY_PM/POST
    "tape",                   # uptrend/downtrend/chop (20-bar classify)
]
ORDINAL_FEATURES = ["et_hour"]

PARQUET = ROOT / "es_master_outrights.parquet"
OUT_MODEL = ROOT / "artifacts" / "signal_gate_2025" / "model_pivot_trail.joblib"
OUT_DATA = ROOT / "artifacts" / "signal_gate_2025" / "pivot_trail_training_data.parquet"


def session_bucket(h: int) -> str:
    if 18 <= h or h < 3: return "ASIA"
    if 3 <= h < 7: return "LONDON"
    if 7 <= h < 9: return "NY_PRE"
    if 9 <= h < 12: return "NY_AM"
    if 12 <= h < 16: return "NY_PM"
    return "POST"


def load_bars(start_year: int = MIN_YEAR) -> pd.DataFrame:
    df = pd.read_parquet(PARQUET)
    df = df[df.index.year >= start_year]
    df = df.sort_index()
    if "symbol" in df.columns and "volume" in df.columns:
        df = df.sort_values("volume", ascending=False).groupby(df.index).first().sort_index()
    elif "symbol" in df.columns:
        df = df[~df.index.duplicated(keep="first")]
    return df[["open", "high", "low", "close", "volume"]]


def detect_pivots(highs: np.ndarray, lows: np.ndarray, idx: int, lookback: int = PIVOT_LOOKBACK):
    """Mirror _detect_pivot_high / _detect_pivot_low. Returns (type, pivot_price)
    for the bar at position (idx - lookback//2), or (None, None) if no pivot."""
    if idx < lookback - 1:
        return None, None
    half = lookback // 2
    window_highs = highs[idx - lookback + 1: idx + 1]
    window_lows = lows[idx - lookback + 1: idx + 1]
    # Center bar of window
    ph = window_highs[half]
    pl = window_lows[half]
    is_ph = all(ph >= window_highs[i] - 1e-9 for i in range(lookback) if i != half)
    is_pl = all(pl <= window_lows[i] + 1e-9 for i in range(lookback) if i != half)
    # If both (degenerate), prefer highs (arbitrary; we treat symmetrically below)
    if is_ph and not is_pl:
        return "HIGH", float(ph)
    if is_pl and not is_ph:
        return "LOW", float(pl)
    if is_ph and is_pl:
        # Doji flat — skip; not a useful structural pivot
        return None, None
    return None, None


def atr_over(highs, lows, closes, end_idx, n=14):
    if end_idx < n:
        return 0.0
    trs = []
    for i in range(end_idx - n + 1, end_idx + 1):
        h = highs[i]; l = lows[i]; cp = closes[i - 1] if i > 0 else closes[i]
        trs.append(max(h - l, abs(h - cp), abs(l - cp)))
    return float(np.mean(trs)) if trs else 0.0


def classify_tape(closes: np.ndarray, end_idx: int, n: int = 20) -> str:
    if end_idx < n:
        return "chop"
    start = closes[end_idx - n]
    now = closes[end_idx]
    rng = closes[end_idx - n: end_idx + 1]
    pct = (now - start) / max(1.0, start) * 100.0
    span = (rng.max() - rng.min()) / max(1.0, start) * 100.0
    if span < 0.15:
        return "chop"
    if pct >= 0.15:
        return "uptrend"
    if pct <= -0.15:
        return "downtrend"
    return "chop"


def generate_training_samples(df: pd.DataFrame):
    """Walk bars and emit one row per confirmed pivot, with forward-looking label."""
    idx_arr = df.index
    opens = df["open"].values.astype(float)
    highs = df["high"].values.astype(float)
    lows = df["low"].values.astype(float)
    closes = df["close"].values.astype(float)
    n = len(df)
    samples = []

    # Track timestamps as python objects only when pivot fires
    half = PIVOT_LOOKBACK // 2

    last_pivot_high_idx = -1
    last_pivot_low_idx = -1

    for i in range(PIVOT_LOOKBACK - 1, n - HORIZON_BARS - 1):
        ptype, ppx = detect_pivots(highs, lows, i, PIVOT_LOOKBACK)
        if ptype is None:
            continue
        pivot_bar_idx = i - half  # center bar
        # Features
        pbar_o = opens[pivot_bar_idx]
        pbar_h = highs[pivot_bar_idx]
        pbar_l = lows[pivot_bar_idx]
        pbar_c = closes[pivot_bar_idx]
        pivot_range = max(1e-9, pbar_h - pbar_l)
        upper_wick = (pbar_h - max(pbar_o, pbar_c)) / pivot_range
        lower_wick = (min(pbar_o, pbar_c) - pbar_l) / pivot_range
        pivot_body = abs(pbar_c - pbar_o)
        window_start = i - PIVOT_LOOKBACK + 1
        window_mean = float(np.mean(closes[window_start: i + 1]))
        pivot_height = ppx - window_mean if ptype == "HIGH" else window_mean - ppx

        atr14 = atr_over(highs, lows, closes, i, 14)
        range30 = float(highs[max(0, i - 29): i + 1].max() - lows[max(0, i - 29): i + 1].min())
        if i >= 20:
            trend20 = (closes[i] - closes[i - 20]) / max(1.0, closes[i - 20]) * 100.0
        else:
            trend20 = 0.0
        if i >= 20:
            hi20 = float(highs[i - 19: i + 1].max())
            lo20 = float(lows[i - 19: i + 1].min())
            dist_hi = (closes[i] - hi20) / max(1.0, closes[i]) * 100.0
            dist_lo = (closes[i] - lo20) / max(1.0, closes[i]) * 100.0
        else:
            dist_hi = 0.0
            dist_lo = 0.0

        # Anchor = Reading-C bank level for the pivot
        if ptype == "HIGH":
            anchor_c = math.floor(ppx / BANK_STEP) * BANK_STEP
            anchor_b = anchor_c - BANK_STEP
        else:
            anchor_c = math.ceil(ppx / BANK_STEP) * BANK_STEP
            anchor_b = anchor_c + BANK_STEP
        dist_pivot_to_bank = abs(ppx - anchor_c)
        reading_b_buffer = abs(anchor_c - anchor_b)

        # 5-bar and 20-bar velocities
        if i >= 5:
            vel5 = (closes[i] - closes[i - 5]) / 5.0  # pts/min on 1m bars
        else:
            vel5 = 0.0
        if i >= 20:
            vel20 = (closes[i] - closes[i - 20]) / 20.0
        else:
            vel20 = 0.0

        # Timestamp + session
        ts = idx_arr[i].to_pydatetime() if hasattr(idx_arr[i], "to_pydatetime") else idx_arr[i]
        et = ts.astimezone(NY) if ts.tzinfo else ts
        et_hour = int(et.hour if hasattr(et, "hour") else 12)
        session = session_bucket(et_hour)
        tape = classify_tape(closes, i, n=20)

        # Forward outcome: did the pivot hold for HORIZON_BARS?
        # Threshold matches the live rule's preferred trail (Reading B =
        # anchor_b - BUFFER for HIGH, anchor_b + BUFFER for LOW).
        #
        # Path A label: wicks don't count. A pivot "broke" only when TWO
        # CONSECUTIVE bar CLOSES sit on the wrong side of threshold. That
        # filters intrabar noise and keeps the label focused on structural
        # break-and-hold moves (which is what actually stops a trade).
        end = min(n, i + 1 + HORIZON_BARS)
        held = True
        violation_bar = None
        CONFIRM = 2   # bars of consecutive closes through threshold
        if ptype == "HIGH":
            threshold = anchor_b - BUFFER   # LONG trail sits here
            streak = 0
            for j in range(i + 1, end):
                if closes[j] <= threshold:
                    streak += 1
                    if streak >= CONFIRM:
                        held = False
                        violation_bar = j - i
                        break
                else:
                    streak = 0
        else:
            threshold = anchor_b + BUFFER   # SHORT trail sits here
            streak = 0
            for j in range(i + 1, end):
                if closes[j] >= threshold:
                    streak += 1
                    if streak >= CONFIRM:
                        held = False
                        violation_bar = j - i
                        break
                else:
                    streak = 0

        samples.append({
            "ts": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
            "pivot_type": ptype,
            "pivot_price": float(ppx),
            "anchor_c": float(anchor_c),
            "anchor_b": float(anchor_b),
            "pivot_range_pts": float(pivot_range),
            "pivot_body_pts": float(pivot_body),
            "upper_wick_pct": float(upper_wick),
            "lower_wick_pct": float(lower_wick),
            "pivot_height_pts": float(pivot_height),
            "atr14_pts": float(atr14),
            "range_30bar_pts": float(range30),
            "trend_20bar_pct": float(trend20),
            "dist_to_20bar_hi_pct": float(dist_hi),
            "dist_to_20bar_lo_pct": float(dist_lo),
            "dist_pivot_to_bank_pts": float(dist_pivot_to_bank),
            "anchor_distance_from_entry_pts": 0.0,  # placeholder (trade-state neutral)
            "vel_5bar_pts_per_min": float(vel5),
            "vel_20bar_pts_per_min": float(vel20),
            "reading_b_buffer_pts": float(reading_b_buffer),
            "session": session,
            "tape": tape,
            "et_hour": float(et_hour),
            "held": int(held),
            "violation_bar": int(violation_bar) if violation_bar is not None else -1,
        })

    return samples


def assemble_X(df, cat_maps=None):
    updated = dict(cat_maps or {})
    parts = [df[NUMERIC_FEATURES].copy()]
    # ordinals as plain numeric
    if ORDINAL_FEATURES:
        parts.append(df[ORDINAL_FEATURES].copy())
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
    df = load_bars(start_year=MIN_YEAR)
    print(f"  {len(df):,} bars ({MIN_YEAR}+)")

    print("[walk bars] detecting pivots + labelling forward outcomes")
    samples = generate_training_samples(df)
    print(f"[samples] {len(samples):,} confirmed pivots")
    if not samples:
        print("ERROR: zero samples — check detector / parquet")
        return

    ds = pd.DataFrame(samples)
    print("\nLabel distribution:")
    hold_ct = int(ds["held"].sum())
    broke_ct = int((ds["held"] == 0).sum())
    print(f"  1 (held)   {hold_ct:>7}  ({hold_ct/len(ds):.1%})")
    print(f"  0 (broke)  {broke_ct:>7}  ({broke_ct/len(ds):.1%})")
    print("\nPivot type distribution:")
    print(ds["pivot_type"].value_counts().to_dict())

    # Rule baseline — rule has NO "will-it-hold" signal, it always ratchets. So
    # "rule accuracy" is just the base rate of hold events (what rule gets when
    # it trails on every pivot). We compute it for the holdout tail below.

    ds = ds.dropna(subset=NUMERIC_FEATURES + CATEGORICAL_FEATURES + ORDINAL_FEATURES).reset_index(drop=True)

    # Stratified downsample for tractable training speed — preserve class ratio
    if MAX_SAMPLES is not None and len(ds) > MAX_SAMPLES:
        rng = np.random.RandomState(42)
        ds["_rand"] = rng.rand(len(ds))
        # Stratified: sample proportionally from each class
        parts = []
        for cls in [0, 1]:
            sub = ds[ds["held"] == cls].copy()
            n_keep = int(MAX_SAMPLES * len(sub) / len(ds))
            parts.append(sub.nsmallest(n_keep, "_rand"))
        ds = pd.concat(parts).drop(columns=["_rand"]).reset_index(drop=True)
        print(f"[downsample] stratified to {len(ds):,} rows (kept ratio: held={ds['held'].mean():.1%})")

    X, cat_maps = assemble_X(ds)
    y = ds["held"].astype(int).values
    print(f"\n[train] {len(ds):,} rows, {X.shape[1]} features, binary classification")

    # CV AUC
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_aucs = []
    for tr, te in skf.split(X, y):
        clf = GradientBoostingClassifier(
            n_estimators=250, max_depth=4, learning_rate=0.05,
            min_samples_leaf=50, random_state=42,
        )
        clf.fit(X.iloc[tr], y[tr])
        p = clf.predict_proba(X.iloc[te])[:, list(clf.classes_).index(1)]
        fold_aucs.append(roc_auc_score(y[te], p))
    print(f"  5-fold CV AUC (held vs broke): {np.mean(fold_aucs):.3f} ± {np.std(fold_aucs):.3f}")

    # Temporal holdout
    ds_sorted = ds.sort_values("ts").reset_index(drop=True)
    X_sorted, _ = assemble_X(ds_sorted, cat_maps=cat_maps)
    y_sorted = ds_sorted["held"].astype(int).values
    split = int(0.85 * len(ds_sorted))
    X_tr, X_te = X_sorted.iloc[:split], X_sorted.iloc[split:]
    y_tr, y_te = y_sorted[:split], y_sorted[split:]
    clf = GradientBoostingClassifier(
        n_estimators=250, max_depth=4, learning_rate=0.05,
        min_samples_leaf=50, random_state=42,
    )
    clf.fit(X_tr, y_tr)
    p_te = clf.predict_proba(X_te)[:, list(clf.classes_).index(1)]
    ho_auc = roc_auc_score(y_te, p_te)
    # ML accuracy at common thresholds
    rule_acc = float(y_te.mean())  # rule always trails → "predicts" held for all
    print(f"  Temporal-tail (last 15%, {len(y_te):,} pivots):")
    print(f"    Rule (always-ratchet) base rate held: {rule_acc:.1%}")
    print(f"    ML holdout AUC: {ho_auc:.3f}")
    for thr in (0.45, 0.50, 0.55, 0.60, 0.65):
        pred = (p_te >= thr).astype(int)
        # Where ML says "hold" (ratchet), how often was it actually held?
        ratcheted = pred == 1
        precision = float(y_te[ratcheted].mean()) if ratcheted.any() else float("nan")
        recall = float(y_te[(y_te == 1)].size and (pred[y_te == 1] == 1).mean())
        print(f"    thr={thr:.2f}: precision={precision:.1%} recall={recall:.1%} ratchets={ratcheted.mean():.1%}")

    # Retrain full + save
    final = GradientBoostingClassifier(
        n_estimators=250, max_depth=4, learning_rate=0.05,
        min_samples_leaf=50, random_state=42,
    )
    final.fit(X, y)
    imps = sorted(zip(X.columns, final.feature_importances_), key=lambda t: -t[1])
    print("\n  top 10 features:")
    for n_, imp in imps[:10]:
        print(f"    {n_:<35} {imp:.4f}")

    joblib.dump({
        "model": final,
        "model_kind": "GBT_d4_pivot_trail_binary",
        "classes": {0: "broke", 1: "held"},
        "feature_names": list(X.columns),
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "ordinal_features": ORDINAL_FEATURES,
        "categorical_maps": cat_maps,
        "training_rows": len(ds),
        "cv_auc_mean": float(np.mean(fold_aucs)),
        "cv_auc_std": float(np.std(fold_aucs)),
        "holdout_auc": float(ho_auc),
        "rule_base_rate_held": float(rule_acc),
        "horizon_bars": HORIZON_BARS,
        "pivot_lookback": PIVOT_LOOKBACK,
        "bank_step": BANK_STEP,
        "buffer": BUFFER,
        "hold_threshold": 0.55,  # default inference threshold (tunable)
    }, OUT_MODEL)
    ds.to_parquet(OUT_DATA)
    print(f"\n[write] {OUT_MODEL}")
    print(f"[write] {OUT_DATA}")


if __name__ == "__main__":
    main()
