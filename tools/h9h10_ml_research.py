#!/usr/bin/env python3
"""H9/H10 ML strategy research pipeline.

End-to-end:
  1. Load es_master_outrights parquet, build continuous front-month series.
  2. Filter to NY-local hours 9 and 10 (cash-session morning).
  3. For each candidate entry bar (09:30 and 10:00 ET on RTH days), build
     no-lookahead features from the trailing window.
  4. Build TP-first vs SL-first labels via path simulation with ATR-scaled
     brackets.
  5. Train HistGradientBoosting classifiers (long + short) with sequential
     60/20/20 train/val/test split.
  6. Random-control validation: shuffle entry timestamps within the same
     hour-window, retrain, confirm signal beats noise.
  7. Backtest with bracket simulation; report PnL by period (Trump 1 / Biden
     / Trump 2) and overall metrics (WR, avg win, max DD, $/year).

Output: artifacts/h9h10_ml/
  - feature_table.parquet
  - labels.parquet
  - models.joblib
  - results.json
  - per_trade.csv
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
PARQUET = ROOT / "es_master_outrights-2.parquet"
OUT_DIR = ROOT / "artifacts" / "h9h10_ml"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Roll calendar (front-month) — copied from simulator_trade_through.py to
# keep this script standalone.
# ---------------------------------------------------------------------------
ROLL_CALENDAR: List[Tuple[pd.Timestamp, str]] = [
    (pd.Timestamp("2010-12-01"), "ESH1"),
    (pd.Timestamp("2011-03-11"), "ESM1"),
    (pd.Timestamp("2011-06-10"), "ESU1"),
    (pd.Timestamp("2011-09-09"), "ESZ1"),
    (pd.Timestamp("2011-12-09"), "ESH2"),
    (pd.Timestamp("2012-03-09"), "ESM2"),
    (pd.Timestamp("2012-06-08"), "ESU2"),
    (pd.Timestamp("2012-09-14"), "ESZ2"),
    (pd.Timestamp("2012-12-14"), "ESH3"),
    (pd.Timestamp("2013-03-08"), "ESM3"),
    (pd.Timestamp("2013-06-14"), "ESU3"),
    (pd.Timestamp("2013-09-13"), "ESZ3"),
    (pd.Timestamp("2013-12-13"), "ESH4"),
    (pd.Timestamp("2014-03-14"), "ESM4"),
    (pd.Timestamp("2014-06-13"), "ESU4"),
    (pd.Timestamp("2014-09-12"), "ESZ4"),
    (pd.Timestamp("2014-12-12"), "ESH5"),
    (pd.Timestamp("2015-03-13"), "ESM5"),
    (pd.Timestamp("2015-06-12"), "ESU5"),
    (pd.Timestamp("2015-09-11"), "ESZ5"),
    (pd.Timestamp("2015-12-11"), "ESH6"),
    (pd.Timestamp("2016-03-11"), "ESM6"),
    (pd.Timestamp("2016-06-10"), "ESU6"),
    (pd.Timestamp("2016-09-09"), "ESZ6"),
    (pd.Timestamp("2016-12-09"), "ESH7"),
    (pd.Timestamp("2017-03-10"), "ESM7"),
    (pd.Timestamp("2017-06-09"), "ESU7"),
    (pd.Timestamp("2017-09-08"), "ESZ7"),
    (pd.Timestamp("2017-12-08"), "ESH8"),
    (pd.Timestamp("2018-03-09"), "ESM8"),
    (pd.Timestamp("2018-06-08"), "ESU8"),
    (pd.Timestamp("2018-09-14"), "ESZ8"),
    (pd.Timestamp("2018-12-14"), "ESH9"),
    (pd.Timestamp("2019-03-08"), "ESM9"),
    (pd.Timestamp("2019-06-14"), "ESU9"),
    (pd.Timestamp("2019-09-13"), "ESZ9"),
    (pd.Timestamp("2019-12-13"), "ESH0"),
    (pd.Timestamp("2020-03-13"), "ESM0"),
    (pd.Timestamp("2020-06-12"), "ESU0"),
    (pd.Timestamp("2020-09-11"), "ESZ0"),
    (pd.Timestamp("2020-12-11"), "ESH1"),  # 2021 contract
    (pd.Timestamp("2021-03-12"), "ESM1"),
    (pd.Timestamp("2021-06-11"), "ESU1"),
    (pd.Timestamp("2021-09-10"), "ESZ1"),
    (pd.Timestamp("2021-12-10"), "ESH2"),
    (pd.Timestamp("2022-03-11"), "ESM2"),
    (pd.Timestamp("2022-06-10"), "ESU2"),
    (pd.Timestamp("2022-09-09"), "ESZ2"),
    (pd.Timestamp("2022-12-09"), "ESH3"),
    (pd.Timestamp("2023-03-10"), "ESM3"),
    (pd.Timestamp("2023-06-09"), "ESU3"),
    (pd.Timestamp("2023-09-08"), "ESZ3"),
    (pd.Timestamp("2023-12-08"), "ESH4"),
    (pd.Timestamp("2024-03-08"), "ESM4"),
    (pd.Timestamp("2024-06-14"), "ESU4"),
    (pd.Timestamp("2024-09-13"), "ESZ4"),
    (pd.Timestamp("2024-12-13"), "ESH5"),
    (pd.Timestamp("2025-03-14"), "ESM5"),
    (pd.Timestamp("2025-06-13"), "ESU5"),
    (pd.Timestamp("2025-09-12"), "ESZ5"),
    (pd.Timestamp("2025-12-12"), "ESH6"),
    (pd.Timestamp("2026-03-12"), "ESM6"),
    (pd.Timestamp("2026-06-11"), "ESU6"),
    (pd.Timestamp("2026-09-10"), "ESZ6"),
    (pd.Timestamp("2026-12-10"), "ESH7"),
]


def front_month_by_calendar(ts: pd.Timestamp) -> str:
    if ts.tzinfo is not None:
        ts_naive = ts.tz_convert("UTC").tz_localize(None)
    else:
        ts_naive = ts
    best = ROLL_CALENDAR[0][1]
    for cut, sym in ROLL_CALENDAR:
        if ts_naive >= cut:
            best = sym
        else:
            break
    return best


def label_period(ts: pd.Timestamp) -> str:
    """Trump 1 / Biden / Trump 2 era partitioning."""
    if ts < pd.Timestamp("2017-01-20", tz=ts.tz):
        return "pre_t1"
    if ts < pd.Timestamp("2021-01-20", tz=ts.tz):
        return "trump1"
    if ts < pd.Timestamp("2025-01-20", tz=ts.tz):
        return "biden"
    return "trump2"


# ---------------------------------------------------------------------------
# 1. Continuous front-month
# ---------------------------------------------------------------------------
def build_front_month_series(parquet_path: Path = PARQUET) -> pd.DataFrame:
    """Load parquet, pick front-month per row by ROLL_CALENDAR, return
    a single OHLCV frame with one row per minute."""
    print(f"[1/8] loading {parquet_path.name} ...")
    df = pd.read_parquet(parquet_path)
    df = df.sort_index()
    # Build a per-row "expected_symbol" via vectorized roll lookup.
    print(f"[1/8] {len(df):,} rows; mapping front-month per timestamp ...")
    # Use searchsorted on ROLL_CALENDAR cuts to find front for each ts.
    # Convert cuts to numpy datetime64[ns] (UTC-naive scalar form).
    cuts_ts = pd.DatetimeIndex(
        [c.tz_localize("UTC") if c.tzinfo is None else c.tz_convert("UTC")
         for c, _ in ROLL_CALENDAR]
    ).tz_convert(None)
    cuts_np = cuts_ts.values  # datetime64[ns]
    syms = np.array([s for _, s in ROLL_CALENDAR])
    idx = df.index
    if idx.tz is not None:
        ts_utc = idx.tz_convert("UTC").tz_convert(None)
    else:
        ts_utc = idx
    pos = np.searchsorted(cuts_np, ts_utc.values, side="right") - 1
    pos = np.clip(pos, 0, len(syms) - 1)
    df["front_symbol"] = syms[pos]
    front = df[df["symbol"] == df["front_symbol"]].copy()
    # If multiple rows per timestamp survived (shouldn't), keep first.
    front = front[~front.index.duplicated(keep="first")]
    print(f"[1/8] front-month frame: {len(front):,} bars  "
          f"({front.index.min()} → {front.index.max()})")
    return front[["open", "high", "low", "close", "volume", "symbol"]]


# ---------------------------------------------------------------------------
# 2. Build entry candidates at hour 9 (09:30 ET RTH open) and hour 10 (10:00).
# ---------------------------------------------------------------------------
def build_entry_candidates(front: pd.DataFrame) -> pd.DataFrame:
    """Return one row per (date, hour) entry candidate with the entry-bar
    open price. Hour 9 = 09:30 ET (RTH open), hour 10 = 10:00 ET."""
    print("[2/8] building hour-9 / hour-10 entry candidates ...")
    df = front.copy()
    df["minute"] = df.index.minute
    df["hour_ny"] = df.index.hour
    h9 = df[(df["hour_ny"] == 9) & (df["minute"] == 30)].copy()
    h9["window"] = "h9"
    h10 = df[(df["hour_ny"] == 10) & (df["minute"] == 0)].copy()
    h10["window"] = "h10"
    cands = pd.concat([h9, h10]).sort_index()
    # Filter to weekday RTH days
    cands = cands[cands.index.dayofweek < 5]
    print(f"[2/8] {len(cands):,} entry candidates "
          f"(h9={len(h9):,}, h10={len(h10):,})")
    return cands


# ---------------------------------------------------------------------------
# 3. No-lookahead features.
# ---------------------------------------------------------------------------
def compute_features(front: pd.DataFrame, cands: pd.DataFrame) -> pd.DataFrame:
    """For each candidate, compute features using bars STRICTLY BEFORE the
    entry bar timestamp. We use a rolling join — front already minute-indexed."""
    print("[3/8] computing features (no-lookahead) ...")
    f = front.copy()
    # Trailing log returns at multiple horizons (in minutes) — shift(1) so
    # the feature at bar t reflects data through t-1.
    for k in (5, 15, 30, 60):
        f[f"ret_{k}m"] = np.log(f["close"]).diff(k)
    # Realized variance proxy: rolling stdev of 1-min returns
    r1 = np.log(f["close"]).diff().fillna(0.0)
    for k in (15, 30, 60):
        f[f"rv_{k}m"] = r1.rolling(k, min_periods=k).std()
    # ATR-14 (in price points), wilder smoothing approximation
    tr = pd.concat([
        f["high"] - f["low"],
        (f["high"] - f["close"].shift(1)).abs(),
        (f["low"] - f["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    f["atr14"] = tr.rolling(14, min_periods=14).mean()
    # Distance from close to 20/60-bar high/low (range location)
    for k in (20, 60):
        hi = f["high"].rolling(k, min_periods=k).max()
        lo = f["low"].rolling(k, min_periods=k).min()
        f[f"hi_dist_{k}"] = (hi - f["close"]) / f["close"]
        f[f"lo_dist_{k}"] = (f["close"] - lo) / f["close"]
        f[f"range_loc_{k}"] = (f["close"] - lo) / (hi - lo).replace(0, np.nan)
    # Volume z-score vs trailing 60-bar mean
    v60_mean = f["volume"].rolling(60, min_periods=60).mean()
    v60_std = f["volume"].rolling(60, min_periods=60).std()
    f["vol_z60"] = (f["volume"] - v60_mean) / v60_std.replace(0, np.nan)
    # SHIFT all features by 1 so they reflect data through bar t-1.
    feature_cols = [c for c in f.columns
                    if c not in ("open", "high", "low", "close", "volume", "symbol")]
    f[feature_cols] = f[feature_cols].shift(1)
    # Now join feature row at each candidate timestamp.
    feat = cands.join(f[feature_cols], how="left", rsuffix="_f")
    # Time-of-day / calendar features (these don't need shifting).
    feat["dow"] = feat.index.dayofweek
    feat["dom"] = feat.index.day
    feat["month"] = feat.index.month
    feat["window_h10"] = (feat["window"] == "h10").astype(int)
    # Drop rows with any NaN feature
    needed = feature_cols + ["dow", "dom", "month", "window_h10"]
    feat = feat.dropna(subset=needed)
    print(f"[3/8] feature table: {len(feat):,} rows × {len(needed)} features")
    return feat[["open", "high", "low", "close", "symbol", "window"] + needed]


# ---------------------------------------------------------------------------
# 4. Path-simulation labels (TP-first vs SL-first).
# ---------------------------------------------------------------------------
def label_outcomes(
    front: pd.DataFrame,
    feat: pd.DataFrame,
    horizon_bars: int = 60,
    tp_atr: float = 1.0,
    sl_atr: float = 0.6,
) -> pd.DataFrame:
    """For each candidate, simulate the LONG and SHORT bracket outcomes:
    label = +1 if TP hit before SL within horizon, -1 if SL hit first, 0 if
    horizon expired. Brackets are sized by atr14 * multipliers."""
    print(f"[4/8] labeling outcomes (horizon={horizon_bars} bars, "
          f"TP={tp_atr}×ATR, SL={sl_atr}×ATR) ...")
    f = front
    # Build numpy arrays for fast lookup
    ts_idx = f.index
    high = f["high"].values
    low = f["low"].values
    # Map timestamps to integer position
    ts_pos = pd.Series(np.arange(len(ts_idx)), index=ts_idx)
    long_lbl = np.zeros(len(feat), dtype=np.int8)
    short_lbl = np.zeros(len(feat), dtype=np.int8)
    long_mfe = np.zeros(len(feat))
    long_mae = np.zeros(len(feat))
    for i, (ts, row) in enumerate(feat.iterrows()):
        pos = ts_pos.get(ts, -1)
        if pos < 0 or pos + horizon_bars >= len(ts_idx):
            long_lbl[i] = 0
            short_lbl[i] = 0
            continue
        entry = row["open"]  # enter at the open of the entry bar
        atr = row["atr14"]
        if not (atr > 0):
            continue
        tp_dist = tp_atr * atr
        sl_dist = sl_atr * atr
        # LONG path: scan bars (pos+1 ... pos+horizon)
        end = pos + horizon_bars
        hi_slice = high[pos + 1: end + 1]
        lo_slice = low[pos + 1: end + 1]
        long_tp = entry + tp_dist
        long_sl = entry - sl_dist
        short_tp = entry - tp_dist
        short_sl = entry + sl_dist
        # First-touch logic for LONG (assume worst case: same-bar SL beats TP)
        long_outcome = 0
        for j in range(len(hi_slice)):
            if lo_slice[j] <= long_sl:
                long_outcome = -1
                break
            if hi_slice[j] >= long_tp:
                long_outcome = 1
                break
        long_lbl[i] = long_outcome
        # SHORT
        short_outcome = 0
        for j in range(len(hi_slice)):
            if hi_slice[j] >= short_sl:
                short_outcome = -1
                break
            if lo_slice[j] <= short_tp:
                short_outcome = 1
                break
        short_lbl[i] = short_outcome
        # Track MFE/MAE for LONG (for diagnostics)
        long_mfe[i] = (np.max(hi_slice) - entry) if len(hi_slice) else 0
        long_mae[i] = (entry - np.min(lo_slice)) if len(lo_slice) else 0
    feat = feat.copy()
    feat["long_lbl"] = long_lbl
    feat["short_lbl"] = short_lbl
    feat["long_mfe"] = long_mfe
    feat["long_mae"] = long_mae
    win_long = (feat["long_lbl"] == 1).sum()
    win_short = (feat["short_lbl"] == 1).sum()
    print(f"[4/8] LONG win={win_long} ({win_long/len(feat)*100:.1f}%) | "
          f"SHORT win={win_short} ({win_short/len(feat)*100:.1f}%)")
    return feat


# ---------------------------------------------------------------------------
# 5. Train HGB classifiers (long + short) with sequential 60/20/20 split.
# ---------------------------------------------------------------------------
HGB_KW = dict(learning_rate=0.05, max_depth=5, max_iter=300, max_leaf_nodes=64,
              min_samples_leaf=20, l2_regularization=0.1)
FEATURE_COLS = [
    "ret_5m", "ret_15m", "ret_30m", "ret_60m",
    "rv_15m", "rv_30m", "rv_60m",
    "atr14",
    "hi_dist_20", "lo_dist_20", "range_loc_20",
    "hi_dist_60", "lo_dist_60", "range_loc_60",
    "vol_z60",
    "dow", "dom", "month", "window_h10",
]


@dataclass
class TrainedSide:
    side: str
    auc_train: float
    auc_val: float
    auc_test: float
    threshold: float
    n_train: int
    n_val: int
    n_test: int


def train_models(feat: pd.DataFrame) -> Tuple[Dict, Dict]:
    print("[5/8] training HGB classifiers ...")
    feat = feat.sort_index()
    n = len(feat)
    n_tr = int(n * 0.6)
    n_va = int(n * 0.2)
    train = feat.iloc[:n_tr]
    val = feat.iloc[n_tr:n_tr + n_va]
    test = feat.iloc[n_tr + n_va:]

    models = {}
    metrics = {}
    for side in ("long", "short"):
        lbl_col = f"{side}_lbl"
        # Binary: 1 if outcome == +1 (TP first), 0 otherwise (SL first OR no
        # touch within horizon — both are "not a winner").
        y_tr = (train[lbl_col] == 1).astype(int)
        y_va = (val[lbl_col] == 1).astype(int)
        y_te = (test[lbl_col] == 1).astype(int)
        Xtr = train[FEATURE_COLS]
        Xva = val[FEATURE_COLS]
        Xte = test[FEATURE_COLS]
        if y_tr.nunique() < 2:
            print(f"[5/8] {side}: degenerate train labels, skipping")
            continue
        clf = HistGradientBoostingClassifier(**HGB_KW, random_state=42)
        clf.fit(Xtr, y_tr)
        ptr = clf.predict_proba(Xtr)[:, 1]
        pva = clf.predict_proba(Xva)[:, 1]
        pte = clf.predict_proba(Xte)[:, 1]
        auc_tr = roc_auc_score(y_tr, ptr) if y_tr.nunique() == 2 else float("nan")
        auc_va = roc_auc_score(y_va, pva) if y_va.nunique() == 2 else float("nan")
        auc_te = roc_auc_score(y_te, pte) if y_te.nunique() == 2 else float("nan")
        # Pick threshold that maximizes expected dollar-PnL on val
        best_thr = 0.5
        best_pnl = -np.inf
        for thr in np.linspace(0.30, 0.70, 41):
            mask = pva >= thr
            if mask.sum() < 5:
                continue
            outcomes = val[lbl_col].values[mask]  # +1 / -1 / 0
            # 1×ATR TP / 0.6×ATR SL → scale by per-trade dollars; here we use
            # outcome as proxy: +tp_atr - 0 - sl_atr scaled later. For
            # threshold selection use directional correctness.
            pnl = float(np.sum(np.where(outcomes == 1, 1.0, np.where(outcomes == -1, -0.6, 0.0))))
            if pnl > best_pnl:
                best_pnl = pnl
                best_thr = float(thr)
        models[side] = clf
        metrics[side] = TrainedSide(
            side=side,
            auc_train=auc_tr, auc_val=auc_va, auc_test=auc_te,
            threshold=best_thr,
            n_train=int(len(train)), n_val=int(len(val)), n_test=int(len(test)),
        )
        print(f"[5/8] {side}: AUC train={auc_tr:.3f} val={auc_va:.3f} "
              f"test={auc_te:.3f} | thr={best_thr:.2f}")
    return models, metrics


# ---------------------------------------------------------------------------
# 6. Random-control validation.
# ---------------------------------------------------------------------------
def random_control(feat: pd.DataFrame, models: Dict, metrics: Dict,
                   n_seeds: int = 5) -> Dict:
    """Verify the model isn't fitting noise: shuffle the labels within the
    test window and retrain. The shuffled-AUC distribution should center
    around 0.50; the real model's test-AUC should sit well outside that."""
    print(f"[6/8] random-control validation ({n_seeds} seeds) ...")
    feat = feat.sort_index()
    n = len(feat)
    n_tr = int(n * 0.6)
    n_va = int(n * 0.2)
    train = feat.iloc[:n_tr]
    test = feat.iloc[n_tr + n_va:]

    out = {}
    for side in ("long", "short"):
        lbl_col = f"{side}_lbl"
        y_tr = (train[lbl_col] == 1).astype(int)
        Xtr = train[FEATURE_COLS]
        Xte = test[FEATURE_COLS]
        if y_tr.nunique() < 2:
            continue
        rng = np.random.RandomState(0)
        shuf_aucs = []
        for seed in range(n_seeds):
            rng = np.random.RandomState(seed)
            shuffled = rng.permutation(y_tr.values)
            clf = HistGradientBoostingClassifier(**HGB_KW, random_state=seed)
            clf.fit(Xtr, shuffled)
            pte = clf.predict_proba(Xte)[:, 1]
            y_te = (test[lbl_col] == 1).astype(int)
            if y_te.nunique() == 2:
                shuf_aucs.append(float(roc_auc_score(y_te, pte)))
        if shuf_aucs:
            real = metrics[side].auc_test
            mean_shuf = float(np.mean(shuf_aucs))
            std_shuf = float(np.std(shuf_aucs)) if len(shuf_aucs) > 1 else 0.01
            z = (real - mean_shuf) / max(std_shuf, 1e-6)
            print(f"[6/8] {side}: real_test_AUC={real:.3f} | "
                  f"shuf_mean={mean_shuf:.3f}±{std_shuf:.3f} | z={z:+.2f}σ")
            out[side] = {
                "real_test_auc": real,
                "shuf_mean": mean_shuf,
                "shuf_std": std_shuf,
                "z_score": z,
                "shuf_aucs": shuf_aucs,
            }
    return out


# ---------------------------------------------------------------------------
# 7. Backtest with bracket simulation.
# ---------------------------------------------------------------------------
DOLLAR_PER_PT_MES = 5.0   # MES = $5 per 1pt = $1.25 per tick
DOLLAR_PER_PT_ES = 50.0   # ES = $50 per 1pt = $12.50 per tick
DEFAULT_SIZE = 1
TICK = 0.25
COMMISSION = 1.50  # round-trip per contract MES (broker-typical)


def backtest(feat: pd.DataFrame, models: Dict, metrics: Dict,
             tp_atr: float = 1.0, sl_atr: float = 0.6,
             size: int = 10,
             dollar_per_pt: float = DOLLAR_PER_PT_MES) -> Tuple[pd.DataFrame, Dict]:
    """Replay test-window candidates: emit a trade if either model exceeds
    its threshold. If both fire, take whichever has higher prob. Apply
    bracket outcome from the labels (already simulated). Single-position
    constraint: only one trade open at a time per day-window pair (one
    candidate per window per day, so this is automatic)."""
    print("[7/8] backtesting on TEST split ...")
    feat = feat.sort_index()
    n = len(feat)
    n_tr = int(n * 0.6)
    n_va = int(n * 0.2)
    test = feat.iloc[n_tr + n_va:].copy()

    long_clf = models.get("long")
    short_clf = models.get("short")
    long_thr = metrics["long"].threshold if "long" in metrics else 1.1
    short_thr = metrics["short"].threshold if "short" in metrics else 1.1

    Xte = test[FEATURE_COLS]
    test["long_prob"] = long_clf.predict_proba(Xte)[:, 1] if long_clf is not None else 0.0
    test["short_prob"] = short_clf.predict_proba(Xte)[:, 1] if short_clf is not None else 0.0
    test["fire_long"] = test["long_prob"] >= long_thr
    test["fire_short"] = test["short_prob"] >= short_thr
    # When both fire, take the higher prob
    both = test["fire_long"] & test["fire_short"]
    test.loc[both & (test["short_prob"] > test["long_prob"]), "fire_long"] = False
    test.loc[both & (test["long_prob"] >= test["short_prob"]), "fire_short"] = False

    trades = []
    for ts, row in test.iterrows():
        if row["fire_long"]:
            outcome = row["long_lbl"]
            side = "LONG"
        elif row["fire_short"]:
            outcome = row["short_lbl"]
            side = "SHORT"
        else:
            continue
        atr = row["atr14"]
        if outcome == 1:
            pnl_pts = tp_atr * atr
        elif outcome == -1:
            pnl_pts = -sl_atr * atr
        else:
            # Horizon expired — close at last bar of window. Approximate with
            # zero-PnL minus commissions. Conservative.
            pnl_pts = 0.0
        gross = pnl_pts * size * dollar_per_pt
        net = gross - COMMISSION * size
        trades.append({
            "ts": ts, "side": side, "window": row["window"],
            "entry": row["open"], "atr": atr,
            "tp_pts": tp_atr * atr, "sl_pts": sl_atr * atr,
            "outcome": int(outcome),
            "pnl_pts": float(pnl_pts), "pnl_dollars_gross": float(gross),
            "pnl_dollars_net": float(net),
            "long_prob": float(row["long_prob"]), "short_prob": float(row["short_prob"]),
            "period": label_period(ts),
        })
    tr = pd.DataFrame(trades)

    if tr.empty:
        return tr, {"n_trades": 0, "warn": "no trades"}

    summary = summarize(tr)
    print(f"[7/8] test trades: {len(tr)}  "
          f"net=${tr['pnl_dollars_net'].sum():.0f}  "
          f"WR={summary['win_rate']:.1f}%  "
          f"avg_win=${summary['avg_win']:.2f}  "
          f"max_dd=${summary['max_drawdown']:.0f}")
    return tr, summary


def summarize(tr: pd.DataFrame) -> Dict:
    if tr.empty:
        return {}
    wins = tr[tr["pnl_dollars_net"] > 0]
    losses = tr[tr["pnl_dollars_net"] < 0]
    cum = tr["pnl_dollars_net"].cumsum()
    peak = cum.cummax()
    dd = (cum - peak).min()
    days = pd.to_datetime(tr["ts"]).dt.normalize().nunique()
    out = {
        "n_trades": int(len(tr)),
        "n_days": int(days),
        "trades_per_day": float(len(tr) / max(days, 1)),
        "n_wins": int(len(wins)),
        "n_losses": int(len(losses)),
        "win_rate": float(len(wins) / len(tr) * 100),
        "total_pnl": float(tr["pnl_dollars_net"].sum()),
        "avg_pnl": float(tr["pnl_dollars_net"].mean()),
        "avg_win": float(wins["pnl_dollars_net"].mean()) if len(wins) else 0.0,
        "avg_loss": float(losses["pnl_dollars_net"].mean()) if len(losses) else 0.0,
        "max_drawdown": float(abs(dd)),
        "by_side": tr.groupby("side")["pnl_dollars_net"].sum().to_dict(),
        "by_window": tr.groupby("window")["pnl_dollars_net"].sum().to_dict(),
        "by_period": tr.groupby("period")["pnl_dollars_net"].agg(["sum", "count"]).to_dict(),
    }
    return out


# ---------------------------------------------------------------------------
# 8. Save artifacts.
# ---------------------------------------------------------------------------
def save_artifacts(feat, models, metrics, ctrl, tr, summary, out_dir=OUT_DIR):
    print(f"[8/8] saving artifacts → {out_dir} ...")
    feat.to_parquet(out_dir / "feature_table.parquet")
    if tr is not None and not tr.empty:
        tr.to_csv(out_dir / "per_trade.csv", index=False)
    joblib.dump({"models": models, "feature_cols": FEATURE_COLS}, out_dir / "models.joblib")

    payload = {
        "feature_count": len(FEATURE_COLS),
        "feature_cols": FEATURE_COLS,
        "metrics": {k: asdict(v) for k, v in metrics.items()},
        "random_control": ctrl,
        "backtest": summary,
        "config": {
            "horizon_bars": 60,
            "tp_atr": 1.0,
            "sl_atr": 0.6,
            "size": 10,
            "dollar_per_pt_mes": DOLLAR_PER_PT_MES,
            "commission_per_round_trip": COMMISSION,
        },
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2, default=str))
    print(f"[8/8] done. Files:")
    for p in sorted(out_dir.glob("*")):
        print(f"  {p.name}  ({p.stat().st_size:,} bytes)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    front = build_front_month_series()
    cands = build_entry_candidates(front)
    feat = compute_features(front, cands)
    feat = label_outcomes(front, feat)
    models, metrics = train_models(feat)
    ctrl = random_control(feat, models, metrics)
    tr, summary = backtest(feat, models, metrics)
    save_artifacts(feat, models, metrics, ctrl, tr, summary)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
