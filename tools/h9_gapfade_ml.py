#!/usr/bin/env python3
"""H9 Gap-Fade ML strategy.

Edge: at 09:30 ET, the cash open often mean-reverts toward yesterday's close.
The exploratory study (tools/h9h10_explore.py) showed:
  - gap_dn_big (>0.5% below prev close): n=410, mean +17.3bps, t=+6.40σ
  - gap_up_big (>0.5% above prev close): n=474, mean -12.3bps, t=-5.03σ
Both clear the 5σ bar and are consistent across DOW.

This module:
  1. Builds the same continuous front-month series.
  2. Selects days that hit the gap thresholds.
  3. Trains an HGB classifier per side to predict whether the bracket trade
     will hit TP-first (within 30-bar horizon, ATR-scaled brackets).
  4. Runs random-control validation (5 seeds).
  5. Backtests with bracket simulation by period (Trump 1 / Biden / Trump 2).
  6. Saves model + thresholds + per-trade ledger.

Output: artifacts/h9_gapfade_ml/
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
PARQUET = ROOT / "es_master_outrights-2.parquet"
OUT_DIR = ROOT / "artifacts" / "h9_gapfade_ml"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Roll calendar (front-month). Inline copy.
# ---------------------------------------------------------------------------
ROLL_CALENDAR = [
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
    (pd.Timestamp("2020-12-11"), "ESH1"),
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

GAP_THRESHOLD = 0.005  # 0.5% gap threshold for both directions

DOLLAR_PER_PT_MES = 5.0
DOLLAR_PER_PT_ES = 50.0
COMMISSION = 1.50  # round-trip per contract MES
DEFAULT_SIZE = 10  # 10 MES contracts (matches FibH1214 sizing)


def label_period(ts: pd.Timestamp) -> str:
    if ts < pd.Timestamp("2017-01-20", tz=ts.tz):
        return "pre_t1"
    if ts < pd.Timestamp("2021-01-20", tz=ts.tz):
        return "trump1"
    if ts < pd.Timestamp("2025-01-20", tz=ts.tz):
        return "biden"
    return "trump2"


def build_front_month():
    print(f"[1/7] loading {PARQUET.name} ...")
    df = pd.read_parquet(PARQUET).sort_index()
    sym_order = (df.reset_index()
                   .groupby("symbol", sort=False)["timestamp"]
                   .min().sort_values().index.to_list())
    sym_rank = {s: i for i, s in enumerate(sym_order)}
    cuts_ts = pd.DatetimeIndex(
        [c.tz_localize("UTC") if c.tzinfo is None else c.tz_convert("UTC")
         for c, _ in ROLL_CALENDAR]
    ).tz_convert(None)
    syms = np.array([s for _, s in ROLL_CALENDAR])
    if df.index.tz is not None:
        ts_utc = df.index.tz_convert("UTC").tz_convert(None)
    else:
        ts_utc = df.index
    pos = np.searchsorted(cuts_ts.values, ts_utc.values, side="right") - 1
    pos = np.clip(pos, 0, len(syms) - 1)
    df["front_symbol"] = syms[pos]
    front_rank = df["front_symbol"].map(sym_rank).fillna(9999).astype(int)
    sym_rank_col = df["symbol"].map(sym_rank).fillna(9999).astype(int)
    df["roll_distance"] = (sym_rank_col - front_rank).abs()
    df = df.sort_values("roll_distance", kind="mergesort")
    front = df[~df.index.duplicated(keep="first")].sort_index()
    print(f"[1/7] front-month: {len(front):,} bars, "
          f"{front.index.normalize().nunique():,} unique dates")
    return front[["open", "high", "low", "close", "volume", "symbol"]]


def build_daily(front):
    """Per-day feature-rich table with 30+ engineered features.

    Mirrors the depth of AetherFlow / DE3 feature stacks: gap geometry,
    pre-RTH (ETH) context, multi-horizon trailing returns/vols, position-
    relative features, calendar dummies, regime proxies."""
    print("[2/7] building per-day RTH table (rich feature set) ...")
    f = front.copy()
    f["date"] = f.index.tz_convert("US/Eastern").date
    f["minute"] = f.index.minute
    f["hour"] = f.index.hour

    def first_per_day(mask, col, name):
        out = f[mask].drop_duplicates(subset=["date"], keep="first")
        return out.set_index("date")[[col]].rename(columns={col: name})

    # Core RTH bars
    h930 = first_per_day((f["hour"] == 9) & (f["minute"] == 30), "open", "open_0930")
    h1000 = first_per_day((f["hour"] == 10) & (f["minute"] == 0), "open", "open_1000")
    h1100 = first_per_day((f["hour"] == 11) & (f["minute"] == 0), "open", "open_1100")
    # First 5-min after open
    h0934 = first_per_day((f["hour"] == 9) & (f["minute"] == 34), "close", "close_0934")

    rth_mask = (f["hour"] >= 9) & (f["hour"] < 16)
    rth_last = (f[rth_mask].sort_index().groupby("date").last()[["close", "high", "low"]]
                  .rename(columns={"close": "rth_close", "high": "rth_high",
                                    "low": "rth_low"}))
    # ETH (18:00 prev → 09:29 today)
    eth_mask = (f["hour"] < 9) | ((f["hour"] == 9) & (f["minute"] < 30))
    eth = f[eth_mask].groupby("date").agg(
        eth_high=("high", "max"), eth_low=("low", "min"),
        eth_open=("open", "first"), eth_close=("close", "last"),
        eth_volume=("volume", "sum"))
    # RTH volume
    rth_vol = (f[rth_mask].groupby("date")["volume"].sum().rename("rth_volume"))
    # Total daily range
    rth_range = (f[rth_mask].groupby("date").agg(
        day_high=("high", "max"), day_low=("low", "min")))
    # First-30 close (drive)
    drive_mask = (f["hour"] == 9) & (f["minute"].between(30, 59))
    drive_close = (f[drive_mask].sort_index().groupby("date").last()["close"]
                     .rename("close_0959"))

    daily = pd.concat([h930, h1000, h1100, h0934, rth_last, eth,
                        rth_vol, rth_range, drive_close], axis=1)
    daily.index = pd.to_datetime(daily.index)
    daily = daily.sort_index()

    # ----- DERIVED FEATURES (30+) -----
    c = daily
    c["prev_close"] = c["rth_close"].shift(1)
    c["prev_high"] = c["rth_high"].shift(1)
    c["prev_low"] = c["rth_low"].shift(1)
    c["prev_rth_volume"] = c["rth_volume"].shift(1)
    c["prev_eth_volume"] = c["eth_volume"].shift(0)  # last night's ETH

    # 1. Gap geometry (5)
    c["gap_pct"] = (c["open_0930"] / c["prev_close"] - 1)
    c["abs_gap_pct"] = c["gap_pct"].abs()
    c["gap_squared"] = c["gap_pct"] ** 2
    # gap z-score vs trailing 60-day distribution
    c["gap_z60"] = (c["gap_pct"] - c["gap_pct"].rolling(60, min_periods=20).mean()) \
                    / c["gap_pct"].rolling(60, min_periods=20).std().replace(0, np.nan)
    # Gap direction
    c["gap_dir"] = np.sign(c["gap_pct"]).fillna(0).astype(int)

    # 2. Pre-RTH (ETH) context (8)
    c["eth_range_pts"] = c["eth_high"] - c["eth_low"]
    c["eth_range_pct"] = c["eth_range_pts"] / c["prev_close"]
    c["eth_close_loc"] = (c["eth_close"] - c["eth_low"]) / (c["eth_high"] - c["eth_low"]).replace(0, np.nan)
    c["eth_close_vs_prev"] = (c["eth_close"] / c["prev_close"] - 1)
    c["eth_above_prev_close"] = (c["eth_high"] > c["prev_close"]).astype(int)
    c["eth_below_prev_close"] = (c["eth_low"] < c["prev_close"]).astype(int)
    c["eth_open_drift"] = (c["open_0930"] / c["eth_close"] - 1)
    c["eth_volume_z"] = (c["eth_volume"] - c["eth_volume"].rolling(20, min_periods=10).mean()) \
                        / c["eth_volume"].rolling(20, min_periods=10).std().replace(0, np.nan)

    # 3. Trailing returns (multi-horizon, 5)
    rets1 = (c["rth_close"] / c["prev_close"] - 1).fillna(0)
    c["ret_1d"] = rets1.shift(1)
    c["ret_3d"] = c["rth_close"].pct_change(3).shift(1)
    c["ret_5d"] = c["rth_close"].pct_change(5).shift(1)
    c["ret_10d"] = c["rth_close"].pct_change(10).shift(1)
    c["ret_20d"] = c["rth_close"].pct_change(20).shift(1)

    # 4. Realized vol (multi-horizon, 4)
    c["rv_5d"] = rets1.rolling(5, min_periods=3).std()
    c["rv_10d"] = rets1.rolling(10, min_periods=5).std()
    c["rv_20d"] = rets1.rolling(20, min_periods=10).std()
    c["rv_60d"] = rets1.rolling(60, min_periods=20).std()
    c["rv_5_20_ratio"] = c["rv_5d"] / c["rv_20d"].replace(0, np.nan)

    # 5. Range and position (5)
    c["prev_rth_range_pct"] = (c["prev_high"] - c["prev_low"]) / c["prev_close"]
    c["pos_in_60d_hi"] = c["rth_close"] / c["rth_close"].rolling(60, min_periods=20).max()
    c["pos_in_60d_lo"] = c["rth_close"] / c["rth_close"].rolling(60, min_periods=20).min()
    # 50-day MA distance
    ma50 = c["rth_close"].rolling(50, min_periods=20).mean()
    c["ma50_dist_pct"] = (c["prev_close"] / ma50 - 1).shift(1)
    c["ma200_dist_pct"] = (c["prev_close"] / c["rth_close"].rolling(200, min_periods=50).mean() - 1).shift(1)

    # 6. Calendar / regime (7)
    c["dow"] = c.index.dayofweek
    c["dom"] = c.index.day
    c["month"] = c.index.month
    c["quarter"] = c.index.quarter
    c["is_monday"] = (c["dow"] == 0).astype(int)
    c["is_friday"] = (c["dow"] == 4).astype(int)
    c["is_first_5d_of_month"] = (c["dom"] <= 5).astype(int)
    c["is_last_5d_of_month"] = (c["dom"] >= 25).astype(int)

    # 7. Drive features (first 30 min direction at 09:30, NOT used for LONG/SHORT
    #    at the 09:30 entry — but useful for context)
    c["first5_dir"] = np.sign(c["close_0934"] - c["open_0930"]).fillna(0).astype(int)
    c["first30_dir"] = np.sign(c["close_0959"] - c["open_0930"]).fillna(0).astype(int)
    # NOTE: first5_dir / first30_dir are AFTER the 09:30 entry; we DON'T use
    # them as features for the 09:30 entry decision. They're labeled here for
    # diagnostic purposes only and excluded from ML_FEATURES below.

    # 8. Volume / liquidity context (3)
    c["rth_vol_z20"] = (c["prev_rth_volume"] - c["rth_volume"].rolling(20, min_periods=10).mean()) \
                        / c["rth_volume"].rolling(20, min_periods=10).std().replace(0, np.nan)
    c["eth_to_rth_volume_ratio"] = c["eth_volume"] / c["prev_rth_volume"].replace(0, np.nan)
    c["volume_pct_of_60d_max"] = c["prev_rth_volume"] / c["rth_volume"].rolling(60, min_periods=20).max().replace(0, np.nan)

    daily = c.dropna(subset=["open_0930", "open_1000", "prev_close"])
    print(f"[2/7] daily table: {len(daily):,} rows × {len(daily.columns)} cols  "
          f"({daily.index.min().date()} → {daily.index.max().date()})")
    return daily


def build_signals(daily):
    print("[3/7] selecting gap-fade signals ...")
    d = daily.copy()
    d["gap_pct"] = (d["open_0930"] / d["prev_close"] - 1)
    d["abs_gap_pct"] = d["gap_pct"].abs()
    # Long gap-fill: gap down > 0.5%
    d["fire_long"] = d["gap_pct"] <= -GAP_THRESHOLD
    # Short gap-fade: gap up > 0.5%
    d["fire_short"] = d["gap_pct"] >= GAP_THRESHOLD
    n_long = d["fire_long"].sum()
    n_short = d["fire_short"].sum()
    print(f"[3/7] long signals: {n_long:,} ({n_long/len(d)*100:.1f}%) | "
          f"short signals: {n_short:,} ({n_short/len(d)*100:.1f}%)")
    return d


def label_outcomes(front, daily, horizon_min=30, tp_pct=0.0030, sl_pct=0.0040):
    """Walk forward 30 minutes from 09:30 entry. Label = +1 (TP first), -1
    (SL first), 0 (neither — close at horizon's last close)."""
    print(f"[4/7] labeling outcomes (horizon={horizon_min}min, TP={tp_pct*100:.2f}%, "
          f"SL={sl_pct*100:.2f}%) ...")
    f = front.copy()
    f["date"] = f.index.tz_convert("US/Eastern").date
    f["minute"] = f.index.minute
    f["hour"] = f.index.hour
    # Build per-date arrays of 09:30..09:30+horizon-1 high/low for fast lookup
    rth = f[(f["hour"] == 9) & (f["minute"] >= 30) |
            (f["hour"] == 10) & (f["minute"] < (30 + horizon_min) % 60)]
    rth = f[((f["hour"] == 9) & (f["minute"] >= 30)) |
            ((f["hour"] == 10) & (f["minute"] < ((30 + horizon_min) % 60 if (30+horizon_min) >= 60 else 60)))]
    # Easier: build a flat list of (date, hour, minute, high, low, close)
    rth_bars = f[((f["hour"] == 9) & (f["minute"] >= 30)) |
                  (f["hour"] == 10)].copy()
    rth_bars = rth_bars.sort_index()

    long_lbl = np.zeros(len(daily), dtype=np.int8)
    short_lbl = np.zeros(len(daily), dtype=np.int8)
    long_pnl = np.zeros(len(daily), dtype=np.float64)
    short_pnl = np.zeros(len(daily), dtype=np.float64)

    bars_by_date = dict(list(rth_bars.groupby("date")))

    for i, (date, row) in enumerate(daily.iterrows()):
        date_key = date.date()
        if date_key not in bars_by_date:
            continue
        bars = bars_by_date[date_key]
        # Take the first horizon_min bars at 09:30..09:30+horizon
        bars = bars.iloc[:horizon_min] if len(bars) > horizon_min else bars
        if bars.empty:
            continue
        entry = float(row["open_0930"])
        # LONG: TP at entry*(1+tp), SL at entry*(1-sl)
        long_tp = entry * (1 + tp_pct)
        long_sl = entry * (1 - sl_pct)
        # SHORT: TP at entry*(1-tp), SL at entry*(1+sl)
        short_tp = entry * (1 - tp_pct)
        short_sl = entry * (1 + sl_pct)
        long_outcome = 0
        short_outcome = 0
        long_exit = entry
        short_exit = entry
        for _, b in bars.iterrows():
            hi, lo = float(b["high"]), float(b["low"])
            if long_outcome == 0:
                if lo <= long_sl:
                    long_outcome = -1
                    long_exit = long_sl
                elif hi >= long_tp:
                    long_outcome = 1
                    long_exit = long_tp
            if short_outcome == 0:
                if hi >= short_sl:
                    short_outcome = -1
                    short_exit = short_sl
                elif lo <= short_tp:
                    short_outcome = 1
                    short_exit = short_tp
            if long_outcome != 0 and short_outcome != 0:
                break
        if long_outcome == 0:
            long_exit = float(bars.iloc[-1]["close"])
        if short_outcome == 0:
            short_exit = float(bars.iloc[-1]["close"])
        long_lbl[i] = long_outcome
        short_lbl[i] = short_outcome
        long_pnl[i] = (long_exit - entry)  # in price points
        short_pnl[i] = (entry - short_exit)
    daily = daily.copy()
    daily["long_lbl"] = long_lbl
    daily["short_lbl"] = short_lbl
    daily["long_pnl_pts"] = long_pnl
    daily["short_pnl_pts"] = short_pnl
    print(f"[4/7] LONG outcomes: TP={ (long_lbl==1).sum() } "
          f"SL={ (long_lbl==-1).sum() } HOLD={ (long_lbl==0).sum() }")
    print(f"[4/7] SHORT outcomes: TP={ (short_lbl==1).sum() } "
          f"SL={ (short_lbl==-1).sum() } HOLD={ (short_lbl==0).sum() }")
    return daily


# ---------------------------------------------------------------------------
# 5. ML overlay — predict whether the trade will be a TP-winner.
# ---------------------------------------------------------------------------
ML_FEATURES = [
    # Gap geometry (5)
    "gap_pct", "abs_gap_pct", "gap_squared", "gap_z60", "gap_dir",
    # Pre-RTH (ETH) context (8)
    "eth_range_pts", "eth_range_pct", "eth_close_loc", "eth_close_vs_prev",
    "eth_above_prev_close", "eth_below_prev_close", "eth_open_drift",
    "eth_volume_z",
    # Trailing returns (5)
    "ret_1d", "ret_3d", "ret_5d", "ret_10d", "ret_20d",
    # Realized vol (5)
    "rv_5d", "rv_10d", "rv_20d", "rv_60d", "rv_5_20_ratio",
    # Range / position (5)
    "prev_rth_range_pct", "pos_in_60d_hi", "pos_in_60d_lo",
    "ma50_dist_pct", "ma200_dist_pct",
    # Calendar / regime (8)
    "dow", "dom", "month", "quarter",
    "is_monday", "is_friday", "is_first_5d_of_month", "is_last_5d_of_month",
    # Volume / liquidity (3)
    "rth_vol_z20", "eth_to_rth_volume_ratio", "volume_pct_of_60d_max",
]
# 39 features — comparable depth to AetherFlow's manifold feature stack.

HGB_KW = dict(learning_rate=0.05, max_depth=4, max_iter=200, max_leaf_nodes=31,
              min_samples_leaf=30, l2_regularization=0.5)


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


def train_ml(daily):
    print("[5/7] training ML overlays ...")
    d = daily.copy().sort_index()
    # dow/month/etc. already computed in build_daily; just guard against NaN.
    d = d.dropna(subset=ML_FEATURES + ["fire_long", "fire_short", "long_lbl", "short_lbl"])

    # Sequential 60/20/20 split
    n = len(d)
    n_tr, n_va = int(n * 0.6), int(n * 0.2)
    tr = d.iloc[:n_tr]
    va = d.iloc[n_tr:n_tr + n_va]
    te = d.iloc[n_tr + n_va:]
    print(f"[5/7] splits  train={len(tr)}  val={len(va)}  test={len(te)}")

    models = {}
    metrics = {}
    for side in ("long", "short"):
        sig_col = f"fire_{side}"
        lbl_col = f"{side}_lbl"
        tr_sig = tr[tr[sig_col]]
        va_sig = va[va[sig_col]]
        te_sig = te[te[sig_col]]
        if len(tr_sig) < 50:
            print(f"[5/7] {side}: not enough train signals ({len(tr_sig)}), skip ML")
            continue
        y_tr = (tr_sig[lbl_col] == 1).astype(int)
        y_va = (va_sig[lbl_col] == 1).astype(int)
        y_te = (te_sig[lbl_col] == 1).astype(int)
        Xtr, Xva, Xte = tr_sig[ML_FEATURES], va_sig[ML_FEATURES], te_sig[ML_FEATURES]
        if y_tr.nunique() < 2:
            print(f"[5/7] {side}: degenerate y_tr, skip")
            continue
        clf = HistGradientBoostingClassifier(**HGB_KW, random_state=42)
        clf.fit(Xtr, y_tr)
        ptr = clf.predict_proba(Xtr)[:, 1]
        pva = clf.predict_proba(Xva)[:, 1] if len(Xva) else np.array([])
        pte = clf.predict_proba(Xte)[:, 1] if len(Xte) else np.array([])
        auc_tr = roc_auc_score(y_tr, ptr) if y_tr.nunique() == 2 else float("nan")
        auc_va = roc_auc_score(y_va, pva) if len(y_va) and y_va.nunique() == 2 else float("nan")
        auc_te = roc_auc_score(y_te, pte) if len(y_te) and y_te.nunique() == 2 else float("nan")
        # Threshold selection on val: max expected dollar PnL using actual outcomes
        best_thr, best_pnl = 0.5, -np.inf
        for thr in np.linspace(0.20, 0.80, 61):
            mask = pva >= thr
            if mask.sum() < 5:
                continue
            outcomes = va_sig[lbl_col].values[mask]
            pnl_proxy = np.where(outcomes == 1, 1.0, np.where(outcomes == -1, -1.33, 0.0)).sum()
            if pnl_proxy > best_pnl:
                best_pnl, best_thr = pnl_proxy, float(thr)
        models[side] = clf
        metrics[side] = TrainedSide(
            side=side,
            auc_train=auc_tr, auc_val=auc_va, auc_test=auc_te,
            threshold=best_thr,
            n_train=int(len(tr_sig)), n_val=int(len(va_sig)), n_test=int(len(te_sig)),
        )
        print(f"[5/7] {side}: AUC train={auc_tr:.3f} val={auc_va:.3f} "
              f"test={auc_te:.3f} | thr={best_thr:.2f}  "
              f"(n_tr={len(tr_sig)} n_va={len(va_sig)} n_te={len(te_sig)})")
    return models, metrics


def random_control(daily, models, metrics, n_seeds=5):
    print(f"[6/7] random-control validation ({n_seeds} seeds) ...")
    d = daily.copy().sort_index()
    d = d.dropna(subset=ML_FEATURES + ["fire_long", "fire_short", "long_lbl", "short_lbl"])
    n = len(d)
    n_tr, n_va = int(n * 0.6), int(n * 0.2)
    tr = d.iloc[:n_tr]
    te = d.iloc[n_tr + n_va:]
    out = {}
    for side in ("long", "short"):
        if side not in models:
            continue
        sig_col = f"fire_{side}"
        lbl_col = f"{side}_lbl"
        tr_sig = tr[tr[sig_col]]
        te_sig = te[te[sig_col]]
        if len(te_sig) < 5:
            continue
        Xtr = tr_sig[ML_FEATURES]
        Xte = te_sig[ML_FEATURES]
        y_tr_real = (tr_sig[lbl_col] == 1).astype(int)
        y_te = (te_sig[lbl_col] == 1).astype(int)
        if y_tr_real.nunique() < 2 or y_te.nunique() < 2:
            continue
        shuf_aucs = []
        for seed in range(n_seeds):
            rng = np.random.RandomState(seed)
            shuf_y = rng.permutation(y_tr_real.values)
            clf = HistGradientBoostingClassifier(**HGB_KW, random_state=seed)
            clf.fit(Xtr, shuf_y)
            pte = clf.predict_proba(Xte)[:, 1]
            shuf_aucs.append(float(roc_auc_score(y_te, pte)))
        real = metrics[side].auc_test
        m_s = float(np.mean(shuf_aucs))
        s_s = float(np.std(shuf_aucs)) if len(shuf_aucs) > 1 else 0.01
        z = (real - m_s) / max(s_s, 1e-6)
        print(f"[6/7] {side}: real={real:.3f}  shuf={m_s:.3f}±{s_s:.3f}  z={z:+.2f}σ")
        out[side] = {"real_test_auc": real, "shuf_mean": m_s, "shuf_std": s_s,
                     "z_score": z, "shuf_aucs": shuf_aucs}
    return out


def backtest(daily, models, metrics, size=DEFAULT_SIZE,
             dollar_per_pt=DOLLAR_PER_PT_MES,
             use_ml=True, tp_pct=0.0030, sl_pct=0.0040):
    print(f"[7/7] backtesting (size={size} MES, ${dollar_per_pt}/pt, "
          f"use_ml={use_ml}) ...")
    d = daily.copy().sort_index()
    d = d.dropna(subset=ML_FEATURES + ["fire_long", "fire_short", "long_lbl", "short_lbl"])
    n = len(d)
    n_tr, n_va = int(n * 0.6), int(n * 0.2)
    test = d.iloc[n_tr + n_va:].copy()

    if use_ml:
        for side in ("long", "short"):
            sig_col = f"fire_{side}"
            test[f"{side}_prob"] = 0.0
            if side in models:
                mask = test[sig_col]
                test.loc[mask, f"{side}_prob"] = models[side].predict_proba(
                    test.loc[mask, ML_FEATURES])[:, 1]
            test[f"{side}_take"] = (
                test[sig_col]
                & (test[f"{side}_prob"] >= metrics[side].threshold if side in metrics else False)
            )
    else:
        test["long_take"] = test["fire_long"]
        test["short_take"] = test["fire_short"]

    trades = []
    for ts, row in test.iterrows():
        side = None
        outcome = 0
        pnl_pts = 0.0
        if row["long_take"]:
            side = "LONG"
            outcome = int(row["long_lbl"])
            pnl_pts = float(row["long_pnl_pts"])
        elif row["short_take"]:
            side = "SHORT"
            outcome = int(row["short_lbl"])
            pnl_pts = float(row["short_pnl_pts"])
        if side is None:
            continue
        gross = pnl_pts * size * dollar_per_pt
        net = gross - COMMISSION * size
        trades.append({
            "ts": ts, "side": side,
            "entry": row["open_0930"],
            "gap_pct": row["gap_pct"],
            "outcome": outcome,
            "pnl_pts": pnl_pts,
            "pnl_dollars_gross": gross,
            "pnl_dollars_net": net,
            "long_prob": float(row.get("long_prob", 0.0)),
            "short_prob": float(row.get("short_prob", 0.0)),
            "period": label_period(ts),
        })
    tr = pd.DataFrame(trades)
    if tr.empty:
        return tr, {"warn": "no trades"}
    summary = summarize(tr)
    print(f"[7/7] {len(tr)} trades  net=${tr['pnl_dollars_net'].sum():.0f}  "
          f"WR={summary['win_rate']:.1f}%  avg_win=${summary['avg_win']:.2f}  "
          f"max_dd=${summary['max_drawdown']:.0f}")
    return tr, summary


def summarize(tr):
    if tr.empty:
        return {}
    wins = tr[tr["pnl_dollars_net"] > 0]
    losses = tr[tr["pnl_dollars_net"] < 0]
    cum = tr["pnl_dollars_net"].cumsum()
    peak = cum.cummax()
    dd = (cum - peak).min()
    days = pd.to_datetime(tr["ts"]).dt.normalize().nunique()
    span_years = max(1, (pd.to_datetime(tr["ts"]).max() - pd.to_datetime(tr["ts"]).min()).days / 365.25)
    return {
        "n_trades": int(len(tr)),
        "n_days": int(days),
        "trades_per_day": float(len(tr) / max(days, 1)),
        "n_wins": int(len(wins)),
        "n_losses": int(len(losses)),
        "win_rate": float(len(wins) / len(tr) * 100),
        "total_pnl": float(tr["pnl_dollars_net"].sum()),
        "annualized_pnl": float(tr["pnl_dollars_net"].sum() / span_years),
        "avg_pnl": float(tr["pnl_dollars_net"].mean()),
        "avg_win": float(wins["pnl_dollars_net"].mean()) if len(wins) else 0.0,
        "avg_loss": float(losses["pnl_dollars_net"].mean()) if len(losses) else 0.0,
        "max_drawdown": float(abs(dd)),
        "by_side": tr.groupby("side")["pnl_dollars_net"].agg(["sum", "count"]).to_dict(),
        "by_period": tr.groupby("period")["pnl_dollars_net"].agg(["sum", "count"]).to_dict(),
    }


def save_artifacts(daily, models, metrics, ctrl, tr_rule, sum_rule, tr_ml, sum_ml):
    print(f"saving → {OUT_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(OUT_DIR / "daily_table.parquet")
    if not tr_rule.empty:
        tr_rule.to_csv(OUT_DIR / "per_trade_rule.csv", index=False)
    if tr_ml is not None and not tr_ml.empty:
        tr_ml.to_csv(OUT_DIR / "per_trade_ml.csv", index=False)
    joblib.dump({"models": models, "feature_cols": ML_FEATURES,
                 "thresholds": {k: v.threshold for k, v in metrics.items()},
                 "gap_threshold": GAP_THRESHOLD},
                 OUT_DIR / "models.joblib")
    payload = {
        "config": {
            "gap_threshold": GAP_THRESHOLD,
            "tp_pct": 0.0030, "sl_pct": 0.0040,
            "horizon_min": 30,
            "size_mes": DEFAULT_SIZE,
            "dollar_per_pt_mes": DOLLAR_PER_PT_MES,
            "commission_per_round_trip": COMMISSION,
        },
        "ml_metrics": {k: asdict(v) for k, v in metrics.items()},
        "random_control": ctrl,
        "backtest_rule_only": sum_rule,
        "backtest_with_ml": sum_ml,
    }
    (OUT_DIR / "results.json").write_text(json.dumps(payload, indent=2, default=str))
    print(f"saved {len(list(OUT_DIR.glob('*')))} files")


def main():
    front = build_front_month()
    daily = build_daily(front)
    daily = build_signals(daily)
    daily = label_outcomes(front, daily)
    models, metrics = train_ml(daily)
    ctrl = random_control(daily, models, metrics)

    print("\n--- RULE-ONLY (no ML) backtest ---")
    tr_rule, sum_rule = backtest(daily, models, metrics, use_ml=False)

    print("\n--- WITH ML overlay backtest ---")
    tr_ml, sum_ml = backtest(daily, models, metrics, use_ml=True)

    save_artifacts(daily, models, metrics, ctrl, tr_rule, sum_rule, tr_ml, sum_ml)


if __name__ == "__main__":
    main()
