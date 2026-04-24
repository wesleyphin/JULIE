"""Shared utilities for the regime-ML reproducible trainers.

Exports: data loading, feature building, rolling metric computation, and
forward-walk PnL simulators. Used by train_model_a, train_model_b,
train_model_c, diagnose.

No LightGBM. HGB-only. Deterministic (fixed seeds in trainers).
Requirements: numpy, pandas, scikit-learn, pyarrow (for parquet).
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ─── Constants — shared with regime_classifier.py + live bot ─────────────

ROOT = Path(__file__).resolve().parents[2]

# Matches regime_classifier._classify boundaries
DEAD_TAPE_VOL_BP = 1.5
EFF_LOW = 0.05
EFF_HIGH = 0.12
WINDOW_BARS = 120

# Bracket policies matching apply_dead_tape_brackets + default DE3
DEAD_TAPE_TP = 3.0
DEAD_TAPE_SL = 5.0
DEFAULT_TP = 6.0
DEFAULT_SL = 4.0
# BE-arm regime (DE3 defaults)
BE_TP = 10.0
BE_SL = 4.0
BE_TRIGGER_MFE = 5.0

MES_PT_VALUE = 5.0
PNL_LOOKAHEAD_BARS = 60
SAMPLE_EVERY = 5            # every 5th bar in feature space
AMBIGUOUS_MARGIN_USD = 30.0 # drop training rows where label margin < this

# NY session filter (ET hours)
SESSION_START_HOUR_ET = 9
SESSION_END_HOUR_ET = 16

# Default training + holdout windows (used to reproduce shipped artifacts)
DEFAULT_TRAIN_START = "2024-07-01"
DEFAULT_TRAIN_END   = "2026-01-26"
DEFAULT_OOS_START   = "2026-01-27"
DEFAULT_OOS_END     = "2026-04-20"

# Sizing knobs
NATURAL_SIZE = 3
SMALL_SIZE = 1

# 40-feature set (matches what the live bot builds in regime_classifier.RegimeClassifier.build_ml_feature_snapshot)
FEATURE_COLS_40 = [
    # vol/eff at 5 lookbacks
    "vol_bp_30", "eff_30",
    "vol_bp_60", "eff_60",
    "vol_bp_120", "eff_120",
    "vol_bp_240", "eff_240",
    "vol_bp_480", "eff_480",
    # slopes
    "vol_slope_10", "vol_slope_30", "vol_slope_60", "eff_slope_30",
    # range / ATR
    "atr14", "atr30",
    "range_pct_20", "range_pct_120",
    # bar shape
    "body_ratio_20", "body_ratio_60", "abs_body_20", "up_bar_pct_20",
    "run_up_max_20", "run_down_max_20",
    # gaps
    "gap_pct", "gap_abs_mean_20",
    # momentum
    "mom_5", "mom_15", "mom_30", "mom_60", "mom_120",
    # volume
    "volume_z_20", "volume_ma_ratio",
    # run-ups
    "max_runup_60", "max_rundown_60",
    # session
    "et_hour", "minutes_into_session", "day_of_week",
    # cross-strategy proxies
    "any_strategy_signal_30", "big_move_10",
]

# v6 additive feature — present on B and C trainers only
FEATURE_COLS_V6 = FEATURE_COLS_40 + ["a_pred_scalp"]


# ─── Data loading ────────────────────────────────────────────────────────

def load_continuous_bars(start: str, end: str,
                          parquet_path: Optional[Path] = None) -> pd.DataFrame:
    """Load minute bars from es_master_outrights.parquet, filter to the dominant
    symbol per day (max volume), sort by time. Returns a DatetimeIndex-keyed
    DataFrame with columns: open, high, low, close, volume, symbol.
    """
    if parquet_path is None:
        parquet_path = ROOT / "es_master_outrights.parquet"
    df = pd.read_parquet(parquet_path)
    lo = pd.Timestamp(start, tz=df.index.tz)
    hi = pd.Timestamp(end, tz=df.index.tz)
    df = df.loc[(df.index >= lo) & (df.index <= hi)].copy()
    date_arr = df.index.date
    dom = (df.assign(_d=date_arr)
             .groupby(["_d", "symbol"])["volume"].sum()
             .reset_index()
             .sort_values(["_d", "volume"], ascending=[True, False])
             .drop_duplicates("_d", keep="first"))
    dmap = dict(zip(dom["_d"], dom["symbol"]))
    mask = pd.Series(date_arr, index=df.index).map(dmap) == df["symbol"]
    return df.loc[mask.values].sort_index()


# ─── Feature building ────────────────────────────────────────────────────

def rolling_vol_eff(closes: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    """Rolling (vol_bp, eff) over `window` bars. Matches regime_classifier._compute_metrics."""
    n = len(closes)
    vol_bp = np.full(n, np.nan)
    eff = np.full(n, np.nan)
    for i in range(window, n):
        p = closes[i - window : i + 1]
        rets = (p[1:] - p[:-1]) / p[:-1]
        if len(rets) == 0:
            continue
        mean = rets.mean()
        var = ((rets - mean) ** 2).sum() / max(1, len(rets) - 1)
        vol_bp[i] = (var ** 0.5) * 10_000.0
        abs_sum = np.abs(rets).sum()
        eff[i] = abs(rets.sum()) / abs_sum if abs_sum > 0 else 0.0
    return vol_bp, eff


def build_feature_frame(bars: pd.DataFrame) -> pd.DataFrame:
    """Build the 40-feature snapshot frame. Output aligns index with bars."""
    c = bars["close"].to_numpy(float)
    o = bars["open"].to_numpy(float)
    h = bars["high"].to_numpy(float)
    l = bars["low"].to_numpy(float)
    v = bars["volume"].to_numpy(float)

    vol120, eff120 = rolling_vol_eff(c, 120)
    vol60,  eff60  = rolling_vol_eff(c, 60)
    vol30,  eff30  = rolling_vol_eff(c, 30)
    vol240, eff240 = rolling_vol_eff(c, 240)
    vol480, eff480 = rolling_vol_eff(c, 480)

    def diff(arr, lag):
        return arr - np.r_[np.full(lag, np.nan), arr[:-lag]]
    vol_slope_10 = diff(vol60, 10)
    vol_slope_30 = diff(vol60, 30)
    vol_slope_60 = diff(vol120, 60)
    eff_slope_30 = diff(eff60, 30)

    prev_c = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    atr14 = pd.Series(tr).rolling(14).mean().to_numpy()
    atr30 = pd.Series(tr).rolling(30).mean().to_numpy()

    rng = h - l
    rng_pct = rng / np.where(c != 0, c, 1.0)
    rng_pct_20 = pd.Series(rng_pct).rolling(20).mean().to_numpy() * 10_000
    rng_pct_120 = pd.Series(rng_pct).rolling(120).mean().to_numpy() * 10_000

    hl = np.maximum(rng, 1e-9)
    body = np.abs(c - o)
    body_ratio = body / hl
    body_ratio_20 = pd.Series(body_ratio).rolling(20).mean().to_numpy()
    body_ratio_60 = pd.Series(body_ratio).rolling(60).mean().to_numpy()
    abs_body_20 = pd.Series(body).rolling(20).mean().to_numpy()

    up_bar = (c >= o).astype(float)
    up_bar_pct_20 = pd.Series(up_bar).rolling(20).mean().to_numpy()

    run_up = np.zeros(len(c), dtype=int)
    run_down = np.zeros(len(c), dtype=int)
    run_up[0] = 1 if up_bar[0] else 0
    for i in range(1, len(c)):
        if up_bar[i]:
            run_up[i] = run_up[i - 1] + 1; run_down[i] = 0
        else:
            run_down[i] = run_down[i - 1] + 1; run_up[i] = 0
    run_up_max_20 = pd.Series(run_up).rolling(20).max().to_numpy()
    run_down_max_20 = pd.Series(run_down).rolling(20).max().to_numpy()

    gap_pct = (o - prev_c) / np.where(prev_c != 0, prev_c, 1.0) * 10_000
    gap_abs_mean_20 = pd.Series(np.abs(gap_pct)).rolling(20).mean().to_numpy()

    def lag(arr, k):
        return np.r_[arr[:k], arr[:-k]]

    mom_5  = (c - lag(c, 5))  / np.where(lag(c, 5)  != 0, lag(c, 5),  1.0) * 10_000
    mom_15 = (c - lag(c, 15)) / np.where(lag(c, 15) != 0, lag(c, 15), 1.0) * 10_000
    mom_30 = (c - lag(c, 30)) / np.where(lag(c, 30) != 0, lag(c, 30), 1.0) * 10_000
    mom_60 = (c - lag(c, 60)) / np.where(lag(c, 60) != 0, lag(c, 60), 1.0) * 10_000
    mom_120 = (c - lag(c, 120)) / np.where(lag(c, 120) != 0, lag(c, 120), 1.0) * 10_000

    vol_mean_200 = pd.Series(v).rolling(200).mean().to_numpy()
    vol_std_200 = pd.Series(v).rolling(200).std().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        volume_z = (v - vol_mean_200) / np.where(vol_std_200 > 0, vol_std_200, 1.0)
    volume_ma_ratio = v / np.where(vol_mean_200 > 0, vol_mean_200, 1.0)

    max_run_up_60 = pd.Series(c).rolling(60).apply(
        lambda x: (x.max() - x.iloc[0]) / max(x.iloc[0], 1e-9), raw=False
    ).to_numpy() * 10_000.0
    max_run_down_60 = pd.Series(c).rolling(60).apply(
        lambda x: (x.iloc[0] - x.min()) / max(x.iloc[0], 1e-9), raw=False
    ).to_numpy() * 10_000.0

    idx = bars.index
    et_hour = idx.hour.to_numpy()
    et_minute = idx.minute.to_numpy()
    minutes_into_session = np.where(
        (et_hour >= SESSION_START_HOUR_ET) & (et_hour < SESSION_END_HOUR_ET),
        (et_hour - SESSION_START_HOUR_ET) * 60 + et_minute, -1)
    day_of_week = idx.dayofweek.to_numpy()

    high_20 = pd.Series(h).rolling(20).max().to_numpy()
    low_20 = pd.Series(l).rolling(20).min().to_numpy()
    broke_high_5 = np.zeros(len(c), dtype=float)
    broke_low_5 = np.zeros(len(c), dtype=float)
    for i in range(5, len(c)):
        h5 = h[i-5:i+1].max()
        l5 = l[i-5:i+1].min()
        if high_20[i-5] is not None and np.isfinite(high_20[i-5]):
            broke_high_5[i] = 1.0 if h5 > high_20[i-5] else 0.0
            broke_low_5[i]  = 1.0 if l5 < low_20[i-5] else 0.0
    any_strategy_signal_30 = (pd.Series(broke_high_5).rolling(30).max().to_numpy()
                               + pd.Series(broke_low_5).rolling(30).max().to_numpy())
    max_move_10 = pd.Series(c).rolling(10).apply(lambda x: x.max() - x.min(), raw=False).to_numpy()
    big_move_10 = (max_move_10 >= 6.0).astype(float)

    return pd.DataFrame({
        "vol_bp_30": vol30, "eff_30": eff30,
        "vol_bp_60": vol60, "eff_60": eff60,
        "vol_bp_120": vol120, "eff_120": eff120,
        "vol_bp_240": vol240, "eff_240": eff240,
        "vol_bp_480": vol480, "eff_480": eff480,
        "vol_slope_10": vol_slope_10, "vol_slope_30": vol_slope_30,
        "vol_slope_60": vol_slope_60, "eff_slope_30": eff_slope_30,
        "atr14": atr14, "atr30": atr30,
        "range_pct_20": rng_pct_20, "range_pct_120": rng_pct_120,
        "body_ratio_20": body_ratio_20, "body_ratio_60": body_ratio_60,
        "abs_body_20": abs_body_20, "up_bar_pct_20": up_bar_pct_20,
        "run_up_max_20": run_up_max_20, "run_down_max_20": run_down_max_20,
        "gap_pct": gap_pct, "gap_abs_mean_20": gap_abs_mean_20,
        "mom_5": mom_5, "mom_15": mom_15, "mom_30": mom_30,
        "mom_60": mom_60, "mom_120": mom_120,
        "volume_z_20": volume_z, "volume_ma_ratio": volume_ma_ratio,
        "max_runup_60": max_run_up_60, "max_rundown_60": max_run_down_60,
        "et_hour": et_hour, "minutes_into_session": minutes_into_session,
        "day_of_week": day_of_week,
        "any_strategy_signal_30": any_strategy_signal_30,
        "big_move_10": big_move_10,
    }, index=idx)


def filter_ny_session(df: pd.DataFrame) -> pd.DataFrame:
    h = df.index.hour
    return df.loc[(h >= SESSION_START_HOUR_ET) & (h < SESSION_END_HOUR_ET)].copy()


# ─── Forward-walk simulators ─────────────────────────────────────────────

def simulate_trade(bh: np.ndarray, bl: np.ndarray, bc: np.ndarray,
                    start_idx: int, tp: float, sl: float, side: int) -> float:
    """Forward-walk a single hypothetical trade. Returns $ PnL at 1 MES."""
    if start_idx + 1 >= len(bc):
        return 0.0
    entry = bc[start_idx]
    end_idx = min(start_idx + 1 + PNL_LOOKAHEAD_BARS, len(bc))
    hs = bh[start_idx + 1 : end_idx]
    ls = bl[start_idx + 1 : end_idx]
    if len(hs) == 0:
        return 0.0
    if side > 0:
        tp_hits = np.where(hs >= entry + tp)[0]
        sl_hits = np.where(ls <= entry - sl)[0]
    else:
        tp_hits = np.where(ls <= entry - tp)[0]
        sl_hits = np.where(hs >= entry + sl)[0]
    tp_i = tp_hits[0] if len(tp_hits) else 1 << 30
    sl_i = sl_hits[0] if len(sl_hits) else 1 << 30
    if tp_i == 1 << 30 and sl_i == 1 << 30:
        last_c = bc[end_idx - 1]
        pts = (last_c - entry) if side > 0 else (entry - last_c)
        return pts * MES_PT_VALUE
    if tp_i < sl_i:
        return tp * MES_PT_VALUE
    return -sl * MES_PT_VALUE


def simulate_be_trade(bh: np.ndarray, bl: np.ndarray, bc: np.ndarray,
                       start_idx: int, tp: float, sl: float, be_trigger: float,
                       side: int, be_on: bool) -> float:
    """Walk forward with BE-arm semantics. BE-OFF = classic TP/SL first-hit.
    BE-ON = if MFE reaches be_trigger, stop moves to entry; subsequent
    touch of entry = $0 exit; TP still $ tp × $5."""
    if start_idx + 1 >= len(bc): return 0.0
    entry = bc[start_idx]
    end_idx = min(start_idx + 1 + PNL_LOOKAHEAD_BARS, len(bc))
    hs = bh[start_idx + 1 : end_idx]
    ls = bl[start_idx + 1 : end_idx]
    if len(hs) == 0: return 0.0
    if side > 0:
        tp_px = entry + tp; sl_px = entry - sl
        be_trigger_px = entry + be_trigger; be_stop_px = entry
        if not be_on:
            return simulate_trade(bh, bl, bc, start_idx, tp, sl, side)
        be_armed = False
        for b in range(len(hs)):
            if not be_armed:
                if ls[b] <= sl_px: return -sl * MES_PT_VALUE
                if hs[b] >= tp_px: return tp * MES_PT_VALUE
                if hs[b] >= be_trigger_px: be_armed = True
            else:
                if ls[b] <= be_stop_px: return 0.0
                if hs[b] >= tp_px: return tp * MES_PT_VALUE
        last = bc[end_idx - 1]
        return (last - entry) * MES_PT_VALUE
    else:
        tp_px = entry - tp; sl_px = entry + sl
        be_trigger_px = entry - be_trigger; be_stop_px = entry
        if not be_on:
            return simulate_trade(bh, bl, bc, start_idx, tp, sl, side)
        be_armed = False
        for b in range(len(hs)):
            if not be_armed:
                if hs[b] >= sl_px: return -sl * MES_PT_VALUE
                if ls[b] <= tp_px: return tp * MES_PT_VALUE
                if ls[b] <= be_trigger_px: be_armed = True
            else:
                if hs[b] >= be_stop_px: return 0.0
                if ls[b] <= tp_px: return tp * MES_PT_VALUE
        last = bc[end_idx - 1]
        return (entry - last) * MES_PT_VALUE


def stats(arr) -> Dict[str, float]:
    """Return {n, pnl, avg, dd, std, sharpe} for an array of per-trade PnL."""
    if len(arr) == 0:
        return {"n": 0, "pnl": 0.0, "avg": 0.0, "dd": 0.0, "std": 0.0, "sharpe": 0.0}
    a = np.asarray(arr, dtype=float)
    cum = np.cumsum(a)
    peak = np.maximum.accumulate(cum)
    std = float(a.std()) or 1e-9
    return {
        "n": int(len(a)),
        "pnl": float(a.sum()),
        "avg": float(a.mean()),
        "dd": float(np.max(peak - cum)),
        "std": std,
        "sharpe": float(a.mean() / std),
    }


def sample_weights_balanced(y, cost_ratio: float = 1.5) -> np.ndarray:
    """Class-balanced sample weights with cost penalty on the minority class."""
    counts = Counter(y)
    base_w = {lbl: len(y) / (2 * counts[lbl]) for lbl in counts}
    minority = min(counts, key=counts.get)
    base_w[minority] *= cost_ratio
    return np.array([base_w[lbl] for lbl in y])
