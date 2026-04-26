#!/usr/bin/env python3
"""Train Kalshi overlay ML v4 — enriched feature set (Option C).

Adds to v2/v3:
  A. Cross-strategy queue state (parsed from same live/replay logs)
     - counts of AetherFlow, RegimeAdaptive STRATEGY_SIGNAL fires in last 10m
       split by side
     - status of most recent AF / RA signal (CANDIDATE / BLOCKED)
     - minutes since last [TRADE_PLACED] event
  B. NQ cross-asset from mnq_master_continuous.parquet (now extended to Apr-24)
     - NQ vol_bp at 60/120-bar lookbacks
     - NQ mom at 15/30/60-bar lags
     - NQ-ES correlation (rolling 60 bars)
     - NQ-ES spread divergence (pct)
  C. VIX daily proxy (from data/vix_daily.parquet)
     - VIX close on the event's date
     - VIX change over last 5 days

Uses regression on forward_pnl with binary override (same action space as v3).

Ship gates unchanged — all 5 must pass.
"""
from __future__ import annotations

import argparse, json, logging, pickle, re, sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    ROOT, FEATURE_COLS_40, MES_PT_VALUE, DEFAULT_TP, DEFAULT_SL,
    load_continuous_bars, build_feature_frame,
    stats,
)
from train_kalshi_v2 import (
    parse_log_events, simulate_trade_horizon, add_v2_features,
    fill_intraday_pnl, V2_EXTRA_FEATURES,
    recency_weights,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("train_kalshi_v4")

LABEL_MARGIN_USD = 15.0
OOS_START = "2026-01-27"
OOS_END = "2026-04-24"

# ─── Cross-strategy queue state parser ───────────────────────────────────

RE_BAR = re.compile(r"Bar: (?P<mts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) ET")
RE_STRATEGY_SIGNAL = re.compile(
    r"\[STRATEGY_SIGNAL\][^|]*\|\s*strategy=(?P<strategy>\S+)\s*\|\s*side=(?P<side>LONG|SHORT)"
    r"[^|]*(?:\|[^|]*?status=(?P<status>\S+))?"
)
RE_TRADE_PLACED = re.compile(r"\[TRADE_PLACED\]")


def parse_cross_strategy_signals(log_paths: list[Path]) -> pd.DataFrame:
    """Return a DataFrame of strategy signals keyed by (ts, strategy, side).
    Columns: ts (tz-aware), strategy, side, status, is_trade_placed."""
    rows = []
    for p in log_paths:
        last_bar_mts = None
        with p.open(errors="ignore") as fh:
            for line in fh:
                bm = RE_BAR.search(line)
                if bm:
                    last_bar_mts = bm.group("mts")
                    continue
                ss = RE_STRATEGY_SIGNAL.search(line)
                if ss and last_bar_mts is not None:
                    rows.append({
                        "ts": last_bar_mts, "strategy": ss.group("strategy"),
                        "side": ss.group("side"), "status": ss.group("status") or "",
                        "kind": "strategy_signal",
                    })
                    continue
                if RE_TRADE_PLACED.search(line) and last_bar_mts is not None:
                    rows.append({
                        "ts": last_bar_mts, "strategy": "ANY", "side": "",
                        "status": "PLACED", "kind": "trade_placed",
                    })
    if not rows:
        return pd.DataFrame(columns=["ts", "strategy", "side", "status", "kind"])
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"]).dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT")
    df = df.dropna(subset=["ts"]).sort_values("ts")
    return df


def compute_cross_strategy_features(event_df: pd.DataFrame,
                                    signals_df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized: for each Kalshi event, count signals in rolling windows.

    Strategy: pre-build per-subset sorted timestamp arrays, then use
    np.searchsorted for each window range. O((events+signals) log n).
    """
    event_df = event_df.copy()
    default_zero_cols = ["xs_af_long_10m", "xs_af_short_10m",
                         "xs_ra_long_10m", "xs_ra_short_10m",
                         "xs_af_any_30m", "xs_ra_any_30m"]
    for c in default_zero_cols:
        event_df[c] = 0
    event_df["xs_min_since_trade"] = 999

    if signals_df.empty:
        return event_df

    # Event timestamps in ns (tz-naive NY)
    ev_ns = np.array([
        np.datetime64(ts.tz_convert("America/New_York").tz_localize(None), "ns")
        for ts in event_df.index
    ], dtype="datetime64[ns]")

    # Build subset sorted-timestamp arrays
    def subset_ts(mask: np.ndarray) -> np.ndarray:
        arr = signals_df["ts"].values.astype("datetime64[ns]")[mask]
        arr.sort()
        return arr

    sig_kind = signals_df["kind"].values
    sig_strat_l = np.array([str(s).lower() for s in signals_df["strategy"].values])
    sig_side = signals_df["side"].values

    is_sig = (sig_kind == "strategy_signal")
    ts_af_long = subset_ts(is_sig & (np.char.find(sig_strat_l, "aetherflow") >= 0)
                           & (sig_side == "LONG"))
    ts_af_short = subset_ts(is_sig & (np.char.find(sig_strat_l, "aetherflow") >= 0)
                            & (sig_side == "SHORT"))
    ts_ra_long = subset_ts(is_sig & (np.char.find(sig_strat_l, "regimeadaptive") >= 0)
                           & (sig_side == "LONG"))
    ts_ra_short = subset_ts(is_sig & (np.char.find(sig_strat_l, "regimeadaptive") >= 0)
                            & (sig_side == "SHORT"))
    ts_af_any = subset_ts(is_sig & (np.char.find(sig_strat_l, "aetherflow") >= 0))
    ts_ra_any = subset_ts(is_sig & (np.char.find(sig_strat_l, "regimeadaptive") >= 0))
    ts_placed = subset_ts(sig_kind == "trade_placed")

    def count_in_window(ts_sub: np.ndarray, window_s: int) -> np.ndarray:
        """Vectorized: # signals in [event-window, event) for each event."""
        lo = ev_ns - np.timedelta64(window_s, "s")
        right = np.searchsorted(ts_sub, ev_ns, side="left")   # < event
        left = np.searchsorted(ts_sub, lo, side="left")        # >= lo
        return (right - left).astype(int)

    event_df["xs_af_long_10m"] = count_in_window(ts_af_long, 600)
    event_df["xs_af_short_10m"] = count_in_window(ts_af_short, 600)
    event_df["xs_ra_long_10m"] = count_in_window(ts_ra_long, 600)
    event_df["xs_ra_short_10m"] = count_in_window(ts_ra_short, 600)
    event_df["xs_af_any_30m"] = count_in_window(ts_af_any, 1800)
    event_df["xs_ra_any_30m"] = count_in_window(ts_ra_any, 1800)

    # Minutes since last [TRADE_PLACED] (cap 999)
    if len(ts_placed):
        idx = np.searchsorted(ts_placed, ev_ns, side="left") - 1
        has_prior = idx >= 0
        mins = np.full(len(ev_ns), 999.0)
        prior_ts = ts_placed[np.clip(idx, 0, len(ts_placed)-1)]
        delta_min = (ev_ns[has_prior] - prior_ts[has_prior]) / np.timedelta64(1, "m")
        mins[has_prior] = np.clip(delta_min.astype(float), 0, 999)
        event_df["xs_min_since_trade"] = mins.astype(int)
    return event_df


# ─── NQ cross-asset features ────────────────────────────────────────────

def load_nq_bars(start: str, end: str) -> pd.DataFrame:
    p = ROOT / "data" / "mnq_master_continuous.parquet"
    nq = pd.read_parquet(p)
    lo = pd.Timestamp(start, tz=nq.index.tz)
    hi = pd.Timestamp(end, tz=nq.index.tz)
    return nq.loc[(nq.index >= lo) & (nq.index <= hi)].sort_index()


def build_nq_features(nq: pd.DataFrame, es_close_by_ts: dict) -> pd.DataFrame:
    """Build per-minute NQ features.
    Returns DataFrame indexed by ts with columns:
      nq_vol_bp_60, nq_vol_bp_120, nq_mom_15, nq_mom_30, nq_mom_60,
      nq_es_corr_60, nq_es_ret_spread_30.
    """
    c = nq["close"].to_numpy(float)
    idx = nq.index
    # vol_bp over window
    def vol_bp(win):
        out = np.full(len(c), np.nan)
        for i in range(win, len(c)):
            p = c[i - win : i + 1]
            rets = (p[1:] - p[:-1]) / p[:-1]
            if len(rets) == 0: continue
            mean = rets.mean()
            var = ((rets - mean) ** 2).sum() / max(1, len(rets) - 1)
            out[i] = (var ** 0.5) * 10_000.0
        return out
    nq_vol_60 = vol_bp(60)
    nq_vol_120 = vol_bp(120)
    # momentum
    def mom(lag):
        out = np.full(len(c), np.nan)
        for i in range(lag, len(c)):
            if c[i - lag] != 0:
                out[i] = (c[i] - c[i - lag]) / c[i - lag] * 10_000
        return out
    nq_mom_15 = mom(15)
    nq_mom_30 = mom(30)
    nq_mom_60 = mom(60)

    # ES series aligned to NQ timestamps for corr / divergence
    es_close = np.array([es_close_by_ts.get(ts, np.nan) for ts in idx], dtype=float)

    # Rolling 60-bar corr of returns
    def rolling_corr(a, b, win=60):
        ar = pd.Series(a).pct_change().to_numpy()
        br = pd.Series(b).pct_change().to_numpy()
        s = pd.Series(ar).rolling(win).corr(pd.Series(br))
        return s.to_numpy()
    corr_60 = rolling_corr(c, es_close, 60)

    # Spread: pct diff of rolling 30-bar returns
    nq_ret_30 = pd.Series(c).pct_change(30).to_numpy() * 10_000
    es_ret_30 = pd.Series(es_close).pct_change(30).to_numpy() * 10_000
    spread_30 = nq_ret_30 - es_ret_30

    df = pd.DataFrame({
        "nq_vol_bp_60": nq_vol_60, "nq_vol_bp_120": nq_vol_120,
        "nq_mom_15": nq_mom_15, "nq_mom_30": nq_mom_30, "nq_mom_60": nq_mom_60,
        "nq_es_corr_60": corr_60, "nq_es_ret_spread_30": spread_30,
    }, index=idx)
    return df


# ─── VIX features ────────────────────────────────────────────────────────

def load_vix() -> pd.DataFrame:
    p = ROOT / "data" / "vix_daily.parquet"
    v = pd.read_parquet(p)
    v = v.sort_index().copy()
    v["vix_close"] = v["close"]
    v["vix_chg_5d"] = v["close"] - v["close"].shift(5)
    v["vix_chg_1d"] = v["close"] - v["close"].shift(1)
    return v[["vix_close", "vix_chg_1d", "vix_chg_5d"]]


# ─── Dataset builder ─────────────────────────────────────────────────────

V4_EXTRA_XS = [
    "xs_af_long_10m", "xs_af_short_10m", "xs_ra_long_10m", "xs_ra_short_10m",
    "xs_af_any_30m", "xs_ra_any_30m", "xs_min_since_trade",
]
V4_EXTRA_NQ = [
    "nq_vol_bp_60", "nq_vol_bp_120",
    "nq_mom_15", "nq_mom_30", "nq_mom_60",
    "nq_es_corr_60", "nq_es_ret_spread_30",
]
V4_EXTRA_VIX = ["vix_close", "vix_chg_1d", "vix_chg_5d"]

# Build on top of v2's feature set
from train_kalshi_v2 import FEATURE_COLS_KALSHI_V1, FEATURE_COLS_KALSHI_V2

def _dedup(seq):
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

FEATURE_COLS_V4 = _dedup(FEATURE_COLS_KALSHI_V2 + V4_EXTRA_XS + V4_EXTRA_NQ + V4_EXTRA_VIX)


def build_v4_dataset(events: list, bars: pd.DataFrame,
                     nq_features: pd.DataFrame, vix_df: pd.DataFrame,
                     signals_df: pd.DataFrame,
                     label_horizon_min: int) -> pd.DataFrame:
    feats = build_feature_frame(bars)
    feats = feats.loc[feats[FEATURE_COLS_40].notna().all(axis=1)].copy()
    feats_minute_idx = {pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M"): ts for ts in feats.index}
    log.info("feature frame rows: %d", len(feats))

    c_bars = bars["close"].to_numpy(float)
    h_bars = bars["high"].to_numpy(float)
    l_bars = bars["low"].to_numpy(float)
    idx_pos = {ts: i for i, ts in enumerate(bars.index)}

    rows = []
    dropped_no_bar = 0
    dropped_ambiguous = 0
    for e in events:
        mts_key = e["market_ts"][:16]
        ts = feats_minute_idx.get(mts_key)
        if ts is None:
            dropped_no_bar += 1
            continue
        pos = idx_pos.get(ts)
        if pos is None:
            dropped_no_bar += 1
            continue
        side_sign = 1 if e["side"] == "LONG" else -1
        pnl = simulate_trade_horizon(h_bars, l_bars, c_bars, int(pos),
                                     DEFAULT_TP, DEFAULT_SL, side_sign, label_horizon_min)
        if abs(pnl) < LABEL_MARGIN_USD:
            dropped_ambiguous += 1
            continue
        row = {**e, "_ts": ts}
        for col in FEATURE_COLS_40:
            row[col] = feats.loc[ts, col]
        row["side_sign"] = side_sign
        row["is_de3"] = 1 if "dynamicengine" in e["strategy"].lower() else 0
        row["is_ra"]  = 1 if "regimeadaptive" in e["strategy"].lower() else 0
        row["is_ml"]  = 1 if "mlphysics" in e["strategy"].lower() else 0
        row["settlement_hour"] = int(pd.Timestamp(mts_key + ":00").hour)
        row["minutes_into_session"] = int(
            (pd.Timestamp(mts_key + ":00").hour - 9) * 60 +
             pd.Timestamp(mts_key + ":00").minute
        )
        row["role_forward"] = 1 if "forward" in e["role"] else 0
        row["role_background"] = 1 if e["role"] == "background" else 0
        row["role_balanced"] = 1 if e["role"] == "balanced" else 0
        row["label"] = "pass" if pnl > 0 else "block"
        row["forward_pnl"] = pnl
        row["rule_decision_pass_int"] = 1 if e["rule_decision"] == "PASS" else 0
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty: return df
    df = df.set_index("_ts")
    df = fill_intraday_pnl(df)
    log.info("labeled rows: %d  dropped (no bar): %d  dropped (ambiguous): %d",
             len(df), dropped_no_bar, dropped_ambiguous)

    # Cross-strategy features
    log.info("computing cross-strategy features...")
    df = compute_cross_strategy_features(df, signals_df)

    # NQ features — join on nearest timestamp
    log.info("joining NQ features...")
    nq_reindexed = nq_features.reindex(df.index, method="ffill", tolerance=pd.Timedelta("5min"))
    for col in V4_EXTRA_NQ:
        df[col] = nq_reindexed[col].values
    # Fill missing NQ features with neutral values
    for col in V4_EXTRA_NQ:
        df[col] = df[col].fillna(df[col].median() if df[col].notna().any() else 0.0)

    # VIX — daily, forward-fill on date
    log.info("joining VIX features...")
    vix_lookup = vix_df.copy()
    vix_lookup.index = vix_lookup.index.tz_convert(df.index.tz) if vix_lookup.index.tz else vix_lookup.index
    # normalize index to date
    vix_by_date = {ts.date(): row for ts, row in vix_lookup.iterrows()}
    for col in V4_EXTRA_VIX:
        vals = []
        for ts in df.index:
            d = ts.date()
            # walk back up to 5 days to find a trading day
            val = None
            for off in range(6):
                test = d - pd.Timedelta(days=off).to_pytimedelta() if hasattr(pd.Timedelta(days=off), "to_pytimedelta") else d
                row = vix_by_date.get(d)
                if row is not None:
                    val = row.get(col)
                    break
                d = d - pd.Timedelta(days=1).to_pytimedelta() if hasattr(pd.Timedelta(days=1), "to_pytimedelta") else None
                if d is None: break
            vals.append(val if val is not None else np.nan)
        df[col] = vals
    # Fill remaining missing VIX with median
    for col in V4_EXTRA_VIX:
        if df[col].isna().any():
            med = df[col].median()
            df[col] = df[col].fillna(med if pd.notna(med) else 0.0)

    log.info("class dist: %s", dict(Counter(df["label"])))
    log.info("rule × label cross-tab:")
    for r_dec in ("PASS", "BLOCK"):
        for lbl in ("pass", "block"):
            n = int(((df["rule_decision"] == r_dec) & (df["label"] == lbl)).sum())
            log.info("  rule=%s × label=%s: %d", r_dec, lbl, n)
    return df


def eval_regressor(pred_pnl, rule_decision, y_true, pnl_if_passed,
                   pass_thr, block_thr):
    pass_override = (rule_decision == "BLOCK") & (pred_pnl >= pass_thr)
    block_override = (rule_decision == "PASS") & (pred_pnl <= -block_thr)
    final_pass = np.where(
        pass_override, True,
        np.where(block_override, False, rule_decision == "PASS"),
    )
    final_pnl = np.where(final_pass, pnl_if_passed, 0.0)
    ml_st = stats(final_pnl)
    n_pass = int(final_pass.sum())
    n_new_pass = int(pass_override.sum())
    n_new_block = int(block_override.sum())
    new_pass_pnls = pnl_if_passed[pass_override]
    new_block_pnls = pnl_if_passed[block_override]
    new_pass_wr = (100 * (new_pass_pnls > 0).sum() / len(new_pass_pnls)) if len(new_pass_pnls) else 0.0
    new_block_wr = (100 * (new_block_pnls <= 0).sum() / len(new_block_pnls)) if len(new_block_pnls) else 0.0
    return {
        "pass_thr": pass_thr, "block_thr": block_thr,
        "n_pass": n_pass, "n_new_pass": n_new_pass, "n_new_block": n_new_block,
        "new_pass_wr": new_pass_wr, "new_block_wr": new_block_wr,
        "pnl": ml_st["pnl"], "dd": ml_st["dd"], "avg": ml_st["avg"],
    }


def run_v4_config_with_cache(df: pd.DataFrame,
                             label_horizon_min: int, half_life_days: float,
                             train_start_date: Optional[str],
                             oos_start: str, oos_end: str,
                             seed: int = 42) -> Optional[dict]:
    log.info("")
    log.info("=" * 74)
    log.info("V4 CONFIG  horizon=%dmin  hl=%.0fd  trstart=%s",
             label_horizon_min, half_life_days, train_start_date or "full")
    log.info("=" * 74)
    if len(df) < 500:
        log.warning("only %d labeled rows — skip", len(df))
        return None

    oos_start_ts = pd.Timestamp(oos_start, tz=df.index.tz)
    oos_end_ts = pd.Timestamp(oos_end, tz=df.index.tz) + pd.Timedelta(days=1)
    tr = df.loc[df.index < oos_start_ts]
    if train_start_date is not None:
        tr = tr.loc[tr.index >= pd.Timestamp(train_start_date, tz=df.index.tz)]
    oos = df.loc[(df.index >= oos_start_ts) & (df.index <= oos_end_ts)]
    log.info("train: %d  OOS: %d", len(tr), len(oos))
    if len(tr) < 200 or len(oos) < 50:
        log.warning("below floors — skip")
        return None

    # Clean NaNs in features
    X_tr_df = tr[FEATURE_COLS_V4].copy()
    X_oos_df = oos[FEATURE_COLS_V4].copy()
    # Median impute any remaining NaNs (shared median from training)
    med = X_tr_df.median(numeric_only=True)
    for col in FEATURE_COLS_V4:
        if X_tr_df[col].isna().any() or X_oos_df[col].isna().any():
            m = med.get(col, 0.0)
            if pd.isna(m): m = 0.0
            X_tr_df[col] = X_tr_df[col].fillna(m)
            X_oos_df[col] = X_oos_df[col].fillna(m)
    X_tr = X_tr_df.to_numpy()
    X_oos = X_oos_df.to_numpy()

    recency_w = recency_weights(tr.index, half_life_days)
    y_pnl = tr["forward_pnl"].to_numpy()
    rule_decision = oos["rule_decision"].to_numpy()
    pnl_if_passed = oos["forward_pnl"].to_numpy()
    y_true = oos["label"].to_numpy()
    rule_pnl_arr = np.where(rule_decision == "PASS", pnl_if_passed, 0.0)
    rule_st = stats(rule_pnl_arr)
    oracle_pnl_arr = np.where(y_true == "pass", pnl_if_passed, 0.0)
    oracle_st = stats(oracle_pnl_arr)
    log.info("rule baseline: PnL=$%+.2f DD=$%.0f  | oracle: PnL=$%+.2f  headroom=$%+.2f",
             rule_st["pnl"], rule_st["dd"], oracle_st["pnl"],
             oracle_st["pnl"] - rule_st["pnl"])

    reg = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.05, max_depth=6,
        l2_regularization=1.0, min_samples_leaf=30, random_state=seed,
    )
    reg.fit(X_tr, y_pnl, sample_weight=recency_w)
    pred_pnl = reg.predict(X_oos)

    corr = np.corrcoef(pred_pnl, pnl_if_passed)[0, 1]
    log.info("OOS pred-vs-actual Pearson: %.3f", corr)
    log.info("pred PnL percentiles: p25=%+.2f p50=%+.2f p75=%+.2f p90=%+.2f",
             np.percentile(pred_pnl, 25), np.percentile(pred_pnl, 50),
             np.percentile(pred_pnl, 75), np.percentile(pred_pnl, 90))

    # Feature importance (permutation-free: use built-in)
    try:
        fi = reg.feature_importances_  # HGB doesn't expose this natively
    except AttributeError:
        fi = None

    results = []
    thr_grid = [
        (5.0, 5.0), (7.5, 5.0), (10.0, 5.0), (10.0, 7.5),
        (12.5, 5.0), (12.5, 10.0), (15.0, 7.5), (15.0, 10.0), (15.0, 15.0),
        (20.0, 10.0), (20.0, 15.0), (25.0, 10.0), (25.0, 15.0), (30.0, 15.0),
    ]
    log.info("%5s %5s %8s %8s %10s %8s %8s %7s gates",
             "p_thr", "b_thr", "n_pass", "new_psn", "newPassWR", "n_new_blk",
             "pnl", "capt")
    for (p_thr, b_thr) in thr_grid:
        r = eval_regressor(pred_pnl, rule_decision, y_true, pnl_if_passed, p_thr, b_thr)
        lift = r["pnl"] - rule_st["pnl"]
        dd_over_pnl = (r["dd"] / r["pnl"] * 100.0) if r["pnl"] > 0 else float("inf")
        headroom = oracle_st["pnl"] - rule_st["pnl"]
        capt = (lift / headroom * 100.0) if headroom > 0 else 0.0
        gates = {
            "pnl_ok":        lift > 0,
            "dd_ratio_ok":   dd_over_pnl <= 30.0,
            "n_ok":          len(oos) >= 50,
            "new_pass_wr_ok":(r["new_pass_wr"] >= 50.0) if r["n_new_pass"] >= 5 else True,
            "capt_ok":       capt >= 20.0,
        }
        ok = all(gates.values())
        log.info("%5.1f %5.1f %8d %8d %9.2f%% %8d $%+7.2f %6.2f%%  %d/5%s",
                 p_thr, b_thr, r["n_pass"], r["n_new_pass"], r["new_pass_wr"],
                 r["n_new_block"], r["pnl"], capt, sum(gates.values()),
                 " SHIP" if ok else "")
        r.update({"lift": lift, "dd_over_pnl": dd_over_pnl, "capt_pct": capt,
                  "gates": gates, "ships": ok})
        results.append(r)

    shippers = [r for r in results if r["ships"]]
    best = max(shippers, key=lambda r: r["lift"]) if shippers else max(results, key=lambda r: r["lift"])
    return {
        "horizon": label_horizon_min, "half_life": half_life_days,
        "train_start": train_start_date,
        "rule_baseline": rule_st, "oracle": oracle_st,
        "n_train": len(tr), "n_oos": len(oos),
        "corr": corr,
        "all": results, "best": best,
        "model": reg if best["ships"] else None,
        "feature_cols": FEATURE_COLS_V4,
        "median_imputes": {c: float(med.get(c, 0.0)) for c in FEATURE_COLS_V4},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oos-start", default=OOS_START)
    ap.add_argument("--oos-end", default=OOS_END)
    ap.add_argument("--out-dir", default=str(ROOT / "artifacts/regime_ml_kalshi_v4"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--horizons", nargs="+", type=int, default=[15, 30, 60])
    ap.add_argument("--half-lives", nargs="+", type=float, default=[90.0, 120.0])
    ap.add_argument("--train-starts", nargs="+", default=["full", "2025-05-01"])
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("=== loading logs ===")
    log_paths = []
    rdir = ROOT / "backtest_reports/full_live_replay"
    for m in "2025_01 2025_02 2025_03 2025_04 2025_05 2025_06 2025_07 2025_08 2025_09 2025_10 2025_11 2025_12".split():
        p = rdir / m / "topstep_live_bot.log"
        if p.exists() and p.stat().st_size > 50_000:
            log_paths.append(p)
    live = ROOT / "topstep_live_bot.log"
    if live.exists(): log_paths.append(live)

    all_events = []
    for p in log_paths:
        ev = parse_log_events(p)
        log.info("  %s → %d events",
                 p.name if p.parent == ROOT else f"{p.parent.name}/{p.name}", len(ev))
        all_events.extend(ev)
    log.info("total Kalshi events (AF excl.): %d", len(all_events))
    all_events = add_v2_features(all_events)

    log.info("=== parsing cross-strategy signals ===")
    signals_df = parse_cross_strategy_signals(log_paths)
    log.info("cross-strategy signals: %d  (unique strategies: %s)",
             len(signals_df),
             dict(Counter(signals_df["strategy"])) if not signals_df.empty else {})

    all_mts = [e["market_ts"] for e in all_events]
    start = pd.Timestamp(min(all_mts)).strftime("%Y-%m-%d")
    end = (pd.Timestamp(max(all_mts)) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    log.info("=== loading ES bars ===")
    es_bars = load_continuous_bars(start, end)
    log.info("ES bars: %d  range %s -> %s", len(es_bars), es_bars.index.min(), es_bars.index.max())

    log.info("=== loading NQ bars ===")
    nq_bars = load_nq_bars(start, end)
    log.info("NQ bars: %d  range %s -> %s", len(nq_bars), nq_bars.index.min(), nq_bars.index.max())

    log.info("=== building NQ features ===")
    es_close_by_ts = dict(zip(es_bars.index, es_bars["close"]))
    # Align NQ to ES minute grid so correlations compute on common timeline
    # Reindex NQ onto ES index (ffill close within 5 min)
    nq_aligned = nq_bars.reindex(es_bars.index, method="ffill", tolerance=pd.Timedelta("5min"))
    nq_features = build_nq_features(nq_aligned, es_close_by_ts)
    log.info("NQ features: %d rows (with non-nan vol_bp_60: %d)",
             len(nq_features), nq_features["nq_vol_bp_60"].notna().sum())

    log.info("=== loading VIX ===")
    vix_df = load_vix()
    log.info("VIX: %d rows range %s -> %s", len(vix_df), vix_df.index.min(), vix_df.index.max())

    # Cache datasets by horizon so we don't rebuild features 12 times
    dataset_cache: dict[int, pd.DataFrame] = {}
    for h in args.horizons:
        log.info("=== building dataset for horizon=%d ===", h)
        df = build_v4_dataset(all_events, es_bars, nq_features, vix_df,
                              signals_df, h)
        dataset_cache[h] = df

    configs = []
    for ts_ in args.train_starts:
        ts_val = None if ts_ == "full" else ts_
        for h in args.horizons:
            for hl in args.half_lives:
                configs.append((h, hl, ts_val))
    log.info("=== running %d configs (using cached datasets) ===", len(configs))

    runs = []
    for cfg in configs:
        r = run_v4_config_with_cache(dataset_cache[cfg[0]], cfg[0], cfg[1], cfg[2],
                                     args.oos_start, args.oos_end, seed=args.seed)
        if r: runs.append(r)

    log.info("\n%s\nV4 SWEEP SUMMARY\n%s", "═" * 140, "═" * 140)
    log.info("%4s %4s %10s %5s %7s %10s %8s %8s %+8s %7s %s",
             "hrz", "hl", "trstart", "gts", "corr",
             "params", "newPs", "WR%", "lift$", "capt%", "ship?")
    for r in runs:
        b = r["best"]
        params = f"p={b['pass_thr']:.1f}/b={b['block_thr']:.1f}"
        log.info("%4d %4.0f %10s %5d %7.3f %10s %8d %6.2f%% %+8.2f %6.2f%% %s",
                 r["horizon"], r["half_life"], str(r["train_start"] or "full")[:10],
                 sum(b["gates"].values()), r["corr"], params,
                 b["n_new_pass"], b["new_pass_wr"], b["lift"], b["capt_pct"],
                 "SHIP" if b["ships"] else "-")

    shippers = [r for r in runs if r["best"]["ships"]]
    if not shippers:
        (out_dir / "sweep_summary.json").write_text(json.dumps({
            "verdict": "KILL", "reason": "no v4 config passes all 5 gates",
            "runs": [{k: v for k, v in r.items() if k not in ("model", "median_imputes")}
                     for r in runs],
        }, indent=2, default=str))
        log.warning("[KILL] no v4 config passes — summary written")
        return 1

    best_run = max(shippers, key=lambda r: r["best"]["lift"])
    b = best_run["best"]
    payload = {
        "model_kind": "regression",
        "reg": best_run["model"],
        "feature_cols": FEATURE_COLS_V4,
        "median_imputes": best_run["median_imputes"],
        "decision": b,
        "horizon_min": best_run["horizon"],
        "half_life_days": best_run["half_life"],
        "train_start": best_run["train_start"],
        "rule_baseline_oos": best_run["rule_baseline"],
        "oracle_oos": best_run["oracle"],
        "n_train": best_run["n_train"], "n_oos": best_run["n_oos"],
        "corr_oos": best_run["corr"],
        "seed": args.seed,
    }
    with (out_dir / "model.pkl").open("wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    (out_dir / "model_meta.json").write_text(json.dumps({
        k: v for k, v in payload.items() if k != "reg"
    }, indent=2, default=str))
    log.info("[SHIP] horizon=%d hl=%.0f  lift=$%+.2f  WR=%.2f%%  capt=%.2f%%  corr=%.3f",
             best_run["horizon"], best_run["half_life"],
             b["lift"], b["new_pass_wr"], b["capt_pct"], best_run["corr"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
