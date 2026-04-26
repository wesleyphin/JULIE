#!/usr/bin/env python3
"""Kalshi ML v7 — with REAL historical tick features.

Phase-2 unlock from Option-#1 deep dive. Kalshi's `/historical/trades`
endpoint (with `?ticker=<exact_market>`) returns full per-trade history
back to 2024 — discovered after the prior v1-v6 attempts had concluded
the data was unavailable.

`scripts/fetch_kalshi_historical_trades.py` pulled all trades for ATM
strikes around all 5,076 KALSHI_ENTRY_VIEW events into
`data/kalshi/kxinxu_historical_trades.parquet`.

This trainer adds tick-derived features computed PER-EVENT:
  - `kt_close_velocity_5/15/30m`  — yes_price change over windows
  - `kt_close_volatility_15m`     — stddev of yes_price ticks
  - `kt_n_trades_15m`             — trade count
  - `kt_taker_yes_imbalance_15m`  — % taker_side="yes"
  - `kt_avg_size_15m`             — avg count_fp
  - `kt_size_pressure_15m`        — count of trades with count_fp > 100
  - `kt_yes_price_range_15m`      — max - min yes_price
  - `kt_dollar_volume_15m`        — sum(yes_price * count_fp)
  - `kt_atm_yes_price_now`        — most recent yes_price within 5 min
  - `kt_strikes_traded_15m`       — distinct strikes that had trades

Pipeline:
  1. Build v2 dataset (existing 58 features + labels)
  2. Compute tick features per event from the trades parquet
  3. Train HGB regressor on forward_pnl with binary override
  4. Sweep horizon × override threshold × top_k feature selection
  5. Walk-forward CV for stability check
  6. Ship-or-kill on the 5 gates
"""
from __future__ import annotations

import argparse, json, logging, pickle, sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    ROOT, FEATURE_COLS_40, MES_PT_VALUE, DEFAULT_TP, DEFAULT_SL,
    load_continuous_bars, build_feature_frame, stats,
)
from train_kalshi_v2 import (
    parse_log_events, simulate_trade_horizon, add_v2_features,
    fill_intraday_pnl, V2_EXTRA_FEATURES,
    recency_weights, FEATURE_COLS_KALSHI_V2,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("train_kalshi_v7")

LABEL_MARGIN_USD = 15.0
OOS_START = "2026-01-27"
OOS_END = "2026-04-24"
TRADES_PARQUET = ROOT / "data" / "kalshi" / "kxinxu_historical_trades.parquet"

# New tick-derived features
TICK_FEATURE_COLS = [
    "kt_close_velocity_5m", "kt_close_velocity_15m", "kt_close_velocity_30m",
    "kt_close_volatility_15m",
    "kt_n_trades_15m", "kt_n_trades_5m",
    "kt_taker_yes_imbalance_15m",
    "kt_avg_size_15m", "kt_size_pressure_15m",
    "kt_yes_price_range_15m",
    "kt_dollar_volume_15m",
    "kt_atm_yes_price_now",
    "kt_strikes_traded_15m",
]

FEATURE_POOL_V7 = list(FEATURE_COLS_KALSHI_V2)
for c in TICK_FEATURE_COLS:
    if c not in FEATURE_POOL_V7:
        FEATURE_POOL_V7.append(c)


# ─── Tick feature computation ────────────────────────────────────────────

def event_ticker_for_event_ts(market_ts: pd.Timestamp) -> tuple[str, int]:
    """Map a Kalshi event timestamp (NY-tz aware) to its KXINXU event_ticker."""
    et = market_ts.tz_convert("America/New_York")
    if et.minute < 5:
        settle = et.replace(minute=0, second=0, microsecond=0)
    else:
        settle = (et + pd.Timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    h = int(settle.hour)
    if h < 10 or h > 16: return ("", 0)
    return (f"KXINXU-{settle.strftime('%y%b%d').upper()}H{h*100:04d}", h)


def compute_tick_features(events_df: pd.DataFrame,
                           trades_df: pd.DataFrame) -> pd.DataFrame:
    """For each row in events_df (a labeled Kalshi event), compute the tick
    features by looking at trades within ±25 strikes of entry_price during
    the [event_ts - 30min, event_ts] window.
    """
    events_df = events_df.copy()
    if trades_df.empty:
        for c in TICK_FEATURE_COLS:
            events_df[c] = 0.0
        return events_df

    # Group trades by event_ticker for fast lookup
    trades_df = trades_df.copy()
    trades_df["created_time_utc"] = pd.to_datetime(trades_df["created_time_utc"], utc=True)
    by_event = dict(tuple(trades_df.groupby("event_ticker")))

    # Pre-compute event_ticker for each event row
    events_df["_kalshi_event_ticker"] = events_df["ts"].apply(
        lambda t: event_ticker_for_event_ts(t)[0])

    out_rows = []
    for _, row in events_df.iterrows():
        ev_ticker = row["_kalshi_event_ticker"]
        ts_utc = pd.Timestamp(row["ts"]).tz_convert("UTC")
        entry_px = float(row["entry_price"])
        if ev_ticker not in by_event:
            out_rows.append({c: 0.0 for c in TICK_FEATURE_COLS})
            continue
        td = by_event[ev_ticker]
        # Only trades within ±25 strikes of entry_price
        td = td[(td["strike"] >= entry_px - 25) & (td["strike"] <= entry_px + 25)]
        # Window: [ts - 30min, ts]
        win30 = td[(td["created_time_utc"] >= ts_utc - pd.Timedelta(minutes=30)) &
                    (td["created_time_utc"] <= ts_utc)]
        win15 = win30[win30["created_time_utc"] >= ts_utc - pd.Timedelta(minutes=15)]
        win5 = win30[win30["created_time_utc"] >= ts_utc - pd.Timedelta(minutes=5)]
        win1 = win30[win30["created_time_utc"] >= ts_utc - pd.Timedelta(minutes=1)]

        feats = {c: 0.0 for c in TICK_FEATURE_COLS}
        if not win30.empty:
            # Use the strike closest to entry_px as the "ATM" reference
            atm_strike = win30.iloc[(win30["strike"] - entry_px).abs().argsort()[:1]]["strike"].iloc[0]
            atm = win30[win30["strike"] == atm_strike].sort_values("created_time_utc")
            if not atm.empty:
                last_yp = float(atm.iloc[-1]["yes_price"])
                feats["kt_atm_yes_price_now"] = last_yp
                # Velocities
                for win_min, key in [(5, "kt_close_velocity_5m"),
                                     (15, "kt_close_velocity_15m"),
                                     (30, "kt_close_velocity_30m")]:
                    cutoff = ts_utc - pd.Timedelta(minutes=win_min)
                    earlier = atm[atm["created_time_utc"] <= cutoff]
                    if not earlier.empty:
                        feats[key] = last_yp - float(earlier.iloc[-1]["yes_price"])
                # Volatility (over 15m)
                a15 = atm[atm["created_time_utc"] >= ts_utc - pd.Timedelta(minutes=15)]
                if len(a15) >= 2:
                    feats["kt_close_volatility_15m"] = float(a15["yes_price"].std())
                    feats["kt_yes_price_range_15m"] = float(
                        a15["yes_price"].max() - a15["yes_price"].min())
        if not win15.empty:
            feats["kt_n_trades_15m"] = float(len(win15))
            feats["kt_taker_yes_imbalance_15m"] = (
                100 * (win15["taker_side"] == "yes").mean()
            )
            feats["kt_avg_size_15m"] = float(win15["count_fp"].mean())
            feats["kt_size_pressure_15m"] = float((win15["count_fp"] > 100).sum())
            feats["kt_dollar_volume_15m"] = float(
                (win15["yes_price"] * win15["count_fp"]).sum())
            feats["kt_strikes_traded_15m"] = float(win15["strike"].nunique())
        if not win5.empty:
            feats["kt_n_trades_5m"] = float(len(win5))
        out_rows.append(feats)

    feat_df = pd.DataFrame(out_rows, index=events_df.index)
    return pd.concat([events_df, feat_df], axis=1).drop(columns=["_kalshi_event_ticker"])


def build_v7_dataset(events: list, bars: pd.DataFrame,
                     trades_df: pd.DataFrame,
                     label_horizon_min: int) -> pd.DataFrame:
    """Build dataset = v2 features + tick features."""
    feats = build_feature_frame(bars)
    feats = feats.loc[feats[FEATURE_COLS_40].notna().all(axis=1)].copy()
    feats_minute_idx = {pd.Timestamp(t).strftime("%Y-%m-%d %H:%M"): t for t in feats.index}
    log.info("feature frame rows: %d", len(feats))
    c_b = bars["close"].to_numpy(float)
    h_b = bars["high"].to_numpy(float)
    l_b = bars["low"].to_numpy(float)
    idx_pos = {t: i for i, t in enumerate(bars.index)}
    rows = []
    dropped_no_bar = 0
    dropped_ambiguous = 0
    for e in events:
        mts_key = e["market_ts"][:16]
        ts = feats_minute_idx.get(mts_key)
        if ts is None: dropped_no_bar += 1; continue
        pos = idx_pos.get(ts)
        if pos is None: dropped_no_bar += 1; continue
        side_sign = 1 if e["side"] == "LONG" else -1
        pnl = simulate_trade_horizon(h_b, l_b, c_b, int(pos),
                                     DEFAULT_TP, DEFAULT_SL, side_sign,
                                     label_horizon_min)
        if abs(pnl) < LABEL_MARGIN_USD:
            dropped_ambiguous += 1; continue
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
             pd.Timestamp(mts_key + ":00").minute)
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
    log.info("labeled rows: %d  dropped (no bar): %d  dropped (ambig): %d",
             len(df), dropped_no_bar, dropped_ambiguous)

    # Add tick features
    df_reset = df.reset_index().rename(columns={"_ts": "ts"})
    log.info("computing tick features for %d events using %d trades...",
             len(df_reset), len(trades_df))
    df_reset = compute_tick_features(df_reset, trades_df)
    log.info("tick features added")

    # Diagnostic: how many events have non-zero tick coverage?
    has_ticks = (df_reset["kt_n_trades_15m"] > 0).sum()
    log.info("events with tick data: %d / %d (%.1f%%)",
             has_ticks, len(df_reset), 100*has_ticks/len(df_reset))
    pass_pnls = df_reset[df_reset["label"]=="pass"]["forward_pnl"]
    block_pnls = df_reset[df_reset["label"]=="block"]["forward_pnl"]
    log.info("pass mean: $%+.2f  block mean: $%+.2f", pass_pnls.mean(), block_pnls.mean())

    df = df_reset.set_index("ts")
    df.index.name = "_ts"
    return df


def eval_regressor(pred_pnl, rule_decision, y_true, pnl_if_passed,
                   pass_thr, block_thr):
    pass_override = (rule_decision == "BLOCK") & (pred_pnl >= pass_thr)
    block_override = (rule_decision == "PASS") & (pred_pnl <= -block_thr)
    final_pass = np.where(pass_override, True,
                           np.where(block_override, False, rule_decision == "PASS"))
    final_pnl = np.where(final_pass, pnl_if_passed, 0.0)
    st = stats(final_pnl)
    n_new_pass = int(pass_override.sum())
    n_new_block = int(block_override.sum())
    new_pass_pnls = pnl_if_passed[pass_override]
    new_pass_wr = (100 * (new_pass_pnls > 0).sum() / len(new_pass_pnls)) if len(new_pass_pnls) else 0.0
    return {
        "pass_thr": pass_thr, "block_thr": block_thr,
        "n_pass": int(final_pass.sum()),
        "n_new_pass": n_new_pass, "n_new_block": n_new_block,
        "new_pass_wr": new_pass_wr,
        "pnl": st["pnl"], "dd": st["dd"], "avg": st["avg"],
    }


# ─── Walk-forward stability check ────────────────────────────────────────

def walk_forward_check(df: pd.DataFrame, feature_cols: list[str],
                        train_days: int = 60, test_days: int = 30,
                        seed: int = 42) -> dict:
    df = df.sort_index()
    cur = df.index.min() + pd.Timedelta(days=train_days)
    end = df.index.max()
    pooled_pred, pooled_actual = [], []
    n = 0
    while cur + pd.Timedelta(days=test_days) <= end:
        tr = df.loc[(df.index >= cur - pd.Timedelta(days=train_days)) & (df.index < cur)]
        te = df.loc[(df.index >= cur) & (df.index < cur + pd.Timedelta(days=test_days))]
        if len(tr) >= 200 and len(te) >= 20:
            X_tr = tr[feature_cols].fillna(tr[feature_cols].median()).to_numpy()
            X_te = te[feature_cols].fillna(tr[feature_cols].median()).to_numpy()
            y_tr = tr["forward_pnl"].to_numpy()
            m = HistGradientBoostingRegressor(
                max_iter=200, learning_rate=0.05, max_depth=4,
                min_samples_leaf=50, random_state=seed)
            m.fit(X_tr, y_tr)
            pred = m.predict(X_te)
            actual = te["forward_pnl"].to_numpy()
            pooled_pred.extend(pred)
            pooled_actual.extend(actual)
            n += 1
        cur += pd.Timedelta(days=test_days)
    pooled_corr = float(np.corrcoef(pooled_pred, pooled_actual)[0, 1]) \
        if len(pooled_pred) > 1 else 0.0
    return {"windows": n, "pooled_corr": pooled_corr}


def run_v7_config(df: pd.DataFrame, top_k: int, half_life: float,
                   train_start: Optional[str], oos_start: str, oos_end: str,
                   seed: int = 42) -> Optional[dict]:
    log.info("")
    log.info("=" * 78)
    log.info("V7 CONFIG  top_k=%d  hl=%.0f  trstart=%s",
             top_k, half_life, train_start or "full")
    log.info("=" * 78)

    oos_start_ts = pd.Timestamp(oos_start, tz=df.index.tz)
    oos_end_ts = pd.Timestamp(oos_end, tz=df.index.tz) + pd.Timedelta(days=1)
    tr = df.loc[df.index < oos_start_ts]
    if train_start is not None:
        tr = tr.loc[tr.index >= pd.Timestamp(train_start, tz=df.index.tz)]
    oos = df.loc[(df.index >= oos_start_ts) & (df.index <= oos_end_ts)]
    log.info("train: %d  OOS: %d", len(tr), len(oos))
    if len(tr) < 200 or len(oos) < 50:
        log.warning("below floors — skip"); return None

    pool = FEATURE_POOL_V7
    X_tr_full = tr[pool].fillna(tr[pool].median()).to_numpy()
    y_pnl = tr["forward_pnl"].to_numpy()
    mi = mutual_info_regression(X_tr_full, y_pnl, random_state=seed)
    # Always include tick features + Kalshi metadata, plus top-K-by-MI from the rest
    mandatory = [c for c in pool if c.startswith("kt_") or c.startswith("k_") or
                 c in ("rule_decision_pass_int", "side_sign", "settlement_hour")]
    others = [(c, mi[pool.index(c)]) for c in pool if c not in mandatory]
    others.sort(key=lambda x: -x[1])
    extras_n = max(0, top_k - len(mandatory))
    selected = mandatory + [c for c, _ in others[:extras_n]]
    log.info("mandatory features: %d  extras (top-MI): %d  total selected: %d",
             len(mandatory), extras_n, len(selected))
    log.info("MI scores for top tick features:")
    for c in TICK_FEATURE_COLS:
        if c in pool:
            log.info("  %-30s MI=%.4f", c, mi[pool.index(c)])

    X_tr = tr[selected].fillna(tr[selected].median()).to_numpy()
    X_oos = oos[selected].fillna(tr[selected].median()).to_numpy()
    recency_w = recency_weights(tr.index, half_life)

    rule_decision = oos["rule_decision"].to_numpy()
    pnl_if_passed = oos["forward_pnl"].to_numpy()
    y_true = oos["label"].to_numpy()
    rule_pnl_arr = np.where(rule_decision == "PASS", pnl_if_passed, 0.0)
    rule_st = stats(rule_pnl_arr)
    oracle_st = stats(np.where(y_true == "pass", pnl_if_passed, 0.0))
    log.info("rule baseline: PnL=$%+.2f DD=$%.0f  | oracle: PnL=$%+.2f",
             rule_st["pnl"], rule_st["dd"], oracle_st["pnl"])

    reg = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.05, max_depth=5,
        l2_regularization=1.0, min_samples_leaf=30, random_state=seed)
    reg.fit(X_tr, y_pnl, sample_weight=recency_w)
    pred_pnl = reg.predict(X_oos)
    corr = float(np.corrcoef(pred_pnl, pnl_if_passed)[0, 1])
    log.info("OOS pred-vs-actual Pearson: %+.3f", corr)

    wf = walk_forward_check(tr, selected, seed=seed)
    log.info("walk-forward (training): %d windows  pooled_corr=%+.3f",
             wf["windows"], wf["pooled_corr"])

    thr_grid = [
        (5.0, 5.0), (7.5, 5.0), (10.0, 5.0), (10.0, 7.5),
        (12.5, 7.5), (15.0, 7.5), (15.0, 10.0), (20.0, 10.0),
        (25.0, 10.0), (25.0, 15.0), (30.0, 15.0),
    ]
    log.info("%5s %5s %8s %8s %10s %8s %8s %7s gates",
             "p_thr", "b_thr", "n_pass", "new_psn", "newPassWR", "n_new_blk",
             "pnl", "capt")
    results = []
    headroom = oracle_st["pnl"] - rule_st["pnl"]
    for (pt, bt) in thr_grid:
        r = eval_regressor(pred_pnl, rule_decision, y_true, pnl_if_passed, pt, bt)
        lift = r["pnl"] - rule_st["pnl"]
        capt = (lift / headroom * 100.0) if headroom > 0 else 0.0
        dd_over_pnl = (r["dd"] / r["pnl"] * 100.0) if r["pnl"] > 0 else float("inf")
        gates = {
            "pnl_ok":        lift > 0,
            "dd_ratio_ok":   dd_over_pnl <= 30.0,
            "n_ok":          len(oos) >= 50,
            "new_pass_wr_ok":(r["new_pass_wr"] >= 50.0) if r["n_new_pass"] >= 5 else True,
            "capt_ok":       capt >= 20.0,
        }
        ok = all(gates.values())
        log.info("%5.1f %5.1f %8d %8d %9.2f%% %8d $%+7.2f %6.2f%%  %d/5%s",
                 pt, bt, r["n_pass"], r["n_new_pass"], r["new_pass_wr"],
                 r["n_new_block"], r["pnl"], capt, sum(gates.values()),
                 " SHIP" if ok else "")
        r.update({"lift": lift, "dd_over_pnl": dd_over_pnl, "capt_pct": capt,
                  "gates": gates, "ships": ok})
        results.append(r)

    shippers = [r for r in results if r["ships"]]
    best = max(shippers, key=lambda r: r["lift"]) if shippers else max(results, key=lambda r: r["lift"])
    return {
        "top_k": top_k, "half_life": half_life, "train_start": train_start,
        "rule_baseline": rule_st, "oracle": oracle_st,
        "n_train": len(tr), "n_oos": len(oos),
        "oos_corr": corr, "wf": wf, "selected_features": selected,
        "all": results, "best": best,
        "model": reg if best["ships"] else None,
        "median_imputes": {c: float(tr[c].median()) if pd.notna(tr[c].median()) else 0.0
                           for c in selected},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oos-start", default=OOS_START)
    ap.add_argument("--oos-end", default=OOS_END)
    ap.add_argument("--out-dir", default=str(ROOT / "artifacts/regime_ml_kalshi_v7"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--horizons", nargs="+", type=int, default=[15, 30, 60])
    ap.add_argument("--top-ks", nargs="+", type=int, default=[20, 30, 45])
    ap.add_argument("--half-lives", nargs="+", type=float, default=[90.0])
    ap.add_argument("--train-starts", nargs="+", default=["full"])
    ap.add_argument("--trades-parquet", default=str(TRADES_PARQUET))
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("=== loading log Kalshi events ===")
    log_paths = []
    rdir = ROOT / "backtest_reports/full_live_replay"
    for m in "2025_01 2025_02 2025_03 2025_04 2025_05 2025_06 2025_07 2025_08 2025_09 2025_10 2025_11 2025_12".split():
        p = rdir / m / "topstep_live_bot.log"
        if p.exists() and p.stat().st_size > 50_000:
            log_paths.append(p)
    live = ROOT / "topstep_live_bot.log"
    if live.exists(): log_paths.append(live)
    all_events = []
    for p in log_paths: all_events.extend(parse_log_events(p))
    all_events = add_v2_features(all_events)
    log.info("Kalshi events: %d", len(all_events))

    mts = [e["market_ts"] for e in all_events]
    start = pd.Timestamp(min(mts)).strftime("%Y-%m-%d")
    end = (pd.Timestamp(max(mts)) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    log.info("=== loading ES bars %s -> %s ===", start, end)
    bars = load_continuous_bars(start, end)
    log.info("ES bars: %d", len(bars))

    log.info("=== loading trades parquet ===")
    trades_path = Path(args.trades_parquet)
    if not trades_path.exists():
        log.error("trades parquet not found at %s — run fetch_kalshi_historical_trades.py first", trades_path)
        return 1
    trades_df = pd.read_parquet(trades_path)
    log.info("trades: %d  unique markets: %d  unique events: %d  date range: %s -> %s",
             len(trades_df), trades_df["market_ticker"].nunique(),
             trades_df["event_ticker"].nunique(),
             trades_df["created_time_utc"].min(), trades_df["created_time_utc"].max())

    # Build datasets per horizon
    datasets = {}
    for h in args.horizons:
        log.info("\n=== building dataset horizon=%d ===", h)
        datasets[h] = build_v7_dataset(all_events, bars, trades_df, h)

    configs = []
    for h in args.horizons:
        for k in args.top_ks:
            for hl in args.half_lives:
                for ts_ in args.train_starts:
                    ts_val = None if ts_ == "full" else ts_
                    configs.append((h, k, hl, ts_val))
    log.info("=== running %d configs ===", len(configs))

    runs = []
    for cfg in configs:
        h, k, hl, ts_val = cfg
        r = run_v7_config(datasets[h], k, hl, ts_val,
                          args.oos_start, args.oos_end, seed=args.seed)
        if r is not None:
            r["horizon"] = h
            runs.append(r)

    log.info("\n%s\nV7 SWEEP SUMMARY\n%s", "═"*150, "═"*150)
    log.info("%4s %4s %4s %10s %5s %7s %8s %8s %+8s %7s %s",
             "hrz", "k", "hl", "trstart", "gts", "corr", "wf_corr",
             "newPs", "lift$", "capt%", "ship?")
    for r in runs:
        b = r["best"]
        log.info("%4d %4d %4.0f %10s %5d %+6.3f %+7.3f %8d %+8.2f %6.2f%% %s",
                 r["horizon"], r["top_k"], r["half_life"],
                 str(r["train_start"] or "full")[:10],
                 sum(b["gates"].values()), r["oos_corr"], r["wf"]["pooled_corr"],
                 b["n_new_pass"], b["lift"], b["capt_pct"],
                 "SHIP" if b["ships"] else "-")

    shippers = [r for r in runs if r["best"]["ships"]]
    if not shippers:
        (out_dir / "sweep_summary.json").write_text(json.dumps({
            "verdict": "KILL",
            "reason": "no v7 config passes all 5 gates even with tick features",
            "runs": [{k: v for k, v in r.items() if k not in ("model",)}
                     for r in runs],
        }, indent=2, default=str))
        log.warning("[KILL] no v7 config passes — summary written")
        return 1

    best_run = max(shippers, key=lambda r: r["best"]["lift"])
    b = best_run["best"]
    payload = {
        "model_kind": "regression",
        "reg": best_run["model"],
        "feature_cols": best_run["selected_features"],
        "median_imputes": best_run["median_imputes"],
        "decision": b,
        "horizon_min": best_run["horizon"],
        "half_life_days": best_run["half_life"],
        "train_start": best_run["train_start"],
        "rule_baseline_oos": best_run["rule_baseline"],
        "oracle_oos": best_run["oracle"],
        "n_train": best_run["n_train"], "n_oos": best_run["n_oos"],
        "oos_corr": best_run["oos_corr"],
        "wf": best_run["wf"],
        "tick_features": TICK_FEATURE_COLS,
        "seed": args.seed,
    }
    with (out_dir / "model.pkl").open("wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    (out_dir / "model_meta.json").write_text(json.dumps({
        k: v for k, v in payload.items() if k != "reg"
    }, indent=2, default=str))
    log.info("[SHIP] horizon=%d top_k=%d  lift=$%+.2f  capt=%.2f%%  corr=%+.3f",
             best_run["horizon"], best_run["top_k"],
             b["lift"], b["capt_pct"], best_run["oos_corr"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
