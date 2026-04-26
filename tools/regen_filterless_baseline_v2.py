#!/usr/bin/env python3
"""Regenerate the 14-month filterless baseline using the corrected
trade-through simulator (Section 8.26 fix).

Inputs:
  - artifacts/full_overlay_stack_simulation_14mo.parquet
        Candidate signal stream (3,438 rows, 14 months) with columns
        ts, strategy, family, side, price, sl, tp, pnl_baseline (broken).
  - es_master_outrights.parquet
        ES outright bars by symbol/timestamp (one symbol per row).

Process:
  1) For each candidate signal, pin contract by close-price match at the
     signal bar (fallback: front-month roll calendar).
  2) Walk forward up to 30 bars on ONLY that contract's bars.
  3) Apply the conservative trade-through TP rule + any-touch SL.
  4) Apply $7.50/trade haircut.
  5) Apply single-position friend-rule: skip a candidate if a prior
     candidate is still open at its entry timestamp.

Outputs:
  artifacts/filterless_reconstruction_14mo_v2.parquet
  artifacts/baseline_v2_summary.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulator_trade_through import (  # noqa: E402
    ES_POINT_VALUE,
    front_month_by_calendar,
    get_walk_forward_bars,
    pin_contract,
    simulate_trade_through,
)

PARQUET_BARS = ROOT / "es_master_outrights.parquet"
PARQUET_SIGNALS = ROOT / "artifacts" / "full_overlay_stack_simulation_14mo.parquet"
OUT_TRADES = ROOT / "artifacts" / "filterless_reconstruction_14mo_v2.parquet"
OUT_SUMMARY = ROOT / "artifacts" / "baseline_v2_summary.json"

POINT_VALUE = 5.0  # MES = $5/pt; matches Section 8.25 audit
HAIRCUT = 7.50      # $/trade slippage+commission (matches existing tools)
HORIZON = 30        # bars
ROLL_GUARD_MINUTES = 5  # min spacing for "open trade" friend-rule


def _coerce_ts(s) -> pd.Timestamp:
    ts = pd.Timestamp(s)
    if ts.tzinfo is None:
        ts = ts.tz_localize("US/Eastern")
    else:
        ts = ts.tz_convert("US/Eastern")
    return ts


def _per_contract_index(bars_df: pd.DataFrame) -> dict:
    """Pre-group bars by symbol for fast lookup."""
    out = {}
    for sym, sub in bars_df.groupby("symbol", observed=True):
        out[sym] = sub.sort_index()
    return out


def main():
    t0 = time.time()
    print(f"[regen] loading bars: {PARQUET_BARS}")
    bars = pd.read_parquet(PARQUET_BARS)
    if bars.index.tz is None:
        bars.index = bars.index.tz_localize("US/Eastern")
    elif str(bars.index.tz) != "US/Eastern":
        bars.index = bars.index.tz_convert("US/Eastern")
    print(f"  bars shape={bars.shape}  range={bars.index.min()} -> {bars.index.max()}")

    print(f"[regen] loading signals: {PARQUET_SIGNALS}")
    sigs = pd.read_parquet(PARQUET_SIGNALS)
    sigs["ts"] = pd.to_datetime(sigs["ts"])
    if sigs["ts"].dt.tz is None:
        sigs["ts"] = sigs["ts"].dt.tz_localize("US/Eastern")
    else:
        sigs["ts"] = sigs["ts"].dt.tz_convert("US/Eastern")
    sigs = sigs.sort_values("ts").reset_index(drop=True)
    print(f"  signals: n={len(sigs)}, range={sigs['ts'].min()} -> {sigs['ts'].max()}")

    print("[regen] grouping bars by symbol")
    by_sym = _per_contract_index(bars)
    print(f"  {len(by_sym)} symbols cached")

    print("[regen] walking signals")
    rows = []
    next_open_at = None  # ts at which the previous trade exits (single-position rule)

    n = len(sigs)
    for i, sig in enumerate(sigs.itertuples(index=False)):
        ts = sig.ts.tz_convert("US/Eastern") if sig.ts.tzinfo else sig.ts.tz_localize("US/Eastern")
        side = str(sig.side).upper()
        price = float(sig.price)
        sl_dist = float(sig.sl)
        tp_dist = float(sig.tp)
        if side == "LONG":
            tp_price = price + tp_dist
            sl_price = price - sl_dist
        else:
            tp_price = price - tp_dist
            sl_price = price + sl_dist

        # Friend rule: single position at a time
        if next_open_at is not None and ts < next_open_at:
            rows.append({
                "ts": ts,
                "strategy": str(getattr(sig, "strategy", "")),
                "family": str(getattr(sig, "family", "")),
                "side": side,
                "entry_price": price,
                "tp": tp_price,
                "sl": sl_price,
                "contract": "",
                "exit_reason": "skipped_friend_rule",
                "exit_price": np.nan,
                "exit_bar": -1,
                "exit_ts": pd.NaT,
                "raw_pnl": 0.0,
                "net_pnl_after_haircut": 0.0,
                "horizon_bars": HORIZON,
            })
            continue

        # Pin contract: lookup all bars at ts, pick the symbol with close ~= price.
        try:
            ts_rows = bars.loc[ts]
        except KeyError:
            ts_rows = None
        contract = None
        if ts_rows is not None:
            if isinstance(ts_rows, pd.Series):
                contract = str(ts_rows["symbol"]) if "symbol" in ts_rows.index else None
            else:
                # Pick by smallest |close - price| with a 0.5pt tolerance
                diffs = (ts_rows["close"].astype(float) - price).abs()
                # Prefer those with vol > 0
                mask_vol = ts_rows["volume"].astype(float) > 0
                if mask_vol.any():
                    diffs = diffs.where(mask_vol, np.inf)
                if diffs.min() <= 0.5:
                    contract = str(ts_rows["symbol"].iloc[diffs.values.argmin()])
        if contract is None:
            contract = front_month_by_calendar(ts)

        sym_bars = by_sym.get(contract)
        if sym_bars is None or sym_bars.empty:
            rows.append({
                "ts": ts, "strategy": str(getattr(sig, "strategy", "")),
                "family": str(getattr(sig, "family", "")),
                "side": side, "entry_price": price, "tp": tp_price, "sl": sl_price,
                "contract": contract, "exit_reason": "no_data",
                "exit_price": np.nan, "exit_bar": -1, "exit_ts": pd.NaT,
                "raw_pnl": 0.0, "net_pnl_after_haircut": 0.0,
                "horizon_bars": HORIZON,
            })
            continue

        fwd = sym_bars.loc[sym_bars.index > ts].head(HORIZON)
        if fwd.empty:
            rows.append({
                "ts": ts, "strategy": str(getattr(sig, "strategy", "")),
                "family": str(getattr(sig, "family", "")),
                "side": side, "entry_price": price, "tp": tp_price, "sl": sl_price,
                "contract": contract, "exit_reason": "no_data",
                "exit_price": np.nan, "exit_bar": -1, "exit_ts": pd.NaT,
                "raw_pnl": 0.0, "net_pnl_after_haircut": 0.0,
                "horizon_bars": HORIZON,
            })
            continue

        fwd_reset = fwd.reset_index()
        outcome = simulate_trade_through(
            fwd_reset, side=side, entry_price=price,
            tp_price=tp_price, sl_price=sl_price,
        )
        raw_pnl = outcome.pnl_points * POINT_VALUE
        net_pnl = raw_pnl - HAIRCUT
        # exit ts (resolve from the original timestamped index)
        if outcome.exit_bar >= 0 and outcome.exit_bar < len(fwd):
            exit_ts = fwd.index[outcome.exit_bar]
        else:
            exit_ts = pd.NaT

        rows.append({
            "ts": ts,
            "strategy": str(getattr(sig, "strategy", "")),
            "family": str(getattr(sig, "family", "")),
            "side": side,
            "entry_price": price,
            "tp": tp_price,
            "sl": sl_price,
            "contract": contract,
            "exit_reason": outcome.exit_reason,
            "exit_price": outcome.exit_price,
            "exit_bar": outcome.exit_bar,
            "exit_ts": exit_ts,
            "raw_pnl": raw_pnl,
            "net_pnl_after_haircut": net_pnl,
            "horizon_bars": HORIZON,
        })

        # Set the friend-rule next_open_at = exit_ts (or +HORIZON if no exit)
        if pd.notna(exit_ts):
            next_open_at = exit_ts
        else:
            next_open_at = ts + pd.Timedelta(minutes=HORIZON)

        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{n}] elapsed={time.time()-t0:.1f}s")

    print(f"[regen] simulated {len(rows)} rows in {time.time()-t0:.1f}s")
    df = pd.DataFrame(rows)
    df.to_parquet(OUT_TRADES, index=False)
    print(f"[regen] wrote {OUT_TRADES}")

    # Aggregates
    taken = df[~df["exit_reason"].isin({"skipped_friend_rule", "no_data"})].copy()
    n_taken = len(taken)
    n_wins = int((taken["raw_pnl"] > 0).sum())
    wr = (100.0 * n_wins / n_taken) if n_taken else 0.0
    pnl_raw = float(taken["raw_pnl"].sum())
    pnl_net = float(taken["net_pnl_after_haircut"].sum())

    # Equity curve + DD on net
    taken_sorted = taken.sort_values("ts").reset_index(drop=True)
    cum = taken_sorted["net_pnl_after_haircut"].cumsum()
    if len(cum):
        running_max = cum.cummax()
        dd_series = cum - running_max
        max_dd = float(dd_series.min())
    else:
        max_dd = 0.0

    # Per-month
    taken_sorted["month"] = taken_sorted["ts"].dt.tz_convert("US/Eastern").dt.strftime("%Y-%m")
    per_month = []
    for month, sub in taken_sorted.groupby("month", sort=True):
        n_m = len(sub)
        wins_m = int((sub["raw_pnl"] > 0).sum())
        wr_m = (100.0 * wins_m / n_m) if n_m else 0.0
        pnl_m = float(sub["net_pnl_after_haircut"].sum())
        # Per-month DD
        cum_m = sub.sort_values("ts")["net_pnl_after_haircut"].cumsum()
        if len(cum_m):
            dd_m = float((cum_m - cum_m.cummax()).min())
        else:
            dd_m = 0.0
        per_month.append({"month": month, "n": int(n_m), "wr": round(wr_m, 2),
                          "pnl": round(pnl_m, 2), "dd": round(dd_m, 2)})

    summary = {
        "trades": int(n_taken),
        "wr": round(wr, 2),
        "pnl_raw": round(pnl_raw, 2),
        "pnl_net": round(pnl_net, 2),
        "max_dd": round(max_dd, 2),
        "n_total_signals": int(len(df)),
        "n_skipped_friend_rule": int((df["exit_reason"] == "skipped_friend_rule").sum()),
        "n_no_data": int((df["exit_reason"] == "no_data").sum()),
        "haircut_per_trade": HAIRCUT,
        "point_value": POINT_VALUE,
        "horizon_bars": HORIZON,
        "rule": "conservative trade-through TP (high>=tp+1tick AND close>=tp-1tick OR next bar trades through); SL any-touch; ES tick=0.25",
        "per_month": per_month,
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2, default=str))
    print(f"[regen] wrote {OUT_SUMMARY}")

    print("\n=== Per-month ===")
    print("month        n     wr%      pnl       dd")
    for m in per_month:
        print(f"  {m['month']}  {m['n']:>4}  {m['wr']:>5.1f}  {m['pnl']:>9.2f}  {m['dd']:>9.2f}")
    print("\n=== Aggregate (taken trades) ===")
    print(f"  trades   = {n_taken}")
    print(f"  WR       = {wr:.2f}%")
    print(f"  PnL raw  = ${pnl_raw:,.2f}")
    print(f"  PnL net  = ${pnl_net:,.2f}  (after ${HAIRCUT}/trade haircut)")
    print(f"  Max DD   = ${max_dd:,.2f}")
    print(f"  Total signals = {len(df)}")
    print(f"  Skipped (friend) = {(df['exit_reason']=='skipped_friend_rule').sum()}")
    print(f"  No-data = {(df['exit_reason']=='no_data').sum()}")


if __name__ == "__main__":
    main()
