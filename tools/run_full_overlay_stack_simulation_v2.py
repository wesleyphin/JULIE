#!/usr/bin/env python3
"""Recompute Section 8.21 — full overlay stack simulation — using the
fixed walk-forward simulator from `simulator_trade_through.py`.

Inputs:
  - artifacts/full_overlay_stack_simulation_14mo.parquet  (v1)
        3,438 candidate signals with precomputed overlay decisions:
        fg_decision, k_decision, lfo_decision, pct_decision, pct_size_mult,
        pct_tp_mult, pivot_decision, blocked_by.
        Overlay decisions DO NOT change — only PnL recomputation.
  - es_master_outrights.parquet
        ES outright bars by symbol/timestamp.

Process (per signal):
  1) Pin contract via close-price match (fallback: roll calendar).
  2) Walk forward up to 30 bars on ONLY that contract's bars.
  3) Apply conservative trade-through TP rule + any-touch SL.
  4) Apply $7.50/trade haircut.
  5) Build TWO streams:
       pnl_baseline  — every signal walks (filterless control), single-position
                       friend-rule chained over ALL signals.
       pnl_final     — only signals where blocked_by is null walk; PCT
                       size_mult scales raw PnL; friend-rule chained over
                       UN-blocked signals.

Outputs:
  artifacts/full_overlay_stack_simulation_14mo_v2.parquet
  artifacts/full_stack_v2_summary.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulator_trade_through import (  # noqa: E402
    front_month_by_calendar,
    simulate_trade_through,
)

PARQUET_BARS = ROOT / "es_master_outrights.parquet"
PARQUET_V1 = ROOT / "artifacts" / "full_overlay_stack_simulation_14mo.parquet"
OUT_TRADES = ROOT / "artifacts" / "full_overlay_stack_simulation_14mo_v2.parquet"
OUT_SUMMARY = ROOT / "artifacts" / "full_stack_v2_summary.json"

POINT_VALUE = 5.0   # MES = $5/pt
HAIRCUT = 7.50      # $/trade
HORIZON = 30        # bars


def _per_contract_index(bars_df: pd.DataFrame) -> dict:
    out = {}
    for sym, sub in bars_df.groupby("symbol", observed=True):
        out[sym] = sub.sort_index()
    return out


def main():
    t0 = time.time()
    print(f"[v2-stack] loading bars: {PARQUET_BARS}")
    bars = pd.read_parquet(PARQUET_BARS)
    if bars.index.tz is None:
        bars.index = bars.index.tz_localize("US/Eastern")
    elif str(bars.index.tz) != "US/Eastern":
        bars.index = bars.index.tz_convert("US/Eastern")
    print(f"  bars shape={bars.shape}  range={bars.index.min()} -> {bars.index.max()}")

    print(f"[v2-stack] loading v1 candidate stream: {PARQUET_V1}")
    sigs = pd.read_parquet(PARQUET_V1)
    sigs["ts"] = pd.to_datetime(sigs["ts"])
    if sigs["ts"].dt.tz is None:
        sigs["ts"] = sigs["ts"].dt.tz_localize("US/Eastern")
    else:
        sigs["ts"] = sigs["ts"].dt.tz_convert("US/Eastern")
    sigs = sigs.sort_values("ts").reset_index(drop=True)
    print(f"  signals: n={len(sigs)}, range={sigs['ts'].min()} -> {sigs['ts'].max()}")

    print("[v2-stack] grouping bars by symbol")
    by_sym = _per_contract_index(bars)
    print(f"  {len(by_sym)} symbols cached")

    print("[v2-stack] walking signals (independent per-signal walk; chains applied later)")
    rows = []
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

        # Pin contract
        try:
            ts_rows = bars.loc[ts]
        except KeyError:
            ts_rows = None
        contract = None
        if ts_rows is not None:
            if isinstance(ts_rows, pd.Series):
                contract = str(ts_rows["symbol"]) if "symbol" in ts_rows.index else None
            else:
                diffs = (ts_rows["close"].astype(float) - price).abs()
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
                "ts": ts, "contract": contract,
                "exit_reason": "no_data", "exit_price": np.nan, "exit_bar": -1,
                "exit_ts": pd.NaT, "raw_pnl_walk": 0.0, "tp_used": tp_price, "sl_used": sl_price,
            })
            continue

        fwd = sym_bars.loc[sym_bars.index > ts].head(HORIZON)
        if fwd.empty:
            rows.append({
                "ts": ts, "contract": contract,
                "exit_reason": "no_data", "exit_price": np.nan, "exit_bar": -1,
                "exit_ts": pd.NaT, "raw_pnl_walk": 0.0, "tp_used": tp_price, "sl_used": sl_price,
            })
            continue

        fwd_reset = fwd.reset_index()
        outcome = simulate_trade_through(
            fwd_reset, side=side, entry_price=price,
            tp_price=tp_price, sl_price=sl_price,
        )
        raw_pnl = outcome.pnl_points * POINT_VALUE
        if outcome.exit_bar >= 0 and outcome.exit_bar < len(fwd):
            exit_ts = fwd.index[outcome.exit_bar]
        else:
            exit_ts = pd.NaT
        rows.append({
            "ts": ts, "contract": contract,
            "exit_reason": outcome.exit_reason, "exit_price": outcome.exit_price,
            "exit_bar": outcome.exit_bar, "exit_ts": exit_ts,
            "raw_pnl_walk": raw_pnl, "tp_used": tp_price, "sl_used": sl_price,
        })

        if (i + 1) % 500 == 0:
            print(f"  [{i+1}/{n}] elapsed={time.time()-t0:.1f}s")

    walk_df = pd.DataFrame(rows)
    print(f"[v2-stack] walked {len(walk_df)} signals in {time.time()-t0:.1f}s")
    print("  exit_reason counts (per-signal walk):")
    print(walk_df["exit_reason"].value_counts())

    # Merge walks back into the candidate stream
    sigs2 = sigs.copy().reset_index(drop=True)
    walk_df = walk_df.reset_index(drop=True)
    out = pd.concat([sigs2, walk_df.drop(columns=["ts"])], axis=1)
    assert len(out) == len(sigs), f"row count mismatch: {len(out)} vs {len(sigs)}"

    # ----- Stream 1: pnl_baseline (filterless control + friend-rule) -----
    # Single-position friend-rule chained over ALL signals. PCT mults NOT applied.
    print("[v2-stack] computing pnl_baseline stream (friend-rule over ALL signals)")
    base_pnl = np.zeros(len(out))
    base_taken = np.zeros(len(out), dtype=bool)
    next_open_at = None
    for i, r in out.iterrows():
        ts = r["ts"]
        exit_reason = r["exit_reason"]
        if exit_reason == "no_data":
            base_pnl[i] = 0.0
            continue
        if next_open_at is not None and ts < next_open_at:
            base_pnl[i] = 0.0
            continue
        # Take this signal
        base_pnl[i] = float(r["raw_pnl_walk"]) - HAIRCUT
        base_taken[i] = True
        ex = r["exit_ts"]
        if pd.notna(ex):
            next_open_at = ex
        else:
            next_open_at = ts + pd.Timedelta(minutes=HORIZON)

    out["pnl_baseline"] = base_pnl
    out["taken_baseline"] = base_taken

    # ----- Stream 2: pnl_final (full ml_full_ny stack) -----
    # Apply: if blocked_by not null → trade NEVER takes (pnl=0, doesn't consume friend-rule chain)
    # Otherwise: walk + PCT size_mult on raw PnL, friend-rule chained over UN-blocked signals only.
    print("[v2-stack] computing pnl_final stream (overlays applied + friend-rule)")
    final_pnl = np.zeros(len(out))
    final_taken = np.zeros(len(out), dtype=bool)
    final_outcome = ["" for _ in range(len(out))]
    next_open_at = None

    blocked_by_arr = out["blocked_by"].fillna("").astype(str).values
    pct_size_arr = out["pct_size_mult"].fillna(1.0).astype(float).values

    for i, r in out.iterrows():
        ts = r["ts"]
        bb = blocked_by_arr[i]
        exit_reason = r["exit_reason"]

        # Hard block by Filter G or Kalshi → no trade, no chain consumption
        if bb and bb != "nan" and bb != "":
            final_pnl[i] = 0.0
            final_outcome[i] = f"blocked_by_{bb}"
            continue

        if exit_reason == "no_data":
            final_pnl[i] = 0.0
            final_outcome[i] = "no_data"
            continue

        # Friend-rule on the un-blocked stream
        if next_open_at is not None and ts < next_open_at:
            final_pnl[i] = 0.0
            final_outcome[i] = "skipped_friend_rule"
            continue

        # Take this signal — apply PCT size_mult on raw PnL (gross), then haircut
        size_mult = pct_size_arr[i]
        raw = float(r["raw_pnl_walk"]) * size_mult
        final_pnl[i] = raw - HAIRCUT
        final_taken[i] = True
        final_outcome[i] = exit_reason  # take/stop/horizon

        ex = r["exit_ts"]
        if pd.notna(ex):
            next_open_at = ex
        else:
            next_open_at = ts + pd.Timedelta(minutes=HORIZON)

    out["pnl_final"] = final_pnl
    out["taken_final"] = final_taken
    out["outcome_final_v2"] = final_outcome

    # Trim to v1 schema-compatible columns + extras (keep v1 schema; replace pnl cols)
    keep_cols = [
        "ts", "month", "strategy", "family", "side", "price", "sl", "tp",
        "fg_proba", "fg_decision", "k_proba", "k_decision",
        "lfo_proba", "lfo_decision", "pct_decision", "pct_size_mult",
        "pct_tp_mult", "pivot_decision", "pivot_proba", "blocked_by",
        # Recomputed fields
        "contract", "exit_reason", "exit_price", "exit_bar", "exit_ts",
        "raw_pnl_walk", "pnl_baseline", "pnl_final",
        "taken_baseline", "taken_final", "outcome_final_v2",
    ]
    keep_cols = [c for c in keep_cols if c in out.columns]
    out_final = out[keep_cols]
    out_final.to_parquet(OUT_TRADES, index=False)
    print(f"[v2-stack] wrote {OUT_TRADES}  rows={len(out_final)}")

    # ===== Summary =====
    def _stream_stats(df, pnl_col, taken_col, label):
        taken = df[df[taken_col]]
        n = len(taken)
        wins = int((taken[pnl_col] > 0).sum())
        wr = (100.0 * wins / n) if n else 0.0
        pnl = float(taken[pnl_col].sum())
        # DD on the taken stream, time-ordered
        ord_ = taken.sort_values("ts")
        cum = ord_[pnl_col].cumsum()
        dd = float((cum - cum.cummax()).min()) if len(cum) else 0.0
        # per-month
        per_month = []
        ord2 = ord_.copy()
        ord2["month_str"] = ord2["ts"].dt.strftime("%Y-%m")
        for m, sub in ord2.groupby("month_str", sort=True):
            sub_pnl = float(sub[pnl_col].sum())
            sub_n = len(sub)
            sub_w = int((sub[pnl_col] > 0).sum())
            sub_wr = (100.0 * sub_w / sub_n) if sub_n else 0.0
            sub_cum = sub.sort_values("ts")[pnl_col].cumsum()
            sub_dd = float((sub_cum - sub_cum.cummax()).min()) if len(sub_cum) else 0.0
            per_month.append({"month": m, "n": int(sub_n), "wr": round(sub_wr, 2),
                              "pnl": round(sub_pnl, 2), "dd": round(sub_dd, 2)})
        return {
            "label": label, "trades": int(n), "wr": round(wr, 2),
            "pnl_net": round(pnl, 2), "max_dd": round(dd, 2),
            "per_month": per_month,
        }

    summary = {
        "n_total_signals": int(len(out_final)),
        "n_blocked_by_filter_g": int((out_final["blocked_by"].fillna("") == "filter_g").sum()),
        "n_blocked_by_kalshi": int((out_final["blocked_by"].fillna("") == "kalshi").sum()),
        "n_no_data": int((out_final["exit_reason"] == "no_data").sum()),
        "haircut_per_trade": HAIRCUT,
        "point_value": POINT_VALUE,
        "horizon_bars": HORIZON,
        "rule": "conservative trade-through TP (high>=tp+1tick AND close>=tp-1tick OR next bar trades through); SL any-touch; ES tick=0.25",
        "baseline": _stream_stats(out_final, "pnl_baseline", "taken_baseline", "baseline (filterless control + friend-rule)"),
        "final_stack": _stream_stats(out_final, "pnl_final", "taken_final", "ml_full_ny stack (Filter G + Kalshi + LFO + PCT + friend-rule)"),
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2, default=str))
    print(f"[v2-stack] wrote {OUT_SUMMARY}")

    print("\n=== Stream summary ===")
    for key in ("baseline", "final_stack"):
        s = summary[key]
        print(f"  {s['label']}")
        print(f"    trades = {s['trades']}, WR = {s['wr']:.2f}%, PnL = ${s['pnl_net']:,.2f}, DD = ${s['max_dd']:,.2f}")

    print("\n=== Per-month (final_stack) ===")
    print("month        n     wr%      pnl       dd")
    for m in summary["final_stack"]["per_month"]:
        print(f"  {m['month']}  {m['n']:>4}  {m['wr']:>5.1f}  {m['pnl']:>9.2f}  {m['dd']:>9.2f}")


if __name__ == "__main__":
    main()
