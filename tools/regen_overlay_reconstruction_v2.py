#!/usr/bin/env python3
"""Regenerate the LFO / Pivot / PCT overlay reconstructions using the corrected
simulator (Section 8.26 fix — contract pinning + conservative trade-through).

Inputs:
  - artifacts/full_overlay_stack_simulation_14mo.parquet
        Pre-scored overlay decisions (fg/k/lfo/pct/pivot) on 3,438 candidates.
  - artifacts/filterless_reconstruction_14mo_v2.parquet
        Corrected per-candidate baseline PnL with the fixed simulator.
  - es_master_outrights.parquet
        ES outright bars by symbol/timestamp.

Process per overlay (LFO, PCT, Pivot):
  1) Take the candidate's corrected baseline PnL as `pnl_no_<overlay>`.
  2) Apply the overlay's gate logic to determine `pnl_with_<overlay>`:
     * LFO WAIT  → re-simulate entry 3 bars later at the same price target;
                   if a bank-level fill (within 2 ES points of price) is found
                   in the 3-bar window, take that price; else use the bar 3
                   close. Walk forward 30 bars from the new entry under the
                   corrected simulator.
     * LFO IMMEDIATE → pnl_with = pnl_no.
     * PCT BREAKOUT_LEAN → re-simulate with tp *= 1.10 (size_mult=1.0).
     * PCT PIVOT_LEAN    → re-simulate with tp *= 0.85, size_mult = 0.5.
     * PCT NEUTRAL/NOT_AT_LEVEL → pnl_with = pnl_no.
     * Pivot HOLD/SKIP → DEFERRED (mid-trade SL ratchet not modelled here);
                         we still emit the parquet with Pivot=NA so callers
                         have the schema, but `pivot_delta = 0` everywhere.
  3) Apply the SAME single-position friend-rule used in
     filterless_reconstruction_14mo_v2 — based on the *no-overlay* exit_ts.
  4) Apply the $7.50/trade haircut to whichever path is realised.
  5) Per-month + 14-mo aggregate (trades, WR, PnL, DD) into
     artifacts/<overlay>_reconstruction_14mo_v2.parquet and a JSON summary.

Outputs:
  artifacts/lfo_reconstruction_14mo_v2.parquet
  artifacts/pivot_reconstruction_14mo_v2.parquet     (deferred / NA)
  artifacts/pct_reconstruction_14mo_v2.parquet
  artifacts/<overlay>_v2_summary.json
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
    front_month_by_calendar,
    simulate_trade_through,
)

PARQUET_BARS = ROOT / "es_master_outrights.parquet"
PARQUET_SIGNALS = ROOT / "artifacts" / "full_overlay_stack_simulation_14mo.parquet"
PARQUET_FILTERLESS_V2 = ROOT / "artifacts" / "filterless_reconstruction_14mo_v2.parquet"

OUT_LFO = ROOT / "artifacts" / "lfo_reconstruction_14mo_v2.parquet"
OUT_PIVOT = ROOT / "artifacts" / "pivot_reconstruction_14mo_v2.parquet"
OUT_PCT = ROOT / "artifacts" / "pct_reconstruction_14mo_v2.parquet"
OUT_SUMMARY = ROOT / "artifacts" / "overlay_v2_summary.json"

POINT_VALUE = 5.0   # MES = $5/pt
HAIRCUT = 7.50
HORIZON = 30        # bars
LFO_WAIT_BARS = 3   # bars LFO waits for a better fill
LFO_BANK_TOL = 2.0  # pts; "found a bank fill" if any bar low/high within tol
LFO_THRESHOLD = 0.50
PCT_BREAKOUT_THRESHOLD = 0.55
PCT_PIVOT_THRESHOLD = 0.45


# ---------------------------------------------------------------------------
# bar / contract helpers
# ---------------------------------------------------------------------------
def _per_contract_index(bars_df: pd.DataFrame) -> dict:
    out = {}
    for sym, sub in bars_df.groupby("symbol", observed=True):
        out[sym] = sub.sort_index()
    return out


def _resolve_contract(bars: pd.DataFrame, ts, price: float) -> str:
    try:
        ts_rows = bars.loc[ts]
    except KeyError:
        return front_month_by_calendar(ts)
    if isinstance(ts_rows, pd.Series):
        return str(ts_rows["symbol"]) if "symbol" in ts_rows.index else front_month_by_calendar(ts)
    diffs = (ts_rows["close"].astype(float) - price).abs()
    mask_vol = ts_rows["volume"].astype(float) > 0
    if mask_vol.any():
        diffs = diffs.where(mask_vol, np.inf)
    if diffs.min() <= 0.5:
        return str(ts_rows["symbol"].iloc[diffs.values.argmin()])
    return front_month_by_calendar(ts)


def _walk_forward(by_sym: dict, contract: str, ts, horizon: int) -> Optional[pd.DataFrame]:
    sym_bars = by_sym.get(contract)
    if sym_bars is None or sym_bars.empty:
        return None
    fwd = sym_bars.loc[sym_bars.index > ts].head(horizon)
    return fwd if len(fwd) else None


def _simulate(by_sym, contract, entry_ts, side, entry_price, tp_price, sl_price, horizon):
    """Run the corrected simulator from `entry_ts` forward `horizon` bars on
    the pinned contract. Returns (raw_pnl, exit_ts) — raw_pnl is in DOLLARS
    pre-haircut. (None, None) if no_data."""
    fwd = _walk_forward(by_sym, contract, entry_ts, horizon)
    if fwd is None:
        return None, None
    fwd_reset = fwd.reset_index()
    out = simulate_trade_through(
        fwd_reset, side=side, entry_price=entry_price,
        tp_price=tp_price, sl_price=sl_price,
    )
    raw = out.pnl_points * POINT_VALUE
    if 0 <= out.exit_bar < len(fwd):
        exit_ts = fwd.index[out.exit_bar]
    else:
        exit_ts = None
    return raw, exit_ts


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------
def _aggregate(df_taken: pd.DataFrame, pnl_col: str) -> dict:
    """Compute per-month + total {trades, wr, pnl, max_dd}."""
    s = df_taken.sort_values("ts").reset_index(drop=True).copy()
    s["month"] = s["ts"].dt.tz_convert("US/Eastern").dt.strftime("%Y-%m")
    cum = s[pnl_col].cumsum()
    if len(cum):
        max_dd = float((cum - cum.cummax()).min())
    else:
        max_dd = 0.0

    n = len(s)
    wins = int((s[pnl_col] > 0).sum())
    wr = (100.0 * wins / n) if n else 0.0

    per_month = []
    for month, sub in s.groupby("month", sort=True):
        n_m = len(sub)
        wins_m = int((sub[pnl_col] > 0).sum())
        wr_m = (100.0 * wins_m / n_m) if n_m else 0.0
        pnl_m = float(sub[pnl_col].sum())
        cum_m = sub.sort_values("ts")[pnl_col].cumsum()
        dd_m = float((cum_m - cum_m.cummax()).min()) if len(cum_m) else 0.0
        per_month.append({
            "month": month, "n": int(n_m), "wr": round(wr_m, 2),
            "pnl": round(pnl_m, 2), "dd": round(dd_m, 2),
        })

    return {
        "trades": int(n),
        "wr": round(wr, 2),
        "pnl_net": round(float(s[pnl_col].sum()), 2),
        "max_dd": round(max_dd, 2),
        "per_month": per_month,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    print(f"[overlay-v2] loading bars: {PARQUET_BARS}")
    bars = pd.read_parquet(PARQUET_BARS)
    if bars.index.tz is None:
        bars.index = bars.index.tz_localize("US/Eastern")
    elif str(bars.index.tz) != "US/Eastern":
        bars.index = bars.index.tz_convert("US/Eastern")
    print(f"  bars shape={bars.shape}")

    by_sym = _per_contract_index(bars)

    print(f"[overlay-v2] loading candidates: {PARQUET_SIGNALS}")
    sigs = pd.read_parquet(PARQUET_SIGNALS)
    sigs["ts"] = pd.to_datetime(sigs["ts"])
    if sigs["ts"].dt.tz is None:
        sigs["ts"] = sigs["ts"].dt.tz_localize("US/Eastern")
    else:
        sigs["ts"] = sigs["ts"].dt.tz_convert("US/Eastern")
    sigs = sigs.sort_values("ts").reset_index(drop=True)
    print(f"  candidates: {len(sigs)}")

    print(f"[overlay-v2] loading corrected baseline: {PARQUET_FILTERLESS_V2}")
    base = pd.read_parquet(PARQUET_FILTERLESS_V2)
    # base ts already tz-aware US/Eastern
    base = base.sort_values("ts").reset_index(drop=True)
    print(f"  baseline rows: {len(base)}")

    # Aligned candidate -> baseline index via merge on (ts, side, price)
    print("[overlay-v2] aligning candidate stream to baseline rows")
    sigs_key = sigs[["ts", "side", "price"]].copy()
    base_key = base[["ts", "side", "entry_price", "contract", "exit_reason",
                     "exit_ts", "raw_pnl", "net_pnl_after_haircut"]].copy()
    base_key = base_key.rename(columns={
        "entry_price": "price",
        "contract": "base_contract",
        "exit_reason": "base_exit_reason",
        "exit_ts": "base_exit_ts",
        "raw_pnl": "base_raw_pnl",
        "net_pnl_after_haircut": "base_net_pnl",
    })
    merged = sigs.merge(base_key, on=["ts", "side", "price"], how="left")
    merged["base_contract"] = merged["base_contract"].fillna("").astype(str)
    merged["base_exit_reason"] = merged["base_exit_reason"].fillna("").astype(str)
    merged["base_raw_pnl"] = merged["base_raw_pnl"].fillna(0.0).astype(float)
    merged["base_net_pnl"] = merged["base_net_pnl"].fillna(0.0).astype(float)
    sigs = merged
    n_match = int((sigs["base_exit_reason"] != "").sum())
    n_miss = int((sigs["base_exit_reason"] == "").sum())
    print(f"  matched={n_match}, missed={n_miss}")

    # Pre-compute friend-rule inclusion: a row is "taken" iff
    # base_exit_reason not in {skipped_friend_rule, no_data}
    sigs["is_taken"] = ~sigs["base_exit_reason"].isin({"skipped_friend_rule", "no_data"})

    # ------------------------------------------------------------------
    # LFO reconstruction
    # ------------------------------------------------------------------
    print("\n[overlay-v2] LFO reconstruction")
    lfo_rows = []
    for i, sig in enumerate(sigs.itertuples(index=False)):
        ts = sig.ts
        side = str(sig.side)
        price = float(sig.price)
        sl_dist = float(sig.sl)
        tp_dist = float(sig.tp)
        proba = float(sig.lfo_proba) if sig.lfo_proba == sig.lfo_proba else 0.0
        decision = "WAIT" if proba >= LFO_THRESHOLD else "IMMEDIATE"
        contract = sig.base_contract or front_month_by_calendar(ts)

        if side == "LONG":
            tp_price = price + tp_dist
            sl_price = price - sl_dist
        else:
            tp_price = price - tp_dist
            sl_price = price + sl_dist

        # pnl_immediate = baseline corrected raw PnL if the candidate was
        # taken, else the candidate's would-be PnL re-simulated standalone.
        # For consistency with the overlay decision, compute IMMEDIATE without
        # friend-rule (to score the model's call vs WAIT) and then reapply
        # friend-rule at the aggregate step.
        if sig.is_taken:
            pnl_immediate = float(sig.base_raw_pnl)
            imm_exit_ts = sig.base_exit_ts
        else:
            # Candidate was friend-skipped in baseline. Re-simulate standalone
            # so the model has a comparable "what if alone".
            raw, ex = _simulate(by_sym, contract, ts, side, price, tp_price, sl_price, HORIZON)
            pnl_immediate = float(raw) if raw is not None else 0.0
            imm_exit_ts = ex

        # WAIT outcome: skip 3 bars; entry = first bar (within 3 bars) whose
        # low<=price (LONG) / high>=price (SHORT) within bank tol; else
        # bar-3 close. Walk forward HORIZON bars from the new entry.
        wait_entry = None
        wait_entry_ts = None
        sym_bars = by_sym.get(contract)
        if sym_bars is not None and not sym_bars.empty:
            wait_window = sym_bars.loc[sym_bars.index > ts].head(LFO_WAIT_BARS)
            if not wait_window.empty:
                # Look for "fill at level" within the window
                if side == "LONG":
                    candidates = wait_window[wait_window["low"] <= price + LFO_BANK_TOL]
                else:
                    candidates = wait_window[wait_window["high"] >= price - LFO_BANK_TOL]
                if not candidates.empty:
                    first_fill = candidates.iloc[0]
                    wait_entry = price  # fill at the level
                    wait_entry_ts = candidates.index[0]
                else:
                    # No fill within window: enter on the bar-3 close
                    wait_entry = float(wait_window.iloc[-1]["close"])
                    wait_entry_ts = wait_window.index[-1]

        if wait_entry is not None and wait_entry_ts is not None:
            if side == "LONG":
                wait_tp = wait_entry + tp_dist
                wait_sl = wait_entry - sl_dist
            else:
                wait_tp = wait_entry - tp_dist
                wait_sl = wait_entry + sl_dist
            raw_w, ex_w = _simulate(
                by_sym, contract, wait_entry_ts, side,
                wait_entry, wait_tp, wait_sl, HORIZON,
            )
            pnl_wait = float(raw_w) if raw_w is not None else 0.0
            wait_exit_ts = ex_w
        else:
            pnl_wait = 0.0
            wait_exit_ts = None

        if decision == "WAIT":
            pnl_lfo = pnl_wait
            chosen_exit_ts = wait_exit_ts
        else:
            pnl_lfo = pnl_immediate
            chosen_exit_ts = imm_exit_ts

        lfo_delta = pnl_lfo - pnl_immediate
        lfo_correct = pnl_lfo >= pnl_immediate

        lfo_rows.append({
            "ts": ts,
            "month": ts.tz_convert("US/Eastern").strftime("%Y-%m"),
            "strategy": str(getattr(sig, "strategy", "")),
            "family": str(getattr(sig, "family", "")),
            "side": side,
            "price": price,
            "contract": contract,
            "lfo_proba": proba,
            "lfo_decision": decision,
            "pnl_immediate": pnl_immediate,
            "pnl_wait": pnl_wait,
            "pnl_lfo_raw": pnl_lfo,
            "pnl_lfo_net": pnl_lfo - HAIRCUT if abs(pnl_lfo) > 1e-9 else 0.0,
            "lfo_delta": lfo_delta,
            "lfo_correct": bool(lfo_correct),
            "is_taken": bool(sig.is_taken),
            "exit_ts": chosen_exit_ts,
        })
        if (i + 1) % 500 == 0:
            print(f"  [LFO {i+1}/{len(sigs)}] elapsed={time.time()-t0:.1f}s")

    df_lfo = pd.DataFrame(lfo_rows)
    df_lfo.to_parquet(OUT_LFO, index=False)
    print(f"[overlay-v2] wrote {OUT_LFO} ({len(df_lfo)} rows)")

    # ------------------------------------------------------------------
    # PCT reconstruction
    # ------------------------------------------------------------------
    print("\n[overlay-v2] PCT reconstruction")
    pct_rows = []
    for i, sig in enumerate(sigs.itertuples(index=False)):
        ts = sig.ts
        side = str(sig.side)
        price = float(sig.price)
        sl_dist = float(sig.sl)
        tp_dist = float(sig.tp)
        decision = str(sig.pct_decision)
        size_mult = float(sig.pct_size_mult)
        tp_mult = float(sig.pct_tp_mult)
        contract = sig.base_contract or front_month_by_calendar(ts)

        if side == "LONG":
            tp_price = price + tp_dist
            sl_price = price - sl_dist
            tp_price_pct = price + tp_dist * tp_mult
        else:
            tp_price = price - tp_dist
            sl_price = price + sl_dist
            tp_price_pct = price - tp_dist * tp_mult

        # pnl_no_pct
        if sig.is_taken:
            pnl_no_pct = float(sig.base_raw_pnl)
        else:
            raw, _ = _simulate(by_sym, contract, ts, side, price, tp_price, sl_price, HORIZON)
            pnl_no_pct = float(raw) if raw is not None else 0.0

        # pnl_with_pct: re-simulate with adjusted TP, then scale by size_mult
        if abs(tp_mult - 1.0) < 1e-9 and abs(size_mult - 1.0) < 1e-9:
            pnl_with_pct = pnl_no_pct
        else:
            raw_p, _ = _simulate(
                by_sym, contract, ts, side, price,
                tp_price_pct, sl_price, HORIZON,
            )
            pnl_with_pct = float(raw_p) * size_mult if raw_p is not None else 0.0

        pct_delta = pnl_with_pct - pnl_no_pct
        pct_correct = pnl_with_pct >= pnl_no_pct

        pct_rows.append({
            "ts": ts,
            "month": ts.tz_convert("US/Eastern").strftime("%Y-%m"),
            "strategy": str(getattr(sig, "strategy", "")),
            "family": str(getattr(sig, "family", "")),
            "side": side,
            "price": price,
            "contract": contract,
            "pct_decision": decision,
            "size_mult": size_mult,
            "tp_mult": tp_mult,
            "pnl_no_pct": pnl_no_pct,
            "pnl_with_pct_raw": pnl_with_pct,
            "pnl_with_pct_net": pnl_with_pct - HAIRCUT if abs(pnl_with_pct) > 1e-9 else 0.0,
            "pct_delta": pct_delta,
            "pct_correct": bool(pct_correct),
            "is_taken": bool(sig.is_taken),
        })
        if (i + 1) % 500 == 0:
            print(f"  [PCT {i+1}/{len(sigs)}] elapsed={time.time()-t0:.1f}s")

    df_pct = pd.DataFrame(pct_rows)
    df_pct.to_parquet(OUT_PCT, index=False)
    print(f"[overlay-v2] wrote {OUT_PCT} ({len(df_pct)} rows)")

    # ------------------------------------------------------------------
    # Pivot — DEFERRED (mid-trade SL ratchet not modelled). Emit pass-through.
    # ------------------------------------------------------------------
    print("\n[overlay-v2] Pivot reconstruction — DEFERRED (mid-trade ratchet not modelled)")
    pivot_rows = []
    for sig in sigs.itertuples(index=False):
        ts = sig.ts
        if sig.is_taken:
            pnl_no_pivot = float(sig.base_raw_pnl)
        else:
            pnl_no_pivot = 0.0
        pivot_rows.append({
            "ts": ts,
            "month": ts.tz_convert("US/Eastern").strftime("%Y-%m"),
            "strategy": str(getattr(sig, "strategy", "")),
            "family": str(getattr(sig, "family", "")),
            "side": str(sig.side),
            "price": float(sig.price),
            "pivot_decision": "DEFERRED",
            "pivot_proba": np.nan,
            "pnl_no_pivot": pnl_no_pivot,
            "pnl_with_pivot_raw": pnl_no_pivot,  # pass-through
            "pnl_with_pivot_net": pnl_no_pivot - HAIRCUT if abs(pnl_no_pivot) > 1e-9 else 0.0,
            "pivot_delta": 0.0,
            "pivot_correct": None,
            "is_taken": bool(sig.is_taken),
        })
    df_pivot = pd.DataFrame(pivot_rows)
    df_pivot.to_parquet(OUT_PIVOT, index=False)
    print(f"[overlay-v2] wrote {OUT_PIVOT} ({len(df_pivot)} rows) — pass-through baseline")

    # ------------------------------------------------------------------
    # Aggregates — apply friend-rule + haircut, compare against filterless v2
    # ------------------------------------------------------------------
    print("\n[overlay-v2] computing aggregates")

    # Filterless baseline (taken trades only)
    base_taken = base[~base["exit_reason"].isin({"skipped_friend_rule", "no_data"})].copy()
    base_taken = base_taken.rename(columns={"net_pnl_after_haircut": "pnl_net"})
    base_agg = _aggregate(base_taken, "pnl_net")
    print(f"  Filterless v2: trades={base_agg['trades']}, "
          f"WR={base_agg['wr']}, PnL=${base_agg['pnl_net']}, DD=${base_agg['max_dd']}")

    # LFO: only the rows that are taken in baseline contribute
    lfo_taken = df_lfo[df_lfo["is_taken"]].copy()
    lfo_taken["pnl_net"] = lfo_taken["pnl_lfo_raw"] - HAIRCUT
    lfo_agg = _aggregate(lfo_taken, "pnl_net")
    print(f"  LFO v2:        trades={lfo_agg['trades']}, "
          f"WR={lfo_agg['wr']}, PnL=${lfo_agg['pnl_net']}, DD=${lfo_agg['max_dd']}")

    pct_taken = df_pct[df_pct["is_taken"]].copy()
    pct_taken["pnl_net"] = pct_taken["pnl_with_pct_raw"] - HAIRCUT
    pct_agg = _aggregate(pct_taken, "pnl_net")
    print(f"  PCT v2:        trades={pct_agg['trades']}, "
          f"WR={pct_agg['wr']}, PnL=${pct_agg['pnl_net']}, DD=${pct_agg['max_dd']}")

    pivot_taken = df_pivot[df_pivot["is_taken"]].copy()
    pivot_taken["pnl_net"] = pivot_taken["pnl_with_pivot_raw"] - HAIRCUT
    pivot_agg = _aggregate(pivot_taken, "pnl_net")
    print(f"  Pivot v2:      trades={pivot_agg['trades']}, "
          f"WR={pivot_agg['wr']}, PnL=${pivot_agg['pnl_net']}, DD=${pivot_agg['max_dd']} (pass-through)")

    # Decision-level deltas (without friend-rule, model audit only)
    lfo_n_wait = int((df_lfo["lfo_decision"] == "WAIT").sum())
    lfo_pnl_delta_total = float(df_lfo[df_lfo["is_taken"]]["lfo_delta"].sum())
    pct_n_breakout = int((df_pct["pct_decision"] == "BREAKOUT_LEAN").sum())
    pct_n_pivot_lean = int((df_pct["pct_decision"] == "PIVOT_LEAN").sum())
    pct_pnl_delta_total = float(df_pct[df_pct["is_taken"]]["pct_delta"].sum())

    summary = {
        "filterless_v2": base_agg,
        "lfo_v2": lfo_agg,
        "lfo_decisions": {
            "n_total": int(len(df_lfo)),
            "n_wait": lfo_n_wait,
            "wait_pct": round(100.0 * lfo_n_wait / max(len(df_lfo), 1), 2),
            "pnl_delta_taken_only": round(lfo_pnl_delta_total, 2),
            "lfo_threshold": LFO_THRESHOLD,
        },
        "pct_v2": pct_agg,
        "pct_decisions": {
            "n_total": int(len(df_pct)),
            "n_breakout_lean": pct_n_breakout,
            "n_pivot_lean": pct_n_pivot_lean,
            "pnl_delta_taken_only": round(pct_pnl_delta_total, 2),
            "breakout_threshold": PCT_BREAKOUT_THRESHOLD,
            "pivot_threshold": PCT_PIVOT_THRESHOLD,
        },
        "pivot_v2": pivot_agg,
        "pivot_decisions": {
            "status": "DEFERRED — mid-trade SL ratchet requires bar-by-bar "
                      "intra-trade replay; candidate parquet has Pivot=NA.",
        },
        "rules": {
            "haircut_per_trade": HAIRCUT,
            "point_value": POINT_VALUE,
            "horizon_bars": HORIZON,
            "tp_rule": "conservative trade-through (high>=tp+tick AND close>=tp-tick OR next_bar through); SL any-touch",
            "lfo_wait_bars": LFO_WAIT_BARS,
            "lfo_bank_tol_pts": LFO_BANK_TOL,
        },
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2, default=str))
    print(f"[overlay-v2] wrote {OUT_SUMMARY}")

    # ------------------------------------------------------------------
    # Print per-month tables
    # ------------------------------------------------------------------
    def _print_month(name, agg):
        print(f"\n=== {name} per-month ===")
        print("month        n     wr%      pnl       dd")
        for m in agg["per_month"]:
            print(f"  {m['month']}  {m['n']:>4}  {m['wr']:>5.1f}  {m['pnl']:>9.2f}  {m['dd']:>9.2f}")
        print(f"  {'-'*47}")
        print(f"  TOTAL  {agg['trades']:>4}  {agg['wr']:>5.1f}  {agg['pnl_net']:>9.2f}  {agg['max_dd']:>9.2f}")

    _print_month("Filterless v2", base_agg)
    _print_month("LFO v2", lfo_agg)
    _print_month("PCT v2", pct_agg)
    _print_month("Pivot v2 (pass-through)", pivot_agg)

    print(f"\n[overlay-v2] done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
