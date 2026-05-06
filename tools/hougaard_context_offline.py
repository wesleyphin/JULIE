"""Hougaard context — OFFLINE batch engine.

Same trigger logic as the analysis script, but emits one record per session
with the bias direction + strength + regime amplifier so other modules
(overlay backtest, live engine) can consume a clean per-day context table.

Output schema (DataFrame, indexed by ET date):
  scenario_b_active  : bool  (only Mondays where Fri high < Thu high)
  scenario_b_dir     : int   (-1 if active else 0; Hougaard B is short-bias)
  scenario_c_active  : bool  (only Wed/Thu/Fri after Tue confirms direction)
  scenario_c_dir     : int   (+1 if Tue broke Mon-high; -1 if Tue broke Mon-low)
  bull_regime        : bool
  high_vol           : bool
  regime_amplifier   : float  (Scenario B: 1.5 in bear, 0.5 in bull;
                                Scenario C: 1.0 flat across regimes)
  bias_direction     : int   (combined: -1, 0, +1)
  bias_strength      : float (0..1; 1.0 when both scenarios fire same dir)
  active_scenarios   : str   ("" | "B" | "C+" | "C-" | "B+C+" | "B+C-")

Usage:
  from tools.hougaard_context_offline import build_context_table
  ctx = build_context_table("2024-01-01", "2025-12-31")
  ctx.loc["2024-04-15"]  -> Series for that date
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PARQUET = ROOT / "es_master_outrights-2.parquet"

RTH_OPEN = "09:30"
RTH_CLOSE = "16:00"


def _load_daily_rth(start: str, end: str) -> pd.DataFrame:
    """Daily RTH OHLC + regime classification (weekdays only)."""
    df = pd.read_parquet(PARQUET)
    # parquet index is ET-tz-aware
    if "symbol" in df.columns:
        df = df.sort_values("volume", ascending=False)
        df = df[~df.index.duplicated(keep="first")].sort_index()
    df = df[["open", "high", "low", "close", "volume"]]
    # filter date range — index is tz-aware; compare date-only
    mask = (df.index.normalize() >= pd.Timestamp(start, tz=df.index.tz)) & \
           (df.index.normalize() <= pd.Timestamp(end, tz=df.index.tz))
    df = df[mask]
    rth = df.between_time(RTH_OPEN, RTH_CLOSE, inclusive="left")
    daily = (rth.resample("1D")
                .agg({"open": "first", "high": "max", "low": "min",
                      "close": "last", "volume": "sum"})
                .dropna())
    daily["weekday"] = daily.index.dayofweek
    daily = daily[daily["weekday"] < 5].copy()
    daily["sma50"] = daily["close"].rolling(50, min_periods=20).mean()
    daily["sma50_slope_5d"] = daily["sma50"].diff(5)
    daily["bull_regime"] = (daily["sma50_slope_5d"] > 0).fillna(True)
    daily["ret_1d"] = daily["close"].pct_change()
    daily["vol_20d"] = daily["ret_1d"].rolling(20).std() * np.sqrt(252) * 100
    vol_thresh = daily["vol_20d"].quantile(0.66)
    daily["high_vol"] = (daily["vol_20d"] > vol_thresh).fillna(False)
    return daily


def build_context_table(start: str, end: str) -> pd.DataFrame:
    """Return a DataFrame with one row per ET trading day in [start, end]
    annotated with Hougaard context. Index is tz-aware ET (date level)."""
    daily = _load_daily_rth(start, end)
    by_date = {d.date(): row for d, row in daily.iterrows()}

    records = []
    for ts, row in daily.iterrows():
        d = ts.date()
        wd = int(row["weekday"])
        rec = {
            "date": d,
            "weekday": wd,
            "scenario_b_active": False,
            "scenario_b_dir": 0,
            "scenario_c_active": False,
            "scenario_c_dir": 0,
            "bull_regime": bool(row["bull_regime"]),
            "high_vol": bool(row["high_vol"]),
        }

        # Scenario B — Monday only. Trigger: Fri_high < Thu_high. Bias: SHORT.
        if wd == 0:
            fri = (ts - pd.Timedelta(days=3)).date()
            thu = (ts - pd.Timedelta(days=4)).date()
            if fri in by_date and thu in by_date:
                if by_date[fri]["high"] < by_date[thu]["high"]:
                    rec["scenario_b_active"] = True
                    rec["scenario_b_dir"] = -1

        # Scenario C — Wed/Thu/Fri. Trigger: Tue broke Mon's H or L.
        # Bias: same direction as the break.
        if wd in (2, 3, 4):
            # Find this week's Tuesday and Monday
            tue = (ts - pd.Timedelta(days=wd - 1)).date()
            mon = (ts - pd.Timedelta(days=wd)).date()
            if tue in by_date and mon in by_date:
                tue_row = by_date[tue]
                mon_row = by_date[mon]
                if tue_row["high"] > mon_row["high"] and tue_row["low"] >= mon_row["low"]:
                    rec["scenario_c_active"] = True
                    rec["scenario_c_dir"] = +1
                elif tue_row["low"] < mon_row["low"] and tue_row["high"] <= mon_row["high"]:
                    rec["scenario_c_active"] = True
                    rec["scenario_c_dir"] = -1
                # If Tue broke BOTH sides, scenario is ambiguous → leave inactive

        # Regime amplifier — only Scenario B has regime amplification
        # (analysis showed bear 68.6% vs bull 34.1% for B; C was regime-flat)
        if rec["scenario_b_active"]:
            rec["regime_amplifier_b"] = 1.5 if not rec["bull_regime"] else 0.5
        else:
            rec["regime_amplifier_b"] = 1.0

        # Combine into single bias_direction + bias_strength
        b_dir = rec["scenario_b_dir"]
        c_dir = rec["scenario_c_dir"]
        b_amp = rec["regime_amplifier_b"]
        if b_dir != 0 and c_dir != 0:
            # Both active (rare — only possible if Mon also Wed/Thu/Fri… so impossible)
            # Defensive: if same direction, full strength
            if b_dir == c_dir:
                rec["bias_direction"] = b_dir
                rec["bias_strength"] = min(1.0, 0.7 * b_amp + 0.3)
            else:
                rec["bias_direction"] = 0
                rec["bias_strength"] = 0.0
        elif b_dir != 0:
            rec["bias_direction"] = b_dir
            # B in bear: 0.85; B in bull: 0.30
            rec["bias_strength"] = min(1.0, 0.55 * b_amp + 0.10)
        elif c_dir != 0:
            rec["bias_direction"] = c_dir
            rec["bias_strength"] = 0.6  # 78%/71% claim → flat 0.6 strength
        else:
            rec["bias_direction"] = 0
            rec["bias_strength"] = 0.0

        # Active-scenarios string
        parts = []
        if rec["scenario_b_active"]:
            parts.append("B")
        if rec["scenario_c_active"]:
            parts.append("C+" if c_dir > 0 else "C-")
        rec["active_scenarios"] = "+".join(parts) if parts else ""

        records.append(rec)

    out = pd.DataFrame(records)
    out["date"] = pd.to_datetime(out["date"])
    out = out.set_index("date").sort_index()
    return out


def main():
    """Smoke test — build & save context for the iter12 backtest window."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2017-01-01")
    ap.add_argument("--end", default="2026-04-30")
    ap.add_argument("--out", default="artifacts/hougaard_overlay/context_table.parquet")
    args = ap.parse_args()

    print(f"[context] building table {args.start} → {args.end}...")
    ctx = build_context_table(args.start, args.end)
    print(f"[context] {len(ctx):,} session rows")

    n_b = int(ctx["scenario_b_active"].sum())
    n_c = int(ctx["scenario_c_active"].sum())
    n_bias = int((ctx["bias_direction"] != 0).sum())
    print(f"[context] Scenario B active: {n_b} sessions ({n_b/len(ctx)*100:.1f}%)")
    print(f"[context] Scenario C active: {n_c} sessions ({n_c/len(ctx)*100:.1f}%)")
    print(f"[context] ANY bias active:   {n_bias} sessions ({n_bias/len(ctx)*100:.1f}%)")
    print(f"[context] bias dir distribution: "
          f"{int((ctx['bias_direction'] == -1).sum())} short, "
          f"{int((ctx['bias_direction'] == +1).sum())} long, "
          f"{int((ctx['bias_direction'] == 0).sum())} neutral")
    print(f"[context] bull regime: {int(ctx['bull_regime'].sum())}/{len(ctx)} "
          f"({ctx['bull_regime'].mean()*100:.1f}%)")

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ctx.to_parquet(out_path)
    print(f"[context] → {out_path}")


if __name__ == "__main__":
    main()
