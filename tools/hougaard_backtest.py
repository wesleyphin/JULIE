"""Tom Hougaard "Highs & Lows" Backtester — ES Futures (last 5 years).

Three situational-analysis scenarios:

  A) Thursday/Wednesday Expansion (short bias):
     If Monday's RTH high is the highest of Mon/Tue/Wed RTH highs, short on
     Thursday at 09:30 ET open.

  B) Friday "Inertia" Rule (weekend carry):
     If Friday's RTH high < Thursday's RTH high, short Monday at 09:30 ET
     open (instead of Friday close — 09:30 lets us check the gap filter).
     Filter: skip if |Monday open - Friday close| / Friday close > 0.5%

  C) Weekly Pivot Rule:
     If Monday's high broken Tuesday → long bias for rest of week
     (Wed-Fri entries each at 09:30 ET).
     If Monday's low broken Tuesday → short bias for rest of week.

Execution rules per Hougaard:
  - Initial SL: 1.5 × ATR(14) on 5-min chart at entry
  - Scaling: each 0.5 × ATR favorable adds 50% of initial size (pyramid)
  - Trailing: after first add, SL → breakeven; thereafter trail to prior
    5-min swing high (short) / low (long)
  - EOD exit: 16:00 ET cash close, or trailing stop hit
  - No fixed take profit

Outputs:
  - Per-scenario win rate, profit factor, max DD
  - With vs without scaling (pyramiding)
  - Equity curve CSV + PNG
  - Trade journal CSV

Usage:
    python3 tools/hougaard_backtest.py
    python3 tools/hougaard_backtest.py --start 2021-01-01 --end 2026-04-24
    python3 tools/hougaard_backtest.py --no-plot   # skip matplotlib
"""
from __future__ import annotations

import argparse
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent.parent
PARQUET = ROOT / "es_master_outrights-2.parquet"
OUT_DIR = ROOT / "artifacts" / "hougaard_backtest"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ES futures: $50/point. MES is $5/point. Use ES for backtest sizing.
POINT_VALUE = 50.0
COMMISSION_PER_SIDE = 2.50  # ES retail-ish; round-trip = $5
SLIPPAGE_TICKS = 1  # 1 tick = 0.25 pts on ES
TICK_SIZE = 0.25
RTH_OPEN = "09:30"
RTH_CLOSE = "16:00"


# ─── data loading + resampling ──────────────────────────────────────────────

def load_bars(start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
    """Load 1-min ES bars, pick highest-volume symbol per timestamp (front-month proxy)."""
    print(f"[load] reading {PARQUET} ...")
    df = pd.read_parquet(PARQUET)
    if start: df = df[df.index >= start]
    if end: df = df[df.index <= end]
    # If multiple symbols share a timestamp (during rollover), keep highest-volume
    if "symbol" in df.columns:
        df = df.sort_values("volume", ascending=False)
        df = df[~df.index.duplicated(keep="first")].sort_index()
    df = df[["open", "high", "low", "close", "volume"]]
    print(f"[load] {len(df):,} rows, {df.index.min()} → {df.index.max()}")
    return df


def resample_5m(bars_1m: pd.DataFrame) -> pd.DataFrame:
    """Resample 1-min bars to 5-min OHLCV."""
    agg = {"open": "first", "high": "max", "low": "min",
           "close": "last", "volume": "sum"}
    df = bars_1m.resample("5min", label="left", closed="left").agg(agg)
    return df.dropna(subset=["open"])


def daily_rth_ohlc(bars_1m: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 09:30-16:00 ET RTH window to daily OHLC. Index = trade date.
    Hougaard's research is equity-day; ETH overnights are noise here."""
    df = bars_1m.between_time(RTH_OPEN, RTH_CLOSE, inclusive="left")
    daily = df.resample("1D").agg({"open": "first", "high": "max",
                                    "low": "min", "close": "last",
                                    "volume": "sum"}).dropna()
    daily["weekday"] = daily.index.dayofweek  # Mon=0, Fri=4
    daily = daily[daily["weekday"] < 5]  # drop weekends explicitly
    return daily


def compute_atr14_5m(bars_5m: pd.DataFrame) -> pd.Series:
    """Wilder's ATR(14) on 5-min chart."""
    high = bars_5m["high"]; low = bars_5m["low"]; close_prev = bars_5m["close"].shift(1)
    tr = pd.concat([high - low, (high - close_prev).abs(), (low - close_prev).abs()],
                   axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()
    return atr


# ─── scenario detectors ─────────────────────────────────────────────────────

def detect_scenario_a(daily: pd.DataFrame) -> pd.DataFrame:
    """Thursday/Wednesday Expansion: Monday's high is highest of Mon/Tue/Wed.
    Returns DataFrame indexed by Thursday date with the trigger info."""
    triggers = []
    daily_idx = daily.index.normalize()
    by_date = {d.date(): row for d, row in daily.iterrows()}
    for thu_ts, thu_row in daily.iterrows():
        if thu_row["weekday"] != 3:  # 3 = Thursday
            continue
        wed = thu_ts - pd.Timedelta(days=1)
        tue = thu_ts - pd.Timedelta(days=2)
        mon = thu_ts - pd.Timedelta(days=3)
        if not (mon.date() in by_date and tue.date() in by_date and wed.date() in by_date):
            continue
        mon_h = by_date[mon.date()]["high"]
        tue_h = by_date[tue.date()]["high"]
        wed_h = by_date[wed.date()]["high"]
        wed_l = by_date[wed.date()]["low"]
        if mon_h > tue_h and mon_h > wed_h:
            triggers.append({
                "date": thu_ts.date(),
                "side": "SHORT",
                "wed_low": wed_l,
                "mon_high": mon_h,
                "scenario": "A",
            })
    return pd.DataFrame(triggers)


def detect_scenario_b(daily: pd.DataFrame, gap_filter_pct: float = 0.5) -> pd.DataFrame:
    """Friday Inertia: Friday's high < Thursday's high → short Monday open.
    Filter: skip if |Mon open - Fri close| / Fri close > gap_filter_pct%
    Returns DataFrame indexed by Monday date."""
    triggers = []
    by_date = {d.date(): row for d, row in daily.iterrows()}
    for mon_ts, mon_row in daily.iterrows():
        if mon_row["weekday"] != 0:  # 0 = Monday
            continue
        # Find prior Friday + Thursday
        fri = mon_ts - pd.Timedelta(days=3)  # Mon - 3 days = Fri
        thu = mon_ts - pd.Timedelta(days=4)
        if not (fri.date() in by_date and thu.date() in by_date):
            continue
        fri_h = by_date[fri.date()]["high"]
        thu_h = by_date[thu.date()]["high"]
        fri_c = by_date[fri.date()]["close"]
        fri_l = by_date[fri.date()]["low"]
        mon_o = mon_row["open"]
        if fri_h >= thu_h:
            continue  # Friday made a new high — no trigger
        gap_pct = abs(mon_o - fri_c) / fri_c * 100.0
        triggers.append({
            "date": mon_ts.date(),
            "side": "SHORT",
            "fri_low": fri_l,
            "fri_close": fri_c,
            "mon_open": mon_o,
            "gap_pct": gap_pct,
            "scenario": "B",
            "filter_blocked": gap_pct > gap_filter_pct,
        })
    return pd.DataFrame(triggers)


def detect_scenario_c(daily: pd.DataFrame) -> pd.DataFrame:
    """Weekly Pivot: if Tuesday breaks Monday's high → long bias rest of week.
    If Tuesday breaks Monday's low → short bias rest of week.
    Generates entries for Wed/Thu/Fri at 09:30 ET in the bias direction."""
    triggers = []
    by_date = {d.date(): row for d, row in daily.iterrows()}
    for mon_ts, mon_row in daily.iterrows():
        if mon_row["weekday"] != 0:
            continue
        tue = mon_ts + pd.Timedelta(days=1)
        if tue.date() not in by_date:
            continue
        tue_row = by_date[tue.date()]
        mon_h = mon_row["high"]; mon_l = mon_row["low"]
        bias = None
        if tue_row["high"] > mon_h and tue_row["low"] >= mon_l:
            bias = "LONG"
        elif tue_row["low"] < mon_l and tue_row["high"] <= mon_h:
            bias = "SHORT"
        else:
            # Both broken (outside day) — no clean bias signal
            continue
        # Generate Wed, Thu, Fri entries
        for offset in (2, 3, 4):
            d = mon_ts + pd.Timedelta(days=offset)
            if d.date() not in by_date:
                continue
            triggers.append({
                "date": d.date(),
                "side": bias,
                "monday_high": mon_h,
                "monday_low": mon_l,
                "scenario": "C",
            })
    return pd.DataFrame(triggers)


# ─── trade walker with scaling logic ────────────────────────────────────────

@dataclass
class Trade:
    entry_date: pd.Timestamp
    entry_ts: pd.Timestamp
    side: str            # "LONG" or "SHORT"
    initial_size: float
    initial_atr: float
    legs: list = field(default_factory=list)  # [(ts, price, size), ...]
    sl_price: Optional[float] = None
    exit_ts: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: str = ""
    pnl_dollars: float = 0.0
    pnl_points: float = 0.0
    n_adds: int = 0
    scenario: str = ""

    @property
    def total_size(self) -> float:
        return sum(leg[2] for leg in self.legs)

    @property
    def avg_entry(self) -> float:
        ts = self.total_size
        if ts <= 0: return 0.0
        return sum(leg[1] * leg[2] for leg in self.legs) / ts


def walk_trade(entry_date: pd.Timestamp, side: str, atr_at_entry: float,
               bars_5m: pd.DataFrame, scaling: bool = True,
               initial_size: float = 1.0,
               sl_atr_mult: float = 1.5,
               add_atr_step: float = 0.5,
               add_size_pct: float = 0.5,
               trail_to_swing: bool = True,
               eod_time: str = "16:00",
               scenario: str = "") -> Optional[Trade]:
    """Walk the day forward from 09:30 ET, applying scaling + trailing logic.
    Returns Trade or None if entry bar is missing.

    Slippage: 1 tick adverse on entry + 1 tick adverse on exit.
    Commission: $5 round-trip per contract (initial size unit).
    """
    day_start = pd.Timestamp(entry_date).tz_localize("US/Eastern").replace(hour=9, minute=30)
    day_end = day_start.replace(hour=int(eod_time[:2]), minute=int(eod_time[3:]))
    try:
        day_bars = bars_5m.loc[day_start:day_end]
    except KeyError:
        return None
    if day_bars.empty or atr_at_entry <= 0 or not math.isfinite(atr_at_entry):
        return None

    # Entry at 09:30 bar open + 1 tick slippage
    entry_bar = day_bars.iloc[0]
    entry_ts = day_bars.index[0]
    raw_open = float(entry_bar["open"])
    if side == "LONG":
        entry_price = raw_open + SLIPPAGE_TICKS * TICK_SIZE
        sl_price = entry_price - sl_atr_mult * atr_at_entry
    else:
        entry_price = raw_open - SLIPPAGE_TICKS * TICK_SIZE
        sl_price = entry_price + sl_atr_mult * atr_at_entry

    trade = Trade(
        entry_date=pd.Timestamp(entry_date),
        entry_ts=entry_ts, side=side,
        initial_size=initial_size, initial_atr=atr_at_entry,
        sl_price=sl_price, scenario=scenario,
    )
    trade.legs.append((entry_ts, entry_price, initial_size))

    # Walk subsequent bars
    sign = 1 if side == "LONG" else -1
    next_add_threshold = 1  # multiplier on add_atr_step (1, 2, 3, ...)
    moved_to_be = False

    for ts, bar in day_bars.iloc[1:].iterrows():
        bar_high = float(bar["high"]); bar_low = float(bar["low"])
        # 1) Check SL first (conservative — if both SL and add level hit in same bar, prefer SL)
        sl_hit = (bar_low <= trade.sl_price) if side == "LONG" else (bar_high >= trade.sl_price)
        if sl_hit:
            trade.exit_ts = ts
            trade.exit_price = trade.sl_price - sign * SLIPPAGE_TICKS * TICK_SIZE
            trade.exit_reason = "stop_loss"
            break

        # 2) Check scaling: each 0.5 × ATR move favorable adds 50% of initial
        if scaling:
            mfe_pts = (bar_high - entry_price) if side == "LONG" else (entry_price - bar_low)
            target_pts = next_add_threshold * add_atr_step * atr_at_entry
            if mfe_pts >= target_pts:
                add_price = entry_price + sign * target_pts
                add_size = initial_size * add_size_pct
                trade.legs.append((ts, add_price, add_size))
                trade.n_adds += 1
                # On first add: move SL to BE
                if not moved_to_be:
                    trade.sl_price = entry_price
                    moved_to_be = True
                # Subsequent adds: trail to previous bar's swing
                elif trail_to_swing:
                    prev_idx = day_bars.index.get_loc(ts) - 1
                    if prev_idx > 0:
                        prev_bar = day_bars.iloc[prev_idx]
                        new_sl = float(prev_bar["low"]) if side == "LONG" else float(prev_bar["high"])
                        if side == "LONG":
                            trade.sl_price = max(trade.sl_price, new_sl)
                        else:
                            trade.sl_price = min(trade.sl_price, new_sl)
                next_add_threshold += 1

        # 3) After first add, on every subsequent bar trail SL to prior swing
        if moved_to_be and trail_to_swing:
            prev_idx = day_bars.index.get_loc(ts) - 1
            if prev_idx > 0:
                prev_bar = day_bars.iloc[prev_idx]
                cand_sl = float(prev_bar["low"]) if side == "LONG" else float(prev_bar["high"])
                if side == "LONG":
                    trade.sl_price = max(trade.sl_price, cand_sl)
                else:
                    trade.sl_price = min(trade.sl_price, cand_sl)

    # EOD exit if not stopped
    if trade.exit_ts is None:
        last_bar = day_bars.iloc[-1]
        last_close = float(last_bar["close"])
        trade.exit_ts = day_bars.index[-1]
        trade.exit_price = last_close - sign * SLIPPAGE_TICKS * TICK_SIZE
        trade.exit_reason = "eod"

    # PnL: sum across legs at common exit
    pnl_pts = 0.0
    n_round_trips = 0.0
    for leg_ts, leg_price, leg_size in trade.legs:
        leg_pnl = (trade.exit_price - leg_price) * sign * leg_size
        pnl_pts += leg_pnl
        n_round_trips += leg_size
    trade.pnl_points = pnl_pts
    trade.pnl_dollars = pnl_pts * POINT_VALUE - n_round_trips * COMMISSION_PER_SIDE * 2

    return trade


# ─── metrics + reporting ────────────────────────────────────────────────────

def metrics(trades: list[Trade]) -> dict:
    if not trades:
        return {"n": 0, "wr": 0, "pf": 0, "net_pnl": 0, "max_dd": 0,
                "avg_pnl": 0, "median_pnl": 0}
    pnls = np.array([t.pnl_dollars for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    pf = (wins.sum() / abs(losses.sum())) if len(losses) > 0 and losses.sum() != 0 else float("inf")
    cum = pnls.cumsum()
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak).min()
    return {
        "n": len(trades),
        "wr": float(len(wins)) / len(trades) * 100,
        "pf": float(pf),
        "net_pnl": float(pnls.sum()),
        "max_dd": float(dd),
        "avg_pnl": float(pnls.mean()),
        "median_pnl": float(np.median(pnls)),
        "wins": int(len(wins)),
        "losses": int(len(losses)),
    }


def trades_to_df(trades: list[Trade]) -> pd.DataFrame:
    rows = []
    for t in trades:
        rows.append({
            "entry_date": t.entry_date.date(),
            "entry_ts": t.entry_ts,
            "side": t.side,
            "scenario": t.scenario,
            "initial_atr": t.initial_atr,
            "n_legs": len(t.legs),
            "n_adds": t.n_adds,
            "total_size": t.total_size,
            "avg_entry": t.avg_entry,
            "exit_ts": t.exit_ts,
            "exit_price": t.exit_price,
            "exit_reason": t.exit_reason,
            "pnl_points": t.pnl_points,
            "pnl_dollars": t.pnl_dollars,
        })
    return pd.DataFrame(rows)


def run_scenario(scenario: str, signals_df: pd.DataFrame, bars_5m: pd.DataFrame,
                 atr_5m: pd.Series, scaling: bool) -> list[Trade]:
    trades = []
    for _, sig in signals_df.iterrows():
        entry_date = pd.Timestamp(sig["date"])
        # Get ATR at entry — last 5-min bar before 09:30 of entry_date
        day_start = entry_date.tz_localize("US/Eastern").replace(hour=9, minute=30)
        atr_window = atr_5m.loc[:day_start]
        if atr_window.empty:
            continue
        atr_at_entry = float(atr_window.iloc[-1])
        if not math.isfinite(atr_at_entry) or atr_at_entry <= 0:
            continue
        t = walk_trade(
            entry_date=entry_date, side=sig["side"], atr_at_entry=atr_at_entry,
            bars_5m=bars_5m, scaling=scaling, scenario=scenario,
        )
        if t is not None:
            trades.append(t)
    return trades


def fmt_metrics_row(label: str, m: dict) -> str:
    return (f"  {label:<35s}  n={m['n']:>4d}  WR={m['wr']:>5.1f}%  "
            f"PF={m['pf']:>5.2f}  net=${m['net_pnl']:>+10,.0f}  "
            f"DD=${m['max_dd']:>+9,.0f}  avg=${m['avg_pnl']:>+7.0f}")


# ─── main ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--end", default="2026-04-24")
    ap.add_argument("--gap-filter-pct", type=float, default=0.5)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    bars_1m = load_bars(args.start, args.end)
    print("[load] resampling to 5-min...")
    bars_5m = resample_5m(bars_1m)
    print(f"[load] 5-min bars: {len(bars_5m):,}")
    print("[load] computing ATR(14) on 5-min...")
    atr_5m = compute_atr14_5m(bars_5m)
    print("[load] computing daily RTH OHLC...")
    daily = daily_rth_ohlc(bars_1m)
    print(f"[load] daily RTH bars: {len(daily):,}")
    print()

    # Detect signals
    sig_a = detect_scenario_a(daily)
    sig_b_all = detect_scenario_b(daily, gap_filter_pct=args.gap_filter_pct)
    sig_b_unfiltered = sig_b_all
    sig_b_filtered = sig_b_all[~sig_b_all["filter_blocked"]] if not sig_b_all.empty else sig_b_all
    sig_c = detect_scenario_c(daily)

    print("=" * 96)
    print(f"SIGNAL COUNTS (Thursdays/Mondays/Wed-Fri triggers, {args.start} → {args.end})")
    print("=" * 96)
    print(f"  Scenario A (Thursday short): {len(sig_a):>4d}")
    print(f"  Scenario B (Monday short, ALL):                {len(sig_b_unfiltered):>4d}")
    print(f"  Scenario B (Monday short, gap-filter ≤{args.gap_filter_pct:.1f}%): {len(sig_b_filtered):>4d}")
    print(f"  Scenario C (Wed/Thu/Fri bias):                 {len(sig_c):>4d}")
    print()

    print("=" * 96)
    print("BACKTEST RESULTS")
    print("=" * 96)

    all_trades = {}

    # Run each scenario with and without scaling
    for scaling_label, scaling in [("WITH SCALING (pyramiding)", True),
                                    ("NO SCALING (single leg)", False)]:
        print(f"\n--- {scaling_label} ---")
        trades_a = run_scenario("A", sig_a, bars_5m, atr_5m, scaling)
        trades_b_unf = run_scenario("B_unfiltered", sig_b_unfiltered, bars_5m, atr_5m, scaling)
        trades_b_flt = run_scenario("B_filtered", sig_b_filtered, bars_5m, atr_5m, scaling)
        trades_c = run_scenario("C", sig_c, bars_5m, atr_5m, scaling)
        combined = trades_a + trades_b_flt + trades_c

        print(fmt_metrics_row(f"A: Thursday Expansion (short)", metrics(trades_a)))
        print(fmt_metrics_row(f"B: Friday Inertia (short, NO filter)", metrics(trades_b_unf)))
        print(fmt_metrics_row(f"B: Friday Inertia (short, +gap-filter)", metrics(trades_b_flt)))
        print(fmt_metrics_row(f"C: Weekly Pivot bias (Wed-Fri)", metrics(trades_c)))
        print(fmt_metrics_row(f"COMBINED (A + B-filtered + C)", metrics(combined)))

        all_trades[scaling_label] = {
            "A": trades_a, "B_unfiltered": trades_b_unf,
            "B_filtered": trades_b_flt, "C": trades_c,
            "combined": combined,
        }

    # Save trade journals + equity curves
    print()
    print("=" * 96)
    print("ARTIFACTS")
    print("=" * 96)
    for scaling_label, scen_trades in all_trades.items():
        slug = "scaled" if "WITH" in scaling_label else "noscale"
        for scen_name, tlist in scen_trades.items():
            if not tlist: continue
            df = trades_to_df(tlist)
            out = OUT_DIR / f"trades_{scen_name}_{slug}.csv"
            df.to_csv(out, index=False)
        # Equity curve for COMBINED
        if scen_trades["combined"]:
            df = trades_to_df(scen_trades["combined"]).sort_values("entry_ts")
            df["cum_pnl"] = df["pnl_dollars"].cumsum()
            eq_path = OUT_DIR / f"equity_curve_combined_{slug}.csv"
            df[["entry_date", "scenario", "side", "pnl_dollars", "cum_pnl"]].to_csv(
                eq_path, index=False)
            print(f"  → {eq_path}")

    # Plot equity curve
    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=False)
            for ax, (scaling_label, scen_trades) in zip(axes, all_trades.items()):
                slug = "scaled" if "WITH" in scaling_label else "noscale"
                for scen_name, tlist in scen_trades.items():
                    if not tlist or scen_name == "B_unfiltered": continue
                    df = trades_to_df(tlist).sort_values("entry_ts")
                    df["cum_pnl"] = df["pnl_dollars"].cumsum()
                    ax.plot(df["entry_ts"], df["cum_pnl"], label=scen_name, alpha=0.8)
                ax.set_title(scaling_label)
                ax.axhline(0, color="black", lw=0.5, alpha=0.5)
                ax.set_ylabel("Cumulative PnL ($)")
                ax.grid(alpha=0.3)
                ax.legend(loc="best", fontsize=8)
            axes[-1].set_xlabel("Date")
            fig.suptitle(f"Hougaard Highs & Lows Backtest — ES Futures ({args.start} → {args.end})")
            fig.tight_layout()
            png_path = OUT_DIR / "equity_curves.png"
            fig.savefig(png_path, dpi=110, bbox_inches="tight")
            print(f"  → {png_path}")
        except Exception as exc:
            print(f"  (plot skipped: {exc})")

    print()
    print(f"Done. Artifacts in: {OUT_DIR}")


if __name__ == "__main__":
    main()
