#!/usr/bin/env python3
"""Exploratory analysis on hour 9 / hour 10 ES bars.

Goal: find any RULE that has positive expectancy on out-of-sample data
before we try ML again. ML can't add signal where none exists; we need
to confirm the data has structure first.

Tests:
 1. Marginal directional bias of next-30min, next-60min returns
 2. Conditional bias by:
      - ETH overnight direction (up/down)
      - Gap (previous RTH close vs today's 09:30 open)
      - Day of week
      - First 5-min of RTH direction (momentum vs reversal)
      - First 30-min direction (for hour-10 entries)
 3. Distributional: tail / non-tail behavior
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PARQUET = ROOT / "es_master_outrights-2.parquet"

# Reuse iter1's roll calendar + front-month builder. Inline copy keeps this
# script self-contained and faster to iterate.
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


def build_front_month():
    """Build a continuous one-row-per-minute series.

    The parquet has multiple symbols per minute around rolls and (annoyingly)
    *some* minutes only have the back-month contract, not the calendar
    front-month. So instead of strict-equality filtering on front_symbol,
    we rank rows by closeness to front (the calendar pick gets rank 0, the
    next-quarter rank 1, etc.) and keep the single lowest-rank row per
    timestamp. This preserves coverage across the full 2011-2026 window."""
    print(f"Loading {PARQUET.name} ...")
    df = pd.read_parquet(PARQUET).sort_index()

    # Unique symbols ordered chronologically by their first-appearance.
    sym_order = (
        df.reset_index()
          .groupby("symbol", sort=False)["timestamp"]
          .min()
          .sort_values()
          .index
          .to_list()
    )
    sym_rank = {s: i for i, s in enumerate(sym_order)}

    # Calendar-front-month per row.
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

    # Distance score: 0 = exact front, larger = further-out contract.
    front_rank = df["front_symbol"].map(sym_rank).fillna(9999).astype(int)
    sym_rank_col = df["symbol"].map(sym_rank).fillna(9999).astype(int)
    df["roll_distance"] = (sym_rank_col - front_rank).abs()

    # Per-timestamp keep the row with smallest roll_distance.
    df = df.sort_values(["roll_distance"], kind="mergesort")
    front = df[~df.index.duplicated(keep="first")].sort_index()
    print(f"  front-month series: {len(front):,} bars  "
          f"({front.index.min()} → {front.index.max()})  "
          f"unique dates: {front.index.normalize().nunique():,}")
    return front[["open", "high", "low", "close", "volume", "symbol"]]


def build_daily_table(front: pd.DataFrame) -> pd.DataFrame:
    """One row per RTH day with the bars we'll condition on:
       prev_rth_close (16:00 close prev day),
       eth_high, eth_low (18:00 prev → 09:29 today),
       open_0930, close_1000, close_1100,
       high_0930_1000, low_0930_1000.
    """
    print("Building per-day table ...")
    f = front.copy()
    f["date"] = f.index.tz_convert("US/Eastern").date
    f["minute"] = f.index.minute
    f["hour"] = f.index.hour
    # 09:30 open (cash open)
    h930 = f[(f["hour"] == 9) & (f["minute"] == 30)][["open", "date"]].rename(columns={"open": "open_0930"})
    h930 = h930.reset_index().drop_duplicates(subset=["date"]).set_index("date")
    # 10:00 open
    h1000 = f[(f["hour"] == 10) & (f["minute"] == 0)][["open", "date"]].rename(columns={"open": "open_1000"})
    h1000 = h1000.reset_index().drop_duplicates(subset=["date"]).set_index("date")
    # 11:00 close
    h1100 = f[(f["hour"] == 11) & (f["minute"] == 0)][["open", "date"]].rename(columns={"open": "open_1100"})
    h1100 = h1100.reset_index().drop_duplicates(subset=["date"]).set_index("date")
    # 16:00 close (previous day cash close)
    h1600 = f[(f["hour"] == 16) & (f["minute"] == 0)][["open", "date"]].rename(columns={"open": "open_1600"})
    h1600 = h1600.reset_index().drop_duplicates(subset=["date"]).set_index("date")
    # ETH range: bars from 18:00 prev → 09:29 today
    # Approximation: take min/max over bars where date == today AND hour < 9.5
    eth = f[(f["hour"] < 9) | ((f["hour"] == 9) & (f["minute"] < 30))]
    eth_hi = eth.groupby("date")["high"].max().rename("eth_high")
    eth_lo = eth.groupby("date")["low"].min().rename("eth_low")
    # 09:30-09:59 first half-hour stats
    drive = f[(f["hour"] == 9) & (f["minute"] >= 30)]
    drive_hi = drive.groupby("date")["high"].max().rename("h930_959_hi")
    drive_lo = drive.groupby("date")["low"].min().rename("h930_959_lo")
    drive_close = drive.groupby("date").last()["close"].rename("close_0959")
    # First 5-min: 09:30-09:34
    first5 = f[(f["hour"] == 9) & (f["minute"].between(30, 34))]
    first5_close = first5.groupby("date").last()["close"].rename("close_0934")
    # Combine
    daily = pd.concat([h930, h1000, h1100, h1600, eth_hi, eth_lo,
                        drive_hi, drive_lo, drive_close, first5_close], axis=1)
    daily.index = pd.to_datetime(daily.index)
    daily["prev_close"] = daily["open_1600"].shift(1)  # previous day's 16:00 (close)
    daily = daily.dropna(subset=["open_0930", "open_1000", "prev_close"])
    print(f"Daily table: {len(daily):,} days "
          f"({daily.index.min()} → {daily.index.max()})")
    return daily


def add_derived(daily: pd.DataFrame) -> pd.DataFrame:
    d = daily.copy()
    d["gap_pct"] = (d["open_0930"] / d["prev_close"] - 1) * 100
    d["eth_range_pct"] = (d["eth_high"] - d["eth_low"]) / d["prev_close"] * 100
    d["eth_dir"] = np.sign(d["open_0930"] - (d["eth_high"] + d["eth_low"]) / 2)
    d["first5_dir"] = np.sign(d["close_0934"] - d["open_0930"])
    d["first30_dir"] = np.sign(d["close_0959"] - d["open_0930"])
    d["dow"] = d.index.dayofweek
    # Targets:
    #  - ret_h9_30m  = open_0930 → open_1000 (LONG perspective, %)
    #  - ret_h10_60m = open_1000 → open_1100 (LONG perspective, %)
    d["ret_h9_30m_pct"] = (d["open_1000"] / d["open_0930"] - 1) * 100
    d["ret_h10_60m_pct"] = (d["open_1100"] / d["open_1000"] - 1) * 100
    return d


def t_stat(x):
    x = x.dropna().values
    if len(x) < 10:
        return float("nan"), float("nan"), float("nan")
    return float(x.mean()), float(x.std() / np.sqrt(len(x))), float(x.mean() / (x.std() / np.sqrt(len(x))))


def report_section(title, df, target):
    print(f"\n=== {title} (target: {target}) ===")
    m, se, t = t_stat(df[target])
    print(f"  ALL: n={len(df):,}  mean={m*100:.4f}bps  se={se*100:.4f}bps  t={t:+.2f}")


def conditional_table(df, target, by, label):
    print(f"\n  by {label}:")
    for k, sub in df.groupby(by):
        m, se, t = t_stat(sub[target])
        if not np.isnan(m):
            n = len(sub)
            print(f"    {label}={k!s:>10}  n={n:>5}  mean={m*100:+8.3f}bps  t={t:+5.2f}  WR={(sub[target] > 0).mean()*100:5.1f}%")


def main():
    front = build_front_month()
    daily = build_daily_table(front)
    daily = add_derived(daily)
    # Strip pre-2012 (data sparsity in ETH window for first year)
    daily = daily[daily.index >= "2012-01-01"]
    print(f"\n=== analyzed sample: {len(daily):,} days  "
          f"({daily.index.min().date()} → {daily.index.max().date()}) ===")

    # Hour 9 entry returns (open_0930 → open_1000, 30 min)
    target = "ret_h9_30m_pct"
    report_section("HOUR 9 (09:30→10:00 LONG return)", daily, target)
    conditional_table(daily, target, "dow", "DOW")
    # Gap buckets
    daily["gap_bucket"] = pd.cut(daily["gap_pct"], bins=[-99, -0.5, -0.1, 0.1, 0.5, 99],
                                  labels=["gap_dn_big", "gap_dn", "flat", "gap_up", "gap_up_big"])
    conditional_table(daily, target, "gap_bucket", "gap")
    daily["eth_range_bucket"] = pd.qcut(daily["eth_range_pct"], q=4, labels=["q1_lo", "q2", "q3", "q4_hi"])
    conditional_table(daily, target, "eth_range_bucket", "eth_range")

    # Hour 10 entry returns (open_1000 → open_1100, 60 min)
    target = "ret_h10_60m_pct"
    report_section("HOUR 10 (10:00→11:00 LONG return)", daily, target)
    conditional_table(daily, target, "dow", "DOW")
    conditional_table(daily, target, "first5_dir", "first5_dir")
    conditional_table(daily, target, "first30_dir", "first30_dir")
    conditional_table(daily, target, "gap_bucket", "gap")

    # Cross-conditional: gap × first30_dir for hour 10
    print("\n  HOUR 10 by (gap_bucket, first30_dir):")
    for (g, f30), sub in daily.groupby(["gap_bucket", "first30_dir"], observed=False):
        m, se, t = t_stat(sub[target])
        if not np.isnan(m) and len(sub) >= 30:
            print(f"    gap={g!s:>10} first30={f30:+.0f}  n={len(sub):>5}  "
                  f"mean={m*100:+8.3f}bps  t={t:+5.2f}  WR={(sub[target] > 0).mean()*100:5.1f}%")


if __name__ == "__main__":
    main()
