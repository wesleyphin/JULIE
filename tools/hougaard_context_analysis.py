"""Hougaard "Highs & Lows" — STATISTICAL CONTEXT analysis (not trading).

Pivot from the executable backtest: instead of asking "does this strategy
make money with mechanical entry/exit?", ask "do these claimed statistical
patterns hold as conditional probabilities — i.e., as REGIME / BIAS context
that could inform other decisions?"

Removes all execution friction (no SL, no slippage, no EOD exit). Just
P(event_X | scenario_Y) vs the baseline P(event_X), with sample sizes,
significance, and regime-conditioned splits.

Three claims tested (per article + Hougaard's research):

  A) "Thursday Expansion" — given Mon's high is highest of Mon/Tue/Wed,
     Thursday should expand the range / break Wed's low.
     Tests: P(Thu_low < Wed_low | A), P(Thu_close < Wed_close | A),
            P(Thu_range > Mon-Wed avg range | A)

  B) "Friday Inertia → Monday weakness" — given Fri's high < Thu's high,
     Monday should break Fri's low (Hougaard cites 90%+).
     Tests: P(Mon_low < Fri_low | B), P(Mon_close < Fri_close | B),
            P(Mon_low < Fri_low ANYTIME during week | B)

  C) "Weekly Pivot" — Monday's H or L is week's H or L 60% of time.
     Tests: P(week_high == Mon_high), P(week_low == Mon_low),
            conditional P(week_high == Mon_high | Tue breaks Mon_low) etc.

Each stat is reported with:
  - Conditional probability under the trigger
  - Baseline (random / always-true probability)
  - Sample size
  - Lift = conditional / baseline
  - Regime-conditioned splits (bull vs bear, high-VIX vs low-VIX, trend day
    detection from prior 5 daily ranges)

Honest answer wins. If a claim doesn't replicate, surface it.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PARQUET = ROOT / "es_master_outrights-2.parquet"
OUT_DIR = ROOT / "artifacts" / "hougaard_context"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RTH_OPEN = "09:30"
RTH_CLOSE = "16:00"


def load_daily_rth(start: str, end: str) -> pd.DataFrame:
    """Daily RTH (09:30-16:00 ET) OHLC, sorted, weekdays only."""
    print(f"[load] reading {PARQUET} {start}→{end}...")
    df = pd.read_parquet(PARQUET)
    df = df[(df.index >= start) & (df.index <= end)]
    if "symbol" in df.columns:
        df = df.sort_values("volume", ascending=False)
        df = df[~df.index.duplicated(keep="first")].sort_index()
    df = df[["open", "high", "low", "close", "volume"]]
    print(f"[load] 1-min rows: {len(df):,}")
    rth = df.between_time(RTH_OPEN, RTH_CLOSE, inclusive="left")
    daily = rth.resample("1D").agg({"open": "first", "high": "max",
                                     "low": "min", "close": "last",
                                     "volume": "sum"}).dropna()
    daily["weekday"] = daily.index.dayofweek
    daily = daily[daily["weekday"] < 5].copy()
    # Add 50-day SMA for regime classification
    daily["sma50"] = daily["close"].rolling(50, min_periods=20).mean()
    daily["sma50_slope_5d"] = daily["sma50"].diff(5)
    daily["bull_regime"] = daily["sma50_slope_5d"] > 0
    # Daily range
    daily["range"] = daily["high"] - daily["low"]
    daily["range_pct"] = daily["range"] / daily["close"] * 100
    # Vol regime (proxy for VIX): rolling 20-day stdev of daily returns
    daily["ret_1d"] = daily["close"].pct_change()
    daily["vol_20d"] = daily["ret_1d"].rolling(20).std() * np.sqrt(252) * 100
    daily["high_vol"] = daily["vol_20d"] > daily["vol_20d"].quantile(0.66)
    print(f"[load] daily RTH bars: {len(daily):,}")
    return daily


# ─── helpers ────────────────────────────────────────────────────────────────

def fmt_prob(p: float, n: int) -> str:
    return f"{p*100:>5.1f}% (n={n})"


def fmt_row(label: str, conditional: float, n_cond: int,
            baseline: float, n_base: int, n_sig: int = None) -> str:
    lift = (conditional / baseline) if baseline > 0 else float("inf")
    lift_str = f"{lift:>4.2f}x"
    sig_str = f"  trigger n={n_sig}" if n_sig is not None else ""
    return (f"  {label:<55s}  "
            f"P|cond={conditional*100:>5.1f}% (n={n_cond})  "
            f"baseline={baseline*100:>5.1f}% (n={n_base})  "
            f"lift={lift_str}{sig_str}")


def regime_split(daily: pd.DataFrame, mask_trigger: pd.Series,
                  mask_event: pd.Series, label: str) -> str:
    """Show the conditional P(event | trigger) split by regime."""
    lines = [f"\n  Regime split — {label}:"]
    for regime_label, regime_mask in [
        ("bull (sma50 rising)", daily["bull_regime"] == True),
        ("bear (sma50 flat/falling)", daily["bull_regime"] == False),
        ("high vol (top 33% 20d)", daily["high_vol"] == True),
        ("low/mid vol", daily["high_vol"] == False),
    ]:
        m = mask_trigger & regime_mask
        n = m.sum()
        if n < 5:
            lines.append(f"    {regime_label:<28s}: n={n} (too few)")
            continue
        p = (mask_event & m).sum() / n
        lines.append(f"    {regime_label:<28s}: P|cond={p*100:>5.1f}% (n={n})")
    return "\n".join(lines)


# ─── Scenario A: Thursday Expansion ─────────────────────────────────────────

def analyze_scenario_a(daily: pd.DataFrame) -> str:
    out = ["", "=" * 110,
           "SCENARIO A — Thursday Expansion (Mon high > Tue high AND Mon high > Wed high)",
           "=" * 110]
    # Build aligned series for each Thursday
    by_date = {d.date(): row for d, row in daily.iterrows()}
    thu_records = []
    for thu_ts, thu in daily.iterrows():
        if thu["weekday"] != 3: continue
        wed = (thu_ts - pd.Timedelta(days=1)).date()
        tue = (thu_ts - pd.Timedelta(days=2)).date()
        mon = (thu_ts - pd.Timedelta(days=3)).date()
        if not (mon in by_date and tue in by_date and wed in by_date): continue
        m, t, w = by_date[mon], by_date[tue], by_date[wed]
        thu_records.append({
            "date": thu_ts.date(),
            "trigger_A": m["high"] > t["high"] and m["high"] > w["high"],
            "thu_low_lt_wed_low": thu["low"] < w["low"],
            "thu_close_lt_wed_close": thu["close"] < w["close"],
            "thu_high_lt_wed_high": thu["high"] < w["high"],
            "thu_range_pct": (thu["high"] - thu["low"]) / thu["close"] * 100,
            "mon_wed_avg_range_pct": (
                ((m["high"] - m["low"]) + (t["high"] - t["low"]) + (w["high"] - w["low"]))
                / 3 / w["close"] * 100
            ),
            "thu_range_expanded": (thu["high"] - thu["low"]) > (
                ((m["high"] - m["low"]) + (t["high"] - t["low"]) + (w["high"] - w["low"])) / 3
            ),
            "bull_regime": thu["bull_regime"],
            "high_vol": thu["high_vol"],
        })
    thu_df = pd.DataFrame(thu_records)
    n_thu = len(thu_df); n_trig = thu_df["trigger_A"].sum()
    out.append(f"\n  Total Thursdays in window: {n_thu}")
    out.append(f"  Scenario-A triggered: {n_trig} ({n_trig/n_thu*100:.1f}% of Thursdays)")

    # Tests
    cond = thu_df[thu_df["trigger_A"]]
    base = thu_df[~thu_df["trigger_A"]]

    tests = [
        ("Thursday low < Wednesday low", "thu_low_lt_wed_low"),
        ("Thursday close < Wednesday close", "thu_close_lt_wed_close"),
        ("Thursday high < Wednesday high (failed-to-break)", "thu_high_lt_wed_high"),
        ("Thursday range > avg of Mon-Wed range", "thu_range_expanded"),
    ]
    out.append("\n  Conditional probabilities:")
    for label, col in tests:
        p_cond = cond[col].mean()
        p_base = base[col].mean()
        out.append(fmt_row(label, p_cond, len(cond), p_base, len(base), n_sig=n_trig))

    # Regime splits for the headline test
    out.append(regime_split(
        daily=thu_df.set_index(pd.to_datetime(thu_df["date"])),
        mask_trigger=thu_df.set_index(pd.to_datetime(thu_df["date"]))["trigger_A"],
        mask_event=thu_df.set_index(pd.to_datetime(thu_df["date"]))["thu_low_lt_wed_low"],
        label="Thursday low < Wednesday low",
    ))
    return "\n".join(out)


# ─── Scenario B: Friday Inertia ─────────────────────────────────────────────

def analyze_scenario_b(daily: pd.DataFrame) -> str:
    out = ["", "=" * 110,
           "SCENARIO B — Friday Inertia (Friday high < Thursday high) → Monday weakness",
           "=" * 110]
    by_date = {d.date(): row for d, row in daily.iterrows()}
    mon_records = []
    for mon_ts, mon in daily.iterrows():
        if mon["weekday"] != 0: continue
        fri = (mon_ts - pd.Timedelta(days=3)).date()
        thu = (mon_ts - pd.Timedelta(days=4)).date()
        if not (fri in by_date and thu in by_date): continue
        f, t = by_date[fri], by_date[thu]
        # Test the broader claim: low touched anytime in next 5 trading days
        next_week_dates = [(mon_ts + pd.Timedelta(days=i)).date() for i in range(0, 5)]
        next_week_lows = [by_date[d]["low"] for d in next_week_dates if d in by_date]
        any_break = any(low < f["low"] for low in next_week_lows) if next_week_lows else False
        gap_pct = (mon["open"] - f["close"]) / f["close"] * 100  # signed
        mon_records.append({
            "date": mon_ts.date(),
            "trigger_B": f["high"] < t["high"],
            "mon_low_lt_fri_low": mon["low"] < f["low"],
            "mon_close_lt_fri_close": mon["close"] < f["close"],
            "mon_close_lt_fri_low": mon["close"] < f["low"],
            "any_day_lt_fri_low_next_week": any_break,
            "gap_pct_abs": abs(gap_pct),
            "gap_pct_signed": gap_pct,
            "bull_regime": mon["bull_regime"],
            "high_vol": mon["high_vol"],
        })
    mon_df = pd.DataFrame(mon_records)
    n_mon = len(mon_df); n_trig = mon_df["trigger_B"].sum()
    out.append(f"\n  Total Mondays in window: {n_mon}")
    out.append(f"  Scenario-B triggered: {n_trig} ({n_trig/n_mon*100:.1f}% of Mondays)")

    cond = mon_df[mon_df["trigger_B"]]
    base = mon_df[~mon_df["trigger_B"]]

    tests = [
        ("Monday low < Friday low (intraday touch)", "mon_low_lt_fri_low"),
        ("Monday close < Friday close", "mon_close_lt_fri_close"),
        ("Monday close < Friday low (strong day)", "mon_close_lt_fri_low"),
        ("Friday low broken ANY day in next week (broader)", "any_day_lt_fri_low_next_week"),
    ]
    out.append("\n  Conditional probabilities:")
    for label, col in tests:
        p_cond = cond[col].mean()
        p_base = base[col].mean()
        out.append(fmt_row(label, p_cond, len(cond), p_base, len(base), n_sig=n_trig))

    # Gap filter
    out.append("\n  Gap-filter conditional (P|trigger AND |gap%|≤0.5%):")
    cond_lowgap = cond[cond["gap_pct_abs"] <= 0.5]
    base_lowgap = base[base["gap_pct_abs"] <= 0.5]
    for label, col in tests:
        if len(cond_lowgap) < 5:
            out.append(f"  {label:<55s}  (n={len(cond_lowgap)} too small)")
            continue
        p_cond = cond_lowgap[col].mean()
        p_base = base_lowgap[col].mean() if len(base_lowgap) >= 5 else float("nan")
        out.append(fmt_row(label, p_cond, len(cond_lowgap), p_base, len(base_lowgap),
                          n_sig=len(cond_lowgap)))

    # Regime split for headline test
    out.append(regime_split(
        daily=mon_df.set_index(pd.to_datetime(mon_df["date"])),
        mask_trigger=mon_df.set_index(pd.to_datetime(mon_df["date"]))["trigger_B"],
        mask_event=mon_df.set_index(pd.to_datetime(mon_df["date"]))["mon_low_lt_fri_low"],
        label="Monday low < Friday low",
    ))
    return "\n".join(out)


# ─── Scenario C: Weekly Pivot ───────────────────────────────────────────────

def analyze_scenario_c(daily: pd.DataFrame) -> str:
    out = ["", "=" * 110,
           "SCENARIO C — Weekly Pivot (Monday's H or L = week's H or L)",
           "=" * 110]
    daily = daily.copy()
    daily["iso_year"] = daily.index.isocalendar().year
    daily["iso_week"] = daily.index.isocalendar().week
    week_records = []
    for (yr, wk), wk_grp in daily.groupby(["iso_year", "iso_week"]):
        wk_grp = wk_grp.sort_index()
        # Need at least Monday + one more day
        if 0 not in wk_grp["weekday"].values: continue
        mon = wk_grp[wk_grp["weekday"] == 0].iloc[0]
        rest = wk_grp[wk_grp["weekday"] > 0]
        if rest.empty: continue
        wk_high = wk_grp["high"].max()
        wk_low = wk_grp["low"].min()
        rest_high = rest["high"].max()
        rest_low = rest["low"].min()
        tue_row = rest[rest["weekday"] == 1]
        tue_breaks_mon_high = (not tue_row.empty) and (tue_row.iloc[0]["high"] > mon["high"])
        tue_breaks_mon_low = (not tue_row.empty) and (tue_row.iloc[0]["low"] < mon["low"])
        week_records.append({
            "iso_year": yr, "iso_week": wk,
            "mon_is_week_high": abs(mon["high"] - wk_high) < 1e-6,
            "mon_is_week_low": abs(mon["low"] - wk_low) < 1e-6,
            "mon_is_week_h_or_l": (abs(mon["high"] - wk_high) < 1e-6) or
                                   (abs(mon["low"] - wk_low) < 1e-6),
            "tue_breaks_mon_high": tue_breaks_mon_high,
            "tue_breaks_mon_low": tue_breaks_mon_low,
            "mon_low_holds_as_week_low": (abs(mon["low"] - wk_low) < 1e-6),
            "mon_high_holds_as_week_high": (abs(mon["high"] - wk_high) < 1e-6),
            "week_continues_up_after_tue_breaks_mon_high":
                tue_breaks_mon_high and (rest_high > tue_row.iloc[0]["high"] if not tue_row.empty else False),
            "week_continues_down_after_tue_breaks_mon_low":
                tue_breaks_mon_low and (rest_low < tue_row.iloc[0]["low"] if not tue_row.empty else False),
        })
    wk_df = pd.DataFrame(week_records)
    n_wk = len(wk_df)
    out.append(f"\n  Total weeks in window: {n_wk}")

    out.append("\n  Base rates (claim: 60% of weeks, Mon's H or L is week's H/L):")
    out.append(f"  {'Mon high == week high':<55s}  P={wk_df['mon_is_week_high'].mean()*100:>5.1f}% (n={n_wk})")
    out.append(f"  {'Mon low  == week low':<55s}  P={wk_df['mon_is_week_low'].mean()*100:>5.1f}% (n={n_wk})")
    out.append(f"  {'Mon EITHER  == week H or L':<55s}  P={wk_df['mon_is_week_h_or_l'].mean()*100:>5.1f}% (n={n_wk})")

    # Conditional: when Tuesday breaks Mon high → does week continue up?
    out.append("\n  Conditional bias claim (article's rule):")
    n_tueh = wk_df["tue_breaks_mon_high"].sum()
    n_tuel = wk_df["tue_breaks_mon_low"].sum()
    n_baseh = (~wk_df["tue_breaks_mon_high"]).sum()
    n_basel = (~wk_df["tue_breaks_mon_low"]).sum()
    if n_tueh >= 5:
        p_cond = wk_df.loc[wk_df["tue_breaks_mon_high"], "week_continues_up_after_tue_breaks_mon_high"].mean()
        out.append(fmt_row("Tue breaks Mon-high → week extends ABOVE Tue high",
                          p_cond, n_tueh, 0.0, n_baseh, n_sig=n_tueh))
    if n_tuel >= 5:
        p_cond = wk_df.loc[wk_df["tue_breaks_mon_low"], "week_continues_down_after_tue_breaks_mon_low"].mean()
        out.append(fmt_row("Tue breaks Mon-low → week extends BELOW Tue low",
                          p_cond, n_tuel, 0.0, n_basel, n_sig=n_tuel))
    # Inverse claim: when Tue breaks Mon-high, does Mon-low STILL hold as week low?
    if n_tueh >= 5:
        p = wk_df.loc[wk_df["tue_breaks_mon_high"], "mon_low_holds_as_week_low"].mean()
        out.append(f"  {'  Tue breaks Mon-high: Mon-low STILL holds as week low':<55s}  P={p*100:>5.1f}% (n={n_tueh})")
    if n_tuel >= 5:
        p = wk_df.loc[wk_df["tue_breaks_mon_low"], "mon_high_holds_as_week_high"].mean()
        out.append(f"  {'  Tue breaks Mon-low: Mon-high STILL holds as week high':<55s}  P={p*100:>5.1f}% (n={n_tuel})")
    return "\n".join(out)


# ─── overall bull/bear regime split summary ─────────────────────────────────

def regime_summary(daily: pd.DataFrame) -> str:
    out = ["", "=" * 110, "REGIME CHARACTERIZATION OF THE TEST PERIOD",
           "=" * 110]
    bull_pct = daily["bull_regime"].mean() * 100
    high_vol_pct = daily["high_vol"].mean() * 100
    drift_total = (daily["close"].iloc[-1] - daily["close"].iloc[0]) / daily["close"].iloc[0] * 100
    out.append(f"\n  Period: {daily.index.min().date()} → {daily.index.max().date()}")
    out.append(f"  Total drift (close): {drift_total:+.1f}%")
    out.append(f"  Days in bull regime (50d-SMA rising):   {bull_pct:>5.1f}%")
    out.append(f"  Days in high-vol regime (top-33% 20d):  {high_vol_pct:>5.1f}%")
    out.append("\n  ⚠ Test window is dominated by uptrending bull market — "
               "any 'short-bias' pattern will under-perform vs Hougaard's claim.")
    return "\n".join(out)


# ─── main ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--end", default="2026-04-24")
    args = ap.parse_args()

    daily = load_daily_rth(args.start, args.end)
    sections = [
        regime_summary(daily),
        analyze_scenario_a(daily),
        analyze_scenario_b(daily),
        analyze_scenario_c(daily),
    ]
    report = "\n".join(sections)
    print(report)
    out = OUT_DIR / "context_report.txt"
    out.write_text(report)
    print(f"\n  → {out}")


if __name__ == "__main__":
    main()
