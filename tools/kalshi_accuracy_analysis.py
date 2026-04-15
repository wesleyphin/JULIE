"""
Kalshi KXINXU S&P 500 Hourly Prediction Market Accuracy Analysis

Each row in the Kalshi CSV represents one hourly settlement event:
  - num_yes / num_no = how many of the 400 (or 60) strike contracts had
    the crowd predicting YES vs NO as dominant
  - A YES-heavy event means the crowd is broadly bullish (expects SPX to
    stay within / above many strike ranges)

This script tests whether that crowd sentiment signal correctly predicts
SPX hourly price direction — the same signal used for 3x trade sizing.
"""

import pandas as pd
import numpy as np


def main():
    # Load data
    kalshi = pd.read_csv("/Users/wes/Downloads/kalshi_sp500_hourly_summary.csv")
    spx = pd.read_csv("/Users/wes/Downloads/spx_hourly_price_3m.csv")

    print(f"Kalshi rows: {len(kalshi)} | Date range: {kalshi['close_date'].min()} to {kalshi['close_date'].max()}")
    print(f"SPX rows: {len(spx)}")
    print()

    # Clean SPX data: drop non-numeric header rows, parse types
    spx = spx[pd.to_numeric(spx["Close"], errors="coerce").notna()].copy()
    spx["Close"] = pd.to_numeric(spx["Close"])
    spx["Datetime"] = pd.to_datetime(spx["Datetime"], utc=True)
    spx = spx.sort_values("Datetime").reset_index(drop=True)

    # Parse Kalshi timestamps
    kalshi["close_time_dt"] = pd.to_datetime(kalshi["close_time"], utc=True)
    kalshi = kalshi.sort_values("close_time_dt").reset_index(drop=True)

    # Compute crowd sentiment
    kalshi["total_votes"] = kalshi["num_yes"] + kalshi["num_no"]
    kalshi["yes_ratio"] = kalshi["num_yes"] / kalshi["total_votes"]
    kalshi["crowd_bullish"] = kalshi["num_yes"] > kalshi["num_no"]
    kalshi["confidence"] = kalshi[["num_yes", "num_no"]].max(axis=1) / kalshi["total_votes"] * 100

    # Compute SPX hourly price change for each Kalshi event
    # Match each Kalshi event to SPX data by finding the closest prior and at-settlement prices
    spx_changes = []
    for _, row in kalshi.iterrows():
        settle_time = row["close_time_dt"]
        prior_time = settle_time - pd.Timedelta(hours=1)

        # Find SPX price at/near settlement and 1 hour before
        at_settle = spx[(spx["Datetime"] >= settle_time - pd.Timedelta(minutes=30)) &
                        (spx["Datetime"] <= settle_time + pd.Timedelta(minutes=30))]
        at_prior = spx[(spx["Datetime"] >= prior_time - pd.Timedelta(minutes=30)) &
                       (spx["Datetime"] <= prior_time + pd.Timedelta(minutes=30))]

        if len(at_settle) > 0 and len(at_prior) > 0:
            settle_price = at_settle.iloc[-1]["Close"]
            prior_price = at_prior.iloc[-1]["Close"]
            spx_changes.append({
                "event_ticker": row["event_ticker"],
                "settle_price": settle_price,
                "prior_price": prior_price,
                "spx_change": settle_price - prior_price,
                "spx_up": settle_price > prior_price,
            })
        else:
            spx_changes.append({
                "event_ticker": row["event_ticker"],
                "settle_price": None,
                "prior_price": None,
                "spx_change": None,
                "spx_up": None,
            })

    changes_df = pd.DataFrame(spx_changes)
    merged = kalshi.merge(changes_df, on="event_ticker", how="left")
    matched = merged.dropna(subset=["spx_change"]).copy()

    print(f"Events matched to SPX price data: {len(matched)} of {len(kalshi)}")
    unmatched = len(kalshi) - len(matched)
    if unmatched > 0:
        print(f"  ({unmatched} events could not be matched to SPX hourly data)")
    print()

    if len(matched) == 0:
        print("No matched data — cannot perform analysis.")
        return

    # Directional accuracy: did crowd bullish/bearish match SPX direction?
    matched["direction_correct"] = matched["crowd_bullish"] == matched["spx_up"]
    flat = (matched["spx_change"] == 0).sum()
    non_flat = matched[matched["spx_change"] != 0].copy()

    print("=" * 60)
    print("DIRECTIONAL PREDICTION ACCURACY")
    print("=" * 60)
    print(f"Total matched events (excl flat): {len(non_flat)}")
    if len(non_flat) > 0:
        dir_acc = non_flat["direction_correct"].mean() * 100
        print(f"Direction correct: {dir_acc:.1f}%")
        print(f"  (baseline random = 50.0%)")
        print(f"  Edge over random: {dir_acc - 50:.1f} pp")
    print(f"Flat moves (excluded): {flat}")
    print()

    # By confidence tier
    print("=" * 60)
    print("DIRECTIONAL ACCURACY BY CONFIDENCE TIER")
    print("=" * 60)
    tiers = [(50, 55), (55, 60), (60, 65), (65, 70), (70, 80), (80, 100)]
    print(f"{'Tier':<12} {'Events':>8} {'Correct':>8} {'Accuracy':>10} {'Avg Move':>10}")
    print("-" * 50)
    for lo, hi in tiers:
        tier = non_flat[(non_flat["confidence"] >= lo) & (non_flat["confidence"] < hi)]
        if len(tier) == 0:
            print(f"{lo}-{hi}%{'':<7} {'0':>8} {'--':>8} {'--':>10} {'--':>10}")
            continue
        correct = tier["direction_correct"].sum()
        acc = correct / len(tier) * 100
        avg_move = tier["spx_change"].abs().mean()
        print(f"{lo}-{hi}%{'':<7} {len(tier):>8} {correct:>8} {acc:>9.1f}% {avg_move:>9.2f}")
    print()

    # By hour
    print("=" * 60)
    print("DIRECTIONAL ACCURACY BY HOUR (ET)")
    print("=" * 60)
    print(f"{'Hour':<8} {'Events':>8} {'Correct':>8} {'Accuracy':>10} {'Avg Move':>10}")
    print("-" * 46)
    for hour in sorted(non_flat["contract_hour_et"].unique()):
        h_df = non_flat[non_flat["contract_hour_et"] == hour]
        correct = h_df["direction_correct"].sum()
        acc = correct / len(h_df) * 100
        avg_move = h_df["spx_change"].abs().mean()
        print(f"{hour:<8} {len(h_df):>8} {correct:>8} {acc:>9.1f}% {avg_move:>9.2f}")
    print()

    # Bullish vs bearish crowd accuracy
    print("=" * 60)
    print("ACCURACY BY CROWD DIRECTION")
    print("=" * 60)
    for label, subset in [("Bullish (YES > NO)", non_flat[non_flat["crowd_bullish"]]),
                          ("Bearish (NO > YES)", non_flat[~non_flat["crowd_bullish"]])]:
        if len(subset) == 0:
            print(f"  {label}: no events")
            continue
        acc = subset["direction_correct"].mean() * 100
        print(f"  {label}: {len(subset)} events, {acc:.1f}% accuracy")
    print()

    # Mean SPX move when crowd agrees vs disagrees with actual direction
    print("=" * 60)
    print("SIGNAL STRENGTH ANALYSIS")
    print("=" * 60)
    correct_moves = non_flat[non_flat["direction_correct"]]["spx_change"].abs()
    wrong_moves = non_flat[~non_flat["direction_correct"]]["spx_change"].abs()
    if len(correct_moves) > 0:
        print(f"  When crowd is RIGHT: avg |move| = {correct_moves.mean():.2f} pts ({len(correct_moves)} events)")
    if len(wrong_moves) > 0:
        print(f"  When crowd is WRONG: avg |move| = {wrong_moves.mean():.2f} pts ({len(wrong_moves)} events)")
    print()

    # High confidence events (>60%) — most relevant for 3x sizing
    print("=" * 60)
    print("HIGH CONFIDENCE (60%+) — TRADE GATING RELEVANCE")
    print("=" * 60)
    high_conf = non_flat[non_flat["confidence"] >= 60.0]
    if len(high_conf) > 0:
        hc_acc = high_conf["direction_correct"].mean() * 100
        print(f"  Events: {len(high_conf)}")
        print(f"  Directional accuracy: {hc_acc:.1f}%")
        print(f"  Avg |SPX move|: {high_conf['spx_change'].abs().mean():.2f} pts")
        hc_correct = high_conf[high_conf["direction_correct"]]["spx_change"].abs().mean()
        hc_wrong = high_conf[~high_conf["direction_correct"]]["spx_change"].abs().mean()
        if not np.isnan(hc_correct):
            print(f"  Avg |move| when RIGHT: {hc_correct:.2f} pts")
        if not np.isnan(hc_wrong):
            print(f"  Avg |move| when WRONG: {hc_wrong:.2f} pts")
        print()
        print("  → This is the signal used for 3x sizing during trade gating hours.")
        edge = hc_acc - 50
        if edge > 5:
            print(f"  → {edge:.1f}pp edge suggests the 3x multiplier adds value.")
        elif edge > 0:
            print(f"  → {edge:.1f}pp edge is marginal — 3x sizing is slightly positive EV.")
        else:
            print(f"  → {edge:.1f}pp — no directional edge at this confidence level.")
    else:
        print("  No events with 60%+ confidence found.")
    print()

    # YES ratio distribution
    print("=" * 60)
    print("YES RATIO DISTRIBUTION")
    print("=" * 60)
    print(f"  Mean yes_ratio: {matched['yes_ratio'].mean():.3f}")
    print(f"  Median yes_ratio: {matched['yes_ratio'].median():.3f}")
    print(f"  Min: {matched['yes_ratio'].min():.3f}, Max: {matched['yes_ratio'].max():.3f}")
    print(f"  Std dev: {matched['yes_ratio'].std():.3f}")


if __name__ == "__main__":
    main()
