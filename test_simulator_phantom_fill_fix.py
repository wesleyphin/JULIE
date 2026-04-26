"""Tests for the simulator_trade_through.py fix.

These tests defend the fix described in
docs/STRATEGY_ARCHITECTURE_JOURNAL.md Section 8.25 / 8.26:

  1. Phantom fills (TP awarded on a contract the trade was NEVER in).
  2. Wick-only TP fills (1-tick spike with no trade-through).

Run:
    python -m pytest test_simulator_phantom_fill_fix.py -v

If you have no pytest, you can also run as:
    python test_simulator_phantom_fill_fix.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulator_trade_through import (
    front_month_by_calendar,
    get_walk_forward_bars,
    pin_contract,
    simulate_trade,
    simulate_trade_through,
)

PARQUET = ROOT / "es_master_outrights.parquet"


@pytest.fixture(scope="module")
def parquet_df():
    if not PARQUET.exists():
        pytest.skip(f"parquet not found at {PARQUET}")
    df = pd.read_parquet(PARQUET)
    return df


def test_phantom_fill_2026_03_05(parquet_df):
    """The smoking gun from Section 8.25.

    LONG ESH6 entered at 6855.00 on 2026-03-05 08:05 ET, TP=6880, SL=6845.
    Real ESH6 max high in next 30 bars was 6864.50 -> TP never approached.
    Old simulator phantom-awarded +$117.50 because it walked merged/ESM6 bars
    where ESM6 was trading $50+ above ESH6.
    """
    result = simulate_trade(
        parquet_df,
        entry_ts="2026-03-05 08:05:00",
        side="LONG",
        entry_price=6855.0,
        tp=6880.0,
        sl=6845.0,
        contract="ESH6",
        horizon_bars=30,
    )
    assert result["exit_reason"] != "take", f"Phantom TP regression: {result}"
    # On real ESH6 bars, the max high in the next 30 bars is 6864.50 — neither TP
    # nor SL (6845) is hit, so we expect a horizon exit.
    assert result["contract"] == "ESH6"
    assert result["exit_reason"] in {"horizon", "stop"}, result


def test_contract_pinning_h6_not_m6(parquet_df):
    """At 2026-03-05 the front month is ESH6 (rolls 2026-03-12 to ESM6).
    Pinning by close-price match to 6855.00 should choose ESH6."""
    ts = pd.Timestamp("2026-03-05 08:05:00", tz="US/Eastern")
    contract = pin_contract(parquet_df, ts, signal_price=6855.0)
    assert contract == "ESH6", f"Expected ESH6, got {contract}"

    bars = get_walk_forward_bars(parquet_df, ts, contract="ESH6", horizon_bars=30)
    assert len(bars) > 0, "Empty walk-forward bars"
    assert (bars["symbol"] == "ESH6").all(), (
        f"Bar contamination: contracts in walk={set(bars['symbol'].unique())}"
    )


def test_legitimate_tp_fill():
    """Synthetic bars: clean trade-through must still award TP.

    Bars (LONG entry @ 100, TP=105, SL=95):
      bar 0: o=100, h=104, l=99, c=103     # not yet at TP
      bar 1: o=103, h=106, l=102, c=105.5  # touches AND closes >= TP-1tick
    """
    bars = pd.DataFrame(
        {
            "open":   [100.0, 103.0],
            "high":   [104.0, 106.0],
            "low":    [99.0,  102.0],
            "close":  [103.0, 105.5],
            "volume": [10, 10],
            "symbol": ["TEST", "TEST"],
        }
    )
    out = simulate_trade_through(bars, side="LONG", entry_price=100.0, tp_price=105.0, sl_price=95.0)
    assert out.exit_reason == "take", f"Legit TP not filled: {out}"
    assert out.exit_price == 105.0


def test_wick_only_no_tp():
    """Synthetic bars: 1-tick wick to TP with close back below — must NOT TP."""
    bars = pd.DataFrame(
        {
            "open":   [100.0, 103.0, 102.0],
            "high":   [104.0, 105.25, 103.0],  # bar 1 wicks to TP+1tick
            "low":    [99.0,  102.0, 101.0],
            "close":  [103.0, 102.5, 102.0],   # close way below TP
            "volume": [10, 10, 10],
            "symbol": ["TEST", "TEST", "TEST"],
        }
    )
    out = simulate_trade_through(bars, side="LONG", entry_price=100.0, tp_price=105.0, sl_price=95.0)
    assert out.exit_reason != "take", f"Wick-only awarded TP: {out}"


def test_short_legitimate_tp():
    """SHORT side trade-through TP."""
    bars = pd.DataFrame(
        {
            "open":   [100.0, 97.0],
            "high":   [101.0, 98.0],
            "low":    [96.0, 94.0],   # bar 1 trades through TP=95
            "close":  [97.0, 94.5],
            "volume": [10, 10],
            "symbol": ["TEST", "TEST"],
        }
    )
    out = simulate_trade_through(bars, side="SHORT", entry_price=100.0, tp_price=95.0, sl_price=105.0)
    assert out.exit_reason == "take", out


def test_sl_any_touch_kept_strict():
    """Any wick to SL must count as a stop (asymmetric / conservative)."""
    bars = pd.DataFrame(
        {
            "open":   [100.0, 99.0],
            "high":   [101.0, 99.5],
            "low":    [99.0, 94.5],   # 1-tick wick below SL=95
            "close":  [99.0, 99.0],
            "volume": [10, 10],
            "symbol": ["TEST", "TEST"],
        }
    )
    out = simulate_trade_through(bars, side="LONG", entry_price=100.0, tp_price=105.0, sl_price=95.0)
    assert out.exit_reason == "stop", out


def test_smoking_gun_mfe_tracked(parquet_df):
    """MFE/MAE tracking added 2026-04-25 (early-exit audit follow-up).

    The 2026-03-05 ESH6 LONG @ 6855.00 trade had a real ESH6 max-high of
    6864.50 over the next 30 bars => MFE should be ~9.50 points.
    """
    result = simulate_trade(
        parquet_df,
        entry_ts="2026-03-05 08:05:00",
        side="LONG",
        entry_price=6855.0,
        tp=6880.0,
        sl=6845.0,
        contract="ESH6",
        horizon_bars=30,
    )
    assert "mfe_points" in result and "mae_points" in result
    assert result["mfe_points"] == pytest.approx(9.5, abs=0.5), (
        f"Expected MFE ~9.5pt (ESH6 max-high 6864.50 - entry 6855.00), got {result['mfe_points']}"
    )
    assert result["mfe_points"] >= 0.0
    assert result["mae_points"] >= 0.0


def test_mfe_mae_synthetic_long():
    """Synthetic LONG: entry 100, bars sweep 95-110, must capture MFE=10, MAE=5."""
    bars = pd.DataFrame(
        {
            "open":   [100.0, 102.0, 105.0],
            "high":   [103.0, 110.0, 107.0],  # bar 1 wicks to 110 -> MFE=10
            "low":    [98.0,  95.0,  101.0],  # bar 1 wicks to 95  -> MAE=5
            "close":  [102.0, 105.0, 102.0],
            "volume": [10, 10, 10],
            "symbol": ["TEST", "TEST", "TEST"],
        }
    )
    out = simulate_trade_through(bars, side="LONG", entry_price=100.0, tp_price=120.0, sl_price=80.0)
    assert out.exit_reason == "horizon"
    assert out.mfe_points == pytest.approx(10.0)
    assert out.mae_points == pytest.approx(5.0)


def test_calendar_fallback():
    """Calendar fallback when price doesn't pin uniquely."""
    assert front_month_by_calendar(pd.Timestamp("2026-03-05")) == "ESH6"
    assert front_month_by_calendar(pd.Timestamp("2026-04-15")) == "ESM6"
    assert front_month_by_calendar(pd.Timestamp("2025-04-15")) == "ESM5"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
