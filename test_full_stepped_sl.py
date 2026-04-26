"""Tests for the full integrated stepped-SL simulator (Phase 6).

These verify:
  a) Smoking gun (2026-03-05 ESH6 LONG @ 6855) STILL exits horizon — MFE 9.5pt
     never crosses BE-arm threshold (10pt for DE3 TP=25pt).
  b) BE-arm rescue: synthetic LONG hits +10pt then reverses to original SL.
     Must exit at entry (raw_pnl ~ 0), NOT at SL (-$50).
  c) Pivot lock-in: synthetic LONG hits +20pt with a swing pivot at +15pt then
     reverses. Must exit at the pivot-ratcheted SL (locking in profit), NOT
     at original SL.
  d) BE-arm + Pivot Trail interaction: BE-arm fires at +10pt then Pivot Trail
     fires at +20pt with swing — final SL is the higher pivot level.
  e) Phantom-fill regression test still passes via the existing
     simulator_trade_through (covered by test_simulator_phantom_fill_fix.py).

Run:
    python -m pytest test_full_stepped_sl.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from tools.full_stepped_sl_simulator import (  # noqa: E402
    BE_TRIGGER_PCT,
    simulate_full_stepped,
    simulate_trade_with_full_steps,
)

PARQUET = ROOT / "es_master_outrights.parquet"


@pytest.fixture(scope="module")
def parquet_df():
    if not PARQUET.exists():
        pytest.skip(f"parquet not found at {PARQUET}")
    df = pd.read_parquet(PARQUET)
    return df


# ------------- (a) smoking gun unchanged -------------

def test_smoking_gun_unchanged(parquet_df):
    """2026-03-05 ESH6 LONG @ 6855 — MFE = 9.5pt < 10pt BE-arm threshold,
    so BE-arm does NOT fire. Pivot Trail also does not fire (no qualifying
    swing). Trade still exits at horizon ~6851.25, no phantom TP.
    """
    result = simulate_trade_with_full_steps(
        parquet_df,
        entry_ts="2026-03-05 08:05:00",
        side="LONG",
        entry_price=6855.0,
        sl_distance=10.0,   # SL=6845
        tp_distance=25.0,   # TP=6880
        contract="ESH6",
        horizon_bars=30,
    )
    assert result["exit_reason"] != "take", f"Phantom TP regression: {result}"
    assert result["contract"] == "ESH6"
    assert result["exit_reason"] in {"horizon", "stop"}, result
    assert result["mfe_points"] == pytest.approx(9.5, abs=0.5), result
    # MFE < 10pt -> BE-arm should not fire. (TP=25, threshold=25*0.4=10)
    assert result["be_armed"] is False, f"BE-arm fired below threshold: {result}"


# ------------- (b) BE-arm rescue -------------

def test_be_arm_rescue():
    """Synthetic LONG @ 100, TP=125, SL=90.
    Bar 0: high=110, low=99, close=108  -> MFE=10pt -> BE-arm fires (SL=100)
    Bar 1: high=105, low=85,  close=88  -> low < new SL=100 -> exit at SL=100
    Without BE-arm, low=85 hits SL=90 -> -$50 (loss).
    With BE-arm, exit at SL=100 -> 0 raw pnl (BE flat).
    """
    bars = pd.DataFrame({
        "open":   [100.0, 108.0],
        "high":   [110.0, 105.0],
        "low":    [99.0,  85.0],
        "close":  [108.0, 88.0],
        "volume": [10, 10],
        "symbol": ["TEST", "TEST"],
    })
    out = simulate_full_stepped(
        bars, side="LONG", entry_price=100.0,
        initial_sl=90.0, initial_tp=125.0,
        be_arm_active=True, pivot_active=False,
    )
    # MFE on bar0=10 -> BE-arm at bar0; SL becomes 100. Bar1 low=85 hits SL=100.
    assert out.be_armed is True
    assert out.be_armed_at_bar == 0
    assert out.exit_reason == "stop"
    assert out.exit_price == pytest.approx(100.0)
    assert out.pnl_points == pytest.approx(0.0)
    assert out.final_sl == pytest.approx(100.0)


def test_be_arm_no_trigger_below_threshold():
    """Below-threshold MFE: BE-arm must NOT fire."""
    # TP=25, threshold=10. MFE=8 < 10, so no arm.
    bars = pd.DataFrame({
        "open":   [100.0, 105.0],
        "high":   [108.0, 105.0],
        "low":    [99.0,  85.0],
        "close":  [105.0, 88.0],
        "volume": [10, 10],
        "symbol": ["TEST", "TEST"],
    })
    out = simulate_full_stepped(
        bars, side="LONG", entry_price=100.0,
        initial_sl=90.0, initial_tp=125.0,
        be_arm_active=True, pivot_active=False,
    )
    assert out.be_armed is False
    assert out.exit_reason == "stop"
    assert out.exit_price == pytest.approx(90.0)
    assert out.pnl_points == pytest.approx(-10.0)


# ------------- (c) Pivot Trail lock-in -------------

def test_pivot_lock_in():
    """LONG @ 100, TP=130, SL=90. Swing pivot at +15pt; pivot anchor C =
    floor(115/12.5)*12.5 = 112.5; anchor B = 100; candidate B = 99.75;
    99.75 <= entry=100, so use anchor C = 112.5 - 0.25 = 112.25.
    Pivot Trail fires when MFE >= 12.5pt (PIVOT_TRAIL_MIN_PROFIT_PTS).
    Then bar reverses, low touches 90 -> but new SL is 112.25, so exit
    at 112.25 (locking in 12.25pt profit).

    To trigger detect_pivot_high(window=5), need 5 bars with the middle
    bar (index 2) the highest. Use US session timestamp.
    """
    times = pd.date_range("2026-03-04 10:00", periods=6, freq="1min", tz="US/Eastern")
    # 5-bar window where middle bar (index 2) high=115 is the max:
    # bar0 high=105, bar1 high=110, bar2 high=115 (PIVOT), bar3 high=112,
    # bar4 high=108. After bar4 (window full of 5 bars: 0..4), pivot detected.
    # Then bar5 plunges low to 90.
    bars = pd.DataFrame({
        "timestamp": times,
        "open":   [100.0, 105.0, 110.0, 115.0, 112.0, 108.0],
        "high":   [105.0, 110.0, 115.0, 112.0, 108.0, 109.0],
        "low":    [99.0,  104.0, 109.0, 110.0, 107.0, 90.0],
        "close":  [105.0, 109.0, 113.0, 111.0, 108.0, 92.0],
        "volume": [10, 10, 10, 10, 10, 10],
        "symbol": ["TEST"] * 6,
    }).set_index("timestamp")

    out = simulate_full_stepped(
        bars, side="LONG", entry_price=100.0,
        initial_sl=90.0, initial_tp=130.0,
        be_arm_active=False,  # isolate pivot trail
        pivot_active=True,
        us_session_only=True,
    )
    assert out.pivot_armed is True, f"Pivot Trail did NOT arm: {out}"
    # Final SL must be above entry
    assert out.final_sl > 100.0, out
    # Reasonable pivot anchor: 112.25 (from anchor C - buffer)
    assert out.final_sl == pytest.approx(112.25, abs=0.01), out
    assert out.exit_reason == "stop"
    assert out.exit_price == pytest.approx(112.25, abs=0.01)
    assert out.pnl_points == pytest.approx(12.25, abs=0.01)


# ------------- (d) BE-arm + Pivot Trail interaction -------------

def test_be_arm_then_pivot_supersedes():
    """BE-arm fires first (bar0 MFE=10 -> SL=entry=100), THEN Pivot Trail
    fires later (bar4-bar5 sweep with pivot at +15pt -> SL=112.25).
    Final SL must be the HIGHER pivot level, not entry.
    """
    times = pd.date_range("2026-03-04 10:00", periods=6, freq="1min", tz="US/Eastern")
    bars = pd.DataFrame({
        "timestamp": times,
        # bar0: high=110 -> MFE=10 -> BE-arm (SL becomes 100). low=100 just
        #       holds (any-touch is l <= sl, but l=100.5 > 100 keeps trade
        #       alive on bar0).
        # bars1..4: 5-bar window with middle bar high=115 -> Pivot fires after
        #       bar4 -> SL ratchets to 112.25
        # bar5: plunge to low=90 -> exit at 112.25 (pivot SL)
        "open":   [100.0, 108.0, 110.0, 113.0, 112.0, 108.0],
        "high":   [110.0, 110.0, 115.0, 113.0, 108.0, 109.0],
        "low":    [100.5, 107.0, 109.0, 111.0, 107.0, 90.0],
        "close":  [108.0, 109.0, 113.0, 112.0, 108.0, 92.0],
        "volume": [10, 10, 10, 10, 10, 10],
        "symbol": ["TEST"] * 6,
    }).set_index("timestamp")

    out = simulate_full_stepped(
        bars, side="LONG", entry_price=100.0,
        initial_sl=90.0, initial_tp=125.0,
        be_arm_active=True, pivot_active=True,
        us_session_only=True,
    )
    assert out.be_armed is True
    assert out.be_armed_at_bar == 0  # MFE=10 first crosses threshold on bar0
    assert out.pivot_armed is True, f"pivot did not arm: {out}"
    # Final SL must be pivot-level (~112.25), not entry (100).
    assert out.final_sl > 100.0 + 1e-6, out
    assert out.final_sl == pytest.approx(112.25, abs=0.01), out
    assert out.exit_reason == "stop"
    assert out.pnl_points == pytest.approx(12.25, abs=0.01)


def test_short_be_arm_rescue():
    """SHORT @ 100, TP=75, SL=110.
    Bar0: low=90, high=101 -> MFE=10 (entry-low) -> BE-arm -> SL=100
    Bar1: high=115, low=92 -> high>=SL=100 -> exit at SL=100, pnl=0
    Without BE-arm: high=115 >= 110 -> exit at SL=110, pnl=-10.
    """
    bars = pd.DataFrame({
        "open":   [100.0, 92.0],
        "high":   [101.0, 115.0],
        "low":    [90.0,  92.0],
        "close":  [92.0,  113.0],
        "volume": [10, 10],
        "symbol": ["TEST", "TEST"],
    })
    out = simulate_full_stepped(
        bars, side="SHORT", entry_price=100.0,
        initial_sl=110.0, initial_tp=75.0,
        be_arm_active=True, pivot_active=False,
    )
    assert out.be_armed is True
    assert out.exit_reason == "stop"
    assert out.exit_price == pytest.approx(100.0)
    assert out.pnl_points == pytest.approx(0.0)


# ------------- (e) phantom-fill regression -------------

def test_phantom_fill_regression(parquet_df):
    """The phantom-fill fix from §8.25 must still hold under the full sim.
    2026-03-05 ESH6 LONG @ 6855 must NOT phantom-TP.
    """
    result = simulate_trade_with_full_steps(
        parquet_df,
        entry_ts="2026-03-05 08:05:00",
        side="LONG",
        entry_price=6855.0,
        sl_distance=10.0,
        tp_distance=25.0,
        contract="ESH6",
        horizon_bars=30,
    )
    assert result["exit_reason"] != "take", f"Phantom TP: {result}"


def test_be_active_off_matches_baseline_path():
    """With both add-ons disabled, the full sim should match the baseline
    trade-through sim's exit reason and price exactly on a synthetic case.
    """
    bars = pd.DataFrame({
        "open":   [100.0, 103.0],
        "high":   [104.0, 106.0],
        "low":    [99.0,  102.0],
        "close":  [103.0, 105.5],
        "volume": [10, 10],
        "symbol": ["TEST", "TEST"],
    })
    out = simulate_full_stepped(
        bars, side="LONG", entry_price=100.0,
        initial_sl=95.0, initial_tp=105.0,
        be_arm_active=False, pivot_active=False,
    )
    assert out.exit_reason == "take"
    assert out.exit_price == pytest.approx(105.0)


def test_be_arm_constants_match_julie001():
    """BE_TRIGGER_PCT must match julie001.py:4678 / config.py:6070 = 0.40.
    If the live config changes, this test must be updated together with the
    sim — and the corpus must be re-walked.
    """
    assert BE_TRIGGER_PCT == pytest.approx(0.40)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
