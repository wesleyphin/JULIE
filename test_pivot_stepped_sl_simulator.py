"""Unit tests for the stepped-SL Pivot Trail simulator.

Three tests called out in the Phase 3 brief:
  a) Smoking gun: 2026-03-05 08:05 LONG ESH6 @ 6855 — MFE never reaches the
     12.5pt arm threshold AND the entry is outside US session (08:05 ET).
     Both pivot_active=True and pivot_active=False must produce identical
     horizon exits matching Phase 1's recorded 6851.25.
  b) Pivot arms correctly: 2025-03-03 13:41 LONG ESH5 @ 5901 SL=10 TP=25.
     Phase 1 says the trade took (exit 5926). With the trail in shadow, the
     pivot should arm somewhere along the run-up and we should still take
     the TP (or exit on a ratcheted SL — never worse than the original SL
     at 5891).
  c) Reversal lock: a LONG with strong intra-trade MFE that reverses past
     the original SL. With pivot_active=True the trailed SL must produce a
     better outcome than pivot_active=False.
"""
from __future__ import annotations

import math
import sys

import pandas as pd

from pivot_stepped_sl_simulator import (
    compute_pivot_trail_sl,
    detect_pivot_high,
    detect_pivot_low,
    simulate_trade_with_pivot_trail,
    simulate_with_pivot_trail,
)
from simulator_trade_through import get_walk_forward_bars

MASTER_PATH = "es_master_outrights.parquet"


def _load_master():
    return pd.read_parquet(MASTER_PATH)


# ---------------------------------------------------------------------------
# Sanity: pivot detection + SL math match julie001 docstring examples
# ---------------------------------------------------------------------------
def test_unit_pivot_detection_and_sl():
    # Pivot-high detection: middle bar of 5 is the highest
    highs = [10.0, 11.0, 12.5, 11.5, 10.5]
    assert detect_pivot_high(highs) == 12.5
    # Not a high (last bar is highest)
    assert detect_pivot_high([10.0, 11.0, 11.5, 11.0, 13.0]) is None
    # Pivot-low symmetric
    lows = [10.0, 9.0, 8.0, 9.0, 9.5]
    assert detect_pivot_low(lows) == 8.0

    # SL math — docstring example: pivot 5248, entry 5200 → Reading B used
    sl = compute_pivot_trail_sl("LONG", 5248.0, 5200.0, current_sl=5190.0)
    # anchor_C = floor(5248/12.5)*12.5 = 5237.5; anchor_B = 5225; cand_B = 5224.75
    # 5224.75 > 5200 → use B → SL = 5224.75
    assert sl is not None and abs(sl - 5224.75) < 1e-6, sl

    # Reading C fallback: pivot 5212.5, entry 5200 → B would be 5199.75 (loss)
    # Use C → 5212.5 - 0.25 = 5212.25
    sl_c = compute_pivot_trail_sl("LONG", 5212.5, 5200.0, current_sl=5190.0)
    assert sl_c is not None and abs(sl_c - 5212.25) < 1e-6, sl_c

    # Min profit gate: pivot only 5pt above entry → no arm
    no_arm = compute_pivot_trail_sl("LONG", 5205.0, 5200.0, current_sl=5190.0)
    assert no_arm is None, no_arm

    print("test_unit_pivot_detection_and_sl: PASS")


# ---------------------------------------------------------------------------
# (a) Phantom-fill smoking gun — pivot off and pivot on must agree
# ---------------------------------------------------------------------------
def test_phantom_fill_pivot_disabled():
    master = _load_master()
    entry_ts = pd.Timestamp("2026-03-05 08:05:00", tz="US/Eastern")
    # pivot off
    out_off = simulate_trade_with_pivot_trail(
        master, entry_ts, "LONG", 6855.0, sl_distance=10.0, tp_distance=25.0,
        contract="ESH6", horizon_bars=30, pivot_active=False,
    )
    # pivot on (but: hour=8 ET → outside US session AND MFE never hits 12.5)
    out_on = simulate_trade_with_pivot_trail(
        master, entry_ts, "LONG", 6855.0, sl_distance=10.0, tp_distance=25.0,
        contract="ESH6", horizon_bars=30, pivot_active=True,
    )
    assert out_off["exit_reason"] == "horizon", out_off
    assert out_on["exit_reason"] == "horizon", out_on
    # Phase 1 reports exit_price = 6851.25 — exit at 30th forward bar's close.
    assert abs(out_off["exit_price"] - 6851.25) < 1e-6, out_off["exit_price"]
    assert abs(out_on["exit_price"] - 6851.25) < 1e-6, out_on["exit_price"]
    assert out_on["pivot_armed"] is False, out_on
    print(f"test_phantom_fill_pivot_disabled: PASS  exit={out_off['exit_price']:.2f}")


# ---------------------------------------------------------------------------
# (b) Pivot arms correctly — 2025-03-03 13:41 LONG ESH5 @ 5901
# ---------------------------------------------------------------------------
def test_pivot_arms_correctly():
    master = _load_master()
    entry_ts = pd.Timestamp("2025-03-03 13:41:00", tz="US/Eastern")
    # pivot on (hour=13 ET, in US session) — bot's recorded behaviour: take @ 5926
    out_on = simulate_trade_with_pivot_trail(
        master, entry_ts, "LONG", 5901.0, sl_distance=10.0, tp_distance=25.0,
        contract="ESH5", horizon_bars=30, pivot_active=True,
    )
    out_off = simulate_trade_with_pivot_trail(
        master, entry_ts, "LONG", 5901.0, sl_distance=10.0, tp_distance=25.0,
        contract="ESH5", horizon_bars=30, pivot_active=False,
    )
    print(f"  pivot_off: {out_off['exit_reason']} @ {out_off['exit_price']:.2f}  "
          f"pnl={out_off['net_pnl_after_haircut']:.2f}")
    print(f"  pivot_on : {out_on['exit_reason']} @ {out_on['exit_price']:.2f}  "
          f"pnl={out_on['net_pnl_after_haircut']:.2f}  "
          f"armed={out_on['pivot_armed']} sl_path_n={len(out_on['sl_path'])}")
    if out_on["sl_path"]:
        print(f"  first ratchet: {out_on['sl_path'][0]}")
    # Pivot-on must have armed at some point during the run-up to 5926 (MFE>=25)
    assert out_on["pivot_armed"] is True, "expected pivot to arm on a 25pt-up trade"
    # And must finish no worse than original SL (entry-10 = 5891)
    assert out_on["exit_price"] >= 5891.0 - 1e-6, out_on
    print("test_pivot_arms_correctly: PASS")


# ---------------------------------------------------------------------------
# (c) Reversal lock — synthetic bars: rally to +20pt then collapse below
#     original SL. Pivot ON should keep more of the gain than pivot OFF.
# ---------------------------------------------------------------------------
def test_pivot_locks_profit_on_reverse():
    # Build a synthetic 1-min bar series that:
    #  - starts at entry 5000
    #  - rallies to 5020 over 10 bars (creates a confirmed pivot-high somewhere)
    #  - prints a swing high then sells off below 4990 (original SL)
    # If pivot armed, trailed SL should be near 5012.25 / 5024.75 etc — well
    # above the original 4990.
    base_ts = pd.Timestamp("2025-03-04 11:00:00", tz="US/Eastern")  # hour 11 = US session
    rows = []
    # 10 bars climbing 5000 → 5020 (steady push)
    for i in range(10):
        o = 5000.0 + i * 2.0
        h = o + 1.0
        l = o - 0.25
        c = o + 1.0
        rows.append((base_ts + pd.Timedelta(minutes=i), o, h, l, c, 100, "ESH5"))
    # 1 bar that creates a clear swing-high at 5021 (pivot center)
    rows.append((base_ts + pd.Timedelta(minutes=10), 5020.0, 5022.0, 5019.0, 5020.0, 100, "ESH5"))
    # 4 bars descending so the bar at index 10 IS the middle of a 5-bar window
    # that will be confirmed when bar 12 prints (window = bars 8..12, middle=10)
    for i in range(4):
        o = 5020.0 - i * 2.0
        h = o + 0.5
        l = o - 1.0
        c = o - 1.0
        rows.append((base_ts + pd.Timedelta(minutes=11 + i), o, h, l, c, 100, "ESH5"))
    # Then collapse to below 4990 (original SL = entry - 10 = 4990)
    for i in range(10):
        o = 5012.0 - i * 3.0
        h = o + 0.5
        l = o - 0.5
        c = o - 0.5
        rows.append((base_ts + pd.Timedelta(minutes=15 + i), o, h, l, c, 100, "ESH5"))
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume", "symbol"]).set_index("timestamp")

    # Use the simulate_with_pivot_trail directly with these bars
    out_on = simulate_with_pivot_trail(
        df, side="LONG", entry_price=5000.0,
        initial_sl=4990.0, initial_tp=5025.0,
        pivot_active=True,
    )
    out_off = simulate_with_pivot_trail(
        df, side="LONG", entry_price=5000.0,
        initial_sl=4990.0, initial_tp=5025.0,
        pivot_active=False,
    )
    print(f"  pivot_off: {out_off.exit_reason} @ {out_off.exit_price:.2f}  pnl_pts={out_off.pnl_points:.2f}")
    print(f"  pivot_on : {out_on.exit_reason} @ {out_on.exit_price:.2f}  pnl_pts={out_on.pnl_points:.2f}  "
          f"armed={out_on.pivot_armed} sl_path={out_on.sl_path[:3]}")
    # Pivot OFF: original SL 4990 fires after the collapse → loss of -10pt
    assert out_off.exit_reason == "stop", out_off
    assert abs(out_off.pnl_points - (-10.0)) < 1e-6, out_off
    # Pivot ON: must arm and lock a positive (or smaller-loss) outcome
    assert out_on.pivot_armed is True, out_on
    assert out_on.pnl_points > out_off.pnl_points, (out_on, out_off)
    print("test_pivot_locks_profit_on_reverse: PASS")


if __name__ == "__main__":
    test_unit_pivot_detection_and_sl()
    test_phantom_fill_pivot_disabled()
    test_pivot_arms_correctly()
    test_pivot_locks_profit_on_reverse()
    print("\nALL TESTS PASS")
