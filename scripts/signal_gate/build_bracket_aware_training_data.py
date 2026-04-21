"""Build a bracket-aware training set: simulate (LONG/SHORT, TP, SL) trade
outcomes from the parquet bar history, recording chart features at entry
+ the bracket geometry as inputs.

Idea: instead of a single big_loss target tied to one strategy's brackets,
let the model learn "given chart context AND tp_pts AND sl_pts, what's
P(SL hits before TP)?" Then at inference time, pass the strategy's actual
TP/SL — works for DE3 (25/10), AetherFlow (3/5), RegimeAdaptive (whatever).

Output: artifacts/signal_gate_2025/bracket_training.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

ROOT = Path("/Users/wes/Downloads/JULIE001")
PARQUET = Path("/Users/wes/Downloads/es_master_outrights-2.parquet")
sys.path.insert(0, str(ROOT / "tools"))
from build_de3_chosen_shape_dataset import _compute_feature_frame, ENTRY_SHAPE_COLUMNS  # noqa: E402

# ROLL MAP — use front-month per period
ROLL_MAP = [
    (pd.Timestamp("2025-01-01", tz="America/New_York"), "ESH5"),
    (pd.Timestamp("2025-03-17", tz="America/New_York"), "ESM5"),
    (pd.Timestamp("2025-06-16", tz="America/New_York"), "ESU5"),
    (pd.Timestamp("2025-09-15", tz="America/New_York"), "ESZ5"),
    (pd.Timestamp("2025-12-15", tz="America/New_York"), "ESH6"),
    (pd.Timestamp("2026-03-16", tz="America/New_York"), "ESM6"),
]

# Bracket configs to simulate. Each strategy's typical TP/SL.
BRACKETS = [
    # (side, tp_pts, sl_pts, name) — covers DE3, AF, regime-adaptive
    ("LONG",  25.0, 10.0, "de3_long_wide"),
    ("SHORT", 25.0, 10.0, "de3_short_wide"),
    ("LONG",  12.5, 10.0, "de3_long_tight"),
    ("SHORT", 12.5, 10.0, "de3_short_tight"),
    ("LONG",   3.0,  5.0, "af_long"),
    ("SHORT",  3.0,  5.0, "af_short"),
    ("LONG",   8.0,  5.0, "ra_long"),     # RegimeAdaptive-ish
    ("SHORT",  8.0,  5.0, "ra_short"),
]

MAX_HOLD_BARS = 60   # 1 hour
SAMPLE_EVERY_N_BARS = 30  # sample every 30 minutes (gives ~50 samples per session)


def active_symbol(ts) -> str:
    if ts.tzinfo is None:
        ts = ts.tz_localize("America/New_York")
    best = ROLL_MAP[0][1]
    for roll_date, sym in ROLL_MAP:
        if ts >= roll_date:
            best = sym
        else:
            break
    return best


def simulate_outcome(
    bars_h, bars_l, bars_c,
    entry_idx: int,
    side: str, entry_price: float,
    tp_pts: float, sl_pts: float,
    max_hold: int,
):
    """Walk forward from entry_idx+1; return ('WIN'|'LOSS'|'OPEN', bars_held)."""
    if side == "LONG":
        tp_level = entry_price + tp_pts
        sl_level = entry_price - sl_pts
    else:
        tp_level = entry_price - tp_pts
        sl_level = entry_price + sl_pts

    end_idx = min(entry_idx + 1 + max_hold, len(bars_c))
    for i in range(entry_idx + 1, end_idx):
        h, l = bars_h[i], bars_l[i]
        if side == "LONG":
            # Worst-case: if both H>=tp AND L<=sl in same bar, assume SL hit first
            # (conservative). Real markets vary tick-by-tick; this matches DE3 backtest behavior.
            if l <= sl_level:
                return "LOSS", i - entry_idx
            if h >= tp_level:
                return "WIN", i - entry_idx
        else:
            if h >= sl_level:
                return "LOSS", i - entry_idx
            if l <= tp_level:
                return "WIN", i - entry_idx
    return "OPEN", end_idx - entry_idx - 1


def main():
    print(f"[load] {PARQUET}")
    master = pd.read_parquet(PARQUET)
    master = master[master.index >= "2025-01-01"].copy()
    print(f"  2025+ rows: {len(master):,}")

    # Pre-segment by symbol for speed
    rows = []
    total_sims = 0
    for sym in sorted(master["symbol"].unique()):
        sub = master[master["symbol"] == sym].copy()
        sub = sub.sort_index()
        # Only consider this symbol over its active window (per ROLL_MAP)
        # Active = the period when this symbol IS the front month
        active_starts = [(d, s) for d, s in ROLL_MAP if s == sym]
        if not active_starts:
            continue
        active_start = active_starts[0][0]
        # Find next roll
        next_roll = None
        for i, (d, s) in enumerate(ROLL_MAP):
            if s == sym and i + 1 < len(ROLL_MAP):
                next_roll = ROLL_MAP[i + 1][0]
                break
        sub = sub[sub.index >= active_start]
        if next_roll is not None:
            sub = sub[sub.index < next_roll]
        if len(sub) < 100:
            continue
        print(f"[sim] symbol={sym} bars={len(sub):,} window={sub.index.min()}..{sub.index.max()}")

        # Compute features once for the whole symbol's bars
        feats = _compute_feature_frame(sub[["open","high","low","close","volume"]])
        h = sub["high"].values
        l = sub["low"].values
        c = sub["close"].values

        # Sample entry indices
        for entry_idx in range(50, len(sub) - MAX_HOLD_BARS, SAMPLE_EVERY_N_BARS):
            entry_price = float(c[entry_idx])
            ts = sub.index[entry_idx]
            # Skip if features are NaN
            f_row = feats.iloc[entry_idx]
            if f_row.isna().any():
                continue
            for side, tp_pts, sl_pts, name in BRACKETS:
                outcome, held = simulate_outcome(h, l, c, entry_idx, side, entry_price, tp_pts, sl_pts, MAX_HOLD_BARS)
                if outcome == "OPEN":
                    continue  # skip indeterminate outcomes
                row = {
                    "symbol": sym,
                    "ts": ts,
                    "et_hour": int(ts.tz_convert("America/New_York").hour),
                    "side": side,
                    "tp_pts": tp_pts,
                    "sl_pts": sl_pts,
                    "rr_ratio": tp_pts / sl_pts,
                    "bracket_name": name,
                    "entry_price": entry_price,
                    "outcome": outcome,
                    "is_loss": 1 if outcome == "LOSS" else 0,
                    "bars_held": held,
                }
                for fc in ENTRY_SHAPE_COLUMNS:
                    row[fc] = float(f_row.get(fc, float("nan")))
                rows.append(row)
                total_sims += 1
        print(f"  cumulative simulations: {total_sims:,}")

    df = pd.DataFrame(rows)
    print(f"\n[done] total simulated rows: {len(df):,}")
    if len(df) == 0:
        return
    print(f"  outcomes: {df['outcome'].value_counts().to_dict()}")
    print(f"  by bracket:")
    g = df.groupby("bracket_name").agg({"is_loss": ["count", "mean"]})
    print(g)

    out = ROOT / "artifacts" / "signal_gate_2025" / "bracket_training.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"\n[write] {out}  ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
