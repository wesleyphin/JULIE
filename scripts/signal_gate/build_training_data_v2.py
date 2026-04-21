#!/usr/bin/env python3
"""v2 training-data builder — uses REAL OHLCV from the ES master parquet
instead of synthetic h=l=c from replay logs.

Adds 2026 trades (April 2026 replays) to the 2025 iter-11 set.

Output: artifacts/signal_gate_2025/training_rows_v2.parquet
"""
from __future__ import annotations

import json
import sys
from bisect import bisect_right
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import numpy as np

ROOT = Path("/Users/wes/Downloads/JULIE001")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_de3_chosen_shape_dataset import _compute_feature_frame, ENTRY_SHAPE_COLUMNS  # noqa: E402
from reconstruct_regime import reconstruct_from_log  # noqa: E402

NY = ZoneInfo("America/New_York")
PARQUET_PATH = Path("/Users/wes/Downloads/es_master_outrights-2.parquet")

# Front-month roll dates (from analysis). Each date is the first trading day
# of the new contract.
ROLL_MAP = [
    (pd.Timestamp("2025-01-01", tz="America/New_York"), "ESH5"),
    (pd.Timestamp("2025-03-17", tz="America/New_York"), "ESM5"),
    (pd.Timestamp("2025-06-16", tz="America/New_York"), "ESU5"),
    (pd.Timestamp("2025-09-15", tz="America/New_York"), "ESZ5"),
    (pd.Timestamp("2025-12-15", tz="America/New_York"), "ESH6"),
    (pd.Timestamp("2026-03-16", tz="America/New_York"), "ESM6"),
]

# iter-11-consistent 2025 sources
SOURCES_2025 = [
    "2025_03_ny_iter11_deadtape",
    "2025_05_ny_iter11_deadtape",
    "2025_06_ny_iter11_deadtape",
    "outrageous_feb",
    "outrageous_jul",
    "outrageous_aug",
    "outrageous_oct",
    "outrageous_dec",
    "outrageous_apr",
]

# April 2026 replays (NEW — included in training this time)
SOURCES_2026 = [
    # Paths relative to backtest_reports/
    "replay_apr2026_p1",
    "replay_apr20/baseline_warm",
]

REPORT_ROOT = ROOT / "backtest_reports" / "full_live_replay"


def parse_ts(s):
    return datetime.fromisoformat(s.strip())


def active_symbol(dt) -> str:
    """Return the front-month symbol for a given datetime."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=NY)
    best = ROLL_MAP[0][1]
    for roll_date, sym in ROLL_MAP:
        if dt >= roll_date:
            best = sym
        else:
            break
    return best


def load_parquet_bars_for_window(
    master_df: pd.DataFrame,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    symbol: str,
) -> pd.DataFrame:
    """Slice master parquet to [start, end] for a given symbol."""
    sub = master_df[(master_df["symbol"] == symbol)]
    sub = sub[(sub.index >= start_dt) & (sub.index <= end_dt)]
    return sub[["open", "high", "low", "close", "volume"]]


def compute_trade_features(
    trade_et: datetime,
    master_df: pd.DataFrame,
    cache: dict,
) -> dict:
    """Given a trade entry time, pull 150 bars of preceding real OHLCV from
    the parquet and compute the feature row at that timestamp."""
    symbol = active_symbol(trade_et)
    # 150 bars window ending at or before trade time
    end_ts = pd.Timestamp(trade_et).tz_convert("UTC") if trade_et.tzinfo else pd.Timestamp(trade_et, tz="UTC")
    start_ts = end_ts - pd.Timedelta(minutes=180)  # 3h buffer for rolling windows

    cache_key = (symbol, end_ts.floor("min"))
    if cache_key in cache:
        return cache[cache_key]

    try:
        bars = master_df.loc[(master_df.index >= start_ts) & (master_df.index <= end_ts) &
                             (master_df["symbol"] == symbol),
                             ["open", "high", "low", "close", "volume"]]
    except Exception:
        cache[cache_key] = {}
        return {}
    if len(bars) < 45:
        cache[cache_key] = {}
        return {}
    bars = bars.tz_convert("America/New_York")
    feats = _compute_feature_frame(bars)
    if feats.empty or feats.iloc[-1].isna().all():
        cache[cache_key] = {}
        return {}
    # Match training lookup: features at the bar BEFORE the trade timestamp.
    # Here last row is the bar AT the trade; features at that row are already
    # shifted-1 (use data through prev bar), so iloc[-1] is correct for the
    # trade's info set.
    row = {c: float(feats.iloc[-1].get(c, float("nan"))) for c in ENTRY_SHAPE_COLUMNS}
    cache[cache_key] = row
    return row


def session_bucket(h: int) -> str:
    if 18 <= h or h < 3:  return "ASIA"
    if 3 <= h < 7:        return "LONDON"
    if 7 <= h < 9:        return "NY_PRE"
    if 9 <= h < 16:       return "NY"
    return "POST"


def process_folder(folder: Path, source_tag: str, master_df, cache: dict, rows: list):
    ct = folder / "closed_trades.json"
    log = folder / "topstep_live_bot.log"
    if not ct.exists() or not log.exists():
        print(f"  [skip] {folder.name}: missing artifacts")
        return
    trades = json.loads(ct.read_text(encoding="utf-8"))
    regime_events = reconstruct_from_log(log)
    regime_keys = [e[0] for e in regime_events]
    print(f"  [load] {folder.name}: {len(trades)} trades")
    n_added = 0
    for t in trades:
        try:
            et = parse_ts(t["entry_time"]).astimezone(NY)
        except Exception:
            continue
        feats = compute_trade_features(et, master_df, cache)
        if not feats:
            continue
        ri = bisect_right(regime_keys, et) - 1
        regime = regime_events[ri][1] if ri >= 0 else "warmup"
        pnl = float(t.get("pnl_dollars", 0.0) or 0.0)
        row = {
            "source_tag": source_tag,
            "src_folder": folder.name,
            "day": et.date().isoformat(),
            "entry_time": et.isoformat(),
            "et_hour": et.hour,
            "session": session_bucket(et.hour),
            "side": str(t.get("side", "")).upper(),
            "size": int(t.get("size", 1) or 1),
            "entry_price": float(t.get("entry_price", 0.0) or 0.0),
            "pnl_dollars": pnl,
            "win": 1 if pnl > 0 else 0,
            "big_loss": 1 if pnl <= -100 else 0,
            "regime": regime,
            "sub_strategy": str(t.get("sub_strategy", "")),
            "strategy": str(t.get("strategy", "")),
        }
        row.update(feats)
        rows.append(row)
        n_added += 1
    print(f"    +{n_added} feature rows")


def main():
    print(f"[parquet] loading {PARQUET_PATH}")
    master = pd.read_parquet(PARQUET_PATH)
    master = master[master.index >= "2025-01-01"].copy()  # just 2025+ to save memory
    print(f"  2025+ rows: {len(master):,}")

    cache: dict = {}
    rows: list = []

    print("\n[2025] processing iter-11 folders with parquet bars...")
    for src in SOURCES_2025:
        process_folder(REPORT_ROOT / src, "2025", master, cache, rows)

    print("\n[2026] processing April 2026 replays...")
    for src in SOURCES_2026:
        folder_root = ROOT / "backtest_reports" / src
        if not folder_root.exists():
            continue
        loops = sorted(folder_root.glob("live_loop_MES_*"))
        if loops:
            process_folder(loops[-1], "2026", master, cache, rows)

    df = pd.DataFrame(rows)
    print(f"\n[done] total rows: {len(df)}")
    if len(df) > 0:
        print(f"  2025 rows: {(df['source_tag']=='2025').sum()}")
        print(f"  2026 rows: {(df['source_tag']=='2026').sum()}")
        print(f"  overall win rate: {df['win'].mean():.1%}")
        print(f"  2025 win rate: {df.loc[df['source_tag']=='2025','win'].mean():.1%}")
        print(f"  2026 win rate: {df.loc[df['source_tag']=='2026','win'].mean():.1%}")
        print(f"  big_loss rate (overall): {df['big_loss'].mean():.1%}")

    out = ROOT / "artifacts" / "signal_gate_2025" / "training_rows_v2.parquet"
    df.to_parquet(out, index=False)
    print(f"\n[write] {out}  ({out.stat().st_size // 1024} KB)")

    # Sanity: show how many features have meaningful (non-zero) variance now
    for col in ENTRY_SHAPE_COLUMNS:
        vals = df[col].dropna()
        print(f"  {col:<32}  n={len(vals):>4}  std={vals.std():>8.3f}  range=[{vals.min():>7.2f}, {vals.max():>7.2f}]")


if __name__ == "__main__":
    main()
