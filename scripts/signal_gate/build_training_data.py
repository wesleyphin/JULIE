#!/usr/bin/env python3
"""Build training dataset for the 2025 signal-gate classifier.

For each trade in the 2025 iter-11 labeled set:
  1. Load the day's bars from the replay log
  2. Compute entry-time features via _compute_feature_frame (18 cols)
  3. Record features + labels (pnl, win)
  4. Also record regime at entry + session (ET hour bucket) as context

Output: artifacts/signal_gate_2025/training_rows.parquet
"""
from __future__ import annotations

import json
import re
import sys
from bisect import bisect_right
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import numpy as np

ROOT = Path("/Users/wes/Downloads/JULIE001")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_de3_chosen_shape_dataset import _compute_feature_frame, ENTRY_SHAPE_COLUMNS  # noqa: E402
from reconstruct_regime import reconstruct_from_log  # noqa: E402 (for folders without regime logs)

NY = ZoneInfo("America/New_York")
REPORT_ROOT = ROOT / "backtest_reports" / "full_live_replay"

# iter-11-consistent sources (same bot config across all these folders).
SOURCES = [
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

RGX_BAR = re.compile(
    r"Bar: (?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) ET \| Price: (?P<price>[\d.]+)"
)


def parse_ts(s):
    return datetime.fromisoformat(s.strip())


def bars_df_from_log(log_path: Path) -> pd.DataFrame:
    """Return a DataFrame indexed by ET-aware datetime with (close) — that's
    all _compute_feature_frame really needs if we synthesize OHLC from close.
    But the function DOES read open/high/low; we fabricate those as close
    (which makes body/wick-based features degenerate but 30-bar trend/range
    features stay meaningful)."""
    rows = []
    with log_path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = RGX_BAR.search(line)
            if not m:
                continue
            ts = datetime.strptime(m.group("ts"), "%Y-%m-%d %H:%M:%S").replace(tzinfo=NY)
            price = float(m.group("price"))
            rows.append((ts, price))
    rows.sort()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame({
        "open":  [r[1] for r in rows],
        "high":  [r[1] for r in rows],  # degenerate; features touching wicks will be flat
        "low":   [r[1] for r in rows],
        "close": [r[1] for r in rows],
        "volume": [np.nan] * len(rows),
    }, index=pd.DatetimeIndex([r[0] for r in rows], name="timestamp_et"))
    return df


def session_bucket(et_hour: int) -> str:
    if 18 <= et_hour or et_hour < 3:  return "ASIA"
    if 3 <= et_hour < 7:              return "LONDON"
    if 7 <= et_hour < 9:              return "NY_PRE"
    if 9 <= et_hour < 16:             return "NY"
    return "POST"


def build():
    rows = []
    for src in SOURCES:
        folder = REPORT_ROOT / src
        ct_path = folder / "closed_trades.json"
        log_path = folder / "topstep_live_bot.log"
        if not ct_path.exists() or not log_path.exists():
            print(f"[skip] {src}")
            continue
        trades = json.loads(ct_path.read_text(encoding="utf-8"))
        bars = bars_df_from_log(log_path)
        if bars.empty:
            print(f"[skip] {src}: no bars")
            continue
        feats = _compute_feature_frame(bars)
        # Regime at each ts (reconstruct from log if no transitions logged)
        regime_events = reconstruct_from_log(log_path)  # always offline-reconstructed
        regime_keys = [e[0] for e in regime_events]

        print(f"[load] {src}: bars={len(bars)}  feats={len(feats.dropna())}  trades={len(trades)}")
        for t in trades:
            try:
                et = parse_ts(t["entry_time"]).astimezone(NY)
            except Exception:
                continue
            # Feature lookup: we want the features AT entry (which _compute_feature_frame
            # shifts by 1 bar to be causal — features at bar t use data through bar t-1).
            # Find the bar index whose timestamp <= et.
            try:
                idx = feats.index.searchsorted(et)
            except Exception:
                continue
            if idx <= 0 or idx > len(feats):
                continue
            feat_row = feats.iloc[idx - 1]
            if feat_row.isna().all():
                continue
            # Regime at entry
            ri = bisect_right(regime_keys, et) - 1
            regime = regime_events[ri][1] if ri >= 0 else "warmup"
            pnl = float(t.get("pnl_dollars", 0.0) or 0.0)
            size = int(t.get("size", 1) or 1)
            per_contract_pnl = pnl / size if size > 0 else pnl
            side = str(t.get("side", "")).upper()
            row = {
                "src_folder": src,
                "day": et.date().isoformat(),
                "entry_time": et.isoformat(),
                "et_hour": et.hour,
                "session": session_bucket(et.hour),
                "side": side,
                "size": size,
                "entry_price": float(t.get("entry_price", 0.0) or 0.0),
                "pnl_dollars": pnl,
                "per_contract_pnl": per_contract_pnl,
                "win": 1 if pnl > 0 else 0,
                "big_loss": 1 if pnl <= -100 else 0,
                "regime": regime,
                "sub_strategy": str(t.get("sub_strategy", "")),
                "strategy": str(t.get("strategy", "")),
            }
            # Copy feature columns
            for col in ENTRY_SHAPE_COLUMNS:
                row[col] = float(feat_row.get(col, float("nan")))
            rows.append(row)
    df = pd.DataFrame(rows)
    print(f"\n[done] {len(df)} rows")
    return df


if __name__ == "__main__":
    df = build()
    out_dir = ROOT / "artifacts" / "signal_gate_2025"
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / "training_rows.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"[write] {parquet_path}  ({parquet_path.stat().st_size // 1024} KB)")
    # Quick sanity stats
    print()
    print("Labels:")
    print(f"  total trades: {len(df)}")
    print(f"  wins: {df['win'].sum()}  ({df['win'].mean():.1%})")
    print(f"  big losses (<=-$100): {df['big_loss'].sum()}  ({df['big_loss'].mean():.1%})")
    print()
    print("By regime:")
    print(df.groupby("regime")[["pnl_dollars", "win"]].agg({"pnl_dollars": ["sum", "count"], "win": "mean"}))
    print()
    print("By session:")
    print(df.groupby("session")[["pnl_dollars", "win"]].agg({"pnl_dollars": ["sum", "count"], "win": "mean"}))
