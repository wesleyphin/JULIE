"""Build v11 corpus with full 30-bar OHLC trajectories per row.

This is the foundation of the §8.30 early-exit hyperparameter sweep. By caching
the full bar_path per candidate ONCE, we can replay 1000+ early-exit configs
post-hoc in microseconds per config (no I/O, no contract pinning).

Output: artifacts/v11_corpus_with_bar_paths.parquet
Schema: original v11 columns + bar_path (list[dict]) + walk_contract.

Each bar in bar_path: {ts, open, high, low, close, volume}.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulator_trade_through import pin_contract, get_walk_forward_bars

CORPUS_IN = ROOT / "artifacts/v11_training_corpus_with_mfe.parquet"
CORPUS_OUT = ROOT / "artifacts/v11_corpus_with_bar_paths.parquet"
BAR_PARQUET = ROOT / "es_master_outrights.parquet"

HORIZON = 30


def _load_bars() -> pd.DataFrame:
    df = pd.read_parquet(BAR_PARQUET)
    if df.index.tz is None:
        df.index = df.index.tz_localize("US/Eastern")
    return df


def _bar_path_for_row(bars_df: pd.DataFrame, ts: pd.Timestamp,
                       entry_price: float, contract_hint: str | None) -> tuple[list[dict], str]:
    """Pin contract and return [{ts, open, high, low, close, volume}, ...]."""
    contract = contract_hint or pin_contract(bars_df, ts, entry_price)
    fwd = get_walk_forward_bars(bars_df, ts, contract, horizon_bars=HORIZON)
    if fwd is None or len(fwd) == 0:
        return [], contract
    out: list[dict] = []
    for tsi, row in fwd.iterrows():
        out.append({
            "ts": pd.Timestamp(tsi).isoformat(),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row.get("volume", 0) or 0),
        })
    return out, str(contract)


def main() -> None:
    print(f"[bar_paths] loading corpus: {CORPUS_IN}")
    corpus = pd.read_parquet(CORPUS_IN)
    print(f"[bar_paths] corpus rows: {len(corpus)}")

    print(f"[bar_paths] loading ES bars: {BAR_PARQUET}")
    t0 = time.time()
    bars = _load_bars()
    print(f"[bar_paths] bars loaded in {time.time()-t0:.1f}s ({len(bars):,} rows)")

    bar_paths: list[list[dict]] = []
    walk_contracts: list[str] = []
    n_empty = 0
    t1 = time.time()
    for i, row in enumerate(corpus.itertuples(index=False)):
        ts = pd.Timestamp(row.ts)
        if ts.tzinfo is None:
            ts = ts.tz_localize("US/Eastern")
        else:
            ts = ts.tz_convert("US/Eastern")
        entry_price = float(row.entry_price)
        contract_hint = getattr(row, "contract", None)
        path, contract = _bar_path_for_row(bars, ts, entry_price, contract_hint)
        if not path:
            n_empty += 1
        bar_paths.append(path)
        walk_contracts.append(contract)
        if (i + 1) % 200 == 0:
            elapsed = time.time() - t1
            rate = (i + 1) / max(elapsed, 1e-6)
            eta = (len(corpus) - i - 1) / max(rate, 1e-6)
            print(f"  ... {i+1}/{len(corpus)} rows ({rate:.1f} rows/sec, ETA {eta:.0f}s, empty={n_empty})")

    print(f"[bar_paths] all rows walked in {time.time()-t1:.1f}s ({n_empty} empty)")

    out = corpus.copy()
    # Convert bar_path lists to JSON-encoded strings for safe parquet round-trip.
    # (Native list-of-dict columns in pandas can be flaky; JSON strings are
    # robust and decode trivially in the replay.)
    import json
    out["bar_path_json"] = [json.dumps(p) for p in bar_paths]
    out["walk_contract"] = walk_contracts

    print(f"[bar_paths] writing: {CORPUS_OUT}")
    out.to_parquet(CORPUS_OUT, index=False)
    print(f"[bar_paths] done — {len(out)} rows, {n_empty} empty bar_paths")

    # Sanity check: smoking-gun trade
    sg_mask = (out["ts"] == pd.Timestamp("2026-03-05 08:06:00", tz="US/Eastern"))
    if sg_mask.any():
        sg = out[sg_mask].iloc[0]
        path = json.loads(sg["bar_path_json"])
        print(f"\n[sanity] smoking gun: ts={sg['ts']} contract={sg['walk_contract']} "
              f"entry={sg['entry_price']} side={sg['side']} bars={len(path)}")
        if path:
            max_high = max(b["high"] for b in path)
            min_low = min(b["low"] for b in path)
            print(f"   first_bar_ts={path[0]['ts']} max_high={max_high} min_low={min_low}")


if __name__ == "__main__":
    main()
