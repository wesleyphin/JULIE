#!/usr/bin/env python3
"""Append the Kalshi-ML extension parquet into es_master_outrights.parquet.

Preserves master schema (tz=US/Eastern, nanosecond precision, column order/dtypes).
Dedupes on timestamp; later rows win (so re-runs are idempotent).
Writes to the symlink target so the symlink stays intact.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path("/Users/wes/Downloads/JULIE001")
SYMLINK = ROOT / "es_master_outrights.parquet"
REAL = Path(os.path.realpath(SYMLINK))

EXT = ROOT / "artifacts/es_extension_apr21_25.parquet"


def main():
    print(f"symlink: {SYMLINK}")
    print(f"real path: {REAL}")
    print(f"extension: {EXT}")

    master = pd.read_parquet(REAL)
    ext = pd.read_parquet(EXT)
    print(f"\nbefore:")
    print(f"  master rows: {len(master):,}")
    print(f"  master range: {master.index.min()} -> {master.index.max()}")
    print(f"  master tz: {master.index.tz}")
    print(f"  ext rows: {len(ext):,}")
    print(f"  ext range: {ext.index.min()} -> {ext.index.max()}")
    print(f"  ext tz: {ext.index.tz}")

    # Normalize ext index to master's tz + precision
    ext = ext.tz_convert(master.index.tz)
    # Cast to datetime64[ns, tz] to match master
    ext.index = pd.DatetimeIndex(ext.index.astype("datetime64[ns, US/Eastern]"))
    ext.index.name = master.index.name

    # Align columns
    assert list(ext.columns) == list(master.columns), (
        f"column mismatch: master={list(master.columns)}  ext={list(ext.columns)}"
    )
    # Match dtypes
    for col in master.columns:
        if str(master[col].dtype) != str(ext[col].dtype):
            print(f"  cast {col}: {ext[col].dtype} -> {master[col].dtype}")
            ext[col] = ext[col].astype(master[col].dtype)

    # Combine, dedupe on (timestamp, symbol) because master carries multiple
    # contracts at the same minute during rollover periods.  Keep ext where it
    # overlaps (later rows win). Preserve DatetimeIndex.
    combined = pd.concat([master, ext])
    combined = combined.reset_index()
    idx_col = combined.columns[0]  # 'timestamp'
    before = len(combined)
    combined = combined.drop_duplicates(subset=[idx_col, "symbol"], keep="last")
    n_dupes = before - len(combined)
    combined = combined.set_index(idx_col).sort_index()
    print(f"\ndedup: dropped {n_dupes} rows duplicated on (timestamp, symbol)")

    print(f"\nafter:")
    print(f"  combined rows: {len(combined):,}  (net +{len(combined) - len(master):,})")
    print(f"  combined range: {combined.index.min()} -> {combined.index.max()}")
    print(f"  combined tz: {combined.index.tz}")
    print(f"  combined dtypes: {dict(combined.dtypes)}")

    # Sanity checks
    assert combined.index.is_monotonic_increasing, "index not sorted"
    assert combined.index.tz is not None, "index tz lost"
    assert len(combined) > len(master), "no new rows added"

    # Write to realpath (preserves symlink)
    tmp = REAL.with_suffix(".parquet.tmp")
    combined.to_parquet(tmp, engine="pyarrow", compression="snappy")
    print(f"\nwrote tmp: {tmp}  ({tmp.stat().st_size:,} bytes)")
    # Atomic swap
    os.replace(tmp, REAL)
    print(f"swapped into place: {REAL}")

    # Verify read-back
    rb = pd.read_parquet(REAL)
    print(f"\nread-back verify:")
    print(f"  rows: {len(rb):,}")
    print(f"  range: {rb.index.min()} -> {rb.index.max()}")
    tail_after = rb[rb.index >= pd.Timestamp("2026-04-20 12:00", tz=rb.index.tz)]
    print(f"  post-2026-04-20 12:00 rows: {len(tail_after):,}")


if __name__ == "__main__":
    sys.exit(main() or 0)
