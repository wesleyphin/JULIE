#!/usr/bin/env python3
"""Append NQ/MNQ extension parquet into data/mnq_master_continuous.parquet."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path("/Users/wes/Downloads/JULIE001")
MASTER = ROOT / "data" / "mnq_master_continuous.parquet"
EXT = ROOT / "artifacts" / "nq_extension_apr21_25.parquet"


def main():
    print(f"master: {MASTER}")
    print(f"ext:    {EXT}")

    master = pd.read_parquet(MASTER)
    ext = pd.read_parquet(EXT)
    print(f"\nbefore: master rows={len(master):,}  range {master.index.min()} -> {master.index.max()}")
    print(f"        ext rows={len(ext):,}  range {ext.index.min()} -> {ext.index.max()}")
    print(f"        master tz={master.index.tz}  ext tz={ext.index.tz}")
    print(f"        master dtypes: {dict(master.dtypes)}")
    print(f"        ext dtypes:    {dict(ext.dtypes)}")

    # Normalize tz/precision to match master (tz-aware NY; ns precision)
    ext = ext.tz_convert(master.index.tz)
    # Cast to ns precision to match master
    ext.index = pd.DatetimeIndex(ext.index.astype(str(master.index.dtype)))
    ext.index.name = master.index.name

    # Relabel symbol to a uniform "NQ_ext" for provenance; the trainer
    # doesn't care about symbol, only on-date OHLCV
    ext = ext.assign(symbol=ext["symbol"].astype(str))

    # Match master symbol dtype
    for col in master.columns:
        if str(master[col].dtype) != str(ext[col].dtype):
            print(f"  cast {col}: {ext[col].dtype} -> {master[col].dtype}")
            ext[col] = ext[col].astype(master[col].dtype)

    combined = pd.concat([master, ext])
    combined = combined.reset_index()
    idx_col = combined.columns[0]
    before = len(combined)
    combined = combined.drop_duplicates(subset=[idx_col, "symbol"], keep="last")
    n_dupes = before - len(combined)
    combined = combined.set_index(idx_col).sort_index()
    print(f"\ndedup: dropped {n_dupes} rows duplicated on (timestamp, symbol)")

    # More important: for our feature joiner we want a *single* price
    # series per timestamp.  If both MNQ.c.0 and NQM6 coexist for some
    # timestamps (they won't — MNQ ends Apr 21, NQ starts Apr 21), prefer
    # NQM6 (newer).  Extra safety: drop on timestamp only, keep='last'.
    combined2 = combined.reset_index().drop_duplicates(subset=[idx_col], keep="last").set_index(idx_col).sort_index()
    extra_dupes = len(combined) - len(combined2)
    if extra_dupes:
        print(f"extra timestamp-only dedup removed {extra_dupes} rows")
    combined = combined2

    print(f"\nafter:  rows={len(combined):,}  range {combined.index.min()} -> {combined.index.max()}")

    # Backup + write
    bak = MASTER.with_suffix(".parquet.bak_preKalshiMLv4")
    if not bak.exists():
        print(f"backing up master -> {bak}")
        import shutil
        shutil.copy2(os.path.realpath(MASTER), bak)
    tmp = Path(os.path.realpath(MASTER)).with_suffix(".parquet.tmp")
    combined.to_parquet(tmp, engine="pyarrow", compression="snappy")
    os.replace(tmp, Path(os.path.realpath(MASTER)))
    print(f"wrote: {os.path.realpath(MASTER)}")

    rb = pd.read_parquet(MASTER)
    print(f"\nread-back: rows={len(rb):,}  range {rb.index.min()} -> {rb.index.max()}")


if __name__ == "__main__":
    sys.exit(main() or 0)
