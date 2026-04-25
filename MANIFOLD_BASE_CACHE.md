# Manifold Base Cache

The live/research manifold base parquet is generated and intentionally not
committed. Build it locally before running workflows that need the full
AetherFlow manifold base cache.

## Full Build

```powershell
python tools/build_manifold_base_cache.py `
  --source es_master_outrights.parquet `
  --output artifacts/aetherflow_corrected_full_2011_2026/manifold_base_outrights_2011_2026.parquet
```

The builder also writes a sibling `.meta.json` file containing the exact engine
state needed for deterministic continuation.

## Extend Existing Cache

```powershell
python tools/extend_manifold_base_cache.py `
  --existing-base artifacts/aetherflow_corrected_full_2011_2026/manifold_base_outrights_2011_2026.parquet `
  --end 2026-04-24
```

Generated `manifold_base_*.parquet`, metadata, and feature parquet files are
ignored by Git so the code can be shared without uploading the computed cache.
