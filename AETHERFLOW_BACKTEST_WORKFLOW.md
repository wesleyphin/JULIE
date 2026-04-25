# AetherFlow Backtest Workflow

This repo runs AetherFlow backtests in two steps:

1. Extend the corrected canonical manifold base parquet with a small overlap window.
2. Run the direct AetherFlow backtest against that extended base cache.

The intended commands are:

```powershell
.\.venv\Scripts\python.exe tools\extend_manifold_base_cache.py `
  --existing-base artifacts\aetherflow_corrected_full_2011_2026\manifold_base_outrights_2011_2026.parquet `
  --end 2026-04-21 `
  --output artifacts\aetherflow_backtest_cache\manifold_base_outrights_corrected_full_extended_20260421.parquet `
  --overwrite
```

```powershell
.\.venv\Scripts\python.exe tools\backtest_aetherflow_direct.py `
  --base-features artifacts\aetherflow_backtest_cache\manifold_base_outrights_corrected_full_extended_20260421.parquet `
  --model-file artifacts\aetherflow_live_erasia_londonnative015_tb_er_af_candidate\model.pkl `
  --thresholds-file artifacts\aetherflow_live_erasia_londonnative015_tb_er_af_candidate\thresholds.json `
  --start 2026-04-21 `
  --end 2026-04-21
```

Notes:

- `tools/extend_manifold_base_cache.py` is the cache-refresh step for AetherFlow backtests. Use the corrected full base as the anchor cache so overlapping historical windows stay identical to the validated research lineage, and only rebuild the trailing overlap window plus the missing tail.
- `tools/backtest_aetherflow_direct.py` is the execution step. It consumes the cached manifold base features directly, derives AetherFlow signals from that cache, and writes the normal backtest report plus converted CSV.
- When the selected model and thresholds files match `CONFIG["AETHERFLOW_STRATEGY"]` and no manual threshold or filter overrides are passed, the direct backtest also applies the live `AetherFlowStrategy` family and slice policy gates instead of only using a flat threshold cut.
- Use the current live AetherFlow artifact paths from `CONFIG["AETHERFLOW_STRATEGY"]` when you want the backtest to reflect the deployed AetherFlow bundle.
- Treat `artifacts/aetherflow_corrected_full_2011_2026/manifold_base_outrights_2011_2026.parquet` as the only canonical full manifold base. Refresh that canonical file via `tools/start_aetherflow_canonical_base_rebuild.ps1`, and keep temporary backtest extensions in `artifacts/aetherflow_backtest_cache/` instead of repointing runtime defaults.
