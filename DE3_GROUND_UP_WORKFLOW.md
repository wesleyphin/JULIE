# DE3 Ground-Up Workflow

Use [tools/run_de3_ground_up_workflow.py](/c:/Users/Timothy/OneDrive/Desktop/trading/topstep2/tools/run_de3_ground_up_workflow.py) for clean DE3 rebuilds off [es_master_outrights.parquet](/c:/Users/Timothy/OneDrive/Desktop/trading/topstep2/es_master_outrights.parquet).

This workflow is for the post-contamination world:
- build a fresh DE3 v2 member DB from outright-only data
- build a fresh DE3 v4 bundle from that DB
- export a fresh current pool through `2024-12-31`
- retrain DE3 entry-policy candidates from that current pool
- emit a validation command pack for `2024`, `2025`, and `2024-2025`

## Default split
- Train: `2011-01-01` to `2023-12-31`
- Tune: `2024-01-01` to `2024-12-31`
- OOS: `2025-01-01` to `2025-12-31`
- Future holdout: `2026-01-01+`

## Typical usage
Full clean prep:

```powershell
python tools\run_de3_ground_up_workflow.py --tag clean_outrights
```

Limit entry-policy candidates to a shortlist:

```powershell
python tools\run_de3_ground_up_workflow.py --tag clean_outrights --candidate-profiles current_pool_pf_coverage,current_pool_tail_guard_v2
```

Dry-run only:

```powershell
python tools\run_de3_ground_up_workflow.py --tag clean_outrights --dry-run
```

Resume from an existing artifact root after v2/v4 are already built:

```powershell
python tools\run_de3_ground_up_workflow.py --artifact-root artifacts\de3_ground_up_YYYYMMDD_HHMMSS_clean_outrights --skip-v2 --skip-v4
```

## Outputs
Each workflow run writes:
- `workflow_manifest.json`
- `workflow_logs/*.log`
- `dynamic_engine3_strategies_v2.outrights.json`
- `reports/base_bundle/*`
- `reports/de3_current_pool_2011_2024.csv`
- `reports/de3_current_pool_2011_2024_trade_attribution.csv`
- `entry_policy_candidates/candidate_summary.json`
- `recommended_validation_commands.json`
- `recommended_validation_commands.ps1`

## Validation loop
The workflow does not auto-run all candidate backtests by default. Instead it writes the command pack so you can choose how broad to go.

Recommended order:
1. Run base rebuilt bundle on `2024`, `2025`, and `2024-2025`.
2. Run only the strongest entry-policy candidates from `candidate_summary.json`.
3. Compare:
   - realized equity
   - `profit_factor`
   - `sqn`
   - Sharpe-like / Monte Carlo `net_pnl_p05`
   - Monte Carlo `max_drawdown_p95` and `max_drawdown_p99`
4. Promote only a candidate that improves `2025` OOS and does not clearly degrade the combined run.

## Promotion rule
Do not point live config to a rebuilt DE3 bundle until:
- the bundle was built from `es_master_outrights.parquet`
- its backtests beat the contaminated baseline on `2025`
- the combined `2024-2025` run still holds up
- the artifact path is explicitly set in [config.py](/c:/Users/Timothy/OneDrive/Desktop/trading/topstep2/config.py)
