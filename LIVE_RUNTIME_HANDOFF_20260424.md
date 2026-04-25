# Live Runtime Handoff (2026-04-24)

This document is the current source of truth for the **live runtime** state of
`DE3` and `AetherFlow`.

Use this before trusting older handoff notes, older promotion summaries, or
large research/backtest numbers.

## Current GitHub Sync

As of the 2026-04-25 cleanup, `main` is the live runtime source of truth.
The old runtime-only draft PR `#206` (`codex/publish-live-ready-stack-20260424`)
is superseded and should not be merged as-is because it was based on an older
launcher/defaults snapshot.

The current sync intentionally includes:

- live runtime code
- promoted runtime artifacts that are small enough to track
- launcher defaults
- runtime-side overlays that are actually loaded by the live bot

That sync intentionally excludes:

- UI work
- broad research history
- backtest output bulk
- temp/probe/local-machine debris

## Read This First

If you are re-entering the repo for live/runtime work, open files in this order:

1. [config.py](./config.py)
2. [launch_filterless_live.py](./launch_filterless_live.py)
3. [julie001.py](./julie001.py)
4. [LIVE_RUNTIME_HANDOFF_20260424.md](./LIVE_RUNTIME_HANDOFF_20260424.md)
5. [PROJECT_HANDOFF.md](./PROJECT_HANDOFF.md)

## Live Launch Path

Preferred live entry path:

- [LaunchFilterlessWorkspace.bat](./LaunchFilterlessWorkspace.bat)
- [launch_filterless_live.py](./launch_filterless_live.py)

If a bot is already running, it must be restarted to pick up config or launcher
changes.

## DE3: Current Live Runtime

### Core bundle

Live is pinned to the explicit promoted bundle:

- [artifacts/de3_v4_live/dynamic_engine3_v4_bundle.decision_side_daytype_soft_v1_20260422_promoted.json](./artifacts/de3_v4_live/dynamic_engine3_v4_bundle.decision_side_daytype_soft_v1_20260422_promoted.json)

`config.py` points directly to that file, not just a moving alias:

- [config.py](./config.py)

`artifacts/de3_v4_live/latest.json` is still updated, but the explicit promoted
file is the safer thing to reference in future work.

### Runtime settings that are intended to be live

- DE3 signal gate: **off**
- dynamic gate thresholding: **off**
- DE3 LFO policy: **ml**
- DE3 Kalshi rule overlay: **on**, but **scoped to DE3 only**
- DE3 family-profile veto: **on**, but only one narrow rule is enabled:
  - `frv_06_09_long_rev_t2_normal_grind_down_distributed_normal_15m`

Relevant files:

- [config.py](./config.py)
- [launch_filterless_live.py](./launch_filterless_live.py)
- [de3_v4_runtime.py](./de3_v4_runtime.py)
- [dynamic_engine3_strategy.py](./dynamic_engine3_strategy.py)
- [dynamic_signal_engine3.py](./dynamic_signal_engine3.py)
- [level_fill_optimizer.py](./level_fill_optimizer.py)
- [kalshi_trade_overlay.py](./kalshi_trade_overlay.py)
- [ml_overlay_shadow.py](./ml_overlay_shadow.py)

### Most relevant validated DE3 numbers

Integrated DE3 live-stack backtests with real backtest-path `ML LFO + DE3-only
Kalshi` enabled:

- `2025-01-01` to `2025-12-31`: net `$20,135.53`, max DD `$2,231.36`, `2092` trades
- `2026-01-01` to `2026-04-17`: net `$1,874.01`, max DD `$2,790.92`, `902` trades

Artifacts:

- [artifacts/de3_live_stack_backtests_20260424/year_2025/summary.json](./artifacts/de3_live_stack_backtests_20260424/year_2025/summary.json)
- [artifacts/de3_live_stack_backtests_20260424/holdout_2026/summary.json](./artifacts/de3_live_stack_backtests_20260424/holdout_2026/summary.json)

### Important DE3 caution

Older DE3 numbers from other research branches may be stronger on paper, but
many of them were:

- backtest-only overlays
- old runtime mixes
- incomplete live-stack replays
- or not the exact promoted runtime path

For current live work, treat the explicit promoted bundle plus the narrow
`06-09` repair rule as the actual baseline.

## AetherFlow: Current Live Runtime

### Canonical manifold path

AetherFlow now uses the **canonical full manifold base cache**, not “newest file
wins” behavior:

- [artifacts/aetherflow_corrected_full_2011_2026/manifold_base_outrights_2011_2026.parquet](./artifacts/aetherflow_corrected_full_2011_2026/manifold_base_outrights_2011_2026.parquet)
- metadata: [artifacts/aetherflow_corrected_full_2011_2026/manifold_base_outrights_2011_2026.parquet.meta.json](./artifacts/aetherflow_corrected_full_2011_2026/manifold_base_outrights_2011_2026.parquet.meta.json)

Important:

- the promoted model/threshold/metrics artifacts are tracked in Git
- the computed manifold parquet is intentionally **not** committed because it is
  large generated data
- a fresh checkout must build the cache locally before relying on full-history
  live-equivalent AetherFlow behavior

Builder:

```powershell
.\.venv\Scripts\python.exe tools\build_manifold_base_cache.py `
  --source es_master_outrights.parquet `
  --output artifacts\aetherflow_corrected_full_2011_2026\manifold_base_outrights_2011_2026.parquet
```

See [MANIFOLD_BASE_CACHE.md](./MANIFOLD_BASE_CACHE.md) for validation details.

### Promoted routed model

Current live AetherFlow model:

- [artifacts/aetherflow_routed_ensemble_candidates_20260422/radical_af_nyam_trend_v1/model.pkl](./artifacts/aetherflow_routed_ensemble_candidates_20260422/radical_af_nyam_trend_v1/model.pkl)
- [thresholds.json](./artifacts/aetherflow_routed_ensemble_candidates_20260422/radical_af_nyam_trend_v1/thresholds.json)
- [metrics.json](./artifacts/aetherflow_routed_ensemble_candidates_20260422/radical_af_nyam_trend_v1/metrics.json)

Relevant files:

- [config.py](./config.py)
- [aetherflow_base_cache.py](./aetherflow_base_cache.py)
- [aetherflow_live_runtime_cache.py](./aetherflow_live_runtime_cache.py)
- [aetherflow_strategy.py](./aetherflow_strategy.py)
- [tools/backtest_aetherflow_direct.py](./tools/backtest_aetherflow_direct.py)
- [tools/run_aetherflow_live_policy_search.py](./tools/run_aetherflow_live_policy_search.py)

### Runtime settings that are intended to be live

- AetherFlow signal gate: **off**
- AetherFlow LFO policy: **off**
- AetherFlow Kalshi overlay: **off / scoped out**
- canonical full manifold cache: **required**
- AetherFlow direct/live drawdown sizing override: **on**
  - starts at `$250` realized drawdown
  - reaches minimum size at `$800` realized drawdown
  - caps from boosted requested size down to `1` contract under drawdown
- AetherFlow direct replay execution window:
  - no new entries from `15:00` to `18:00` ET
  - force-flat at `16:00` ET
- AetherFlow policy promotion, 2026-04-25:
  - blocks weak full-history pockets:
    - transition_burst / ASIA / DISPERSED
    - transition_burst / NY_AM / TREND_GEODESIC / LONG
    - aligned_flow / NY_AM / DISPERSED / SHORT
  - boosts high-confidence pockets with a post-policy `2.2x` size multiplier:
    - transition_burst / LONDON at confidence `>= 0.70`
    - transition_burst / NY_AM / SHORT at confidence `>= 0.70`
    - exhaustion_reversal / ASIA at confidence `>= 0.70`

### Most relevant validated AetherFlow numbers

Promoted live-equivalent goal candidate, exact direct replay:

- full history `2011-01-01` to `2026-04-20`: net `$151,120.07`, max DD `$1,287.84`, PF `3.3198`, `4606` trades
- winrate `65.37%`
- exact local report:
  - [backtest_reports/af_direct_goal_candidate_20260425_boost22/backtest_AUTO_BY_DAY_20110101_0000_20260420_2359_20260425_043655.json](./backtest_reports/af_direct_goal_candidate_20260425_boost22/backtest_AUTO_BY_DAY_20110101_0000_20260420_2359_20260425_043655.json)

Previous corrected live-equivalent runtime after the runtime refactor:

- full history `2011-01-01` to `2026-04-20`: net `$141,251.55`, max DD `$1,908.94`, PF `2.804`, `4515` trades
- exact `2025`: net `$10,785.24`, max DD `$1,096.36`, `287` trades

Previous baseline artifacts:

- [backtest_reports/af_direct_livealigned_full_runtime_refactor3_20260424/backtest_AUTO_BY_DAY_20110101_0000_20260420_2359_20260424_170858.json](./backtest_reports/af_direct_livealigned_full_runtime_refactor3_20260424/backtest_AUTO_BY_DAY_20110101_0000_20260420_2359_20260424_170858.json)
- [backtest_reports/af_direct_livealigned_2025_runtime_refactor3_20260424/backtest_AUTO_BY_DAY_20250101_0000_20251231_2359_20260424_170652.json](./backtest_reports/af_direct_livealigned_2025_runtime_refactor3_20260424/backtest_AUTO_BY_DAY_20250101_0000_20251231_2359_20260424_170652.json)

### Important AetherFlow caution

Do **not** treat these older numbers as the current live-equivalent truth:

- `$271k` old dense-harness AF core replay
- `$294k` / `$332k` AF gate replay results

Why:

- those came from older, looser trade universes or gate replays
- they were not the corrected manifold-backed live-equivalent runtime

The current trustworthy AetherFlow baseline is the corrected runtime-refactor
path above.

## Overlay / Module State

### Signal gates

Per-strategy signal gates are available but intentionally default **off** in
the live launcher because full-history utility checks did not justify keeping
them on for DE3 or AF.

Relevant files:

- [signal_gate_2025.py](./signal_gate_2025.py)
- [loss_factor_guard.py](./loss_factor_guard.py)
- [launch_filterless_live.py](./launch_filterless_live.py)

### LFO

Current intended live policy:

- `DE3 -> ml`
- `RegimeAdaptive -> rule`
- `AetherFlow -> off`
- `MLPhysics -> off`

Relevant files:

- [ml_overlay_shadow.py](./ml_overlay_shadow.py)
- [launch_filterless_live.py](./launch_filterless_live.py)
- [level_fill_optimizer.py](./level_fill_optimizer.py)

### Kalshi

Current intended live state:

- rule overlay enabled in config
- scoped to `DynamicEngine3` only
- ML Kalshi entry / TP models remain shadow-only by launcher default

Relevant files:

- [config.py](./config.py)
- [julie001.py](./julie001.py)
- [kalshi_trade_overlay.py](./kalshi_trade_overlay.py)
- [services/kalshi_provider.py](./services/kalshi_provider.py)

### RL management

Current launcher default:

- `JULIE_ML_RL_MGMT_ACTIVE=1`

That means the runtime currently expects the promoted RL management artifact to
exist:

- [artifacts/signal_gate_2025/model_rl_management.zip](./artifacts/signal_gate_2025/model_rl_management.zip)
- [artifacts/signal_gate_2025/bar_encoder.pt](./artifacts/signal_gate_2025/bar_encoder.pt)

Relevant files:

- [launch_filterless_live.py](./launch_filterless_live.py)
- [ml_overlay_shadow.py](./ml_overlay_shadow.py)
- [rl/inference.py](./rl/inference.py)
- [rl/trade_env.py](./rl/trade_env.py)
- [rl/cross_market.py](./rl/cross_market.py)

## Live Sanity Checks

These are the quickest checks future workers should run after touching runtime
code or pulling onto a new machine:

```powershell
.\.venv\Scripts\python.exe -m py_compile aetherflow_strategy.py julie001.py config.py signal_gate_2025.py
.\.venv\Scripts\python.exe -m unittest test_ml_overlay_lfo_policy.py test_signal_gate_2025_runtime.py test_aetherflow_live_runtime_cache.py test_aetherflow_cached_base_runtime.py
@'
from aetherflow_strategy import AetherFlowStrategy
s = AetherFlowStrategy()
print(s.model_loaded, s._live_base_features_path, s.threshold)
'@ | .\.venv\Scripts\python.exe -
```

If the AetherFlow smoke load does not show the canonical full manifold path, or
that path does not exist locally, build the manifold cache before trusting any
AF runtime results.

## What Future Workers Should Not Redo Blindly

Do not assume:

- newest artifact = best artifact
- higher backtest number = live-equivalent result
- older AetherFlow gate replay numbers are the real runtime
- older DE3 safer overlays are still the current baseline

Do:

- start from the pinned config paths
- verify launcher defaults
- verify the canonical AF manifold parquet has been built locally
- compare against the exact current runtime baselines above before promoting changes

## Short Version

If you only remember three things:

1. `DE3` live = promoted percent bundle + one narrow `06-09` repair + `ML LFO` + DE3-scoped Kalshi.
2. `AetherFlow` live = canonical full manifold + routed promoted model, **not** the older `$271k/$294k/$332k` research/gate numbers.
3. The AetherFlow model artifact is tracked; the AF manifold parquet is generated locally with the builder in `MANIFOLD_BASE_CACHE.md`.
