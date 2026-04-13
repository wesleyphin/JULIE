# Project Handoff

This file is the shortest practical way to re-enter this repo on another machine or in a later session without re-discovering the whole system.

## What This Repo Is

This is a Python trading/research repo for MES live trading and ES/MES backtesting on the TopstepX / ProjectX stack.

Main modes in the repo:

- Live bot and UI
- Backtesting
- Artifact-driven ML and hybrid strategy research
- Strategy-specific training workflows

Core reference doc:

- [README.md](./README.md)

## Read These First

Open these in this order:

1. [README.md](./README.md)
2. [config.py](./config.py)
3. [REGIME_ADAPTIVE_HANDOFF.md](./REGIME_ADAPTIVE_HANDOFF.md)
4. [DE3_GROUND_UP_WORKFLOW.md](./DE3_GROUND_UP_WORKFLOW.md)

Then open the strategy/runtime files relevant to the task.

## Current Trusted State

### RegimeAdaptive

RegimeAdaptive is the cleanest current handoff path in the repo.

- Live config points to `artifacts/regimeadaptive_v19_live/latest.json`
- Trusted research baseline is `artifacts/regimeadaptive_v19_walkforward_holdout2024_research/`
- Best older but less trustworthy branch is `artifacts/regimeadaptive_v15_gated_sharpe/`

The main rule for future work:

- continue from `v19` walk-forward plus untouched holdout discipline
- do not treat older higher full-sample metrics as more trustworthy

Primary docs/files:

- [REGIME_ADAPTIVE_HANDOFF.md](./REGIME_ADAPTIVE_HANDOFF.md)
- [regime_strategy.py](./regime_strategy.py)
- [regimeadaptive_artifact.py](./regimeadaptive_artifact.py)
- [regimeadaptive_gate.py](./regimeadaptive_gate.py)
- [tools/train_regimeadaptive_gate_walkforward.py](./tools/train_regimeadaptive_gate_walkforward.py)
- [tools/backtest_regimeadaptive_robust.py](./tools/backtest_regimeadaptive_robust.py)

### DE3

DE3 has a dedicated rebuild workflow intended for cleaner post-contamination research.

Current notable config state:

- `DE3_VERSION` is currently `"v1"` in [config.py](./config.py)
- filterless live mode forces DE3 to `v4` in [julie001.py](./julie001.py)
- the promoted DE3v4 live bundle now points to `artifacts/de3_v4_live/latest.json`
- there is a DE3 walk-forward gate artifact path in config, but it is backtest-only and disabled by default

Current DE3 promotion note:

- live DE3v4 now points at a promoted copy of `artifacts/de3_decision_policy_sideaware_20260330_biasgrid/dynamic_engine3_v4_bundle.decision_direct_compare_sideaware.json` under `artifacts/de3_v4_live/latest.json`
- previous promoted compare-expansion live bundle is preserved as `artifacts/de3_v4_live/latest.pre_sideaware_compare_20260330.json`
- previous live entry-policy promotion came from `artifacts/de3_entry_policy_shape_research_20260326/dynamic_engine3_v4_bundle.shape_tail_guard_sharpe.json`
- prior shape-tail research summary remains `backtest_reports/de3_shape_rework_20260326/summary.md`
- live runtime now also applies an exact DE3 variant multiplier trim on `5min_03-06_Long_Rev_T3_SL10_TP25` at `0.50x`
- filterless live now also applies a runtime DE3v4 signal-size trim on `15min_09-12_Long_Mom_T3_SL10_TP25` when `de3_entry_close_pos1 >= 0.40` and `de3_entry_dist_low5_atr >= 0.50`, reducing size by `0.50x`
- validation summary for that trim: `backtest_reports/de3_live_filterless_exact_20260327/summary.md`
- current live-wired full-span exact report: `backtest_reports/de3_live_filterless_exact_20260327/current_live_2022_2026jan/backtest_AUTO_BY_DAY_20220101_0000_20260126_2359_20260327_143949.json`
- latest promoted runtime follow-up now lives in `backtest_reports/de3_variant_trim_sweep_sync_20260401/summary.md`
- current live config now tightens `5min_03-06_Long_Rev_T3_SL10_TP25` from `0.50x` to `0.25x`
- current live config now also applies `live_06_09_long_rev_t2_size3_vol090_defensive_067`, trimming `15min_06-09_Long_Rev_T2_SL10_TP25` from `3 -> 2` when `de3_entry_vol1_rel20 >= 0.90`
- key takeaway from that pass: this is the first post-side-aware DE3 follow-up here that improved `2022`, `2024`, `2025`, untouched `2026-01`, and the full `2022-2026-01` span while staying entirely inside the existing live DE3 sizing stack
- latest live source-of-truth repair now lives in `backtest_reports/de3_live_bundle_sync_20260401/summary.md`
- current live config now also applies `live_09_12_long_mom_t3_body025_vol120_defensive_050`, `live_18_21_long_rev_t2_distlow020_range565_defensive_050`, and `live_12_15_long_rev_t2_distlow050_range400_defensive_050`
- current live config now sets `DE3_V4.runtime.execution_policy.calibrated_entry_model.use_bundle_model = True`
- key takeaway from that pass: the promoted DE3 live bundle had drifted away from runtime entry-model config; once `de3_v4_runtime.py` was fixed to use the bundle entry-policy parameters directly, `2024`, `2025`, untouched `2026-01`, and the full `2022-2026-01` span all improved materially versus the stale-config live path
- latest learned short-term side-choice research now lives in `backtest_reports/de3_short_term_condition_20260402/summary.md`
- live DE3v4 now points at a promoted copy of `artifacts/de3_short_term_condition_candidates_20260402_run1/dynamic_engine3_v4_bundle.short_term_local_sideaware.json` under `artifacts/de3_v4_live/latest.json`
- previous repaired live bundle is preserved as `artifacts/de3_v4_live/latest.pre_short_term_local_sideaware_20260402.json`
- key takeaway from that pass: this is the strongest full-range validated DE3 bundle so far that materially learns from chosen-bar short-term conditions; it improved the repaired live baseline on full-span `2022-2026-01`, `2025`, and untouched `2026-01`, while `2022` stayed positive but gave back modestly and long share remained a bit higher
- supporting structural research also fixed a real issue where some “short-term” scopes were using side-coded `strategy_type`; the new directionless `strategy_style` path lives in `de3_v4_decision_policy_trainer.py`, `de3_v4_runtime.py`, and `tools/train_de3_short_term_condition_candidates.py`, but the style-neutral and hybrid candidates did not beat `short_term_local_sideaware` on full-range robustness, so they remain research-only
- latest learned decision-side time-fix research now lives in `backtest_reports/de3_decision_side_timefix_eval_20260403/summary.md`
- live DE3v4 now points at a promoted copy of `artifacts/de3_decision_side_candidates_20260403_hourguards_split/dynamic_engine3_v4_bundle.decision_side_both_baseline_edge_ex_hour_12.json` under `artifacts/de3_v4_live/latest.json`
- previous promoted short-term-local-sideaware bundle is preserved as `artifacts/de3_v4_live/latest.pre_decision_side_ex_hour12_20260403.json`
- key takeaway from that pass: the decision-side trainer was missing usable time context on many exports, so `tools/train_de3_decision_side_model.py` now derives `ctx_hour_et`, `ctx_hour_bucket`, `ctx_session_substate`, and `year` from timestamps when needed, and both trainer/runtime now support hour-bucket-scoped side overrides; the best exact DE3-only result was the learned side-choice bundle with `exclude_hour_buckets=["12"]`, which improved the full `2022-2026-01` span over both the live baseline and the earlier `exclude 10 + 12` guard, improved `2023`, `2024`, and `2025`, and held untouched `2026-01` unchanged while keeping `2022` strongly positive
- latest exact-substrategy side-choice follow-up now lives in `backtest_reports/de3_decision_side_exact_substrategy_20260403/summary.md`
- important source changes from that pass: `tools/train_de3_decision_side_model.py` now supports exact `chosen_sub_strategy` / `long_sub_strategy` / `short_sub_strategy` context plus exact long-only and exact both-side profile families, and `de3_v4_runtime.py` now exposes exact sub-strategy context to decision-side runtime models and supports `apply_prior_only_when_predicted` for safer soft-prior overlays
- key takeaway from that pass: additive exact-substrategy side models did not beat the promoted `decision_side_both_baseline_edge_ex_hour_12` live bundle on a full-range robustness basis, but the solo exact paired-side chooser (`artifacts/de3_decision_side_exact_both_20260403_noex12/dynamic_engine3_v4_bundle.decision_side_both_exact_pair_solo.json`) materially improved short stressed windows like `2025-01` and untouched `2026-01`; it still failed promotion because it gave back too much on `2022` and full `2025`, so live stayed unchanged
- latest fresh-live DE3 side follow-up now lives in `backtest_reports/de3_freshlive_side_followup_20260404/summary.md`
- key takeaway from that pass: once the side/direction work was re-run off fresh live-valid decision exports and a fresh current-pool export from the current promoted runtime, none of the tempting follow-ons actually beat the promoted bundle on a broad robustness basis; fresh exact paired-side retrains still only helped stressed windows, fresh current-pool side-aware retrains still gave back too much on `2022` / `2024` / `2025`, the new `liveplus_local_*` trainer profiles fixed the missing short-term-condition term but still failed promotion, and even a runtime book-gated `pfstrict` alternate model for hour `20` did not produce a real uplift
- code behind that follow-up now also extends `tools/train_de3_entry_policy_from_current_pool.py` with `decision_direct_compare_liveplus_local_side` and `decision_direct_compare_liveplus_local_guarded`, which are useful fresh-live research baselines but remain non-promotable
- DE3 sanity sweep before any new rebuild: the current promoted live bundle is a mixed stack, not a single clean chooser; `decision_policy_model` is the main selector, but `decision_side_model` is still enabled as a `soft_prior` layer with `prior_component_weight=0.14`
- that live `decision_side_model` is also much narrower than it first appears: it only applies on `apply_side_patterns=["both"]` and excludes hour bucket `12`
- fresh live-valid side dataset summary (`reports/de3_decision_side_dataset_fresh_current_live_2011_2024.summary.json`) shows why this matters: only `13,478 / 399,011` decision events (`3.3779%`) are `both_side_decisions`, so the current side layer cannot solve DE3's general long-bias problem by itself
- practical implication for future DE3 work: do not treat the current side model as a full solution; the next real formulation should learn `long / short / no-trade` from exact local decision events and ideally collapse the mixed direct-chooser + soft-side-prior stack into one cleaner action-selection path
- new sanity/rebuild support now also lives in `tools/train_de3_decision_action_stack.py`; it can train stack bundles from the decision-side dataset and combine multiple `decision_side_models` into one bundle for real runtime testing
- `tools/train_de3_decision_side_model.py` now also includes missing `short_only` exact/guarded profile families, which closes a structural hole in the older side-learning inventory
- real trainer bug fixed on 2026-04-04: `_evaluate_thresholds()` in `tools/train_de3_decision_side_model.py` previously chose the best “valid” override trial even when every override was worse than baseline; it now falls back to baseline if no override actually beats the baseline objective
- real stack-eval bug fixed on 2026-04-04: `tools/train_de3_decision_action_stack.py` now replays trained models with their selected thresholds instead of empty params
- first full action-stack pass after those fixes did not beat the current live DE3 baseline: once the bugs were removed, the hard `both + long_only + short_only` action stacks all collapsed to baseline/no-op on tune, and the soft full-coverage stacks also selected `baseline_no_override`
- important reality check from the same pass: simply removing the current live `decision_side_model` and backtesting the cleaned bundle under `artifacts/de3_decision_side_cleanup_20260404/dynamic_engine3_v4_bundle.no_decision_side.json` reverts performance back toward the older `short_term_local_sideaware` level (`2022`: `3347.91`, `2025`: `31656.10`, `2026-01`: `3730.72`), which means the current live side-prior is not dead weight even though it only acts on a small both-side slice
- current best practical conclusion: keep `artifacts/de3_v4_live/latest.json` as the live baseline; use the new exact-decision/action-stack tooling as research infrastructure, but do not promote any of the first action-stack candidates from `artifacts/de3_decision_action_stack_20260404*` or `artifacts/de3_decision_side_softstack_20260404/`
- latest direct action-condition follow-up now lives in `backtest_reports/de3_decision_action_eval_20260404_v2/summary.md`
- important implementation detail from that pass: `de3_v4_decision_policy_trainer.py` and `de3_v4_runtime.py` now support a fresh decision-event `action_condition_model` inside the main direct chooser, including asymmetric positive/negative score weighting and `apply_only_top_side_candidate`
- key takeaway from that pass: the new direct action-condition path is real research infrastructure, but even the cleaned `current live + bonus-only local action` variant (`artifacts/de3_decision_action_candidates_20260404_v2/dynamic_engine3_v4_bundle.decision_action_bonus_only_v2.json`) still underperformed the current promoted bundle on untouched `2026-01` (`3153.76` vs `3458.81`) and full `2025` (`30460.64` vs `32958.26`), so live stayed unchanged
- DE3v5-lite direct chooser research now lives in `backtest_reports/de3_decision_policy_rework_20260328_hybrid/summary.md`
- current best compromise shadow artifact from that path is `artifacts/de3_decision_policy_rework_20260328_hybrid_thresholds/dynamic_engine3_v4_bundle.decision_direct_hybrid_v1_thr_m0p3.json`
- key takeaway from that pass: raw direct chooser is very strong on `2024-2026` windows but too aggressive for `2022`; stricter hybrid thresholding improves the compromise but still does not beat the current live-style exact baseline on `2022`
- latest DE3v5-lite follow-on research now lives in `backtest_reports/de3_decision_policy_rework_20260328_compare/summary.md`
- current best shadow artifact from the compare-against-baseline path is `artifacts/de3_decision_policy_rework_20260328_compare/dynamic_engine3_v4_bundle.decision_direct_compare_expansion.json`
- key takeaway from that pass: compare-mode keeps most of the raw direct chooser's recent upside and clearly beats the current live-style exact baseline on `2025`, `2026-01`, `2024-2025`, and full-span `2022-2026-01`, but `2022` still remains negative, so it is a strong research base and shadow candidate rather than a clean full-regime promotion
- latest robustness follow-up now lives in `backtest_reports/de3_decision_policy_rework_20260330_followup/summary.md`
- older full-range add-on note for context: the first robustness follow-up promoted `5min_03-06_Long_Rev_T3_SL10_TP25=0.5`; that was later superseded by the stronger synced live-portable combo in `backtest_reports/de3_variant_trim_sweep_sync_20260401/summary.md`
- best old-tape repair found in the follow-up is disabled research rule `exact_03_06_long_rev_t3_lower015_defensive_050`; it turns `2022` positive, but it still underperforms the `T3 0.5x` base on the full span and recent windows, so it stays shadow-only
- additive follow-up trims that looked good in trade-log mining did not hold up once backtested on the full trade path; treat the `T3 0.5x` base as the real bar for future DE3v5-lite work
- latest structural skew repair now lives in `backtest_reports/de3_sideaware_eval_20260330/summary.md`
- key takeaway from that pass: moving side-awareness into the direct chooser score itself was the first DE3v5-lite compare-mode fix that turned `2022` positive, improved the untouched `2026-01` holdout, improved the full `2022-2026-01` span, and cut the full-span long share from `88.0%` to `76.3%`
- code behind that pass lives in `de3_v4_decision_policy_trainer.py`, `de3_v4_runtime.py`, `de3_v4_entry_policy_trainer.py`, and `tools/train_de3_entry_policy_from_current_pool.py`
- latest PF-core follow-up now lives in `backtest_reports/de3_pfcore_eval_20260330/summary.md`
- key takeaway from that pass: adding a training-stat candidate filter can create narrower, more efficient DE3 cores, but the only path that gets near the desired PF target collapses into an asymmetric short-only sub-book that fails `2024`; no PF-core bundle beat the promoted side-aware live bundle on a full-range robustness basis, so live stayed unchanged
- code behind that pass currently lives in `de3_v4_runtime.py`
- latest multibook recheck now lives in `backtest_reports/de3_multibook_gate_livevalidfix_20260331/summary.md`
- current live-valid base export for that pass is `reports/de3_multibook_gate_livebase_2022_2024.csv` with trade attribution in `reports/de3_multibook_gate_livebase_2022_2024_trade_attribution.csv`
- key takeaway from the recheck: the earlier multibook uplift was contaminated by dead pre-runtime candidate context, especially `5min_09-12_Long_Rev_T6_SL10_TP25`; once the base export was regenerated from the actual promoted live bundle and `de3_v4_book_gate_trainer.py` was fixed to only trust runtime-populated / chosen context, the apparent edge disappeared on the `2024` tune window
- current best multibook artifact from the cleaned pass is `artifacts/de3_multibook_gate_livevalidfix_20260331/dynamic_engine3_v4_bundle.multibook_session_subvar_pf.json`, but it is research-only and not promoted; live remains on the promoted side-aware bundle under `artifacts/de3_v4_live/latest.json`
- code behind the multibook path now lives in `de3_v4_book_gate_trainer.py`, `de3_v4_runtime.py`, `tools/build_de3_multibook_variant_books.py`, `tools/train_de3_multibook_gate.py`, `tools/run_de3_multibook_gate_pipeline.py`, and `tools/run_de3_multibook_gate_eval_parallel.py`
- source-level cleanup is now in place too: `dynamic_engine3_strategy.py` exports DE3v4 post-runtime `feasible_rows` when available, so dead `5min_09-12_Long_Rev_T6_SL10_TP25` rows no longer leak into new decision exports
- clean post-runtime multibook base export now lives in `reports/de3_multibook_gate_livebase_2022_2024_postfix.csv` with trade attribution in `reports/de3_multibook_gate_livebase_2022_2024_postfix_trade_attribution.csv`
- retraining from the source-fixed export under `artifacts/de3_multibook_gate_livevalidsrcfix_20260401/` leaves the best `multibook_session_subvar_pf` candidate as a true tune no-op, which is the strongest evidence that the old multibook uplift was contamination rather than a robust live edge
- first clean active-divergence reroute (`5min_03-06_Long_Rev_T3_SL10_TP25 -> short_direct_soft`) was backtested under `artifacts/de3_multibook_targeted_20260401/` and failed the `2022` robustness bar, so it remains research-only
- next clean active-divergence branch now lives in `backtest_reports/de3_multibook_targeted_20260401b/summary.md`; the `12-15 Long_Rev T2 -> shortrev`, `12-15 Long_Mom T3 -> shortrev`, and combined `both12` reroutes all failed to beat the promoted side-aware `2022` baseline, so that branch is currently closed out as non-promotable

Primary workflow/doc:

- [DE3_GROUND_UP_WORKFLOW.md](./DE3_GROUND_UP_WORKFLOW.md)
- [tools/run_de3_ground_up_workflow.py](./tools/run_de3_ground_up_workflow.py)

Important DE3 rule:

- DE3 rebuilds should use `es_master_outrights.parquet`

### AetherFlow

AetherFlow is wired in config as a live-enabled strategy using a narrower deploy artifact choice, not the broader research bundle.

Current config points to:

- model: `model_aetherflow_deploy_2026oos.pkl`
- thresholds: `aetherflow_thresholds_deploy_2026oos.json`
- metrics: `aetherflow_metrics_deploy_2026oos.json`
- live config now also sets `AETHERFLOW_STRATEGY.threshold_override = 0.55`
- live config keeps `allowed_setup_families = ["compression_release", "transition_burst"]`
- coverage repair summary now lives in `backtest_reports/aetherflow_cov_20260401/summary.md`
- key takeaway from that pass: the main live bottleneck was not “AetherFlow never signals,” it was a combination of an overly tight `0.59` deploy threshold and a setup-family selection bug where the strategy chose the best global family before applying the family allowlist; fixing that to prefer the best allowed family plus using a `0.55` override materially increased coverage without opening the noisy `aligned_flow` / `exhaustion_reversal` families

Primary files:

- [aetherflow_strategy.py](./aetherflow_strategy.py)
- [aetherflow_features.py](./aetherflow_features.py)
- [train_aetherflow.py](./train_aetherflow.py)
- [tools/train_aetherflow_walkforward.py](./tools/train_aetherflow_walkforward.py)
- [tools/run_aetherflow_walkforward_direct.py](./tools/run_aetherflow_walkforward_direct.py)
- [tools/backtest_aetherflow_direct.py](./tools/backtest_aetherflow_direct.py)

### ML Physics

ML Physics is still an important system, but the repo is configured around the dist-bracket artifact path rather than a simple single-model flow.

Notable config direction:

- `ML_PHYSICS_REPLACE_WITH_DIST` is enabled in [config.py](./config.py)
- runtime points at a specific `dist_bracket_ml_runs/...` artifact directory

Primary files:

- [ml_physics_strategy.py](./ml_physics_strategy.py)
- [ml_physics_pipeline.py](./ml_physics_pipeline.py)
- [dist_bracket_ml/README.md](./dist_bracket_ml/README.md)

## Main Runtime Entry Points

Use these depending on task:

- live bot: [julie001.py](./julie001.py)
- UI launcher: [launch_ui.py](./launch_ui.py)
- filterless live launcher: [launch_filterless_live.py](./launch_filterless_live.py)
- workspace-style launcher: [launch_filterless_workspace.py](./launch_filterless_workspace.py)
- main backtest: [backtest_mes_et.py](./backtest_mes_et.py)
- backtest UI: [backtest_mes_et_ui.py](./backtest_mes_et_ui.py)

Support modules worth remembering:

- [async_market_stream.py](./async_market_stream.py)
- [async_tasks.py](./async_tasks.py)
- [client.py](./client.py)
- [session_manager.py](./session_manager.py)
- [process_singleton.py](./process_singleton.py)

## Config Is The Real Control Plane

Before changing behavior, inspect [config.py](./config.py).

It controls:

- active live artifact paths
- strategy enable/disable flags
- session rules
- filter gates and guardrails
- DE3 version selection
- ML Physics dist replacement
- RegimeAdaptive live artifact wiring
- AetherFlow deploy artifact wiring

If behavior looks surprising, check config before touching strategy logic.

## Data And Artifact Reality

This repo is not a small clean code-only project. It contains a mix of:

- code
- local datasets
- trained models
- cached outputs
- logs
- research artifacts
- backup files

Files/directories that matter often:

- `es_master.csv`
- `es_master.parquet`
- `es_master_outrights.parquet`
- `artifacts/`
- `backtest_reports/`
- `checkpoints/`
- `dist_bracket_ml_runs/`
- `runpod_results/`
- `tools/`

Assume many workflows depend on local files already present in the repo root.

## Safe Research Heuristics

These are the practical rules that matter most:

- Prefer walk-forward plus untouched holdout validation over prettier full-sample metrics.
- For rebuilds, default to `es_master_outrights.parquet` instead of older mixed or contaminated paths.
- For RegimeAdaptive, start from the `v19` walk-forward holdout branch.
- For DE3, rebuild from outright-only data when doing clean research.
- Treat `config.py` as source of truth for what is actually live or active.
- Use the narrow direct validators before the broad mixed backtest engine when iterating on one subsystem.

## Common Commands

Typical commands already implied by repo docs:

```powershell
python julie001.py
python launch_ui.py
python backtest_mes_et.py
python backtest_mes_et_ui.py
python tools\run_de3_ground_up_workflow.py --tag clean_outrights
python tools\train_regimeadaptive_gate_walkforward.py
```

Install baseline dependencies with:

```powershell
pip install -r freeze.txt
```

## Secrets And Portability

Important:

- [config.py](./config.py) imports local secrets from `config_secrets.py`
- treat credentials and API keys as machine-local
- do not rely on hardcoded secrets being safe to share

If moving machines, preserve:

- this repo
- this handoff file
- the artifact/model directories you actually need
- any local secret/config mechanism used by the current machine

## If You Only Have Five Minutes

Do this:

1. Read [PROJECT_HANDOFF.md](./PROJECT_HANDOFF.md)
2. Read [README.md](./README.md)
3. Check the active paths in [config.py](./config.py)
4. If the task is RegimeAdaptive, read [REGIME_ADAPTIVE_HANDOFF.md](./REGIME_ADAPTIVE_HANDOFF.md)
5. If the task is DE3, read [DE3_GROUND_UP_WORKFLOW.md](./DE3_GROUND_UP_WORKFLOW.md)

## One-Line Summary

Treat `config.py` as runtime truth, use `es_master_outrights.parquet` for rebuilds, trust the RegimeAdaptive `v19` walk-forward holdout path over older flashy results, and use the dedicated docs before touching strategy logic.
