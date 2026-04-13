# RegimeAdaptive Handoff

This note is the quickest way to re-enter `RegimeAdaptive` work without rediscovering the whole history.

## Current Status

- Live runtime currently points to the promoted `v19` bundle in [config.py](./config.py).
  - `artifacts/regimeadaptive_v19_live/latest.json`
- The most trustworthy research baseline is the walk-forward + final-holdout build in:
  - `artifacts/regimeadaptive_v19_walkforward_holdout2024_research/`
- The best old "looks strong but has more leakage risk" branch was:
  - `artifacts/regimeadaptive_v15_gated_sharpe/`

## Files To Open First

1. [regime_strategy.py](./regime_strategy.py)
   - Runtime entrypoint.
   - `RegimeAdaptiveStrategy` starts here.
   - This is where rule selection, side reversal, gate application, and payload emission happen.

2. [regimeadaptive_artifact.py](./regimeadaptive_artifact.py)
   - Defines the artifact schema and resolution order.
   - This is the source of truth for:
     - `combo_policies`
     - `signal_policies`
     - `group_signal_policies`
     - `rule_catalog`
     - `session_defaults`
     - `global_default`
     - `signal_gate`

3. [regimeadaptive_gate.py](./regimeadaptive_gate.py)
   - Gate model loader and runtime feature-row builder.
   - If gate behavior looks wrong, start here before touching the strategy.

4. [tools/backtest_regimeadaptive_robust.py](./tools/backtest_regimeadaptive_robust.py)
   - Fast direct validator for a single artifact.
   - Use this before touching the full mixed backtest engine.
   - Also writes converted trade CSVs.

5. [tools/train_regimeadaptive_gate_walkforward.py](./tools/train_regimeadaptive_gate_walkforward.py)
   - Best current research workflow.
   - This is the first trainer to add a real untouched holdout year.
   - If you are trying to improve the strategy responsibly, start here, not from the older gate trainer.

## Where The Trustworthy Evidence Lives

- Main report:
  - [artifacts/regimeadaptive_v19_walkforward_holdout2024_research/walkforward_report.json](./artifacts/regimeadaptive_v19_walkforward_holdout2024_research/walkforward_report.json)
- Final holdout trade CSV:
  - [artifacts/regimeadaptive_v19_walkforward_holdout2024_research/final_holdout_trades.csv](./artifacts/regimeadaptive_v19_walkforward_holdout2024_research/final_holdout_trades.csv)
- Candidate research artifact:
  - [artifacts/regimeadaptive_v19_walkforward_holdout2024_research/regimeadaptive_gated_walkforward_research.json](./artifacts/regimeadaptive_v19_walkforward_holdout2024_research/regimeadaptive_gated_walkforward_research.json)

What mattered from that run:

- Source artifact for gating research was the dense baseline:
  - `artifacts/regimeadaptive_v14_dense_balanced/latest.json`
- Walk-forward out-of-sample period:
  - `2021-2023`
- Final untouched holdout:
  - `2024`
- Holdout result was materially better quality than the ungated dense baseline, even with fewer trades.

That is the reason `v19` is the safest place to continue from.

## Current Runtime Wiring

- Live artifact path is set in [config.py](./config.py) under `REGIME_ADAPTIVE_TUNING`.
- Runtime loads the artifact and optional gate model in [regime_strategy.py](./regime_strategy.py).
- The filterless live bot executes it through [julie001.py](./julie001.py).

Important live details already wired:

- RegimeAdaptive is part of the filterless live roster.
- Live sizing includes the RegimeAdaptive-specific growth sizing path.
- New entries are blocked from `4:00 PM` to `6:00 PM ET`; existing trades are still managed.

## Artifact Lineage That Actually Mattered

- `v14_dense_balanced`
  - Solved the "not enough trades" problem.
  - Too loose on quality by itself.

- `v15_gated_sharpe`
  - Big improvement from adding the ML gate.
  - Strong results, but threshold selection still had leakage risk.

- `v19_walkforward_holdout2024_research`
  - First branch with cleaner walk-forward selection and untouched holdout validation.
  - This is the baseline to trust more than the earlier flashy numbers.

- `v19_live`
  - Live packaging of the approved artifact/model pair.

## What Failed And Should Not Be Repeated Blindly

- Blind exact-bucket tuning.
  - It found local improvements, but eventually just moved weakness from one period to another.

- Simple "prune the weak combos" passes.
  - The pruned variants often looked better on the selection window.
  - They got worse on the untouched 2024 holdout.

- Treating high SQN or high full-sample Sharpe as proof.
  - Earlier versions benefited from trainer leakage and full-history threshold selection.
  - The walk-forward holdout result is more important than the prettiest aggregate metric.

- Reverting to the old `v15` path as the research starting point.
  - Keep it as reference only.

## What To Improve Next

If someone wants to push RegimeAdaptive further, this is the order I would use:

1. Keep the `v19` walk-forward discipline.
   - Any new trainer should preserve:
     - walk-forward fold selection
     - untouched final holdout
     - separate reporting for baseline vs selected

2. Improve pre-gate signal quality, not just the gate threshold.
   - The dense baseline created the trade flow.
   - The gate cleaned it up.
   - The next edge probably comes from making the dense candidate set less noisy before gating.

3. Add new rule families carefully.
   - Continuation helped.
   - More threshold churn on the same SMA/ATR logic had diminishing returns.
   - New archetypes should still be validated through the same walk-forward pipeline.

4. Preserve exact SL/TP overrides.
   - Earlier trainer paths accidentally lost good side-specific SL/TP overrides.
   - Be careful whenever rebuilding `combo_policies`.

5. Prefer the direct validator for fast iteration.
   - Use [tools/backtest_regimeadaptive_robust.py](./tools/backtest_regimeadaptive_robust.py) to sanity-check artifacts before running broader mixed-engine work.

## Practical Start Checklist

If you are picking this up later, do this first:

1. Read [REGIME_ADAPTIVE_HANDOFF.md](./REGIME_ADAPTIVE_HANDOFF.md).
2. Open [config.py](./config.py) and confirm which artifact path is active.
3. Read [regime_strategy.py](./regime_strategy.py) and [regimeadaptive_artifact.py](./regimeadaptive_artifact.py).
4. Read [tools/train_regimeadaptive_gate_walkforward.py](./tools/train_regimeadaptive_gate_walkforward.py).
5. Inspect [artifacts/regimeadaptive_v19_walkforward_holdout2024_research/walkforward_report.json](./artifacts/regimeadaptive_v19_walkforward_holdout2024_research/walkforward_report.json).
6. Only then decide whether the next experiment is:
   - new rule family
   - gate regularization
   - cleaner dense baseline retrain

## One-Line Summary

If you only remember one thing: start from `v19` walk-forward holdout research, not from the older higher-metric branches.
