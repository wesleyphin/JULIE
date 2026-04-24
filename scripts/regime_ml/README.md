# Regime ML — Reproducible Trainer Kit

Three small HGB classifiers that replace pieces of the rule-based
`regime_classifier.apply_dead_tape_brackets()`. Each trainer is
standalone, deterministic, and runs from a single command.

## Shipped models (as of commit reference below)

| Model | File | Action | OOS lift vs rule baseline |
|---|---|---|---|
| **A** — scalp brackets | `artifacts/regime_ml_v5_brackets/model.pkl` | Rewrite TP/SL to 3/5 | +$11,160 PnL, −$780 DD (HGB-only @ thr=0.50) |
| **B** — size reduction | `artifacts/regime_ml_v6_size/model.pkl` | Force size=1 | +$9,188 PnL, −$4,756 DD vs A-only baseline (@ thr=0.70) |
| **C** — BE-arm disable | `artifacts/regime_ml_v6_be/model.pkl` | Skip BE move-to-entry | +$2,453 PnL, −$1,185 DD vs A-only baseline (@ thr=0.60) |

All three are HGB-only. **No LightGBM is pickled** into any shipped
artifact (OpenMP conflicts with the bot's asyncio+torch stack caused
SIGSEGV-11 crashes in an early deployment; HGB is pure sklearn/Cython
and survives).

Training data: `es_master_outrights.parquet` (minute OHLCV bars, ES
outrights 2011 → 2026-04-20). Dominant contract per day auto-selected
by volume.

## Prerequisites

- Python 3.10+
- `numpy`, `pandas`, `scikit-learn`, `pyarrow`
- `es_master_outrights.parquet` present at repo root

No LightGBM / xgboost / catboost. No GPU.

## One-command reproduction — all three models

```bash
# 1) Model A first (B and C depend on A's artifact being present)
python3 scripts/regime_ml/train_model_a.py

# 2) Model B (reads artifacts/regime_ml_v5_brackets/model.pkl)
python3 scripts/regime_ml/train_model_b.py

# 3) Model C (reads same Model A artifact)
python3 scripts/regime_ml/train_model_c.py

# 4) Verify the shipped state still clears gates
python3 scripts/regime_ml/diagnose.py
```

Defaults produce bit-similar artifacts to the currently-shipped versions
(HGB is seeded to 42; HistGradientBoostingClassifier is deterministic
given the same seed + sklearn version).

## Custom training windows

Each trainer accepts the same four window flags:

```bash
python3 scripts/regime_ml/train_model_a.py \
    --start 2024-07-01 \
    --end   2026-04-20 \
    --holdout-start 2026-01-27 \
    --holdout-end   2026-04-20 \
    --out-dir artifacts/regime_ml_v5_brackets \
    --seed 42
```

`[--start, --holdout-start)` is the training window. `[--holdout-start,
--holdout-end]` is the OOS holdout where ship gates are evaluated. The
trainer refuses to write a model if gates fail — use `--force` to
override (DEBUG only, do not ship).

## Ship gates (evaluated at end of training)

### Model A
1. OOS PnL ≥ rule baseline + $500 (rule = `vol_bp_120 < 1.5 → scalp`)
2. OOS MaxDD ≤ 110% of rule MaxDD
3. Prediction rate in [10%, 90%] (non-degenerate)

### Model B (A-conditional)
Combined OOS (A=ML + B=ML + C=rule) must beat
(A=ML + B=rule + C=rule) on BOTH PnL AND DD. No free DD improvement
at PnL cost.

### Model C (A-conditional)
Combined OOS (A=ML + B=rule + C=ML) must beat
(A=ML + B=rule + C=rule) on BOTH PnL AND DD.

## Features the models expect at inference

- **Model A**: 40 columns listed in `_common.FEATURE_COLS_40`. Built live
  by `regime_classifier.RegimeClassifier.build_ml_feature_snapshot()`.
- **Model B, C**: 40 columns + `a_pred_scalp` (Model A's binary
  prediction). The live bot chains: build 40 features → query A →
  append `a_pred_scalp` → query B/C. See
  `regime_classifier._features_with_a_pred()`.

The bot caches 520 bars of OHLCV in `RegimeClassifier._ml_{o,h,l,c,v}`
deques (widest lookback is vol_bp_480). Pre-warmed at startup from
`master_df.tail(520)`. If history is < 480 bars, the feature builder
returns `None` and the decoupled actions fall through to rule.

## Rollback — disable any model without touching code

Environment flags in `launch_filterless_live.py` (or set at bot-launch time):

```bash
export JULIE_REGIME_ML_BRACKETS=0   # Model A → rule
export JULIE_REGIME_ML_SIZE=0       # Model B → rule
export JULIE_REGIME_ML_BE=0         # Model C → rule
```

All three default to `1` (ML active) in the launcher. Setting all three
to `0` restores pre-v5 rule-based behavior exactly (i.e. what
`apply_dead_tape_brackets` did unconditionally: rewrite to 3/5 + size=1
+ BE-off when `vol_bp < 1.5`).

## Shipped commit reference

Model A ship commit: `09942e9` — 2026-04-24
Model A HGB-only hotfix: `4c7ceba` — 2026-04-24
Models B + C v6 ship: `095b4b7` — 2026-04-24
Reproducible trainer kit (this README): see git log for current branch.

## Historical / killed attempts (preserved in `scripts/`)

These scripts trained earlier versions that were killed on their
respective gates. Kept for audit trail; do not use them to train
production artifacts.

- `scripts/ml_regime_classifier.py` — v1 supervised-on-rules
- `scripts/ml_regime_classifier_v2.py` — v2 outcome-labeled per-bar
- `scripts/ml_regime_classifier_v3.py` — v3 confidence-hybrid
- `scripts/ml_regime_classifier_v4.py` — v4 stacked improvements
- `scripts/ml_regime_v4_rule_agreement_check.py` — v4 strict-gate verify
- `scripts/ml_regime_v5_three_models.py` — v5 original 3-model trainer
  (used LightGBM + HGB ensemble; produced current shipped Model A via
  HGB splice — retained because v6's v5-import chain is historical)
- `scripts/ml_regime_v5_diagnose.py` — v5 post-mortem diagnostic
- `scripts/ml_regime_v6_conditional.py` — v6 B+C trainer that produced
  currently-shipped B+C artifacts

The `scripts/regime_ml/` kit (this directory) is the clean-slate
reproducible replacement — same features, same labels, same gates,
no LightGBM, full CLI, identical artifacts.
