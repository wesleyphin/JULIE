# Experimental 2011-2017 Mode (No Overwrite)

This repo now supports a safe experimental training mode that does not overwrite your existing full-data artifacts.

## Safety rule

- Experimental training always writes suffixed files.
- Default suffix: `_exp2011_2017`
- Example:
  - `model_ny_am.joblib` stays untouched
  - experimental output is `model_ny_am_exp2011_2017.joblib`

Even if you pass `--experimental-window` without `--artifact-suffix`, the trainer forces a non-empty suffix.

## Train experimental artifacts

Use `--experimental-window` on each trainer.

Examples:

```powershell
python ml_train_physics.py --csv es_master.parquet --experimental-window
python ml_train_physics_regime_loro.py --manifest regimes.json --experimental-window
python train_manifold_strategy.py --input es_master.parquet --experimental-window
python train_dynamic_engine3.py --csv ml_mes_et.csv --experimental-window
python train_de3_context_veto.py --dataset cache/de3_context_dataset --experimental-window
python train_all.py --csv es_master.parquet --experimental-window
```

Optional custom suffix:

```powershell
python ml_train_physics.py --csv es_master.parquet --experimental-window --artifact-suffix _exp_custom
```

## Swap runtime between full vs experimental artifacts

Open `config.py`, then set:

- Full artifacts (default):
  - `CONFIG["EXPERIMENTAL_TRAINING"]["enabled_runtime"] = False`
- Experimental artifacts:
  - `CONFIG["EXPERIMENTAL_TRAINING"]["enabled_runtime"] = True`

That one toggle remaps all configured model/JSON artifact paths to the suffixed version.
