# DE3 v2 Config Schema

`CONFIG["DE3_VERSION"]`
- `v1` or `v2`
- Runtime selector used by `dynamic_signal_engine3.py`

`CONFIG["DE3_V2"]`
- `enabled`: bool
- `db_path`: string path to v2 DB JSON
- `mode`: `fixed_split` or `rolling`
- `train_end`: date string
- `valid_start`: date string
- `valid_end`: date string or null
- `purge_bars`: int embargo around split boundary

`DE3_V2.plateau`
- `enabled`: bool
- `min_neighbors`: int
- `neighbor_def`: `adjacent_grid`
- `min_plateau_score`: float

`DE3_V2.scoring`
- `lambda_std`: float
- `gamma_dd`: float
- `min_oos_trades`: int
- `min_profitable_blocks`: int
- `min_train_trades`: int

`DE3_V2.search_space`
- `thresholds`: list[float]
- `sl_list`: list[float]
- `rr_list`: list[float]
- `max_per_bucket`: int

`DE3_V2.rolling`
- `train_years`: int
- `valid_years`: int
- `step_years`: int

Execution:
```bash
python de3_v2_generator.py --source es_master.csv --out dynamic_engine3_strategies_v2.json
```

