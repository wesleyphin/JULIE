#!/bin/bash
# Wide-window retrain chain: 2020-01-01 → 2026-04-20 (5.5 years).
# Runs after manifold_base_wide.parquet exists.
set -e
cd /Users/wes/Downloads/JULIE001
OUT=artifacts/aetherflow_retrain_wide
BASE=$OUT/manifold_base_wide.parquet

echo "=== Shared model training — wide window 2020-01-01 → 2026-04-20 ==="
python3 train_aetherflow.py \
    --input es_master_outrights.parquet \
    --base-features $BASE \
    --reuse-cached-base \
    --start 2020-01-01 \
    --end   2026-04-20 \
    --out-dir $OUT \
    --model-file model_shared.pkl \
    --thresholds-file thresholds_shared.json \
    --metrics-file metrics_shared.json \
    --features-parquet features_shared.parquet \
    --workers 4 \
    --force-rebuild-features \
    2>&1 | tee $OUT/train_shared.log
echo "=== Shared training complete ==="

echo "=== Family-head dataset build (transition_burst) ==="
python3 tools/build_aetherflow_family_dataset.py \
    --input es_master_outrights.parquet \
    --base-features $BASE \
    --family transition_burst \
    --start 2020-01-01 \
    --end   2026-04-20 \
    --output-parquet $OUT/family_tb_features.parquet \
    --force-rebuild \
    2>&1 | tee $OUT/build_family_tb.log
echo "=== Family dataset complete ==="

echo "=== Family-head training (transition_burst) ==="
python3 train_aetherflow.py \
    --input es_master_outrights.parquet \
    --base-features $BASE \
    --reuse-features-parquet \
    --features-parquet family_tb_features.parquet \
    --start 2020-01-01 \
    --end   2026-04-20 \
    --out-dir $OUT \
    --model-file model_family_tb.pkl \
    --thresholds-file thresholds_family_tb.json \
    --metrics-file metrics_family_tb.json \
    --workers 4 \
    --min-fold-trades 10 \
    --min-val-trades 40 \
    2>&1 | tee $OUT/train_family_tb.log
echo "=== Family training complete ==="

echo "=== Splice bundle (Path B — shared + family head) ==="
python3 scripts/splice_aetherflow_retrained_bundle.py \
    --new-shared $OUT/model_shared.pkl \
    --new-thresholds $OUT/thresholds_shared.json \
    --new-conditional $OUT/model_family_tb.pkl \
    2>&1 | tee $OUT/splice.log

echo "=== Compare retrained vs deployed ==="
python3 scripts/aetherflow_compare_retrained_vs_deployed.py \
    --retrained model_aetherflow_deploy_2026full.pkl \
    --retrained-threshold aetherflow_thresholds_deploy_2026full.json \
    2>&1 | tee $OUT/compare.log
echo "=== All done ==="
