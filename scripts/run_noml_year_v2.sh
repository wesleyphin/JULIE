#!/usr/bin/env bash
# Orchestrator v2: serialized 14-month NO-ML Kalshi-only replay.
# Memory-safe: runs ONE month at a time. Slower (~14h estimated) but no
# OOM risk. Skips months that already have a populated training parquet.

set -uo pipefail
cd /Users/wes/Downloads/JULIE001

MONTHS_AND_SYMBOLS=(
    "2025-03 ES.M5"
    "2025-04 ES.M5"
    "2025-05 ES.M5"
    "2025-06 ES.U5"
    "2025-07 ES.U5"
    "2025-08 ES.U5"
    "2025-09 ES.U5"
    "2025-10 ES.Z5"
    "2025-11 ES.Z5"
    "2025-12 ES.H6"
    "2026-01 ES.H6"
    "2026-02 ES.H6"
    "2026-03 ES.M6"
    "2026-04 ES.M6"
)

ORCH_LOG="logs/noml_year_v2_orchestrator.log"
mkdir -p logs
{
    echo "================================================================"
    date
    echo "Starting 14-month NO-ML Kalshi-only sweep — SERIAL (memory-safe)"
    echo "================================================================"
} | tee -a "$ORCH_LOG"

START_TIME=$(date +%s)

for entry in "${MONTHS_AND_SYMBOLS[@]}"; do
    month="${entry% *}"
    sym="${entry#* }"
    train_pq="artifacts/kalshi_training_v8_${month//-/_}.parquet"
    if [ -s "$train_pq" ]; then
        rows=$(python3 -c "import pandas as pd; print(len(pd.read_parquet('$train_pq')))" 2>/dev/null || echo 0)
        if [ "$rows" -gt "10" ]; then
            echo "[$(date +%H:%M:%S)] SKIP $month — $train_pq has $rows rows already" | tee -a "$ORCH_LOG"
            continue
        fi
    fi
    echo "[$(date +%H:%M:%S)] LAUNCH $month ($sym)" | tee -a "$ORCH_LOG"
    if bash scripts/run_noml_kalshi_month.sh "$month" "$sym" >> "$ORCH_LOG" 2>&1; then
        echo "[$(date +%H:%M:%S)] DONE $month" | tee -a "$ORCH_LOG"
    else
        echo "[$(date +%H:%M:%S)] FAIL $month — see $ORCH_LOG" | tee -a "$ORCH_LOG"
    fi
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINS=$(((ELAPSED % 3600) / 60))
{
    echo ""
    echo "[$(date +%H:%M:%S)] ALL 14 months complete — elapsed ${HOURS}h${MINS}m"
    echo ""
    ls -lh artifacts/kalshi_training_v8_*.parquet 2>&1
    python3 -c "
import pandas as pd
import glob
files = sorted(glob.glob('artifacts/kalshi_training_v8_*.parquet'))
total = 0
for f in files:
    try:
        df = pd.read_parquet(f)
        total += len(df)
        print(f'  {f}: {len(df)} rows')
    except: pass
print(f'TOTAL across all months: {total} rows')
"
} | tee -a "$ORCH_LOG"

# Chain into v8 retrain
echo "" | tee -a "$ORCH_LOG"
echo "================================================================" | tee -a "$ORCH_LOG"
echo "[$(date +%H:%M:%S)] Starting v8 retrain on combined dataset" | tee -a "$ORCH_LOG"
echo "================================================================" | tee -a "$ORCH_LOG"
python3 scripts/regime_ml/train_kalshi_v8.py --horizon 60 --half-life 120 2>&1 | tee -a "$ORCH_LOG"
TRAIN_RC=$?
if [ "$TRAIN_RC" -eq 0 ]; then
    echo "[$(date +%H:%M:%S)] v8 SHIPPED — model.pkl written to artifacts/regime_ml_kalshi_v8/" | tee -a "$ORCH_LOG"
else
    echo "[$(date +%H:%M:%S)] v8 KILLED — see logs above for sweep details" | tee -a "$ORCH_LOG"
fi
