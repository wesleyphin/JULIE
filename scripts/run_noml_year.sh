#!/usr/bin/env bash
# Orchestrator: run NO-ML Kalshi-only replays for Mar 2025 → Apr 2026.
# Parallelism: 3 concurrent month-replays at a time, using GNU parallel-style
# job control with `wait -n` (no GNU parallel dependency).

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

PARALLELISM=3
ORCH_LOG="logs/noml_year_orchestrator.log"
mkdir -p logs
{
    echo "================================================================"
    date
    echo "Starting 14-month NO-ML Kalshi-only sweep, parallelism=$PARALLELISM"
    echo "================================================================"
} | tee -a "$ORCH_LOG"

active=0
declare -a pids

for entry in "${MONTHS_AND_SYMBOLS[@]}"; do
    month="${entry% *}"
    sym="${entry#* }"
    while [ "$active" -ge "$PARALLELISM" ]; do
        wait -n
        active=$((active - 1))
    done
    {
        echo "[$(date +%H:%M:%S)] LAUNCH $month ($sym)"
        if bash scripts/run_noml_kalshi_month.sh "$month" "$sym"; then
            echo "[$(date +%H:%M:%S)] DONE $month"
        else
            echo "[$(date +%H:%M:%S)] FAIL $month"
        fi
    } >> "$ORCH_LOG" 2>&1 &
    pids+=($!)
    active=$((active + 1))
    echo "  queued $month ($sym) — active=$active"
done

# Wait for all remaining jobs
wait
echo "[$(date +%H:%M:%S)] ALL 14 months complete" | tee -a "$ORCH_LOG"
ls -lh artifacts/kalshi_training_v8_*.parquet 2>&1 | tee -a "$ORCH_LOG"
