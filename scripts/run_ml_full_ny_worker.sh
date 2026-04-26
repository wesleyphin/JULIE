#!/usr/bin/env bash
# Single ml_full_ny worker — runs assigned months sequentially with the
# ml_full_ny config (NY-only + every ML overlay + rule blockers + AF loaded).
# Usage: ./run_ml_full_ny_worker.sh WORKER_ID "MONTH_1 SYMBOL_1" "MONTH_2 SYMBOL_2" ...
# WORKER_ID is e.g. workerA / workerB / workerC / workerD.
# Each (month, symbol) pair must be a single quoted argument.

set -uo pipefail
cd /Users/wes/Downloads/JULIE001
WORKER="$1"
shift

ORCH_LOG="logs/4config_ml_full_ny_${WORKER}.log"
mkdir -p logs
{
    echo "================================================================"
    date
    echo "ML_FULL_NY worker=$WORKER · ${#@} months assigned"
    echo "================================================================"
} | tee -a "$ORCH_LOG"

START_TIME=$(date +%s)

is_done() {
    local month="$1"
    local replay_dir="backtest_reports/ml_full_ny_${month//-/_}"
    [ -f "$replay_dir/closed_trades.json" ] && [ -f "$replay_dir/live_replay_summary.json" ]
}

for entry in "$@"; do
    month="${entry% *}"
    sym="${entry#* }"
    if is_done "$month"; then
        echo "[$(date +%H:%M:%S)] SKIP $month (already done)" | tee -a "$ORCH_LOG"
        continue
    fi
    echo "[$(date +%H:%M:%S)] $WORKER LAUNCH ml_full_ny/$month ($sym)" | tee -a "$ORCH_LOG"
    if bash scripts/run_4config_month.sh ml_full_ny "$month" "$sym" >> "$ORCH_LOG" 2>&1; then
        echo "[$(date +%H:%M:%S)] $WORKER DONE ml_full_ny/$month" | tee -a "$ORCH_LOG"
    else
        echo "[$(date +%H:%M:%S)] $WORKER FAIL ml_full_ny/$month" | tee -a "$ORCH_LOG"
    fi
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINS=$(((ELAPSED % 3600) / 60))
echo "[$(date +%H:%M:%S)] $WORKER ALL months complete — elapsed ${HOURS}h${MINS}m" | tee -a "$ORCH_LOG"
