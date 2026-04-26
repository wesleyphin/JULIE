#!/usr/bin/env bash
# Watch for orchestrator completion, then run v8 retrain.
# Polls every 5 minutes. Exits when the orchestrator log shows "ALL 14 months complete".
set -uo pipefail
cd /Users/wes/Downloads/JULIE001
ORCH_LOG="logs/noml_year_v2_orchestrator.log"
WAIT_LOG="logs/wait_and_train.log"
mkdir -p logs

echo "[$(date +%H:%M:%S)] watching $ORCH_LOG for completion" | tee -a "$WAIT_LOG"

while true; do
    if grep -q "ALL 14 months complete" "$ORCH_LOG" 2>/dev/null; then
        echo "[$(date +%H:%M:%S)] orchestrator complete — launching v8 retrain" | tee -a "$WAIT_LOG"
        break
    fi
    sleep 300
done

# Run the trainer
python3 scripts/regime_ml/train_kalshi_v8.py --horizon 60 --half-life 120 2>&1 | tee -a "$WAIT_LOG"
TRAIN_RC=${PIPESTATUS[0]}

if [ "$TRAIN_RC" -eq 0 ]; then
    echo "[$(date +%H:%M:%S)] v8 SHIPPED — see artifacts/regime_ml_kalshi_v8/" | tee -a "$WAIT_LOG"
else
    echo "[$(date +%H:%M:%S)] v8 KILLED — see logs above" | tee -a "$WAIT_LOG"
fi
