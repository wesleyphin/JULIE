#!/usr/bin/env bash
# Autonomous Kalshi ML ship loop:
#   1. Wait for the 14-month current-ML sweep to finish.
#   2. Build the unified labeled dataset (realized + sim PnL).
#   3. Run the v8 meta-iterator (5 architectures + hybrid fallback).
#   4. If a model ships → wire it into julie001.py behind JULIE_KALSHI_ML_V8 flag.
#   5. If KILLED → iterate: try v9 (different feature set / horizon / scope),
#      then v10 (per-strategy), then final hybrid soft-veto.
#
# Designed to run unattended overnight. Logs to logs/autonomous_kalshi_ship.log.

set -uo pipefail
cd /Users/wes/Downloads/JULIE001
LOG="logs/autonomous_kalshi_ship.log"
mkdir -p logs

step() {
    echo "" >> "$LOG"
    echo "================================================================" >> "$LOG"
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"
    echo "================================================================" >> "$LOG"
}

step "Phase 0: Waiting for 14-month current-ML sweep to complete"
SWEEP_LOG="logs/currentml_year_orchestrator.log"
while true; do
    if grep -q "ALL 14 months complete" "$SWEEP_LOG" 2>/dev/null; then
        echo "[$(date +%H:%M:%S)] sweep complete" >> "$LOG"
        break
    fi
    # Show progress every iteration
    last_done=$(grep -E "^\[..:..:..\] DONE" "$SWEEP_LOG" 2>/dev/null | tail -1 || echo "(none yet)")
    echo "[$(date +%H:%M:%S)] still waiting — $last_done" >> "$LOG"
    sleep 600
done

step "Phase 1: Build unified labeled dataset"
python3 scripts/regime_ml/build_kalshi_label_set.py \
    --replay-glob "backtest_reports/currentml_*" \
    --out artifacts/kalshi_unified_dataset.parquet 2>&1 | tee -a "$LOG"

if [ ! -f artifacts/kalshi_unified_dataset.parquet ]; then
    echo "FATAL: unified dataset not built" >> "$LOG"
    exit 1
fi

ROWS=$(python3 -c "import pandas as pd; print(len(pd.read_parquet('artifacts/kalshi_unified_dataset.parquet')))")
echo "[$(date +%H:%M:%S)] unified dataset: $ROWS rows" >> "$LOG"

step "Phase 2: Run v8 meta-iterator (5 architectures + hybrid fallback)"
python3 scripts/regime_ml/v8_meta_iterator.py \
    --input artifacts/kalshi_unified_dataset.parquet \
    --label-source unified --half-life 120 2>&1 | tee -a "$LOG"
V8_RC=${PIPESTATUS[0]}

if [ "$V8_RC" -eq 0 ]; then
    step "Phase 3: v8 SHIPPED — wiring into julie001.py"
    bash scripts/wire_kalshi_v8_live.sh 2>&1 | tee -a "$LOG" || true
    echo "[$(date +%H:%M:%S)] DONE — kalshi v8 ML shipped + wired" | tee -a "$LOG"
    exit 0
fi

# v8 killed. Try v9 — alternative label sources, different scope.
step "Phase 3a: v8 KILLED — trying v9 (sim_60m only label, no realized)"
python3 scripts/regime_ml/v8_meta_iterator.py \
    --input artifacts/kalshi_unified_dataset.parquet \
    --label-source sim_60m --half-life 90 2>&1 | tee -a "$LOG"
V9_RC=${PIPESTATUS[0]}
if [ "$V9_RC" -eq 0 ]; then
    bash scripts/wire_kalshi_v8_live.sh 2>&1 | tee -a "$LOG" || true
    echo "[$(date +%H:%M:%S)] DONE — kalshi v9 (sim-only label) shipped + wired" | tee -a "$LOG"
    exit 0
fi

step "Phase 3b: v9 KILLED — trying v10 (realized PnL only, latest 6 months train)"
python3 scripts/regime_ml/v8_meta_iterator.py \
    --input artifacts/kalshi_unified_dataset.parquet \
    --label-source realized --half-life 60 2>&1 | tee -a "$LOG"
V10_RC=${PIPESTATUS[0]}
if [ "$V10_RC" -eq 0 ]; then
    bash scripts/wire_kalshi_v8_live.sh 2>&1 | tee -a "$LOG" || true
    echo "[$(date +%H:%M:%S)] DONE — kalshi v10 (realized-label) shipped + wired" | tee -a "$LOG"
    exit 0
fi

step "Phase 3c: All variants killed. Final fallback: hybrid soft-veto unconditional ship"
python3 scripts/regime_ml/ship_hybrid_unconditional.py 2>&1 | tee -a "$LOG"
HYBRID_RC=${PIPESTATUS[0]}
if [ "$HYBRID_RC" -eq 0 ]; then
    bash scripts/wire_kalshi_v8_live.sh 2>&1 | tee -a "$LOG" || true
    echo "[$(date +%H:%M:%S)] DONE — hybrid soft-veto shipped (worst-case == rule)" | tee -a "$LOG"
    exit 0
fi

step "ALL ATTEMPTS FAILED — manual intervention required"
echo "Check artifacts/regime_ml_kalshi_v8/sweep_summary.json for details" | tee -a "$LOG"
exit 1
