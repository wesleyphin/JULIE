#!/usr/bin/env bash
# Orchestrator: PARALLEL 4-at-a-time 14-month CURRENT-ML-STACK backtest.
# Skips months that already have a populated training parquet (so the
# in-flight Mar 2025 run gets picked up when it finishes its extract step).

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

PARALLELISM=4
ORCH_LOG="logs/currentml_year_orchestrator.log"
mkdir -p logs
{
    echo "================================================================"
    date
    echo "RESUMING current-ML-stack backtest — PARALLEL=$PARALLELISM"
    echo "(skips months that already produced a training parquet)"
    echo "================================================================"
} | tee -a "$ORCH_LOG"

START_TIME=$(date +%s)

# Track active jobs in an array
declare -a active_pids
declare -a active_months

run_one() {
    local month="$1"; local sym="$2"
    local train_pq="artifacts/kalshi_training_currentml_${month//-/_}.parquet"
    local replay_dir="backtest_reports/currentml_${month//-/_}"
    # If already done (parquet has rows OR closed_trades.json exists), skip
    if [ -s "$train_pq" ]; then
        local rows
        rows=$(python3 -c "import pandas as pd; print(len(pd.read_parquet('$train_pq')))" 2>/dev/null || echo 0)
        if [ "$rows" -gt "10" ]; then
            echo "[$(date +%H:%M:%S)] SKIP $month (parquet has $rows rows)" >> "$ORCH_LOG"
            return 0
        fi
    fi
    if [ -f "$replay_dir/closed_trades.json" ] && [ -f "$replay_dir/topstep_live_bot.log" ]; then
        local size
        size=$(stat -f %z "$replay_dir/topstep_live_bot.log" 2>/dev/null || echo 0)
        if [ "$size" -gt "1000000" ]; then
            # replay log substantial — just run extract
            echo "[$(date +%H:%M:%S)] EXTRACT-ONLY $month (replay log already exists, $size bytes)" >> "$ORCH_LOG"
            local year_int="${month%-*}"
            local mm_int="${month#*-}"
            local next
            next=$(python3 -c "
import datetime
d = datetime.date(int('$year_int'), int('$mm_int'), 1)
m = d.month + 1; y = d.year + (1 if m > 12 else 0); m = m if m <= 12 else 1
print(f'{y}-{m:02d}')")
            python3 scripts/extract_kalshi_training_set.py \
                --log "$replay_dir/topstep_live_bot.log" \
                --start "$year_int-$mm_int-01" \
                --end "$next-01" \
                --out "$train_pq" >> "$ORCH_LOG" 2>&1
            return 0
        fi
    fi
    echo "[$(date +%H:%M:%S)] LAUNCH $month ($sym)" >> "$ORCH_LOG"
    if bash scripts/run_currentml_kalshi_month.sh "$month" "$sym" >> "$ORCH_LOG" 2>&1; then
        echo "[$(date +%H:%M:%S)] DONE $month" >> "$ORCH_LOG"
    else
        echo "[$(date +%H:%M:%S)] FAIL $month" >> "$ORCH_LOG"
    fi
}

# Job-pool with parallelism=4
launch_idx=0
for entry in "${MONTHS_AND_SYMBOLS[@]}"; do
    month="${entry% *}"
    sym="${entry#* }"
    # Wait if pool is full
    while [ "${#active_pids[@]}" -ge "$PARALLELISM" ]; do
        # Check each PID; remove dead ones
        new_pids=()
        new_months=()
        for i in "${!active_pids[@]}"; do
            pid="${active_pids[$i]}"
            mo="${active_months[$i]}"
            if kill -0 "$pid" 2>/dev/null; then
                new_pids+=("$pid")
                new_months+=("$mo")
            fi
        done
        active_pids=("${new_pids[@]}")
        active_months=("${new_months[@]}")
        if [ "${#active_pids[@]}" -ge "$PARALLELISM" ]; then
            sleep 30
        fi
    done
    run_one "$month" "$sym" &
    pid=$!
    active_pids+=("$pid")
    active_months+=("$month")
    echo "  queued $month ($sym) pid=$pid  active=${#active_pids[@]}" | tee -a "$ORCH_LOG"
    sleep 10  # stagger launches a bit so they don't all hit manifold-build at once
done

# Wait for all remaining
echo "[$(date +%H:%M:%S)] all queued, waiting for completion..." >> "$ORCH_LOG"
wait

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINS=$(((ELAPSED % 3600) / 60))
{
    echo ""
    echo "[$(date +%H:%M:%S)] ALL 14 months complete — elapsed ${HOURS}h${MINS}m"
    echo ""
    ls -lh artifacts/kalshi_training_currentml_*.parquet 2>&1 | head -20
} | tee -a "$ORCH_LOG"

# Auto-chain into v8 retrain
{
    echo ""
    echo "================================================================"
    echo "[$(date +%H:%M:%S)] Starting v8 retrain on combined currentml dataset"
    echo "================================================================"
} >> "$ORCH_LOG"
python3 scripts/regime_ml/build_kalshi_label_set.py \
    --replay-glob "backtest_reports/currentml_*" \
    --out artifacts/kalshi_unified_dataset.parquet >> "$ORCH_LOG" 2>&1
python3 scripts/regime_ml/v8_meta_iterator.py \
    --input artifacts/kalshi_unified_dataset.parquet \
    --label-source unified --half-life 120 >> "$ORCH_LOG" 2>&1
TRAIN_RC=$?

if [ "$TRAIN_RC" -eq 0 ]; then
    echo "[$(date +%H:%M:%S)] v8 SHIPPED — wiring live" >> "$ORCH_LOG"
    bash scripts/wire_kalshi_v8_live.sh >> "$ORCH_LOG" 2>&1
else
    echo "[$(date +%H:%M:%S)] v8 KILLED — trying ship_hybrid_unconditional" >> "$ORCH_LOG"
    python3 scripts/regime_ml/ship_hybrid_unconditional.py >> "$ORCH_LOG" 2>&1
    if [ -f artifacts/regime_ml_kalshi_v8/model.pkl ]; then
        bash scripts/wire_kalshi_v8_live.sh >> "$ORCH_LOG" 2>&1
    fi
fi
echo "[$(date +%H:%M:%S)] CHAIN COMPLETE" >> "$ORCH_LOG"
