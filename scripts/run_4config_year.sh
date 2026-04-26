#!/usr/bin/env bash
# Run 4 configs × 14 months. CONFIG order: baseline, rules, mlstack (then later v8).
# Memory-safe: 2 concurrent replays max across ALL configs.
# Pass single arg = config to run: baseline | rules | mlstack | v8

set -uo pipefail
cd /Users/wes/Downloads/JULIE001
CONFIG="${1:-baseline}"

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

PARALLELISM=1
ORCH_LOG="logs/4config_${CONFIG}_orchestrator.log"
mkdir -p logs
{
    echo "================================================================"
    date
    echo "Running config=$CONFIG · 14 months · parallelism=$PARALLELISM"
    echo "================================================================"
} | tee -a "$ORCH_LOG"

is_done() {
    local month="$1"
    local replay_dir="backtest_reports/${CONFIG}_${month//-/_}"
    if [ -f "$replay_dir/closed_trades.json" ] && [ -f "$replay_dir/live_replay_summary.json" ]; then
        return 0
    fi
    return 1
}

active_count() {
    pgrep -f "run_full_live_replay_parquet.*--run-name ${CONFIG}_" 2>/dev/null | wc -l | tr -d ' '
}

run_one() {
    local month="$1"; local sym="$2"
    if is_done "$month"; then
        echo "[$(date +%H:%M:%S)] $CONFIG/$month already done, skip" >> "$ORCH_LOG"
        return 0
    fi
    echo "[$(date +%H:%M:%S)] LAUNCH $CONFIG/$month ($sym)" >> "$ORCH_LOG"
    if bash scripts/run_4config_month.sh "$CONFIG" "$month" "$sym" >> "$ORCH_LOG" 2>&1; then
        echo "[$(date +%H:%M:%S)] DONE $CONFIG/$month" >> "$ORCH_LOG"
    else
        echo "[$(date +%H:%M:%S)] FAIL $CONFIG/$month" >> "$ORCH_LOG"
    fi
}

for entry in "${MONTHS_AND_SYMBOLS[@]}"; do
    month="${entry% *}"; sym="${entry#* }"
    while [ "$(active_count)" -ge "$PARALLELISM" ]; do
        sleep 30
    done
    run_one "$month" "$sym" &
    echo "  queued $CONFIG/$month  active=$(active_count)" | tee -a "$ORCH_LOG"
    sleep 15
done
wait
echo "[$(date +%H:%M:%S)] $CONFIG ALL months done" | tee -a "$ORCH_LOG"
