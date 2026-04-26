#!/usr/bin/env bash
# orchestrate_v18.sh — autonomous V18 Kronos pipeline.
#
# Sequence (each step's failure is logged and the next step still runs):
#   1. Wait for Phase B-DE3 process (PID arg, default 54206) to finish.
#   2. python3 tools/v18_train.py --target de3
#   3. python3 tools/kronos_phase_b_extract_ra.py
#   4. python3 tools/v18_train.py --target ra
#   5. python3 tools/kronos_phase_b_extract_af.py
#   6. python3 tools/v18_train.py --target af
#   7. python3 tools/v18_train.py --target final_report
#
# Status updates are appended to /tmp/v18_orchestrate_status.txt.
# Each step's stdout/stderr is captured in /tmp/v18_orchestrate_<step>.log.

set -u  # fail on undefined vars (but NOT on errors — we WANT to continue)

ROOT="/Users/wes/Downloads/JULIE001"
cd "$ROOT" || exit 1

# Activate Kronos venv (sets PATH so python3 is the venv python)
if [ -f "$ROOT/.kronos_venv/bin/activate" ]; then
    # shellcheck source=/dev/null
    source "$ROOT/.kronos_venv/bin/activate"
fi
PY="$ROOT/.kronos_venv/bin/python"
if [ ! -x "$PY" ]; then
    PY="python3"
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

PHASE_B_PID="${1:-54206}"
STATUS="/tmp/v18_orchestrate_status.txt"
LOG_DIR="/tmp"

ts() { date '+%Y-%m-%dT%H:%M:%S%z'; }
status() {
    local msg="$1"
    echo "$(ts)  $msg" | tee -a "$STATUS"
}

status "START orchestrator pid=$$ phase_b_pid=$PHASE_B_PID"
status "PY=$PY"

# Step 1: wait for Phase B-DE3 to finish
status "STEP1 waiting for Phase B-DE3 PID=$PHASE_B_PID"
while kill -0 "$PHASE_B_PID" 2>/dev/null; do
    sleep 60
done
status "STEP1 done — Phase B-DE3 process exited"

FAILURES=()

# Helper: run a command, capture stderr+stdout, never abort
run_step() {
    local name="$1"; shift
    local logf="$LOG_DIR/v18_step_${name}.log"
    status "STEP ${name} STARTED  log=$logf  cmd=$*"
    set +e
    "$@" > "$logf" 2>&1
    local rc=$?
    set -e
    if [ "$rc" -eq 0 ]; then
        status "STEP ${name} DONE rc=0"
    else
        status "STEP ${name} FAILED rc=$rc (continuing)"
        FAILURES+=("${name}(rc=$rc)")
    fi
    return $rc
}

# Disable -e so step failures do not abort orchestrator
set +e

# Step 2: V18-DE3 train
run_step "v18_train_de3"   "$PY" "$ROOT/tools/v18_train.py" --target de3

# Step 3: Phase B-RA extract
run_step "phase_b_ra"      "$PY" "$ROOT/tools/kronos_phase_b_extract_ra.py" --resume

# Step 4: V18-RA train
run_step "v18_train_ra"    "$PY" "$ROOT/tools/v18_train.py" --target ra

# Step 5: Phase B-AF extract
run_step "phase_b_af"      "$PY" "$ROOT/tools/kronos_phase_b_extract_af.py" --resume

# Step 6: V18-AF train
run_step "v18_train_af"    "$PY" "$ROOT/tools/v18_train.py" --target af

# Step 7: final unified report
run_step "v18_final_report" "$PY" "$ROOT/tools/v18_train.py" --target final_report

if [ ${#FAILURES[@]} -eq 0 ]; then
    status "ALL_DONE"
else
    status "PARTIAL_FAILURE: ${FAILURES[*]}"
fi
