#!/usr/bin/env bash
# Run a NO-ML-stack Kalshi-only replay for a single month.
# Usage: ./run_noml_kalshi_month.sh YYYY-MM TARGET_SYMBOL
# Example: ./run_noml_kalshi_month.sh 2025-09 ES.U5
set -euo pipefail

MONTH="$1"          # e.g. 2025-09
TARGET_SYMBOL="$2"  # e.g. ES.U5

YEAR="${MONTH%-*}"
MM="${MONTH#*-}"
NEXT=$(python3 -c "
import datetime
d = datetime.date(int('$YEAR'), int('$MM'), 1)
m = d.month + 1
y = d.year + (1 if m > 12 else 0)
m = m if m <= 12 else 1
print(f'{y}-{m:02d}')")
START="$MONTH-01 00:00"
END="$NEXT-01 00:00"

RUN_NAME="kalshi_only_noml_${MONTH//-/_}"
REPORT_DIR="backtest_reports/$RUN_NAME"
LOG_FILE="$REPORT_DIR/topstep_live_bot.log"
mkdir -p "$REPORT_DIR"

echo "[$MONTH] start=$START end=$END  symbol=$TARGET_SYMBOL  outdir=$REPORT_DIR"

cd /Users/wes/Downloads/JULIE001

JULIE_FILTERLESS_ONLY=1 \
JULIE_DISABLE_STRATEGY_FILTERS=1 \
JULIE_BYPASS_SAMESIDE=1 \
JULIE_CB=0 \
JULIE_DLB=0 \
JULIE_DD_SCALE=0 \
JULIE_LOSS_FACTOR_GUARD=0 \
JULIE_REGIME_ADAPTIVE_CB=0 \
JULIE_REGIME_CLASSIFIER=0 \
JULIE_SIGNAL_GATE_2025=0 \
JULIE_FILTERG_PER_CELL_ACTIVE=0 \
JULIE_CASCADE_BLOCKER_ACTIVE=0 \
JULIE_ANTI_FLIP_BLOCKER_ACTIVE=0 \
JULIE_TRIATHLON_ACTIVE=0 \
JULIE_SAMESIDE_ML=0 \
JULIE_REGIME_ML_BRACKETS=0 \
JULIE_REGIME_ML_SIZE=0 \
JULIE_REGIME_ML_BE=0 \
JULIE_KALSHI_DE3_MODE=on \
JULIE_KALSHI_CONTINUATION_TP=1 \
JULIE_ML_LFO_ACTIVE=0 \
JULIE_ML_PCT_ACTIVE=0 \
JULIE_ML_PIVOT_TRAIL_ACTIVE=0 \
JULIE_ML_KALSHI_ACTIVE=0 \
JULIE_ML_KALSHI_TP_ACTIVE=0 \
JULIE_ML_RL_MGMT_ACTIVE=0 \
JULIE_RL_REGIME_GATE_ACTIVE=0 \
JULIE_KALSHI_CM_GATE_V2_ACTIVE=0 \
JULIE_LFG_CHART_VETO=0 \
JULIE_FREEZE_AUTO_CONFIG=1 \
JULIE_AILOOP_APPLY=0 \
python3 tools/run_full_live_replay_parquet.py \
    --start "$START" --end "$END" \
    --bars-parquet es_master_outrights.parquet \
    --target-symbol "$TARGET_SYMBOL" \
    --report-dir backtest_reports \
    --run-name "$RUN_NAME" \
    --account-id 99999 \
    --initial-balance 50000 > "$REPORT_DIR/replay_stdout.log" 2>&1

# Extract training set from this month's log
python3 scripts/extract_kalshi_training_set.py \
    --log "$LOG_FILE" \
    --start "$YEAR-$MM-01" \
    --end "$NEXT-01" \
    --out "artifacts/kalshi_training_v8_${MONTH//-/_}.parquet" >> "$REPORT_DIR/replay_stdout.log" 2>&1

echo "[$MONTH] done"
