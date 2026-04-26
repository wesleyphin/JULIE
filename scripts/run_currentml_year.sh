#!/usr/bin/env bash
# Orchestrator: serial 14-month CURRENT-ML-STACK backtest.
# Production-realistic config. Auto-chains into v8 retrain at the end.

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

ORCH_LOG="logs/currentml_year_orchestrator.log"
mkdir -p logs
{
    echo "================================================================"
    date
    echo "Starting 14-month CURRENT-ML-STACK backtest — SERIAL"
    echo "All ML overlays ON · all production blockers ON · SameSide ML ON"
    echo "================================================================"
} | tee -a "$ORCH_LOG"

START_TIME=$(date +%s)

for entry in "${MONTHS_AND_SYMBOLS[@]}"; do
    month="${entry% *}"
    sym="${entry#* }"
    train_pq="artifacts/kalshi_training_currentml_${month//-/_}.parquet"
    if [ -s "$train_pq" ]; then
        rows=$(python3 -c "import pandas as pd; print(len(pd.read_parquet('$train_pq')))" 2>/dev/null || echo 0)
        if [ "$rows" -gt "10" ]; then
            echo "[$(date +%H:%M:%S)] SKIP $month — $train_pq has $rows rows already" | tee -a "$ORCH_LOG"
            continue
        fi
    fi
    echo "[$(date +%H:%M:%S)] LAUNCH $month ($sym)" | tee -a "$ORCH_LOG"
    if bash scripts/run_currentml_kalshi_month.sh "$month" "$sym" >> "$ORCH_LOG" 2>&1; then
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
    ls -lh artifacts/kalshi_training_currentml_*.parquet 2>&1 | head
    python3 -c "
import pandas as pd
import glob
import json
files = sorted(glob.glob('artifacts/kalshi_training_currentml_*.parquet'))
total_rows = 0
total_pnl = 0.0
total_trades = 0
print('Per-month summary:')
for f in files:
    try:
        df = pd.read_parquet(f)
        total_rows += len(df)
        # Trades come from closed_trades.json in the report dir
        month = f.split('_')[-2:]
        month_str = '_'.join(month).replace('.parquet','')
        ct_path = f'backtest_reports/currentml_{month_str}/closed_trades.json'
        try:
            trades = json.load(open(ct_path))
            n = len(trades)
            pnl = sum(t.get('pnl_dollars',0) for t in trades)
            total_pnl += pnl
            total_trades += n
            print(f'  {f.split(\"/\")[-1]}: {len(df)} kalshi events, {n} trades, \${pnl:+,.2f}')
        except Exception as e:
            print(f'  {f.split(\"/\")[-1]}: {len(df)} kalshi events, no trades file')
    except Exception as e:
        print(f'  {f}: error {e}')
print(f'\\nTOTAL: {total_rows:,} kalshi events  {total_trades:,} trades  \${total_pnl:+,.2f} PnL')
"
} | tee -a "$ORCH_LOG"

# Auto-chain into v8 retrain
echo "" | tee -a "$ORCH_LOG"
echo "================================================================" | tee -a "$ORCH_LOG"
echo "[$(date +%H:%M:%S)] Starting v8 retrain on combined currentml dataset" | tee -a "$ORCH_LOG"
echo "================================================================" | tee -a "$ORCH_LOG"
python3 scripts/regime_ml/train_kalshi_v8.py \
    --horizon 60 --half-life 120 \
    --input-glob "artifacts/kalshi_training_currentml_*.parquet" \
    --out-dir artifacts/regime_ml_kalshi_v8 2>&1 | tee -a "$ORCH_LOG"
TRAIN_RC=${PIPESTATUS[0]}
if [ "$TRAIN_RC" -eq 0 ]; then
    echo "[$(date +%H:%M:%S)] v8 SHIPPED — model.pkl in artifacts/regime_ml_kalshi_v8/" | tee -a "$ORCH_LOG"
else
    echo "[$(date +%H:%M:%S)] v8 KILLED — see sweep summary" | tee -a "$ORCH_LOG"
fi
