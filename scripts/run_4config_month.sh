#!/usr/bin/env bash
# Run a single month with one of 4 config profiles.
# Usage: ./run_4config_month.sh CONFIG MONTH SYMBOL
#   CONFIG: baseline | rules | mlstack | v8
#   MONTH:  YYYY-MM
#   SYMBOL: e.g. ES.M5
set -euo pipefail
CONFIG="$1"; MONTH="$2"; SYMBOL="$3"
cd /Users/wes/Downloads/JULIE001
YEAR="${MONTH%-*}"; MM="${MONTH#*-}"
NEXT=$(python3 -c "
import datetime
d = datetime.date(int('$YEAR'), int('$MM'), 1)
m = d.month + 1; y = d.year + (1 if m > 12 else 0); m = m if m <= 12 else 1
print(f'{y}-{m:02d}')")
START="$MONTH-01 00:00"
END="$NEXT-01 00:00"

RUN_NAME="${CONFIG}_${MONTH//-/_}"
REPORT_DIR="backtest_reports/$RUN_NAME"
mkdir -p "$REPORT_DIR"

# Common env (always set)
export JULIE_FILTERLESS_ONLY=1
export JULIE_DISABLE_STRATEGY_FILTERS=1
export JULIE_FREEZE_AUTO_CONFIG=1
export JULIE_AILOOP_APPLY=0
export JULIE_TRIATHLON_RETRAIN_QUEUE=0
export JULIE_LFG_CHART_VETO=0
export JULIE_KALSHI_ML_V8=0  # default off, only v8 config flips on
# SPEED: 4 workers × 2 threads chosen empirically. Per-worker rate is ~7 b/s
# regardless of thread count (Python GIL-bound, not BLAS-bound) so we
# maximize parallel streams. 4 workers split a 4/3/3/3 month load — beats
# 3 workers × 4 threads on wall-clock (~5h 36m vs ~7h 14m bottleneck).
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
export VECLIB_MAXIMUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
# SPEED: skip the news calendar HTTP fetch (it 429s every time and retries)
export JULIE_DISABLE_NEWS_CALENDAR=1
# SPEED: skip the AF runtime cache periodic flush (writes parquet every 30s
# during replay — pure I/O overhead with no effect on backtest results)
export JULIE_DISABLE_AF_RUNTIME_CACHE=1

case "$CONFIG" in
  baseline)
    # No overlays — bot fires every signal that passes basic safety
    export JULIE_CB=0 JULIE_DLB=0 JULIE_DD_SCALE=0 JULIE_LOSS_FACTOR_GUARD=0
    export JULIE_REGIME_ADAPTIVE_CB=0 JULIE_REGIME_CLASSIFIER=1
    export JULIE_SIGNAL_GATE_2025=0 JULIE_FILTERG_PER_CELL_ACTIVE=0
    export JULIE_CASCADE_BLOCKER_ACTIVE=0 JULIE_ANTI_FLIP_BLOCKER_ACTIVE=0
    export JULIE_TRIATHLON_ACTIVE=0 JULIE_SAMESIDE_ML=0
    export JULIE_BYPASS_SAMESIDE=1   # also disable hard-rule same-side
    export JULIE_REGIME_ML_BRACKETS=0 JULIE_REGIME_ML_SIZE=0 JULIE_REGIME_ML_BE=0
    export JULIE_KALSHI_DE3_MODE=off JULIE_KALSHI_CONTINUATION_TP=0
    export JULIE_ML_LFO_ACTIVE=0 JULIE_ML_PCT_ACTIVE=0 JULIE_ML_PIVOT_TRAIL_ACTIVE=0
    export JULIE_ML_KALSHI_ACTIVE=0 JULIE_ML_KALSHI_TP_ACTIVE=0
    export JULIE_ML_RL_MGMT_ACTIVE=0 JULIE_RL_REGIME_GATE_ACTIVE=0
    export JULIE_KALSHI_CM_GATE_V2_ACTIVE=0
    # SPEED: kill PPO SHADOW per-bar inference (~30-40% of wall-time
    # for nothing — RL is irrelevant to baseline)
    export JULIE_DISABLE_RL_SHADOW=1
    # SPEED: disable AetherFlow strategy (manifold + routed model
    # inference per bar) — irrelevant to baseline since baseline = "what
    # does DE3 do unfiltered" and AF brings nothing to a no-overlay test
    export JULIE_DISABLE_STRATEGIES=aetherflow,regimeadaptive
    ;;
  rules)
    # Rule-based overlays only — Kalshi rule, CB, DLB, news, etc. NO ML overlays
    export JULIE_CB=1 JULIE_DLB=1 JULIE_DD_SCALE=1 JULIE_LOSS_FACTOR_GUARD=1
    export JULIE_REGIME_ADAPTIVE_CB=1 JULIE_REGIME_CLASSIFIER=1
    export JULIE_SIGNAL_GATE_2025=0
    export JULIE_FILTERG_PER_CELL_ACTIVE=0
    export JULIE_CASCADE_BLOCKER_ACTIVE=1 JULIE_ANTI_FLIP_BLOCKER_ACTIVE=1
    export JULIE_TRIATHLON_ACTIVE=0
    export JULIE_SAMESIDE_ML=0   # hard rule same-side
    export JULIE_BYPASS_SAMESIDE=0
    export JULIE_REGIME_ML_BRACKETS=0 JULIE_REGIME_ML_SIZE=0 JULIE_REGIME_ML_BE=0
    export JULIE_KALSHI_DE3_MODE=on JULIE_KALSHI_CONTINUATION_TP=1
    export JULIE_ML_LFO_ACTIVE=0 JULIE_ML_PCT_ACTIVE=0 JULIE_ML_PIVOT_TRAIL_ACTIVE=0
    export JULIE_ML_KALSHI_ACTIVE=0 JULIE_ML_KALSHI_TP_ACTIVE=0
    export JULIE_ML_RL_MGMT_ACTIVE=0 JULIE_RL_REGIME_GATE_ACTIVE=0
    export JULIE_KALSHI_CM_GATE_V2_ACTIVE=0
    ;;
  mlstack)
    # Full ML stack ON — everything you had on before the PR
    export JULIE_CB=1 JULIE_DLB=1 JULIE_DD_SCALE=1 JULIE_LOSS_FACTOR_GUARD=1
    export JULIE_REGIME_ADAPTIVE_CB=1 JULIE_REGIME_CLASSIFIER=1
    export JULIE_SIGNAL_GATE_2025=1 JULIE_FILTERG_PER_CELL_ACTIVE=1
    export JULIE_CASCADE_BLOCKER_ACTIVE=1 JULIE_ANTI_FLIP_BLOCKER_ACTIVE=1
    export JULIE_TRIATHLON_ACTIVE=1
    export JULIE_SAMESIDE_ML=1 JULIE_SAMESIDE_ML_MAX_CONTRACTS=2
    export JULIE_BYPASS_SAMESIDE=0
    export JULIE_REGIME_ML_BRACKETS=1 JULIE_REGIME_ML_SIZE=1 JULIE_REGIME_ML_BE=1
    export JULIE_KALSHI_DE3_MODE=on JULIE_KALSHI_CONTINUATION_TP=1
    export JULIE_ML_LFO_ACTIVE=1 JULIE_ML_PCT_ACTIVE=1 JULIE_ML_PIVOT_TRAIL_ACTIVE=1
    export JULIE_ML_KALSHI_ACTIVE=1 JULIE_ML_KALSHI_TP_ACTIVE=1
    export JULIE_ML_RL_MGMT_ACTIVE=1 JULIE_RL_REGIME_GATE_ACTIVE=1
    export JULIE_KALSHI_CM_GATE_V2_ACTIVE=1
    ;;
  v8)
    # Full ML stack + new Kalshi v8 ML replaces Kalshi rule
    export JULIE_CB=1 JULIE_DLB=1 JULIE_DD_SCALE=1 JULIE_LOSS_FACTOR_GUARD=1
    export JULIE_REGIME_ADAPTIVE_CB=1 JULIE_REGIME_CLASSIFIER=1
    export JULIE_SIGNAL_GATE_2025=1 JULIE_FILTERG_PER_CELL_ACTIVE=1
    export JULIE_CASCADE_BLOCKER_ACTIVE=1 JULIE_ANTI_FLIP_BLOCKER_ACTIVE=1
    export JULIE_TRIATHLON_ACTIVE=1
    export JULIE_SAMESIDE_ML=1 JULIE_SAMESIDE_ML_MAX_CONTRACTS=2
    export JULIE_BYPASS_SAMESIDE=0
    export JULIE_REGIME_ML_BRACKETS=1 JULIE_REGIME_ML_SIZE=1 JULIE_REGIME_ML_BE=1
    export JULIE_KALSHI_DE3_MODE=on JULIE_KALSHI_CONTINUATION_TP=1
    export JULIE_ML_LFO_ACTIVE=1 JULIE_ML_PCT_ACTIVE=1 JULIE_ML_PIVOT_TRAIL_ACTIVE=1
    export JULIE_ML_KALSHI_ACTIVE=1 JULIE_ML_KALSHI_TP_ACTIVE=1
    export JULIE_ML_RL_MGMT_ACTIVE=1 JULIE_RL_REGIME_GATE_ACTIVE=1
    export JULIE_KALSHI_CM_GATE_V2_ACTIVE=1
    export JULIE_KALSHI_ML_V8=1   # ← v8 active
    ;;
  ml_full_ny)
    # NY-only sweep with EVERYTHING on: ML overlays (LFO, Pivot, PCT, Kalshi,
    # Kalshi-TP, RL-mgmt, Regime ML brackets/size/be, SameSide ML), rule
    # blockers (CB, DLB, Cascade, AntiFlip), AF strategy LOADED (no skip).
    # Friend's swapped overlay model artifacts on disk are used as-is —
    # results show what THAT stack does, not whether the models are correct.
    # NY-only filter restricts replay bars to 08:00–17:00 ET (~67% bar drop).
    export JULIE_REPLAY_NY_ONLY=1            # ← NY-hours only
    export JULIE_CB=1 JULIE_DLB=1 JULIE_DD_SCALE=1 JULIE_LOSS_FACTOR_GUARD=1
    export JULIE_REGIME_ADAPTIVE_CB=1 JULIE_REGIME_CLASSIFIER=1
    export JULIE_SIGNAL_GATE_2025=1 JULIE_FILTERG_PER_CELL_ACTIVE=1
    export JULIE_CASCADE_BLOCKER_ACTIVE=1 JULIE_ANTI_FLIP_BLOCKER_ACTIVE=1
    export JULIE_TRIATHLON_ACTIVE=1
    export JULIE_SAMESIDE_ML=1 JULIE_SAMESIDE_ML_MAX_CONTRACTS=2
    export JULIE_BYPASS_SAMESIDE=0
    export JULIE_REGIME_ML_BRACKETS=1 JULIE_REGIME_ML_SIZE=1 JULIE_REGIME_ML_BE=1
    export JULIE_KALSHI_DE3_MODE=on JULIE_KALSHI_CONTINUATION_TP=1
    export JULIE_ML_LFO_ACTIVE=1 JULIE_ML_PCT_ACTIVE=1 JULIE_ML_PIVOT_TRAIL_ACTIVE=1
    export JULIE_ML_KALSHI_ACTIVE=1 JULIE_ML_KALSHI_TP_ACTIVE=1
    export JULIE_ML_RL_MGMT_ACTIVE=1 JULIE_RL_REGIME_GATE_ACTIVE=1
    export JULIE_KALSHI_CM_GATE_V2_ACTIVE=1
    # Note: DirectionalLossBlocker only fully activates when the filter
    # stack is on (JULIE_DISABLE_STRATEGY_FILTERS=0). In filterless-only
    # mode (kept here, like all other configs) DLB is a no-op. CascadeLoss
    # and AntiFlip have their own gates and DO activate.
    # Don't disable AF (user wants it loaded) and don't disable RL shadow
    # (user said SHADOW_RL OFF — that means the RL-shadow per-bar inference
    # is OFF; meanwhile JULIE_ML_RL_MGMT_ACTIVE=1 turns on the live RL
    # management bracket overlay, which is a different code path).
    export JULIE_DISABLE_RL_SHADOW=1
    # AF stays loaded — no JULIE_DISABLE_STRATEGIES override
    ;;
  *)
    echo "unknown CONFIG: $CONFIG"; exit 1;;
esac

echo "[$CONFIG/$MONTH] launching with start=$START end=$END"
python3 tools/run_full_live_replay_parquet.py \
    --start "$START" --end "$END" \
    --bars-parquet es_master_outrights.parquet \
    --target-symbol "$SYMBOL" \
    --report-dir backtest_reports \
    --run-name "$RUN_NAME" \
    --account-id 99999 \
    --initial-balance 50000 > "$REPORT_DIR/replay_stdout.log" 2>&1

echo "[$CONFIG/$MONTH] replay done"
