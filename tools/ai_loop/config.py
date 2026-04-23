"""Safety rails + whitelist for the AI-loop auto-adjust stack.

CHANGE THIS FILE CAREFULLY — every constant here defines the bounds of
what the automated system is allowed to do to the live bot's config.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "ai_loop_data"
JOURNALS_DIR = DATA_DIR / "journals"
RECS_DIR = DATA_DIR / "recommendations"
VALIDATED_DIR = DATA_DIR / "validated"
APPLIED_DIR = DATA_DIR / "applied"
STATE_FILE = DATA_DIR / "state.json"
AUDIT_LOG = DATA_DIR / "audit.jsonl"

for d in (JOURNALS_DIR, RECS_DIR, VALIDATED_DIR, APPLIED_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# WHITELIST
# A param is auto-adjustable ONLY if it appears here with a valid spec.
# Adding a param to this list = authorizing the AI loop to change it.
# ─────────────────────────────────────────────────────────────
# Spec fields:
#   target         — "env" for env-var in launch_filterless_live.py,
#                    "config" for a key path in CONFIG dict (future)
#   key            — the env var name or CONFIG path
#   dtype          — "float" or "int"
#   bounds         — (min, max) absolute clamp
#   max_step_delta — max change per single auto-apply
#   description    — human-readable
AUTO_ADJUSTABLE_PARAMS: dict[str, dict] = {
    # Kalshi cross-market ML gate v2 override threshold — lives inside
    # the joblib payload, not env. Handled specially in applier.py.
    "cm_gate_v2_override_threshold": {
        "target": "joblib",
        "key": "artifacts/signal_gate_2025/model_cm_breakout_long.joblib:override_threshold",
        "also_update": [
            "artifacts/signal_gate_2025/model_cm_breakout_short.joblib:override_threshold",
        ],
        "dtype": "float",
        "bounds": (0.45, 0.80),
        "max_step_delta": 0.05,
        "description": "Min p for CM v2 gate to override a Kalshi block.",
    },
    # Kalshi TP PnL threshold — set via env in launcher
    "JULIE_ML_KALSHI_TP_PNL_THR": {
        "target": "env",
        "key": "JULIE_ML_KALSHI_TP_PNL_THR",
        "dtype": "float",
        "bounds": (-15.0, 25.0),
        "max_step_delta": 2.5,
        "description": "Kalshi TP regressor: block when predicted pnl <= X.",
    },
    # CM gate v2 active flag — allow toggle but treat as special (binary)
    "JULIE_KALSHI_CM_GATE_V2_ACTIVE": {
        "target": "env",
        "key": "JULIE_KALSHI_CM_GATE_V2_ACTIVE",
        "dtype": "bool",
        "bounds": (0, 1),
        "max_step_delta": 1,   # can toggle either way
        "description": "Whether the v2 CM gate is authoritative for Kalshi overrides.",
    },
    # RL live-steering flag — binary but safety-critical. Default to OFF
    # being the "safe" direction; any toggle back to 1 requires 2 consec
    # positive backtests.
    "JULIE_ML_RL_MGMT_ACTIVE": {
        "target": "env",
        "key": "JULIE_ML_RL_MGMT_ACTIVE",
        "dtype": "bool",
        "bounds": (0, 1),
        "max_step_delta": 1,
        "description": "Whether RL v3 actively steers stops.",
        "high_risk": True,
    },
    # Cascade loss blocker — time-window loss-cluster veto (separate from
    # DirectionalLossBlocker which counts strictly-consecutive losses).
    # Backtest on 2025+2026 (5,237 trades / 370 days) at count=2 /
    # window=30min / cool=30min showed +$1.1k (2026) and +$4.2k (2025)
    # lift with lower DD in both. High-risk flag ensures the applier
    # won't flip it without manual confirmation.
    "JULIE_CASCADE_BLOCKER_ACTIVE": {
        "target": "env",
        "key": "JULIE_CASCADE_BLOCKER_ACTIVE",
        "dtype": "bool",
        "bounds": (0, 1),
        "max_step_delta": 1,
        "description": "Whether the time-window cascade loss blocker is active.",
        "high_risk": True,
    },
    "JULIE_CASCADE_BLOCKER_COUNT": {
        "target": "env",
        "key": "JULIE_CASCADE_BLOCKER_COUNT",
        "dtype": "int",
        "bounds": (2, 4),
        "max_step_delta": 1,
        "description": "N same-side losses within window that trip the cascade blocker.",
    },
    "JULIE_CASCADE_BLOCKER_WINDOW_MIN": {
        "target": "env",
        "key": "JULIE_CASCADE_BLOCKER_WINDOW_MIN",
        "dtype": "int",
        "bounds": (10, 60),
        "max_step_delta": 15,
        "description": "Rolling window length (minutes) for the cascade blocker.",
    },
    "JULIE_CASCADE_BLOCKER_COOLDOWN_MIN": {
        "target": "env",
        "key": "JULIE_CASCADE_BLOCKER_COOLDOWN_MIN",
        "dtype": "int",
        "bounds": (10, 60),
        "max_step_delta": 15,
        "description": "Cooldown (minutes) after the most recent qualifying loss.",
    },
    # Anti-flip blocker — rejects new opposite-side signals that fire
    # close to the price where the last trade just stopped out. Catches
    # the "flip at the stop" failure mode (2026-04-23 SHORT stopped at
    # 7172.50 then LONG flipped at 7171.75). The activation flag is
    # marked high_risk so the applier won't flip it without manual
    # confirmation; window/distance are auto-adjustable within bounds.
    "JULIE_ANTI_FLIP_BLOCKER_ACTIVE": {
        "target": "env",
        "key": "JULIE_ANTI_FLIP_BLOCKER_ACTIVE",
        "dtype": "bool",
        "bounds": (0, 1),
        "max_step_delta": 1,
        "description": "Whether the anti-flip blocker is active.",
        "high_risk": True,
    },
    "JULIE_ANTI_FLIP_WINDOW_MIN": {
        "target": "env",
        "key": "JULIE_ANTI_FLIP_WINDOW_MIN",
        "dtype": "int",
        "bounds": (5, 60),
        "max_step_delta": 15,
        "description": "Minutes after a stop-out during which opposite-side near-price signals are blocked.",
    },
    "JULIE_ANTI_FLIP_MAX_DIST_PTS": {
        "target": "env",
        "key": "JULIE_ANTI_FLIP_MAX_DIST_PTS",
        "dtype": "float",
        "bounds": (2.0, 15.0),
        "max_step_delta": 2.0,
        "description": "Max distance in points from the last stop-out price to trigger the block.",
    },
    # Triathlon Engine — per-cell medal-driven size/priority.
    # Activation flag marked high_risk so the applier won't flip it
    # without manual confirmation.
    "JULIE_TRIATHLON_ACTIVE": {
        "target": "env",
        "key": "JULIE_TRIATHLON_ACTIVE",
        "dtype": "bool",
        "bounds": (0, 1),
        "max_step_delta": 1,
        "description": "Whether the Triathlon Engine applies medal-driven size/priority effects live.",
        "high_risk": True,
    },
    # Filter G per-cell threshold override (Idea 1). Loads the pre-April-
    # derived multiplier table at runtime; bleeding cells get 0.75×
    # threshold (more aggressive veto), strong cells get 1.15×. high_risk
    # because flipping off reverts to the undifferentiated-threshold
    # behavior and removes a PnL optimization.
    "JULIE_FILTERG_PER_CELL_ACTIVE": {
        "target": "env",
        "key": "JULIE_FILTERG_PER_CELL_ACTIVE",
        "dtype": "bool",
        "bounds": (0, 1),
        "max_step_delta": 1,
        "description": "Whether Filter G applies per-(strategy × regime × time-bucket) threshold overrides.",
        "high_risk": True,
    },
    # Triathlon time-decay half-life (Option B, 2026-04-23). Zero
    # disables decay (all trades weight 1.0); positive values weight
    # recent trades higher. OOS picked 120 days on the April 2026
    # holdout, but the right value may drift as more live data
    # accumulates, so allow AI-loop auto-adjustment within bounds.
    "JULIE_TRIATHLON_HALFLIFE_DAYS": {
        "target": "env",
        "key": "JULIE_TRIATHLON_HALFLIFE_DAYS",
        "dtype": "int",
        "bounds": (0, 365),
        "max_step_delta": 30,
        "description": "Half-life in days for Triathlon time-decay weighting (0 = off).",
    },
}

# Absolutely non-auto-adjustable — even if the analyzer proposes, validator
# rejects. Things that could destroy the stack.
HARDCODED_FORBIDDEN = {
    # Can't modify model architecture / feature schema
    "model architecture", "feature_names", "numeric_features",
    # Can't change bracket behavior
    "sl_dist", "tp_dist", "TICK_SIZE",
    # Can't touch account config
    "ACCOUNT_ID", "account_id",
    # Can't disable the filterless core
    "JULIE_FILTERLESS_ONLY", "JULIE_DISABLE_STRATEGY_FILTERS",
}


# ─────────────────────────────────────────────────────────────
# GLOBAL SAFETY
# ─────────────────────────────────────────────────────────────
COOLDOWN_DAYS = 7
"""After a successful auto-apply, the same param can't change again for
this many days. Prevents oscillation and compounding drift."""

MAX_APPLIES_PER_DAY = 2
"""Daily cap across ALL params."""

STOP_LOSS_48H_DOLLARS = 500.0
"""If live bot's drawdown exceeds this in the last 48 hours of trades,
freeze all auto-applies for 7 days. Human re-enables."""

BACKTEST_MIN_LIFT_PCT = 0.05
"""Validator: candidate PnL must beat baseline by at least this fraction."""

BACKTEST_MAX_DD_INFLATE = 1.20
"""Validator: candidate max-DD must be ≤ baseline max-DD × this."""

BACKTEST_PERIOD_DAYS = 30
"""How many recent days of tape to use as the backtest window."""

BACKTEST_OOS_SPLIT_FRAC = 0.70
"""Rolling-origin split: use first X% as train/config-proposal data,
last 1-X% as genuinely out-of-sample for validation."""

KILL_SWITCH_ENV = "JULIE_FREEZE_AUTO_CONFIG"
"""Set this env var to '1' and the orchestrator short-circuits before
applying anything. Journal + analyzer + validator still run (read-only)."""

LIVE_VS_BACKTEST_SIGMA = 2.0
"""Monitor: if live realized PnL drifts this many std-devs below the
backtest forecast over 5 sessions, auto-revert the most recent change."""

LIVE_VS_BACKTEST_WINDOW = 5
"""Sessions to average over for the drift check."""


def is_frozen() -> bool:
    """Check the kill switch."""
    import os
    return os.environ.get(KILL_SWITCH_ENV, "").strip() == "1"


def validate_param_delta(param: str, current: float, proposed: float) -> tuple[bool, str]:
    """Return (ok, reason). Every auto-apply MUST pass this gate."""
    if param in HARDCODED_FORBIDDEN:
        return False, f"param '{param}' is hard-coded forbidden"
    spec = AUTO_ADJUSTABLE_PARAMS.get(param)
    if spec is None:
        return False, f"param '{param}' not in whitelist"
    lo, hi = spec["bounds"]
    if not (lo <= proposed <= hi):
        return False, f"proposed {proposed} outside bounds [{lo}, {hi}]"
    step = abs(proposed - current)
    if step > spec["max_step_delta"] + 1e-9:
        return False, f"step Δ={step:.4f} > max {spec['max_step_delta']}"
    if spec["dtype"] == "bool" and proposed not in (0, 1):
        return False, f"bool param must be 0 or 1"
    return True, "ok"
