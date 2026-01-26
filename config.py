import datetime
import os
from typing import Optional
from zoneinfo import ZoneInfo


# Core runtime configuration for the bot. This was pulled directly from the
# original julie001.py entrypoint so other modules can import it without
# bringing in the full bot runtime.
CONFIG = {
    # --- CREDENTIALS ---
    "USERNAME": "",
    "API_KEY": "",

    # --- ACCOUNT/CONTRACT (will be fetched dynamically) ---
    "ACCOUNT_ID": os.environ.get("JULIE_ACCOUNT_ID"),  # Can be set via env var or fetched via /Account/search
    "CONTRACT_ID": None,  # Fetched via /Contract/available (e.g., "CON.F.US.MES.H25")
    "CONTRACT_ROOT": "MES",  # Symbol root used to determine current MES contract (e.g., CON.F.US.MES.Z25)
    "TARGET_SYMBOL": None,  # Determined dynamically from date and CONTRACT_ROOT

    # --- API ENDPOINTS (ProjectX Gateway LIVE) ---
    # Switched from 'gateway-api-demo' to 'gateway-api'
    "REST_BASE_URL": "https://api.topstepx.com",
    "RTC_USER_HUB": "https://rtc.topstepx.com/hubs/user",
    "RTC_MARKET_HUB": "https://rtc.topstepx.com/hubs/market",

    # --- SYSTEM SETTINGS ---
    "MAX_DAILY_LOSS": 1000.0,
    "TIMEZONE": "US/Eastern",

    # --- RISK & FEE MANAGEMENT ---
    "RISK": {
        "POINT_VALUE": 5.0,      # $ value per point per contract (MES=$5, ES=$50)
        "FEES_PER_SIDE": 2.50,   # Commission + Exchange fee per side (estimated)
        "MIN_NET_PROFIT": 10.0,  # Minimum expected profit (after round-trip fees) to take a trade
        "CONTRACTS": 1           # Number of contracts traded
    },

    # --- ML SESSION-BASED STRATEGY SETTINGS ---
    "WINDOW_SIZE": 15,
    "ML_PHYSICS_TIMEFRAME_MINUTES": 1,
    # Optional: use ML training output files to override session thresholds
    "ML_PHYSICS_THRESHOLDS_FILE": "ml_physics_thresholds.json",
    "ML_PHYSICS_METRICS_FILE": "ml_physics_metrics.json",
    # Guardrails to auto-disable weak sessions based on training metrics
    "ML_PHYSICS_GUARD": {
        "enabled": True,
        "min_trades": 500,     # Require enough samples for stability
        "min_win_rate": 0.55,  # Basic edge over 50/50
        "min_avg_pnl": 0.2,    # Points per trade after fees (training eval)
    },
    # Optional per-session/regime overrides for ML physics guard thresholds
    "ML_PHYSICS_GUARD_OVERRIDES": {
        "ASIA": {
            "low": {"min_trades": 200, "min_win_rate": 0.55, "min_avg_pnl": 0.2},
            "high": {"min_trades": 50, "min_win_rate": 0.60, "min_avg_pnl": 0.3},
        },
    },
    # Disable MLPhysics sessions in live bot (backtest still evaluates them)
    "ML_PHYSICS_LIVE_DISABLED_SESSIONS": [],
    # Disable MLPhysics sessions in backtest (default empty to allow evaluation)
    "ML_PHYSICS_BACKTEST_DISABLED_SESSIONS": [],
    # Backtest-only: learned continuation allowlist from walk-forward reports
    "BACKTEST_CONTINUATION_ALLOWLIST": {
        "enabled": True,
        # modes: "reports" (walk-forward backtest files) or "csv_fast" (single CSV pass)
        "mode": "csv_fast",
        "reports_glob": "backtest_reports/backtest_*.json",
        "min_total_trades": 8,
        "min_fold_trades": 2,
        "min_avg_pnl_points": 0.1,
        "min_fold_expectancy_points": 0.0,
        "min_folds": 2,
        "min_positive_fold_ratio": 0.60,
        "cache_file": "backtest_reports/continuation_allowlist.json",
        "fast": {
            "folds": 4,
            "max_horizon_bars": 120,
            "exit_at_horizon": "close",
            "assume_sl_first": True,
            "use_dynamic_sltp": True,
            "default_tp": 6.0,
            "default_sl": 4.0,
            "min_win_rate": 0.45,
            "symbol_contains": ["MES"],
        },
    },
    # Backtest-only: require market confirmation for continuation
    "BACKTEST_CONTINUATION_CONFIRM": {
        "enabled": True,
        "use_adx": True,
        "use_trend_alt": True,
        "use_vwap": True,
        "use_structure_break": True,
        "vwap_sigma_min": 1.0,
        "require_any": True,
    },
    # Backtest-only: continuation signal generation mode ("calendar" or "structure")
    "BACKTEST_CONTINUATION_SIGNAL_MODE": "structure",
    # Backtest-only: allow continuation only in proven regimes
    "BACKTEST_CONTINUATION_ALLOWED_REGIMES": ["high"],
    # Backtest-only: continuation rescues do not bypass core filters
    "BACKTEST_CONTINUATION_NO_BYPASS": True,
    # Live continuation guardrails (allowlist + confirmation + regime gating)
    "CONTINUATION_GUARD": {
        "enabled": True,
        # Signal generation mode for live continuation ("calendar" or "structure")
        "signal_mode": "structure",
        "allowlist_file": "backtest_reports/continuation_allowlist.json",
        "allowed_regimes": ["high"],
        "confirm": {
            "enabled": True,
            "use_adx": True,
            "use_trend_alt": True,
            "use_vwap": True,
            "use_structure_break": True,
            "vwap_sigma_min": 1.0,
            "require_any": True,
        },
        "no_bypass": True,
    },
    # Backtest-only: require stronger MLPhysics confidence before it can support consensus
    "BACKTEST_CONSENSUS_ML_MIN_CONF": 0.65,
    "BACKTEST_CONSENSUS_ML_EXTRA_MARGIN": 0.05,
    # Backtest-only: extend vol-split ML sessions without affecting live defaults
    "ML_PHYSICS_VOL_SPLIT_BACKTEST_SESSIONS": [],
    # Volatility guard: skip MLPhysics in selected sessions during high vol
    "ML_PHYSICS_VOL_GUARD": {
        "enabled": True,
        "sessions": ["NY_AM"],
        "feature": "High_Volatility",
    },
    # Robust NY_AM fix: split ML models by volatility regime
    "ML_PHYSICS_VOL_SPLIT": {
        "enabled": True,
        "sessions": ["ASIA", "LONDON", "NY_AM", "NY_PM"],
        "feature": "High_Volatility",
    },
    # Session-specific training presets (used by ml_train_physics.py)
    "ML_PHYSICS_TRAINING_PRESETS": {
        "ASIA": {
            "timeframe_minutes": 3,
            "horizon_bars": 12,
            "label_mode": "atr",
            "drop_neutral": True,
            "thr_min": 0.62,
            "thr_max": 0.85,
            "thr_step": 0.01,
            "drop_gap_minutes": 30.0,
        },
        "LONDON": {
            "label_mode": "barrier",
            "drop_neutral": True,
            "thr_min": 0.60,
            "thr_max": 0.80,
            "thr_step": 0.01,
            "drop_gap_minutes": 20.0,
        },
        "NY_AM": {
            "timeframe_minutes": 1,
            "horizon_bars": 8,
            "label_mode": "barrier",
            "drop_neutral": True,
            "thr_min": 0.60,
            "thr_max": 0.80,
            "thr_step": 0.01,
            "drop_gap_minutes": 20.0,
        },
        "NY_PM": {
            "label_mode": "barrier",
            "drop_neutral": True,
            "thr_min": 0.60,
            "thr_max": 0.80,
            "thr_step": 0.01,
            "drop_gap_minutes": 20.0,
        },
    },

    # --- EARLY EXIT OPTIMIZATION (from 2023-2025 backtest analysis) ---
    # Combined early exit rules:
    # 1. exit_if_not_green_by: Exit if not profitable within X bars
    # 2. max_profit_crosses: Exit if price crosses profit/loss threshold X times
    "EARLY_EXIT": {
        "Confluence": {
            "enabled": True,
            "exit_if_not_green_by": 5,  # Exit if not profitable within 5 bars)
            "max_profit_crosses": 1,     # Exit if crosses > 1
        },
        "ICT_Model": {
            "enabled": True,
            "exit_if_not_green_by": 1,   # Exit if not profitable within 1 bar
            "max_profit_crosses": 0,     # Exit if ANY profit cross
        },
        "MLPhysics_ASIA": {
            "enabled": True,
            "exit_if_not_green_by": 30,   # Give trade room to develop
            "max_profit_crosses": 4,      # Allow some chop
        },
        "MLPhysics_LONDON": {
            "enabled": True,
            "exit_if_not_green_by": 30,   # Give trade room to develop
            "max_profit_crosses": 4,      # Allow some chop
        },
        "MLPhysics_NY_AM": {
            "enabled": True,
            "exit_if_not_green_by": 30,   # Give trade room to develop
            "max_profit_crosses": 4,      # Allow some chop
        },
        "MLPhysics_NY_PM": {
            "enabled": True,
            "exit_if_not_green_by": 30,   # Give trade room to develop
            "max_profit_crosses": 4,      # Allow some chop
        },
        # These strategies don't benefit from early exit (wins happen faster than losses)
        "RegimeAdaptive": {
            "enabled": True,
            "exit_if_not_green_by": 30,   # If still red after 30 bars, bail
            "max_profit_crosses": 4       # Max 2 green/red flips before we exit as chop
        },
        "IntradayDip": {
            "enabled": True,
            "exit_if_not_green_by": 30,   # Give trade room to develop
            "max_profit_crosses": 4,      # Allow some chop
        },
        "DynamicEngine": {
            "enabled": True,
            "exit_if_not_green_by": 30,   # Give trade room to develop
            "max_profit_crosses": 4,      # Allow some chop
        },
        "ORB_Long": {"enabled": False},
    },

    # --- BREAK-EVEN LOGIC ---
    # Move stop to entry when profit reaches X% of TP distance
    "BREAK_EVEN": {
        "enabled": True,
        "trigger_pct": 0.40,  # Trigger at 40% of TP distance
        "buffer_ticks": 1,    # Add 1 tick buffer above entry for longs (below for shorts)
        "trail_pct": 0.25,    # Lock in 25% of profit as trailing stop
    },

    # --- GEMINI 3.0 PRO OPTIMIZATION ---
    "GEMINI": {
        "enabled": True,
        "api_key": "",
        "model": "gemini-3-pro-preview",
    },

    # Dynamic Multipliers (Updated by Bot at runtime)
    "DYNAMIC_SL_MULTIPLIER": 1.0,
    "DYNAMIC_TP_MULTIPLIER": 1.0,

    # --- SESSION DEFINITIONS (From Optimization Results) ---
    "SESSIONS": {
        "ASIA": {
            # 6:00 PM - 3:00 AM ET
            "HOURS": [18, 19, 20, 21, 22, 23, 0, 1, 2],
            "MODEL_FILE": "model_asia.joblib",
            "MODEL_FILE_LOW": "model_asia_low.joblib",
            "MODEL_FILE_HIGH": "model_asia_high.joblib",
            "TIMEFRAME_MINUTES": 3,
            "THRESHOLD": 0.65,  # Strict Entry
            "SL": 4.0,          # Tight Stop
            "TP": 6.0           # Moderate Target
        },
        "LONDON": {
            # 3:00 AM - 8:00 AM ET
            "HOURS": [3, 4, 5, 6, 7],
            "MODEL_FILE": "model_london.joblib",
            "MODEL_FILE_LOW": "model_london_low.joblib",
            "MODEL_FILE_HIGH": "model_london_high.joblib",
            "THRESHOLD": 0.55,  # Standard Entry
            "SL": 4.0,          # Scalper Stop
            "TP": 4.0           # Scalper Target
        },
        "NY_AM": {
            # 8:00 AM - 12:00 PM ET
            "HOURS": [8, 9, 10, 11],
            "MODEL_FILE": "model_ny_am.joblib",
            "MODEL_FILE_LOW": "model_ny_am_low.joblib",
            "MODEL_FILE_HIGH": "model_ny_am_high.joblib",
            "THRESHOLD": 0.55,
            "SL": 10.0,         # Wide Stop (Breathing Room)
            "TP": 4.0           # High Probability Target (80% WR)
        },
        "NY_PM": {
            # 12:00 PM - 5:00 PM ET
            "HOURS": [12, 13, 14, 15, 16],
            "MODEL_FILE": "model_ny_pm.joblib",
            "MODEL_FILE_LOW": "model_ny_pm_low.joblib",
            "MODEL_FILE_HIGH": "model_ny_pm_high.joblib",
            "THRESHOLD": 0.55,
            "SL": 10.0,         # Wide Stop
            "TP": 8.0           # Trend Target (Highest PnL)
        }
    }
}


CONTRACT_MONTH_CODES = {
    1: "H",  # March
    2: "H",  # March
    3: "H",  # March
    4: "M",  # June
    5: "M",  # June
    6: "M",  # June
    7: "U",  # September
    8: "U",  # September
    9: "U",  # September
    10: "Z",  # December
    11: "Z",  # December
    12: "Z",  # December
}


def determine_current_contract_symbol(
    root: str = "MES",
    tz_name: str = "US/Eastern",
    today: Optional[datetime.date] = None,
) -> str:
    """Return the active contract symbol, handling rollover logic."""
    tz = ZoneInfo(tz_name.replace("US/Eastern", "America/New_York"))
    current_date = today or datetime.datetime.now(tz).date()

    year = current_date.year
    month = current_date.month

    # Contract Months: H (Mar), M (Jun), U (Sep), Z (Dec)
    # Rollover logic: If past the 10th of an expiration month, move to next
    if month == 3 and current_date.day > 10: target_code = "M"
    elif month == 6 and current_date.day > 10: target_code = "U"
    elif month == 9 and current_date.day > 10: target_code = "Z"
    elif month == 12 and current_date.day > 10:
        target_code = "H"
        year += 1  # Roll to next year (March 2026)
    else:
        # Standard Mapping
        if month <= 3: target_code = "H"
        elif month <= 6: target_code = "M"
        elif month <= 9: target_code = "U"
        else: target_code = "Z"

    year_code = str(year % 100).zfill(2)
    return f"{root}.{target_code}{year_code}"


def refresh_target_symbol():
    """Update CONFIG['TARGET_SYMBOL'] based on today's date and configured root."""

    CONFIG["TARGET_SYMBOL"] = determine_current_contract_symbol(
        root=CONFIG.get("CONTRACT_ROOT", "MES"),
        tz_name=CONFIG.get("TIMEZONE", "US/Eastern"),
    )


# Initialize TARGET_SYMBOL at import time
refresh_target_symbol()
