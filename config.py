import datetime
from typing import Optional
import pytz


# Core runtime configuration for the bot. This was pulled directly from the
# original julie001.py entrypoint so other modules can import it without
# bringing in the full bot runtime.
CONFIG = {
    # --- CREDENTIALS ---
    "USERNAME": "",
    "API_KEY": "",

    # --- ACCOUNT/CONTRACT (will be fetched dynamically) ---
    "ACCOUNT_ID": None,  # Fetched via /Account/search
    "CONTRACT_ID": None,  # Fetched via /Contract/available (e.g., "CON.F.US.ES.H25")
    "CONTRACT_ROOT": "ES",  # Symbol root used to determine current ES contract (e.g., ESZ25)
    "TARGET_SYMBOL": None,  # Determined dynamically from date and CONTRACT_ROOT

    # --- API ENDPOINTS (ProjectX Gateway LIVE) ---
    # Switched from 'gateway-api-demo' to 'gateway-api'
    "REST_BASE_URL": "https://api.topstepx.com",
    "RTC_USER_HUB": "https://rtc.topstepx.com/hubs/user",
    "RTC_MARKET_HUB": "https://rtc.topstepx.com/hubs/market",

    # --- SYSTEM SETTINGS ---
    "MAX_DAILY_LOSS": 1000.0,
    "TIMEZONE": "US/Eastern",

    # --- ML SESSION-BASED STRATEGY SETTINGS ---
    "WINDOW_SIZE": 15,

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
        "DynamicEngine2": {
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
    },

    # --- SESSION DEFINITIONS (From Optimization Results) ---
    "SESSIONS": {
        "ASIA": {
            # 6:00 PM - 3:00 AM ET
            "HOURS": [18, 19, 20, 21, 22, 23, 0, 1, 2],
            "MODEL_FILE": "model_asia.joblib",
            "THRESHOLD": 0.65,  # Strict Entry
            "SL": 4.0,          # Tight Stop
            "TP": 6.0           # Moderate Target
        },
        "LONDON": {
            # 3:00 AM - 8:00 AM ET
            "HOURS": [3, 4, 5, 6, 7],
            "MODEL_FILE": "model_london.joblib",
            "THRESHOLD": 0.55,  # Standard Entry
            "SL": 4.0,          # Scalper Stop
            "TP": 4.0           # Scalper Target
        },
        "NY_AM": {
            # 8:00 AM - 12:00 PM ET
            "HOURS": [8, 9, 10, 11],
            "MODEL_FILE": "model_ny_am.joblib",
            "THRESHOLD": 0.55,
            "SL": 10.0,         # Wide Stop (Breathing Room)
            "TP": 4.0           # High Probability Target (80% WR)
        },
        "NY_PM": {
            # 12:00 PM - 5:00 PM ET
            "HOURS": [12, 13, 14, 15, 16],
            "MODEL_FILE": "model_ny_pm.joblib",
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
    root: str = "ES",
    tz_name: str = "US/Eastern",
    today: Optional[datetime.date] = None,
) -> str:
    """Return the ES contract symbol for the current date (e.g., ESZ25)."""

    tz = pytz.timezone(tz_name)
    current_date = today or datetime.datetime.now(tz).date()
    month_code = CONTRACT_MONTH_CODES.get(current_date.month)

    if month_code is None:
        raise ValueError(f"Unsupported month for contract mapping: {current_date.month}")

    year_code = str(current_date.year % 100).zfill(2)
    return f"{root}{month_code}{year_code}"


def refresh_target_symbol():
    """Update CONFIG['TARGET_SYMBOL'] based on today's date and configured root."""

    CONFIG["TARGET_SYMBOL"] = determine_current_contract_symbol(
        root=CONFIG.get("CONTRACT_ROOT", "ES"),
        tz_name=CONFIG.get("TIMEZONE", "US/Eastern"),
    )


# Initialize TARGET_SYMBOL at import time
refresh_target_symbol()
