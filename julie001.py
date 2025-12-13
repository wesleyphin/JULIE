import requests
import pandas as pd
import numpy as np
import datetime
import time
import logging
import pytz
import joblib
import uuid
from abc import ABC, abstractmethod
from typing import Dict, Optional, List, Tuple

from config import CONFIG, refresh_target_symbol
from dynamic_sltp_params import dynamic_sltp_engine, get_sltp
from volatility_filter import volatility_filter, check_volatility, VolRegime
from regime_strategy import RegimeAdaptiveStrategy
from htf_fvg_filter import HTFFVGFilter
from dynamic_signal_engine import get_signal_engine
from dynamic_signal_engine2 import get_signal_engine as get_signal_engine2
from rejection_filter import RejectionFilter
from chop_filter import ChopFilter
from extension_filter import ExtensionFilter
from dynamic_structure_blocker import DynamicStructureBlocker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler("topstep_live_bot.log"), logging.StreamHandler()]
)

NY_TZ = pytz.timezone('America/New_York')

# ==========================================
# 2a. REJECTION FILTER (Trade Direction Filters)
# ==========================================
# Implementation moved to rejection_filter.py to keep this entrypoint focused on
# bot orchestration.

# ==========================================
# 2d. HTF FVG REJECTION
# ==========================================
try:
    from htf_fvg_filter import HTFFVGFilter
    logging.info("✅ HTFFVGFilter module loaded")
except ImportError as e:
    logging.error(f"❌ Failed to import htf_fvg_filter.py: {e}")
    # Dummy class to prevent crash if file missing
    class HTFFVGFilter:
        def check_signal_blocked(self, *args): return False, None


class BankLevelQuarterFilter:
    """Filter trades based on bank level rejection relative to prev PM, prev session, and midnight ORB."""

    BANK_GRID = 12.5  # $12.50 bank levels
    REJECTION_TOLERANCE = 0.5  # Points tolerance for rejection detection
    REJECTIONS_REQUIRED = 2  # Require at least 2 full candles to confirm / flip bias

    def __init__(self):
        self.reset()

    def reset(self):
        # === PREVIOUS PM SESSION (12:00-17:00 ET) ===
        self.prev_day_pm_high: Optional[float] = None
        self.prev_day_pm_low: Optional[float] = None
        self.current_pm_high: Optional[float] = None
        self.current_pm_low: Optional[float] = None

        # Bank levels relative to prev PM
        self.bank_below_prev_pm_low: Optional[float] = None
        self.bank_above_prev_pm_high: Optional[float] = None

        # === PREVIOUS SESSION ===
        self.prev_session_high: Optional[float] = None
        self.prev_session_low: Optional[float] = None
        self.curr_session_high: Optional[float] = None
        self.curr_session_low: Optional[float] = None
        self.last_session: Optional[str] = None

        # Bank levels relative to prev session
        self.bank_below_prev_session_low: Optional[float] = None
        self.bank_above_prev_session_high: Optional[float] = None

        # === MIDNIGHT ORB (00:00-00:15 ET) ===
        self.midnight_orb_high: Optional[float] = None
        self.midnight_orb_low: Optional[float] = None
        self.midnight_orb_set: bool = False

        # Bank levels relative to midnight ORB
        self.bank_below_orb_low: Optional[float] = None
        self.bank_above_orb_high: Optional[float] = None

        # === COMMON ===
        self.current_date: Optional[datetime.date] = None
        self.current_session: Optional[str] = None
        self.current_quarter: Optional[int] = None

        # Bias from bank level rejection (reset each quarter)
        self.bank_rejection_bias: Optional[str] = None
        self.rejection_source: Optional[str] = None  # Track which level triggered the bias

        # Pending state for 2-candle confirmation
        self.pending_dir: Optional[str] = None      # 'LONG' or 'SHORT'
        self.pending_source: Optional[str] = None   # description of level
        self.pending_count: int = 0

    # -------------------------------------------------
    # Utility helpers
    # -------------------------------------------------
    def _get_closest_bank_below(self, price: float) -> Optional[float]:
        """Get the $12.50 bank level closest below the price"""
        if price is None:
            return None
        return (price // self.BANK_GRID) * self.BANK_GRID

    def _get_closest_bank_above(self, price: float) -> Optional[float]:
        """Get the $12.50 bank level closest above the price"""
        if price is None:
            return None
        return ((price // self.BANK_GRID) + 1) * self.BANK_GRID

    def get_session(self, hour: int, minute: int = 0) -> str:
        """Determine session from hour (ET)"""
        if 18 <= hour <= 23 or 0 <= hour < 3:
            return 'ASIA'
        elif 3 <= hour < 8:
            return 'LONDON'
        elif 8 <= hour < 12:
            return 'NY_AM'
        elif 12 <= hour < 17:
            return 'NY_PM'
        return 'CLOSED'

    def get_quarter(self, hour: int, minute: int, session: str) -> int:
        """Determine which quarter (1-4) based on Daye's Quarterly Theory."""
        if session == 'ASIA':
            if hour >= 18:
                mins_since_start = (hour - 18) * 60 + minute
            else:
                mins_since_start = (24 - 18 + hour) * 60 + minute
            quarter_length = 135
        elif session == 'LONDON':
            mins_since_start = (hour - 3) * 60 + minute
            quarter_length = 75
        elif session == 'NY_AM':
            mins_since_start = (hour - 8) * 60 + minute
            quarter_length = 60
        elif session == 'NY_PM':
            mins_since_start = (hour - 12) * 60 + minute
            quarter_length = 75
        else:
            return 0

        quarter = min(4, (mins_since_start // quarter_length) + 1)
        return quarter

    def check_bank_rejection(self, high: float, low: float, close: float,
                             bank_level: Optional[float], direction: str) -> bool:
        """Check if bar shows rejection of a bank level."""
        if bank_level is None:
            return False

        if direction == 'BULLISH':
            # Wick through or tag slightly below the level, close back above
            if low < bank_level + self.REJECTION_TOLERANCE and close > bank_level:
                return True
        elif direction == 'BEARISH':
            # Wick through or tag slightly above the level, close back below
            if high > bank_level - self.REJECTION_TOLERANCE and close < bank_level:
                return True

        return False

    def _update_bank_levels(self):
        """Recalculate all bank levels based on current reference levels"""
        # Bank levels from prev PM
        self.bank_below_prev_pm_low = self._get_closest_bank_below(self.prev_day_pm_low)
        self.bank_above_prev_pm_high = self._get_closest_bank_above(self.prev_day_pm_high)

        # Bank levels from prev session
        self.bank_below_prev_session_low = self._get_closest_bank_below(self.prev_session_low)
        self.bank_above_prev_session_high = self._get_closest_bank_above(self.prev_session_high)

        # Bank levels from midnight ORB (only if set)
        if self.midnight_orb_set:
            self.bank_below_orb_low = self._get_closest_bank_below(self.midnight_orb_low)
            self.bank_above_orb_high = self._get_closest_bank_above(self.midnight_orb_high)

    # -------------------------------------------------
    # 2-candle confirmation logic for bank-level bias
    # -------------------------------------------------
    def _process_rejection(self, direction: str, source: str, current_quarter: int):
        """Apply 2-candle confirmation logic for bank-level bias."""
        if direction not in ('LONG', 'SHORT'):
            return

        # No existing bias: build initial bias
        if self.bank_rejection_bias is None:
            if self.pending_dir == direction:
                self.pending_count += 1
            else:
                self.pending_dir = direction
                self.pending_source = source
                self.pending_count = 1

            if self.pending_count >= self.REJECTIONS_REQUIRED:
                self.bank_rejection_bias = direction
                self.rejection_source = self.pending_source
                logging.info(
                    f"🎯 BANK BIAS CONFIRMED Q{current_quarter}: "
                    f"{direction} from {self.rejection_source} "
                    f"after {self.REJECTIONS_REQUIRED} rejection candles"
                )
                # Reset pending state
                self.pending_dir = None
                self.pending_source = None
                self.pending_count = 0
            return

        # There is an existing bias
        if direction == self.bank_rejection_bias:
            # Same-direction rejection reinforces current view, clear any flip attempts
            self.pending_dir = None
            self.pending_source = None
            self.pending_count = 0
            return

        # Opposite rejection: candidate flip
        if self.pending_dir == direction:
            self.pending_count += 1
        else:
            self.pending_dir = direction
            self.pending_source = source
            self.pending_count = 1

        if self.pending_count >= self.REJECTIONS_REQUIRED:
            logging.info(
                f"🔁 BANK BIAS FLIP Q{current_quarter}: "
                f"{self.bank_rejection_bias} -> {direction} from {self.pending_source} "
                f"after {self.REJECTIONS_REQUIRED} opposite rejection candles"
            )
            self.bank_rejection_bias = direction
            self.rejection_source = self.pending_source
            # Reset pending state
            self.pending_dir = None
            self.pending_source = None
            self.pending_count = 0

    # -------------------------------------------------
    # Main update loop
    # -------------------------------------------------
    def update(self, ts_et, high: float, low: float, close: float):
        """Update filter state with new bar data"""
        date = ts_et.date()
        hour = ts_et.hour
        minute = ts_et.minute

        current_session = self.get_session(hour, minute)
        current_quarter = self.get_quarter(hour, minute, current_session) if current_session != 'CLOSED' else 0

        # === NEW DAY ===
        if self.current_date != date:
            # Save previous day's PM session as prev_day_pm
            if self.current_pm_high is not None:
                self.prev_day_pm_high = self.current_pm_high
                self.prev_day_pm_low = self.current_pm_low
                logging.info(
                    f"🏦 NEW DAY: Prev PM levels set - "
                    f"High: {self.prev_day_pm_high:.2f}, Low: {self.prev_day_pm_low:.2f}"
                )

            self.current_pm_high = None
            self.current_pm_low = None
            self.current_date = date

            # Reset midnight ORB for new calendar day
            self.midnight_orb_high = None
            self.midnight_orb_low = None
            self.midnight_orb_set = False

            # Recompute PM-based bank levels now that prev_day_pm_* is set
            self._update_bank_levels()

            # New day: clear any partial confirmation streak
            self.pending_dir = None
            self.pending_source = None
            self.pending_count = 0

        # === SESSION CHANGE ===
        if current_session != self.last_session and current_session != 'CLOSED':
            # Save outgoing session's range as prev_session
            if self.curr_session_high is not None:
                self.prev_session_high = self.curr_session_high
                self.prev_session_low = self.curr_session_low
                logging.info(
                    f"🏦 SESSION CHANGE: Prev {self.last_session} "
                    f"High: {self.prev_session_high:.2f}, Low: {self.prev_session_low:.2f}"
                )

            self.curr_session_high = None
            self.curr_session_low = None
            self.last_session = current_session

            # Update bank levels whenever the reference session rolls
            self._update_bank_levels()

            # Session change: clear any partial confirmation streak
            self.pending_dir = None
            self.pending_source = None
            self.pending_count = 0

        # === QUARTER CHANGE: Reset bias ===
        if current_session != 'CLOSED':
            if (self.current_session != current_session or
                    self.current_quarter != current_quarter):
                if self.bank_rejection_bias is not None:
                    logging.info(
                        f"🔄 QUARTER CHANGE: {current_session} Q{current_quarter} "
                        f"| Resetting bank level bias (was {self.bank_rejection_bias} "
                        f"from {self.rejection_source})"
                    )
                self.bank_rejection_bias = None
                self.rejection_source = None
                self.current_session = current_session
                self.current_quarter = current_quarter

                # Clear any partial confirmation streak
                self.pending_dir = None
                self.pending_source = None
                self.pending_count = 0

        # === UPDATE CURRENT SESSION HIGH/LOW ===
        if current_session != 'CLOSED':
            if self.curr_session_high is None:
                self.curr_session_high = high
                self.curr_session_low = low
            else:
                self.curr_session_high = max(self.curr_session_high, high)
                self.curr_session_low = min(self.curr_session_low, low)

        # === UPDATE PM SESSION (12:00-17:00 ET) ===
        if 12 <= hour < 17:
            if self.current_pm_high is None:
                self.current_pm_high = high
                self.current_pm_low = low
            else:
                self.current_pm_high = max(self.current_pm_high, high)
                self.current_pm_low = min(self.current_pm_low, low)

        # === BUILD MIDNIGHT ORB (00:00 - 00:15 ET) ===
        if hour == 0 and minute < 15:
            if self.midnight_orb_high is None:
                self.midnight_orb_high = high
                self.midnight_orb_low = low
            else:
                self.midnight_orb_high = max(self.midnight_orb_high, high)
                self.midnight_orb_low = min(self.midnight_orb_low, low)

        # === COMPLETE ORB AT 00:15 ===
        elif hour == 0 and minute >= 15 and not self.midnight_orb_set:
            self.midnight_orb_set = True
            if self.midnight_orb_high is not None and self.midnight_orb_low is not None:
                self.bank_below_orb_low = self._get_closest_bank_below(self.midnight_orb_low)
                self.bank_above_orb_high = self._get_closest_bank_above(self.midnight_orb_high)
                logging.info(
                    f"🏦 MIDNIGHT ORB SET: High: {self.midnight_orb_high:.2f}, "
                    f"Low: {self.midnight_orb_low:.2f}"
                )
                logging.info(
                    f"🏦 ORB Bank Levels: Below ${self.bank_below_orb_low:.2f} | "
                    f"Above ${self.bank_above_orb_high:.2f}"
                )

        # === CHECK FOR BANK LEVEL REJECTIONS (with 2-candle confirm) ===
        if current_session != 'CLOSED' and current_quarter > 0:
            # 1. Midnight ORB (if set)
            if self.midnight_orb_set:
                # Bullish: bank below ORB low
                if self.bank_below_orb_low is not None and self.check_bank_rejection(
                        high, low, close, self.bank_below_orb_low, 'BULLISH'):
                    self._process_rejection(
                        'LONG',
                        f"Midnight ORB (bank {self.bank_below_orb_low:.2f} below ORB low)",
                        current_quarter,
                    )
                    return  # Respect midnight ORB as top priority

                # Bearish: bank above ORB high
                if self.bank_above_orb_high is not None and self.check_bank_rejection(
                        high, low, close, self.bank_above_orb_high, 'BEARISH'):
                    self._process_rejection(
                        'SHORT',
                        f"Midnight ORB (bank {self.bank_above_orb_high:.2f} above ORB high)",
                        current_quarter,
                    )
                    return

            # 2. Prev session banks
            if self.prev_session_high is not None and self.prev_session_low is not None:
                if self.bank_below_prev_session_low is not None and self.check_bank_rejection(
                        high, low, close, self.bank_below_prev_session_low, 'BULLISH'):
                    self._process_rejection(
                        'LONG',
                        f"Prev Session (bank {self.bank_below_prev_session_low:.2f} below prev session low)",
                        current_quarter,
                    )
                    return

                if self.bank_above_prev_session_high is not None and self.check_bank_rejection(
                        high, low, close, self.bank_above_prev_session_high, 'BEARISH'):
                    self._process_rejection(
                        'SHORT',
                        f"Prev Session (bank {self.bank_above_prev_session_high:.2f} above prev session high)",
                        current_quarter,
                    )
                    return

            # 3. Prev PM banks
            if self.prev_day_pm_high is not None and self.prev_day_pm_low is not None:
                if self.bank_below_prev_pm_low is not None and self.check_bank_rejection(
                        high, low, close, self.bank_below_prev_pm_low, 'BULLISH'):
                    self._process_rejection(
                        'LONG',
                        f"Prev PM (bank {self.bank_below_prev_pm_low:.2f} below prev PM low)",
                        current_quarter,
                    )
                    return

                if self.bank_above_prev_pm_high is not None and self.check_bank_rejection(
                        high, low, close, self.bank_above_prev_pm_high, 'BEARISH'):
                    self._process_rejection(
                        'SHORT',
                        f"Prev PM (bank {self.bank_above_prev_pm_high:.2f} above prev PM high)",
                        current_quarter,
                    )
                    return

    # -------------------------------------------------
    # Public API
    # -------------------------------------------------
    def should_block_trade(self, side: str) -> Tuple[bool, str]:
        """Check if trade should be blocked based on bank level rejection bias."""
        if self.bank_rejection_bias is None:
            return False, ""

        if self.bank_rejection_bias == 'LONG' and side == 'SHORT':
            return True, f"Bank level bias: LONG only Q{self.current_quarter} ({self.rejection_source})"

        if self.bank_rejection_bias == 'SHORT' and side == 'LONG':
            return True, f"Bank level bias: SHORT only Q{self.current_quarter} ({self.rejection_source})"

        return False, ""

    def get_status(self) -> str:
        """Return current filter status for logging"""
        bias_str = f"{self.bank_rejection_bias} ({self.rejection_source})" if self.bank_rejection_bias else "None"
        return f"Bank Filter: {self.current_session} Q{self.current_quarter} | Bias: {bias_str}"
class SessionManager:
    """
    Manages the 4 Neural Networks and switches them based on NY Time.
    """
    def __init__(self):
        self.brains = {}
        self.load_all_brains()
    
    def load_all_brains(self):
        logging.info("🧠 Initializing Neural Network Array...")
        for name, settings in CONFIG["SESSIONS"].items():
            path = settings["MODEL_FILE"]
            try:
                self.brains[name] = joblib.load(path)
                logging.info(f"  ✅ {name} Specialist Loaded (Thresh: {settings['THRESHOLD']})")
            except Exception as e:
                logging.error(f"  ❌ Failed to load {name} ({path}): {e}")
    
    def get_current_setup(self):
        """
        Returns the active Model, Threshold, SL, and TP for the current minute.
        """
        now_ny = datetime.datetime.now(NY_TZ)
        current_hour = now_ny.hour
        
        for name, settings in CONFIG["SESSIONS"].items():
            if current_hour in settings["HOURS"]:
                model = self.brains.get(name)
                return {
                    "name": name,
                    "model": model,
                    "threshold": settings["THRESHOLD"],
                    "sl": settings["SL"],
                    "tp": settings["TP"]
                }
        
        return None  # Market Closed or Gap Time


# ==========================================
# 3. SHARED LOGIC: OPTIMIZED TP ENGINE
# ==========================================
class OptimizedTPEngine:
    """
    Optimized take profit calculation that scales around 2.0pt base.
    Uses volatility and regime detection to adjust multiplier.
    From esbacktest002.py - proven to work with Confluence strategy.
    """
    def __init__(self, lookback: int = 60):
        self.lookback = lookback
    
    def estimate_volatility(self, prices: np.ndarray) -> float:
        """Quick volatility estimate (std of returns)"""
        if len(prices) < 2:
            return 0.5
        returns = np.diff(prices) / prices[:-1]
        return np.std(returns) if len(returns) > 0 else 0.5
    
    def estimate_garch_volatility(self, returns: np.ndarray) -> float:
        """GARCH(1,1) volatility estimate"""
        if len(returns) < 5:
            return np.std(returns) if len(returns) > 0 else 1.0
        sigma2 = np.var(returns)
        omega, alpha, beta = 0.00001, 0.1, 0.85
        for r in returns[-20:]:
            sigma2 = omega + alpha * r**2 + beta * sigma2
        return np.sqrt(sigma2)
    
    def calculate_entropy(self, returns: np.ndarray) -> float:
        """Market regime detection: trending (low) vs choppy (high)"""
        if len(returns) < 10:
            return 0.5
        hist, _ = np.histogram(returns, bins=10, density=True)
        hist = hist / (hist.sum() + 1e-10)
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log2(hist + 1e-10)) if len(hist) > 0 else 0.5
        max_entropy = np.log2(10)
        return entropy / max_entropy
    
    def calculate_optimized_tp(self, prices: np.ndarray) -> float:
        """
        Calculate TP as a multiplier around 2.0pt base.
        Returns DISTANCE in points.
        """
        BASE_TP = 2.0
        if len(prices) < 20:
            return BASE_TP
        
        vol = self.estimate_volatility(prices[-60:])
        returns = np.diff(prices) / prices[:-1]
        garch_vol = self.estimate_garch_volatility(returns[-60:] if len(returns) >= 60 else returns)
        entropy = self.calculate_entropy(returns[-60:] if len(returns) >= 60 else returns)
        
        volatility_signal = vol + (garch_vol * 0.5)
        regime_signal = entropy
        multiplier = 1.0 + (volatility_signal * 0.12) + (regime_signal * 0.08)
        multiplier = np.clip(multiplier, 0.95, 1.35)
        
        return float(np.round(BASE_TP * multiplier, 2))



# ==========================================
# 4. PROJECTX GATEWAY API CLIENT
# ==========================================
class ProjectXClient:
    """
    Client for ProjectX Gateway API (Live)
    Based on API Documentation:
    - REST API: https://gateway-api.s2f.projectx.com
    - Auth: JWT tokens via /api/Auth/loginKey (valid 24 hours)
    - Rate Limits:
        - /api/History/retrieveBars: 50 requests / 30 seconds
        - All other endpoints: 200 requests / 60 seconds
    """
    def __init__(self):
        self.session = requests.Session()
        self.token = None
        self.token_expiry = None
        self.base_url = CONFIG['REST_BASE_URL']
        self.et = pytz.timezone('US/Eastern')
        
        # Account and contract info (fetched after login)
        self.account_id = CONFIG.get('ACCOUNT_ID')
        self.contract_id = CONFIG.get('CONTRACT_ID')
        
        # Rate limiting for /History/retrieveBars: 50 requests / 30 seconds
        self.bar_fetch_timestamps = []
        self.last_bar_fetch_time = None
        self.cached_df = pd.DataFrame()
        self.last_bar_timestamp = None
        
        # Rate limit config
        self.BAR_RATE_LIMIT = 50
        self.BAR_RATE_WINDOW = 30
        self.MIN_FETCH_INTERVAL = 0.1
        
        # General rate limiting: 200 requests / 60 seconds
        self.general_request_timestamps = []
        self.GENERAL_RATE_LIMIT = 200
        self.GENERAL_RATE_WINDOW = 60
        
        # Shadow Position State (avoids unnecessary API calls)
        self._local_position = {'side': None, 'size': 0, 'avg_price': 0.0}
        
        # Stop order tracking (avoids search_orders calls)
        self._active_stop_order_id = None
    
    def _check_general_rate_limit(self) -> bool:
        """Check if we're within general rate limits"""
        now = time.time()
        self.general_request_timestamps = [
            t for t in self.general_request_timestamps
            if now - t < self.GENERAL_RATE_WINDOW
        ]
        if len(self.general_request_timestamps) >= self.GENERAL_RATE_LIMIT - 10:
            logging.warning(f"⚠️ Approaching general rate limit ({len(self.general_request_timestamps)}/{self.GENERAL_RATE_LIMIT})")
            return False
        return True
    
    def _track_general_request(self):
        """Track a general API request for rate limiting"""
        self.general_request_timestamps.append(time.time())
    
    def login(self):
        """
        Authenticate via API Key
        Endpoint: POST /api/Auth/loginKey
        Returns JWT token valid for 24 hours
        """
        url = f"{self.base_url}/api/Auth/loginKey"
        payload = {
            "userName": CONFIG['USERNAME'],
            "apiKey": CONFIG['API_KEY']
        }
        try:
            logging.info(f"🔐 Authenticating to {self.base_url}...")
            resp = self.session.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            
            # Check for API error response
            if data.get('errorCode') and data.get('errorCode') != 0:
                error_msg = data.get('errorMessage', 'Unknown Error')
                raise ValueError(f"Login Failed. ErrorCode: {data.get('errorCode')} | Msg: {error_msg}")
            
            self.token = data.get('token')
            if not self.token:
                raise ValueError("Login response missing 'token' field")
            
            # Token valid for 24 hours
            self.token_expiry = datetime.datetime.now() + datetime.timedelta(hours=24)
            
            # Set auth header for all subsequent requests
            self.session.headers.update({"Authorization": f"Bearer {self.token}"})
            logging.info("✅ Authentication successful (JWT token acquired, valid 24h)")
            self._track_general_request()
        except Exception as e:
            logging.error(f"❌ Login Failed: {e}")
            raise
    
    def validate_session(self) -> bool:
        """
        Validate and refresh session token if needed
        Endpoint: POST /api/Auth/validate
        """
        if not self._check_general_rate_limit():
            return self.token is not None
        
        url = f"{self.base_url}/api/Auth/validate"
        try:
            resp = self.session.post(url)
            self._track_general_request()
            if resp.status_code == 200:
                data = resp.json()
                if 'newToken' in data:
                    self.token = data['newToken']
                    self.token_expiry = datetime.datetime.now() + datetime.timedelta(hours=24)
                    self.session.headers.update({"Authorization": f"Bearer {self.token}"})
                    logging.info("🔄 Session token refreshed")
                return True
            else:
                logging.warning(f"Session validation failed: {resp.status_code}")
                return False
        except Exception as e:
            logging.error(f"Session validation error: {e}")
            return False
    
    def fetch_accounts(self) -> Optional[int]:
        """
        Retrieve active accounts and PROMPT USER for selection.
        """
        # If a specific ID is hardcoded in CONFIG, use it automatically (good for automation)
        if CONFIG.get('ACCOUNT_ID'):
            self.account_id = CONFIG['ACCOUNT_ID']
            logging.info(f"✅ Using Hardcoded Account ID from Config: {self.account_id}")
            return self.account_id
        
        url = f"{self.base_url}/api/Account/search"
        payload = {"onlyActiveAccounts": True}
        
        try:
            resp = self.session.post(url, json=payload)
            self._track_general_request()
            resp.raise_for_status()
            data = resp.json()
            
            if 'accounts' in data and len(data['accounts']) > 0:
                print("\n" + "="*40)
                print("📋 SELECT AN ACCOUNT TO TRADE")
                print("="*40)
                accounts = data['accounts']
                
                # Print options nicely
                for idx, acc in enumerate(accounts):
                    print(f"  [{idx + 1}] Name: {acc.get('name')}")
                    print(f"      ID: {acc.get('id')}")
                    print("-" * 30)
                
                # Loop until valid input is received
                while True:
                    try:
                        selection = input(f"👉 Enter number (1-{len(accounts)}): ")
                        choice_idx = int(selection) - 1
                        if 0 <= choice_idx < len(accounts):
                            selected_acc = accounts[choice_idx]
                            self.account_id = selected_acc.get('id')
                            print(f"✅ Selected: {selected_acc.get('name')} (ID: {self.account_id})")
                            logging.info(f"User selected account ID: {self.account_id}")
                            return self.account_id
                        else:
                            print(f"❌ Invalid number. Please enter 1-{len(accounts)}.")
                    except ValueError:
                        print("❌ Please enter a valid number.")
            else:
                logging.warning("No active accounts found")
                return None
        except Exception as e:
            logging.error(f"Failed to fetch accounts: {e}")
            return None
    
    def fetch_contracts(self) -> Optional[str]:
        """
        Get available contracts using Search to find ES futures specifically.
        Endpoint: POST /api/Contract/search
        """
        refresh_target_symbol()

        if not self._check_general_rate_limit():
            return self.contract_id

        url = f"{self.base_url}/api/Contract/search"
        # We explicitly search for "ES" to ensure we get the right list
        payload = {
            "live": False,  # Set to False to find Topstep tradable contracts
            "searchText": CONFIG.get('TARGET_SYMBOL', 'ESZ25')
        }

        try:
            logging.info(f"🔎 Searching for contracts with symbol: {payload['searchText']}...")
            resp = self.session.post(url, json=payload)
            self._track_general_request()
            resp.raise_for_status()
            data = resp.json()

            if 'contracts' in data and len(data['contracts']) > 0:
                target = CONFIG.get('TARGET_SYMBOL', 'ESZ25')
                for contract in data['contracts']:
                    contract_id = contract.get('id', '')
                    contract_name = contract.get('name', '')
                    logging.info(f"  Found: {contract_name} ({contract_id})")
                    
                    if (f".{target}." in contract_id or contract_id.endswith(f".{target}")):
                        self.contract_id = contract_id
                        logging.info(f"✅ Selected Contract ID: {self.contract_id}")
                        return self.contract_id
                
                # Fallback: Just take the first one if exact matching logic above misses
                self.contract_id = data['contracts'][0].get('id')
                logging.warning(f"⚠️ Exact match not confirmed, using first result: {self.contract_id}")
                return self.contract_id
            else:
                logging.warning("❌ No contracts found in search results.")
                return None
        except Exception as e:
            logging.error(f"Failed to fetch contracts: {e}")
            return None
    
    def get_market_data(self, lookback_minutes: int = 500, force_fetch: bool = False) -> pd.DataFrame:
        """
        Fetch historical bars with rate limiting
        Endpoint: POST /api/History/retrieveBars
        Rate Limit: 50 requests / 30 seconds
        """
        now = time.time()
        
        # Clean old timestamps
        self.bar_fetch_timestamps = [
            t for t in self.bar_fetch_timestamps
            if now - t < self.BAR_RATE_WINDOW
        ]
        
        # Check rate limit
        if len(self.bar_fetch_timestamps) >= self.BAR_RATE_LIMIT - 5:
            logging.warning(f"⚠️ Bar rate limit ({len(self.bar_fetch_timestamps)}/{self.BAR_RATE_LIMIT}). Using cache.")
            return self.cached_df
        
        # Enforce minimum interval
        if self.last_bar_fetch_time is not None:
            if now - self.last_bar_fetch_time < self.MIN_FETCH_INTERVAL and not force_fetch:
                return self.cached_df
        
        if self.contract_id is None:
            logging.error("No contract ID set. Call fetch_contracts() first.")
            return self.cached_df
        
        end_time = datetime.datetime.now(datetime.timezone.utc)
        start_time = end_time - datetime.timedelta(minutes=lookback_minutes)
        
        url = f"{self.base_url}/api/History/retrieveBars"
        payload = {
            "accountId": self.account_id,
            "contractId": self.contract_id,
            "live": False,
            "limit": 1000,
            "startTime": start_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
            "endTime": end_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
            "unit": 2,
            "unitNumber": 1
        }
        
        # Add cache-busting headers
        headers = {
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'Pragma': 'no-cache',
            'Expires': '0'
        }
        
        try:
            resp = self.session.post(url, json=payload, headers=headers)
            
            if resp.status_code == 429:
                logging.warning("🚫 Rate limited (429). Backing off 5s...")
                time.sleep(5)
                return self.cached_df
            
            resp.raise_for_status()
            self.bar_fetch_timestamps.append(now)
            self.last_bar_fetch_time = now
            data = resp.json()
            
            # DEBUG: Print last bar timestamp from raw response
            if 'bars' in data and data['bars']:
                newest_raw_bar = data['bars'][0]
                logging.debug(f"📡 API raw: newest bar t={newest_raw_bar.get('t')}, c={newest_raw_bar.get('c')}")
            
            if 'bars' in data and data['bars']:
                df = pd.DataFrame(data['bars'])
                df = df.rename(columns={
                    't': 'ts', 'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'
                })
                # FASTEST - Direct fast-path access
                df['ts'] = pd.to_datetime(df['ts'], format='ISO8601')
                df['ts'] = df['ts'].dt.tz_convert(self.et)
                df = df.set_index('ts')
                
                # API returns data in REVERSE chronological order (newest first)
                df = df.iloc[::-1]  # Reverse to get oldest->newest (chronological)
                
                logging.debug(f"📊 Final df: first={df.index[0]}, last={df.index[-1]}, len={len(df)}")
                
                self.cached_df = df
                if not df.empty:
                    new_bar_ts = df.index[-1]
                    if self.last_bar_timestamp is None or new_bar_ts > self.last_bar_timestamp:
                        self.last_bar_timestamp = new_bar_ts
                return df
            else:
                logging.warning("API returned no bars")
                return self.cached_df if not self.cached_df.empty else pd.DataFrame()
        
        except requests.exceptions.HTTPError as e:
            if hasattr(e, 'response'):
                logging.error(f"❌ HTTP Error: {e}")
                logging.error(f"❌ Server Response: {e.response.text}")
                if e.response.status_code == 429:
                    logging.warning("🚫 Rate limited. Backing off...")
                    time.sleep(5)
            else:
                logging.error(f"Data fetch error: {e}")
            return self.cached_df
        except Exception as e:
            logging.error(f"Data fetch error: {e}")
            return self.cached_df if not self.cached_df.empty else pd.DataFrame()

    def fetch_custom_bars(self, lookback_bars: int, minutes_per_bar: int) -> pd.DataFrame:
            """
            Fetch historical bars with custom timeframe (for HTF analysis).
            minutes_per_bar: 60 for 1H, 240 for 4H.
            """
            # Rate limit check (reuse existing logic)
            if not self._check_general_rate_limit():
                return pd.DataFrame()

            end_time = datetime.datetime.now(datetime.timezone.utc)
            # Calculate start time based on bars needed * minutes per bar
            total_mins = lookback_bars * minutes_per_bar
            start_time = end_time - datetime.timedelta(minutes=total_mins + 1000) # Buffer

            url = f"{self.base_url}/api/History/retrieveBars"
            payload = {
                "accountId": self.account_id,
                "contractId": self.contract_id,
                "live": False,
                "limit": lookback_bars + 50, # Request slightly more to be safe
                "startTime": start_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
                "endTime": end_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
                "unit": 2,              # 2 = Minutes
                "unitNumber": minutes_per_bar  # 60 or 240
            }

            try:
                resp = self.session.post(url, json=payload)
                self._track_general_request()
            
                if resp.status_code == 200:
                    data = resp.json()
                    if 'bars' in data and data['bars']:
                        df = pd.DataFrame(data['bars'])
                        df = df.rename(columns={'t': 'ts', 'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
                        df['ts'] = pd.to_datetime(df['ts'])
                        df = df.set_index('ts').sort_index()
                        return df
                return pd.DataFrame()
            except Exception as e:
                logging.error(f"HTF Data fetch error: {e}")
                return pd.DataFrame()
            
    def get_rate_limit_status(self) -> str:
        """Returns current rate limit usage"""
        now = time.time()
        bar_recent = len([t for t in self.bar_fetch_timestamps if now - t < self.BAR_RATE_WINDOW])
        general_recent = len([t for t in self.general_request_timestamps if now - t < self.GENERAL_RATE_WINDOW])
        return f"Bars: {bar_recent}/{self.BAR_RATE_LIMIT} (30s) | General: {general_recent}/{self.GENERAL_RATE_LIMIT} (60s)"
    
    def place_order(self, signal: Dict, current_price: float):
        """
        Place a market order with brackets using SIGNED RELATIVE TICKS.
        
        CRITICAL FIX: 
        - Long TP is positive (+), Long SL is negative (-)
        - Short TP is negative (-), Short SL is positive (+)
        This satisfies the engine requirement: "Ticks should be less than zero when going short."
        """
        if not self._check_general_rate_limit():
            logging.error("Rate limit reached, cannot place order")
            return
        
        if self.account_id is None:
            logging.error("No account ID set. Call fetch_accounts() first.")
            return
        
        if self.contract_id is None:
            logging.error("No contract ID set. Call fetch_contracts() first.")
            return
        
        url = f"{self.base_url}/api/Order/place"
        
        # 1. Determine Side (0=Buy, 1=Sell)
        is_long = (signal['side'] == "LONG")
        side_code = 0 if is_long else 1
        
        # 2. Calculate Ticks Distance (Absolute)
        # ES/MES tick size is 0.25
        sl_points = float(signal['sl_dist'])
        tp_points = float(signal['tp_dist'])
        
        # If sl_dist and tp_dist are already in POINTS:
        abs_sl_ticks = int(abs(sl_points / 0.25))  # Convert points to ticks

        # OR if they're already in TICKS:
        abs_sl_ticks = int(abs(sl_points))  # Use directly
        abs_tp_ticks = int(abs(tp_points))  # Use directly

        
        # 3. Apply Directional Signs based on Side
        if is_long:
            # LONG: Profit is UP (+), Stop is DOWN (-)
            final_tp_ticks = abs_tp_ticks
            final_sl_ticks = -abs_sl_ticks
        else:
            # SHORT: Profit is DOWN (-), Stop is UP (+)
            final_tp_ticks = -abs_tp_ticks
            final_sl_ticks = abs_sl_ticks

        # 4. Generate Unique Client Order ID
        unique_order_id = str(uuid.uuid4())

        # 5. Construct Payload
        # Using 'ticks' with the correct signs calculated above
        payload = {
            "accountId": self.account_id,
            "contractId": self.contract_id,
            "clOrdId": unique_order_id,
            "type": 2,  # Market Order
            "side": side_code,
            "size": 1,  # Fixed size per your config
            "stopLossBracket": {
                "type": 4,      # Stop Market
                "ticks": final_sl_ticks
            },
            "takeProfitBracket": {
                "type": 1,      # Limit
                "ticks": final_tp_ticks
            }
        }
        
        try:
            # Log exact details for verification
            direction_str = "UP (+)" if is_long else "DOWN (-)"
            logging.info(f"🚀 SENDING ORDER: {signal['side']} @ ~{current_price:.2f}")
            logging.info(f"   TP: {tp_points}pts ({final_tp_ticks} ticks)")
            logging.info(f"   SL: {sl_points}pts ({final_sl_ticks} ticks)")
            
            resp = self.session.post(url, json=payload)
            self._track_general_request()
            
            if resp.status_code == 429:
                logging.error("🚫 Rate limited on order placement!")
                return None
            
            resp_data = resp.json()
            
            if resp.status_code == 200:
                # Check for business logic success
                if resp_data.get('success') is False:
                     err_msg = resp_data.get('errorMessage', 'Unknown Rejection')
                     logging.error(f"❌ Order Rejected by Engine: {err_msg}")
                     return None
                else:
                     logging.info(f"✅ Order Placed Successfully [{unique_order_id[:8]}]")
                     
                     # Update shadow position state
                     self._local_position = {
                         'side': signal['side'],
                         'size': 1,  # Fixed size
                         'avg_price': current_price
                     }
                     
                     # Try to capture stop order ID from response if available
                     # The exact field name depends on the API response structure
                     if 'stopLossOrderId' in resp_data:
                         self._active_stop_order_id = resp_data['stopLossOrderId']
                         logging.debug(f"📝 Captured stop order ID: {self._active_stop_order_id}")
                     elif 'orderId' in resp_data:
                         # Main order ID - we'll still need to search for bracket orders
                         logging.debug(f"📝 Main order ID: {resp_data['orderId']}")
                     
                     return resp_data
            else:
                logging.error(f"❌ HTTP Error {resp.status_code}: {resp.text}")
                return None
                
        except Exception as e:
            logging.error(f"❌ Order exception: {e}")
            return None
            
    def get_position(self) -> Dict:
        """
        Get current position. Tries Search (POST) first, then GET fallback.
        UPDATED: Treats 404 as 'Flat Position' to stop log errors when no trades are open.
        """
        if not self._check_general_rate_limit():
            return {'side': None, 'size': 0, 'avg_price': 0.0}
        
        if self.account_id is None:
            return {'side': None, 'size': 0, 'avg_price': 0.0}
        
        # Primary: POST /api/Position/search
        url = f"{self.base_url}/api/Position/search"
        payload = {"accountId": self.account_id}
        
        try:
            resp = self.session.post(url, json=payload)
            self._track_general_request()
            
            # If 404, try Fallback Endpoint (GET /api/Position) - ORIGINAL FEATURE KEPT
            if resp.status_code == 404:
                # logging.debug("⚠️ Position/search 404. Trying GET fallback...") 
                fallback_url = f"{self.base_url}/api/Position"
                resp = self.session.get(fallback_url, params=payload)
                self._track_general_request()

            # Handle Success (200)
            if resp.status_code == 200:
                data = resp.json()
                # Handle different response structures (list vs dict)
                positions = data.get('positions', data) if isinstance(data, dict) else data
                
                # Find position for our contract
                for pos in positions:
                    if pos.get('contractId') == self.contract_id:
                        size = pos.get('size', 0)
                        avg_price = pos.get('averagePrice', 0.0)
                        if size > 0:
                            return {'side': 'LONG', 'size': size, 'avg_price': avg_price}
                        elif size < 0:
                            return {'side': 'SHORT', 'size': abs(size), 'avg_price': avg_price}
                
                # If 200 OK but contract not in list -> We are Flat
                return {'side': None, 'size': 0, 'avg_price': 0.0}

            # Handle 404 on Fallback (The Fix: This means NO positions exist = Flat)
            elif resp.status_code == 404:
                # Valid state: User has no open positions. Return clean flat state.
                return {'side': None, 'size': 0, 'avg_price': 0.0}

            # Handle Actual Errors
            else:
                logging.warning(f"Position check failed: {resp.status_code} - {resp.text}")
                return {'side': None, 'size': 0, 'avg_price': 0.0}

        except Exception as e:
            logging.error(f"Position check error: {e}")
            return {'side': None, 'size': 0, 'avg_price': 0.0}
    
    def close_position(self, position: Dict) -> bool:
        """
        Close an existing position by placing an opposite market order
        """
        if position['side'] is None or position['size'] == 0:
            return True  # Nothing to close
            
        if not self._check_general_rate_limit():
            logging.error("Rate limit reached, cannot close position")
            return False
        
        url = f"{self.base_url}/api/Order/place"
        
        # To close: sell if long, buy if short
        if position['side'] == 'LONG':
            side_code = 1  # Sell to close long
            action = "SELL"
        else:
            side_code = 0  # Buy to close short
            action = "BUY"
        
        payload = {
            "accountId": self.account_id,
            "contractId": self.contract_id,
            "type": 2,  # Market Order
            "side": side_code,
            "size": position['size']
            # No brackets - just close the position
        }
        
        try:
            logging.info(f"🔄 CLOSING POSITION: {action} {position['size']} contracts to close {position['side']} @ ~{position['avg_price']:.2f}")
            resp = self.session.post(url, json=payload)
            self._track_general_request()
            
            if resp.status_code == 429:
                logging.error("🚫 Rate limited on position close!")
                return False
            
            resp_data = resp.json()
            if resp.status_code == 200 and resp_data.get('success', False):
                logging.info(f"✅ Position close order submitted: {resp_data}")
                
                # Reset shadow position state
                self._local_position = {'side': None, 'size': 0, 'avg_price': 0.0}
                self._active_stop_order_id = None
                
                return True
            else:
                logging.error(f"❌ Position close failed: {resp_data}")
                return False
        except Exception as e:
            logging.error(f"❌ Position close exception: {e}")
            return False
    
    def close_and_reverse(self, new_signal: Dict, current_price: float, opposite_signal_count: int) -> Tuple[bool, int]:
        """
        Check current position, close if 3 opposite signals received, then place new order.
        Uses shadow position state to reduce API calls.
        """
        # OPTIMIZATION: Check local shadow position first
        if self._local_position['side'] == new_signal['side']:
            logging.debug(f"Shadow position shows already {new_signal['side']}. Skipping API check.")
            return True, 0  # Already in same direction
        
        # Only fetch real position if we intend to switch or local state is uncertain
        position = self.get_position()
        
        # Update shadow position from API response
        self._local_position = position.copy()
        
        # If no position, just place the order and reset count
        if position['side'] is None:
            self.place_order(new_signal, current_price)
            return True, 0
        
        # If signal is SAME direction as position, place order and reset count
        if position['side'] == new_signal['side']:
            self.place_order(new_signal, current_price)
            return True, 0
        
        # Signal is OPPOSITE direction - increment counter
        opposite_signal_count += 1
        logging.info(f"⚠️ OPPOSITE SIGNAL #{opposite_signal_count}/3: Current {position['side']} {position['size']} contracts, Signal: {new_signal['side']}")
        
        # If we've received 3 opposite signals, close and reverse
        if opposite_signal_count >= 3:
            logging.info(f"🔄 3 OPPOSITE SIGNALS RECEIVED - Closing {position['side']} position and reversing to {new_signal['side']}")
            
            # Close the existing position
            close_success = self.close_position(position)
            if not close_success:
                logging.error("Failed to close existing position, aborting new order")
                return False, opposite_signal_count
            
            # Small delay to let the close order process
            time.sleep(0.5)
            
            # Place the new order
            self.place_order(new_signal, current_price)
            return True, 0  # Reset counter after closing
        
        # Not yet 3 signals - don't place order, just return updated count
        logging.info(f"⏳ Waiting for {3 - opposite_signal_count} more opposite signals before closing position")
        return False, opposite_signal_count
    
    def search_orders(self) -> List[Dict]:
        """
        Search for open orders (bracket orders) for the account.
        Endpoint: POST /api/Order/search
        Returns: List of order dicts with orderId, type, side, price, etc.
        """
        if not self._check_general_rate_limit():
            return []
        
        if self.account_id is None:
            return []
        
        url = f"{self.base_url}/api/Order/search"
        payload = {"accountId": self.account_id}
        
        try:
            resp = self.session.post(url, json=payload)
            self._track_general_request()
            
            if resp.status_code == 200:
                data = resp.json()
                orders = data.get('orders', [])
                # Filter for our contract
                return [o for o in orders if o.get('contractId') == self.contract_id]
            elif resp.status_code == 400:
                # 400 often means no orders exist - treat as empty
                logging.debug("Order search returned 400 - treating as no orders")
                return []
            elif resp.status_code == 404:
                # 404 means no orders found - valid empty state
                return []
            else:
                logging.warning(f"Order search failed: {resp.status_code} - {resp.text}")
                return []
        except Exception as e:
            logging.error(f"Order search error: {e}")
            return []
    
    def cancel_order(self, order_id: int) -> bool:
        """
        Cancel an order by ID.
        Endpoint: POST /api/Order/cancel
        """
        if not self._check_general_rate_limit():
            return False
        
        url = f"{self.base_url}/api/Order/cancel"
        payload = {
            "accountId": self.account_id,  # Include accountId for API
            "orderId": order_id
        }
        
        try:
            resp = self.session.post(url, json=payload)
            self._track_general_request()
            if resp.status_code == 200:
                data = resp.json()
                if data.get('success', False):
                    logging.info(f"✅ Order {order_id} cancelled")
                    return True
                else:
                    err = data.get('errorMessage', 'Unknown error')
                    logging.warning(f"Order cancel rejected: {err}")
                    return False
            logging.warning(f"Order cancel failed: {resp.status_code} - {resp.text}")
            return False
        except Exception as e:
            logging.error(f"Order cancel error: {e}")
            return False
    
    def modify_order(self, order_id: int, stop_price: float = None, limit_price: float = None, size: int = None) -> bool:
        """
        Modify an existing order using the /api/Order/modify endpoint.
        
        Args:
            order_id: The order ID to modify
            stop_price: New stop price (for stop orders)
            limit_price: New limit price (for limit orders)
            size: New size (optional)
        
        Returns:
            True if modification successful
        """
        if not self._check_general_rate_limit():
            logging.error("Rate limit reached, cannot modify order")
            return False
        
        url = f"{self.base_url}/api/Order/modify"
        
        payload = {
            "accountId": self.account_id,
            "orderId": order_id
        }
        
        # Only include fields that are being modified
        if size is not None:
            payload["size"] = size
        if limit_price is not None:
            payload["limitPrice"] = limit_price
        if stop_price is not None:
            payload["stopPrice"] = stop_price
        
        try:
            logging.info(f"📝 MODIFYING ORDER {order_id}: stopPrice={stop_price}, limitPrice={limit_price}")
            resp = self.session.post(url, json=payload)
            self._track_general_request()
            
            if resp.status_code == 200:
                data = resp.json()
                if data.get('success', False):
                    logging.info(f"✅ Order {order_id} modified successfully")
                    return True
                else:
                    err = data.get('errorMessage', str(data))
                    logging.error(f"❌ Order modification rejected: {err}")
                    return False
            else:
                logging.error(f"❌ Order modification failed: {resp.status_code} - {resp.text}")
                return False
                
        except Exception as e:
            logging.error(f"❌ Order modification exception: {e}")
            return False
    
    def modify_stop_to_breakeven(self, entry_price: float, side: str, known_size: int = None, stop_order_id: int = None) -> bool:
        """
        Modify stop to break-even using the /api/Order/modify endpoint.
        
        Args:
            entry_price: Original entry price
            side: 'LONG' or 'SHORT'
            known_size: Known position size (avoids API call)
            stop_order_id: Known stop order ID (avoids search_orders call)
        
        Returns:
            True if break-even was set successfully
        """
        # 1. Determine position size
        position_size = known_size if known_size is not None else 1
        
        if position_size == 0:
            logging.warning("❌ No position size provided. Aborting BE.")
            return False

        # 2. Calculate break-even price with buffer
        buffer = 0.25  # 1 tick buffer
        if side == 'LONG':
            be_price = round((entry_price + buffer) * 4) / 4
        else:  # SHORT
            be_price = round((entry_price - buffer) * 4) / 4

        logging.info(f"🔒 BREAK-EVEN: Moving Stop to {be_price:.2f} for {position_size} contracts")

        # 3. Find stop order ID (use cached if available)
        target_stop_id = stop_order_id or self._active_stop_order_id
        
        if target_stop_id is None:
            # Fall back to searching for stop order
            orders = self.search_orders()
            for order in orders:
                order_type = order.get('type', 0)
                order_status = str(order.get('status', '')).lower()
                if order_type in [4, 5] and order_status in ['working', 'pending', 'accepted', 'active']:
                    target_stop_id = order.get('orderId')
                    current_stop = order.get('price', 0)
                    
                    # Smart check: Don't move if we are already better than BE
                    if (side == 'LONG' and current_stop >= be_price) or (side == 'SHORT' and current_stop <= be_price):
                        logging.info(f"✋ Stop already at better price ({current_stop}). Skipping.")
                        return True
                    break
        
        if target_stop_id is None:
            logging.warning("❌ No stop order found to modify. Placing new stop.")
            return self._place_breakeven_stop(be_price, side, position_size)
        
        # 4. Use modify_order API instead of cancel+place
        if self.modify_order(target_stop_id, stop_price=be_price):
            logging.info(f"✅ BREAK-EVEN SET: Stop modified to {be_price:.2f}")
            return True
        else:
            # Fallback: cancel old stop and place new one
            logging.warning("⚠️ Order modify failed. Falling back to cancel+place.")
            self.cancel_order(target_stop_id)
            time.sleep(0.3)
            return self._place_breakeven_stop(be_price, side, position_size)
        
    def _place_breakeven_stop(self, be_price: float, side: str, size: int) -> bool:
        """
        Internal method to place a new stop order at break-even price.

        Args:
            be_price: Break-even price for the stop
            side: 'LONG' or 'SHORT' (determines stop side)
            size: Position size

        Returns:
            True if stop successfully placed
        """
        if not self._check_general_rate_limit():
            logging.error("Rate limit reached, cannot place break-even stop")
            return False

        url = f"{self.base_url}/api/Order/place"

        # For break-even stop:
        # - If LONG position, we need a SELL stop (side=1)
        # - If SHORT position, we need a BUY stop (side=0)
        side_code = 1 if side == 'LONG' else 0

        payload = {
            "accountId": self.account_id,
            "contractId": self.contract_id,
            "type": 4,  # Stop Market
            "side": side_code,
            "size": size,
            "stopPrice": be_price  # FIXED: Use stopPrice, not price
        }

        try:
            logging.info(f"Placing break-even stop: {be_price:.2f} for {size} contracts ({side} position)")
            resp = self.session.post(url, json=payload)
            self._track_general_request()

            if resp.status_code == 200:
                data = resp.json()
                if data.get('success', False):
                    logging.info(f"✓ BREAK-EVEN STOP PLACED at {be_price:.2f}")
                    # Capture the new stop order ID
                    if 'orderId' in data:
                        self._active_stop_order_id = data['orderId']
                    return True
                else:
                    err = data.get('errorMessage', str(data))
                    logging.error(f"Break-even stop rejected: {err}")
                    return False
            else:
                logging.error(f"Break-even stop failed: {resp.status_code} - {resp.text}")
                return False

        except Exception as e:
            logging.error(f"Break-even stop exception: {e}")
            return False

    def _try_bracket_modification(self, entry_price: float, side: str, size: int) -> bool:
        """Alternative method: try to modify stop via position bracket endpoint."""
        be_price = round(entry_price * 4) / 4
        
        # Try direct stop placement without cancelling first
        url = f"{self.base_url}/api/Order/place"
        side_code = 1 if side == 'LONG' else 0
        
        payload = {
            "accountId": self.account_id,
            "contractId": self.contract_id,
            "type": 4,  # Stop Market
            "side": side_code,
            "size": size,
            "stopPrice": be_price  # FIXED: Use stopPrice
        }
        
        try:
            logging.info(f"🔒 BREAK-EVEN (alt): Placing stop at {be_price:.2f} for {size} contracts")
            resp = self.session.post(url, json=payload)
            self._track_general_request()
            
            if resp.status_code == 200:
                data = resp.json()
                if data.get('success', False):
                    logging.info(f"✅ Break-even stop placed at {be_price:.2f}")
                    return True
                else:
                    err = data.get('errorMessage', str(data))
                    logging.error(f"❌ Break-even alt rejected: {err}")
            else:
                logging.error(f"❌ Break-even alt failed: {resp.status_code}")
            return False
        except Exception as e:
            logging.error(f"❌ Break-even alt exception: {e}")
            return False

# ==========================================
# 5. STRATEGY BASE CLASS
# ==========================================
class Strategy(ABC):
    @abstractmethod
    def on_bar(self, df: pd.DataFrame) -> Optional[Dict]:
        pass

# ==========================================
# 7. ORB STRATEGY - LONG ONLY + ORB < 5pt
# ==========================================
"""
From orb_long_orb5_strategy.py:
- 15-minute Opening Range Breakout (9:30-9:45 ET)
- LONG entries only
- ORB range must be < 5 points
- Midpoint must be retested before entry
- Price must be above daily open
- Morning session only (9:30-11:30 ET)
- TP: 5.0pt, SL: 3.5pt
"""

class OrbStrategy(Strategy):
    """ORB Strategy - Now uses DynamicSLTPEngine for SL/TP"""
    def __init__(self):
        self.reset_daily()
        
    def reset_daily(self):
        self.orb_high = None
        self.orb_low = None
        self.orb_mid = None
        self.orb_range = None
        self.mid_retested = False
        self.orb_complete = False
        self.daily_open = None
        self.current_date = None
        self.trade_taken_today = False

    def on_bar(self, df: pd.DataFrame) -> Optional[Dict]:
        if len(df) < 2:
            return None
            
        ts = df.index[-1]
        curr = df.iloc[-1]
        curr_date = ts.date()
        
        if self.current_date != curr_date:
            self.reset_daily()
            self.current_date = curr_date
            
        t = ts.time()
        
        if self.daily_open is None and t >= datetime.time(9, 30):
            self.daily_open = curr['open']

        if datetime.time(9, 30) <= t < datetime.time(9, 45):
            if self.orb_high is None:
                self.orb_high = curr['high']
                self.orb_low = curr['low']
            else:
                self.orb_high = max(self.orb_high, curr['high'])
                self.orb_low = min(self.orb_low, curr['low'])
        
        if t >= datetime.time(9, 45) and not self.orb_complete and self.orb_high is not None:
            self.orb_range = self.orb_high - self.orb_low
            self.orb_mid = (self.orb_high + self.orb_low) / 2.0
            self.orb_complete = True
            
            if self.orb_range >= 15.0:
                self.orb_complete = False
                logging.info(f"ORB: Range {self.orb_range:.2f} >= 15.0pt. Strategy disabled for day.")
                
        if not self.orb_complete:
            return None
        if not (datetime.time(9, 45) <= t <= datetime.time(11, 30)):
            return None
        if self.trade_taken_today:
            return None
        if self.daily_open is None:
            return None
            
        if not self.mid_retested:
            if curr['low'] <= self.orb_mid <= curr['high']:
                self.mid_retested = True
                logging.info(f"ORB: Midpoint {self.orb_mid:.2f} retested")
        
        if self.mid_retested and len(df) >= 2:
            prev_close = df.iloc[-2]['close']
            curr_close = curr['close']
            
            if (prev_close <= self.orb_high) and (curr_close > self.orb_high):
                if curr_close > self.daily_open:
                    self.trade_taken_today = True
                    sltp = dynamic_sltp_engine.calculate_sltp("Confluence", df)
                    sltp = dynamic_sltp_engine.calculate_sltp("IntradayDip", df)
                    sltp = dynamic_sltp_engine.calculate_sltp("RegimeAdaptive", df)
                    logging.info(f"ORB: LONG signal - Breakout above {self.orb_high:.2f}")
                    dynamic_sltp_engine.log_params(sltp)
                    return {
                        "strategy": "ORB_Long",
                        "side": "LONG",
                        "tp_dist": sltp['tp_dist'],
                        "sl_dist": sltp['sl_dist']
                    }
        
        return None


# ==========================================
# 8. INTRADAY DIP STRATEGY
# ==========================================
"""
From intraday_dip_strategy.py:
- LONG: Price down >= 1.0% from session open, z-score < -0.5, volatility spike
- SHORT: Price up >= 1.25% from session open, z-score > 1.0, volatility spike
- TP: 5.0pt, SL: 5.0pt
"""

class IntradayDipStrategy(Strategy):
    """
    IntradayDip Strategy - Now uses DynamicSLTPEngine
    - LONG: Price down >= 1.0% from session open, z-score < -0.5, volatility spike
    - SHORT: Price up >= 1.25% from session open, z-score > 1.0, volatility spike
    """
    def __init__(self):
        self.session_open = None
        self.current_date = None
        
    def on_bar(self, df: pd.DataFrame) -> Optional[Dict]:
        if len(df) < 20:
            return None
            
        ts = df.index[-1]
        curr = df.iloc[-1]
        
        # Reset on new day
        if self.current_date != ts.date():
            self.session_open = None
            self.current_date = ts.date()
        
        # Capture session open at 9:30 ET
        if ts.time() == datetime.time(9, 30):
            self.session_open = curr['open']
            
        if self.session_open is None:
            return None
        
        # Calculate intraday % change from session open
        pct_change = (curr['close'] - self.session_open) / self.session_open * 100
        
        # Z-score calculation
        sma20 = df['close'].rolling(20).mean().iloc[-1]
        std20 = df['close'].rolling(20).std().iloc[-1]
        if std20 == 0:
            return None
        z_score = (curr['close'] - sma20) / std20
        
        # Volatility spike detection
        range_series = df['high'] - df['low']
        range_sma = range_series.rolling(20).mean().iloc[-1]
        curr_range = curr['high'] - curr['low']
        is_vol_spike = curr_range > range_sma
        
        # Get dynamic SL/TP
        sltp = dynamic_sltp_engine.calculate_dynamic_sltp(df)
        
        # LONG: Down 1%+, oversold (z < -0.5), volatility spike
        if (pct_change <= -1.0) and (z_score < -0.5) and is_vol_spike:
            logging.info(f"IntradayDip: LONG signal - Down {pct_change:.2f}%, Z={z_score:.2f}")
            dynamic_sltp_engine.log_params(sltp)
            return {"strategy": "IntradayDip", "side": "LONG", 
                    "tp_dist": sltp['tp_dist'], "sl_dist": sltp['sl_dist']}
        
        # SHORT: Up 1.25%+, overbought (z > 1.0), volatility spike
        if (pct_change >= 1.25) and (z_score > 1.0) and is_vol_spike:
            logging.info(f"IntradayDip: SHORT signal - Up {pct_change:.2f}%, Z={z_score:.2f}")
            dynamic_sltp_engine.log_params(sltp)
            return {"strategy": "IntradayDip", "side": "SHORT", 
                    "tp_dist": sltp['tp_dist'], "sl_dist": sltp['sl_dist']}
        
        return None


# ==========================================
# 9. CONFLUENCE STRATEGY (STRICT)
# ==========================================
"""
From esbacktest002.py - STRICT implementation:
- Liquidity sweep (previous session high/low) + re-cross back inside
- Hourly body-gap FVG (not wick-based)
- Bank level ($12.50 grid, ±0.25 tolerance)
- Macro window filter: only minutes 0-10 or 50-59 of each hour
- Day filter: Remove Tuesday/Thursday
- Time filter: Remove 08:50-08:59 and 13:50-13:59

BACKTEST RESULTS (strict logic):
- 31 trades over 1 year with 64.5% win rate
- Win rate stays constant regardless of TP (all wins hit even 5pt TP)
- Optimal: TP=5.0pt, SL=2.0pt -> Best risk/reward
"""

class ConfluenceStrategy(Strategy):
    def __init__(self):
        self.et = pytz.timezone('US/Eastern')
        # Session tracking
        self.current_session = None
        self.prev_session_high = None
        self.prev_session_low = None
        self.session_high = None
        self.session_low = None
        self.bear_swept_this_session = False
        self.bull_swept_this_session = False
        # Hourly FVG tracking
        self.last_hour_processed = None
        self.bull_fvg_low = None
        self.bull_fvg_high = None
        self.bear_fvg_low = None
        self.bear_fvg_high = None
        # Fixed TP/SL based on backtest optimization
        self.TAKE_PROFIT = 5.0  # Backtest showed 64.5% WR holds for all TP values up to 5pt
        self.STOP_LOSS = 2.0
        
    def _get_session_date(self, ts) -> datetime.date:
        """Assign session date: if before 18:00, belongs to previous day's session"""
        hour = ts.hour
        dte = ts.date()
        if hour >= 18:
            return dte
        else:
            return dte - datetime.timedelta(days=1)
    
    def _near_bank_level(self, price: float) -> bool:
        """Check if price is near a $12.50 bank level (±0.25 tolerance)"""
        BANK_GRID = 12.5
        BANK_TOL = 0.25
        remainder = price % BANK_GRID
        distance = min(remainder, BANK_GRID - remainder)
        return distance <= BANK_TOL
    
    def _in_macro_window(self, minute: int) -> bool:
        """Check if minute is in macro window (0-10 or 50-59)"""
        return (minute <= 10) or (minute >= 50)
    
    def _update_hourly_fvg(self, df: pd.DataFrame) -> None:
        """Update hourly body-gap FVGs (STRICT: body-based, not wick-based)"""
        # Resample to hourly
        h1 = df.resample('1h').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }).dropna()
        
        if len(h1) < 2:
            return
            
        current_hour = df.index[-1].floor('h')
        if self.last_hour_processed == current_hour:
            return
        self.last_hour_processed = current_hour
        
        # Get last 2 complete hourly candles
        curr_h = h1.iloc[-1]
        prev_h = h1.iloc[-2]
        
        # STRICT: Body calculations (not wick-based)
        curr_body_hi = max(curr_h['open'], curr_h['close'])
        curr_body_lo = min(curr_h['open'], curr_h['close'])
        prev_body_hi = max(prev_h['open'], prev_h['close'])
        prev_body_lo = min(prev_h['open'], prev_h['close'])
        
        # Bullish body gap: current body low > previous body high (gap up)
        if curr_body_lo > prev_body_hi:
            self.bull_fvg_low = prev_body_hi
            self.bull_fvg_high = curr_body_lo
            logging.debug(f"Confluence: Bullish FVG detected {self.bull_fvg_low:.2f} - {self.bull_fvg_high:.2f}")
        
        # Bearish body gap: current body high < previous body low (gap down)
        if curr_body_hi < prev_body_lo:
            self.bear_fvg_low = curr_body_hi
            self.bear_fvg_high = prev_body_lo
            logging.debug(f"Confluence: Bearish FVG detected {self.bear_fvg_low:.2f} - {self.bear_fvg_high:.2f}")
    
    def on_bar(self, df: pd.DataFrame) -> Optional[Dict]:
        if len(df) < 120:
            return None
            
        ts = df.index[-1]
        curr = df.iloc[-1]
        price = curr['close']
        
        # ========== STRICT DAY/TIME FILTERS ==========
        
        # Filter 1: Remove Tuesday/Thursday
        weekday = ts.weekday()
        if weekday in [1, 3]:  # Tuesday=1, Thursday=3
            return None
        
        # Filter 2: Remove 08:50-08:59
        hour = ts.hour
        minute = ts.minute
        if hour == 8 and minute >= 50:
            return None
        
        # Filter 3: Remove 13:50-13:59
        if hour == 13 and minute >= 50:
            return None
        
        # Filter 4: Macro window (only minutes 0-10 or 50-59)
        if not self._in_macro_window(minute):
            return None
        
        # ========== BANK LEVEL CHECK ==========
        if not self._near_bank_level(price):
            return None
        
        # ========== SESSION TRACKING ==========
        session_date = self._get_session_date(ts)
        
        if self.current_session != session_date:
            # New session: save previous session H/L
            self.prev_session_high = self.session_high
            self.prev_session_low = self.session_low
            self.session_high = curr['high']
            self.session_low = curr['low']
            self.current_session = session_date
            self.bear_swept_this_session = False
            self.bull_swept_this_session = False
        else:
            # Update session H/L
            self.session_high = max(self.session_high, curr['high']) if self.session_high else curr['high']
            self.session_low = min(self.session_low, curr['low']) if self.session_low else curr['low']
        
        # Need previous session data
        if self.prev_session_high is None or self.prev_session_low is None:
            return None
        
        # ========== SWEEP DETECTION (wick-based with re-cross) ==========
        
        # Bear sweep: wick above previous session high
        if curr['high'] > self.prev_session_high:
            self.bear_swept_this_session = True
        
        # Bull sweep: wick below previous session low
        if curr['low'] < self.prev_session_low:
            self.bull_swept_this_session = True
        
        # Re-cross back inside after sweep
        back_inside_after_bear = self.bear_swept_this_session and (curr['close'] < self.prev_session_high)
        back_inside_after_bull = self.bull_swept_this_session and (curr['close'] > self.prev_session_low)
        
        # ========== HOURLY FVG TRACKING ==========
        self._update_hourly_fvg(df)
        
        # Check if price is inside FVG
        in_bull_fvg = False
        if self.bull_fvg_low is not None and self.bull_fvg_high is not None:
            in_bull_fvg = self.bull_fvg_low <= price <= self.bull_fvg_high
        
        in_bear_fvg = False
        if self.bear_fvg_low is not None and self.bear_fvg_high is not None:
            in_bear_fvg = self.bear_fvg_low <= price <= self.bear_fvg_high
        
        # ========== SIGNAL GENERATION ==========
        
        # Get dynamic SL/TP
        sltp = dynamic_sltp_engine.calculate_dynamic_sltp(df)
        
        # LONG: Bull sweep (took low) + re-crossed back inside + in bullish FVG + at bank level
        if back_inside_after_bull and in_bull_fvg:
            logging.info(f"Confluence: LONG signal - Bull sweep + FVG + Bank level @ {price:.2f}")
            dynamic_sltp_engine.log_params(sltp)
            return {
                "strategy": "Confluence", 
                "side": "LONG", 
                "tp_dist": sltp['tp_dist'], 
                "sl_dist": sltp['sl_dist']
            }
        
        # SHORT: Bear sweep (took high) + re-crossed back inside + in bearish FVG + at bank level
        if back_inside_after_bear and in_bear_fvg:
            logging.info(f"Confluence: SHORT signal - Bear sweep + FVG + Bank level @ {price:.2f}")
            dynamic_sltp_engine.log_params(sltp)
            return {
                "strategy": "Confluence", 
                "side": "SHORT", 
                "tp_dist": sltp['tp_dist'], 
                "sl_dist": sltp['sl_dist']
            }
        
        return None


# ==========================================
# 10. ICT MODEL STRATEGY - LONG ONLY
# ==========================================
"""
From ict002.py:
- HTF Bias: 4H candle (10AM open) + 5m FVG structure respect/disrespect
- Key Levels: PDL (Previous Day Low) + 5m Bullish FVGs
- Manipulation: Price sweeps/touches key level
- Entry Trigger: 1-minute Inversion FVG (bearish FVG closed above)
- Filters: NY AM session (9:30-11:00 ET), LRL filter (no equal lows)
- TP: 5.0pt, SL: 3.5pt (asymmetric R:R)
"""

class ICTModelStrategy(Strategy):
    def __init__(self):
        self.reset_daily()
        # 5m FVG state
        self.curr_bull_fvg_low = np.nan
        self.curr_bull_fvg_high = np.nan
        self.curr_bear_fvg_low = np.nan
        self.curr_bear_fvg_high = np.nan
        self.fvg_bias = None
        self.last_5m_bar_ts = None
        # 1m bearish FVG for inversion trigger
        self.bear_fvg_1m_active = False
        self.bear_fvg_1m_high = np.nan
        
    def reset_daily(self):
        """Reset daily state"""
        self.current_date = None
        self.pdl = None
        self.pdh = None
        self.open_10am = None
        # Manipulation/setup tracking
        self.pending_long = False
        self.long_stop = np.nan
        self.setup_bar_count = 0
        
    def _is_in_ny_am_session(self, ts) -> bool:
        """Check if timestamp is within NY AM session (9:30-11:00 ET)"""
        t = ts.time()
        return (
            (t.hour == 9 and t.minute >= 30) or
            (t.hour == 10) or
            (t.hour == 11 and t.minute == 0)
        )
    
    def _update_pdh_pdl(self, df: pd.DataFrame, current_date) -> None:
        """Calculate Previous Day High/Low from historical data"""
        df_reset = df.reset_index()
        df_reset['date'] = df_reset['ts'].dt.date
        
        dates = sorted(df_reset['date'].unique())
        if len(dates) >= 2:
            prev_dates = [d for d in dates if d < current_date]
            if prev_dates:
                prev_date = prev_dates[-1]
                prev_day_data = df_reset[df_reset['date'] == prev_date]
                if not prev_day_data.empty:
                    self.pdh = prev_day_data['high'].max()
                    self.pdl = prev_day_data['low'].min()
    
    def _update_10am_open(self, df: pd.DataFrame, current_date) -> None:
        """Get the 10AM candle open for the current day"""
        df_reset = df.reset_index()
        df_reset['date'] = df_reset['ts'].dt.date
        df_reset['hour'] = df_reset['ts'].dt.hour
        df_reset['minute'] = df_reset['ts'].dt.minute
        
        today_10am = df_reset[
            (df_reset['date'] == current_date) & 
            (df_reset['hour'] == 10) & 
            (df_reset['minute'] == 0)
        ]
        if not today_10am.empty:
            self.open_10am = today_10am.iloc[0]['open']
    
    def _detect_5m_fvgs(self, df: pd.DataFrame) -> None:
        """Detect 5-minute FVGs and track respect/disrespect for bias"""
        df_5m = df.resample('5min').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
        }).dropna()
        
        if len(df_5m) < 3:
            return
            
        latest_5m_ts = df_5m.index[-1]
        if self.last_5m_bar_ts == latest_5m_ts:
            return
        self.last_5m_bar_ts = latest_5m_ts
        
        highs = df_5m['high'].values
        lows = df_5m['low'].values
        closes = df_5m['close'].values
        
        i = len(df_5m) - 1
        h, l, c = highs[i], lows[i], closes[i]
        
        # Bullish FVG (gap up)
        if lows[i] > highs[i-2]:
            self.curr_bull_fvg_low = highs[i-2]
            self.curr_bull_fvg_high = lows[i]
        
        # Bearish FVG (gap down)
        if highs[i] < lows[i-2]:
            self.curr_bear_fvg_low = highs[i]
            self.curr_bear_fvg_high = lows[i-2]
        
        # Track interactions for bias
        last_interaction = None
        
        if not np.isnan(self.curr_bull_fvg_low):
            if l <= self.curr_bull_fvg_high and l >= self.curr_bull_fvg_low and c >= self.curr_bull_fvg_low:
                last_interaction = 'bull_respect'
            if c < self.curr_bull_fvg_low:
                last_interaction = 'bull_disrespect'
                self.curr_bull_fvg_low = np.nan
                self.curr_bull_fvg_high = np.nan
        
        if not np.isnan(self.curr_bear_fvg_low):
            if h >= self.curr_bear_fvg_low and h <= self.curr_bear_fvg_high and c <= self.curr_bear_fvg_high:
                last_interaction = 'bear_respect'
            if c > self.curr_bear_fvg_high:
                last_interaction = 'bear_disrespect'
                self.curr_bear_fvg_low = np.nan
                self.curr_bear_fvg_high = np.nan
        
        if last_interaction in ['bull_respect', 'bear_disrespect']:
            self.fvg_bias = 'BULL'
        elif last_interaction in ['bear_respect', 'bull_disrespect']:
            self.fvg_bias = 'BEAR'
    
    def _detect_1m_bearish_fvg(self, df: pd.DataFrame) -> None:
        """Detect 1-minute bearish FVG for inversion trigger"""
        if len(df) < 3:
            return
        
        highs = df['high'].values
        lows = df['low'].values
        
        # Bearish FVG: current high < 2-bars-ago low
        if highs[-1] < lows[-3]:
            self.bear_fvg_1m_high = lows[-3]
            self.bear_fvg_1m_active = True
    
    def _check_ifvg_long_trigger(self, curr_close: float) -> bool:
        """Check for Inversion FVG trigger (bearish FVG closed above)"""
        if self.bear_fvg_1m_active and curr_close > self.bear_fvg_1m_high:
            self.bear_fvg_1m_active = False
            return True
        return False
    
    def _lrl_ok_for_long(self, df: pd.DataFrame, stop_price: float, 
                         lookback: int = 20, tolerance: float = 0.75) -> bool:
        """LRL filter: Avoid longs if equal lows clustered near stop level"""
        if len(df) < lookback:
            return True
        lows = df['low'].values[-lookback:]
        near = np.abs(lows - stop_price) <= tolerance
        return near.sum() < 2
    
    def on_bar(self, df: pd.DataFrame) -> Optional[Dict]:
        if len(df) < 100:
            return None
            
        ts = df.index[-1]
        curr = df.iloc[-1]
        curr_date = ts.date()
        
        # Reset on new day
        if self.current_date != curr_date:
            self.reset_daily()
            self.current_date = curr_date
            self._update_pdh_pdl(df, curr_date)
        
        # Update 10AM open
        if ts.hour >= 10 and self.open_10am is None:
            self._update_10am_open(df, curr_date)
        
        # Detect 1m bearish FVG
        self._detect_1m_bearish_fvg(df)
        
        # Check for IFVG trigger
        ifvg_long = self._check_ifvg_long_trigger(curr['close'])
        
        # Update 5m FVG tracking
        self._detect_5m_fvgs(df)
        
        # Only process during NY AM session (9:30-11:00 ET)
        if not self._is_in_ny_am_session(ts):
            return None
        
        # Expire old setups (90 bars max)
        if self.pending_long:
            self.setup_bar_count += 1
            if self.setup_bar_count > 90:
                self.pending_long = False
                self.setup_bar_count = 0
        
        # Determine 4H candle bias (after 10AM)
        candle_bias = None
        if ts.hour >= 10 and self.open_10am is not None:
            if curr['close'] > self.open_10am:
                candle_bias = 'BULL'
            elif curr['close'] < self.open_10am:
                candle_bias = 'BEAR'
        
        # Combined bullish bias check
        is_bullish = False
        if candle_bias == 'BULL' and self.fvg_bias == 'BULL':
            is_bullish = True
        elif candle_bias == 'BULL' and self.fvg_bias is None:
            is_bullish = True
        elif self.fvg_bias == 'BULL' and candle_bias is None:
            is_bullish = True
        
        # MANIPULATION DETECTION (Key Level Touch)
        if is_bullish and not self.pending_long:
            key_level_touched = False
            
            # Check PDL sweep/touch
            if self.pdl is not None and curr['low'] <= self.pdl:
                key_level_touched = True
                self.long_stop = curr['low'] - 1
            
            # Check bullish 5m FVG touch
            if not np.isnan(self.curr_bull_fvg_low):
                if curr['low'] <= self.curr_bull_fvg_high and curr['low'] >= self.curr_bull_fvg_low:
                    key_level_touched = True
                    self.long_stop = self.curr_bull_fvg_low - 1
            
            if key_level_touched:
                self.pending_long = True
                self.setup_bar_count = 0
                logging.info(f"ICT: Manipulation detected. PDL={self.pdl}, Bull FVG Low={self.curr_bull_fvg_low}")
        
        # ENTRY on IFVG trigger + LRL filter
        if self.pending_long and ifvg_long and is_bullish:
            if self._lrl_ok_for_long(df, self.long_stop):
                self.pending_long = False
                self.setup_bar_count = 0
                sltp = dynamic_sltp_engine.calculate_dynamic_sltp(df)
                logging.info(f"ICT: LONG signal triggered at {curr['close']:.2f}")
                dynamic_sltp_engine.log_params(sltp)
                return {
                    "strategy": "ICT_Model",
                    "side": "LONG",
                    "tp_dist": sltp['tp_dist'],
                    "sl_dist": sltp['sl_dist']
                }
            else:
                logging.info("ICT: Signal rejected by LRL filter")
                self.pending_long = False
        
        return None


# ==========================================
# 11. ML PHYSICS STRATEGY (SESSION-BASED)
# ==========================================
"""
From juliemlsession.py - Session-specialized ML strategy:
- Uses 4 separate neural network models for different sessions
- ASIA (6PM-3AM ET): Threshold 0.65, SL 4.0, TP 6.0
- LONDON (3AM-8AM ET): Threshold 0.55, SL 4.0, TP 4.0
- NY_AM (8AM-12PM ET): Threshold 0.55, SL 10.0, TP 4.0
- NY_PM (12PM-5PM ET): Threshold 0.55, SL 10.0, TP 8.0
- Features: Velocity, Acceleration, Candle shape, Volume, Cyclic time
"""


# ==========================================
# ML FEATURE PIPELINE (Synced with ml_train_v11.py)
# ==========================================
# These constants & helpers mirror the institutional ML training script.
# They are used to build the exact same feature vector at runtime.

RSI_PERIOD = 9       # Faster for V-bottoms in high-vol regimes
ADX_PERIOD = 14      # Standard, robust
ATR_PERIOD = 14      # For dynamic volatility context
ZSCORE_WINDOW = 50   # Lookback for Z-score normalization
RVOL_LOOKBACK_DAYS = 20

ML_FEATURE_COLUMNS = [
    'Close_ZScore',
    'High_ZScore',
    'Low_ZScore',
    'ATR_ZScore',
    'Volatility_ZScore',
    'Range_ZScore',
    'Volume_ZScore',
    'Slope_ZScore',
    'RVol_ZScore',
    'RSI_Vel',
    'Adx_Vel',
    'RSI_Norm',
    'ADX_Norm',
    'Return_1',
    'Return_5',
    'Return_15',
    'Hour_Sin',
    'Hour_Cos',
    'Minute_Sin',
    'Minute_Cos',
    'DOW_Sin',
    'DOW_Cos',
    'Is_Trending',
    'Trend_Direction',
    'High_Volatility',
]

def ml_calculate_rsi(series, period=RSI_PERIOD):
    """RSI calculation (same as ml_train_v11.calculate_rsi)"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def ml_calculate_atr(high, low, close, period=ATR_PERIOD):
    """ATR for volatility-adjusted features (same as ml_train_v11.calculate_atr)."""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

def ml_calculate_adx(high, low, close, period=ADX_PERIOD):
    """ADX + DI lines for regime detection (same as ml_train_v11.calculate_adx)."""
    # True Range
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    # Smoothed values
    atr = tr.rolling(window=period).mean()
    plus_di = 100 * pd.Series(plus_dm).rolling(window=period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=period).mean() / atr

    # ADX
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx = dx.rolling(window=period).mean()

    return adx, plus_di, minus_di

def ml_calculate_slope(series, window=15):
    """Rolling linear regression slope (same as ml_train_v11.calculate_slope)."""
    def slope_calc(y):
        if len(y) < 2:
            return 0.0
        x = np.arange(len(y))
        covariance = np.cov(x, y)[0, 1]
        variance = np.var(x)
        return float(covariance / variance) if variance != 0 else 0.0
    return series.rolling(window=window).apply(slope_calc, raw=False)

def ml_zscore_normalize(series, window=ZSCORE_WINDOW):
    """Z-score normalization (same as ml_train_v11.zscore_normalize)."""
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    zscore = (series - rolling_mean) / (rolling_std + 1e-10)
    return zscore

def ml_calculate_rvol(volume, datetime_index, lookback_days=RVOL_LOOKBACK_DAYS):
    """Relative Volume: current vol / avg vol at this time of day (ml_train_v11.calculate_rvol)."""
    df_temp = pd.DataFrame({'volume': volume, 'datetime': datetime_index})
    df_temp['hour'] = df_temp['datetime'].dt.hour
    df_temp['minute'] = df_temp['datetime'].dt.minute
    df_temp['time_bucket'] = df_temp['hour'] * 60 + df_temp['minute']

    avg_vol_by_time = df_temp.groupby('time_bucket')['volume'].transform(
        lambda x: x.rolling(window=lookback_days * 5, min_periods=10).mean()
    )
    rvol = volume / (avg_vol_by_time + 1)
    return rvol

def ml_encode_cyclical_time(datetime_series):
    """Cyclical time encoding using sin/cos (same as ml_train_v11.encode_cyclical_time)."""
    hour = datetime_series.dt.hour
    minute = datetime_series.dt.minute
    day_of_week = datetime_series.dt.dayofweek

    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)

    minute_sin = np.sin(2 * np.pi * minute / 60)
    minute_cos = np.cos(2 * np.pi * minute / 60)

    dow_sin = np.sin(2 * np.pi * day_of_week / 5)
    dow_cos = np.cos(2 * np.pi * day_of_week / 5)

    return hour_sin, hour_cos, minute_sin, minute_cos, dow_sin, dow_cos

class MLPhysicsStrategy(Strategy):
    """
    Exact port of PhysicsStrategy from juliemlsession.py
    Uses SessionManager to switch between 4 session-specific models.
    """
    def __init__(self):
        self.sm = SessionManager()
        self.window_size = CONFIG.get("WINDOW_SIZE", 15)
        self.model_loaded = any(self.sm.brains.values())  # True if at least one model loaded

    def calculate_slope(self, values):
        """Exact copy from juliemlsession.py"""
        y = np.array(values)
        x = np.arange(len(y))
        if len(y) < 2: return 0
        cov = np.cov(x, y)[0, 1]
        var = np.var(x)
        return cov / var if var != 0 else 0

    
    def prepare_features(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Build ML feature vector for the latest bar using the same pipeline
        as ml_train_v11.prepare_data + get_feature_columns().

        Expects:
            df index or 'datetime' column: timezone-aware or naive timestamps
            price/volume columns: open, high, low, close, volume (any case)
        Returns:
            Single-row DataFrame with ML_FEATURE_COLUMNS, or None if not enough history.
        """
        if df is None or len(df) < 200:
            # Need sufficient history for ATR / ADX / Z-score / RVol windows
            return None

        w_df = df.copy()

        # Ensure we have a datetime column
        if 'datetime' in w_df.columns:
            w_df['datetime'] = pd.to_datetime(w_df['datetime'])
        else:
            w_df['datetime'] = pd.to_datetime(w_df.index)

        # Timezone handling – align to US/Eastern like training script
        try:
            if w_df['datetime'].dt.tz is None:
                # Assume timestamps are already ET but naive
                w_df['datetime'] = w_df['datetime'].dt.tz_localize('US/Eastern')
            else:
                w_df['datetime'] = w_df['datetime'].dt.tz_convert('US/Eastern')
        except Exception:
            # Fallback: best-effort localization
            try:
                w_df['datetime'] = w_df['datetime'].dt.tz_localize('US/Eastern')
            except Exception:
                pass

        w_df.sort_values('datetime', inplace=True)

        # Normalize column names to the format used during training
        col_map = {}
        for col in w_df.columns:
            cl = col.lower()
            if cl == 'open':
                col_map[col] = 'Open'
            elif cl == 'high':
                col_map[col] = 'High'
            elif cl == 'low':
                col_map[col] = 'Low'
            elif cl == 'close':
                col_map[col] = 'Close'
            elif cl == 'volume':
                col_map[col] = 'Volume'
        w_df.rename(columns=col_map, inplace=True)

        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for rc in required_cols:
            if rc not in w_df.columns:
                logging.warning(f"MLPhysics: missing column '{rc}' in feature builder.")
                return None

        # ==========================
        # RAW TECHNICAL INDICATORS
        # ==========================
        w_df['RSI'] = ml_calculate_rsi(w_df['Close'], period=RSI_PERIOD)
        w_df['ATR'] = ml_calculate_atr(w_df['High'], w_df['Low'], w_df['Close'], period=ATR_PERIOD)
        adx, plus_di, minus_di = ml_calculate_adx(w_df['High'], w_df['Low'], w_df['Close'], period=ADX_PERIOD)
        w_df['ADX'] = adx
        w_df['PLUS_DI'] = plus_di
        w_df['MINUS_DI'] = minus_di

        # Velocity-style features
        w_df['RSI_Vel'] = w_df['RSI'].diff(3)
        w_df['Adx_Vel'] = w_df['ADX'].diff(3)
        w_df['Slope'] = ml_calculate_slope(w_df['Close'], window=15)

        # Volatility and range
        w_df['Volatility'] = w_df['Close'].rolling(window=15).std()
        w_df['Range'] = w_df['High'] - w_df['Low']

        # Returns (already stationary)
        w_df['Return_1'] = w_df['Close'].pct_change(1)
        w_df['Return_5'] = w_df['Close'].pct_change(5)
        w_df['Return_15'] = w_df['Close'].pct_change(15)

        # ==========================
        # Z-SCORE NORMALIZATION
        # ==========================
        w_df['Close_ZScore'] = ml_zscore_normalize(w_df['Close'], window=ZSCORE_WINDOW)
        w_df['High_ZScore'] = ml_zscore_normalize(w_df['High'], window=ZSCORE_WINDOW)
        w_df['Low_ZScore'] = ml_zscore_normalize(w_df['Low'], window=ZSCORE_WINDOW)
        w_df['ATR_ZScore'] = ml_zscore_normalize(w_df['ATR'], window=ZSCORE_WINDOW)
        w_df['Volatility_ZScore'] = ml_zscore_normalize(w_df['Volatility'], window=ZSCORE_WINDOW)
        w_df['Range_ZScore'] = ml_zscore_normalize(w_df['Range'], window=ZSCORE_WINDOW)
        w_df['Volume_ZScore'] = ml_zscore_normalize(w_df['Volume'], window=ZSCORE_WINDOW)

        # RSI / ADX bounded normalization
        w_df['RSI_Norm'] = (w_df['RSI'] - 50.0) / 50.0
        w_df['ADX_Norm'] = (w_df['ADX'] / 50.0) - 1.0

        # Slope Z-score
        w_df['Slope_ZScore'] = ml_zscore_normalize(w_df['Slope'], window=ZSCORE_WINDOW)

        # ==========================
        # RELATIVE VOLUME
        # ==========================
        w_df['RVol'] = ml_calculate_rvol(w_df['Volume'], w_df['datetime'], lookback_days=RVOL_LOOKBACK_DAYS)
        w_df['RVol_ZScore'] = ml_zscore_normalize(w_df['RVol'], window=ZSCORE_WINDOW)

        # ==========================
        # CYCLICAL TIME ENCODING
        # ==========================
        (w_df['Hour_Sin'], w_df['Hour_Cos'],
         w_df['Minute_Sin'], w_df['Minute_Cos'],
         w_df['DOW_Sin'], w_df['DOW_Cos']) = ml_encode_cyclical_time(w_df['datetime'])

        # ==========================
        # REGIME FEATURES
        # ==========================
        w_df['Is_Trending'] = (w_df['ADX'] > 25).astype(int)
        w_df['Trend_Direction'] = np.sign(w_df['PLUS_DI'] - w_df['MINUS_DI'])
        w_df['ATR_MA'] = w_df['ATR'].rolling(window=100).mean()
        w_df['High_Volatility'] = (w_df['ATR'] > w_df['ATR_MA'] * 1.5).astype(int)

        # Final row must have a complete feature vector (no NaNs)
        last = w_df.iloc[-1]
        feature_vals = []
        for col in ML_FEATURE_COLUMNS:
            val = last.get(col, np.nan)
            feature_vals.append(val)

        if any(pd.isna(feature_vals)):
            # Still warming up rolling windows
            return None

        X = pd.DataFrame([feature_vals], columns=ML_FEATURE_COLUMNS)
        return X
    def on_bar(self, df: pd.DataFrame) -> Optional[Dict]:
            """
            Adapted from juliemlsession.py on_bar method.
            Uses SessionManager to get the correct model and parameters.
            Uses the full dataframe passed in (which already has 500 bars of history).
            """
            # 1. Get current session setup
            setup = self.sm.get_current_setup()
        
            if setup is None:
                logging.info("💤 Market Closed (No active session strategy)")
                return None
            
            if setup['model'] is None:
                logging.info(f"⚠️ Session {setup['name']} active, but brain file is missing!")
                return None
        
            # 2. Convert df to the format expected by prepare_features
            # Need columns: datetime, Open, High, Low, Close, Volume
            hist_df = df.copy()
            hist_df['datetime'] = hist_df.index
        
            # Rename columns to match expected format (capitalize)
            col_map = {}
            for col in hist_df.columns:
                if col.lower() == 'open':
                    col_map[col] = 'open'
                elif col.lower() == 'high':
                    col_map[col] = 'high'
                elif col.lower() == 'low':
                    col_map[col] = 'low'
                elif col.lower() == 'close':
                    col_map[col] = 'close'
                elif col.lower() == 'volume':
                    col_map[col] = 'volume'
            hist_df = hist_df.rename(columns=col_map)
        
            # Ensure we have enough data
            if len(hist_df) < 20:
                logging.info(f"📊 {setup['name']}: Building Physics Data ({len(hist_df)}/20)...")
                return None
            
            # 3. Run Prediction
            X = self.prepare_features(hist_df)
        
            if X is not None:
                # Align columns to what the specific brain expects
                if hasattr(setup['model'], "feature_names_in_"):
                    X = X.reindex(columns=setup['model'].feature_names_in_, fill_value=0)
            
                try:
                    # Ask the Specialist Brain
                    prob_up = setup['model'].predict_proba(X)[0][1]
                
                    status = "💚" if prob_up > 0.5 else "🔴"
                    req = setup['threshold']
                
                    logging.info(f"{setup['name']} Analysis {status} | Conf: {prob_up:.1%} | Req: {req:.1%}")
                
                    # LONG signal: high probability of up move
                    if prob_up >= req:
                        # Get dynamic SL/TP from engine
                        sltp = dynamic_sltp_engine.calculate_dynamic_sltp(hist_df)
                        logging.info(f"🎯 {setup['name']} LONG SIGNAL CONFIRMED (prob={prob_up:.1%})")
                        dynamic_sltp_engine.log_params(sltp)
                        return {
                            "strategy": f"MLPhysics_{setup['name']}",
                            "side": "LONG",
                            "tp_dist": sltp['tp_dist'],
                            "sl_dist": sltp['sl_dist']
                        }
                    
                    # SHORT signal: high probability of down move (low prob_up)
                    elif prob_up <= (1.0 - req):
                        sltp = dynamic_sltp_engine.calculate_dynamic_sltp(hist_df)
                        logging.info(f"🎯 {setup['name']} SHORT SIGNAL CONFIRMED (prob={prob_up:.1%})")
                        dynamic_sltp_engine.log_params(sltp)
                        return {
                            "strategy": f"MLPhysics_{setup['name']}",
                            "side": "SHORT",
                            "tp_dist": sltp['tp_dist'],
                            "sl_dist": sltp['sl_dist']
                        }
                    
                except Exception as e:
                    logging.error(f"Prediction Error: {e}")
                
            return None

# ==========================================
# 13. DYNAMIC ENGINE STRATEGY WRAPPER
# ==========================================
class DynamicEngineStrategy(Strategy):
    """
    Wrapper for DynamicSignalEngine to run 235 hardcoded strategies
    alongside Julie's existing filters.
    """
    def __init__(self):
        self.engine = get_signal_engine()
        self.last_processed_time = None

    def on_bar(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Resamples 1m data to 5m/15m and queries the engine.
        """
        # We need enough data to resample accurately
        if df is None or len(df) < 60:
            return None

        # Ensure we only check once per candle close (handled by is_new_bar in main loop,
        # but good to have a safety check here)
        current_time = df.index[-1]
        
        # 1. Resample 1m data to 5m and 15m
        # We use standard pandas resampling. 
        # 'closed="left", label="left"' ensures 09:00-09:05 is labeled 09:00
        
        # Resample to 5 Minute
        df_5m = df.resample('5min', closed='left', label='left').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
        }).dropna()

        # Resample to 15 Minute
        df_15m = df.resample('15min', closed='left', label='left').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
        }).dropna()

        # 2. Query the Engine
        # The engine checks df.iloc[-2] (the previously closed candle), 
        # so we pass the current state and it handles the lookback.
        signal_data = self.engine.check_signal(current_time, df_5m, df_15m)

        if signal_data:
            # 3. Map Engine Output to Julie's Signal Format
            # Note: We use the engine's 'Best_SL' and 'Best_TP' because these 
            # strategies were optimized specifically for those values.
            
            logging.info(f"🚀 DYNAMIC ENGINE TRIGGER: {signal_data['strategy_id']}")
            
            return {
                "strategy": "DynamicEngine",  # General name for logging
                "sub_strategy": signal_data['strategy_id'], # Specific ID for debugging
                "side": signal_data['signal'], # 'LONG' or 'SHORT'
                "tp_dist": signal_data['tp'],
                "sl_dist": signal_data['sl']
            }

        return None

class DynamicEngine2Strategy(Strategy):
    """
    Wrapper for DynamicSignalEngine2 to run 167 hardcoded Price Action strategies
    alongside Julie's existing filters.
    Contains Quarter/Day/Session specific edges with positive Risk:Reward.
    """
    def __init__(self):
        self.engine = get_signal_engine2()
        self.last_processed_time = None

    def on_bar(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Resamples 1m data to 5m/15m and queries the engine.
        """
        # We need enough data to resample accurately
        if df is None or len(df) < 60:
            return None

        current_time = df.index[-1]
        
        # 1. Resample 1m data to 5m and 15m
        df_5m = df.resample('5min', closed='left', label='left').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
        }).dropna()

        df_15m = df.resample('15min', closed='left', label='left').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
        }).dropna()

        # 2. Query the Engine
        signal_data = self.engine.check_signal(current_time, df_5m, df_15m)

        if signal_data:
            logging.info(f"🚀 DYNAMIC ENGINE 2 TRIGGER: {signal_data['strategy_id']}")
            
            return {
                "strategy": "DynamicEngine2",  # General name for logging
                "sub_strategy": signal_data['strategy_id'], # Specific ID for debugging
                "side": signal_data['signal'], # 'LONG' or 'SHORT'
                "tp_dist": signal_data['tp'],
                "sl_dist": signal_data['sl']
            }

        return None

# ==========================================
# 12. MAIN EXECUTION LOOP
# ==========================================
def run_bot():
    refresh_target_symbol()
    print("=" * 60)
    print("PROJECTX GATEWAY - ES FUTURES BOT (LIVE)")
    print("--- Julie Pro (Session Specialized) ---")
    print("--- DYNAMIC SL/TP ENGINE ENABLED ---")
    print(f"REST API: {CONFIG['REST_BASE_URL']}")
    print(f"Target Symbol: {CONFIG['TARGET_SYMBOL']}")
    print("=" * 60)
    
    client = ProjectXClient()
    
    # Step 1: Authenticate
    try:
        client.login()
    except Exception as e:
        print(f"CRITICAL: Failed to login. Check credentials. Error: {e}")
        return

    # Step 2: Fetch Account ID
    print("\n📋 Fetching account information...")
    account_id = client.fetch_accounts()
    if account_id is None:
        print("CRITICAL: Could not retrieve account ID")
        return
    
    # Step 3: Fetch Contract ID
    print("\n📋 Fetching available contracts...")
    contract_id = client.fetch_contracts()
    if contract_id is None:
        print("CRITICAL: Could not retrieve contract ID")
        return
    
    print(f"\n✅ Setup complete:")
    print(f"   Account ID: {client.account_id}")
    print(f"   Contract ID: {client.contract_id}")

    # Initialize all strategies
    
    # HIGH PRIORITY - Execute immediately on signal
    fast_strategies = [
        RegimeAdaptiveStrategy(),
        IntradayDipStrategy(),
    ]
    
    # STANDARD PRIORITY - Normal execution
    ml_strategy = MLPhysicsStrategy()
    dynamic_engine_strat = DynamicEngineStrategy()
    dynamic_engine2_strat = DynamicEngine2Strategy()
    
    standard_strategies = [
        ConfluenceStrategy(),
        dynamic_engine_strat,
        dynamic_engine2_strat,
    ]
    
    # Only add ML strategy if at least one model loaded successfully
    if ml_strategy.model_loaded:
        standard_strategies.append(ml_strategy)
    else:
        print(f"⚠️ MLPhysicsStrategy disabled - no session model files found")
    
    # LOW PRIORITY / LOOSE EXECUTION - Wait for next bar
    loose_strategies = [
        OrbStrategy(),
        ICTModelStrategy(),
    ]
    
    # Initialize filters
    rejection_filter = RejectionFilter()
    bank_filter = BankLevelQuarterFilter()
    chop_filter = ChopFilter(lookback=20)
    extension_filter = ExtensionFilter()
    htf_fvg_filter = HTFFVGFilter() # Now uses Memory-Based Class
    structure_blocker = DynamicStructureBlocker(lookback=20)
    
    print("\nActive Strategies:")
    print("  [FAST EXECUTION]")
    for strat in fast_strategies: print(f"    • {strat.__class__.__name__}")
    print("  [STANDARD EXECUTION]")
    for strat in standard_strategies: print(f"    • {strat.__class__.__name__}")
    print("  [LOOSE EXECUTION]")
    for strat in loose_strategies: print(f"    • {strat.__class__.__name__}")
    
    print("\nListening for market data (polling every 1 second)...")
    
    # === TRACKING VARIABLES ===
    last_htf_fetch_time = 0
    
    # Track pending signals for delayed execution
    pending_loose_signals = {}
    last_processed_bar = None
    opposite_signal_count = 0
    
    # Early Exit Tracking
    active_trade = None
    bar_count = 0
    
    # Token refresh
    last_token_check = time.time()
    TOKEN_CHECK_INTERVAL = 3600
    
    # One-time backfill flag
    data_backfilled = False

    while True:
        try:
            # Periodic token validation
            if time.time() - last_token_check > TOKEN_CHECK_INTERVAL:
                client.validate_session()
                last_token_check = time.time()
            
            # 1. Fetch Latest Data (Fast loop)
            new_df = client.get_market_data(lookback_minutes=500, force_fetch=True) 
            
            if new_df.empty:
                time.sleep(1)
                continue
            
            # === ONE-TIME BACKFILL ===
            if not data_backfilled:
                logging.info("🔄 Performing one-time backfill of filter state from history...")
                # Replay the 500 minutes of history we just fetched
                # This restores Midnight ORB, Prev Session, etc. instantly
                rejection_filter.backfill(new_df)
                
                # Also backfill bank_filter (has same update() signature)
                for ts, row in new_df.sort_index().iterrows():
                    bank_filter.update(ts, row['high'], row['low'], row['close'])
                
                data_backfilled = True
                logging.info("✅ State restored from history.")
            
            # Heartbeat
            current_price = new_df.iloc[-1]['close']
            current_time = new_df.index[-1]
            now_ts = time.time()
            
            if not hasattr(client, '_heartbeat_counter'):
                client._heartbeat_counter = 0
            client._heartbeat_counter += 1
            if client._heartbeat_counter % 30 == 0:
                print(f"💓 Heartbeat: {datetime.datetime.now().strftime('%H:%M:%S')} | Price: {current_price:.2f}")

            # === UPDATE HTF FVG MEMORY (THROTTLED) ===
            # Only update memory once every 60 seconds to save API calls.
            # The filter will use this memory for checks every second.
            # === UPDATE HTF FVG MEMORY (THROTTLED) ===
            # Only update memory once every 60 seconds to save API calls.
            if now_ts - last_htf_fetch_time > 60:
                try:
                    # Use PRINT here so you definitely see it
                    print(f"\n⏳ [{datetime.datetime.now().strftime('%H:%M:%S')}] Updating HTF FVG Memory (1H/4H)...")
                    
                    # Fetch new data
                    df_1h_new = client.fetch_custom_bars(lookback_bars=240, minutes_per_bar=60)
                    df_4h_new = client.fetch_custom_bars(lookback_bars=200, minutes_per_bar=240)
                    
                    # Update Memory
                    if not df_1h_new.empty and not df_4h_new.empty:
                        htf_fvg_filter.check_signal_blocked('CHECK', current_price, df_1h_new, df_4h_new)
                        print(f"   ✅ Memory Updated. Active Structures: {len(htf_fvg_filter.memory)}")
                        last_htf_fetch_time = now_ts
                    else:
                        print("   ⚠️ HTF Data Fetch returned empty.")
                except Exception as e:
                    logging.error(f"Error updating HTF memory: {e}")
            
            # Initial Startup Fetch (if empty)
            if (not htf_fvg_filter.memory) and client._check_general_rate_limit() and (now_ts - last_htf_fetch_time > 60):
                 logging.info("🚀 Startup: Fetching initial HTF data...")
                 try:
                     df_1h_new = client.fetch_custom_bars(lookback_bars=240, minutes_per_bar=60)
                     df_4h_new = client.fetch_custom_bars(lookback_bars=200, minutes_per_bar=240)
                     htf_fvg_filter.check_signal_blocked('CHECK', current_price, df_1h_new, df_4h_new)
                     last_htf_fetch_time = now_ts
                 except Exception as e:
                     logging.error(f"Startup HTF fetch failed: {e}")
            
            # === BREAK-EVEN CHECK (EVERY TICK) ===
            if active_trade is not None and not active_trade.get('break_even_triggered', False):
                be_config = CONFIG.get('BREAK_EVEN', {})
                if be_config.get('enabled', False):
                    tp_dist = active_trade.get('tp_dist', 6.0)
                    entry_price = active_trade['entry_price']
                    trigger_pct = be_config.get('trigger_pct', 0.40)
                    
                    if active_trade['side'] == 'LONG':
                        current_profit = current_price - entry_price
                    else:
                        current_profit = entry_price - current_price
                    
                    profit_threshold = tp_dist * trigger_pct
                    
                    if current_profit >= profit_threshold:
                        logging.info(f"🔒 BREAK-EVEN TRIGGER: Profit {current_profit:.2f} >= {profit_threshold:.2f}")
                        buffer = be_config.get('buffer_ticks', 1) * 0.25
                        be_price = entry_price + buffer if active_trade['side'] == 'LONG' else entry_price - buffer
                        
                        # Use known size from active_trade (default to 1) and cached stop order ID
                        known_size = active_trade.get('size', 1)
                        stop_order_id = active_trade.get('stop_order_id')
                        
                        if client.modify_stop_to_breakeven(be_price, active_trade['side'], known_size, stop_order_id):
                            active_trade['break_even_triggered'] = True
                            logging.info(f"✅ BREAK-EVEN SET: Stop moved to {be_price:.2f}")

            currbar = new_df.iloc[-1]
            rejection_filter.update(current_time, currbar['high'], currbar['low'], currbar['close'])
            bank_filter.update(current_time, currbar['high'], currbar['low'], currbar['close'])
            chop_filter.update(currbar['high'], currbar['low'], currbar['close'], current_time)
            extension_filter.update(currbar['high'], currbar['low'], currbar['close'], current_time)
            structure_blocker.update(new_df)

            
            # Only process signals on NEW bars
            is_new_bar = (last_processed_bar is None or current_time > last_processed_bar)
            
            if is_new_bar:
                bar_count += 1
                logging.info(f"Bar: {current_time.strftime('%Y-%m-%d %H:%M:%S')} ET | Price: {current_price:.2f}")
                last_processed_bar = current_time
                
                # === UPDATE FILTERS ===
                curr_bar = new_df.iloc[-1]
                rejection_filter.update(current_time, curr_bar['high'], curr_bar['low'], curr_bar['close'])
                bank_filter.update(current_time, curr_bar['high'], curr_bar['low'], curr_bar['close'])
                chop_filter.update(curr_bar['high'], curr_bar['low'], curr_bar['close'], current_time)
                extension_filter.update(curr_bar['high'], curr_bar['low'], curr_bar['close'], current_time)
                structure_blocker.update(new_df)
                
                # === EARLY EXIT CHECK ===
                if active_trade is not None:
                    active_trade['bars_held'] += 1
                    strategy_name = active_trade['strategy']
                    early_exit_config = CONFIG.get('EARLY_EXIT', {}).get(strategy_name, {})
                        
                    if active_trade['side'] == 'LONG':
                        is_green = current_price > active_trade['entry_price']
                    else:
                        is_green = current_price < active_trade['entry_price']
                        
                    was_green = active_trade.get('was_green')
                    if was_green is not None and is_green != was_green:
                        active_trade['profit_crosses'] = active_trade.get('profit_crosses', 0) + 1
                    active_trade['was_green'] = is_green
                    
                    if early_exit_config.get('enabled', False):
                        exit_time = early_exit_config.get('exit_if_not_green_by', 50)
                        exit_cross = early_exit_config.get('max_profit_crosses', 100)
                        should_exit = False
                        
                        if active_trade['bars_held'] >= exit_time and not is_green:
                            should_exit = True
                            exit_reason = f"not green after {active_trade['bars_held']} bars"
                        if active_trade.get('profit_crosses', 0) > exit_cross:
                            should_exit = True
                            exit_reason = f"choppy ({active_trade['profit_crosses']} crosses)"
                            
                        if should_exit:
                            logging.info(f"⏰ EARLY EXIT: {strategy_name} - {exit_reason}")
                            position = client.get_position()
                            if position['side'] is not None:
                                client.close_position(position)
                            active_trade = None

                # === STRATEGY EXECUTION ===
                strategy_results = {'checked': [], 'rejected': [], 'executed': None}
                
                # Run ML Analysis
                ml_signal = None
                if ml_strategy.model_loaded:
                    try:
                        ml_signal = ml_strategy.on_bar(new_df)
                        if ml_signal: strategy_results['checked'].append('MLPhysics')
                    except Exception as e:
                        logging.error(f"ML Strategy Error: {e}")
                
                # 2a. FAST STRATEGIES
                signal_executed = False
                for strat in fast_strategies:
                    strat_name = strat.__class__.__name__
                    try:
                        signal = strat.on_bar(new_df)
                        if signal:
                            strategy_results['checked'].append(strat_name)
                            
                            # Filters
                            if rejection_filter.should_block_trade(signal['side'])[0]: continue
                            
                            # HTF FVG (Memory Based)
                            fvg_blocked, fvg_reason = htf_fvg_filter.check_signal_blocked(signal['side'], current_price, None, None)
                            if fvg_blocked:
                                logging.info(f"🚫 BLOCKED (HTF FVG): {fvg_reason}")
                                continue
                            
                            # Weak Level Blocker (EQH/EQL)
                            struct_blocked, struct_reason = structure_blocker.should_block_trade(signal['side'], current_price)
                            if struct_blocked:
                                logging.info(f"🚫 {struct_reason}")
                                continue

                            # Chop
                            daily_bias = rejection_filter.prev_day_pm_bias
                            if chop_filter.should_block_trade(signal['side'], daily_bias)[0]: continue
                            
                            # Extension
                            if extension_filter.should_block_trade(signal['side'])[0]: continue
                            
                            # Volatility
                            should_trade, vol_adj = check_volatility(new_df, signal.get('sl_dist', 4.0), signal.get('tp_dist', 6.0))
                            if not should_trade: continue
                            
                            if vol_adj.get('adjustment_applied', False):
                                signal['sl_dist'] = vol_adj['sl_dist']
                                signal['tp_dist'] = vol_adj['tp_dist']
                            
                            # Bank Filter (RegimeAdaptive Only)
                            if bank_filter.should_block_trade(signal['side'])[0]: continue
                            
                            # Execute
                            strategy_results['executed'] = strat_name
                            logging.info(f"✅ FAST EXEC: {signal['strategy']} signal")
                            result = client.close_and_reverse(signal, current_price, opposite_signal_count)
                            if result[0]:
                                active_trade = {
                                    'strategy': signal['strategy'], 
                                    'side': signal['side'], 
                                    'entry_price': current_price, 
                                    'entry_bar': bar_count, 
                                    'bars_held': 0, 
                                    'tp_dist': signal['tp_dist'],
                                    'size': 1,  # Fixed contract size
                                    'stop_order_id': client._active_stop_order_id  # Cached stop ID
                                }
                            
                            signal_executed = True
                            break
                    except Exception as e:
                        logging.error(f"Error in {strat_name}: {e}")
                
                # 2b. STANDARD STRATEGIES
                if not signal_executed:
                    for strat in standard_strategies:
                        strat_name = strat.__class__.__name__
                        signal = None
                        
                        if strat_name == "MLPhysicsStrategy":
                            signal = ml_signal
                        else:
                            try:
                                signal = strat.on_bar(new_df)
                            except Exception as e:
                                logging.error(f"Error in {strat_name}: {e}")
                        
                        if signal:
                            strategy_results['checked'].append(strat_name)
                            
                            if rejection_filter.should_block_trade(signal['side'])[0]: continue
                            
                            # HTF FVG (Memory Based)
                            fvg_blocked, fvg_reason = htf_fvg_filter.check_signal_blocked(signal['side'], current_price, None, None)
                            if fvg_blocked:
                                logging.info(f"🚫 BLOCKED (HTF FVG): {fvg_reason}")
                                continue
                            
                            # Weak Level Blocker (EQH/EQL)
                            struct_blocked, struct_reason = structure_blocker.should_block_trade(signal['side'], current_price)
                            if struct_blocked:
                                logging.info(f"🚫 {struct_reason}")
                                continue

                            # Chop (Except DynamicEngine and DynamicEngine2)
                            if signal['strategy'] not in ["DynamicEngine", "DynamicEngine2"]:
                                if chop_filter.should_block_trade(signal['side'], rejection_filter.prev_day_pm_bias)[0]: continue
                            
                            # Extension (Except DynamicEngine and DynamicEngine2)
                            if signal['strategy'] not in ["DynamicEngine", "DynamicEngine2"]:
                                if extension_filter.should_block_trade(signal['side'])[0]: continue
                            
                            # Volatility
                            should_trade, vol_adj = check_volatility(new_df, signal.get('sl_dist', 4.0), signal.get('tp_dist', 6.0))
                            if not should_trade: continue
                            
                            if vol_adj.get('adjustment_applied', False):
                                signal['sl_dist'] = vol_adj['sl_dist']
                                signal['tp_dist'] = vol_adj['tp_dist']
                                
                            # Bank Filter (ML Only)
                            if strat_name == "MLPhysicsStrategy":
                                if bank_filter.should_block_trade(signal['side'])[0]: continue

                            strategy_results['executed'] = strat_name
                            logging.info(f"✅ STANDARD EXEC: {signal['strategy']} signal")
                            result = client.close_and_reverse(signal, current_price, opposite_signal_count)
                            if result[0]:
                                active_trade = {
                                    'strategy': signal['strategy'], 
                                    'side': signal['side'], 
                                    'entry_price': current_price, 
                                    'entry_bar': bar_count, 
                                    'bars_held': 0, 
                                    'tp_dist': signal['tp_dist'],
                                    'size': 1,  # Fixed contract size
                                    'stop_order_id': client._active_stop_order_id  # Cached stop ID
                                }
                            
                            signal_executed = True
                            break

# 2c. LOOSE STRATEGIES (Queued)
                if not signal_executed:
                    if is_new_bar:
                        # Process Pending
                        for s_name in list(pending_loose_signals.keys()):
                            pending = pending_loose_signals[s_name]
                            pending['bar_count'] += 1
                            if pending['bar_count'] >= 1:
                                sig = pending['signal']
                                # Re-check filters
                                if rejection_filter.should_block_trade(sig['side'])[0]: 
                                    del pending_loose_signals[s_name]; continue
                                
                                # HTF FVG
                                fvg_blocked, fvg_reason = htf_fvg_filter.check_signal_blocked(sig['side'], current_price, None, None)
                                if fvg_blocked:
                                    logging.info(f"🚫 BLOCKED (HTF FVG): {fvg_reason}")
                                    del pending_loose_signals[s_name]; continue
                                
                                # === [FIX 1] UPDATED BLOCKER CHECK ===
                                struct_blocked, struct_reason = structure_blocker.should_block_trade(sig['side'], current_price)
                                if struct_blocked:
                                    logging.info(f"🚫 {struct_reason}")
                                    del pending_loose_signals[s_name]; continue
                                # =====================================

                                if chop_filter.should_block_trade(sig['side'], rejection_filter.prev_day_pm_bias)[0]: 
                                    del pending_loose_signals[s_name]; continue
                                if extension_filter.should_block_trade(sig['side'])[0]: 
                                    del pending_loose_signals[s_name]; continue
                                
                                logging.info(f"✅ LOOSE EXEC: {s_name}")
                                result = client.close_and_reverse(sig, current_price, opposite_signal_count)
                                if result[0]:
                                    active_trade = {
                                        'strategy': s_name, 
                                        'side': sig['side'], 
                                        'entry_price': current_price, 
                                        'entry_bar': bar_count, 
                                        'bars_held': 0, 
                                        'tp_dist': sig['tp_dist'],
                                        'size': 1, 
                                        'stop_order_id': client._active_stop_order_id 
                                    }
                                
                                del pending_loose_signals[s_name]
                                signal_executed = True
                                break
                        
                        # Check New Loose Signals
                        if not signal_executed:
                            for strat in loose_strategies:
                                try:
                                    signal = strat.on_bar(new_df)
                                    s_name = strat.__class__.__name__
                                    if signal and s_name not in pending_loose_signals:
                                        if rejection_filter.should_block_trade(signal['side'])[0]: continue
                                        
                                        fvg_blocked, fvg_reason = htf_fvg_filter.check_signal_blocked(signal['side'], current_price, None, None)
                                        if fvg_blocked: continue
                                        
                                        # === [FIX 2] UPDATED BLOCKER CHECK ===
                                        struct_blocked, struct_reason = structure_blocker.should_block_trade(signal['side'], current_price)
                                        if struct_blocked: continue
                                        # =====================================

                                        if chop_filter.should_block_trade(signal['side'], rejection_filter.prev_day_pm_bias)[0]: continue
                                        if extension_filter.should_block_trade(signal['side'])[0]: continue
                                        
                                        logging.info(f"🕐 Queuing {s_name} signal")
                                        pending_loose_signals[s_name] = {'signal': signal, 'bar_count': 0}
                                except Exception as e:
                                    logging.error(f"Error in {s_name}: {e}")

            time.sleep(1)
            
        except KeyboardInterrupt:
            print("\nBot Stopped by User.")
            break
        except Exception as e:
            logging.error(f"Main Loop Error: {e}")
            time.sleep(10)


if __name__ == "__main__":
    run_bot()