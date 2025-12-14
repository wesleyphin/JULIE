import requests
import pandas as pd
import numpy as np
import datetime
import time
import logging
import pytz
import uuid
from typing import Dict, Optional, List, Tuple

from config import CONFIG, refresh_target_symbol
from dynamic_sltp_params import dynamic_sltp_engine, get_sltp
from volatility_filter import volatility_filter, check_volatility, VolRegime
from regime_strategy import RegimeAdaptiveStrategy
from htf_fvg_filter import HTFFVGFilter
from rejection_filter import RejectionFilter
from chop_filter import ChopFilter
from extension_filter import ExtensionFilter
from dynamic_structure_blocker import DynamicStructureBlocker
from bank_level_quarter_filter import BankLevelQuarterFilter
from orb_strategy import OrbStrategy
from intraday_dip_strategy import IntradayDipStrategy
from confluence_strategy import ConfluenceStrategy
from ict_model_strategy import ICTModelStrategy
from ml_physics_strategy import MLPhysicsStrategy
from dynamic_engine_strategy import DynamicEngineStrategy
from dynamic_engine2_strategy import DynamicEngine2Strategy
from event_logger import event_logger

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
        Get available contracts using Search to find MES futures specifically.
        Endpoint: POST /api/Contract/search
        """
        refresh_target_symbol()

        if not self._check_general_rate_limit():
            return self.contract_id

        url = f"{self.base_url}/api/Contract/search"
        # We explicitly search for "MES" to ensure we get the right list
        payload = {
            "live": False,  # Set to False to find Topstep tradable contracts
            "searchText": CONFIG.get('TARGET_SYMBOL', 'MESZ25')
        }

        try:
            logging.info(f"🔎 Searching for contracts with symbol: {payload['searchText']}...")
            resp = self.session.post(url, json=payload)
            self._track_general_request()
            resp.raise_for_status()
            data = resp.json()

            if 'contracts' in data and len(data['contracts']) > 0:
                target = CONFIG.get('TARGET_SYMBOL', 'MESZ25')
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
        # MES tick size is 0.25
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
            tp_price = current_price + tp_points if is_long else current_price - tp_points
            sl_price = current_price - sl_points if is_long else current_price + sl_points

            # Enhanced event logging: Order about to be placed
            event_logger.log_trade_signal_generated(
                strategy=signal.get('strategy', 'Unknown'),
                side=signal['side'],
                price=current_price,
                tp_dist=tp_points,
                sl_dist=sl_points
            )

            logging.info(f"🚀 SENDING ORDER: {signal['side']} @ ~{current_price:.2f}")
            logging.info(f"   TP: {tp_points}pts ({final_tp_ticks} ticks)")
            logging.info(f"   SL: {sl_points}pts ({final_sl_ticks} ticks)")

            resp = self.session.post(url, json=payload)
            self._track_general_request()

            if resp.status_code == 429:
                logging.error("🚫 Rate limited on order placement!")
                event_logger.log_error("RATE_LIMIT", "Order placement rate limited")
                return None

            resp_data = resp.json()

            if resp.status_code == 200:
                # Check for business logic success
                if resp_data.get('success') is False:
                     err_msg = resp_data.get('errorMessage', 'Unknown Rejection')
                     logging.error(f"❌ Order Rejected by Engine: {err_msg}")

                     # Enhanced event logging: Order rejected
                     event_logger.log_trade_order_rejected(
                         side=signal['side'],
                         price=current_price,
                         error_msg=err_msg,
                         strategy=signal.get('strategy', 'Unknown')
                     )
                     return None
                else:
                     logging.info(f"✅ Order Placed Successfully [{unique_order_id[:8]}]")

                     # Enhanced event logging: Order placed successfully
                     event_logger.log_trade_order_placed(
                         order_id=unique_order_id,
                         side=signal['side'],
                         price=current_price,
                         tp_price=tp_price,
                         sl_price=sl_price,
                         strategy=signal.get('strategy', 'Unknown')
                     )
                     
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
                event_logger.log_error("RATE_LIMIT", "Position close rate limited")
                return False

            resp_data = resp.json()
            if resp.status_code == 200 and resp_data.get('success', False):
                logging.info(f"✅ Position close order submitted: {resp_data}")

                # Enhanced event logging: Position closed
                # Note: We don't have the exact exit price yet, but we can estimate
                event_logger.log_trade_closed(
                    side=position['side'],
                    entry_price=position['avg_price'],
                    exit_price=position['avg_price'],  # Actual exit price not available yet
                    pnl=0.0,  # PnL will be calculated later
                    reason="Manual Close"
                )

                # Reset shadow position state
                self._local_position = {'side': None, 'size': 0, 'avg_price': 0.0}
                self._active_stop_order_id = None

                return True
            else:
                logging.error(f"❌ Position close failed: {resp_data}")
                event_logger.log_error("POSITION_CLOSE_FAILED", f"Failed to close position: {resp_data}")
                return False
        except Exception as e:
            logging.error(f"❌ Position close exception: {e}")
            event_logger.log_error("POSITION_CLOSE_EXCEPTION", f"Exception closing position: {e}", exception=e)
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

            # Enhanced event logging: Close and reverse
            event_logger.log_close_and_reverse(
                old_side=position['side'],
                new_side=new_signal['side'],
                price=current_price,
                strategy=new_signal.get('strategy', 'Unknown')
            )

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
# 12. MAIN EXECUTION LOOP
# ==========================================
def run_bot():
    refresh_target_symbol()
    print("=" * 60)
    print("PROJECTX GATEWAY - MES FUTURES BOT (LIVE)")
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

                            # Enhanced event logging: Breakeven adjustment
                            old_sl = entry_price - tp_dist if active_trade['side'] == 'LONG' else entry_price + tp_dist
                            event_logger.log_breakeven_adjustment(
                                old_sl=old_sl,
                                new_sl=be_price,
                                current_price=current_price,
                                profit_points=current_profit
                            )

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

                            # Enhanced event logging: Early exit
                            event_logger.log_early_exit(
                                reason=exit_reason,
                                bars_held=active_trade['bars_held'],
                                current_price=current_price,
                                entry_price=active_trade['entry_price']
                            )

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

                            # Enhanced event logging: Strategy signal generated
                            event_logger.log_strategy_signal(
                                strategy_name=signal.get('strategy', strat_name),
                                side=signal['side'],
                                tp_dist=signal.get('tp_dist', 6.0),
                                sl_dist=signal.get('sl_dist', 4.0),
                                price=current_price,
                                additional_info={"execution_type": "FAST"}
                            )

                            # Filters - Rejection Filter
                            rej_blocked, rej_reason = rejection_filter.should_block_trade(signal['side'])
                            if rej_blocked:
                                event_logger.log_rejection_block("RejectionFilter", signal['side'], rej_reason or "Rejection bias")
                                continue

                            # HTF FVG (Memory Based)
                            fvg_blocked, fvg_reason = htf_fvg_filter.check_signal_blocked(signal['side'], current_price, None, None)
                            if fvg_blocked:
                                logging.info(f"🚫 BLOCKED (HTF FVG): {fvg_reason}")
                                event_logger.log_filter_check("HTF_FVG", signal['side'], False, fvg_reason)
                                continue
                            else:
                                event_logger.log_filter_check("HTF_FVG", signal['side'], True)

                            # Weak Level Blocker (EQH/EQL)
                            struct_blocked, struct_reason = structure_blocker.should_block_trade(signal['side'], current_price)
                            if struct_blocked:
                                logging.info(f"🚫 {struct_reason}")
                                event_logger.log_filter_check("StructureBlocker", signal['side'], False, struct_reason)
                                continue
                            else:
                                event_logger.log_filter_check("StructureBlocker", signal['side'], True)

                            # Chop
                            daily_bias = rejection_filter.prev_day_pm_bias
                            chop_blocked, chop_reason = chop_filter.should_block_trade(signal['side'], daily_bias)
                            if chop_blocked:
                                event_logger.log_filter_check("ChopFilter", signal['side'], False, chop_reason)
                                continue
                            else:
                                event_logger.log_filter_check("ChopFilter", signal['side'], True)

                            # Extension
                            ext_blocked, ext_reason = extension_filter.should_block_trade(signal['side'])
                            if ext_blocked:
                                event_logger.log_filter_check("ExtensionFilter", signal['side'], False, ext_reason)
                                continue
                            else:
                                event_logger.log_filter_check("ExtensionFilter", signal['side'], True)

                            # Volatility
                            should_trade, vol_adj = check_volatility(new_df, signal.get('sl_dist', 4.0), signal.get('tp_dist', 6.0))
                            if not should_trade:
                                event_logger.log_filter_check("VolatilityFilter", signal['side'], False, "Volatility check failed")
                                continue
                            else:
                                event_logger.log_filter_check("VolatilityFilter", signal['side'], True)

                            if vol_adj.get('adjustment_applied', False):
                                signal['sl_dist'] = vol_adj['sl_dist']
                                signal['tp_dist'] = vol_adj['tp_dist']
                                event_logger.log_trade_modified(
                                    "VolatilityAdjustment",
                                    signal.get('tp_dist', 6.0),
                                    vol_adj['tp_dist'],
                                    "Volatility regime adjustment"
                                )

                            # Bank Filter (RegimeAdaptive Only)
                            bank_blocked, bank_reason = bank_filter.should_block_trade(signal['side'])
                            if bank_blocked:
                                event_logger.log_filter_check("BankFilter", signal['side'], False, bank_reason)
                                continue
                            else:
                                event_logger.log_filter_check("BankFilter", signal['side'], True)

                            # Execute
                            strategy_results['executed'] = strat_name
                            logging.info(f"✅ FAST EXEC: {signal['strategy']} signal")
                            event_logger.log_strategy_execution(signal.get('strategy', strat_name), "FAST")

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

                            # Enhanced event logging: Strategy signal generated
                            event_logger.log_strategy_signal(
                                strategy_name=signal.get('strategy', strat_name),
                                side=signal['side'],
                                tp_dist=signal.get('tp_dist', 6.0),
                                sl_dist=signal.get('sl_dist', 4.0),
                                price=current_price,
                                additional_info={"execution_type": "STANDARD"}
                            )

                            # Rejection Filter
                            rej_blocked, rej_reason = rejection_filter.should_block_trade(signal['side'])
                            if rej_blocked:
                                event_logger.log_rejection_block("RejectionFilter", signal['side'], rej_reason or "Rejection bias")
                                continue
                            
                            # HTF FVG (Memory Based)
                            fvg_blocked, fvg_reason = htf_fvg_filter.check_signal_blocked(signal['side'], current_price, None, None)
                            if fvg_blocked:
                                logging.info(f"🚫 BLOCKED (HTF FVG): {fvg_reason}")
                                event_logger.log_filter_check("HTF_FVG", signal['side'], False, fvg_reason)
                                continue
                            else:
                                event_logger.log_filter_check("HTF_FVG", signal['side'], True)

                            # Weak Level Blocker (EQH/EQL)
                            struct_blocked, struct_reason = structure_blocker.should_block_trade(signal['side'], current_price)
                            if struct_blocked:
                                logging.info(f"🚫 {struct_reason}")
                                event_logger.log_filter_check("StructureBlocker", signal['side'], False, struct_reason)
                                continue
                            else:
                                event_logger.log_filter_check("StructureBlocker", signal['side'], True)

                            # Chop (Except DynamicEngine and DynamicEngine2)
                            if signal['strategy'] not in ["DynamicEngine", "DynamicEngine2"]:
                                chop_blocked, chop_reason = chop_filter.should_block_trade(signal['side'], rejection_filter.prev_day_pm_bias)
                                if chop_blocked:
                                    event_logger.log_filter_check("ChopFilter", signal['side'], False, chop_reason)
                                    continue
                                else:
                                    event_logger.log_filter_check("ChopFilter", signal['side'], True)

                            # Extension (Except DynamicEngine and DynamicEngine2)
                            if signal['strategy'] not in ["DynamicEngine", "DynamicEngine2"]:
                                ext_blocked, ext_reason = extension_filter.should_block_trade(signal['side'])
                                if ext_blocked:
                                    event_logger.log_filter_check("ExtensionFilter", signal['side'], False, ext_reason)
                                    continue
                                else:
                                    event_logger.log_filter_check("ExtensionFilter", signal['side'], True)
                            
                            # Volatility
                            should_trade, vol_adj = check_volatility(new_df, signal.get('sl_dist', 4.0), signal.get('tp_dist', 6.0))
                            if not should_trade:
                                event_logger.log_filter_check("VolatilityFilter", signal['side'], False, "Volatility check failed")
                                continue
                            else:
                                event_logger.log_filter_check("VolatilityFilter", signal['side'], True)

                            if vol_adj.get('adjustment_applied', False):
                                signal['sl_dist'] = vol_adj['sl_dist']
                                signal['tp_dist'] = vol_adj['tp_dist']
                                event_logger.log_trade_modified(
                                    "VolatilityAdjustment",
                                    signal.get('tp_dist', 6.0),
                                    vol_adj['tp_dist'],
                                    "Volatility regime adjustment"
                                )

                            # Bank Filter (ML Only)
                            if strat_name == "MLPhysicsStrategy":
                                bank_blocked, bank_reason = bank_filter.should_block_trade(signal['side'])
                                if bank_blocked:
                                    event_logger.log_filter_check("BankFilter", signal['side'], False, bank_reason)
                                    continue
                                else:
                                    event_logger.log_filter_check("BankFilter", signal['side'], True)

                            strategy_results['executed'] = strat_name
                            logging.info(f"✅ STANDARD EXEC: {signal['strategy']} signal")
                            event_logger.log_strategy_execution(signal.get('strategy', strat_name), "STANDARD")

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
                                rej_blocked, rej_reason = rejection_filter.should_block_trade(sig['side'])
                                if rej_blocked:
                                    event_logger.log_rejection_block("RejectionFilter", sig['side'], rej_reason or "Rejection bias")
                                    del pending_loose_signals[s_name]; continue

                                # HTF FVG
                                fvg_blocked, fvg_reason = htf_fvg_filter.check_signal_blocked(sig['side'], current_price, None, None)
                                if fvg_blocked:
                                    logging.info(f"🚫 BLOCKED (HTF FVG): {fvg_reason}")
                                    event_logger.log_filter_check("HTF_FVG", sig['side'], False, fvg_reason)
                                    del pending_loose_signals[s_name]; continue
                                else:
                                    event_logger.log_filter_check("HTF_FVG", sig['side'], True)

                                # === [FIX 1] UPDATED BLOCKER CHECK ===
                                struct_blocked, struct_reason = structure_blocker.should_block_trade(sig['side'], current_price)
                                if struct_blocked:
                                    logging.info(f"🚫 {struct_reason}")
                                    event_logger.log_filter_check("StructureBlocker", sig['side'], False, struct_reason)
                                    del pending_loose_signals[s_name]; continue
                                else:
                                    event_logger.log_filter_check("StructureBlocker", sig['side'], True)
                                # =====================================

                                chop_blocked, chop_reason = chop_filter.should_block_trade(sig['side'], rejection_filter.prev_day_pm_bias)
                                if chop_blocked:
                                    event_logger.log_filter_check("ChopFilter", sig['side'], False, chop_reason)
                                    del pending_loose_signals[s_name]; continue
                                else:
                                    event_logger.log_filter_check("ChopFilter", sig['side'], True)

                                ext_blocked, ext_reason = extension_filter.should_block_trade(sig['side'])
                                if ext_blocked:
                                    event_logger.log_filter_check("ExtensionFilter", sig['side'], False, ext_reason)
                                    del pending_loose_signals[s_name]; continue
                                else:
                                    event_logger.log_filter_check("ExtensionFilter", sig['side'], True)

                                # Volatility
                                should_trade, vol_adj = check_volatility(new_df, sig.get('sl_dist', 4.0), sig.get('tp_dist', 6.0))
                                if not should_trade:
                                    event_logger.log_filter_check("VolatilityFilter", sig['side'], False, "Volatility check failed")
                                    del pending_loose_signals[s_name]; continue
                                else:
                                    event_logger.log_filter_check("VolatilityFilter", sig['side'], True)

                                if vol_adj.get('adjustment_applied', False):
                                    sig['sl_dist'] = vol_adj['sl_dist']
                                    sig['tp_dist'] = vol_adj['tp_dist']
                                    event_logger.log_trade_modified(
                                        "VolatilityAdjustment",
                                        sig.get('tp_dist', 6.0),
                                        vol_adj['tp_dist'],
                                        "Volatility regime adjustment"
                                    )

                                logging.info(f"✅ LOOSE EXEC: {s_name}")
                                event_logger.log_strategy_execution(s_name, "LOOSE")
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
                                        # Enhanced event logging: Strategy signal generated
                                        event_logger.log_strategy_signal(
                                            strategy_name=signal.get('strategy', s_name),
                                            side=signal['side'],
                                            tp_dist=signal.get('tp_dist', 6.0),
                                            sl_dist=signal.get('sl_dist', 4.0),
                                            price=current_price,
                                            additional_info={"execution_type": "LOOSE"}
                                        )

                                        rej_blocked, rej_reason = rejection_filter.should_block_trade(signal['side'])
                                        if rej_blocked:
                                            event_logger.log_rejection_block("RejectionFilter", signal['side'], rej_reason or "Rejection bias")
                                            continue

                                        fvg_blocked, fvg_reason = htf_fvg_filter.check_signal_blocked(signal['side'], current_price, None, None)
                                        if fvg_blocked:
                                            event_logger.log_filter_check("HTF_FVG", signal['side'], False, fvg_reason)
                                            continue
                                        else:
                                            event_logger.log_filter_check("HTF_FVG", signal['side'], True)

                                        # === [FIX 2] UPDATED BLOCKER CHECK ===
                                        struct_blocked, struct_reason = structure_blocker.should_block_trade(signal['side'], current_price)
                                        if struct_blocked:
                                            event_logger.log_filter_check("StructureBlocker", signal['side'], False, struct_reason)
                                            continue
                                        else:
                                            event_logger.log_filter_check("StructureBlocker", signal['side'], True)
                                        # =====================================

                                        chop_blocked, chop_reason = chop_filter.should_block_trade(signal['side'], rejection_filter.prev_day_pm_bias)
                                        if chop_blocked:
                                            event_logger.log_filter_check("ChopFilter", signal['side'], False, chop_reason)
                                            continue
                                        else:
                                            event_logger.log_filter_check("ChopFilter", signal['side'], True)

                                        ext_blocked, ext_reason = extension_filter.should_block_trade(signal['side'])
                                        if ext_blocked:
                                            event_logger.log_filter_check("ExtensionFilter", signal['side'], False, ext_reason)
                                            continue
                                        else:
                                            event_logger.log_filter_check("ExtensionFilter", signal['side'], True)

                                        # Volatility
                                        should_trade, vol_adj = check_volatility(new_df, signal.get('sl_dist', 4.0), signal.get('tp_dist', 6.0))
                                        if not should_trade:
                                            event_logger.log_filter_check("VolatilityFilter", signal['side'], False, "Volatility check failed")
                                            continue
                                        else:
                                            event_logger.log_filter_check("VolatilityFilter", signal['side'], True)

                                        if vol_adj.get('adjustment_applied', False):
                                            signal['sl_dist'] = vol_adj['sl_dist']
                                            signal['tp_dist'] = vol_adj['tp_dist']
                                            event_logger.log_trade_modified(
                                                "VolatilityAdjustment",
                                                signal.get('tp_dist', 6.0),
                                                vol_adj['tp_dist'],
                                                "Volatility regime adjustment"
                                            )

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
