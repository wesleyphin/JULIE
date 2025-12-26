"""
ProjectX Gateway API Client

Client for interacting with the ProjectX Gateway API for live futures trading.
Handles authentication, market data retrieval, order placement, and position management.
"""
import requests
import pandas as pd
import datetime
import time
import logging
from zoneinfo import ZoneInfo
import uuid
from typing import Dict, Optional, List, Tuple

from config import CONFIG, refresh_target_symbol, determine_current_contract_symbol
from event_logger import event_logger


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
    # Class-level (shared) rate limiting - all instances share these
    _shared_bar_timestamps = []
    _shared_general_timestamps = []
    _shared_last_bar_fetch = None
    _shared_lock = None  # Will be initialized on first use

    # Class-level rate limit config
    SHARED_BAR_RATE_LIMIT = 50
    SHARED_BAR_RATE_WINDOW = 30
    SHARED_GENERAL_RATE_LIMIT = 200
    SHARED_GENERAL_RATE_WINDOW = 60
    SHARED_MIN_FETCH_INTERVAL = 0.5  # Minimum 500ms between any bar fetches across all instances

    @classmethod
    def _get_lock(cls):
        """Thread-safe lock initialization"""
        if cls._shared_lock is None:
            import threading
            cls._shared_lock = threading.Lock()
        return cls._shared_lock

    @classmethod
    def _shared_check_bar_rate_limit(cls) -> bool:
        """Check shared rate limit for bar fetches across all client instances"""
        with cls._get_lock():
            now = time.time()
            # Clean old timestamps
            cls._shared_bar_timestamps = [
                t for t in cls._shared_bar_timestamps
                if now - t < cls.SHARED_BAR_RATE_WINDOW
            ]
            # Check if we're approaching limit (leave buffer of 10)
            if len(cls._shared_bar_timestamps) >= cls.SHARED_BAR_RATE_LIMIT - 10:
                logging.warning(f"Shared bar rate limit ({len(cls._shared_bar_timestamps)}/{cls.SHARED_BAR_RATE_LIMIT}). Using cache.")
                return False
            # Enforce minimum interval between ANY bar fetches
            if cls._shared_last_bar_fetch is not None:
                elapsed = now - cls._shared_last_bar_fetch
                if elapsed < cls.SHARED_MIN_FETCH_INTERVAL:
                    wait_time = cls.SHARED_MIN_FETCH_INTERVAL - elapsed
                    time.sleep(wait_time)
            return True

    @classmethod
    def _shared_track_bar_fetch(cls):
        """Track a bar fetch request in shared rate limiter"""
        with cls._get_lock():
            now = time.time()
            cls._shared_bar_timestamps.append(now)
            cls._shared_last_bar_fetch = now

    def __init__(self, contract_root: Optional[str] = None, target_symbol: Optional[str] = None):
        self.session = requests.Session()
        self.token = None
        self.token_expiry = None
        self.base_url = CONFIG['REST_BASE_URL']
        self.et = ZoneInfo('America/New_York')

        # Contract configuration (allows per-instance override)
        self.contract_root = contract_root or CONFIG.get('CONTRACT_ROOT', 'MES')
        self.target_symbol = target_symbol or CONFIG.get('TARGET_SYMBOL')

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
            logging.warning(f"Approaching general rate limit ({len(self.general_request_timestamps)}/{self.GENERAL_RATE_LIMIT})")
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
            logging.info(f"Authenticating to {self.base_url}...")
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
            logging.info("Authentication successful (JWT token acquired, valid 24h)")
            self._track_general_request()
        except Exception as e:
            logging.error(f"Login Failed: {e}")
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
                    logging.info("Session token refreshed")
                return True
            else:
                logging.warning(f"Session validation failed: {resp.status_code}")
                return False
        except Exception as e:
            logging.error(f"Session validation error: {e}")
            return False

    def fetch_accounts(self) -> Optional[int]:
        """
        Retrieve active accounts and PROMPT USER for selection using beautiful UI.
        """
        # If a specific ID is hardcoded in CONFIG, use it automatically (good for automation)
        if CONFIG.get('ACCOUNT_ID'):
            self.account_id = CONFIG['ACCOUNT_ID']
            logging.info(f"Using Hardcoded Account ID from Config: {self.account_id}")
            return self.account_id

        try:
            # Try to use the beautiful account selector UI
            from account_selector import select_account_interactive

            selected = select_account_interactive(self.session)

            if selected is None:
                logging.warning("Account selection cancelled")
                return None

            # Handle single account selection (julie001 doesn't support multi-account)
            if isinstance(selected, list):
                # User selected "Monitor All" but julie001 can only trade one account
                print("\n⚠️  Note: Main trading bot can only trade ONE account at a time.")
                print("    Using the first account from your selection.\n")
                self.account_id = selected[0] if selected else None
            else:
                self.account_id = selected

            if self.account_id:
                logging.info(f"User selected account ID: {self.account_id}")
                return self.account_id
            else:
                logging.warning("No account selected")
                return None

        except ImportError:
            # Fallback to simple text-based selection if account_selector not available
            logging.warning("Beautiful UI not available, using simple selection")
            return self._fetch_accounts_simple()
        except Exception as e:
            logging.error(f"Error in account selection: {e}")
            return self._fetch_accounts_simple()

    def _fetch_accounts_simple(self) -> Optional[int]:
        """Fallback simple text-based account selection"""
        url = f"{self.base_url}/api/Account/search"
        payload = {"onlyActiveAccounts": True}

        try:
            resp = self.session.post(url, json=payload)
            self._track_general_request()
            resp.raise_for_status()
            data = resp.json()

            if 'accounts' in data and len(data['accounts']) > 0:
                print("\n" + "="*40)
                print("SELECT AN ACCOUNT TO TRADE")
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
                        selection = input(f"Enter number (1-{len(accounts)}): ")
                        choice_idx = int(selection) - 1
                        if 0 <= choice_idx < len(accounts):
                            selected_acc = accounts[choice_idx]
                            self.account_id = selected_acc.get('id')
                            print(f"Selected: {selected_acc.get('name')} (ID: {self.account_id})")
                            logging.info(f"User selected account ID: {self.account_id}")
                            return self.account_id
                        else:
                            print(f"Invalid number. Please enter 1-{len(accounts)}.")
                    except ValueError:
                        print("Please enter a valid number.")
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
        # Search using the root symbol (e.g., "MES") to find all contracts
        payload = {
            "live": False,  # Set to False to find Topstep tradable contracts
            "searchText": self.contract_root
        }

        try:
            logging.info(f"Searching for contracts with symbol: {payload['searchText']}...")
            resp = self.session.post(url, json=payload)
            self._track_general_request()
            resp.raise_for_status()
            data = resp.json()

            if 'contracts' in data and len(data['contracts']) > 0:
                # TARGET_SYMBOL is short form like "MES.Z25" for matching
                target = self.target_symbol or determine_current_contract_symbol(self.contract_root)
                for contract in data['contracts']:
                    contract_id = contract.get('id', '')
                    contract_name = contract.get('name', '')
                    logging.info(f"  Found: {contract_name} ({contract_id})")

                    # Match contract IDs like "CON.F.US.MES.Z25" that end with ".MES.Z25"
                    if contract_id.endswith(f".{target}"):
                        self.contract_id = contract_id
                        logging.info(f"Selected Contract ID: {self.contract_id}")
                        return self.contract_id

                # Fallback: Just take the first one if exact matching logic above misses
                self.contract_id = data['contracts'][0].get('id')
                logging.warning(f"Exact match not confirmed, using first result: {self.contract_id}")
                return self.contract_id
            else:
                logging.warning("No contracts found in search results.")
                return None
        except Exception as e:
            logging.error(f"Failed to fetch contracts: {e}")
            return None

    def get_market_data(self, lookback_minutes: int = 20000, force_fetch: bool = False) -> pd.DataFrame:
        """
        Fetch historical bars with rate limiting.
        UPDATED: limit increased to 20,000 for deep history (~14 days of 1m data).
        Endpoint: POST /api/History/retrieveBars
        Rate Limit: 50 requests / 30 seconds
        """
        # SHARED rate limit check first (coordinates across MES/MNQ clients)
        if not ProjectXClient._shared_check_bar_rate_limit():
            return self.cached_df

        now = time.time()

        # Instance-level tracking (for per-client diagnostics)
        self.bar_fetch_timestamps = [
            t for t in self.bar_fetch_timestamps
            if now - t < self.BAR_RATE_WINDOW
        ]

        # Instance-level minimum interval (skip if force_fetch)
        if self.last_bar_fetch_time is not None:
            if now - self.last_bar_fetch_time < self.MIN_FETCH_INTERVAL and not force_fetch:
                return self.cached_df

        if self.contract_id is None:
            logging.error("No contract ID set. Call fetch_contracts() first.")
            return self.cached_df

        # Calculate start time based on the massive lookback
        end_time = datetime.datetime.now(datetime.timezone.utc)
        start_time = end_time - datetime.timedelta(minutes=lookback_minutes)

        url = f"{self.base_url}/api/History/retrieveBars"
        payload = {
            "accountId": self.account_id,
            "contractId": self.contract_id,
            "live": False,
            "limit": 20000,  # UPDATED TO 20,000 for deep history
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

            # Track request in shared limiter immediately after making the call
            ProjectXClient._shared_track_bar_fetch()

            if resp.status_code == 429:
                logging.warning(f"Rate limited (429) for {self.contract_root}. Backing off 5s...")
                time.sleep(5)
                return self.cached_df

            resp.raise_for_status()
            self.bar_fetch_timestamps.append(now)
            self.last_bar_fetch_time = now
            data = resp.json()

            # DEBUG: Print last bar timestamp from raw response
            if 'bars' in data and data['bars']:
                newest_raw_bar = data['bars'][0]
                logging.debug(f"API raw: newest bar t={newest_raw_bar.get('t')}, c={newest_raw_bar.get('c')}")

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

                logging.debug(f"Final df: first={df.index[0]}, last={df.index[-1]}, len={len(df)}")

                self.cached_df = df
                if not df.empty:
                    new_bar_ts = df.index[-1]
                    if self.last_bar_timestamp is None or new_bar_ts > self.last_bar_timestamp:
                        self.last_bar_timestamp = new_bar_ts
                return df
            else:
                logging.warning(f"API returned no bars for {self.contract_id} (timeframe: {start_time} to {end_time})")
                return self.cached_df if not self.cached_df.empty else pd.DataFrame()

        except requests.exceptions.HTTPError as e:
            if hasattr(e, 'response'):
                logging.error(f"HTTP Error: {e}")
                logging.error(f"Server Response: {e.response.text}")
                if e.response.status_code == 429:
                    logging.warning("Rate limited. Backing off...")
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
            # SHARED rate limit check first (coordinates across MES/MNQ clients)
            if not ProjectXClient._shared_check_bar_rate_limit():
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
                # Track in shared limiter immediately
                ProjectXClient._shared_track_bar_fetch()
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
        """Returns current rate limit usage (shared across all clients)"""
        now = time.time()
        # Use shared timestamps for accurate cross-client tracking
        shared_bar_recent = len([t for t in ProjectXClient._shared_bar_timestamps if now - t < ProjectXClient.SHARED_BAR_RATE_WINDOW])
        general_recent = len([t for t in self.general_request_timestamps if now - t < self.GENERAL_RATE_WINDOW])
        return f"Bars: {shared_bar_recent}/{ProjectXClient.SHARED_BAR_RATE_LIMIT} (30s) | General: {general_recent}/{self.GENERAL_RATE_LIMIT} (60s)"

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
        # sl_dist and tp_dist are in POINTS, convert to ticks
        # Use .get() with defaults for safety (all strategies should set these, but just in case)
        sl_points = float(signal.get('sl_dist', 4.0))
        tp_points = float(signal.get('tp_dist', 6.0))

        # Log warning if defaults were used (helps debug strategy issues)
        if 'sl_dist' not in signal:
            logging.warning(f"⚠️ Strategy {signal.get('strategy', 'Unknown')} missing sl_dist, using default 4.0")
        if 'tp_dist' not in signal:
            logging.warning(f"⚠️ Strategy {signal.get('strategy', 'Unknown')} missing tp_dist, using default 6.0")

        # Convert Points to Ticks
        # SL: Full conversion (Points / 0.25) - wider stops survive noise
        # TP: Raw points as ticks (1/4 size) - tighter targets bank profit early
        abs_sl_ticks = int(abs(sl_points / 0.25))  # e.g., 4.0 pts → 16 ticks (4.0 pts)
        abs_tp_ticks = int(abs(tp_points))          # e.g., 8.0 pts → 8 ticks (2.0 pts)


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

        # 5. Get order size from signal (allows volatility filter to reduce size)
        order_size = int(signal.get('size', 5))

        # 6. Construct Payload
        # Using 'ticks' with the correct signs calculated above
        payload = {
            "accountId": self.account_id,
            "contractId": self.contract_id,
            "clOrdId": unique_order_id,
            "type": 2,  # Market Order
            "side": side_code,
            "size": order_size,  # Use size from signal (volatility-adjusted)
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

            logging.info(f"SENDING ORDER: {signal['side']} @ ~{current_price:.2f}")
            logging.info(f"   TP: {tp_points}pts ({final_tp_ticks} ticks)")
            logging.info(f"   SL: {sl_points}pts ({final_sl_ticks} ticks)")

            resp = self.session.post(url, json=payload)
            self._track_general_request()

            if resp.status_code == 429:
                logging.error("Rate limited on order placement!")
                event_logger.log_error("RATE_LIMIT", "Order placement rate limited")
                return None

            if resp.status_code != 200:
                logging.error(f"HTTP Error {resp.status_code}: {resp.text[:500] if resp.text else 'Empty response'}")
                return None

            # Only parse JSON after confirming 200 status
            try:
                resp_data = resp.json()
            except Exception as json_err:
                logging.error(f"Failed to parse order response: {json_err}")
                return None

            # Check for business logic success
            if resp_data.get('success') is False:
                err_msg = resp_data.get('errorMessage', 'Unknown Rejection')
                logging.error(f"Order Rejected by Engine: {err_msg}")

                # Enhanced event logging: Order rejected
                event_logger.log_trade_order_rejected(
                    side=signal['side'],
                    price=current_price,
                    error_msg=err_msg,
                    strategy=signal.get('strategy', 'Unknown')
                )
                return None

            logging.info(f"Order Placed Successfully [{unique_order_id[:8]}]")

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
                'size': 5,  # Fixed size
                'avg_price': current_price
            }

            # Try to capture stop order ID from response if available
            # The exact field name depends on the API response structure
            if 'stopLossOrderId' in resp_data:
                self._active_stop_order_id = resp_data['stopLossOrderId']
                logging.debug(f"Captured stop order ID: {self._active_stop_order_id}")
            elif 'orderId' in resp_data:
                # Main order ID - we'll still need to search for bracket orders
                logging.debug(f"Main order ID: {resp_data['orderId']}")

            return resp_data

        except Exception as e:
            logging.error(f"Order exception: {e}")
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
                # logging.debug("Position/search 404. Trying GET fallback...")
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
            "clOrdId": str(uuid.uuid4()),  # Unique order ID for close
            "type": 2,  # Market Order
            "side": side_code,
            "size": position['size']
            # No brackets - just close the position
        }

        try:
            logging.info(f"CLOSING POSITION: {action} {position['size']} contracts to close {position['side']} @ ~{position['avg_price']:.2f}")

            resp = self.session.post(url, json=payload)
            self._track_general_request()

            if resp.status_code == 429:
                logging.error("Rate limited on position close!")
                event_logger.log_error("RATE_LIMIT", "Position close rate limited")
                return False

            if resp.status_code != 200:
                logging.error(f"Position close HTTP Error {resp.status_code}: {resp.text[:500] if resp.text else 'Empty response'}")
                event_logger.log_error("POSITION_CLOSE_FAILED", f"HTTP {resp.status_code}")
                return False

            # Only parse JSON after confirming 200 status
            try:
                resp_data = resp.json()
            except Exception as json_err:
                logging.error(f"Failed to parse close response: {json_err}")
                return False

            if resp_data.get('success', False):
                logging.info(f"Position close order submitted: {resp_data}")

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
                logging.error(f"Position close rejected: {resp_data}")
                event_logger.log_error("POSITION_CLOSE_FAILED", f"Failed to close position: {resp_data}")
                return False
        except Exception as e:
            logging.error(f"Position close exception: {e}")
            event_logger.log_error("POSITION_CLOSE_EXCEPTION", f"Exception closing position: {e}", exception=e)
            return False

    def close_and_reverse(self, new_signal: Dict, current_price: float, opposite_signal_count: int) -> Tuple[bool, int]:
        """
        Check current position, close if 3 opposite signals received, then place new order.
        Uses shadow position state to reduce API calls.
        """
        # Always sync shadow position with broker before deciding
        position = self.get_position()
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
        logging.info(f"OPPOSITE SIGNAL #{opposite_signal_count}/3: Current {position['side']} {position['size']} contracts, Signal: {new_signal['side']}")

        # If we've received 3 opposite signals, close and reverse
        if opposite_signal_count >= 3:
            logging.info(f"3 OPPOSITE SIGNALS RECEIVED - Closing {position['side']} position and reversing to {new_signal['side']}")

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
        logging.info(f"Waiting for {3 - opposite_signal_count} more opposite signals before closing position")
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
                    logging.info(f"Order {order_id} cancelled")
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
            logging.info(f"MODIFYING ORDER {order_id}: stopPrice={stop_price}, limitPrice={limit_price}")
            resp = self.session.post(url, json=payload)
            self._track_general_request()

            if resp.status_code == 200:
                data = resp.json()
                if data.get('success', False):
                    logging.info(f"Order {order_id} modified successfully")
                    return True
                else:
                    err = data.get('errorMessage', str(data))
                    logging.error(f"Order modification rejected: {err}")
                    return False
            else:
                logging.error(f"Order modification failed: {resp.status_code} - {resp.text}")
                return False

        except Exception as e:
            logging.error(f"Order modification exception: {e}")
            return False

    def modify_stop_to_breakeven(self, stop_price: float, side: str, known_size: int = None, stop_order_id: int = None) -> bool:
        """
        Aggressively modify stop to break-even.
        Updates:
        1. Removed 'Skipping' logic - Forces update to ensure safety.
        2. Improved Search - Logs exactly what it finds.
        3. Robust Fallback - If modify fails, immediately cancels and places new stop.
        """
        # 1. Determine position size
        position_size = known_size if known_size is not None else 1
        if position_size == 0:
            logging.warning("No position size provided. Aborting stop modification.")
            return False

        # 2. Use the stop price directly
        be_price = round(stop_price * 4) / 4  # Tick alignment

        # 3. Find stop order ID
        target_stop_id = stop_order_id or self._active_stop_order_id

        # If we don't have an ID, search for it
        if target_stop_id is None:
            orders = self.search_orders()
            # Filter for working stop orders (Type 4=Stop Market, 5=Stop Limit)
            stop_orders = [
                o for o in orders
                if o.get('type') in [4, 5]
                and str(o.get('status', '')).lower() in ['working', 'pending', 'accepted', 'active']
            ]

            if stop_orders:
                # Use the most recent one
                target_stop_id = stop_orders[0].get('orderId')
                current_stop_val = stop_orders[0].get('stopPrice') or stop_orders[0].get('triggerPrice') or stop_orders[0].get('price') or 0
                logging.info(f"Found active stop order {target_stop_id} @ {current_stop_val}")
            else:
                logging.warning("No active stop orders found via search.")

        # 4. EXECUTE MODIFICATION
        if target_stop_id:
            logging.info(f"🔒 MOVING STOP: {side} -> {be_price:.2f} (Order ID: {target_stop_id})")

            # Try standard modification
            if self.modify_order(target_stop_id, stop_price=be_price):
                logging.info(f"✅ STOP UPDATED to {be_price:.2f}")
                return True
            else:
                logging.warning(f"⚠️ Modify failed for {target_stop_id}. Attempting Cancel/Replace...")
                self.cancel_order(target_stop_id)
                # Clear cached ID since we just cancelled it
                self._active_stop_order_id = None
                time.sleep(0.5)  # Short wait for cancel to process

        # 5. FALLBACK: Place New Stop Order
        # This runs if target_stop_id was None OR if modify failed/cancelled
        logging.info(f"🔄 PLACING NEW STOP at {be_price:.2f}...")
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
            "clOrdId": str(uuid.uuid4()),  # Unique order ID for break-even stop
            "type": 4,  # Stop Market
            "side": side_code,
            "size": size,
            "stopPrice": be_price
        }

        try:
            logging.info(f"Placing break-even stop: {be_price:.2f} for {size} contracts ({side} position)")
            resp = self.session.post(url, json=payload)
            self._track_general_request()

            if resp.status_code == 200:
                data = resp.json()
                if data.get('success', False):
                    logging.info(f"BREAK-EVEN STOP PLACED at {be_price:.2f}")
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
            "clOrdId": str(uuid.uuid4()),  # Unique order ID
            "type": 4,  # Stop Market
            "side": side_code,
            "size": size,
            "stopPrice": be_price
        }

        try:
            logging.info(f"BREAK-EVEN (alt): Placing stop at {be_price:.2f} for {size} contracts")
            resp = self.session.post(url, json=payload)
            self._track_general_request()

            if resp.status_code == 200:
                data = resp.json()
                if data.get('success', False):
                    logging.info(f"Break-even stop placed at {be_price:.2f}")
                    return True
                else:
                    err = data.get('errorMessage', str(data))
                    logging.error(f"Break-even alt rejected: {err}")
            else:
                logging.error(f"Break-even alt failed: {resp.status_code}")
            return False
        except Exception as e:
            logging.error(f"Break-even alt exception: {e}")
            return False

    # ==========================================
    # ASYNC METHODS FOR ASYNCIO UPGRADE
    # ==========================================

    async def async_get_position(self) -> Dict:
        """
        Async version of get_position() for use in independent async tasks.

        Returns:
            Position dict with 'side', 'size', 'avg_price'
        """
        import aiohttp

        if not self._check_general_rate_limit():
            return {'side': None, 'size': 0, 'avg_price': 0.0}

        if self.account_id is None:
            return {'side': None, 'size': 0, 'avg_price': 0.0}

        url = f"{self.base_url}/api/Position/search"
        payload = {"accountId": self.account_id}
        headers = {"Authorization": f"Bearer {self.token}"}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers) as resp:
                    self._track_general_request()

                    # Fallback to GET if POST returns 404
                    if resp.status == 404:
                        fallback_url = f"{self.base_url}/api/Position"
                        async with session.get(fallback_url, params=payload, headers=headers) as resp:
                            self._track_general_request()

                    if resp.status == 200:
                        data = await resp.json()
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

                        return {'side': None, 'size': 0, 'avg_price': 0.0}

                    elif resp.status == 404:
                        return {'side': None, 'size': 0, 'avg_price': 0.0}
                    else:
                        logging.warning(f"Async position check failed: {resp.status}")
                        return {'side': None, 'size': 0, 'avg_price': 0.0}

        except Exception as e:
            logging.error(f"Async position check error: {e}")
            return {'side': None, 'size': 0, 'avg_price': 0.0}

    async def async_validate_session(self) -> bool:
        """
        Async version of validate_session() for heartbeat task.

        Returns:
            True if session is valid, False otherwise
        """
        import aiohttp

        if not self._check_general_rate_limit():
            return self.token is not None

        url = f"{self.base_url}/api/Auth/validate"
        headers = {"Authorization": f"Bearer {self.token}"}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers) as resp:
                    self._track_general_request()

                    if resp.status == 200:
                        data = await resp.json()
                        if 'newToken' in data:
                            self.token = data['newToken']
                            self.token_expiry = datetime.datetime.now() + datetime.timedelta(hours=24)
                            self.session.headers.update({"Authorization": f"Bearer {self.token}"})
                            logging.info("Session token refreshed (async)")
                        return True
                    else:
                        logging.warning(f"Async session validation failed: {resp.status}")
                        return False

        except Exception as e:
            logging.error(f"Async session validation error: {e}")
            return False
