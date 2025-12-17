#!/usr/bin/env python3
"""
JULIE Trading Bot - Standalone Terminal Monitor
Displays real-time signals, positions, and market data without modifying the main bot
"""

import requests
import time
from zoneinfo import ZoneInfo
import re
from datetime import datetime
from pathlib import Path
from threading import Thread
from config import CONFIG
from terminal_ui import get_ui
from account_selector import select_account_interactive


def get_current_session(dt):
    """Determine current trading session based on ET time"""
    hour = dt.hour

    # ASIA: 18:00-03:00 ET
    if hour >= 18 or hour < 3:
        return 'ASIA'
    # LONDON: 03:00-08:00 ET
    elif 3 <= hour < 8:
        return 'LONDON'
    # NY_AM: 08:00-12:00 ET
    elif 8 <= hour < 12:
        return 'NY_AM'
    # NY_PM: 12:00-17:00 ET
    elif 12 <= hour < 17:
        return 'NY_PM'
    # OFF_HOURS: 17:00-18:00 ET
    else:
        return 'OFF_HOURS'


class LogMonitor:
    """Monitors the bot's log file for signals and events"""

    def __init__(self, log_file="topstep_live_bot.log"):
        self.log_file = Path(log_file)
        self.ui = get_ui()
        self.last_position = 0

    def tail_log(self):
        """Tail the log file and extract events"""
        if not self.log_file.exists():
            self.ui.add_event("MONITOR", f"Waiting for log file: {self.log_file}")
            return

        with open(self.log_file, 'r') as f:
            # Seek to end on first run
            if self.last_position == 0:
                f.seek(0, 2)  # Go to end
                self.last_position = f.tell()
            else:
                f.seek(self.last_position)

            lines = f.readlines()
            self.last_position = f.tell()

            for line in lines:
                self.parse_log_line(line.strip())

    def parse_log_line(self, line):
        """Extract meaningful information from log lines"""
        if not line:
            return

        # Extract signal executions
        if "FAST EXEC:" in line or "STANDARD EXEC:" in line or "LOOSE EXEC:" in line:
            # Example: "2024-12-14 15:30:45 [INFO] ✅ FAST EXEC: RegimeAdaptive signal"
            match = re.search(r'(FAST|STANDARD|LOOSE) EXEC:\s+(\w+)', line)
            if match:
                exec_type = match.group(1)
                strategy = match.group(2)
                self.ui.add_event("SIGNAL", f"{exec_type} execution: {strategy}")

        # Extract order placements
        elif "SENDING ORDER:" in line:
            # Example: "🚀 SENDING ORDER: LONG @ ~5875.25"
            match = re.search(r'SENDING ORDER:\s+(LONG|SHORT)\s+@\s+~?(\d+\.?\d*)', line)
            if match:
                side = match.group(1)
                price = match.group(2)
                self.ui.add_event("ORDER", f"Placing {side} order @ {price}")

        # Extract TP/SL info
        elif "TP:" in line and "pts" in line:
            match = re.search(r'TP:\s+([\d.]+)pts', line)
            if match:
                tp = match.group(1)
                # Will be on next line usually, but we'll capture it

        # Extract filter blocks
        elif "BLOCKED" in line:
            if "HTF FVG" in line:
                self.ui.add_event("FILTER", "Blocked by HTF FVG filter")
                self.ui.update_filter_status('HTF FVG', False, "Signal blocked")
            elif "Chop" in line or "CHOP" in line:
                self.ui.add_event("FILTER", "Blocked by Chop filter")
                self.ui.update_filter_status('Chop', False, "Signal blocked")
            elif "Extension" in line:
                self.ui.add_event("FILTER", "Blocked by Extension filter")
                self.ui.update_filter_status('Extension', False, "Signal blocked")

        # Extract position close
        elif "CLOSING POSITION:" in line:
            match = re.search(r'CLOSING POSITION:\s+(BUY|SELL)', line)
            if match:
                action = match.group(1)
                self.ui.add_event("TRADE", f"Closing position: {action}")

        # Extract break-even triggers
        elif "BREAK-EVEN" in line and "trigger" in line.lower():
            self.ui.add_event("TRADE", "Break-even stop triggered")

        # Extract session info
        elif "Bar:" in line:
            # Example: "Bar: 2024-12-14 15:30:00 ET | Price: 5875.25"
            match = re.search(r'Price:\s+([\d.]+)', line)
            if match:
                price = float(match.group(1))
                # Don't spam the UI, will be updated by API poller

        # Extract strategy signals before filters
        elif any(strat in line for strat in ["RegimeAdaptive", "IntradayDip", "Confluence", "ORB", "ICT", "MLPhysics", "DynamicEngine"]):
            if "signal" in line.lower():
                for strategy in ["RegimeAdaptive", "IntradayDip", "Confluence", "ORB", "ICTModel", "MLPhysics", "DynamicEngine"]:
                    if strategy in line:
                        # Try to extract side
                        side_match = re.search(r'(LONG|SHORT)', line)
                        if side_match:
                            side = side_match.group(1)
                            # Don't add to signals here, wait for execution or block


class APIMonitor:
    """Monitors the TopstepX API for positions and market data"""

    def __init__(self):
        self.ui = get_ui()
        self.session = requests.Session()
        self.token = None
        self.base_url = CONFIG['REST_BASE_URL']
        self.account_ids = []  # Support multiple accounts
        self.account_id = None  # Primary account for backward compatibility
        self.contract_id = None
        self.et = ZoneInfo('America/New_York')
        self.monitor_all = False  # Flag to indicate if monitoring all accounts

    def login(self):
        """Authenticate with the API"""
        url = f"{self.base_url}/api/Auth/loginKey"
        payload = {
            "userName": CONFIG['USERNAME'],
            "apiKey": CONFIG['API_KEY']
        }

        try:
            self.ui.add_event("API", "Authenticating...")
            resp = self.session.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

            if data.get('errorCode') and data.get('errorCode') != 0:
                raise ValueError(f"Login Failed: {data.get('errorMessage', 'Unknown Error')}")

            self.token = data.get('token')
            if not self.token:
                raise ValueError("Login response missing token")

            self.session.headers.update({"Authorization": f"Bearer {self.token}"})
            self.ui.add_event("API", "✓ Authentication successful")
            return True

        except Exception as e:
            self.ui.add_event("ERROR", f"Login failed: {e}")
            return False

    def fetch_account_id(self):
        """Get the account ID using beautiful interactive selection UI"""
        # Use the beautiful account selector
        self.ui.add_event("API", "Opening account selection interface...")

        selected = select_account_interactive(self.session)

        if selected is None:
            self.ui.add_event("ERROR", "Account selection cancelled")
            return False

        # Handle list of accounts (Monitor All)
        if isinstance(selected, list):
            self.account_ids = selected
            self.account_id = selected[0] if selected else None  # Use first as primary
            self.monitor_all = True
            self.ui.add_event("API", f"✓ Monitoring ALL {len(selected)} accounts")
            self.ui.update_account_info({
                'account_id': f"{len(selected)} accounts",
                'monitor_all': True
            })
            return True

        # Handle single account
        self.account_ids = [selected]
        self.account_id = selected
        self.monitor_all = False
        self.ui.add_event("API", f"✓ Monitoring account: {selected}")
        self.ui.update_account_info({
            'account_id': selected,
            'monitor_all': False
        })
        return True

    def fetch_contract_id(self):
        """Get the contract ID for current symbol"""
        from config import refresh_target_symbol
        refresh_target_symbol()

        url = f"{self.base_url}/api/Contract/search"
        payload = {
            "live": False,
            "searchText": CONFIG.get('TARGET_SYMBOL', 'CON.F.US.MES.Z25')
        }

        try:
            resp = self.session.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

            if 'contracts' in data and len(data['contracts']) > 0:
                target = CONFIG.get('TARGET_SYMBOL', 'CON.F.US.MES.Z25')
                for contract in data['contracts']:
                    contract_id = contract.get('id', '')
                    if f".{target}." in contract_id or contract_id.endswith(f".{target}"):
                        self.contract_id = contract_id
                        self.ui.add_event("API", f"✓ Contract: {target}")
                        return True

                # Fallback to first result
                self.contract_id = data['contracts'][0].get('id')
                self.ui.add_event("API", f"✓ Contract: {self.contract_id}")
                return True
            else:
                self.ui.add_event("ERROR", "No contracts found")
                return False

        except Exception as e:
            self.ui.add_event("ERROR", f"Failed to fetch contract: {e}")
            return False

    def fetch_position(self):
        """Fetch current position from API (supports multiple accounts)"""
        if not self.account_ids:
            return None

        # If monitoring single account, use simple fetch
        if not self.monitor_all:
            return self._fetch_single_account_position(self.account_id)

        # If monitoring all accounts, aggregate positions
        all_positions = []
        for acc_id in self.account_ids:
            pos = self._fetch_single_account_position(acc_id)
            if pos and pos.get('side') is not None:
                all_positions.append(pos)

        # For now, return first active position (can be enhanced to show all)
        if all_positions:
            return all_positions[0]

        return {'side': None, 'size': 0, 'avg_price': 0.0}

    def _fetch_single_account_position(self, account_id):
        """Fetch position for a single account"""
        if not account_id:
            return None

        url = f"{self.base_url}/api/Position/search"
        payload = {"accountId": account_id}

        try:
            resp = self.session.post(url, json=payload, timeout=5)

            if resp.status_code == 404:
                # No position
                return {'side': None, 'size': 0, 'avg_price': 0.0}

            if resp.status_code == 200:
                data = resp.json()
                positions = data.get('positions', data) if isinstance(data, dict) else data

                for pos in positions:
                    if pos.get('contractId') == self.contract_id:
                        size = pos.get('size', 0)
                        avg_price = pos.get('averagePrice', 0.0)
                        if size > 0:
                            return {'side': 'LONG', 'size': size, 'avg_price': avg_price, 'account_id': account_id}
                        elif size < 0:
                            return {'side': 'SHORT', 'size': abs(size), 'avg_price': avg_price, 'account_id': account_id}

                return {'side': None, 'size': 0, 'avg_price': 0.0}

            return None

        except Exception as e:
            # Don't spam errors for timeout/network issues
            return None

    def fetch_market_data(self):
        """Fetch latest market price"""
        if not self.account_id or not self.contract_id:
            return None

        from datetime import datetime, timedelta, timezone
        import pandas as pd

        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(minutes=10)

        url = f"{self.base_url}/api/History/retrieveBars"
        payload = {
            "accountId": self.account_id,
            "contractId": self.contract_id,
            "live": False,
            "limit": 10,
            "startTime": start_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
            "endTime": end_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
            "unit": 2,
            "unitNumber": 1
        }

        try:
            resp = self.session.post(url, json=payload, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                if 'bars' in data and data['bars']:
                    latest_bar = data['bars'][0]  # API returns newest first
                    return float(latest_bar.get('c', 0.0))
            return None
        except:
            return None


def main():
    """Main monitoring application"""
    print("=" * 60)
    print("JULIE TRADING BOT - TERMINAL MONITOR")
    print("Real-time display of signals, positions, and market data")
    print("=" * 60)
    print()

    # Initialize UI
    ui = get_ui()
    ui.update_market_context({
        'symbol': CONFIG.get('TARGET_SYMBOL', 'MES'),
        'session': 'CONNECTING'
    })
    ui.start(refresh_rate=1.0)

    ui.add_event("SYSTEM", "JULIE Monitor starting...")
    ui.add_event("SYSTEM", f"Target: {CONFIG.get('TARGET_SYMBOL', 'MES')}")

    # Initialize API monitor
    api_monitor = APIMonitor()

    if not api_monitor.login():
        ui.add_event("ERROR", "Failed to authenticate - check config.py credentials")
        time.sleep(5)
        ui.stop()
        return

    if not api_monitor.fetch_account_id():
        ui.add_event("ERROR", "Failed to fetch account")
        time.sleep(5)
        ui.stop()
        return

    if not api_monitor.fetch_contract_id():
        ui.add_event("ERROR", "Failed to fetch contract")
        time.sleep(5)
        ui.stop()
        return

    # Initialize log monitor
    log_monitor = LogMonitor()

    ui.add_event("SYSTEM", "✓ Monitor initialized - watching for activity")

    # Main monitoring loop
    last_position_check = 0
    last_price_check = 0
    last_log_check = 0

    try:
        while True:
            now = time.time()

            # Check position every 2 seconds
            if now - last_position_check > 2.0:
                position = api_monitor.fetch_position()
                if position:
                    if position['side'] is not None:
                        # Estimate TP/SL (we don't have exact values without log parsing)
                        # Using common defaults
                        tp_dist = 6.0
                        sl_dist = 4.0

                        if position['side'] == 'LONG':
                            tp_price = position['avg_price'] + tp_dist
                            sl_price = position['avg_price'] - sl_dist
                        else:
                            tp_price = position['avg_price'] - tp_dist
                            sl_price = position['avg_price'] + sl_dist

                        ui.update_position({
                            'active': True,
                            'side': position['side'],
                            'entry_price': position['avg_price'],
                            'tp_price': tp_price,
                            'sl_price': sl_price,
                            'bars_held': 0,  # Can't track without integration
                            'strategy': 'Unknown'  # Can't track without integration
                        })
                    else:
                        ui.update_position({'active': False})

                last_position_check = now

            # Check price every 3 seconds
            if now - last_price_check > 3.0:
                price = api_monitor.fetch_market_data()
                if price:
                    # Determine session
                    current_time = datetime.now(ZoneInfo('America/New_York'))
                    session = get_current_session(current_time)

                    ui.update_market_context({
                        'price': price,
                        'session': session
                    })

                last_price_check = now

            # Check log file every 0.5 seconds
            if now - last_log_check > 0.5:
                log_monitor.tail_log()
                last_log_check = now

            time.sleep(0.1)

    except KeyboardInterrupt:
        ui.add_event("SYSTEM", "Monitor stopped by user")
        ui.stop()
        print("\n\nMonitor stopped.")


if __name__ == "__main__":
    main()
