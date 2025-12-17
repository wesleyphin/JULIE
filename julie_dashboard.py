#!/usr/bin/env python3
"""
Julie Dashboard - Integrated Trading Bot + Monitor
Runs the bot on multiple accounts and displays everything in a beautiful UI
"""

import requests
import time
import threading
import logging
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Dict, List, Optional
from collections import defaultdict

from config import CONFIG, refresh_target_symbol
from dashboard_ui import get_dashboard
from account_selector import select_account_interactive
from client import ProjectXClient


class MultiAccountTracker:
    """Tracks positions, P&L, and trades across multiple accounts"""

    def __init__(self):
        self.accounts = {}  # account_id -> account data
        self.positions = {}  # account_id -> position
        self.pnl = {}  # account_id -> P&L
        self.active_trades = {}  # account_id -> active trade info
        self.lock = threading.Lock()

    def add_account(self, account_id: str, account_name: str, is_trading: bool = True):
        """Register an account for tracking"""
        with self.lock:
            self.accounts[account_id] = {
                'id': account_id,
                'name': account_name,
                'daily_pnl': 0.0,
                'total_pnl': 0.0,
                'trades_today': 0,
                'wins': 0,
                'losses': 0,
                'is_trading': is_trading,
                'bot_active': True
            }
            self.positions[account_id] = None
            self.pnl[account_id] = 0.0
            self.active_trades[account_id] = None

    def update_position(self, account_id: str, position: Optional[Dict]):
        """Update position for an account"""
        with self.lock:
            self.positions[account_id] = position

    def update_pnl(self, account_id: str, pnl: float):
        """Update P&L for an account"""
        with self.lock:
            self.pnl[account_id] = pnl
            if account_id in self.accounts:
                self.accounts[account_id]['daily_pnl'] = pnl

    def record_trade(self, account_id: str, side: str, entry_price: float, exit_price: float, size: int):
        """Record a completed trade"""
        with self.lock:
            if account_id in self.accounts:
                self.accounts[account_id]['trades_today'] += 1

                # Calculate P&L
                if side == 'LONG':
                    pnl = (exit_price - entry_price) * size * 5.0  # $5 per point for MES
                else:
                    pnl = (entry_price - exit_price) * size * 5.0

                if pnl > 0:
                    self.accounts[account_id]['wins'] += 1
                else:
                    self.accounts[account_id]['losses'] += 1

                self.accounts[account_id]['total_pnl'] += pnl

    def get_summary(self) -> Dict:
        """Get summary of all accounts"""
        with self.lock:
            total_pnl = sum(acc.get('daily_pnl', 0) for acc in self.accounts.values())
            total_trades = sum(acc.get('trades_today', 0) for acc in self.accounts.values())
            total_wins = sum(acc.get('wins', 0) for acc in self.accounts.values())
            total_losses = sum(acc.get('losses', 0) for acc in self.accounts.values())

            return {
                'total_accounts': len(self.accounts),
                'total_pnl': total_pnl,
                'total_trades': total_trades,
                'win_rate': (total_wins / total_trades * 100) if total_trades > 0 else 0,
                'accounts': list(self.accounts.values()),
                'active_positions': sum(1 for p in self.positions.values() if p and p.get('side'))
            }


class AccountBotRunner(threading.Thread):
    """Runs the trading bot for a single account in a separate thread"""

    def __init__(self, account_id: str, account_name: str, tracker: MultiAccountTracker, ui, is_trading_account: bool = True):
        super().__init__(daemon=True)
        self.account_id = account_id
        self.account_name = account_name
        self.tracker = tracker
        self.ui = ui
        self.running = True
        self.client = None
        self.is_trading_account = is_trading_account  # False = monitor only

        # Log mode
        mode = "TRADING" if is_trading_account else "MONITOR"
        self.mode_indicator = f"[{mode}]"

    def run(self):
        """Main bot loop for this account"""
        try:
            # Initialize client for this account
            self.client = ProjectXClient()
            self.client.login()
            self.client.account_id = self.account_id
            self.client.fetch_contracts()

            self.ui.add_event("BOT", f"{self.mode_indicator} Started for {self.account_name}")

            # Main trading loop (simplified - you'd add full strategy logic here)
            while self.running:
                try:
                    # Fetch position
                    position = self.client.get_position()
                    self.tracker.update_position(self.account_id, position)

                    # Calculate P&L if position exists
                    if position and position.get('side'):
                        # Get current price (simplified)
                        market_data = self.client.get_market_data(lookback_minutes=1)
                        if not market_data.empty:
                            current_price = market_data.iloc[-1]['close']
                            entry_price = position.get('avg_price', 0)
                            size = position.get('size', 1)

                            if position['side'] == 'LONG':
                                pnl = (current_price - entry_price) * size * 5.0
                            else:
                                pnl = (entry_price - current_price) * size * 5.0

                            self.tracker.update_pnl(self.account_id, pnl)

                    time.sleep(5)  # Check every 5 seconds

                except Exception as e:
                    self.ui.add_event("ERROR", f"{self.account_name}: {str(e)[:50]}")
                    time.sleep(10)

        except Exception as e:
            self.ui.add_event("ERROR", f"Failed to start bot for {self.account_name}: {e}")

    def stop(self):
        """Stop the bot"""
        self.running = False


class JulieDashboard:
    """Main dashboard orchestrator"""

    def __init__(self):
        self.ui = get_dashboard()
        self.tracker = MultiAccountTracker()
        self.bot_threads = []
        self.session = requests.Session()
        self.running = False

    def authenticate(self) -> bool:
        """Authenticate with API"""
        url = f"{CONFIG['REST_BASE_URL']}/api/Auth/loginKey"
        payload = {
            "userName": CONFIG['USERNAME'],
            "apiKey": CONFIG['API_KEY']
        }

        try:
            self.ui.add_event("AUTH", "Authenticating...")
            resp = self.session.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

            if data.get('errorCode') and data.get('errorCode') != 0:
                raise ValueError(f"Login Failed: {data.get('errorMessage', 'Unknown Error')}")

            token = data.get('token')
            if not token:
                raise ValueError("Login response missing token")

            self.session.headers.update({"Authorization": f"Bearer {token}"})
            self.ui.add_event("AUTH", "✓ Authentication successful")
            return True

        except Exception as e:
            self.ui.add_event("ERROR", f"Authentication failed: {e}")
            return False

    def select_accounts(self) -> Optional[List[str]]:
        """Let user select accounts"""
        self.ui.add_event("SYSTEM", "Opening account selection...")

        selected = select_account_interactive(self.session)

        if selected is None:
            return None

        # Convert to list if single account
        if isinstance(selected, str):
            return [selected]

        return selected

    def fetch_account_details(self, account_ids: List[str]) -> tuple[List[Dict], bool, Optional[str]]:
        """
        Fetch full account details and detect copy trade configuration

        Returns:
            (accounts, is_copy_trade_enabled, master_account_id)
        """
        url = f"{CONFIG['REST_BASE_URL']}/api/Account/search"
        payload = {"onlyActiveAccounts": True}

        try:
            resp = self.session.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

            accounts = []
            copy_trade_master = None
            is_copy_trade = False

            if 'accounts' in data:
                for acc in data['accounts']:
                    if acc.get('id') in account_ids:
                        accounts.append(acc)

                        # Check for copy trade indicators in account data
                        # Common field names that might indicate copy trade:
                        # - 'copyTradeEnabled', 'isCopyTrader', 'copyTradeMaster', etc.
                        if acc.get('copyTradeEnabled') or acc.get('isCopyTrader'):
                            is_copy_trade = True

                        if acc.get('copyTradeMaster') or acc.get('isMasterAccount'):
                            copy_trade_master = acc.get('id')

            # If copy trade detected but no master specified, use first account as master
            if is_copy_trade and not copy_trade_master and accounts:
                copy_trade_master = accounts[0].get('id')

            return accounts, is_copy_trade, copy_trade_master

        except Exception as e:
            self.ui.add_event("ERROR", f"Failed to fetch account details: {e}")
            return [], False, None

    def start_bots(self, accounts: List[Dict], is_copy_trade: bool = False, master_account_id: Optional[str] = None):
        """Start bot threads for all accounts"""
        if is_copy_trade and len(accounts) > 1:
            master_name = next((acc.get('name') for acc in accounts if acc.get('id') == master_account_id), accounts[0].get('name'))
            self.ui.add_event("SYSTEM", f"🔗 COPY TRADE DETECTED: Trading on {master_name}, monitoring others")

        for account in accounts:
            account_id = account.get('id')
            account_name = account.get('name', 'Unknown')

            # Determine if this account should trade
            # In copy trade mode: only master account trades, others monitor
            if is_copy_trade:
                is_trading = (account_id == master_account_id)
            else:
                is_trading = True  # All accounts trade independently

            # Register with tracker (include is_trading flag)
            self.tracker.add_account(account_id, account_name, is_trading=is_trading)

            # Start bot thread
            bot = AccountBotRunner(account_id, account_name, self.tracker, self.ui, is_trading_account=is_trading)
            bot.start()
            self.bot_threads.append(bot)

            mode = "🔹 TRADING" if is_trading else "👁️  MONITOR"
            self.ui.add_event("SYSTEM", f"{mode}: {account_name}")

    def update_dashboard_ui(self):
        """Update the dashboard UI with current data"""
        while self.running:
            try:
                summary = self.tracker.get_summary()

                # Update UI with account summary
                self.ui.update_account_info({
                    'account_id': f"{summary['total_accounts']} accounts",
                    'daily_pnl': summary['total_pnl'],
                    'total_trades': summary['total_trades'],
                    'win_rate': summary['win_rate']
                })

                # Update market context
                self.ui.update_market_context({
                    'session': self.get_current_session(),
                    'symbol': CONFIG.get('TARGET_SYMBOL', 'MES')
                })

                time.sleep(2)

            except Exception as e:
                logging.error(f"Dashboard UI update error: {e}")
                time.sleep(5)

    def get_current_session(self) -> str:
        """Determine current trading session"""
        now = datetime.now(ZoneInfo('America/New_York'))
        hour = now.hour

        if hour >= 18 or hour < 3:
            return 'ASIA'
        elif 3 <= hour < 8:
            return 'LONDON'
        elif 8 <= hour < 12:
            return 'NY_AM'
        elif 12 <= hour < 17:
            return 'NY_PM'
        else:
            return 'OFF_HOURS'

    def select_trading_account(self, accounts: List[Dict]) -> str:
        """Let user select which account should execute trades (when copy trade is enabled)"""
        from rich.console import Console
        from rich.prompt import IntPrompt
        from rich.table import Table
        from rich.panel import Panel

        console = Console()

        console.print("\n[bold yellow]Copy Trade Detected![/bold yellow]")
        console.print("Select which account should execute trades:")
        console.print("[dim](Other accounts will mirror the trades automatically)[/dim]\n")

        # Show accounts table
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Option", justify="center", style="bold yellow", width=8)
        table.add_column("Account Name", style="cyan", width=30)
        table.add_column("Account ID", style="dim", width=25)

        for idx, account in enumerate(accounts, start=1):
            table.add_row(
                f"[bold]{idx}[/bold]",
                str(account.get('name', 'Unknown')),
                str(account.get('id', 'N/A'))
            )

        console.print(table)
        console.print()

        # Get selection
        while True:
            try:
                choice = IntPrompt.ask(
                    f"[cyan]Select trading account (1-{len(accounts)})[/cyan]",
                    default=1
                )

                if 1 <= choice <= len(accounts):
                    selected = accounts[choice - 1]
                    console.print(f"\n[green]✓ Trading account: {selected.get('name')}[/green]")
                    console.print(f"[dim]All other accounts will mirror these trades[/dim]\n")
                    return selected.get('id')
                else:
                    console.print(f"[red]Invalid choice. Please enter 1-{len(accounts)}[/red]\n")

            except KeyboardInterrupt:
                return accounts[0].get('id')  # Default to first if cancelled
            except Exception:
                console.print("[red]Invalid input. Please try again.[/red]\n")

    def run(self):
        """Main dashboard loop"""
        print("=" * 60)
        print("JULIE DASHBOARD - Integrated Trading Bot + Monitor")
        print("Multi-Account Trading with Real-Time P&L")
        print("=" * 60)
        print()

        # Initialize UI
        refresh_target_symbol()
        self.ui.update_market_context({
            'symbol': CONFIG.get('TARGET_SYMBOL', 'MES'),
            'session': 'CONNECTING'
        })
        self.ui.start(refresh_rate=1.0)

        self.ui.add_event("SYSTEM", "Julie Dashboard starting...")

        # Authenticate
        if not self.authenticate():
            self.ui.add_event("ERROR", "Failed to authenticate - check credentials")
            time.sleep(5)
            self.ui.stop()
            return

        # Select accounts
        account_ids = self.select_accounts()
        if not account_ids:
            self.ui.add_event("ERROR", "No accounts selected")
            time.sleep(5)
            self.ui.stop()
            return

        # Fetch account details and detect copy trade
        accounts, is_copy_trade, auto_master_id = self.fetch_account_details(account_ids)
        if not accounts:
            self.ui.add_event("ERROR", "Failed to fetch account details")
            time.sleep(5)
            self.ui.stop()
            return

        self.ui.add_event("SYSTEM", f"✓ Loaded {len(accounts)} account(s)")

        # If copy trade and multiple accounts, let user select trading account
        master_account_id = auto_master_id
        if is_copy_trade and len(accounts) > 1:
            master_account_id = self.select_trading_account(accounts)
        elif len(accounts) > 1 and not is_copy_trade:
            # Multiple accounts, no copy trade detected
            self.ui.add_event("SYSTEM", "Multiple accounts - all will trade independently")

        # Start bot threads
        self.running = True
        self.start_bots(accounts, is_copy_trade=is_copy_trade, master_account_id=master_account_id)

        # Start UI update thread
        ui_thread = threading.Thread(target=self.update_dashboard_ui, daemon=True)
        ui_thread.start()

        self.ui.add_event("SYSTEM", "✓ Dashboard active - all bots running")

        # Main loop
        try:
            while True:
                time.sleep(1)

        except KeyboardInterrupt:
            self.ui.add_event("SYSTEM", "Shutting down dashboard...")
            self.running = False

            # Stop all bots
            for bot in self.bot_threads:
                bot.stop()

            self.ui.stop()
            print("\n\nDashboard stopped.")


if __name__ == "__main__":
    dashboard = JulieDashboard()
    dashboard.run()
