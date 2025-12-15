#!/usr/bin/env python3
"""
JULIE Trading Bot - Modern GUI Monitor
Multi-account support with real-time position tracking, signals, and market data
"""

import tkinter as tk
from tkinter import ttk, messagebox
import requests
import pytz
import threading
import time
from datetime import datetime
from pathlib import Path
from config import CONFIG
import re


class ModernGUI:
    """Modern trading dashboard GUI using tkinter"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("JULIE Trading Dashboard")
        self.root.geometry("1400x900")
        self.root.configure(bg='#1e1e1e')

        # Data stores
        self.accounts = []
        self.selected_accounts = []
        self.positions = {}
        self.signals = []
        self.events = []
        self.filter_status = {}
        self.market_data = {
            'session': 'CONNECTING',
            'price': 0.0,
            'symbol': CONFIG.get('TARGET_SYMBOL', 'MES'),
            'bias': 'NEUTRAL'
        }

        # Colors
        self.colors = {
            'bg_dark': '#1e1e1e',
            'bg_medium': '#2d2d2d',
            'bg_light': '#3d3d3d',
            'text': '#ffffff',
            'text_dim': '#888888',
            'green': '#00ff00',
            'red': '#ff4444',
            'yellow': '#ffaa00',
            'blue': '#4488ff',
            'cyan': '#00ffff'
        }

        # API client
        self.api_client = None

        self._setup_ui()

    def _setup_ui(self):
        """Setup the main UI layout"""
        # Top bar with account selection
        self._create_top_bar()

        # Main content area
        main_frame = tk.Frame(self.root, bg=self.colors['bg_dark'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Left panel (70%)
        left_panel = tk.Frame(main_frame, bg=self.colors['bg_dark'])
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        # Right panel (30%)
        right_panel = tk.Frame(main_frame, bg=self.colors['bg_dark'])
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(5, 0))

        # Left panel sections
        self._create_positions_panel(left_panel)
        self._create_signals_panel(left_panel)
        self._create_events_panel(left_panel)

        # Right panel sections
        self._create_market_panel(right_panel)
        self._create_filters_panel(right_panel)

    def _create_top_bar(self):
        """Create top bar with title and account selection"""
        top_bar = tk.Frame(self.root, bg=self.colors['bg_medium'], height=60)
        top_bar.pack(fill=tk.X, padx=10, pady=(10, 5))
        top_bar.pack_propagate(False)

        # Title
        title = tk.Label(
            top_bar,
            text="JULIE - MES Futures Trading Dashboard",
            font=('Helvetica', 16, 'bold'),
            bg=self.colors['bg_medium'],
            fg=self.colors['cyan']
        )
        title.pack(side=tk.LEFT, padx=20, pady=15)

        # Account selection frame
        account_frame = tk.Frame(top_bar, bg=self.colors['bg_medium'])
        account_frame.pack(side=tk.RIGHT, padx=20, pady=10)

        tk.Label(
            account_frame,
            text="Accounts:",
            font=('Helvetica', 10),
            bg=self.colors['bg_medium'],
            fg=self.colors['text']
        ).pack(side=tk.LEFT, padx=(0, 10))

        self.account_var = tk.StringVar(value="Loading...")
        self.account_dropdown = ttk.Combobox(
            account_frame,
            textvariable=self.account_var,
            state='readonly',
            width=30
        )
        self.account_dropdown.pack(side=tk.LEFT, padx=5)
        self.account_dropdown.bind('<<ComboboxSelected>>', self._on_account_selected)

        # All Accounts checkbox
        self.all_accounts_var = tk.BooleanVar(value=False)
        self.all_accounts_check = tk.Checkbutton(
            account_frame,
            text="All Accounts",
            variable=self.all_accounts_var,
            command=self._on_all_accounts_toggle,
            bg=self.colors['bg_medium'],
            fg=self.colors['text'],
            selectcolor=self.colors['bg_light'],
            activebackground=self.colors['bg_medium'],
            activeforeground=self.colors['green']
        )
        self.all_accounts_check.pack(side=tk.LEFT, padx=10)

    def _create_positions_panel(self, parent):
        """Create positions display panel"""
        panel = self._create_panel(parent, "Current Positions", height=250)

        # Treeview for positions
        columns = ('Account', 'Side', 'Entry', 'Current', 'P&L', 'TP', 'SL', 'Strategy')
        self.positions_tree = ttk.Treeview(panel, columns=columns, show='headings', height=8)

        # Column headings
        for col in columns:
            self.positions_tree.heading(col, text=col)
            self.positions_tree.column(col, width=120, anchor='center')

        # Scrollbar
        scrollbar = ttk.Scrollbar(panel, orient=tk.VERTICAL, command=self.positions_tree.yview)
        self.positions_tree.configure(yscrollcommand=scrollbar.set)

        self.positions_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)

        # Configure tags for colors
        self.positions_tree.tag_configure('LONG', foreground=self.colors['green'])
        self.positions_tree.tag_configure('SHORT', foreground=self.colors['red'])
        self.positions_tree.tag_configure('FLAT', foreground=self.colors['text_dim'])

    def _create_signals_panel(self, parent):
        """Create signals display panel"""
        panel = self._create_panel(parent, "Recent Signals", height=200)

        columns = ('Time', 'Account', 'Strategy', 'Side', 'TP', 'SL', 'Status')
        self.signals_tree = ttk.Treeview(panel, columns=columns, show='headings', height=6)

        widths = [80, 100, 150, 60, 60, 60, 100]
        for col, width in zip(columns, widths):
            self.signals_tree.heading(col, text=col)
            self.signals_tree.column(col, width=width, anchor='center')

        scrollbar = ttk.Scrollbar(panel, orient=tk.VERTICAL, command=self.signals_tree.yview)
        self.signals_tree.configure(yscrollcommand=scrollbar.set)

        self.signals_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)

        # Tags
        self.signals_tree.tag_configure('EXECUTED', foreground=self.colors['green'])
        self.signals_tree.tag_configure('BLOCKED', foreground=self.colors['red'])
        self.signals_tree.tag_configure('PENDING', foreground=self.colors['yellow'])

    def _create_events_panel(self, parent):
        """Create events log panel"""
        panel = self._create_panel(parent, "Event Log")

        # Text widget for events
        self.events_text = tk.Text(
            panel,
            bg=self.colors['bg_light'],
            fg=self.colors['text'],
            font=('Courier', 9),
            height=12,
            wrap=tk.WORD
        )

        scrollbar = ttk.Scrollbar(panel, orient=tk.VERTICAL, command=self.events_text.yview)
        self.events_text.configure(yscrollcommand=scrollbar.set)

        self.events_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)

        # Tags for colored events
        self.events_text.tag_configure('TRADE', foreground=self.colors['green'])
        self.events_text.tag_configure('ERROR', foreground=self.colors['red'])
        self.events_text.tag_configure('SIGNAL', foreground=self.colors['yellow'])
        self.events_text.tag_configure('SYSTEM', foreground=self.colors['blue'])
        self.events_text.tag_configure('FILTER', foreground=self.colors['cyan'])

    def _create_market_panel(self, parent):
        """Create market context panel"""
        panel = self._create_panel(parent, "Market Context", height=200)

        content = tk.Frame(panel, bg=self.colors['bg_light'])
        content.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.market_labels = {}

        fields = [
            ('Session:', 'session'),
            ('Symbol:', 'symbol'),
            ('Price:', 'price'),
            ('Bias:', 'bias'),
            ('Volatility:', 'volatility')
        ]

        for i, (label_text, key) in enumerate(fields):
            row = tk.Frame(content, bg=self.colors['bg_light'])
            row.pack(fill=tk.X, pady=5)

            tk.Label(
                row,
                text=label_text,
                font=('Helvetica', 10),
                bg=self.colors['bg_light'],
                fg=self.colors['text_dim'],
                width=12,
                anchor='w'
            ).pack(side=tk.LEFT)

            value_label = tk.Label(
                row,
                text='--',
                font=('Helvetica', 10, 'bold'),
                bg=self.colors['bg_light'],
                fg=self.colors['text'],
                anchor='w'
            )
            value_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

            self.market_labels[key] = value_label

    def _create_filters_panel(self, parent):
        """Create filters status panel"""
        panel = self._create_panel(parent, "Filter Status")

        content = tk.Frame(panel, bg=self.colors['bg_light'])
        content.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.filter_labels = {}

        filters = ['Rejection', 'HTF FVG', 'Chop', 'Extension', 'Structure', 'Bank Level']

        for filter_name in filters:
            row = tk.Frame(content, bg=self.colors['bg_light'])
            row.pack(fill=tk.X, pady=5)

            tk.Label(
                row,
                text=filter_name + ':',
                font=('Helvetica', 10),
                bg=self.colors['bg_light'],
                fg=self.colors['text'],
                width=15,
                anchor='w'
            ).pack(side=tk.LEFT)

            status_label = tk.Label(
                row,
                text='IDLE',
                font=('Helvetica', 10, 'bold'),
                bg=self.colors['bg_light'],
                fg=self.colors['text_dim'],
                width=10,
                anchor='center'
            )
            status_label.pack(side=tk.LEFT)

            self.filter_labels[filter_name] = status_label

    def _create_panel(self, parent, title, height=None):
        """Create a styled panel with title"""
        container = tk.Frame(parent, bg=self.colors['bg_dark'])
        container.pack(fill=tk.BOTH, expand=(height is None), pady=5)

        if height:
            container.configure(height=height)
            container.pack_propagate(False)

        # Title bar
        title_bar = tk.Frame(container, bg=self.colors['bg_medium'], height=30)
        title_bar.pack(fill=tk.X)
        title_bar.pack_propagate(False)

        tk.Label(
            title_bar,
            text=title,
            font=('Helvetica', 11, 'bold'),
            bg=self.colors['bg_medium'],
            fg=self.colors['cyan']
        ).pack(side=tk.LEFT, padx=10, pady=5)

        # Content area
        content = tk.Frame(container, bg=self.colors['bg_light'])
        content.pack(fill=tk.BOTH, expand=True)

        return content

    def _on_account_selected(self, event=None):
        """Handle account selection"""
        if self.all_accounts_var.get():
            return

        selection = self.account_var.get()
        if selection and selection != "Loading...":
            # Extract account ID from selection
            account_id = selection.split('(')[-1].strip(')')
            self.selected_accounts = [account_id]
            self.add_event('SYSTEM', f'Selected account: {account_id}')

    def _on_all_accounts_toggle(self):
        """Handle all accounts checkbox toggle"""
        if self.all_accounts_var.get():
            self.selected_accounts = [acc['id'] for acc in self.accounts]
            self.account_dropdown.configure(state='disabled')
            self.add_event('SYSTEM', f'Monitoring all {len(self.accounts)} accounts')
        else:
            self.account_dropdown.configure(state='readonly')
            self._on_account_selected()

    def set_accounts(self, accounts):
        """Set available accounts"""
        self.accounts = accounts
        if accounts:
            choices = [f"{acc.get('name', 'Unknown')} ({acc['id']})" for acc in accounts]
            self.account_dropdown['values'] = choices
            self.account_dropdown.current(0)
            self._on_account_selected()
        else:
            self.account_var.set("No accounts found")

    def update_position(self, account_id, position_data):
        """Update position for an account"""
        self.positions[account_id] = position_data
        self._refresh_positions()

    def _refresh_positions(self):
        """Refresh positions display"""
        # Clear existing
        for item in self.positions_tree.get_children():
            self.positions_tree.delete(item)

        # Add current positions
        for account_id, pos in self.positions.items():
            if pos.get('active'):
                side = pos.get('side', 'FLAT')
                entry = pos.get('entry_price', 0.0)
                current = pos.get('current_price', entry)

                # Calculate P&L
                if side == 'LONG':
                    pnl = (current - entry) * 5
                elif side == 'SHORT':
                    pnl = (entry - current) * 5
                else:
                    pnl = 0.0

                pnl_str = f"${pnl:+.2f}"
                tag = side

                self.positions_tree.insert('', 'end', values=(
                    account_id[:8],
                    side,
                    f"{entry:.2f}",
                    f"{current:.2f}",
                    pnl_str,
                    f"{pos.get('tp_price', 0.0):.2f}",
                    f"{pos.get('sl_price', 0.0):.2f}",
                    pos.get('strategy', 'Unknown')
                ), tags=(tag,))
            else:
                # Flat position
                self.positions_tree.insert('', 'end', values=(
                    account_id[:8],
                    'FLAT',
                    '--',
                    f"{pos.get('current_price', 0.0):.2f}",
                    '$0.00',
                    '--',
                    '--',
                    '--'
                ), tags=('FLAT',))

    def add_signal(self, account_id, signal_data):
        """Add a signal to the display"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        signal_data['time'] = timestamp
        signal_data['account'] = account_id
        self.signals.insert(0, signal_data)

        # Keep only last 20 signals
        if len(self.signals) > 20:
            self.signals = self.signals[:20]

        self._refresh_signals()

    def _refresh_signals(self):
        """Refresh signals display"""
        for item in self.signals_tree.get_children():
            self.signals_tree.delete(item)

        for sig in self.signals[:10]:
            status = sig.get('status', 'PENDING')
            self.signals_tree.insert('', 'end', values=(
                sig.get('time', ''),
                sig.get('account', '')[:8],
                sig.get('strategy', '')[:15],
                sig.get('side', ''),
                f"{sig.get('tp_dist', 0.0):.1f}",
                f"{sig.get('sl_dist', 0.0):.1f}",
                status
            ), tags=(status,))

    def add_event(self, event_type, message):
        """Add event to the log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.events.insert(0, f"[{timestamp}] {event_type}: {message}")

        if len(self.events) > 100:
            self.events = self.events[:100]

        # Update text widget
        self.events_text.insert('1.0', f"[{timestamp}] {message}\n", event_type)

        # Limit text widget size
        lines = int(self.events_text.index('end-1c').split('.')[0])
        if lines > 100:
            self.events_text.delete(f'{lines-100}.0', tk.END)

    def update_market_context(self, data):
        """Update market context display"""
        self.market_data.update(data)

        # Update labels
        for key, label in self.market_labels.items():
            value = self.market_data.get(key, '--')

            # Format price
            if key == 'price' and isinstance(value, (int, float)):
                value = f"{value:.2f}"

            # Color coding
            color = self.colors['text']
            if key == 'session':
                color = self.colors['yellow']
            elif key == 'bias':
                if value == 'LONG':
                    color = self.colors['green']
                elif value == 'SHORT':
                    color = self.colors['red']

            label.configure(text=str(value), fg=color)

    def update_filter_status(self, filter_name, passed, reason=""):
        """Update filter status"""
        if filter_name in self.filter_labels:
            label = self.filter_labels[filter_name]

            if passed:
                label.configure(text='PASS ✓', fg=self.colors['green'])
            else:
                label.configure(text='BLOCK ✗', fg=self.colors['red'])

    def run(self):
        """Start the GUI"""
        self.root.mainloop()

    def stop(self):
        """Stop the GUI"""
        self.root.quit()


def get_current_session(dt):
    """Determine current trading session"""
    hour = dt.hour
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


class APIMonitor:
    """Monitor TopstepX API for multiple accounts"""

    def __init__(self, gui):
        self.gui = gui
        self.session = requests.Session()
        self.token = None
        self.base_url = CONFIG['REST_BASE_URL']
        self.et = pytz.timezone('US/Eastern')
        self.accounts = []
        self.contracts = {}

    def login(self):
        """Authenticate"""
        url = f"{self.base_url}/api/Auth/loginKey"
        payload = {
            "userName": CONFIG['USERNAME'],
            "apiKey": CONFIG['API_KEY']
        }

        try:
            self.gui.add_event("API", "Authenticating...")
            resp = self.session.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

            if data.get('errorCode') and data.get('errorCode') != 0:
                raise ValueError(f"Login Failed: {data.get('errorMessage')}")

            self.token = data.get('token')
            if not self.token:
                raise ValueError("Login response missing token")

            self.session.headers.update({"Authorization": f"Bearer {self.token}"})
            self.gui.add_event("API", "✓ Authentication successful")
            return True
        except Exception as e:
            self.gui.add_event("ERROR", f"Login failed: {e}")
            return False

    def fetch_accounts(self):
        """Get all active accounts"""
        url = f"{self.base_url}/api/Account/search"
        payload = {"onlyActiveAccounts": True}

        try:
            resp = self.session.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

            if 'accounts' in data and len(data['accounts']) > 0:
                self.accounts = data['accounts']
                self.gui.set_accounts(self.accounts)
                self.gui.add_event("API", f"✓ Found {len(self.accounts)} accounts")
                return True
            else:
                self.gui.add_event("ERROR", "No active accounts found")
                return False
        except Exception as e:
            self.gui.add_event("ERROR", f"Failed to fetch accounts: {e}")
            return False

    def fetch_contract_for_account(self, account_id):
        """Get contract ID for symbol"""
        if account_id in self.contracts:
            return self.contracts[account_id]

        from config import refresh_target_symbol
        refresh_target_symbol()

        url = f"{self.base_url}/api/Contract/search"
        payload = {
            "live": False,
            "searchText": CONFIG.get('TARGET_SYMBOL', 'MESZ25')
        }

        try:
            resp = self.session.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

            if 'contracts' in data and len(data['contracts']) > 0:
                target = CONFIG.get('TARGET_SYMBOL', 'MESZ25')
                for contract in data['contracts']:
                    contract_id = contract.get('id', '')
                    if f".{target}." in contract_id or contract_id.endswith(f".{target}"):
                        self.contracts[account_id] = contract_id
                        return contract_id

                self.contracts[account_id] = data['contracts'][0].get('id')
                return self.contracts[account_id]
        except:
            pass

        return None

    def fetch_position(self, account_id):
        """Fetch position for account"""
        url = f"{self.base_url}/api/Position/search"
        payload = {"accountId": account_id}

        try:
            resp = self.session.post(url, json=payload, timeout=5)

            if resp.status_code == 404:
                return {'side': None, 'size': 0, 'avg_price': 0.0}

            if resp.status_code == 200:
                data = resp.json()
                positions = data.get('positions', data) if isinstance(data, dict) else data

                contract_id = self.contracts.get(account_id)
                for pos in positions:
                    if pos.get('contractId') == contract_id:
                        size = pos.get('size', 0)
                        avg_price = pos.get('averagePrice', 0.0)
                        if size > 0:
                            return {'side': 'LONG', 'size': size, 'avg_price': avg_price}
                        elif size < 0:
                            return {'side': 'SHORT', 'size': abs(size), 'avg_price': avg_price}

                return {'side': None, 'size': 0, 'avg_price': 0.0}
        except:
            pass

        return None

    def fetch_market_data(self, account_id):
        """Fetch latest price"""
        contract_id = self.contracts.get(account_id)
        if not contract_id:
            return None

        from datetime import timedelta, timezone

        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(minutes=10)

        url = f"{self.base_url}/api/History/retrieveBars"
        payload = {
            "accountId": account_id,
            "contractId": contract_id,
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
                    return float(data['bars'][0].get('c', 0.0))
        except:
            pass

        return None


def main():
    """Main application"""
    # Create GUI
    gui = ModernGUI()

    # Create API monitor
    api_monitor = APIMonitor(gui)

    def initialize():
        """Initialize API connection"""
        if not api_monitor.login():
            messagebox.showerror("Error", "Failed to authenticate. Check your config.py credentials.")
            gui.stop()
            return

        if not api_monitor.fetch_accounts():
            messagebox.showerror("Error", "Failed to fetch accounts.")
            gui.stop()
            return

        # Fetch contracts for all accounts
        for acc in api_monitor.accounts:
            api_monitor.fetch_contract_for_account(acc['id'])

        # Start monitoring thread
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()

    def monitor_loop():
        """Main monitoring loop"""
        last_check = 0

        while True:
            try:
                now = time.time()

                if now - last_check > 2.0:
                    # Get current price and session
                    current_time = datetime.now(pytz.timezone('US/Eastern'))
                    session = get_current_session(current_time)

                    # Monitor selected accounts
                    for account_id in gui.selected_accounts:
                        # Fetch position
                        position = api_monitor.fetch_position(account_id)
                        if position:
                            price = api_monitor.fetch_market_data(account_id)
                            if price:
                                # Update position display
                                if position['side'] is not None:
                                    tp_dist = 6.0
                                    sl_dist = 4.0

                                    if position['side'] == 'LONG':
                                        tp_price = position['avg_price'] + tp_dist
                                        sl_price = position['avg_price'] - sl_dist
                                    else:
                                        tp_price = position['avg_price'] - tp_dist
                                        sl_price = position['avg_price'] + sl_dist

                                    gui.update_position(account_id, {
                                        'active': True,
                                        'side': position['side'],
                                        'entry_price': position['avg_price'],
                                        'current_price': price,
                                        'tp_price': tp_price,
                                        'sl_price': sl_price,
                                        'strategy': 'Unknown'
                                    })
                                else:
                                    gui.update_position(account_id, {
                                        'active': False,
                                        'current_price': price
                                    })

                                # Update market context
                                gui.update_market_context({
                                    'price': price,
                                    'session': session
                                })

                    last_check = now

                time.sleep(0.5)
            except Exception as e:
                gui.add_event("ERROR", f"Monitor error: {e}")
                time.sleep(5)

    # Initialize in background
    init_thread = threading.Thread(target=initialize, daemon=True)
    init_thread.start()

    # Run GUI
    gui.run()


if __name__ == "__main__":
    main()
