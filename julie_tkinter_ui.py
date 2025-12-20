#!/usr/bin/env python3
"""
JULIE Tkinter UI - Professional Trading Dashboard
Real integration with julie001.py bot via log monitoring and API
"""

import tkinter as tk
from tkinter import ttk, font as tkfont
import threading
import time
import requests
from datetime import datetime
from pathlib import Path
import re

# Import from existing monitoring infrastructure
try:
    from config import CONFIG
    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False
    CONFIG = {}

class JulieUI:
    def __init__(self, root):
        self.root = root
        self.root.title("JULIE")
        self.root.geometry("1600x900")
        self.root.configure(bg='#000000')

        # Color scheme matching the screenshots
        self.colors = {
            'bg_dark': '#000000',
            'bg_gradient_start': '#0a1a0e',
            'bg_gradient_end': '#0a0e1a',
            'panel_bg': '#1a1f2e',
            'panel_border': '#2a4a3a',
            'input_bg': '#2d3340',
            'input_border': '#3a4a3a',
            'text_white': '#ffffff',
            'text_gray': '#9ca3af',
            'text_dim': '#6b7280',
            'green': '#22c55e',
            'green_light': '#4ade80',
            'red': '#ef4444',
            'yellow': '#fbbf24',
            'blue': '#3b82f6',
        }

        # Session and data
        self.session = None
        self.token = None
        self.accounts = []
        self.selected_account = None
        self.current_price = 5880.25
        self.logged_in = False
        self.monitoring_active = False

        # Bot integration
        self.log_file = Path("topstep_live_bot.log")
        self.log_position = 0
        self.contract_id = None

        # Show login page
        self.show_login_page()

    def show_login_page(self):
        """Create login page matching the screenshot"""
        # Clear root
        for widget in self.root.winfo_children():
            widget.destroy()

        # Create gradient background effect
        canvas = tk.Canvas(self.root, bg='#000000', highlightthickness=0)
        canvas.pack(fill='both', expand=True)

        # Create gradient effect with rectangles
        canvas.create_rectangle(0, 0, 1600, 900, fill='#0a0e1a', outline='')
        canvas.create_oval(-200, -200, 800, 800, fill='#0a1a0e', outline='')
        canvas.create_oval(1000, 200, 1800, 1100, fill='#0a1a0e', outline='')

        # Create login panel
        panel_frame = tk.Frame(canvas, bg=self.colors['panel_bg'],
                              highlightbackground=self.colors['panel_border'],
                              highlightthickness=2)
        canvas.create_window(800, 450, window=panel_frame, width=500, height=550)

        # JULIE Title
        title_font = tkfont.Font(family="Helvetica", size=72, weight="bold")
        title = tk.Label(panel_frame, text="JULIE",
                        font=title_font,
                        fg=self.colors['text_white'],
                        bg=self.colors['panel_bg'])
        title.pack(pady=(50, 80))

        # Account Number Label
        account_label = tk.Label(panel_frame, text="ACCOUNT NUMBER",
                                font=("Helvetica", 11, "bold"),
                                fg=self.colors['text_gray'],
                                bg=self.colors['panel_bg'])
        account_label.pack(pady=(0, 15))

        # Fetch accounts first
        self.fetch_accounts_for_login()

        # Account Dropdown
        self.setup_dropdown_style()
        self.account_var = tk.StringVar()

        if self.accounts:
            account_names = [acc.get('name', acc.get('id', 'Unknown')) for acc in self.accounts]
            self.account_var.set(account_names[0] if account_names else "No accounts")
        else:
            account_names = ["Loading..."]
            self.account_var.set("Loading...")

        dropdown = ttk.Combobox(panel_frame,
                               textvariable=self.account_var,
                               values=account_names,
                               state='readonly',
                               font=("Helvetica", 14),
                               style='Login.TCombobox',
                               width=32)
        dropdown.pack(pady=(0, 50), padx=60)

        # Login Button
        login_btn = tk.Button(panel_frame,
                             text="🔒 LOGIN",
                             font=("Helvetica", 16, "bold"),
                             bg=self.colors['green'],
                             fg='white',
                             activebackground=self.colors['green_light'],
                             activeforeground='white',
                             relief='flat',
                             cursor='hand2',
                             command=self.handle_login)
        login_btn.pack(pady=(0, 25), padx=60, fill='x', ipady=12)

        # Forgot Password Link
        forgot = tk.Label(panel_frame, text="FORGOT PASSWORD?",
                         font=("Helvetica", 10, "underline"),
                         fg=self.colors['text_gray'],
                         bg=self.colors['panel_bg'],
                         cursor='hand2')
        forgot.pack(pady=(15, 0))

    def setup_dropdown_style(self):
        """Setup custom combobox style"""
        style = ttk.Style()
        style.theme_use('clam')

        style.configure('Login.TCombobox',
                       fieldbackground=self.colors['input_bg'],
                       background=self.colors['input_bg'],
                       foreground=self.colors['text_white'],
                       arrowcolor=self.colors['text_white'],
                       bordercolor=self.colors['input_border'],
                       lightcolor=self.colors['input_bg'],
                       darkcolor=self.colors['input_bg'],
                       selectbackground=self.colors['green'],
                       selectforeground='white')

        style.map('Login.TCombobox',
                 fieldbackground=[('readonly', self.colors['input_bg'])],
                 selectbackground=[('readonly', self.colors['input_bg'])])

    def fetch_accounts_for_login(self):
        """Fetch accounts from API for login dropdown"""
        if not HAS_CONFIG or not CONFIG.get('USERNAME') or not CONFIG.get('API_KEY'):
            self.accounts = [{'name': 'ACCT-001', 'id': 'ACCT-001'}]
            return

        try:
            # Create session and login
            self.session = requests.Session()
            url = f"{CONFIG['REST_BASE_URL']}/api/Auth/loginKey"
            payload = {
                "userName": CONFIG['USERNAME'],
                "apiKey": CONFIG['API_KEY']
            }

            resp = self.session.post(url, json=payload, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                self.token = data.get('token')
                if self.token:
                    self.session.headers.update({"Authorization": f"Bearer {self.token}"})

                    # Fetch accounts
                    acc_url = f"{CONFIG['REST_BASE_URL']}/api/Account/search"
                    acc_resp = self.session.post(acc_url, json={"onlyActiveAccounts": True}, timeout=10)
                    if acc_resp.status_code == 200:
                        acc_data = acc_resp.json()
                        self.accounts = acc_data.get('accounts', [])
                        return
        except Exception as e:
            print(f"Error fetching accounts: {e}")

        # Fallback
        self.accounts = [{'name': 'ACCT-001', 'id': 'ACCT-001'}]

    def handle_login(self):
        """Handle login button click"""
        selected_name = self.account_var.get()

        # Find the account
        for acc in self.accounts:
            if acc.get('name') == selected_name or acc.get('id') == selected_name:
                self.selected_account = acc
                break

        if not self.selected_account:
            self.selected_account = {'name': selected_name, 'id': selected_name}

        self.logged_in = True
        self.show_dashboard()

    def show_dashboard(self):
        """Create main dashboard matching the screenshot"""
        # Clear root
        for widget in self.root.winfo_children():
            widget.destroy()

        # Main container
        main = tk.Frame(self.root, bg=self.colors['bg_dark'])
        main.pack(fill='both', expand=True)

        # Header
        header = tk.Frame(main, bg=self.colors['bg_dark'], height=60)
        header.pack(fill='x', padx=25, pady=(15, 10))
        header.pack_propagate(False)

        title = tk.Label(header, text="JULIE",
                        font=("Helvetica", 32, "bold"),
                        fg=self.colors['text_white'],
                        bg=self.colors['bg_dark'])
        title.pack(side='left')

        # Main content area
        content = tk.Frame(main, bg=self.colors['bg_dark'])
        content.pack(fill='both', expand=True, padx=25, pady=10)

        # Left panel (70%)
        left_panel = tk.Frame(content, bg=self.colors['bg_dark'])
        left_panel.pack(side='left', fill='both', expand=True, padx=(0, 15))

        # Right panel (30%) - Event Log
        right_panel = tk.Frame(content, bg=self.colors['bg_dark'], width=450)
        right_panel.pack(side='right', fill='both', padx=(15, 0))
        right_panel.pack_propagate(False)

        # Build left panel sections
        self.create_market_section(left_panel)

        # Bottom row: Strategy List, Positions, Filters
        bottom_row = tk.Frame(left_panel, bg=self.colors['bg_dark'])
        bottom_row.pack(fill='both', expand=True, pady=(10, 0))

        # Strategy list (left column)
        strategy_frame = tk.Frame(bottom_row, bg=self.colors['bg_dark'])
        strategy_frame.pack(side='left', fill='both', expand=True, padx=(0, 5))

        # Right column (Positions + Filters)
        right_col = tk.Frame(bottom_row, bg=self.colors['bg_dark'])
        right_col.pack(side='right', fill='both', expand=True, padx=(5, 0))

        self.create_strategy_list(strategy_frame)
        self.create_positions_section(right_col)
        self.create_filters_section(right_col)

        # Right panel - Event Log
        self.create_event_log(right_panel)

        # Fetch contract ID
        self.fetch_contract_id()

        # Start monitoring
        self.start_monitoring()

    def create_market_section(self, parent):
        """Create market context section"""
        section = tk.Frame(parent, bg=self.colors['panel_bg'],
                          highlightbackground=self.colors['panel_border'],
                          highlightthickness=1)
        section.pack(fill='x', pady=(0, 10))

        # Header
        header = tk.Label(section, text="SIGNAL MONITOR & MARKET CONTEXT",
                         font=("Helvetica", 10, "bold"),
                         fg=self.colors['text_gray'],
                         bg=self.colors['panel_bg'],
                         anchor='w')
        header.pack(fill='x', padx=20, pady=(15, 30))

        # Market display
        self.market_symbol_label = tk.Label(section, text="MES FUTURES",
                                           font=("Helvetica", 56, "bold"),
                                           fg=self.colors['text_white'],
                                           bg=self.colors['panel_bg'])
        self.market_symbol_label.pack(pady=(0, 10))

        # Price row
        price_row = tk.Frame(section, bg=self.colors['panel_bg'])
        price_row.pack(pady=(0, 40))

        self.price_label = tk.Label(price_row, text="5880.25",
                                    font=("Helvetica", 64, "bold"),
                                    fg=self.colors['text_white'],
                                    bg=self.colors['panel_bg'])
        self.price_label.pack(side='left', padx=15)

        self.change_label = tk.Label(price_row, text="▲ 20.73%",
                                     font=("Helvetica", 64, "bold"),
                                     fg=self.colors['green_light'],
                                     bg=self.colors['panel_bg'])
        self.change_label.pack(side='left', padx=15)

    def create_strategy_list(self, parent):
        """Create strategy list section"""
        section = tk.Frame(parent, bg=self.colors['panel_bg'],
                          highlightbackground=self.colors['panel_border'],
                          highlightthickness=1)
        section.pack(fill='both', expand=True)

        # Header
        header = tk.Label(section, text="STRATEGY LIST",
                         font=("Helvetica", 10, "bold"),
                         fg=self.colors['text_gray'],
                         bg=self.colors['panel_bg'],
                         anchor='w')
        header.pack(fill='x', padx=20, pady=(15, 10))

        # Scrollable container
        canvas = tk.Canvas(section, bg=self.colors['panel_bg'],
                          highlightthickness=0)
        scrollbar = tk.Scrollbar(section, orient='vertical', command=canvas.yview)
        scrollable = tk.Frame(canvas, bg=self.colors['panel_bg'])

        scrollable.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable, anchor='nw')
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side='left', fill='both', expand=True, padx=20, pady=(0, 15))
        scrollbar.pack(side='right', fill='y')

        # Strategy entries - All 9 from JULIE
        strategies = [
            "Regime Adaptive",
            "Intraday Dip",
            "Confluence",
            "ICT Model",
            "ORB Strategy",
            "ML Physics",
            "Dynamic Engine 1",
            "Dynamic Engine 2",
            "SMT Divergence"
        ]

        self.strategy_labels = {}
        for name in strategies:
            entry = tk.Frame(scrollable, bg=self.colors['input_bg'],
                           highlightbackground=self.colors['input_border'],
                           highlightthickness=1)
            entry.pack(fill='x', pady=3)

            name_label = tk.Label(entry, text=name,
                                 font=("Helvetica", 12),
                                 fg=self.colors['text_white'],
                                 bg=self.colors['input_bg'],
                                 anchor='w')
            name_label.pack(side='left', padx=15, pady=10)

            status_label = tk.Label(entry, text="WAITING",
                                   font=("Helvetica", 10),
                                   fg=self.colors['text_gray'],
                                   bg=self.colors['input_bg'],
                                   anchor='e')
            status_label.pack(side='right', padx=15)

            self.strategy_labels[name] = status_label

    def create_positions_section(self, parent):
        """Create active positions section"""
        section = tk.Frame(parent, bg=self.colors['panel_bg'],
                          highlightbackground=self.colors['panel_border'],
                          highlightthickness=1)
        section.pack(fill='x', pady=(0, 10))

        # Header
        header = tk.Label(section, text="ACTIVE POSITIONS",
                         font=("Helvetica", 10, "bold"),
                         fg=self.colors['text_gray'],
                         bg=self.colors['panel_bg'],
                         anchor='w')
        header.pack(fill='x', padx=20, pady=(15, 10))

        # Position container
        self.position_container = tk.Frame(section, bg=self.colors['panel_bg'])
        self.position_container.pack(fill='x', padx=20, pady=(0, 15))

        # Initially empty
        self.no_position_label = tk.Label(self.position_container,
                                          text="No active positions",
                                          font=("Helvetica", 10),
                                          fg=self.colors['text_dim'],
                                          bg=self.colors['panel_bg'])
        self.no_position_label.pack(pady=10)

    def create_filters_section(self, parent):
        """Create filter status dashboard - ALL 12 filters"""
        section = tk.Frame(parent, bg=self.colors['panel_bg'],
                          highlightbackground=self.colors['panel_border'],
                          highlightthickness=1)
        section.pack(fill='both', expand=True)

        # Header
        header = tk.Label(section, text="FILTER STATUS DASHBOARD",
                         font=("Helvetica", 10, "bold"),
                         fg=self.colors['text_gray'],
                         bg=self.colors['panel_bg'],
                         anchor='w')
        header.pack(fill='x', padx=20, pady=(15, 10))

        # Grid container
        grid = tk.Frame(section, bg=self.colors['panel_bg'])
        grid.pack(fill='both', expand=True, padx=20, pady=(0, 15))

        # Configure grid (4 columns x 3 rows for 12 filters)
        for i in range(4):
            grid.columnconfigure(i, weight=1, uniform='col')
        for i in range(3):
            grid.rowconfigure(i, weight=1, uniform='row')

        # ALL 12 filter entries (10 filters + 2 blockers)
        filters = [
            ("Rejection", "SAFE", "❌"),
            ("Chop", "SAFE", "🌊"),
            ("Extension", "SAFE", "📈"),
            ("Volatility", "SAFE", "🎯"),
            ("Trend", "SAFE", "📉"),
            ("Impulse", "SAFE", "⚡"),
            ("HTF FVG", "SAFE", "🎚️"),
            ("Bank Level", "SAFE", "💰"),
            ("Memory S/R", "SAFE", "🧠"),
            ("News", "SAFE", "📰"),
            ("Structure", "SAFE", "🏗️"),
            ("Loss Block", "SAFE", "🛡️"),
        ]

        self.filter_labels = {}
        row, col = 0, 0
        for name, status, icon in filters:
            self.create_filter_box(grid, row, col, name, status, icon)
            col += 1
            if col >= 4:
                col = 0
                row += 1

    def create_filter_box(self, parent, row, col, name, status, icon):
        """Create individual filter indicator"""
        box = tk.Frame(parent, bg=self.colors['input_bg'],
                      highlightbackground=self.colors['input_border'],
                      highlightthickness=1)
        box.grid(row=row, column=col, padx=4, pady=4, sticky='nsew')

        icon_label = tk.Label(box, text=icon,
                             font=("Helvetica", 24),
                             bg=self.colors['input_bg'])
        icon_label.pack(pady=(6, 2))

        name_label = tk.Label(box, text=name,
                             font=("Helvetica", 8, "bold"),
                             fg=self.colors['text_white'],
                             bg=self.colors['input_bg'])
        name_label.pack(pady=2)

        status_color = self.colors['green'] if status in ["PASS", "SAFE"] else self.colors['red']
        status_label = tk.Label(box, text=f"[{status}]",
                               font=("Helvetica", 7, "bold"),
                               fg=status_color,
                               bg=self.colors['input_bg'])
        status_label.pack(pady=(2, 6))

        self.filter_labels[name] = status_label

    def create_event_log(self, parent):
        """Create live event log section"""
        section = tk.Frame(parent, bg=self.colors['panel_bg'],
                          highlightbackground=self.colors['panel_border'],
                          highlightthickness=1)
        section.pack(fill='both', expand=True)

        # Header
        header = tk.Label(section, text="LIVE EVENT LOG",
                         font=("Helvetica", 10, "bold"),
                         fg=self.colors['text_gray'],
                         bg=self.colors['panel_bg'],
                         anchor='w')
        header.pack(fill='x', padx=20, pady=(15, 10))

        # Log text widget
        log_frame = tk.Frame(section, bg=self.colors['panel_bg'])
        log_frame.pack(fill='both', expand=True, padx=20, pady=(0, 15))

        scrollbar = tk.Scrollbar(log_frame)
        scrollbar.pack(side='right', fill='y')

        self.log_text = tk.Text(log_frame,
                               bg=self.colors['input_bg'],
                               fg=self.colors['text_gray'],
                               font=("Courier", 9),
                               wrap='word',
                               yscrollcommand=scrollbar.set,
                               state='disabled',
                               relief='flat')
        self.log_text.pack(fill='both', expand=True)
        scrollbar.config(command=self.log_text.yview)

        self.add_log("Waiting for bot activity...")

    def add_log(self, message):
        """Add entry to event log"""
        def update():
            self.log_text.config(state='normal')
            self.log_text.insert('end', message + '\n')
            self.log_text.see('end')
            self.log_text.config(state='disabled')

        if threading.current_thread() != threading.main_thread():
            self.root.after(0, update)
        else:
            update()

    def fetch_contract_id(self):
        """Fetch contract ID for position monitoring"""
        if not self.session or not self.token:
            return

        try:
            from config import refresh_target_symbol
            refresh_target_symbol()

            url = f"{CONFIG['REST_BASE_URL']}/api/Contract/search"
            payload = {
                "live": False,
                "searchText": CONFIG.get('TARGET_SYMBOL', 'MES.Z25')
            }

            resp = self.session.post(url, json=payload, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if 'contracts' in data and data['contracts']:
                    self.contract_id = data['contracts'][0].get('id')
                    self.add_log(f"Contract: {CONFIG.get('TARGET_SYMBOL')}")
        except Exception as e:
            self.add_log(f"Error fetching contract: {e}")

    def start_monitoring(self):
        """Start real-time monitoring of bot log and API"""
        self.monitoring_active = True

        # Thread 1: Monitor log file
        def monitor_log():
            while self.monitoring_active:
                self.tail_log_file()
                time.sleep(0.5)

        # Thread 2: Monitor positions
        def monitor_positions():
            while self.monitoring_active:
                self.fetch_position()
                time.sleep(2)

        # Thread 3: Monitor market price
        def monitor_price():
            while self.monitoring_active:
                self.fetch_price()
                time.sleep(3)

        threading.Thread(target=monitor_log, daemon=True).start()
        threading.Thread(target=monitor_positions, daemon=True).start()
        threading.Thread(target=monitor_price, daemon=True).start()

        self.add_log("Monitoring started")

    def tail_log_file(self):
        """Monitor bot log file for updates"""
        if not self.log_file.exists():
            return

        try:
            with open(self.log_file, 'r') as f:
                if self.log_position == 0:
                    # First time - seek to end
                    f.seek(0, 2)
                    self.log_position = f.tell()
                else:
                    f.seek(self.log_position)

                lines = f.readlines()
                self.log_position = f.tell()

                for line in lines:
                    self.parse_log_line(line.strip())
        except Exception as e:
            pass

    def parse_log_line(self, line):
        """Parse bot log line and update UI"""
        if not line:
            return

        # Add to event log
        self.add_log(line)

        # Parse strategy signals
        for strategy in self.strategy_labels.keys():
            if strategy in line or strategy.replace(" ", "") in line:
                if "EXEC" in line or "EXECUTED" in line:
                    match = re.search(r'(LONG|SHORT).*?(\d+\.?\d*)', line)
                    if match:
                        side = match.group(1)
                        price = match.group(2)
                        self.update_strategy(strategy, f"EXECUTED {side} @ {price}", self.colors['green'])
                elif "SIGNAL" in line or "signal" in line:
                    self.update_strategy(strategy, "PENDING SIGNAL", self.colors['yellow'])
                elif "BLOCK" in line:
                    self.update_strategy(strategy, "BLOCKED", self.colors['red'])

        # Parse filter status
        filter_map = {
            "Rejection": "Rejection",
            "Chop": "Chop",
            "Extension": "Extension",
            "Volatility": "Volatility",
            "Trend": "Trend",
            "Impulse": "Impulse",
            "HTF FVG": "HTF FVG",
            "Bank": "Bank Level",
            "Memory": "Memory S/R",
            "News": "News",
            "Structure": "Structure",
            "Loss": "Loss Block"
        }

        for keyword, filter_name in filter_map.items():
            if keyword in line:
                if "BLOCK" in line or "blocked" in line:
                    self.update_filter(filter_name, "BLOCK", self.colors['red'])
                elif "PASS" in line:
                    self.update_filter(filter_name, "PASS", self.colors['green'])

    def update_strategy(self, name, status, color):
        """Update strategy status"""
        def update():
            if name in self.strategy_labels:
                self.strategy_labels[name].config(text=status, fg=color)
        self.root.after(0, update)

    def update_filter(self, name, status, color):
        """Update filter status"""
        def update():
            if name in self.filter_labels:
                self.filter_labels[name].config(text=f"[{status}]", fg=color)
        self.root.after(0, update)

    def fetch_position(self):
        """Fetch current position from API"""
        if not self.session or not self.selected_account:
            return

        try:
            url = f"{CONFIG['REST_BASE_URL']}/api/Position/search"
            payload = {"accountId": self.selected_account.get('id')}

            resp = self.session.post(url, json=payload, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                positions = data.get('positions', [])

                # Update UI with position
                def update():
                    # Clear container
                    for widget in self.position_container.winfo_children():
                        widget.destroy()

                    active_pos = None
                    for pos in positions:
                        if pos.get('contractId') == self.contract_id:
                            size = pos.get('size', 0)
                            if size != 0:
                                active_pos = pos
                                break

                    if active_pos:
                        size = active_pos.get('size', 0)
                        avg_price = active_pos.get('averagePrice', 0.0)
                        side = "LONG" if size > 0 else "SHORT"

                        # Calculate P&L
                        pnl = (self.current_price - avg_price) * 5 * abs(size)
                        if side == "SHORT":
                            pnl = -pnl

                        # Create position display
                        pos_frame = tk.Frame(self.position_container, bg=self.colors['input_bg'],
                                           highlightbackground=self.colors['input_border'],
                                           highlightthickness=1)
                        pos_frame.pack(fill='x', pady=3)

                        pos_row = tk.Frame(pos_frame, bg=self.colors['input_bg'])
                        pos_row.pack(fill='x', padx=15, pady=10)

                        acc = tk.Label(pos_row,
                                      text=self.selected_account.get('name', 'ACCT'),
                                      font=("Helvetica", 13, "bold"),
                                      fg=self.colors['text_white'],
                                      bg=self.colors['input_bg'])
                        acc.pack(side='left')

                        side_label = tk.Label(pos_row, text=side,
                                             font=("Helvetica", 12, "bold"),
                                             fg=self.colors['green'] if side == "LONG" else self.colors['red'],
                                             bg=self.colors['input_bg'])
                        side_label.pack(side='left', padx=20)

                        price_label = tk.Label(pos_row, text=f"{avg_price:.2f}",
                                              font=("Helvetica", 12),
                                              fg=self.colors['text_white'],
                                              bg=self.colors['input_bg'])
                        price_label.pack(side='left', padx=10)

                        pnl_color = self.colors['green'] if pnl >= 0 else self.colors['red']
                        pnl_text = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
                        pnl_label = tk.Label(pos_row, text=pnl_text,
                                            font=("Helvetica", 12, "bold"),
                                            fg=pnl_color,
                                            bg=self.colors['input_bg'])
                        pnl_label.pack(side='right')
                    else:
                        # No position
                        no_pos = tk.Label(self.position_container,
                                         text="No active positions",
                                         font=("Helvetica", 10),
                                         fg=self.colors['text_dim'],
                                         bg=self.colors['panel_bg'])
                        no_pos.pack(pady=10)

                self.root.after(0, update)
        except Exception as e:
            pass

    def fetch_price(self):
        """Fetch current market price"""
        if not self.session or not self.contract_id:
            return

        try:
            from datetime import timezone, timedelta
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(minutes=10)

            url = f"{CONFIG['REST_BASE_URL']}/api/History/retrieveBars"
            payload = {
                "accountId": self.selected_account.get('id'),
                "contractId": self.contract_id,
                "live": False,
                "limit": 10,
                "startTime": start_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
                "endTime": end_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
                "unit": 2,
                "unitNumber": 1
            }

            resp = self.session.post(url, json=payload, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                if 'bars' in data and data['bars']:
                    latest_bar = data['bars'][0]
                    price = float(latest_bar.get('c', 0.0))
                    self.current_price = price

                    # Update UI
                    def update():
                        self.price_label.config(text=f"{price:.2f}")
                    self.root.after(0, update)
        except Exception as e:
            pass

def main():
    root = tk.Tk()
    app = JulieUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
