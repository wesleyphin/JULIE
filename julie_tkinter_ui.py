#!/usr/bin/env python3
"""
JULIE Tkinter UI - Professional Trading Dashboard
Matches the design from reference screenshots with real API integration
"""

import tkinter as tk
from tkinter import ttk, font as tkfont
import threading
import time
import requests
from datetime import datetime
from zoneinfo import ZoneInfo
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
        self.monitoring_thread = None

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

        # Strategy entries
        strategies = [
            ("Regime Adaptive", "PENDING SIGNAL", self.colors['yellow']),
            ("Silver Bullet", "EXECUTED LONG @ 5880.00", self.colors['green']),
            ("Intraday Dip", "WAITING", self.colors['text_gray']),
            ("Mean Reversion", "PENDING SIGNAL", self.colors['yellow']),
            ("Trend Follower", "EXECUTED LONG @ 5880.00", self.colors['green']),
            ("Confluence", "PASS", self.colors['text_gray']),
            ("ORB Strategy", "WAITING", self.colors['text_gray']),
            ("ML Physics", "PENDING SIGNAL", self.colors['yellow']),
            ("SMT Divergence", "WAITING", self.colors['text_gray']),
        ]

        self.strategy_labels = {}
        for name, status, color in strategies:
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

            status_label = tk.Label(entry, text=status,
                                   font=("Helvetica", 10),
                                   fg=color,
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

        # Sample position
        pos = tk.Frame(self.position_container, bg=self.colors['input_bg'],
                      highlightbackground=self.colors['input_border'],
                      highlightthickness=1)
        pos.pack(fill='x', pady=3)

        pos_row = tk.Frame(pos, bg=self.colors['input_bg'])
        pos_row.pack(fill='x', padx=15, pady=10)

        acc = tk.Label(pos_row,
                      text=self.selected_account.get('name', 'ACCT-001'),
                      font=("Helvetica", 13, "bold"),
                      fg=self.colors['text_white'],
                      bg=self.colors['input_bg'])
        acc.pack(side='left')

        side = tk.Label(pos_row, text="LONG",
                       font=("Helvetica", 12, "bold"),
                       fg=self.colors['green'],
                       bg=self.colors['input_bg'])
        side.pack(side='left', padx=20)

        price = tk.Label(pos_row, text="580.20",
                        font=("Helvetica", 12),
                        fg=self.colors['text_white'],
                        bg=self.colors['input_bg'])
        price.pack(side='left', padx=10)

        self.pnl_label = tk.Label(pos_row, text="+$350.00",
                                 font=("Helvetica", 12, "bold"),
                                 fg=self.colors['green'],
                                 bg=self.colors['input_bg'])
        self.pnl_label.pack(side='right')

    def create_filters_section(self, parent):
        """Create filter status dashboard"""
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

        # Configure grid
        for i in range(4):
            grid.columnconfigure(i, weight=1, uniform='col')
        for i in range(3):
            grid.rowconfigure(i, weight=1, uniform='row')

        # Filter entries
        filters = [
            ("Rejection", "PASS", "❌"),
            ("Chop Filter", "BLOCK", "🌊"),
            ("News", "SAFE", "📰"),
            ("Volatility", "SAFE", "🎯"),
            ("Correlation", "SAFE", "🔄"),
            ("Depth", "SAFE", "📊"),
            ("Time", "SAFE", "🕐"),
            ("System", "SAFE", "⚙️"),
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
                             font=("Helvetica", 28),
                             bg=self.colors['input_bg'])
        icon_label.pack(pady=(8, 2))

        name_label = tk.Label(box, text=name,
                             font=("Helvetica", 9, "bold"),
                             fg=self.colors['text_white'],
                             bg=self.colors['input_bg'])
        name_label.pack(pady=2)

        status_color = self.colors['green'] if status in ["PASS", "SAFE"] else self.colors['red']
        status_label = tk.Label(box, text=f"[{status}]",
                               font=("Helvetica", 8, "bold"),
                               fg=status_color,
                               bg=self.colors['input_bg'])
        status_label.pack(pady=(2, 8))

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

        # Add initial logs
        self.add_log("[2022-01-30 10:52:33.341] SYSTEM: Regime Adaptive summary")
        self.add_log("[2022-01-30 10:52:32.341] SYSTEM: Regime Adaptive summary")
        self.add_log("[2022-01-30 10:52:33.341] SYSTEM: Regime Adaptive summary")
        self.add_log("[2022-01-30 10:52:33.341] SYSTEM: Regime Adaptive summary")

    def add_log(self, message):
        """Add entry to event log"""
        self.log_text.config(state='normal')
        self.log_text.insert('end', message + '\n')
        self.log_text.see('end')
        self.log_text.config(state='disabled')

    def start_monitoring(self):
        """Start monitoring thread"""
        def monitor():
            while self.logged_in:
                # Update mock data
                import random
                self.add_log(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] SYSTEM: Monitoring active")
                time.sleep(5)

        self.monitoring_thread = threading.Thread(target=monitor, daemon=True)
        self.monitoring_thread.start()

def main():
    root = tk.Tk()
    app = JulieUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
