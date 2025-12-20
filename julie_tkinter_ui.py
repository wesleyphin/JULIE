#!/usr/bin/env python3
"""
JULIE Tkinter UI - Modern Trading Bot Interface
Replicates the professional trading dashboard design with login page
Integrates with live bot monitoring via LogMonitor and APIMonitor
"""

import tkinter as tk
from tkinter import ttk, font
import json
import os
from datetime import datetime
import threading
import time
import sys
from pathlib import Path

# Import from existing monitoring infrastructure
try:
    from julie_ui import LogMonitor, APIMonitor, get_current_session
    from config import CONFIG
    HAS_MONITORING = True
except ImportError:
    HAS_MONITORING = False
    print("Warning: Could not import monitoring modules. Running in mock mode.")

class JulieUI:
    def __init__(self, root):
        self.root = root
        self.root.title("JULIE")
        self.root.geometry("1400x900")
        self.root.configure(bg='#0a0e1a')

        # Color scheme - dark professional theme
        self.colors = {
            'bg_primary': '#0a0e1a',      # Main dark background
            'bg_secondary': '#141824',     # Slightly lighter panels
            'bg_tertiary': '#1e2330',      # Input backgrounds
            'accent_green': '#4ade80',     # Success/long green
            'accent_red': '#f87171',       # Danger/short red
            'accent_yellow': '#fbbf24',    # Warning/pending yellow
            'text_primary': '#e5e7eb',     # Main text white
            'text_secondary': '#9ca3af',   # Secondary text gray
            'border': '#2d3548',           # Border color
            'button_green': '#22c55e',     # Button green
            'button_hover': '#16a34a',     # Button hover
        }

        # Load or create config
        self.config_file = 'config.py'
        self.load_accounts()

        # State variables
        self.current_account = None
        self.logged_in = False
        self.mock_data = {
            'price': 5880.25,
            'change_pct': 20.73,
            'positions': [],
            'strategies': {},
            'filters': {},
            'event_log': []
        }

        # Create login page first
        self.show_login_page()

    def load_accounts(self):
        """Load available accounts from config or create mock accounts"""
        if HAS_MONITORING and CONFIG.get('USERNAME') and CONFIG.get('API_KEY'):
            # Try to fetch real accounts later after login
            self.accounts = ["Loading accounts..."]
        else:
            # Mock accounts for testing
            self.accounts = [
                "ACCT-001",
                "ACCT-002",
                "ACCT-003",
                "ACCT-004"
            ]

    def show_login_page(self):
        """Create the login page interface"""
        # Clear any existing widgets
        for widget in self.root.winfo_children():
            widget.destroy()

        # Create gradient-like background
        self.login_frame = tk.Frame(self.root, bg='#0a0e1a')
        self.login_frame.pack(fill='both', expand=True)

        # Add gradient effect with overlapping frames
        gradient_frame = tk.Frame(self.login_frame, bg='#0a1a0e')
        gradient_frame.place(relx=0.5, rely=0.5, anchor='center', relwidth=0.8, relheight=0.8)

        # Main login container
        login_container = tk.Frame(self.login_frame, bg=self.colors['bg_secondary'],
                                   highlightbackground=self.colors['accent_green'],
                                   highlightthickness=1)
        login_container.place(relx=0.5, rely=0.5, anchor='center', width=500, height=500)

        # JULIE Title
        title_font = font.Font(family="Helvetica", size=64, weight="bold")
        title = tk.Label(login_container, text="JULIE",
                        font=title_font,
                        fg=self.colors['text_primary'],
                        bg=self.colors['bg_secondary'])
        title.pack(pady=(60, 80))

        # Account Number Label
        account_label = tk.Label(login_container, text="ACCOUNT NUMBER",
                                font=("Helvetica", 11, "bold"),
                                fg=self.colors['text_secondary'],
                                bg=self.colors['bg_secondary'])
        account_label.pack(pady=(0, 10))

        # Account Dropdown
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Custom.TCombobox',
                       fieldbackground=self.colors['bg_tertiary'],
                       background=self.colors['bg_tertiary'],
                       foreground=self.colors['text_primary'],
                       arrowcolor=self.colors['text_primary'],
                       bordercolor=self.colors['border'],
                       lightcolor=self.colors['bg_tertiary'],
                       darkcolor=self.colors['bg_tertiary'])

        self.account_var = tk.StringVar(value=self.accounts[0])
        account_dropdown = ttk.Combobox(login_container,
                                       textvariable=self.account_var,
                                       values=self.accounts,
                                       state='readonly',
                                       font=("Helvetica", 14),
                                       style='Custom.TCombobox',
                                       width=30)
        account_dropdown.pack(pady=(0, 40), padx=50)

        # Login Button
        login_btn = tk.Button(login_container,
                             text="🔒 LOGIN",
                             font=("Helvetica", 14, "bold"),
                             bg=self.colors['button_green'],
                             fg='white',
                             activebackground=self.colors['button_hover'],
                             activeforeground='white',
                             relief='flat',
                             cursor='hand2',
                             command=self.handle_login)
        login_btn.pack(pady=(0, 20), padx=50, fill='x', ipady=10)

        # Forgot Password Link
        forgot_link = tk.Label(login_container, text="FORGOT PASSWORD?",
                              font=("Helvetica", 10, "underline"),
                              fg=self.colors['text_secondary'],
                              bg=self.colors['bg_secondary'],
                              cursor='hand2')
        forgot_link.pack(pady=(10, 0))

    def handle_login(self):
        """Handle login button click"""
        self.current_account = self.account_var.get()
        self.logged_in = True
        self.show_dashboard()

    def show_dashboard(self):
        """Create the main dashboard interface"""
        # Clear login page
        for widget in self.root.winfo_children():
            widget.destroy()

        # Main dashboard container
        dashboard = tk.Frame(self.root, bg=self.colors['bg_primary'])
        dashboard.pack(fill='both', expand=True)

        # Header with JULIE title
        header = tk.Frame(dashboard, bg=self.colors['bg_primary'], height=60)
        header.pack(fill='x', padx=20, pady=10)
        header.pack_propagate(False)

        title = tk.Label(header, text="JULIE",
                        font=("Helvetica", 28, "bold"),
                        fg=self.colors['text_primary'],
                        bg=self.colors['bg_primary'])
        title.pack(side='left')

        # Account info in header
        account_label = tk.Label(header, text=f"Account: {self.current_account}",
                                font=("Helvetica", 12),
                                fg=self.colors['text_secondary'],
                                bg=self.colors['bg_primary'])
        account_label.pack(side='right', padx=20)

        # Main content area
        content = tk.Frame(dashboard, bg=self.colors['bg_primary'])
        content.pack(fill='both', expand=True, padx=20, pady=10)

        # Left column (70% width)
        left_column = tk.Frame(content, bg=self.colors['bg_primary'])
        left_column.pack(side='left', fill='both', expand=True, padx=(0, 10))

        # Right column (30% width)
        right_column = tk.Frame(content, bg=self.colors['bg_primary'])
        right_column.pack(side='right', fill='both', padx=(10, 0))

        # Create all dashboard sections
        self.create_market_context_section(left_column)
        self.create_strategy_list_section(left_column)
        self.create_active_positions_section(left_column)
        self.create_filter_dashboard_section(left_column)
        self.create_event_log_section(right_column)

        # Start monitoring thread
        self.start_monitoring()

    def create_market_context_section(self, parent):
        """Create the market context display section"""
        section = tk.Frame(parent, bg=self.colors['bg_secondary'],
                          highlightbackground=self.colors['border'],
                          highlightthickness=1)
        section.pack(fill='x', pady=(0, 10))

        # Section header
        header = tk.Label(section, text="SIGNAL MONITOR & MARKET CONTEXT",
                         font=("Helvetica", 10, "bold"),
                         fg=self.colors['text_secondary'],
                         bg=self.colors['bg_secondary'],
                         anchor='w')
        header.pack(fill='x', padx=15, pady=(10, 20))

        # Market display
        market_container = tk.Frame(section, bg=self.colors['bg_secondary'])
        market_container.pack(fill='x', padx=15, pady=(0, 20))

        self.market_symbol = tk.Label(market_container, text="MES FUTURES",
                                     font=("Helvetica", 42, "bold"),
                                     fg=self.colors['text_primary'],
                                     bg=self.colors['bg_secondary'])
        self.market_symbol.pack()

        price_frame = tk.Frame(market_container, bg=self.colors['bg_secondary'])
        price_frame.pack()

        self.market_price = tk.Label(price_frame, text="5880.25",
                                    font=("Helvetica", 48, "bold"),
                                    fg=self.colors['text_primary'],
                                    bg=self.colors['bg_secondary'])
        self.market_price.pack(side='left', padx=10)

        self.market_change = tk.Label(price_frame, text="▲ 20.73%",
                                     font=("Helvetica", 48, "bold"),
                                     fg=self.colors['accent_green'],
                                     bg=self.colors['bg_secondary'])
        self.market_change.pack(side='left')

    def create_strategy_list_section(self, parent):
        """Create the strategy list section"""
        section = tk.Frame(parent, bg=self.colors['bg_secondary'],
                          highlightbackground=self.colors['border'],
                          highlightthickness=1)
        section.pack(fill='both', expand=True, pady=(0, 10))

        # Section header
        header = tk.Label(section, text="STRATEGY LIST",
                         font=("Helvetica", 10, "bold"),
                         fg=self.colors['text_secondary'],
                         bg=self.colors['bg_secondary'],
                         anchor='w')
        header.pack(fill='x', padx=15, pady=(10, 10))

        # Scrollable strategy list
        list_container = tk.Frame(section, bg=self.colors['bg_secondary'])
        list_container.pack(fill='both', expand=True, padx=15, pady=(0, 15))

        # Create strategy entries
        strategies = [
            ("Regime Adaptive", "PENDING SIGNAL"),
            ("Intraday Dip", "WAITING"),
            ("Confluence", "PASS"),
            ("ICT Model (Silver Bullet)", "EXECUTED LONG @ 5880.00"),
            ("ORB Strategy", "WAITING"),
            ("ML Physics", "PENDING SIGNAL"),
            ("Dynamic Engine 1", "PASS"),
            ("Dynamic Engine 2", "PASS"),
            ("SMT Divergence", "WAITING")
        ]

        self.strategy_labels = {}
        for strategy_name, status in strategies:
            self.create_strategy_entry(list_container, strategy_name, status)

    def create_strategy_entry(self, parent, name, status):
        """Create a single strategy entry"""
        entry = tk.Frame(parent, bg=self.colors['bg_tertiary'],
                        highlightbackground=self.colors['border'],
                        highlightthickness=1)
        entry.pack(fill='x', pady=3)

        name_label = tk.Label(entry, text=name,
                             font=("Helvetica", 11),
                             fg=self.colors['text_primary'],
                             bg=self.colors['bg_tertiary'],
                             anchor='w')
        name_label.pack(side='left', padx=15, pady=12)

        # Status color coding
        if "EXECUTED" in status:
            status_color = self.colors['accent_green']
        elif "PENDING" in status or "WAITING" in status:
            status_color = self.colors['accent_yellow']
        else:
            status_color = self.colors['text_secondary']

        status_label = tk.Label(entry, text=status,
                               font=("Helvetica", 10),
                               fg=status_color,
                               bg=self.colors['bg_tertiary'],
                               anchor='e')
        status_label.pack(side='right', padx=15)

        self.strategy_labels[name] = status_label

    def create_active_positions_section(self, parent):
        """Create the active positions section"""
        section = tk.Frame(parent, bg=self.colors['bg_secondary'],
                          highlightbackground=self.colors['border'],
                          highlightthickness=1)
        section.pack(fill='x', pady=(0, 10))

        # Section header
        header = tk.Label(section, text="ACTIVE POSITIONS",
                         font=("Helvetica", 10, "bold"),
                         fg=self.colors['text_secondary'],
                         bg=self.colors['bg_secondary'],
                         anchor='w')
        header.pack(fill='x', padx=15, pady=(10, 10))

        # Position display
        self.position_frame = tk.Frame(section, bg=self.colors['bg_secondary'])
        self.position_frame.pack(fill='x', padx=15, pady=(0, 15))

        # Sample position
        pos_container = tk.Frame(self.position_frame, bg=self.colors['bg_tertiary'],
                                highlightbackground=self.colors['border'],
                                highlightthickness=1)
        pos_container.pack(fill='x', pady=5)

        pos_info = tk.Frame(pos_container, bg=self.colors['bg_tertiary'])
        pos_info.pack(fill='x', padx=15, pady=10)

        account_label = tk.Label(pos_info, text=self.current_account,
                                font=("Helvetica", 12, "bold"),
                                fg=self.colors['text_primary'],
                                bg=self.colors['bg_tertiary'])
        account_label.pack(side='left')

        side_label = tk.Label(pos_info, text="LONG",
                             font=("Helvetica", 11, "bold"),
                             fg=self.colors['accent_green'],
                             bg=self.colors['bg_tertiary'])
        side_label.pack(side='left', padx=20)

        price_label = tk.Label(pos_info, text="580.20",
                              font=("Helvetica", 11),
                              fg=self.colors['text_primary'],
                              bg=self.colors['bg_tertiary'])
        price_label.pack(side='left', padx=10)

        pnl_label = tk.Label(pos_info, text="+$350.00",
                            font=("Helvetica", 11, "bold"),
                            fg=self.colors['accent_green'],
                            bg=self.colors['bg_tertiary'])
        pnl_label.pack(side='right')

    def create_filter_dashboard_section(self, parent):
        """Create the filter status dashboard section"""
        section = tk.Frame(parent, bg=self.colors['bg_secondary'],
                          highlightbackground=self.colors['border'],
                          highlightthickness=1)
        section.pack(fill='both', expand=True)

        # Section header
        header = tk.Label(section, text="FILTER STATUS DASHBOARD",
                         font=("Helvetica", 10, "bold"),
                         fg=self.colors['text_secondary'],
                         bg=self.colors['bg_secondary'],
                         anchor='w')
        header.pack(fill='x', padx=15, pady=(10, 10))

        # Filter grid
        grid_container = tk.Frame(section, bg=self.colors['bg_secondary'])
        grid_container.pack(fill='both', expand=True, padx=15, pady=(0, 15))

        # Configure grid
        for i in range(4):
            grid_container.columnconfigure(i, weight=1)

        filters = [
            ("Rejection", "PASS", "❌"),
            ("Chop Filter", "BLOCK", "🌊"),
            ("News", "SAFE", "📰"),
            ("Volatility", "SAFE", "🎯"),
            ("Correlation", "SAFE", "🔄"),
            ("Depth", "SAFE", "📊"),
            ("Time", "SAFE", "🕐"),
            ("System", "SAFE", "⚙️"),
            ("Extension", "PASS", "📈"),
            ("Trend", "PASS", "📉"),
            ("Impulse", "SAFE", "⚡"),
            ("HTF FVG", "SAFE", "🎚️")
        ]

        self.filter_labels = {}
        row, col = 0, 0
        for filter_name, status, icon in filters:
            self.create_filter_indicator(grid_container, row, col,
                                        filter_name, status, icon)
            col += 1
            if col >= 4:
                col = 0
                row += 1

    def create_filter_indicator(self, parent, row, col, name, status, icon):
        """Create a single filter indicator"""
        container = tk.Frame(parent, bg=self.colors['bg_tertiary'],
                           highlightbackground=self.colors['border'],
                           highlightthickness=1)
        container.grid(row=row, column=col, padx=5, pady=5, sticky='nsew')

        icon_label = tk.Label(container, text=icon,
                             font=("Helvetica", 24),
                             bg=self.colors['bg_tertiary'])
        icon_label.pack(pady=(10, 5))

        name_label = tk.Label(container, text=name,
                             font=("Helvetica", 9, "bold"),
                             fg=self.colors['text_primary'],
                             bg=self.colors['bg_tertiary'])
        name_label.pack()

        # Status color
        if status == "PASS":
            status_color = self.colors['accent_green']
        elif status == "BLOCK":
            status_color = self.colors['accent_red']
        else:  # SAFE
            status_color = self.colors['accent_green']

        status_label = tk.Label(container, text=f"[{status}]",
                               font=("Helvetica", 8, "bold"),
                               fg=status_color,
                               bg=self.colors['bg_tertiary'])
        status_label.pack(pady=(0, 10))

        self.filter_labels[name] = status_label

    def create_event_log_section(self, parent):
        """Create the live event log section"""
        section = tk.Frame(parent, bg=self.colors['bg_secondary'],
                          highlightbackground=self.colors['border'],
                          highlightthickness=1,
                          width=400)
        section.pack(fill='both', expand=True)
        section.pack_propagate(False)

        # Section header
        header = tk.Label(section, text="LIVE EVENT LOG",
                         font=("Helvetica", 10, "bold"),
                         fg=self.colors['text_secondary'],
                         bg=self.colors['bg_secondary'],
                         anchor='w')
        header.pack(fill='x', padx=15, pady=(10, 10))

        # Log container with scrollbar
        log_frame = tk.Frame(section, bg=self.colors['bg_secondary'])
        log_frame.pack(fill='both', expand=True, padx=15, pady=(0, 15))

        scrollbar = tk.Scrollbar(log_frame)
        scrollbar.pack(side='right', fill='y')

        self.event_log = tk.Text(log_frame,
                                bg=self.colors['bg_tertiary'],
                                fg=self.colors['text_secondary'],
                                font=("Courier", 8),
                                wrap='word',
                                yscrollcommand=scrollbar.set,
                                state='disabled')
        self.event_log.pack(fill='both', expand=True)
        scrollbar.config(command=self.event_log.yview)

        # Add sample logs
        self.add_log_entry("SYSTEM: Regime Adaptive summary")
        self.add_log_entry("SYSTEM: Market session NY_AM detected")
        self.add_log_entry("SIGNAL: ICT Model generated LONG signal")
        self.add_log_entry("FILTER: All filters PASSED")
        self.add_log_entry("TRADE: Executed LONG @ 5880.00")
        self.add_log_entry("SYSTEM: Position monitoring active")

    def add_log_entry(self, message):
        """Add an entry to the event log"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        log_line = f"[{timestamp}] {message}\n"

        self.event_log.config(state='normal')
        self.event_log.insert('end', log_line)
        self.event_log.see('end')
        self.event_log.config(state='disabled')

    def start_monitoring(self):
        """Start background monitoring thread"""
        if HAS_MONITORING:
            self.start_real_monitoring()
        else:
            self.start_mock_monitoring()

    def start_real_monitoring(self):
        """Start real monitoring with API and log file integration"""
        def monitor_loop():
            # Initialize monitors
            self.api_monitor = APIMonitor()
            self.log_monitor = LogMonitor()

            # Authenticate
            self.add_log_entry("API: Authenticating...")
            if not self.api_monitor.login():
                self.add_log_entry("ERROR: Authentication failed")
                return

            # Set account ID (bypass interactive selection and use the one selected in login)
            # Note: In a real implementation, you might want to fetch accounts via API
            # For now, we use the selected account from login
            self.api_monitor.account_id = self.current_account
            self.api_monitor.account_ids = [self.current_account]
            self.add_log_entry(f"API: Using account {self.current_account}")

            # Fetch contract
            if not self.api_monitor.fetch_contract_id():
                self.add_log_entry("ERROR: Failed to fetch contract")
                return

            self.add_log_entry("SYSTEM: Monitoring started")

            last_position_check = 0
            last_price_check = 0
            last_log_check = 0

            while self.logged_in:
                now = time.time()

                # Check position every 2 seconds
                if now - last_position_check > 2.0:
                    try:
                        position = self.api_monitor.fetch_position()
                        if position:
                            self.update_position_display(position)
                    except Exception as e:
                        pass
                    last_position_check = now

                # Check price every 3 seconds
                if now - last_price_check > 3.0:
                    try:
                        price = self.api_monitor.fetch_market_data()
                        if price:
                            self.update_market_price(price)
                    except Exception as e:
                        pass
                    last_price_check = now

                # Check log file every 0.5 seconds
                if now - last_log_check > 0.5:
                    try:
                        self.log_monitor.tail_log()
                    except Exception as e:
                        pass
                    last_log_check = now

                time.sleep(0.1)

        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()

    def start_mock_monitoring(self):
        """Start mock monitoring with simulated data"""
        def monitor_loop():
            while self.logged_in:
                # Update mock data (in real implementation, read from bot)
                self.update_dashboard_data()
                time.sleep(2)

        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()

    def update_position_display(self, position):
        """Update position display with real data"""
        if not hasattr(self, 'position_frame'):
            return

        # Update in UI thread
        def update():
            # Clear old position display
            for widget in self.position_frame.winfo_children():
                widget.destroy()

            if position and position.get('side'):
                # Create position display
                pos_container = tk.Frame(self.position_frame, bg=self.colors['bg_tertiary'],
                                        highlightbackground=self.colors['border'],
                                        highlightthickness=1)
                pos_container.pack(fill='x', pady=5)

                pos_info = tk.Frame(pos_container, bg=self.colors['bg_tertiary'])
                pos_info.pack(fill='x', padx=15, pady=10)

                account_label = tk.Label(pos_info, text=self.current_account,
                                        font=("Helvetica", 12, "bold"),
                                        fg=self.colors['text_primary'],
                                        bg=self.colors['bg_tertiary'])
                account_label.pack(side='left')

                side_color = self.colors['accent_green'] if position['side'] == 'LONG' else self.colors['accent_red']
                side_label = tk.Label(pos_info, text=position['side'],
                                     font=("Helvetica", 11, "bold"),
                                     fg=side_color,
                                     bg=self.colors['bg_tertiary'])
                side_label.pack(side='left', padx=20)

                price_label = tk.Label(pos_info, text=f"{position['avg_price']:.2f}",
                                      font=("Helvetica", 11),
                                      fg=self.colors['text_primary'],
                                      bg=self.colors['bg_tertiary'])
                price_label.pack(side='left', padx=10)

                # Calculate P&L if we have current price
                if hasattr(self, 'current_price') and self.current_price:
                    pnl = (self.current_price - position['avg_price']) * 5  # MES $5 per point
                    if position['side'] == 'SHORT':
                        pnl = -pnl
                    pnl_color = self.colors['accent_green'] if pnl >= 0 else self.colors['accent_red']
                    pnl_text = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"

                    pnl_label = tk.Label(pos_info, text=pnl_text,
                                        font=("Helvetica", 11, "bold"),
                                        fg=pnl_color,
                                        bg=self.colors['bg_tertiary'])
                    pnl_label.pack(side='right')
            else:
                # No position
                no_pos = tk.Label(self.position_frame, text="No active positions",
                                 font=("Helvetica", 10),
                                 fg=self.colors['text_secondary'],
                                 bg=self.colors['bg_secondary'])
                no_pos.pack(pady=20)

        self.root.after(0, update)

    def update_market_price(self, price):
        """Update market price display"""
        if not hasattr(self, 'market_price'):
            return

        self.current_price = price

        def update():
            # Update price
            self.market_price.config(text=f"{price:.2f}")

            # Calculate change percent (mock for now, would need previous close)
            # For now, just show a random small change
            import random
            change_pct = random.uniform(-0.5, 2.0)
            change_color = self.colors['accent_green'] if change_pct >= 0 else self.colors['accent_red']
            change_symbol = "▲" if change_pct >= 0 else "▼"

            self.market_change.config(
                text=f"{change_symbol} {abs(change_pct):.2f}%",
                fg=change_color
            )

        self.root.after(0, update)

    def update_dashboard_data(self):
        """Update dashboard with latest data"""
        # This would read from julie_ui.py log monitor or direct bot integration
        # For now, just add periodic log entries
        if hasattr(self, 'event_log'):
            messages = [
                "SYSTEM: Monitoring market conditions",
                "FILTER: Chop filter analyzing structure",
                "SYSTEM: No new signals generated",
                "SYSTEM: Position P&L: +$350.00"
            ]
            import random
            self.add_log_entry(random.choice(messages))

def main():
    root = tk.Tk()
    app = JulieUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
