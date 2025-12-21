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
import subprocess
import sys
import os
import signal

# Import from existing monitoring infrastructure
try:
    from config import CONFIG
    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False
    CONFIG = {}

# Import copy trading config management
try:
    from copy_trading_config import (
        load_copy_trading_config,
        enable_copy_trading as enable_ct,
        disable_copy_trading as disable_ct
    )
    HAS_COPY_TRADING = True
except ImportError:
    HAS_COPY_TRADING = False
    def load_copy_trading_config():
        return {'enabled': False, 'followers': []}
    def enable_ct():
        return False
    def disable_ct():
        return False

class JulieUI:
    def __init__(self, root):
        self.root = root
        self.root.title("JULIE")
        self.root.geometry("1600x900")
        self.root.configure(bg='#000000')

        # Color scheme - all solid black backgrounds
        self.colors = {
            'bg_dark': '#000000',
            'bg_gradient_start': '#000000',
            'bg_gradient_end': '#000000',
            'panel_bg': '#000000',
            'panel_border': '#2a4a3a',
            'input_bg': '#000000',
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
        self.paused = False  # Pause state for log analysis

        # Bot integration
        self.log_file = Path("topstep_live_bot.log")
        self.log_position = 0
        self.contract_id = None
        self.bot_process = None  # Track the julie001.py subprocess

        # Animated logo tracking
        self.logo_frames = []
        self.logo_frame_index = 0
        self.logo_label = None
        self.logo_animation_id = None

        # Setup cleanup on window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Show login page
        self.show_login_page()

    def show_login_page(self):
        """Create login page matching the screenshot"""
        # Clear root
        for widget in self.root.winfo_children():
            widget.destroy()

        # Create solid black background
        canvas = tk.Canvas(self.root, bg='#000000', highlightthickness=0)
        canvas.pack(fill='both', expand=True)

        # Create login panel
        panel_frame = tk.Frame(canvas, bg=self.colors['panel_bg'],
                              highlightthickness=0)
        canvas.create_window(800, 450, window=panel_frame, width=500, height=650)

        # JULIE Logo (Animated GIF)
        try:
            logo_path = Path(__file__).parent / "logo.gif"
            if logo_path.exists():
                from PIL import Image, ImageTk

                # Load all frames from the animated GIF
                gif = Image.open(logo_path)
                self.logo_frames = []

                try:
                    while True:
                        # Copy the frame and convert to PhotoImage
                        frame = gif.copy()
                        # Resize if needed (max 300px wide, max 200px tall)
                        max_width = 300
                        max_height = 200

                        width_ratio = max_width / frame.width if frame.width > max_width else 1
                        height_ratio = max_height / frame.height if frame.height > max_height else 1
                        ratio = min(width_ratio, height_ratio)

                        if ratio < 1:
                            new_width = int(frame.width * ratio)
                            new_height = int(frame.height * ratio)
                            frame = frame.resize((new_width, new_height), Image.Resampling.LANCZOS)

                        photo = ImageTk.PhotoImage(frame)
                        self.logo_frames.append(photo)
                        gif.seek(len(self.logo_frames))  # Move to next frame
                except EOFError:
                    pass  # End of frames

                # Create label and start animation
                if self.logo_frames:
                    self.logo_label = tk.Label(panel_frame, bg=self.colors['panel_bg'])
                    self.logo_label.pack(pady=(30, 40))
                    self.logo_frame_index = 0
                    self.animate_logo()
                else:
                    raise ValueError("No frames found in GIF")
            else:
                # Fallback to text if logo.gif not found
                title_font = tkfont.Font(family="Helvetica", size=72, weight="bold")
                title = tk.Label(panel_frame, text="JULIE",
                                font=title_font,
                                fg=self.colors['text_white'],
                                bg=self.colors['panel_bg'])
                title.pack(pady=(50, 80))
        except Exception as e:
            # Fallback if logo cannot be loaded
            print(f"Could not load logo.gif: {e}")
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
        dropdown.pack(pady=(0, 20), padx=60)

        # Copy Trading Status Section
        self.create_copy_trading_status(panel_frame)

        # Login Button
        login_btn = tk.Button(panel_frame,
                             text="LOGIN",
                             font=("Helvetica", 16, "bold"),
                             bg='#2d2d2d',  # Dark grey
                             fg='#000000',  # Black font
                             activebackground='#3d3d3d',  # Lighter grey on hover
                             activeforeground='#000000',  # Black font on hover
                             relief='flat',
                             cursor='hand2',
                             command=self.handle_login)
        login_btn.pack(pady=(0, 25), padx=60, fill='x', ipady=12)

    def create_copy_trading_status(self, parent):
        """Create copy trading status indicator on login page"""
        # Load copy trading configuration from persistent storage
        copy_config = load_copy_trading_config()
        enabled = copy_config.get('enabled', False)
        followers = copy_config.get('followers', [])
        active_followers = [f for f in followers if f.get('enabled', True)]

        # Create status frame
        status_frame = tk.Frame(parent, bg=self.colors['panel_bg'])
        status_frame.pack(pady=(0, 20), padx=60)

        if enabled and active_followers:
            # Copy trading is enabled
            status_icon = tk.Label(status_frame,
                                  text="📋",
                                  font=("Helvetica", 16),
                                  bg=self.colors['panel_bg'],
                                  fg=self.colors['green'])
            status_icon.pack(side='left', padx=(0, 8))

            status_text = tk.Label(status_frame,
                                  text=f"Copy Trading: {len(active_followers)} follower{'s' if len(active_followers) > 1 else ''} active",
                                  font=("Helvetica", 11),
                                  bg=self.colors['panel_bg'],
                                  fg=self.colors['green'])
            status_text.pack(side='left')

            # Add detail text below
            detail_text = f"Leader trades will be replicated to {len(active_followers)} account{'s' if len(active_followers) > 1 else ''}"
            detail_label = tk.Label(parent,
                                   text=detail_text,
                                   font=("Helvetica", 9),
                                   bg=self.colors['panel_bg'],
                                   fg=self.colors['text_dim'])
            detail_label.pack(pady=(0, 15))

        elif enabled and not active_followers:
            # Copy trading enabled but no followers
            status_icon = tk.Label(status_frame,
                                  text="⚠️",
                                  font=("Helvetica", 16),
                                  bg=self.colors['panel_bg'],
                                  fg=self.colors['yellow'])
            status_icon.pack(side='left', padx=(0, 8))

            status_text = tk.Label(status_frame,
                                  text="Copy Trading: Enabled (No followers configured)",
                                  font=("Helvetica", 11),
                                  bg=self.colors['panel_bg'],
                                  fg=self.colors['yellow'])
            status_text.pack(side='left')

            detail_label = tk.Label(parent,
                                   text="Configure follower accounts in config.py",
                                   font=("Helvetica", 9),
                                   bg=self.colors['panel_bg'],
                                   fg=self.colors['text_dim'])
            detail_label.pack(pady=(0, 15))

        else:
            # Copy trading disabled
            status_text = tk.Label(status_frame,
                                  text="Copy Trading: Disabled",
                                  font=("Helvetica", 10),
                                  bg=self.colors['panel_bg'],
                                  fg=self.colors['text_dim'])
            status_text.pack()

            detail_label = tk.Label(parent,
                                   text="Enable in config.py to replicate trades",
                                   font=("Helvetica", 9),
                                   bg=self.colors['panel_bg'],
                                   fg=self.colors['text_dim'])
            detail_label.pack(pady=(0, 15))

    def animate_logo(self):
        """Animate the logo GIF by cycling through frames"""
        if self.logo_label and self.logo_frames:
            # Update to next frame
            self.logo_label.config(image=self.logo_frames[self.logo_frame_index])
            self.logo_frame_index = (self.logo_frame_index + 1) % len(self.logo_frames)

            # Schedule next frame (50ms = ~20 FPS)
            self.logo_animation_id = self.root.after(50, self.animate_logo)

    def stop_logo_animation(self):
        """Stop the logo animation"""
        if self.logo_animation_id:
            self.root.after_cancel(self.logo_animation_id)
            self.logo_animation_id = None

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
        """Handle login button click and launch julie001.py in background"""
        selected_name = self.account_var.get()

        # Find the account
        for acc in self.accounts:
            if acc.get('name') == selected_name or acc.get('id') == selected_name:
                self.selected_account = acc
                break

        if not self.selected_account:
            self.selected_account = {'name': selected_name, 'id': selected_name}

        # Launch julie001.py in background with selected account
        self.launch_bot()

        self.logged_in = True
        self.show_dashboard()

    def launch_bot(self):
        """Launch julie001.py as a background subprocess with selected account"""
        if self.bot_process:
            # Bot already running
            return

        account_id = self.selected_account.get('id')
        if not account_id:
            print("No account ID selected, cannot launch bot")
            return

        # Set environment variable for account ID
        env = os.environ.copy()
        env['JULIE_ACCOUNT_ID'] = str(account_id)

        # Find julie001.py in current directory
        bot_script = Path(__file__).parent / "julie001.py"

        if not bot_script.exists():
            print(f"Error: {bot_script} not found")
            return

        try:
            # Launch julie001.py as subprocess
            self.bot_process = subprocess.Popen(
                [sys.executable, str(bot_script)],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,  # Prevent interactive prompts
                cwd=str(bot_script.parent),
                bufsize=1  # Line buffered
            )
            print(f"✓ julie001.py launched in background (PID: {self.bot_process.pid})")
            print(f"  Account: {account_id}")

            # Monitor bot output in background thread
            def monitor_bot_output():
                """Monitor bot stdout/stderr and log any errors"""
                if not self.bot_process:
                    return
                for line in iter(self.bot_process.stderr.readline, b''):
                    if line:
                        decoded = line.decode('utf-8', errors='ignore').strip()
                        if decoded and any(kw in decoded.upper() for kw in ['ERROR', 'CRITICAL', 'FAILED']):
                            print(f"[BOT] {decoded}")

            threading.Thread(target=monitor_bot_output, daemon=True).start()

        except Exception as e:
            print(f"Error launching julie001.py: {e}")
            self.bot_process = None

    def on_closing(self):
        """Clean up when window is closed"""
        # Stop logo animation
        self.stop_logo_animation()

        # Stop monitoring
        self.monitoring_active = False

        # Terminate bot subprocess if running
        if self.bot_process:
            print("\nStopping julie001.py bot...")
            try:
                # Try graceful termination first
                self.bot_process.terminate()
                self.bot_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Force kill if it doesn't stop
                self.bot_process.kill()
                self.bot_process.wait()
            print("Bot stopped.")

        # Close window
        self.root.destroy()

    def show_dashboard(self):
        """Create main dashboard matching the screenshot"""
        # Stop logo animation before clearing widgets
        self.stop_logo_animation()

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

        # Pause/Play controls (top right)
        controls_frame = tk.Frame(header, bg=self.colors['bg_dark'])
        controls_frame.pack(side='right')

        self.pause_btn = tk.Button(controls_frame,
                                   text="⏸ PAUSE",
                                   font=("Helvetica", 12, "bold"),
                                   bg=self.colors['yellow'],
                                   fg='#000000',
                                   activebackground='#fcd34d',
                                   activeforeground='#000000',
                                   relief='flat',
                                   cursor='hand2',
                                   command=self.toggle_pause,
                                   padx=20,
                                   pady=8)
        self.pause_btn.pack(side='left', padx=5)

        self.play_btn = tk.Button(controls_frame,
                                  text="▶ PLAY",
                                  font=("Helvetica", 12, "bold"),
                                  bg=self.colors['green'],
                                  fg='#000000',
                                  activebackground=self.colors['green_light'],
                                  activeforeground='#000000',
                                  relief='flat',
                                  cursor='hand2',
                                  command=self.toggle_pause,
                                  padx=20,
                                  pady=8,
                                  state='disabled')
        self.play_btn.pack(side='left', padx=5)

        # Main content area
        content = tk.Frame(main, bg=self.colors['bg_dark'])
        content.pack(fill='both', expand=True, padx=25, pady=10)

        # Left panel (70%)
        left_panel = tk.Frame(content, bg=self.colors['bg_dark'])
        left_panel.pack(side='left', fill='both', expand=True, padx=(0, 15))

        # Right panel (30%) - Market Context
        right_panel = tk.Frame(content, bg=self.colors['bg_dark'], width=450)
        right_panel.pack(side='right', fill='both', padx=(15, 0))
        right_panel.pack_propagate(False)

        # Build left panel sections - Event Log at top
        self.create_event_log(left_panel)

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
        self.create_gemini_logs_section(strategy_frame)
        self.create_positions_section(right_col)
        self.create_copy_trading_stats_section(right_col)
        self.create_filters_section(right_col)

        # Right panel - Market Context
        self.create_market_section(right_panel)

        # Fetch contract ID
        self.fetch_contract_id()

        # Start monitoring
        self.start_monitoring()

    def create_market_section(self, parent):
        """Create market context section"""
        section = tk.Frame(parent, bg=self.colors['panel_bg'],
                          highlightbackground=self.colors['panel_border'],
                          highlightthickness=1)
        section.pack(fill='both', expand=True)  # Fill entire right panel height

        # Header
        header = tk.Label(section, text="SIGNAL MONITOR & MARKET CONTEXT",
                         font=("Helvetica", 10, "bold"),
                         fg=self.colors['text_gray'],
                         bg=self.colors['panel_bg'],
                         anchor='w')
        header.pack(fill='x', padx=20, pady=(15, 10))

        # Market context log (scrollable)
        log_frame = tk.Frame(section, bg=self.colors['panel_bg'])
        log_frame.pack(fill='both', expand=True, padx=20, pady=(0, 15))

        scrollbar = tk.Scrollbar(log_frame)
        scrollbar.pack(side='right', fill='y')

        self.market_log = tk.Text(log_frame,
                                 bg=self.colors['input_bg'],
                                 fg=self.colors['text_gray'],
                                 font=("Courier", 8),
                                 wrap='word',
                                 height=12,
                                 yscrollcommand=scrollbar.set,
                                 state='disabled',
                                 relief='flat')
        self.market_log.pack(fill='both', expand=True)
        scrollbar.config(command=self.market_log.yview)

        # Configure text tags for color coding
        self.market_log.tag_config('session', foreground=self.colors['blue'])
        self.market_log.tag_config('rejection', foreground=self.colors['yellow'])
        self.market_log.tag_config('bank', foreground=self.colors['green_light'])
        self.market_log.tag_config('warning', foreground=self.colors['red'])
        self.market_log.tag_config('drift', foreground='#00d4ff')  # Cyan for drift/calibration
        self.market_log.tag_config('info', foreground=self.colors['text_gray'])

        self.add_market_log("Waiting for bot market context...")

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

        # Grid container (2 columns)
        grid = tk.Frame(section, bg=self.colors['panel_bg'])
        grid.pack(fill='both', expand=True, padx=20, pady=(0, 15))

        # Configure grid
        grid.columnconfigure(0, weight=1, uniform='col')
        grid.columnconfigure(1, weight=1, uniform='col')

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
        for idx, name in enumerate(strategies):
            row = idx // 2
            col = idx % 2

            entry = tk.Frame(grid, bg=self.colors['input_bg'],
                           highlightbackground=self.colors['input_border'],
                           highlightthickness=1)
            entry.grid(row=row, column=col, padx=3, pady=3, sticky='ew')

            name_label = tk.Label(entry, text=name,
                                 font=("Helvetica", 11),
                                 fg=self.colors['text_white'],
                                 bg=self.colors['input_bg'],
                                 anchor='w')
            name_label.pack(side='left', padx=12, pady=8)

            status_label = tk.Label(entry, text="WAITING",
                                   font=("Helvetica", 9),
                                   fg=self.colors['text_gray'],
                                   bg=self.colors['input_bg'],
                                   anchor='e')
            status_label.pack(side='right', padx=12)

            self.strategy_labels[name] = status_label

    def create_gemini_logs_section(self, parent):
        """Create Gemini LLM logs section"""
        section = tk.Frame(parent, bg=self.colors['panel_bg'],
                          highlightbackground=self.colors['panel_border'],
                          highlightthickness=1)
        section.pack(fill='both', expand=True, pady=(10, 0))

        # Header
        header = tk.Label(section, text="GEMINI LLM LOGS",
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

        self.gemini_log_text = tk.Text(log_frame,
                                       bg=self.colors['input_bg'],
                                       fg=self.colors['text_gray'],
                                       font=("Courier", 9),
                                       wrap='word',
                                       yscrollcommand=scrollbar.set,
                                       state='disabled',
                                       relief='flat')
        self.gemini_log_text.pack(fill='both', expand=True)
        scrollbar.config(command=self.gemini_log_text.yview)

        # Configure text tags for color coding
        self.gemini_log_text.tag_config('gemini', foreground=self.colors['blue'])
        self.gemini_log_text.tag_config('reasoning', foreground='#a78bfa')  # Purple for reasoning
        self.gemini_log_text.tag_config('info', foreground=self.colors['text_gray'])
        self.gemini_log_text.tag_config('success', foreground=self.colors['green_light'])
        self.gemini_log_text.tag_config('warning', foreground=self.colors['yellow'])
        self.gemini_log_text.tag_config('error', foreground=self.colors['red'])

        self.add_gemini_log("Waiting for Gemini LLM activity...")

    def create_positions_section(self, parent):
        """Create active positions section - compact height"""
        section = tk.Frame(parent, bg=self.colors['panel_bg'],
                          highlightbackground=self.colors['panel_border'],
                          highlightthickness=1,
                          height=150)  # Fixed compact height - half of strategy list
        section.pack(fill='x', pady=(0, 10))
        section.pack_propagate(False)  # Prevent expansion

        # Header
        header = tk.Label(section, text="ACTIVE POSITIONS",
                         font=("Helvetica", 10, "bold"),
                         fg=self.colors['text_gray'],
                         bg=self.colors['panel_bg'],
                         anchor='w')
        header.pack(fill='x', padx=20, pady=(15, 10))

        # Position container
        self.position_container = tk.Frame(section, bg=self.colors['panel_bg'])
        self.position_container.pack(fill='both', expand=True, padx=20, pady=(0, 15))

        # Initially empty
        self.no_position_label = tk.Label(self.position_container,
                                          text="No active positions",
                                          font=("Helvetica", 10),
                                          fg=self.colors['text_dim'],
                                          bg=self.colors['panel_bg'])
        self.no_position_label.pack(pady=10)

    def create_copy_trading_stats_section(self, parent):
        """Create copy trading statistics and controls section"""
        # Load copy trading configuration from persistent storage
        copy_config = load_copy_trading_config()
        enabled = copy_config.get('enabled', False)
        followers = copy_config.get('followers', [])
        active_followers = [f for f in followers if f.get('enabled', True)]

        # Always show the section (even if disabled)
        section = tk.Frame(parent, bg=self.colors['panel_bg'],
                          highlightbackground=self.colors['panel_border'],
                          highlightthickness=1)
        section.pack(fill='x', pady=(0, 10))

        # Header with icon
        header_frame = tk.Frame(section, bg=self.colors['panel_bg'])
        header_frame.pack(fill='x', padx=20, pady=(15, 10))

        header_icon = tk.Label(header_frame, text="📋",
                              font=("Helvetica", 10),
                              bg=self.colors['panel_bg'],
                              fg=self.colors['green'] if enabled else self.colors['text_dim'])
        header_icon.pack(side='left', padx=(0, 5))

        header = tk.Label(header_frame, text="COPY TRADING",
                         font=("Helvetica", 10, "bold"),
                         fg=self.colors['text_gray'],
                         bg=self.colors['panel_bg'],
                         anchor='w')
        header.pack(side='left')

        # Stats container
        stats_container = tk.Frame(section, bg=self.colors['panel_bg'])
        stats_container.pack(fill='both', expand=True, padx=20, pady=(0, 15))

        if enabled and active_followers:
            # Show follower count
            follower_frame = tk.Frame(stats_container, bg=self.colors['input_bg'],
                                     highlightbackground=self.colors['input_border'],
                                     highlightthickness=1)
            follower_frame.pack(fill='x', pady=3)

            follower_row = tk.Frame(follower_frame, bg=self.colors['input_bg'])
            follower_row.pack(fill='x', padx=15, pady=8)

            follower_label = tk.Label(follower_row,
                                     text=f"{len(active_followers)} Follower{'s' if len(active_followers) > 1 else ''}",
                                     font=("Helvetica", 11, "bold"),
                                     fg=self.colors['text_white'],
                                     bg=self.colors['input_bg'])
            follower_label.pack(side='left')

            status_label = tk.Label(follower_row,
                                   text="ACTIVE",
                                   font=("Helvetica", 10, "bold"),
                                   fg=self.colors['green'],
                                   bg=self.colors['input_bg'])
            status_label.pack(side='right')

            # Store reference for updates
            self.copy_trading_status_label = status_label
            self.copy_trading_follower_label = follower_label
        else:
            # Show status message
            if enabled and not active_followers:
                status_text = "⚠️ No followers configured"
                status_color = self.colors['yellow']
            else:
                status_text = "Copy trading is disabled"
                status_color = self.colors['text_dim']

            status_label = tk.Label(stats_container,
                                    text=status_text,
                                    font=("Helvetica", 10),
                                    fg=status_color,
                                    bg=self.colors['panel_bg'])
            status_label.pack(pady=5)

            self.copy_trading_status_label = None
            self.copy_trading_follower_label = None

        # Control buttons
        button_frame = tk.Frame(stats_container, bg=self.colors['panel_bg'])
        button_frame.pack(fill='x', pady=(10, 0))

        # Enable/Disable button
        if enabled:
            btn_text = "🛑 Disable Copy Trading"
            btn_color = self.colors['red']
            btn_command = self.disable_copy_trading
        else:
            btn_text = "✅ Enable Copy Trading"
            btn_color = self.colors['green']
            btn_command = self.enable_copy_trading

        toggle_btn = tk.Button(button_frame,
                              text=btn_text,
                              font=("Helvetica", 9, "bold"),
                              bg=btn_color,
                              fg='#FFFFFF',
                              activebackground=btn_color,
                              activeforeground='#FFFFFF',
                              relief='flat',
                              cursor='hand2',
                              command=btn_command,
                              padx=10,
                              pady=6)
        toggle_btn.pack(side='left', padx=(0, 5))

        # Select Accounts button
        select_btn = tk.Button(button_frame,
                              text="🔧 Select Follower Accounts",
                              font=("Helvetica", 9, "bold"),
                              bg=self.colors['blue'],
                              fg='#FFFFFF',
                              activebackground='#4a90e2',
                              activeforeground='#FFFFFF',
                              relief='flat',
                              cursor='hand2',
                              command=self.select_copy_trading_accounts,
                              padx=10,
                              pady=6)
        select_btn.pack(side='left')

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
            ("Rejection", "IDLE"),
            ("Chop", "IDLE"),
            ("Extension", "IDLE"),
            ("Volatility", "IDLE"),
            ("Trend", "IDLE"),
            ("Impulse", "IDLE"),
            ("HTF FVG", "IDLE"),
            ("Bank Level", "IDLE"),
            ("Memory S/R", "IDLE"),
            ("News", "IDLE"),
            ("Structure", "IDLE"),
            ("Loss Block", "IDLE"),
        ]

        self.filter_labels = {}
        row, col = 0, 0
        for name, status in filters:
            self.create_filter_box(grid, row, col, name, status)
            col += 1
            if col >= 4:
                col = 0
                row += 1

    def create_filter_box(self, parent, row, col, name, status):
        """Create individual filter indicator"""
        # Determine initial background color
        if status in ["PASS", "SAFE"]:
            bg_color = '#1a3d2e'  # Dark green
            status_color = self.colors['green_light']
        elif status in ["BLOCK", "FAIL"]:
            bg_color = '#3d1a1a'  # Dark red
            status_color = self.colors['red']
        else:  # IDLE or neutral
            bg_color = self.colors['input_bg']  # Neutral gray
            status_color = self.colors['text_gray']

        box = tk.Frame(parent, bg=bg_color,
                      highlightbackground=self.colors['input_border'],
                      highlightthickness=1)
        box.grid(row=row, column=col, padx=4, pady=4, sticky='nsew')

        name_label = tk.Label(box, text=name,
                             font=("Helvetica", 10, "bold"),
                             fg=self.colors['text_white'],
                             bg=bg_color)
        name_label.pack(pady=(12, 4))

        status_label = tk.Label(box, text=f"[{status}]",
                               font=("Helvetica", 9, "bold"),
                               fg=status_color,
                               bg=bg_color)
        status_label.pack(pady=(4, 12))

        # Store both the box frame, labels, and status label for updates
        self.filter_labels[name] = {
            'box': box,
            'name_label': name_label,
            'status_label': status_label
        }

    def create_event_log(self, parent):
        """Create live event log section"""
        section = tk.Frame(parent, bg=self.colors['panel_bg'],
                          highlightbackground=self.colors['panel_border'],
                          highlightthickness=1)
        section.pack(fill='x', pady=(0, 10))  # Don't expand, fixed height
        section.pack_propagate(True)

        # Header
        header = tk.Label(section, text="LIVE EVENT LOG",
                         font=("Helvetica", 10, "bold"),
                         fg=self.colors['text_gray'],
                         bg=self.colors['panel_bg'],
                         anchor='w')
        header.pack(fill='x', padx=20, pady=(15, 10))

        # Log text widget
        log_frame = tk.Frame(section, bg=self.colors['panel_bg'])
        log_frame.pack(fill='x', expand=False, padx=20, pady=(0, 15))

        scrollbar = tk.Scrollbar(log_frame)
        scrollbar.pack(side='right', fill='y')

        self.log_text = tk.Text(log_frame,
                               bg=self.colors['input_bg'],
                               fg=self.colors['text_gray'],
                               font=("Courier", 9),
                               wrap='word',
                               height=10,  # Fixed height - half of original
                               yscrollcommand=scrollbar.set,
                               state='disabled',
                               relief='flat')
        self.log_text.pack(fill='both', expand=False)
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

    def add_market_log(self, message):
        """Add entry to market context log with color coding"""
        def update():
            self.market_log.config(state='normal')

            # Determine tag based on content
            tag = 'info'
            if '🕒 Session' in message or 'SESSION HANDOVER' in message or 'SESSION CHANGE' in message:
                tag = 'session'
            elif '🎯 REJECTION' in message or 'REJECTION' in message:
                tag = 'rejection'
            elif '🏦' in message or 'BANK' in message or 'ORB' in message:
                tag = 'bank'
            elif '⚠️' in message or 'WARNING' in message or 'BLOCK' in message:
                tag = 'warning'
            elif '🌊 DRIFT' in message or 'CALIBRATION' in message or 'Calibrated' in message:
                tag = 'drift'

            self.market_log.insert('end', message + '\n', tag)
            self.market_log.see('end')
            self.market_log.config(state='disabled')

        if threading.current_thread() != threading.main_thread():
            self.root.after(0, update)
        else:
            update()

    def add_gemini_log(self, message):
        """Add entry to Gemini LLM log with color coding"""
        def update():
            self.gemini_log_text.config(state='normal')

            # Determine tag based on content (priority order matters)
            tag = 'info'
            if 'ERROR' in message.upper() or 'FAILED' in message.upper():
                tag = 'error'
            elif 'WARNING' in message.upper() or 'CAUTION' in message.upper():
                tag = 'warning'
            elif 'SUCCESS' in message.upper() or 'APPROVED' in message.upper():
                tag = 'success'
            elif any(kw in message.upper() for kw in ['REASONING', 'THINK', 'THOUGHT', 'RATIONALE', 'DECISION', 'CONCLUSION', 'INFERENCE']):
                tag = 'reasoning'
            elif 'GEMINI' in message.upper() or 'LLM' in message.upper():
                tag = 'gemini'

            self.gemini_log_text.insert('end', message + '\n', tag)
            self.gemini_log_text.see('end')
            self.gemini_log_text.config(state='disabled')

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

    def toggle_pause(self):
        """Toggle pause/play state for log analysis"""
        self.paused = not self.paused

        if self.paused:
            # Paused - enable play, disable pause
            self.pause_btn.config(state='disabled')
            self.play_btn.config(state='normal')
            self.add_log("⏸ Monitoring PAUSED - logs frozen for analysis")
        else:
            # Playing - enable pause, disable play
            self.pause_btn.config(state='normal')
            self.play_btn.config(state='disabled')
            self.add_log("▶ Monitoring RESUMED")

    def enable_copy_trading(self):
        """Enable copy trading and prompt for account selection"""
        import tkinter.messagebox as messagebox

        result = messagebox.askyesno(
            "Enable Copy Trading",
            "Do you want to enable copy trading?\n\n"
            "This will allow trades on your main account to be automatically copied to follower accounts."
        )

        if not result:
            return

        # Prompt to select follower accounts
        self.select_copy_trading_accounts(enable_after_selection=True)

    def disable_copy_trading(self):
        """Disable copy trading"""
        import tkinter.messagebox as messagebox

        result = messagebox.askyesno(
            "Disable Copy Trading",
            "Are you sure you want to disable copy trading?\n\n"
            "Trades will no longer be copied to follower accounts."
        )

        if result:
            # Disable using persistent configuration
            if disable_ct():
                self.add_log("🛑 Copy trading disabled")
                # Refresh the dashboard to update UI
                self.show_dashboard()
            else:
                messagebox.showerror("Error", "Failed to disable copy trading")

    def select_copy_trading_accounts(self, enable_after_selection=False):
        """Open account selection dialog for copy trading followers"""
        import tkinter.messagebox as messagebox
        from copy_trading_setup import setup_copy_trading_from_accounts

        if not self.session:
            messagebox.showerror("Error", "Not authenticated. Please login first.")
            return

        # Create a new window for account selection
        dialog = tk.Toplevel(self.root)
        dialog.title("Select Follower Accounts")
        dialog.geometry("600x500")
        dialog.configure(bg=self.colors['bg_dark'])
        dialog.transient(self.root)
        dialog.grab_set()

        # Title
        title = tk.Label(dialog,
                        text="Select Follower Accounts",
                        font=("Helvetica", 16, "bold"),
                        fg=self.colors['text_white'],
                        bg=self.colors['bg_dark'])
        title.pack(pady=20)

        # Instructions
        instructions = tk.Label(dialog,
                               text="Select accounts that will copy trades from your leader account:",
                               font=("Helvetica", 10),
                               fg=self.colors['text_gray'],
                               bg=self.colors['bg_dark'])
        instructions.pack(pady=(0, 10))

        # Fetch accounts
        from account_selector import AccountSelector
        selector = AccountSelector(self.session)
        accounts = selector.fetch_accounts()

        if not accounts:
            messagebox.showerror("Error", "Could not fetch accounts")
            dialog.destroy()
            return

        # Create scrollable frame for accounts
        canvas = tk.Canvas(dialog, bg=self.colors['bg_dark'], highlightthickness=0)
        scrollbar = tk.Scrollbar(dialog, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=self.colors['bg_dark'])

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Store selected accounts and size ratios
        selected_accounts = {}

        # Create checkboxes for each account
        for account in accounts:
            account_id = account.get('id')
            account_name = account.get('name', 'Unknown')

            frame = tk.Frame(scrollable_frame, bg=self.colors['panel_bg'],
                           highlightbackground=self.colors['panel_border'],
                           highlightthickness=1)
            frame.pack(fill='x', padx=20, pady=5)

            var = tk.BooleanVar()
            ratio_var = tk.DoubleVar(value=1.0)

            cb = tk.Checkbutton(frame,
                              text=f"{account_name} ({account_id})",
                              variable=var,
                              font=("Helvetica", 10),
                              fg=self.colors['text_white'],
                              bg=self.colors['panel_bg'],
                              selectcolor=self.colors['input_bg'],
                              activebackground=self.colors['panel_bg'],
                              activeforeground=self.colors['text_white'])
            cb.pack(side='left', padx=10, pady=10)

            # Size ratio input
            ratio_label = tk.Label(frame,
                                  text="Size Ratio:",
                                  font=("Helvetica", 9),
                                  fg=self.colors['text_gray'],
                                  bg=self.colors['panel_bg'])
            ratio_label.pack(side='left', padx=(10, 5))

            ratio_entry = tk.Entry(frame,
                                  textvariable=ratio_var,
                                  font=("Helvetica", 9),
                                  bg=self.colors['input_bg'],
                                  fg=self.colors['text_white'],
                                  width=8)
            ratio_entry.pack(side='left', padx=5)

            selected_accounts[account_id] = {
                'name': account_name,
                'var': var,
                'ratio_var': ratio_var
            }

        canvas.pack(side="left", fill="both", expand=True, padx=20)
        scrollbar.pack(side="right", fill="y")

        # Buttons
        button_frame = tk.Frame(dialog, bg=self.colors['bg_dark'])
        button_frame.pack(pady=20)

        def save_selection():
            # Get selected account IDs and ratios
            follower_ids = []
            ratios = []

            for acc_id, data in selected_accounts.items():
                if data['var'].get():
                    follower_ids.append(acc_id)
                    ratios.append(data['ratio_var'].get())

            if not follower_ids:
                messagebox.showwarning("No Selection", "Please select at least one follower account")
                return

            # Setup copy trading
            copy_trader = setup_copy_trading_from_accounts(self.session, follower_ids, ratios)

            if copy_trader:
                # Save to persistent configuration
                from copy_trading_setup import save_copy_trading_config
                success = save_copy_trading_config(copy_trader.follower_accounts, enabled=enable_after_selection)

                if success:
                    self.add_log(f"✅ Copy trading configured with {len(follower_ids)} follower(s)")
                    dialog.destroy()
                    # Refresh dashboard
                    self.show_dashboard()
                else:
                    messagebox.showerror("Error", "Failed to save copy trading configuration")
            else:
                messagebox.showerror("Error", "Failed to setup copy trading")

        save_btn = tk.Button(button_frame,
                           text="Save Configuration",
                           font=("Helvetica", 11, "bold"),
                           bg=self.colors['green'],
                           fg='#FFFFFF',
                           activebackground=self.colors['green_light'],
                           activeforeground='#FFFFFF',
                           relief='flat',
                           cursor='hand2',
                           command=save_selection,
                           padx=20,
                           pady=10)
        save_btn.pack(side='left', padx=5)

        cancel_btn = tk.Button(button_frame,
                             text="Cancel",
                             font=("Helvetica", 11, "bold"),
                             bg=self.colors['red'],
                             fg='#FFFFFF',
                             activebackground='#dc2626',
                             activeforeground='#FFFFFF',
                             relief='flat',
                             cursor='hand2',
                             command=dialog.destroy,
                             padx=20,
                             pady=10)
        cancel_btn.pack(side='left', padx=5)

    def start_monitoring(self):
        """Start real-time monitoring of bot log and API"""
        self.monitoring_active = True

        # Thread 1: Monitor log file
        def monitor_log():
            while self.monitoring_active:
                if not self.paused:
                    self.tail_log_file()
                time.sleep(0.5)

        # Thread 2: Monitor positions
        def monitor_positions():
            while self.monitoring_active:
                if not self.paused:
                    self.fetch_position()
                time.sleep(2)

        # Thread 3: Monitor market price
        def monitor_price():
            while self.monitoring_active:
                if not self.paused:
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

        # Determine if this is event log content (heartbeats, trades, blocks, API)
        is_event_log = any(keyword in line for keyword in [
            '💓', 'heartbeat', 'Heartbeat',  # Heartbeats
            '🚀 SENDING ORDER', 'ORDER PLACED', 'FILL', 'EXECUTED',  # Trade execution
            '⛔ Signal ignored', 'BLOCKED', 'FILTER_CHECK', '✗ BLOCK',  # Blocked trades
            'API', 'returned no bars', 'No bars', 'rate limit',  # API logs
            'Trade closed', 'Position', 'P&L',  # Trade results
            'STRATEGY_SIGNAL',  # Strategy signals
            '📋 Copying trade', '📊 Leader trade', '📈 Copy trade complete',  # Copy trading
            'Follower', 'Copy trade', 'Circuit breaker', 'Copy close'  # Copy trading events
        ])

        # Determine if this is market context
        is_market_context = any(keyword in line for keyword in [
            '🕒 Session', 'SESSION CHANGE', 'SESSION change', 'SESSION HANDOVER',
            '🎯 REJECTION', 'REJECTION_DETECTED', 'REJECTION CONFIRMED',
            '🏦', 'ORB', 'BANK', 'Prev PM', 'Prev Session',
            '📅 New day', 'QUARTER CHANGE',
            '⚠️ CHOP', '⚠️ PENALTY', 'CEILING', 'FLOOR',
            '🧠 HTF FVG Memory', 'Bar:', 'Price:',  # HTF FVG Memory goes here
            '📈 CONTINUATION', '📉 CONTINUATION',
            '🔁 BIAS FLIP', '🔄 QUARTER',
            'Backfill Complete', 'ExtFilter',
            '🌊 DRIFT DETECTED', 'CALIBRATION COMPLETE', 'DynamicChop',
            'Calibrated', 'Threshold',
            '⚙️ STARTUP CALIBRATION',  # Startup calibration goes here
            '🌊 UPDATING TREND FILTER',  # Trend filter updates go here
            '📉 Tightening Risk', '📈 Tightening Risk',  # Risk management goes here
        ])

        # Determine if this is Gemini LLM activity (case-insensitive)
        line_upper = line.upper()

        # First, check if it's explicitly NOT a Gemini log (exclusions)
        is_excluded = any(keyword in line_upper for keyword in [
            'HTF FVG MEMORY',  # This uses 🧠 but is not Gemini
            'STARTUP CALIBRATION',  # This uses "Analyzing" but is not Gemini
        ])

        # Only check Gemini keywords if not excluded
        is_gemini_log = False
        if not is_excluded:
            is_gemini_log = any(keyword in line_upper for keyword in [
                # Explicit Gemini mentions (most specific)
                'GEMINI', 'GEMINI 3.0', 'GEMINI MULTIPLIER', 'GEMINI CONTEXT',
                '🧠 GEMINI', 'GEMINI OPTIMIZED',
                'UPDATED GEMINI MULTIPLIER', '[DYNAMICCHOP] UPDATED GEMINI',

                # Neural Network initialization (Gemini-specific)
                'INITIALIZING NEURAL NETWORK ARRAY',

                # Gemini output markers
                '📝 REASONING:', '🎯 NEW MULTIPLIERS',
                '🌊 TREND REGIME:',

                # Session analysis by Gemini
                'ANALYZING SESSION-ALIGNED CONTEXT',
            ])

        # Route to appropriate log
        if is_gemini_log:
            self.add_gemini_log(line)
        elif is_event_log:
            self.add_log(line)
        elif is_market_context:
            self.add_market_log(line)
        # Otherwise, don't display (filter out noise)

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
        """Update filter status and background color"""
        def update():
            if name in self.filter_labels:
                filter_info = self.filter_labels[name]

                # Determine background color based on status
                if status in ["PASS", "SAFE"]:
                    bg_color = '#1a3d2e'  # Dark green
                    status_color = self.colors['green_light']
                elif status in ["BLOCK", "FAIL"]:
                    bg_color = '#3d1a1a'  # Dark red
                    status_color = self.colors['red']
                else:  # IDLE or neutral
                    bg_color = self.colors['input_bg']  # Neutral gray
                    status_color = self.colors['text_gray']

                # Update all components
                filter_info['box'].config(bg=bg_color)
                filter_info['name_label'].config(bg=bg_color)
                filter_info['status_label'].config(text=f"[{status}]", fg=status_color, bg=bg_color)

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
