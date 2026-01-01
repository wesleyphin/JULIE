#!/usr/bin/env python3
"""
JULIE Tkinter UI - Professional Trading Dashboard
Pure log viewer that monitors topstep_live_bot.log (no API polling)
Includes Emergency Stop button to terminate the bot process
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

class ToolTip:
    """Simple ToolTip implementation for Tkinter widgets"""
    def __init__(self, widget, text, bg='#333333', fg='#ffffff'):
        self.widget = widget
        self.text = text
        self.bg = bg
        self.fg = fg
        self.tooltip = None
        self.widget.bind("<Enter>", self.enter)
        self.widget.bind("<Leave>", self.leave)

    def enter(self, event=None):
        x = self.widget.winfo_rootx() + 25
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5

        self.tooltip = tk.Toplevel(self.widget)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{x}+{y}")

        label = tk.Label(self.tooltip, text=self.text, justify='left',
                         background=self.bg, foreground=self.fg,
                         relief='solid', borderwidth=1,
                         font=("Helvetica", "9", "normal"))
        label.pack(ipadx=5, ipady=2)

    def leave(self, event=None):
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None

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
        self.current_price = 0.0
        self.logged_in = False
        self.monitoring_active = False
        self.paused = False  # Pause state for log analysis

        # Bot integration
        self.log_file = Path("topstep_live_bot.log")
        self.log_position = 0
        self.bot_process = None  # Track the julie001.py subprocess

        # Position tracking from logs
        self.active_position = None  # {'side': 'LONG/SHORT', 'entry_price': float, 'size': 1}

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
        dropdown.pack(pady=(0, 30), padx=60)

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
                stdin=subprocess.DEVNULL,
                cwd=str(bot_script.parent),
                bufsize=0  # Unbuffered (Binary compatible)
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

        # Controls (top right)
        controls_frame = tk.Frame(header, bg=self.colors['bg_dark'])
        controls_frame.pack(side='right')

        # Emergency Stop Button
        self.stop_btn = tk.Button(controls_frame,
                                   text="STOP",
                                   font=("Helvetica", 12, "bold"),
                                   bg=self.colors['red'],
                                   fg='#000000',
                                   activebackground='#dc2626',
                                   activeforeground='#000000',
                                   relief='flat',
                                   cursor='hand2',
                                   command=self.emergency_stop,
                                   padx=20,
                                   pady=8)
        self.stop_btn.pack(side='left', padx=5)
        ToolTip(self.stop_btn, "IMMEDIATELY terminate the bot process\nUse only in emergencies", bg='#3d1a1a')

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
        ToolTip(self.pause_btn, "Pause monitoring and freeze logs\nUseful for analyzing current state")

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
        ToolTip(self.play_btn, "Resume real-time monitoring")

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
        self.create_continuation_logs_section(strategy_frame)
        self.create_positions_section(right_col)
        self.create_filters_section(right_col)

        # Right panel - Market Context
        self.create_market_section(right_panel)

        # Start monitoring (log file only - no API polling)
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

        # Strategy entries - All strategies from JULIE
        strategies = [
            "Regime Adaptive",
            "Intraday Dip",
            "Confluence",
            "ICT Model",
            "ORB Strategy",
            "ML Physics",
            "Dynamic Engine 1",
            "SMT Divergence",
            "VIX Reversion",
            "Continuation"
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

    def create_continuation_logs_section(self, parent):
        """Create Continuation Strategy logs section"""
        section = tk.Frame(parent, bg=self.colors['panel_bg'],
                          highlightbackground=self.colors['panel_border'],
                          highlightthickness=1)
        section.pack(fill='both', expand=True, pady=(10, 0))

        # Header
        header = tk.Label(section, text="CONTINUATION STRATEGY LOGS",
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

        self.continuation_log_text = tk.Text(log_frame,
                                              bg=self.colors['input_bg'],
                                              fg=self.colors['text_gray'],
                                              font=("Courier", 9),
                                              wrap='word',
                                              yscrollcommand=scrollbar.set,
                                              state='disabled',
                                              relief='flat')
        self.continuation_log_text.pack(fill='both', expand=True)
        scrollbar.config(command=self.continuation_log_text.yview)

        # Configure text tags for color coding
        self.continuation_log_text.tag_config('rescue_success', foreground='#ec4899')  # Pink for rescue success
        self.continuation_log_text.tag_config('rescue_fail', foreground='#f97316')     # Orange for rescue failed
        self.continuation_log_text.tag_config('bias_block', foreground=self.colors['yellow'])  # Yellow for bias block
        self.continuation_log_text.tag_config('active', foreground=self.colors['green_light'])  # Green for active window
        self.continuation_log_text.tag_config('info', foreground=self.colors['text_gray'])

        self.add_continuation_log("Waiting for Continuation Strategy activity...")

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

        # Configure text tags for color coding new event types
        self.log_text.tag_config('normal', foreground=self.colors['text_gray'])
        self.log_text.tag_config('strategy_detail', foreground='#a5f3fc')  # Light Cyan for STRATEGY_METRIC
        self.log_text.tag_config('risk_alert', foreground='#fca5a5')       # Light Red for RISK_TELEMETRY
        self.log_text.tag_config('latency', foreground='#fbbf24')          # Amber for SLOW EXECUTION
        self.log_text.tag_config('candidate', foreground='#facc15')        # Yellow for CANDIDATE signals
        self.log_text.tag_config('execution', foreground=self.colors['green_light'])  # Green for EXECUTED signals
        self.log_text.tag_config('rescue', foreground='#ec4899')           # Pink/Magenta for RESCUE events
        self.log_text.tag_config('rescue_fail', foreground='#f97316')      # Orange for RESCUE FAILED

        self.add_log("Waiting for bot activity...")

    def add_log(self, message):
        """Add entry to event log with color coding"""
        def update():
            self.log_text.config(state='normal')

            # Determine tag based on message content (priority order matters)
            tag = 'normal'
            if 'RESCUE SUCCESSFUL' in message or '🚑' in message:
                tag = 'rescue'
            elif 'RESCUE FAILED' in message:
                tag = 'rescue_fail'
            elif 'STRATEGY_EXEC' in message or '✅ FAST EXEC' in message or '✅ STANDARD EXEC' in message:
                tag = 'execution'
            elif 'CANDIDATE' in message or 'status=CANDIDATE' in message:
                tag = 'candidate'
            elif 'STRATEGY_METRIC' in message:
                tag = 'strategy_detail'
            elif 'RISK_TELEMETRY' in message or 'RISK MONITOR' in message:
                tag = 'risk_alert'
            elif 'SLOW EXECUTION' in message or 'SYSTEM_LAG' in message:
                tag = 'latency'

            self.log_text.insert('end', message + '\n', tag)
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

    def add_continuation_log(self, message):
        """Add entry to Continuation Strategy log with color coding"""
        def update():
            self.continuation_log_text.config(state='normal')

            # Determine tag based on content (priority order matters)
            tag = 'info'
            if 'RESCUE SUCCESSFUL' in message or '🚑 RESCUE' in message:
                tag = 'rescue_success'
            elif 'RESCUE FAILED' in message or '❌ RESCUE FAILED' in message:
                tag = 'rescue_fail'
            elif '🛑 BIAS BLOCK' in message or 'Attempting Rescue' in message:
                tag = 'bias_block'
            elif 'Continuation_Q' in message or 'ACTIVE' in message.upper():
                tag = 'active'

            self.continuation_log_text.insert('end', message + '\n', tag)
            self.continuation_log_text.see('end')
            self.continuation_log_text.config(state='disabled')

        if threading.current_thread() != threading.main_thread():
            self.root.after(0, update)
        else:
            update()

    def emergency_stop(self):
        """Emergency stop - terminate the bot process immediately"""
        if self.bot_process:
            try:
                self.add_log("🛑 EMERGENCY STOP ACTIVATED - Terminating bot...")
                # Force kill the bot process
                self.bot_process.kill()
                self.bot_process.wait()
                self.bot_process = None
                self.add_log("✓ Bot process terminated")

                # Clear active position
                self.active_position = None
                self.update_position_display()
            except Exception as e:
                self.add_log(f"Error stopping bot: {e}")
        else:
            self.add_log("No active bot process to stop")

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

    def start_monitoring(self):
        """Start real-time monitoring of bot log file (no API polling)"""
        self.monitoring_active = True

        # Single thread: Monitor log file only
        def monitor_log():
            while self.monitoring_active:
                if not self.paused:
                    self.tail_log_file()
                time.sleep(0.5)

        threading.Thread(target=monitor_log, daemon=True).start()

        self.add_log("Log monitoring started (read-only mode)")

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

        # Extract price from heartbeat or bar messages
        # Format: "💓 Heartbeat: HH:MM:SS | Price: 5880.25" or "Bar: ... | Price: 5880.25"
        if ('💓' in line or 'Heartbeat' in line or 'Bar:' in line) and 'Price:' in line:
            price_match = re.search(r'Price:\s*(\d+\.?\d*)', line)
            if price_match:
                self.current_price = float(price_match.group(1))
                # Update position display with new price (for P&L calculation)
                if self.active_position:
                    self.update_position_display()

        # Extract position opening from "SENDING ORDER" messages
        # Format: "SENDING ORDER: LONG @ ~5880.25" or "SENDING ORDER: SHORT @ ~5880.25"
        if 'SENDING ORDER:' in line:
            order_match = re.search(r'SENDING ORDER:\s*(LONG|SHORT)\s*@\s*~?(\d+\.?\d*)', line)
            if order_match:
                side = order_match.group(1)
                entry_price = float(order_match.group(2))
                self.active_position = {
                    'side': side,
                    'entry_price': entry_price,
                    'size': 1  # Default size, actual size may vary
                }
                self.update_position_display()
                self.add_log(f"✅ Position opened: {side} @ {entry_price:.2f}")

        # Extract position closing from "Trade closed" messages
        # Format: "📊 Trade closed: LONG | Entry: 5880.25 | Exit: 5885.50 | PnL: 5.25 pts ($26.25)"
        if '📊 Trade closed' in line or 'Trade closed' in line:
            close_match = re.search(r'Trade closed.*?(LONG|SHORT).*?Entry:\s*(\d+\.?\d*).*?Exit:\s*(\d+\.?\d*).*?PnL:\s*([-\d\.]+)\s*pts.*?\$([-\d\.]+)', line)
            if close_match:
                side = close_match.group(1)
                entry = float(close_match.group(2))
                exit_price = float(close_match.group(3))
                pnl_pts = float(close_match.group(4))
                pnl_dollars = float(close_match.group(5))
                self.active_position = None
                self.update_position_display()
                self.add_log(f"✓ Position closed: {side} | Entry: {entry:.2f} | Exit: {exit_price:.2f} | P&L: ${pnl_dollars:.2f}")

        # Determine if this is event log content (heartbeats, trades, blocks, API)
        is_event_log = any(keyword in line for keyword in [
            '💓', 'heartbeat', 'Heartbeat',  # Heartbeats
            '🚀 SENDING ORDER', 'ORDER PLACED', 'FILL', 'EXECUTED',  # Trade execution
            '⛔ Signal ignored', 'BLOCKED', 'FILTER_CHECK', '✗ BLOCK',  # Blocked trades
            'API', 'returned no bars', 'No bars', 'rate limit',  # API logs
            'Trade closed', 'Position', 'P&L',  # Trade results
            'STRATEGY_SIGNAL',  # Strategy signals
            'Bar:', 'Price:',  # Live bar/price updates
            'missing sl_dist', 'missing tp_dist',  # Missing TP/SL warnings
            '🛑 BIAS BLOCK', '🚑 RESCUE', 'RESCUE_TRIGGER',  # Continuation rescue events
            'RESCUE SUCCESSFUL', 'RESCUE FAILED', 'Continuation',  # Continuation strategy logs
        ])

        # Determine if this is market context
        is_market_context = any(keyword in line for keyword in [
            '🕒 Session', 'SESSION CHANGE', 'SESSION change', 'SESSION HANDOVER',
            '🎯 REJECTION', 'REJECTION_DETECTED', 'REJECTION CONFIRMED',
            '🏦', 'ORB', 'BANK', 'Prev PM', 'Prev Session',
            '📅 New day', 'QUARTER CHANGE',
            '⚠️ CHOP', '⚠️ PENALTY', 'CEILING', 'FLOOR',
            '🧠 HTF FVG Memory',  # HTF FVG Memory goes here
            '📈 CONTINUATION', '📉 CONTINUATION',
            '🔁 BIAS FLIP', '🔄 QUARTER',
            'Backfill Complete', 'ExtFilter',
            '🌊 DRIFT DETECTED', 'CALIBRATION COMPLETE', 'DynamicChop',
            'Calibrated', 'Threshold',
            '⚙️ STARTUP CALIBRATION',  # Startup calibration goes here
            '🌊 UPDATING TREND FILTER',  # Trend filter updates go here
            '📉 Tightening Risk', '📈 Tightening Risk',  # Risk management goes here
            '🎄 HOLIDAY',  # Holiday multiplier detection
            '🎯 TARGET CALCULATION',  # Target calculation logs
            'Layer 1 - BASE', 'Layer 2 - GEMINI AI', 'Layer 3 - HOLIDAY',  # Target layers
            '📊 COMPOSITE EFFECT', '✅ FINAL TARGETS',  # Composite and final targets
        ])

        # Determine if this is a Continuation Strategy log
        is_continuation_log = any(keyword in line for keyword in [
            'CONTINUATION_BIAS_BLOCK', 'CONTINUATION_RESCUE', 'CONTINUATION_NO_MATCH',
            'CONTINUATION_WINDOW', 'CONTINUATION_SIGNAL',
            '🛑 BIAS BLOCK', '🚑 RESCUE SUCCESSFUL', '❌ RESCUE FAILED',
            'Continuation_Q', '📅 CONTINUATION WINDOW',
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
        if is_continuation_log:
            self.add_continuation_log(line)
            # Also add to event log for visibility
            if is_event_log:
                self.add_log(line)
        elif is_gemini_log:
            self.add_gemini_log(line)
        elif is_event_log:
            self.add_log(line)
        elif is_market_context:
            self.add_market_log(line)
        # Otherwise, don't display (filter out noise)

        # Reset strategies on new bar
        if "NEW BAR" in line or "Bar:" in line:
            for strategy in self.strategy_labels.keys():
                current = self.strategy_labels[strategy].cget('text')
                # Reset to WAITING unless it's currently EXECUTED
                if "EXECUTED" not in current:
                    self.update_strategy(strategy, "WAITING", self.colors['text_gray'])

        # Parse FILTER_CHECK logs with strategy names
        if "[FILTER_CHECK]" in line:
            # Extract strategy from the log details: strategy=StrategyName
            strategy_match = re.search(r'strategy=([^\s|]+)', line)
            if strategy_match:
                raw_strategy = strategy_match.group(1)

                # Map strategy names (handle both with and without spaces)
                strategy_map = {
                    "RegimeAdaptive": "Regime Adaptive",
                    "IntradayDip": "Intraday Dip",
                    "Confluence": "Confluence",
                    "ICTModel": "ICT Model",
                    "ORBStrategy": "ORB Strategy",
                    "MLPhysicsStrategy": "ML Physics",
                    "MLPhysics": "ML Physics",
                    "DynamicEngine": "Dynamic Engine 1",
                    "SMTStrategy": "SMT Divergence",
                    "VIXMeanReversion": "VIX Reversion",
                    "VIXReversion": "VIX Reversion"
                }

                display_name = strategy_map.get(raw_strategy, raw_strategy)

                # Check if it's a PASS or BLOCK
                if "✗ BLOCK" in line or "BLOCK" in line:
                    self.update_strategy(display_name, "BLOCKED", self.colors['red'])
                elif "✓ PASS" in line:
                    # Only update to CHECKING if not already BLOCKED or EXECUTED
                    if display_name in self.strategy_labels:
                        current = self.strategy_labels[display_name].cget('text')
                        if current not in ["EXECUTED", "BLOCKED"]:
                            self.update_strategy(display_name, "CHECKING", self.colors['yellow'])

        # Parse strategy signals
        for strategy in self.strategy_labels.keys():
            # Check if this log line is about this strategy
            if strategy in line or strategy.replace(" ", "") in line:

                # CASE 1: Execution (Winner) -> GREEN
                if "EXEC" in line or "EXECUTED" in line or "STRATEGY_EXEC" in line:
                    match = re.search(r'(LONG|SHORT).*?(\d+\.?\d*)', line)
                    if match:
                        side = match.group(1)
                        price = match.group(2)
                        self.update_strategy(strategy, f"EXECUTED {side} @ {price}", self.colors['green'])

                # CASE 2: Candidate/Signal (Generated but maybe not chosen) -> YELLOW
                # We look for "CANDIDATE" or "STRATEGY_SIGNAL"
                elif "CANDIDATE" in line or ("STRATEGY_SIGNAL" in line and "status=CANDIDATE" in line):
                    # Update text to show side
                    side_match = re.search(r'(LONG|SHORT)', line)
                    side = side_match.group(1) if side_match else "SIGNAL"

                    # Only update if not already executed (don't downgrade Green to Yellow)
                    current_status = self.strategy_labels[strategy].cget('text')
                    if "EXECUTED" not in current_status:
                        self.update_strategy(strategy, f"{side} FOUND", self.colors['yellow'])

                # CASE 3: Queued -> BLUE/CYAN
                elif "QUEUED" in line or ("STRATEGY_SIGNAL" in line and "status=QUEUED" in line):
                    self.update_strategy(strategy, "QUEUED", '#06b6d4')  # Cyan

                # CASE 4: Blocked -> RED
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

        # Parse Continuation Strategy / Rescue events
        if 'RESCUE SUCCESSFUL' in line or '🚑 RESCUE' in line:
            # Extract bias direction
            side_match = re.search(r'(LONG|SHORT)', line)
            side = side_match.group(1) if side_match else ""
            self.update_strategy("Continuation", f"RESCUED {side}", '#ec4899')  # Pink for rescue
            self.add_continuation_log(line)
        elif 'RESCUE FAILED' in line or '❌ RESCUE FAILED' in line:
            self.update_strategy("Continuation", "NO MATCH", '#f97316')  # Orange for failed
            self.add_continuation_log(line)
        elif '🛑 BIAS BLOCK' in line and 'Attempting Rescue' in line:
            self.update_strategy("Continuation", "CHECKING", self.colors['yellow'])
            self.add_continuation_log(line)
        elif 'RESCUE_TRIGGER' in line or 'Continuation Strategy' in line:
            # Log rescue trigger details
            self.add_continuation_log(line)
        elif 'Continuation_Q' in line:
            # Active continuation window detected
            self.update_strategy("Continuation", "ACTIVE", self.colors['green'])
            self.add_continuation_log(line)

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

    def update_position_display(self):
        """Update position display from log data (no API calls)"""
        def update():
            # Clear container
            for widget in self.position_container.winfo_children():
                widget.destroy()

            if self.active_position and self.current_price > 0:
                side = self.active_position['side']
                entry_price = self.active_position['entry_price']
                size = self.active_position.get('size', 1)

                # Calculate P&L (MES = $5 per point per contract)
                if side == "LONG":
                    pnl_points = self.current_price - entry_price
                else:
                    pnl_points = entry_price - self.current_price
                pnl = pnl_points * 5 * size

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

                price_label = tk.Label(pos_row, text=f"{entry_price:.2f}",
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

        if threading.current_thread() != threading.main_thread():
            self.root.after(0, update)
        else:
            update()


def main():
    root = tk.Tk()
    app = JulieUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
