"""
Terminal UI for JULIE Trading Bot
Displays real-time signals, positions, and market context
"""

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.text import Text
from rich.align import Align
from datetime import datetime
from collections import deque
import threading
import time


class TerminalUI:
    """
    Real-time terminal UI for JULIE trading bot
    Displays signals, open trades, filter status, and market context
    """

    def __init__(self):
        self.console = Console()
        self.layout = Layout()

        # Data stores
        self.current_position = None
        self.recent_signals = deque(maxlen=10)
        self.filter_status = {}
        self.market_context = {
            'session': 'UNKNOWN',
            'price': 0.0,
            'symbol': 'MES',
            'bias': 'NEUTRAL',
            'volatility': 'NORMAL'
        }
        self.event_log = deque(maxlen=15)
        self.strategy_signals = {}
        self.account_info = {
            'account_id': '',
            'pnl': 0.0,
            'daily_pnl': 0.0
        }

        # UI control
        self.running = False
        self.live = None
        self.lock = threading.Lock()

        # Initialize layout
        self._setup_layout()

    def _setup_layout(self):
        """Setup the terminal layout structure"""
        self.layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3)
        )

        self.layout["main"].split_row(
            Layout(name="left", ratio=2),
            Layout(name="right", ratio=1)
        )

        self.layout["left"].split_column(
            Layout(name="position", size=12),
            Layout(name="signals", ratio=1),
            Layout(name="events", size=10)
        )

        self.layout["right"].split_column(
            Layout(name="market", size=12),
            Layout(name="filters", ratio=1)
        )

    def update_position(self, position_data):
        """Update current position information"""
        with self.lock:
            self.current_position = position_data

    def add_signal(self, strategy_name, signal_data):
        """Add a new signal to the display"""
        with self.lock:
            timestamp = datetime.now().strftime("%H:%M:%S")
            signal_entry = {
                'time': timestamp,
                'strategy': strategy_name,
                'side': signal_data.get('side', 'N/A'),
                'tp': signal_data.get('tp_dist', 0.0),
                'sl': signal_data.get('sl_dist', 0.0),
                'status': signal_data.get('status', 'PENDING')
            }
            self.recent_signals.appendleft(signal_entry)

            # Update strategy-specific signal tracking
            self.strategy_signals[strategy_name] = signal_entry

    def update_filter_status(self, filter_name, passed, reason=""):
        """Update filter status"""
        with self.lock:
            self.filter_status[filter_name] = {
                'passed': passed,
                'reason': reason
            }

    def update_market_context(self, context_data):
        """Update market context information"""
        with self.lock:
            self.market_context.update(context_data)

    def add_event(self, event_type, message):
        """Add event to the log"""
        with self.lock:
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.event_log.appendleft(f"[{timestamp}] {event_type}: {message}")

    def update_account_info(self, account_data):
        """Update account information"""
        with self.lock:
            self.account_info.update(account_data)

    def _render_header(self):
        """Render the header section"""
        title = Text("JULIE - Advanced MES Futures Trading Bot", style="bold cyan")
        subtitle = Text(f"Session: {self.market_context['session']} | Symbol: {self.market_context['symbol']} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", style="dim")
        header_text = Text.assemble(title, "\n", subtitle)
        return Panel(Align.center(header_text), style="bold blue")

    def _render_position(self):
        """Render current position panel"""
        table = Table(show_header=True, header_style="bold magenta", expand=True)
        table.add_column("Status", style="cyan", width=12)
        table.add_column("Side", justify="center", width=8)
        table.add_column("Entry", justify="right", width=10)
        table.add_column("Current", justify="right", width=10)
        table.add_column("P&L", justify="right", width=10)
        table.add_column("TP", justify="right", width=10)
        table.add_column("SL", justify="right", width=10)
        table.add_column("Bars", justify="right", width=6)

        if self.current_position and self.current_position.get('active', False):
            pos = self.current_position
            side = pos.get('side', 'N/A')
            entry = pos.get('entry_price', 0.0)
            current = self.market_context.get('price', entry)

            # Calculate P&L
            if side == 'LONG':
                pnl = (current - entry) * 5  # $5 per point for MES
                pnl_color = "green" if pnl >= 0 else "red"
            elif side == 'SHORT':
                pnl = (entry - current) * 5
                pnl_color = "green" if pnl >= 0 else "red"
            else:
                pnl = 0.0
                pnl_color = "white"

            tp_price = pos.get('tp_price', 0.0)
            sl_price = pos.get('sl_price', 0.0)
            bars_held = pos.get('bars_held', 0)
            strategy = pos.get('strategy', 'Unknown')

            side_color = "green" if side == 'LONG' else "red"

            table.add_row(
                f"[yellow]ACTIVE[/yellow]\n{strategy}",
                f"[{side_color}]{side}[/{side_color}]",
                f"{entry:.2f}",
                f"{current:.2f}",
                f"[{pnl_color}]${pnl:+.2f}[/{pnl_color}]",
                f"{tp_price:.2f}",
                f"{sl_price:.2f}",
                f"{bars_held}"
            )
        else:
            table.add_row(
                "[dim]NO POSITION[/dim]",
                "-",
                "-",
                f"{self.market_context.get('price', 0.0):.2f}",
                "[dim]$0.00[/dim]",
                "-",
                "-",
                "-"
            )

        return Panel(table, title="[bold]Current Position", border_style="green")

    def _render_signals(self):
        """Render recent signals panel"""
        table = Table(show_header=True, header_style="bold yellow", expand=True)
        table.add_column("Time", style="dim", width=8)
        table.add_column("Strategy", width=18)
        table.add_column("Side", justify="center", width=6)
        table.add_column("TP", justify="right", width=6)
        table.add_column("SL", justify="right", width=6)
        table.add_column("Status", width=12)

        for signal in list(self.recent_signals)[:8]:
            side = signal['side']
            side_color = "green" if side == 'LONG' else "red" if side == 'SHORT' else "white"
            status = signal['status']
            status_color = "green" if status == 'EXECUTED' else "yellow" if status == 'PENDING' else "red"

            table.add_row(
                signal['time'],
                signal['strategy'][:18],
                f"[{side_color}]{side}[/{side_color}]",
                f"{signal['tp']:.1f}",
                f"{signal['sl']:.1f}",
                f"[{status_color}]{status}[/{status_color}]"
            )

        if len(self.recent_signals) == 0:
            table.add_row("[dim]No signals yet...", "", "", "", "", "")

        return Panel(table, title="[bold]Recent Signals", border_style="yellow")

    def _render_market(self):
        """Render market context panel"""
        table = Table(show_header=False, expand=True, box=None)
        table.add_column("Key", style="cyan", width=15)
        table.add_column("Value", style="bold white")

        session = self.market_context.get('session', 'UNKNOWN')
        session_color = {
            'ASIA': 'blue',
            'LONDON': 'magenta',
            'NY_AM': 'green',
            'NY_PM': 'yellow'
        }.get(session, 'white')

        bias = self.market_context.get('bias', 'NEUTRAL')
        bias_color = "green" if bias == 'LONG' else "red" if bias == 'SHORT' else "white"

        volatility = self.market_context.get('volatility', 'NORMAL')
        vol_color = "red" if volatility == 'HIGH' else "yellow" if volatility == 'LOW' else "white"

        table.add_row("Session:", f"[{session_color}]{session}[/{session_color}]")
        table.add_row("Current Price:", f"{self.market_context.get('price', 0.0):.2f}")
        table.add_row("Bias:", f"[{bias_color}]{bias}[/{bias_color}]")
        table.add_row("Volatility:", f"[{vol_color}]{volatility}[/{vol_color}]")
        table.add_row("", "")
        table.add_row("Account ID:", f"{self.account_info.get('account_id', 'N/A')[:12]}")
        table.add_row("Daily P&L:", f"${self.account_info.get('daily_pnl', 0.0):+.2f}")

        return Panel(table, title="[bold]Market Context", border_style="cyan")

    def _render_filters(self):
        """Render filter status panel"""
        table = Table(show_header=True, header_style="bold blue", expand=True)
        table.add_column("Filter", width=20)
        table.add_column("Status", justify="center", width=8)

        filter_names = [
            'Rejection',
            'HTF FVG',
            'Chop',
            'Extension',
            'Structure',
            'Bank Level'
        ]

        for filter_name in filter_names:
            status = self.filter_status.get(filter_name, {})
            passed = status.get('passed', None)

            if passed is None:
                status_text = "[dim]IDLE[/dim]"
            elif passed:
                status_text = "[green]PASS ✓[/green]"
            else:
                status_text = "[red]BLOCK ✗[/red]"

            table.add_row(filter_name, status_text)

        return Panel(table, title="[bold]Filter Status", border_style="blue")

    def _render_events(self):
        """Render event log panel"""
        events_text = "\n".join(list(self.event_log)[:8]) if self.event_log else "[dim]No events yet...[/dim]"
        return Panel(events_text, title="[bold]Event Log", border_style="white", height=10)

    def _render_footer(self):
        """Render the footer section"""
        strategies_active = len(self.strategy_signals)
        signals_count = len(self.recent_signals)

        footer_text = Text.assemble(
            ("Strategies Active: ", "dim"),
            (f"{strategies_active}", "bold green"),
            ("  |  ", "dim"),
            ("Total Signals: ", "dim"),
            (f"{signals_count}", "bold yellow"),
            ("  |  ", "dim"),
            ("Press Ctrl+C to exit", "dim italic")
        )
        return Panel(Align.center(footer_text), style="dim")

    def render(self):
        """Render the complete UI"""
        with self.lock:
            self.layout["header"].update(self._render_header())
            self.layout["position"].update(self._render_position())
            self.layout["signals"].update(self._render_signals())
            self.layout["market"].update(self._render_market())
            self.layout["filters"].update(self._render_filters())
            self.layout["events"].update(self._render_events())
            self.layout["footer"].update(self._render_footer())

        return self.layout

    def start(self, refresh_rate=1.0):
        """Start the live UI display"""
        self.running = True
        self.live = Live(self.render(), console=self.console, refresh_per_second=refresh_rate)
        self.live.start()

        # Start background refresh thread
        def refresh_loop():
            while self.running:
                try:
                    self.live.update(self.render())
                    time.sleep(1.0 / refresh_rate)
                except Exception as e:
                    self.add_event("ERROR", f"UI refresh error: {str(e)}")

        self.refresh_thread = threading.Thread(target=refresh_loop, daemon=True)
        self.refresh_thread.start()

        self.add_event("SYSTEM", "Terminal UI started")

    def stop(self):
        """Stop the live UI display"""
        self.running = False
        if self.live:
            self.live.stop()
        self.add_event("SYSTEM", "Terminal UI stopped")

    def is_running(self):
        """Check if UI is currently running"""
        return self.running


# Singleton instance for easy access
_ui_instance = None

def get_ui():
    """Get or create the UI singleton instance"""
    global _ui_instance
    if _ui_instance is None:
        _ui_instance = TerminalUI()
    return _ui_instance
