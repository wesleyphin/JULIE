"""
Enhanced Dashboard UI for Julie - Multi-Account Support
Displays all accounts with P&L, positions, and trading activity
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


class DashboardUI:
    """
    Enhanced dashboard UI for multi-account trading
    Shows all accounts, their P&L, positions, and signals
    """

    def __init__(self):
        self.console = Console()
        self.layout = Layout()

        # Data stores
        self.accounts = {}  # account_id -> account data
        self.positions = {}  # account_id -> position
        self.recent_signals = deque(maxlen=10)
        self.market_context = {
            'session': 'UNKNOWN',
            'price': 0.0,
            'symbol': 'MES'
        }
        self.event_log = deque(maxlen=12)

        # UI control
        self.running = False
        self.live = None
        self.lock = threading.Lock()

        # Initialize layout
        self._setup_layout()

    def _setup_layout(self):
        """Setup the dashboard layout"""
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
            Layout(name="accounts", size=15),  # Account overview
            Layout(name="signals", ratio=1),
            Layout(name="events", size=10)
        )

        self.layout["right"].split_column(
            Layout(name="market", size=10),
            Layout(name="summary", ratio=1)
        )

    def update_account(self, account_id: str, account_data: dict):
        """Update or add an account"""
        with self.lock:
            if account_id not in self.accounts:
                self.accounts[account_id] = {
                    'id': account_id,
                    'name': account_data.get('name', 'Unknown'),
                    'daily_pnl': 0.0,
                    'trades': 0,
                    'wins': 0,
                    'losses': 0,
                    'is_trading': account_data.get('is_trading', True),
                    'bot_active': account_data.get('bot_active', True)
                }
            # Always update with latest data
            self.accounts[account_id].update(account_data)

    def update_position(self, account_id: str, position: dict):
        """Update position for an account"""
        with self.lock:
            self.positions[account_id] = position

    def add_signal(self, strategy_name: str, signal_data: dict):
        """Add a trading signal"""
        with self.lock:
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.recent_signals.appendleft({
                'time': timestamp,
                'strategy': strategy_name,
                'side': signal_data.get('side', 'N/A'),
                'account': signal_data.get('account_name', 'Unknown')[:15],
                'status': signal_data.get('status', 'PENDING')
            })

    def update_market_context(self, context_data: dict):
        """Update market context"""
        with self.lock:
            self.market_context.update(context_data)

    def add_event(self, event_type: str, message: str):
        """Add event to log"""
        with self.lock:
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.event_log.appendleft(f"[{timestamp}] {event_type}: {message}")

    def _render_header(self):
        """Render header"""
        title = Text("Julie Dashboard - Multi-Account Trading Platform", style="bold cyan")
        subtitle = Text(f"Session: {self.market_context['session']} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", style="dim")
        header_text = Text.assemble(title, "\n", subtitle)
        return Panel(Align.center(header_text), style="bold blue")

    def _render_accounts(self):
        """Render all accounts overview"""
        table = Table(show_header=True, header_style="bold magenta", expand=True, title="Active Accounts")
        table.add_column("Account", style="cyan", width=15)
        table.add_column("Mode", justify="center", width=7)
        table.add_column("Pos", justify="center", width=6)
        table.add_column("Entry", justify="right", width=8)
        table.add_column("Current", justify="right", width=8)
        table.add_column("P&L", justify="right", width=10)
        table.add_column("W/L", justify="center", width=7)

        if not self.accounts:
            table.add_row("[dim]No accounts loaded...", "", "", "", "", "", "")
        else:
            for account_id, account in sorted(self.accounts.items(), key=lambda x: x[1].get('name', '')):
                position = self.positions.get(account_id)
                account_name = account.get('name', 'Unknown')[:15]
                daily_pnl = account.get('daily_pnl', 0.0)
                trades = account.get('trades', 0)
                wins = account.get('wins', 0)
                losses = account.get('losses', 0)
                is_trading = account.get('is_trading', True)

                # Mode indicator
                mode_indicator = "[green]TRADE[/green]" if is_trading else "[dim]WATCH[/dim]"

                # Position info
                if position and position.get('side'):
                    side = position['side']
                    entry = position.get('entry_price', 0.0)
                    current = self.market_context.get('price', entry)

                    # Calculate P&L
                    if side == 'LONG':
                        pnl = (current - entry) * position.get('size', 1) * 5.0
                    else:
                        pnl = (entry - current) * position.get('size', 1) * 5.0

                    side_color = "green" if side == 'LONG' else "red"
                    pnl_color = "green" if pnl >= 0 else "red"

                    table.add_row(
                        account_name,
                        mode_indicator,
                        f"[{side_color}]{side[:1]}[/{side_color}]",  # L or S
                        f"{entry:.2f}",
                        f"{current:.2f}",
                        f"[{pnl_color}]${pnl:+.2f}[/{pnl_color}]",
                        f"{wins}/{losses}"
                    )
                else:
                    # No position
                    pnl_color = "green" if daily_pnl >= 0 else "red"
                    table.add_row(
                        account_name,
                        mode_indicator,
                        "[dim]-[/dim]",
                        "-",
                        f"{self.market_context.get('price', 0.0):.2f}",
                        f"[{pnl_color}]${daily_pnl:+.2f}[/{pnl_color}]",
                        f"{wins}/{losses}" if trades > 0 else "-"
                    )

        return Panel(table, border_style="green")

    def _render_signals(self):
        """Render recent signals"""
        table = Table(show_header=True, header_style="bold yellow", expand=True)
        table.add_column("Time", style="dim", width=8)
        table.add_column("Account", width=15)
        table.add_column("Strategy", width=16)
        table.add_column("Side", justify="center", width=6)
        table.add_column("Status", width=10)

        for signal in list(self.recent_signals)[:6]:
            side = signal['side']
            side_color = "green" if side == 'LONG' else "red" if side == 'SHORT' else "white"
            status = signal['status']
            status_color = "green" if status == 'EXECUTED' else "yellow" if status == 'PENDING' else "red"

            table.add_row(
                signal['time'],
                signal['account'],
                signal['strategy'][:16],
                f"[{side_color}]{side}[/{side_color}]",
                f"[{status_color}]{status}[/{status_color}]"
            )

        if len(self.recent_signals) == 0:
            table.add_row("[dim]No signals yet...", "", "", "", "")

        return Panel(table, title="[bold]Recent Signals", border_style="yellow")

    def _render_market(self):
        """Render market context"""
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

        table.add_row("Session:", f"[{session_color}]{session}[/{session_color}]")
        table.add_row("Symbol:", f"{self.market_context.get('symbol', 'MES')}")
        table.add_row("Price:", f"{self.market_context.get('price', 0.0):.2f}")

        return Panel(table, title="[bold]Market", border_style="cyan")

    def _render_summary(self):
        """Render summary statistics"""
        total_accounts = len(self.accounts)
        total_pnl = sum(acc.get('daily_pnl', 0) for acc in self.accounts.values())
        total_trades = sum(acc.get('trades', 0) for acc in self.accounts.values())
        total_wins = sum(acc.get('wins', 0) for acc in self.accounts.values())
        total_losses = sum(acc.get('losses', 0) for acc in self.accounts.values())
        active_positions = sum(1 for p in self.positions.values() if p and p.get('side'))

        win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0
        pnl_color = "green" if total_pnl >= 0 else "red"

        table = Table(show_header=False, expand=True, box=None)
        table.add_column("Key", style="dim")
        table.add_column("Value", style="bold")

        table.add_row("Total Accounts:", f"{total_accounts}")
        table.add_row("Active Positions:", f"[yellow]{active_positions}[/yellow]")
        table.add_row("Daily P&L:", f"[{pnl_color}]${total_pnl:+.2f}[/{pnl_color}]")
        table.add_row("Total Trades:", f"{total_trades}")
        table.add_row("Win Rate:", f"[green]{win_rate:.1f}%[/green]" if win_rate >= 50 else f"[red]{win_rate:.1f}%[/red]")

        return Panel(table, title="[bold]Summary", border_style="blue")

    def _render_events(self):
        """Render event log"""
        events_text = "\n".join(list(self.event_log)[:7]) if self.event_log else "[dim]No events yet...[/dim]"
        return Panel(events_text, title="[bold]Event Log", border_style="white", height=10)

    def _render_footer(self):
        """Render footer"""
        footer_text = Text.assemble(
            ("Active Accounts: ", "dim"),
            (f"{len(self.accounts)}", "bold green"),
            ("  |  ", "dim"),
            ("Active Bots: ", "dim"),
            (f"{len([a for a in self.accounts.values() if a.get('bot_active', False)])}", "bold yellow"),
            ("  |  ", "dim"),
            ("Press Ctrl+C to exit", "dim italic")
        )
        return Panel(Align.center(footer_text), style="dim")

    def render(self):
        """Render the complete UI"""
        with self.lock:
            self.layout["header"].update(self._render_header())
            self.layout["accounts"].update(self._render_accounts())
            self.layout["signals"].update(self._render_signals())
            self.layout["market"].update(self._render_market())
            self.layout["summary"].update(self._render_summary())
            self.layout["events"].update(self._render_events())
            self.layout["footer"].update(self._render_footer())

        return self.layout

    def start(self, refresh_rate=1.0):
        """Start the live UI"""
        self.running = True
        self.live = Live(self.render(), console=self.console, refresh_per_second=refresh_rate)
        self.live.start()

        def refresh_loop():
            while self.running:
                try:
                    self.live.update(self.render())
                    time.sleep(1.0 / refresh_rate)
                except Exception as e:
                    self.add_event("ERROR", f"UI refresh error: {str(e)}")

        self.refresh_thread = threading.Thread(target=refresh_loop, daemon=True)
        self.refresh_thread.start()

        self.add_event("SYSTEM", "Dashboard UI started")

    def stop(self):
        """Stop the live UI"""
        self.running = False
        if self.live:
            self.live.stop()
        self.add_event("SYSTEM", "Dashboard UI stopped")


# Singleton instance
_dashboard_instance = None

def get_dashboard():
    """Get or create the dashboard singleton"""
    global _dashboard_instance
    if _dashboard_instance is None:
        _dashboard_instance = DashboardUI()
    return _dashboard_instance
