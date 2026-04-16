"""
Beautiful Account Selection UI for JULIE Trading Bot
Provides an interactive interface to select which account(s) to monitor
"""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt
from rich.text import Text
from rich.align import Align
import requests
from typing import List, Dict, Optional, Union
from config import CONFIG


class AccountSelector:
    """
    Interactive account selection UI using Rich library
    Allows users to select a specific account or monitor all accounts
    """

    def __init__(self, session: requests.Session):
        self.console = Console()
        self.session = session
        self.base_url = CONFIG['REST_BASE_URL']

    def fetch_accounts(self) -> List[Dict]:
        """Fetch all active accounts from the API"""
        url = f"{self.base_url}/api/Account/search"
        payload = {"onlyActiveAccounts": True}

        try:
            resp = self.session.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

            if 'accounts' in data and len(data['accounts']) > 0:
                return data['accounts']
            else:
                return []
        except Exception as e:
            self.console.print(f"[red]Error fetching accounts: {e}[/red]")
            return []

    def display_welcome(self):
        """Display welcome banner"""
        welcome_text = Text()
        welcome_text.append("JULIE TRADING BOT\n", style="bold cyan")
        welcome_text.append("Account Selection", style="bold white")

        welcome_panel = Panel(
            Align.center(welcome_text),
            border_style="cyan",
            padding=(1, 2)
        )

        self.console.print()
        self.console.print(welcome_panel)
        self.console.print()

    def display_accounts_table(self, accounts: List[Dict]) -> Table:
        """Create and display a beautiful table of accounts"""
        table = Table(
            title="Available Accounts",
            title_style="bold magenta",
            show_header=True,
            header_style="bold cyan",
            border_style="blue",
            expand=False
        )

        table.add_column("Option", justify="center", style="bold yellow", width=8)
        table.add_column("Account Name", style="cyan", width=30)
        table.add_column("Account ID", style="dim", width=25)
        table.add_column("Status", justify="center", style="green", width=10)

        # Add "Monitor All" option
        table.add_row(
            "[bold green]0[/bold green]",
            "[bold green]MONITOR ALL ACCOUNTS[/bold green]",
            "[dim]All active accounts[/dim]",
            "[bold green]●[/bold green]"
        )

        # Add individual accounts
        for idx, account in enumerate(accounts, start=1):
            account_name = str(account.get('name', 'Unknown'))
            account_id = str(account.get('id', 'N/A'))

            table.add_row(
                f"[bold]{idx}[/bold]",
                account_name,
                account_id,
                "[green]●[/green]"
            )

        return table

    def select_account(self) -> Union[str, List[str], None]:
        """
        Interactive account selection

        Returns:
            - str: Single account ID if user selects a specific account
            - List[str]: List of all account IDs if user selects "Monitor All"
            - None: If no accounts available or error occurs
        """
        # Display welcome
        self.display_welcome()

        # Fetch accounts
        self.console.print("[yellow]Fetching active accounts...[/yellow]")
        accounts = self.fetch_accounts()

        if not accounts:
            self.console.print("[red]No active accounts found.[/red]")
            return None

        self.console.print(f"[green]Found {len(accounts)} active account(s)[/green]\n")

        # Display accounts table
        table = self.display_accounts_table(accounts)
        self.console.print(table)
        self.console.print()

        # Prompt for selection
        while True:
            try:
                self.console.print("[bold cyan]Select an option:[/bold cyan]")
                self.console.print("  • Enter [bold green]0[/bold green] to monitor ALL accounts")
                self.console.print(f"  • Enter [bold yellow]1-{len(accounts)}[/bold yellow] to monitor a specific account")
                self.console.print()

                choice = IntPrompt.ask(
                    "[cyan]Your choice[/cyan]",
                    default=0,
                    show_default=True
                )

                # Validate choice
                if choice == 0:
                    # Monitor all accounts
                    self.console.print()
                    self.console.print(Panel(
                        f"[bold green]✓ Monitoring ALL {len(accounts)} accounts[/bold green]",
                        border_style="green"
                    ))
                    self.console.print()
                    return [acc.get('id') for acc in accounts]

                elif 1 <= choice <= len(accounts):
                    # Monitor specific account
                    selected_account = accounts[choice - 1]
                    account_name = selected_account.get('name', 'Unknown')
                    account_id = selected_account.get('id')

                    self.console.print()
                    self.console.print(Panel(
                        f"[bold green]✓ Selected: {account_name}[/bold green]\n"
                        f"[dim]Account ID: {account_id}[/dim]",
                        border_style="green"
                    ))
                    self.console.print()
                    return account_id

                else:
                    self.console.print(f"[red]Invalid choice. Please enter 0-{len(accounts)}[/red]\n")

            except KeyboardInterrupt:
                self.console.print("\n[yellow]Selection cancelled by user[/yellow]")
                return None
            except Exception as e:
                self.console.print(f"[red]Error: {e}. Please try again.[/red]\n")


def select_account_interactive(session: requests.Session) -> Union[str, List[str], None]:
    """
    Standalone function to select account(s) interactively

    Args:
        session: Authenticated requests session with Authorization header

    Returns:
        - str: Single account ID
        - List[str]: List of account IDs (if "Monitor All" selected)
        - None: If cancelled or error
    """
    selector = AccountSelector(session)
    return selector.select_account()


# For backwards compatibility with existing code
def fetch_accounts_with_selection(session: requests.Session, base_url: str) -> Optional[str]:
    """
    Fetch accounts and prompt for selection (returns single account ID)
    This maintains compatibility with the original fetch_accounts behavior

    Returns:
        str: Single account ID (if user selects specific account)
        None: If user selects "Monitor All" or error occurs
    """
    selector = AccountSelector(session)
    result = selector.select_account()

    # If result is a list (Monitor All), return None to indicate multi-account mode
    if isinstance(result, list):
        return None

    return result
