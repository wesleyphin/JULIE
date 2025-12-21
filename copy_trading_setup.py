"""
Copy Trading Interactive Setup Module
======================================

Provides interactive setup for copy trading from both CLI and UI.
Allows selecting follower accounts using the beautiful account selector.

Author: Wes (with Claude)
"""

import logging
import json
from typing import List, Dict, Optional, Tuple
from rich.console import Console
from rich.prompt import Prompt, Confirm, FloatPrompt
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from account_selector import AccountSelector
from copy_trader import FollowerAccount, CopyTrader, create_copy_trader_from_config
from client import ProjectXClient
from config import CONFIG

logger = logging.getLogger(__name__)
console = Console()


def setup_copy_trading_interactive(session) -> Optional[CopyTrader]:
    """
    Interactive CLI setup for copy trading.

    Allows user to:
    1. Choose to enable/disable copy trading
    2. Select follower accounts using the account selector
    3. Configure size ratios for each follower

    Args:
        session: Authenticated requests session

    Returns:
        CopyTrader instance if enabled, None if disabled
    """
    console.print()
    console.print(Panel(
        Text("Copy Trading Setup", justify="center", style="bold cyan"),
        border_style="cyan"
    ))
    console.print()

    # Ask if user wants to enable copy trading
    enable = Confirm.ask(
        "[cyan]Do you want to enable copy trading?[/cyan]",
        default=False
    )

    if not enable:
        console.print("[yellow]Copy trading disabled.[/yellow]")
        return None

    console.print()
    console.print("[green]✓ Copy trading enabled[/green]")
    console.print()
    console.print("[cyan]Now select follower accounts (accounts that will copy the leader's trades)[/cyan]")
    console.print()

    # Get follower accounts using the account selector
    follower_configs = []

    while True:
        # Use account selector to pick a follower account
        selector = AccountSelector(session)
        result = selector.select_account()

        if result is None:
            console.print("[yellow]Account selection cancelled[/yellow]")
            break

        # If user selected "Monitor All", that's not valid for copy trading
        if isinstance(result, list):
            console.print("[red]Cannot use 'Monitor All' for copy trading. Please select individual accounts.[/red]")
            console.print()
            if not Confirm.ask("[cyan]Select another follower account?[/cyan]", default=True):
                break
            continue

        # Result is a single account ID
        follower_account_id = result

        # Ask for additional follower configuration
        console.print()
        console.print("[cyan]Configure this follower account:[/cyan]")

        # Ask for size ratio
        size_ratio = FloatPrompt.ask(
            "[cyan]Size ratio (e.g., 1.0 = same size as leader, 0.5 = half size)[/cyan]",
            default=1.0
        )

        # We need username and api_key for each follower
        # For now, we'll use the same credentials from CONFIG
        # In a production system, you'd want separate credentials per follower
        follower_configs.append({
            'username': CONFIG['USERNAME'],  # In production, prompt for this
            'api_key': CONFIG['API_KEY'],    # In production, prompt for this
            'account_id': follower_account_id,
            'contract_id': None,  # Will be fetched during initialization
            'size_ratio': size_ratio,
            'enabled': True
        })

        console.print()
        console.print(f"[green]✓ Added follower account: {follower_account_id} (ratio: {size_ratio})[/green]")
        console.print()

        # Ask if they want to add more followers
        if not Confirm.ask("[cyan]Add another follower account?[/cyan]", default=False):
            break

    if not follower_configs:
        console.print("[yellow]No follower accounts configured. Copy trading will be disabled.[/yellow]")
        return None

    # Display summary
    console.print()
    console.print(Panel(
        f"[bold green]Copy Trading Configuration Complete[/bold green]\n\n"
        f"Followers: {len(follower_configs)} account(s)",
        border_style="green"
    ))

    # Create follower summary table
    table = Table(title="Follower Accounts", border_style="cyan")
    table.add_column("Account ID", style="cyan")
    table.add_column("Size Ratio", justify="right", style="yellow")
    table.add_column("Status", style="green")

    for fc in follower_configs:
        table.add_row(
            fc['account_id'],
            f"{fc['size_ratio']:.2f}x",
            "✓ Enabled" if fc['enabled'] else "Disabled"
        )

    console.print(table)
    console.print()

    # Now we need to authenticate these followers and get contract IDs
    console.print("[yellow]Authenticating follower accounts...[/yellow]")

    followers = []
    for fc in follower_configs:
        try:
            # Create a client for this follower to get contract ID
            follower_client = ProjectXClient()
            follower_client.login()  # Uses CONFIG credentials

            # Fetch contract for this account
            # We need to temporarily set the account_id to fetch the right contract
            follower_client.account_id = fc['account_id']
            contract_id = follower_client.fetch_contracts()

            if contract_id:
                fc['contract_id'] = contract_id
                follower = FollowerAccount(
                    username=fc['username'],
                    api_key=fc['api_key'],
                    account_id=fc['account_id'],
                    contract_id=fc['contract_id'],
                    size_ratio=fc['size_ratio'],
                    enabled=fc['enabled']
                )
                followers.append(follower)
                console.print(f"[green]✓ {fc['account_id']} authenticated[/green]")
            else:
                console.print(f"[red]✗ Failed to get contract for {fc['account_id']}[/red]")
        except Exception as e:
            console.print(f"[red]✗ Failed to authenticate {fc['account_id']}: {e}[/red]")

    if not followers:
        console.print("[red]No followers could be authenticated. Copy trading disabled.[/red]")
        return None

    # Create the CopyTrader instance
    copy_trader = CopyTrader(followers)

    console.print()
    console.print(f"[bold green]✓ Copy trading initialized with {len(followers)} follower(s)[/bold green]")
    console.print()

    return copy_trader


def setup_copy_trading_from_accounts(
    session,
    follower_account_ids: List[str],
    size_ratios: Optional[List[float]] = None
) -> Optional[CopyTrader]:
    """
    Setup copy trading with pre-selected account IDs (for UI use).

    Args:
        session: Authenticated requests session
        follower_account_ids: List of account IDs to use as followers
        size_ratios: Optional list of size ratios (default 1.0 for all)

    Returns:
        CopyTrader instance or None if setup fails
    """
    if not follower_account_ids:
        return None

    # Default all ratios to 1.0 if not provided
    if size_ratios is None:
        size_ratios = [1.0] * len(follower_account_ids)
    elif len(size_ratios) != len(follower_account_ids):
        logger.warning(f"Size ratio count mismatch. Using 1.0 for all.")
        size_ratios = [1.0] * len(follower_account_ids)

    followers = []

    for account_id, ratio in zip(follower_account_ids, size_ratios):
        try:
            # Create a client for this follower to get contract ID
            follower_client = ProjectXClient()
            follower_client.login()

            # Fetch contract for this account
            follower_client.account_id = account_id
            contract_id = follower_client.fetch_contracts()

            if contract_id:
                follower = FollowerAccount(
                    username=CONFIG['USERNAME'],
                    api_key=CONFIG['API_KEY'],
                    account_id=account_id,
                    contract_id=contract_id,
                    size_ratio=ratio,
                    enabled=True
                )
                followers.append(follower)
                logger.info(f"✓ Follower {account_id} configured (ratio: {ratio})")
            else:
                logger.error(f"Failed to get contract for {account_id}")
        except Exception as e:
            logger.error(f"Failed to setup follower {account_id}: {e}")

    if not followers:
        logger.error("No followers could be configured")
        return None

    copy_trader = CopyTrader(followers)
    logger.info(f"✓ Copy trading initialized with {len(followers)} follower(s)")

    return copy_trader


def get_copy_trading_status() -> Dict:
    """
    Get current copy trading status from CONFIG.

    Returns:
        Dict with keys: enabled, follower_count, follower_accounts
    """
    copy_config = CONFIG.get('COPY_TRADING', {})
    enabled = copy_config.get('enabled', False)
    followers = copy_config.get('followers', [])
    active_followers = [f for f in followers if f.get('enabled', True)]

    return {
        'enabled': enabled,
        'follower_count': len(active_followers),
        'follower_accounts': active_followers
    }


def save_copy_trading_config(followers: List[FollowerAccount], enabled: bool = True):
    """
    Save copy trading configuration to CONFIG (in-memory).

    Note: This updates the in-memory CONFIG. To persist, you'd need to
    write to config.py or a separate JSON file.

    Args:
        followers: List of FollowerAccount instances
        enabled: Whether copy trading is enabled
    """
    follower_configs = []
    for f in followers:
        follower_configs.append({
            'username': f.username,
            'api_key': f.api_key,
            'account_id': f.account_id,
            'contract_id': f.contract_id,
            'size_ratio': f.size_ratio,
            'enabled': f.enabled
        })

    CONFIG['COPY_TRADING'] = {
        'enabled': enabled,
        'followers': follower_configs
    }

    logger.info(f"Copy trading config updated: {len(follower_configs)} followers, enabled={enabled}")
