"""
Copy Trader Module for JULIE
============================

Implements a Leader-Follower architecture for copying trades across multiple accounts.

Features:
- Multi-account trade replication
- Rate limiting and circuit breaker protection
- Flexible position sizing ratios
- Comprehensive error handling and logging
- Integration with existing ProjectXClient

Author: Wes (with Claude)
"""

import time
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import deque
from threading import Lock
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

from client import ProjectXClient

logger = logging.getLogger(__name__)


@dataclass
class FollowerAccount:
    """Configuration for a follower account."""
    username: str
    api_key: str
    account_id: str
    contract_id: str
    size_ratio: float = 1.0  # Ratio relative to leader (1.0 = same size, 0.5 = half size)
    enabled: bool = True


class CircuitBreaker:
    """
    Circuit breaker to prevent runaway order placement.

    Monitors order rate and trips if too many orders are placed too quickly.
    """

    def __init__(self, max_orders: int = 5, time_window: float = 1.0):
        """
        Args:
            max_orders: Maximum orders allowed in time window
            time_window: Time window in seconds
        """
        self.max_orders = max_orders
        self.time_window = time_window
        self.order_times = deque(maxlen=max_orders)
        self.is_tripped = False
        self.lock = Lock()

    def record_order(self) -> bool:
        """
        Record an order attempt and check if circuit should trip.

        Returns:
            True if order is allowed, False if circuit is tripped
        """
        with self.lock:
            if self.is_tripped:
                logger.error("🚨 CIRCUIT BREAKER TRIPPED - Copy trading disabled")
                return False

            current_time = time.time()
            self.order_times.append(current_time)

            # Check if we've exceeded rate limit
            if len(self.order_times) >= self.max_orders:
                time_span = current_time - self.order_times[0]
                if time_span < self.time_window:
                    self.is_tripped = True
                    logger.critical(
                        f"🚨 CIRCUIT BREAKER TRIPPED: {self.max_orders} orders "
                        f"in {time_span:.2f}s (limit: {self.time_window}s)"
                    )
                    return False

            return True

    def reset(self):
        """Manually reset the circuit breaker."""
        with self.lock:
            self.is_tripped = False
            self.order_times.clear()
            logger.info("✅ Circuit breaker reset")


class RateLimiter:
    """
    Rate limiter to stay within TopStepX API limits.

    Limits: 200 requests / 60 seconds for most endpoints
    """

    def __init__(self, max_requests: int = 180, time_window: float = 60.0):
        """
        Args:
            max_requests: Maximum requests allowed (conservative buffer)
            time_window: Time window in seconds
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.request_times = deque()
        self.lock = Lock()

    def wait_if_needed(self):
        """Wait if necessary to stay within rate limits."""
        with self.lock:
            current_time = time.time()

            # Remove old requests outside the time window
            while self.request_times and current_time - self.request_times[0] > self.time_window:
                self.request_times.popleft()

            # Check if we need to wait
            if len(self.request_times) >= self.max_requests:
                oldest_request = self.request_times[0]
                wait_time = self.time_window - (current_time - oldest_request)
                if wait_time > 0:
                    logger.warning(f"⏳ Rate limit approaching, waiting {wait_time:.2f}s")
                    time.sleep(wait_time)
                    # Clean up again after waiting
                    current_time = time.time()
                    while self.request_times and current_time - self.request_times[0] > self.time_window:
                        self.request_times.popleft()

            # Record this request
            self.request_times.append(current_time)


class CopyTrader:
    """
    Leader-Follower copy trading system.

    Replicates trades from a leader account to multiple follower accounts
    with proper rate limiting and safety mechanisms.
    """

    def __init__(
        self,
        follower_accounts: List[FollowerAccount],
        enable_circuit_breaker: bool = True,
        enable_rate_limiting: bool = True,
        max_workers: int = 3  # Limit concurrent API calls
    ):
        """
        Args:
            follower_accounts: List of follower account configurations
            enable_circuit_breaker: Enable circuit breaker protection
            enable_rate_limiting: Enable rate limiting
            max_workers: Maximum concurrent API workers
        """
        self.follower_accounts = [acc for acc in follower_accounts if acc.enabled]
        self.follower_clients: Dict[str, ProjectXClient] = {}

        # Safety mechanisms
        self.circuit_breaker = CircuitBreaker() if enable_circuit_breaker else None
        self.rate_limiter = RateLimiter() if enable_rate_limiting else None
        self.max_workers = max_workers

        # Statistics
        self.stats = {
            'total_copies': 0,
            'successful_copies': 0,
            'failed_copies': 0,
            'circuit_breaker_trips': 0
        }

        logger.info(f"📋 CopyTrader initialized with {len(self.follower_accounts)} follower accounts")

    def authenticate_followers(self) -> Dict[str, bool]:
        """
        Authenticate all follower accounts.

        Returns:
            Dict mapping account_id to authentication success status
        """
        results = {}

        for follower in self.follower_accounts:
            try:
                logger.info(f"🔐 Authenticating follower: {follower.username}")

                client = ProjectXClient(
                    username=follower.username,
                    api_key=follower.api_key,
                    account_id=follower.account_id,
                    contract_id=follower.contract_id
                )

                # Test authentication
                if client.session.headers.get('Authorization'):
                    self.follower_clients[follower.account_id] = client
                    results[follower.account_id] = True
                    logger.info(f"✅ Authenticated: {follower.username} ({follower.account_id})")
                else:
                    results[follower.account_id] = False
                    logger.error(f"❌ Authentication failed: {follower.username}")

            except Exception as e:
                results[follower.account_id] = False
                logger.error(f"❌ Error authenticating {follower.username}: {e}")

        successful = sum(1 for v in results.values() if v)
        logger.info(f"🔐 Authentication complete: {successful}/{len(results)} successful")

        return results

    def copy_trade(
        self,
        signal: Dict,
        leader_price: float,
        leader_account_id: str,
        dry_run: bool = False
    ) -> Dict[str, Tuple[bool, Optional[str]]]:
        """
        Copy a trade from leader to all follower accounts.

        Args:
            signal: Trade signal dict (same format as ProjectXClient.place_order)
            leader_price: Price at which leader executed
            leader_account_id: Leader account ID (for logging)
            dry_run: If True, simulate without placing actual orders

        Returns:
            Dict mapping follower account_id to (success, error_message)
        """
        # Circuit breaker check
        if self.circuit_breaker and not self.circuit_breaker.record_order():
            self.stats['circuit_breaker_trips'] += 1
            return {acc.account_id: (False, "Circuit breaker tripped")
                    for acc in self.follower_accounts}

        logger.info(
            f"📊 Leader trade: {signal['strategy']} {signal['side']} "
            f"{signal.get('size', 5)} contracts @ {leader_price:.2f}"
        )

        results = {}

        if dry_run:
            logger.info("🧪 DRY RUN MODE - No actual orders will be placed")
            for follower in self.follower_accounts:
                results[follower.account_id] = (True, "Dry run success")
            return results

        # Place orders on follower accounts in parallel (with limited workers)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_account = {
                executor.submit(
                    self._place_follower_order,
                    follower,
                    signal,
                    leader_price
                ): follower.account_id
                for follower in self.follower_accounts
            }

            for future in as_completed(future_to_account):
                account_id = future_to_account[future]
                try:
                    success, error_msg = future.result(timeout=10.0)
                    results[account_id] = (success, error_msg)

                    if success:
                        self.stats['successful_copies'] += 1
                    else:
                        self.stats['failed_copies'] += 1

                except Exception as e:
                    results[account_id] = (False, f"Exception: {str(e)}")
                    self.stats['failed_copies'] += 1
                    logger.error(f"❌ Unexpected error for {account_id}: {e}")

        self.stats['total_copies'] += len(results)

        # Log summary
        successful = sum(1 for success, _ in results.values() if success)
        logger.info(
            f"📈 Copy trade complete: {successful}/{len(results)} followers succeeded"
        )

        return results

    def _place_follower_order(
        self,
        follower: FollowerAccount,
        signal: Dict,
        leader_price: float
    ) -> Tuple[bool, Optional[str]]:
        """
        Place an order on a single follower account.

        Args:
            follower: Follower account configuration
            signal: Trade signal
            leader_price: Leader execution price

        Returns:
            (success, error_message)
        """
        # Rate limiting
        if self.rate_limiter:
            self.rate_limiter.wait_if_needed()

        # Get client
        client = self.follower_clients.get(follower.account_id)
        if not client:
            return (False, "Client not authenticated")

        try:
            # Adjust size based on follower's ratio
            adjusted_signal = signal.copy()
            original_size = signal.get('size', 5)
            adjusted_size = max(1, int(original_size * follower.size_ratio))
            adjusted_signal['size'] = adjusted_size

            logger.info(
                f"  → Follower {follower.account_id}: {signal['side']} "
                f"{adjusted_size} contracts (ratio: {follower.size_ratio})"
            )

            # Place order using existing client method
            # Note: Using market order to ensure fill (avoid leader/follower desync)
            result = client.place_order(adjusted_signal, leader_price)

            if result and result.get('success'):
                logger.info(f"  ✅ {follower.account_id}: Order placed successfully")
                return (True, None)
            else:
                error_msg = result.get('message', 'Unknown error') if result else 'No response'
                logger.error(f"  ❌ {follower.account_id}: {error_msg}")
                return (False, error_msg)

        except Exception as e:
            logger.error(f"  ❌ {follower.account_id}: Exception - {e}")
            return (False, str(e))

    def copy_position_close(
        self,
        leader_account_id: str,
        dry_run: bool = False
    ) -> Dict[str, Tuple[bool, Optional[str]]]:
        """
        Copy a position close from leader to all followers.

        Args:
            leader_account_id: Leader account ID
            dry_run: If True, simulate without closing positions

        Returns:
            Dict mapping follower account_id to (success, error_message)
        """
        logger.info(f"🔄 Closing positions on all follower accounts")

        results = {}

        if dry_run:
            logger.info("🧪 DRY RUN MODE - No actual closes will be executed")
            for follower in self.follower_accounts:
                results[follower.account_id] = (True, "Dry run success")
            return results

        for follower in self.follower_accounts:
            if self.rate_limiter:
                self.rate_limiter.wait_if_needed()

            client = self.follower_clients.get(follower.account_id)
            if not client:
                results[follower.account_id] = (False, "Client not authenticated")
                continue

            try:
                result = client.close_position()
                if result:
                    results[follower.account_id] = (True, None)
                    logger.info(f"  ✅ {follower.account_id}: Position closed")
                else:
                    results[follower.account_id] = (False, "Close failed")
                    logger.error(f"  ❌ {follower.account_id}: Close failed")
            except Exception as e:
                results[follower.account_id] = (False, str(e))
                logger.error(f"  ❌ {follower.account_id}: {e}")

        return results

    def get_stats(self) -> Dict:
        """Get copy trading statistics."""
        return self.stats.copy()

    def reset_circuit_breaker(self):
        """Manually reset the circuit breaker."""
        if self.circuit_breaker:
            self.circuit_breaker.reset()


# Convenience function for quick setup
def create_copy_trader_from_config(config: Dict) -> Optional[CopyTrader]:
    """
    Create a CopyTrader instance from configuration dict.

    Expected config format:
    {
        'enabled': True,
        'followers': [
            {
                'username': 'user1',
                'api_key': 'key1',
                'account_id': 'acc1',
                'contract_id': 'contract1',
                'size_ratio': 1.0,
                'enabled': True
            },
            ...
        ]
    }

    Returns:
        CopyTrader instance or None if disabled/invalid
    """
    if not config.get('enabled', False):
        logger.info("Copy trading is disabled in config")
        return None

    followers_config = config.get('followers', [])
    if not followers_config:
        logger.warning("No follower accounts configured")
        return None

    followers = []
    for f in followers_config:
        try:
            follower = FollowerAccount(
                username=f['username'],
                api_key=f['api_key'],
                account_id=f['account_id'],
                contract_id=f['contract_id'],
                size_ratio=f.get('size_ratio', 1.0),
                enabled=f.get('enabled', True)
            )
            followers.append(follower)
        except KeyError as e:
            logger.error(f"Invalid follower config, missing key: {e}")
            continue

    if not followers:
        logger.error("No valid follower accounts found in config")
        return None

    copy_trader = CopyTrader(followers)
    copy_trader.authenticate_followers()

    return copy_trader


if __name__ == "__main__":
    # Example usage / testing
    logging.basicConfig(level=logging.INFO)

    # Test with mock follower accounts
    test_followers = [
        FollowerAccount(
            username="test_follower_1",
            api_key="test_key_1",
            account_id="test_acc_1",
            contract_id="test_contract_1",
            size_ratio=1.0,
            enabled=True
        ),
        FollowerAccount(
            username="test_follower_2",
            api_key="test_key_2",
            account_id="test_acc_2",
            contract_id="test_contract_2",
            size_ratio=0.5,  # Half size of leader
            enabled=True
        )
    ]

    copy_trader = CopyTrader(test_followers)

    # Test signal
    test_signal = {
        'side': 'LONG',
        'tp_dist': 6.0,
        'sl_dist': 4.0,
        'size': 5,
        'strategy': 'TestStrategy'
    }

    # Dry run test
    print("\n=== DRY RUN TEST ===")
    results = copy_trader.copy_trade(
        signal=test_signal,
        leader_price=4500.0,
        leader_account_id="leader_acc",
        dry_run=True
    )

    print(f"\nResults: {results}")
    print(f"Stats: {copy_trader.get_stats()}")
