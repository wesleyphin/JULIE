"""
Example: How to integrate Copy Trading into JULIE

This file demonstrates the minimal changes needed to add copy trading
to your existing JULIE bot.
"""

import logging
from copy_trader import create_copy_trader_from_config
from client import ProjectXClient
from config import CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)


def run_bot_with_copy_trading():
    """
    Example bot setup with copy trading integration.

    This shows the 3-step integration:
    1. Initialize copy trader from config
    2. Pass copy trader to ProjectXClient
    3. Trade normally (copying happens automatically)
    """

    # ============================================================
    # STEP 1: Initialize Copy Trader (if enabled in config)
    # ============================================================
    copy_trader = None
    if CONFIG.get('COPY_TRADING', {}).get('enabled', False):
        logging.info("📋 Copy trading is enabled, initializing...")
        copy_trader = create_copy_trader_from_config(CONFIG['COPY_TRADING'])

        if copy_trader:
            logging.info(f"✅ Copy trader initialized with {len(copy_trader.follower_accounts)} followers")
            # Log follower account IDs
            for follower in copy_trader.follower_accounts:
                logging.info(f"  → Follower: {follower.account_id} (ratio: {follower.size_ratio})")
        else:
            logging.warning("⚠️ Copy trader enabled in config but failed to initialize")
    else:
        logging.info("Copy trading is disabled in config")

    # ============================================================
    # STEP 2: Initialize Client with Copy Trader
    # ============================================================
    client = ProjectXClient(copy_trader=copy_trader)

    # Authenticate
    client.login()
    if not client.token:
        logging.error("Failed to authenticate")
        return

    # Fetch account and contract info
    accounts = client.fetch_accounts()
    if not accounts:
        logging.error("No accounts found")
        return

    contracts = client.fetch_contracts()
    if not contracts:
        logging.error("No contracts found")
        return

    logging.info(f"✅ Leader account authenticated: {client.account_id}")
    logging.info(f"✅ Trading contract: {client.contract_id}")

    # ============================================================
    # STEP 3: Trade Normally (Copying Happens Automatically)
    # ============================================================

    # Example: Place a test trade
    test_signal = {
        'side': 'LONG',
        'tp_dist': 6.0,   # 6 point take profit
        'sl_dist': 4.0,   # 4 point stop loss
        'size': 2,        # 2 contracts
        'strategy': 'TestStrategy'
    }

    current_price = 4500.0  # Example price

    logging.info("=" * 60)
    logging.info("PLACING TEST ORDER")
    logging.info("=" * 60)

    # This single call will:
    # 1. Place order on leader account
    # 2. Copy to all enabled follower accounts (if copy_trader is not None)
    # 3. Apply size ratios
    # 4. Log all results
    result = client.place_order(test_signal, current_price)

    if result and result.get('success'):
        logging.info("✅ Leader order placed successfully")

        if copy_trader:
            # Check copy trading stats
            stats = copy_trader.get_stats()
            logging.info("=" * 60)
            logging.info("COPY TRADING STATISTICS")
            logging.info("=" * 60)
            logging.info(f"Total copies: {stats['total_copies']}")
            logging.info(f"Successful: {stats['successful_copies']}")
            logging.info(f"Failed: {stats['failed_copies']}")
            logging.info(f"Circuit breaker trips: {stats['circuit_breaker_trips']}")
    else:
        logging.error("❌ Leader order failed")

    # ============================================================
    # STEP 4: Monitor Position (Optional)
    # ============================================================

    # Check position on leader account
    position = client.get_position()
    logging.info(f"Leader position: {position}")

    # If you want to manually check follower positions, you can access follower clients:
    if copy_trader:
        logging.info("=" * 60)
        logging.info("FOLLOWER POSITIONS")
        logging.info("=" * 60)
        for acc_id, follower_client in copy_trader.follower_clients.items():
            pos = follower_client.get_position()
            logging.info(f"Follower {acc_id}: {pos}")

    # ============================================================
    # STEP 5: Close Position (Optional)
    # ============================================================

    # When you close the leader position, followers will automatically close too
    # logging.info("Closing position...")
    # close_result = client.close_position(position)
    # if close_result:
    #     logging.info("✅ Positions closed on leader and all followers")


if __name__ == "__main__":
    # Before running this example:
    # 1. Configure follower accounts in config.py
    # 2. Set COPY_TRADING['enabled'] = True
    # 3. Ensure all credentials are correct

    # WARNING: This will place REAL orders if credentials are configured!
    # Comment out the actual order placement if you just want to test initialization

    print("\n" + "=" * 60)
    print("COPY TRADING INTEGRATION EXAMPLE")
    print("=" * 60)
    print("\nThis example demonstrates how to integrate copy trading into JULIE.")
    print("Review the code and configuration before running.")
    print("\nTo run: python copy_trading_example.py")
    print("=" * 60 + "\n")

    # Uncomment to run:
    # run_bot_with_copy_trading()

    print("\nExample script loaded successfully.")
    print("Uncomment the last line to execute the example.\n")
