# Copy Trading Setup Guide

## Overview

Copy trading has been enhanced with interactive setup from both JULIE001 bot and the TKinter UI. You can now easily enable/disable copy trading and select follower accounts using a beautiful interface.

## Features

### ✅ Enable from JULIE001 Bot
When you run `python julie001.py`, you'll be prompted to:
1. Enable or disable copy trading
2. Select follower accounts using the beautiful account selector
3. Configure size ratios for each follower account

### ✅ Enable from TKinter UI
The dashboard now includes:
1. **Copy Trading Status Panel** - Shows current status and follower count
2. **Enable/Disable Button** - Toggle copy trading on/off
3. **Select Follower Accounts Button** - Choose which accounts copy trades
4. **Size Ratio Configuration** - Set custom size ratios for each follower

## How to Use

### From JULIE001 Bot (CLI)

1. Run the bot:
   ```bash
   python julie001.py
   ```

2. After authentication, you'll see:
   ```
   ============================================================
   Copy Trading Setup

   Do you want to enable copy trading? (y/n):
   ```

3. Type `y` to enable, then select follower accounts using the interactive menu

4. Configure size ratios for each follower:
   - **1.0** = Same size as leader
   - **0.5** = Half size of leader
   - **2.0** = Double size of leader

### From TKinter UI

1. Launch the UI:
   ```bash
   python julie_tkinter_ui.py
   ```

2. Login to your account

3. In the dashboard, find the **COPY TRADING** section (right panel)

4. Click **"✅ Enable Copy Trading"** button

5. A dialog will open showing all available accounts:
   - Check the accounts you want as followers
   - Set size ratios for each (default: 1.0)
   - Click **"Save Configuration"**

6. To modify followers, click **"🔧 Select Follower Accounts"**

7. To disable, click **"🛑 Disable Copy Trading"**

## Account Selection Interface

Both CLI and UI use the same beautiful account selector that:
- Fetches all active accounts from your TopStepX account
- Displays account names and IDs in a formatted table
- Allows selecting multiple accounts as followers
- Prevents selecting "Monitor All" for copy trading (must be individual accounts)

## Architecture

### Files Created/Modified

1. **`copy_trading_setup.py`** (NEW)
   - Interactive copy trading setup functions
   - Account selection logic
   - Configuration management

2. **`julie001.py`** (MODIFIED)
   - Added import for `setup_copy_trading_interactive`
   - Added copy trading setup prompt after authentication
   - Initializes ProjectXClient with copy trader

3. **`julie_tkinter_ui.py`** (MODIFIED)
   - Enhanced `create_copy_trading_stats_section()` to always show
   - Added enable/disable buttons
   - Added account selection dialog
   - Added three new methods:
     - `enable_copy_trading()`
     - `disable_copy_trading()`
     - `select_copy_trading_accounts()`

### Data Flow

```
User enables copy trading
    ↓
Selects follower accounts (via AccountSelector)
    ↓
Configures size ratios
    ↓
Creates FollowerAccount objects
    ↓
Authenticates each follower
    ↓
Creates CopyTrader instance
    ↓
Passes to ProjectXClient
    ↓
Trades automatically copy to followers
```

## Configuration

The configuration is stored in `CONFIG['COPY_TRADING']`:

```python
CONFIG['COPY_TRADING'] = {
    'enabled': True,
    'followers': [
        {
            'username': 'user@example.com',
            'api_key': 'your-api-key',
            'account_id': 'follower-account-id',
            'contract_id': 'contract-id',
            'size_ratio': 1.0,
            'enabled': True
        },
        # More followers...
    ]
}
```

## Safety Features

1. **Circuit Breaker** - Prevents runaway order placement
2. **Rate Limiting** - Stays within TopStepX API limits
3. **Error Isolation** - Follower failures don't affect leader trades
4. **Confirmation Dialogs** - Asks before enabling/disabling
5. **Size Ratio Validation** - Ensures valid ratios are configured

## Example Workflow

### Scenario: Copy trades from main account to 2 test accounts

1. **Enable from UI:**
   - Click "✅ Enable Copy Trading"
   - Select Account A (set ratio 1.0)
   - Select Account B (set ratio 0.5)
   - Click "Save Configuration"

2. **Result:**
   - Leader places 10 contracts → Account A gets 10 contracts, Account B gets 5 contracts
   - Leader closes position → Both followers close automatically
   - All trades logged with copy trading statistics

3. **Monitor:**
   - Dashboard shows "2 Followers ACTIVE"
   - Bot logs show "📋 Copying trade to 2 followers..."
   - Copy trading stats tracked in real-time

## Troubleshooting

### Copy trading not working?

1. **Check authentication**: All follower accounts must authenticate successfully
2. **Check contracts**: Each follower must have access to the same contract
3. **Check API limits**: Too many followers may hit rate limits
4. **Check logs**: Look for circuit breaker trips or authentication failures

### Can't select accounts?

1. Ensure you're logged in first
2. Check that you have active accounts in TopStepX
3. Verify API access is enabled

### Size ratios not working?

1. Ensure ratios are positive numbers (e.g., 0.5, 1.0, 2.0)
2. Check that follower accounts have sufficient margin
3. Verify position sizing is calculated correctly in logs

## Advanced Features

### Programmatic Setup

You can also setup copy trading programmatically:

```python
from copy_trading_setup import setup_copy_trading_from_accounts

# Setup with account IDs and ratios
copy_trader = setup_copy_trading_from_accounts(
    session=authenticated_session,
    follower_account_ids=['acc-1', 'acc-2'],
    size_ratios=[1.0, 0.5]
)

# Use with ProjectXClient
client = ProjectXClient(copy_trader=copy_trader)
```

### Custom Credentials per Follower

Currently, all followers use the same credentials from CONFIG. For production use with separate credentials:

Modify `setup_copy_trading_from_accounts()` to accept credential dictionaries.

## Next Steps

1. Enable copy trading from either interface
2. Select your follower accounts
3. Configure appropriate size ratios
4. Monitor the dashboard for copy trading statistics
5. Check logs for successful trade replication

## Support

For issues or questions:
- Check `topstep_live_bot.log` for detailed logs
- Review copy trading statistics in the dashboard
- Verify all accounts are properly authenticated
