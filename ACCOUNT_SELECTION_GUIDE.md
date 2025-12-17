# Account Selection UI Guide

## Overview

Julie-UI and JULIE Trading Bot now feature a beautiful, interactive account selection interface that allows you to:
- Select a specific account to monitor/trade
- Monitor ALL accounts simultaneously
- Works with both the monitoring UI and the trading bot

## Features

✨ **Beautiful Terminal UI** - Clean, colorful interface using Rich library
🎯 **Multiple Account Support** - Monitor all your accounts at once
🖥️ **Cross-Platform** - Works on Windows 11, macOS, and Linux
🔄 **Smart Integration** - Automatically integrates with existing code
⚡ **Fast & Intuitive** - Simple number-based selection

## How to Use

### For Monitoring (julie_ui.py)

1. Start Julie-UI:
   ```bash
   python julie_ui.py
   ```

2. The beautiful account selection interface will appear automatically after authentication

3. You'll see a table with all your active accounts:
   ```
   ┌────────┬──────────────────────────────┬─────────────────────────┬────────┐
   │ Option │ Account Name                 │ Account ID              │ Status │
   ├────────┼──────────────────────────────┼─────────────────────────┼────────┤
   │   0    │ MONITOR ALL ACCOUNTS         │ All active accounts     │   ●    │
   │   1    │ Your First Account           │ abc123...               │   ●    │
   │   2    │ Your Second Account          │ def456...               │   ●    │
   └────────┴──────────────────────────────┴─────────────────────────┴────────┘
   ```

4. Choose an option:
   - **Enter 0**: Monitor all accounts simultaneously
   - **Enter 1-N**: Monitor a specific account

### For Trading (julie001.py)

1. Start the trading bot:
   ```bash
   python julie001.py
   ```

2. The same beautiful interface will appear

3. Select the account you want to trade with
   - Note: The trading bot can only trade ONE account at a time (for safety)
   - If you select "Monitor All", the bot will use the first account

## Visual Examples

### Account Selection Screen
```
╭─────────────────────────────────────────────────╮
│            JULIE TRADING BOT                    │
│            Account Selection                    │
╰─────────────────────────────────────────────────╯

Fetching active accounts...
Found 3 active account(s)

[Beautiful table appears here]

Select an option:
  • Enter 0 to monitor ALL accounts
  • Enter 1-3 to monitor a specific account

Your choice [0]:
```

### Success Confirmation
```
╭─────────────────────────────────────────────────╮
│ ✓ Monitoring ALL 3 accounts                    │
╰─────────────────────────────────────────────────╯
```

or

```
╭─────────────────────────────────────────────────╮
│ ✓ Selected: My Trading Account                 │
│ Account ID: abc123def456                        │
╰─────────────────────────────────────────────────╯
```

## Technical Details

### Files Added/Modified

1. **account_selector.py** (NEW)
   - Beautiful account selection interface
   - Uses Rich library for terminal UI
   - Supports single and multi-account selection

2. **julie_ui.py** (MODIFIED - formerly monitor_ui.py)
   - Integrated account selector
   - Support for monitoring multiple accounts
   - Enhanced position tracking

3. **client.py** (MODIFIED)
   - Updated fetch_accounts() to use new UI
   - Fallback to simple text selection if UI unavailable
   - Backward compatible

### Cross-Platform Compatibility

The account selection UI works on all major platforms:
- ✅ **Windows 11**: Fully supported
- ✅ **macOS**: Fully supported (tested on M1/M2 and Intel)
- ✅ **Linux**: Fully supported (tested on Ubuntu/Debian)

### Requirements

The Rich library is required for the beautiful UI:
```bash
pip install rich
```

If Rich is not available, the system automatically falls back to simple text-based selection.

## Troubleshooting

### Issue: UI doesn't appear
**Solution**: Make sure Rich library is installed:
```bash
pip install rich
```

### Issue: Colors not showing
**Solution**: Your terminal may not support colors. Try:
- Windows: Use Windows Terminal or PowerShell 7+
- macOS: Use Terminal.app or iTerm2
- Linux: Most modern terminals support colors by default

### Issue: Can't select account with keyboard
**Solution**: Make sure you're entering numbers, not trying to click with mouse

## FAQ

**Q: Can I monitor multiple accounts with the trading bot?**
A: No, the trading bot (julie001.py) can only trade ONE account at a time for safety. However, Julie-UI (julie_ui.py) supports monitoring all accounts simultaneously.

**Q: Will this work with my existing strategies?**
A: Yes! The account selection is completely separate from trading strategies. No strategy code has been modified.

**Q: Can I automate account selection?**
A: Yes! Set `ACCOUNT_ID` in `config.py` to skip the interactive selection:
```python
CONFIG = {
    "ACCOUNT_ID": "your_account_id_here",
    # ... other settings
}
```

**Q: Does this use more API calls?**
A: When monitoring all accounts, yes - it makes one position check per account. However, this is throttled and respects rate limits.

## Support

For issues or questions:
- Check the log files: `topstep_live_bot.log`
- Review the console output for error messages
- Ensure your API credentials are correct in `config.py`

---

**Note**: Without changing any strategies, this enhancement provides a much better user experience for account management while maintaining full compatibility with existing code.
