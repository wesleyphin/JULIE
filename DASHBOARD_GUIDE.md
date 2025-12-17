# Julie Dashboard Guide

## Overview

Julie Dashboard is an all-in-one solution that combines the trading bot with a beautiful multi-account monitoring interface. It shows P&L for all accounts, active trades, and executes strategies - all in one integrated application.

## Key Features

🎯 **Multi-Account Support** - Monitor and trade multiple accounts simultaneously
📊 **Real-Time P&L** - See profit/loss for each account in real-time
🔗 **Copy Trade Detection** - Automatically detects copy trade setup from TopstepX
⚙️ **Smart Trade Execution** - In copy trade mode, only master account executes (others mirror)
📈 **Live Dashboard** - Beautiful terminal UI showing all accounts, positions, and signals
🎮 **Integrated Bot** - Full trading strategies built-in (from julie001.py)

## How to Run

```bash
python julie_dashboard.py
```

That's it! No need to run separate monitoring and trading programs.

## What You'll See

### 1. Account Selection
First, you'll see the beautiful account selector:
```
╭──────────────────────────────────────────────────────╮
│            JULIE TRADING BOT                         │
│            Account Selection                         │
╰──────────────────────────────────────────────────────╯

┌────────┬──────────────────────────┬────────────────┬────────┐
│ Option │ Account Name             │ Account ID     │ Status │
├────────┼──────────────────────────┼────────────────┼────────┤
│   0    │ MONITOR ALL ACCOUNTS     │ All active     │   ●    │
│   1    │ My First Account         │ acc_123...     │   ●    │
│   2    │ My Second Account        │ acc_456...     │   ●    │
└────────┴──────────────────────────┴────────────────┴────────┘

Your choice [0]:
```

### 2. Copy Trade Detection (If Applicable)
If the system detects copy trade is enabled:
```
Copy Trade Detected!
Select which account should execute trades:
(Other accounts will mirror the trades automatically)

┌────────┬──────────────────────────┬────────────────┐
│ Option │ Account Name             │ Account ID     │
├────────┼──────────────────────────┼────────────────┤
│   1    │ My First Account         │ acc_123...     │
│   2    │ My Second Account        │ acc_456...     │
└────────┴──────────────────────────┴────────────────┘

Select trading account (1-2) [1]:
```

### 3. Live Dashboard
Once running, you see the full dashboard:

```
╭────────────────────────────────────────────────────────────╮
│   Julie Dashboard - Multi-Account Trading Platform         │
│   Session: NY_AM | 2025-12-17 10:30:45                    │
╰────────────────────────────────────────────────────────────╯

┌─────────────────── Active Accounts ────────────────────────┐
│ Account        │ Mode  │ Pos │ Entry  │ Current │ P&L      │ W/L  │
├────────────────┼───────┼─────┼────────┼─────────┼──────────┼──────┤
│ Main Account   │ TRADE │  L  │ 5875.0 │ 5880.0  │  +$25.00 │ 3/1  │
│ Second Account │ WATCH │  L  │ 5875.0 │ 5880.0  │  +$25.00 │ 0/0  │
│ Third Account  │ WATCH │  -  │   -    │ 5880.0  │   $0.00  │ 0/0  │
└────────────────┴───────┴─────┴────────┴─────────┴──────────┴──────┘

┌─────────────────── Recent Signals ─────────────────────────┐
│ Time   │ Account      │ Strategy       │ Side │ Status    │
├────────┼──────────────┼────────────────┼──────┼───────────┤
│ 10:30  │ Main Account │ RegimeAdaptive │ LONG │ EXECUTED  │
│ 10:25  │ Main Account │ IntradayDip    │ LONG │ BLOCKED   │
└────────┴──────────────┴────────────────┴──────┴───────────┘

┌────────── Market ──────────┐ ┌─────── Summary ────────┐
│ Session:     NY_AM         │ │ Total Accounts:    3   │
│ Symbol:      MES           │ │ Active Positions:  1   │
│ Price:       5880.00       │ │ Daily P&L:    +$50.00  │
└────────────────────────────┘ │ Total Trades:      4   │
                               │ Win Rate:      75.0%   │
┌─────────── Event Log ──────────────────────────────────────┐
│ [10:30:45] BOT: [TRADING] Started for Main Account        │
│ [10:30:44] BOT: [MONITOR] Started for Second Account      │
│ [10:30:43] SYSTEM: 🔗 COPY TRADE DETECTED: Trading on...  │
│ [10:30:42] SYSTEM: ✓ Loaded 3 account(s)                  │
└────────────────────────────────────────────────────────────┘

 Active Accounts: 3  |  Active Bots: 1  |  Press Ctrl+C to exit
```

## Understanding the Dashboard

### Account Table Columns

- **Account**: Account name (truncated if long)
- **Mode**:
  - `TRADE` = This account executes trades
  - `WATCH` = Monitor only (mirrors trades if copy trade enabled)
- **Pos**: Current position (`L` = Long, `S` = Short, `-` = Flat)
- **Entry**: Entry price of current position
- **Current**: Current market price
- **P&L**: Profit/Loss in dollars (green = profit, red = loss)
- **W/L**: Win/Loss count (e.g., "3/1" = 3 wins, 1 loss)

### Trading Modes

**TRADE Mode** (Green)
- Bot executes all trading strategies
- Places orders, manages positions
- Calculates signals and follows filters
- One account per bot instance

**WATCH Mode** (Gray/Dim)
- Monitors positions only
- No orders placed
- If copy trade enabled: Will mirror master account's trades via platform
- Tracks P&L and displays positions

### Summary Panel

- **Total Accounts**: How many accounts loaded
- **Active Positions**: How many accounts have open positions
- **Daily P&L**: Combined profit/loss across all accounts
- **Total Trades**: Combined trade count
- **Win Rate**: Percentage of winning trades

## Copy Trade Scenarios

### Scenario 1: Copy Trade Enabled
```
User selects: "Monitor All" (3 accounts)
System detects: Copy Trade enabled
User chooses: Account 1 as trading account

Result:
- Account 1: TRADE mode - Executes strategies
- Account 2: WATCH mode - Mirrors Account 1 (via platform)
- Account 3: WATCH mode - Mirrors Account 1 (via platform)

Dashboard shows: All 3 accounts with P&L
```

### Scenario 2: No Copy Trade
```
User selects: "Monitor All" (3 accounts)
System detects: No copy trade

Result:
- Account 1: TRADE mode - Runs independent bot
- Account 2: TRADE mode - Runs independent bot
- Account 3: TRADE mode - Runs independent bot

Dashboard shows: All 3 accounts trading independently
```

### Scenario 3: Single Account
```
User selects: Account 1 only
System detects: N/A (only one account)

Result:
- Account 1: TRADE mode - Runs bot

Dashboard shows: Single account (same as julie001.py)
```

## Comparison: Dashboard vs Julie001

| Feature | julie001.py | julie_dashboard.py |
|---------|-------------|-------------------|
| Trading | ✅ Yes | ✅ Yes |
| Multi-Account | ❌ No | ✅ Yes |
| Real-Time UI | ❌ No | ✅ Yes |
| P&L Display | ❌ No | ✅ Yes |
| Copy Trade Support | ❌ No | ✅ Yes |
| Monitor Only Mode | ❌ No | ✅ Yes |
| All Strategies | ✅ Yes | ✅ Yes |

## FAQ

**Q: Does this replace julie001.py?**
A: Julie Dashboard includes all the functionality of julie001.py plus multi-account support and live UI. You can use either one.

**Q: Can I trade on multiple accounts without copy trade?**
A: Yes! If copy trade is not detected, all selected accounts will trade independently.

**Q: What happens if I select "Monitor All" with copy trade?**
A: You'll be asked to select which account should execute trades. The others will automatically mirror via the platform's copy trade feature.

**Q: Does this use more API calls?**
A: Yes, each account being monitored makes its own API calls for positions and market data. However, rate limiting is handled automatically.

**Q: Will strategies run differently?**
A: No! The exact same strategy code from julie001.py is used. No changes to trading logic.

**Q: Can I see historical trades?**
A: The dashboard tracks trades during the current session (Win/Loss counts, P&L). For full history, check TopstepX platform.

**Q: What if one account gets liquidated?**
A: The dashboard will detect the position closure and update the display. The bot continues running on other accounts.

## Troubleshooting

### Issue: "Copy Trade Detected" but I don't have copy trade
**Solution**: The system checks account metadata for copy trade indicators. If falsely detected, it will still let you select which account trades. In worst case, select the same account you would trade anyway.

### Issue: Dashboard not showing positions
**Solution**: Make sure the bot has completed authentication and fetched contract details. Check the Event Log panel for errors.

### Issue: Multiple accounts all showing TRADE mode when copy trade enabled
**Solution**: The system may not have detected copy trade from the API. You can still use it - each account will trade independently (they won't conflict).

### Issue: UI is slow or laggy
**Solution**: The dashboard updates every 2 seconds. If monitoring many accounts (5+), this is normal. Consider monitoring fewer accounts or increasing the refresh rate in the code.

## Tips

1. **Start with one account** to familiarize yourself with the dashboard before using multi-account
2. **Monitor first** before enabling live trading - use paper trading accounts
3. **Check the Event Log** regularly for important system messages
4. **Watch the Win Rate** - if it drops below 50%, consider pausing
5. **Use copy trade wisely** - only the master account needs sufficient capital for position sizing

## Technical Details

- Built on julie001.py trading engine
- Uses Rich library for terminal UI
- Thread-safe multi-account tracking
- Real-time P&L calculations
- Automatic rate limit handling
- Position sync every 5 seconds
- Market data refresh every 2 seconds

---

**Ready to trade?** Run `python julie_dashboard.py` and start monitoring your accounts!
