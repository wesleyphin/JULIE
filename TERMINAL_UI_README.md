# JULIE Terminal UI - Real-Time Trading Monitor

A beautiful, real-time terminal interface for monitoring your JULIE trading bot's activity, positions, signals, and market data.

## Description

The JULIE Terminal UI is a **standalone monitoring application** that provides a live dashboard view of your trading bot's operations without modifying any of your existing bot code. It displays:

### 📊 **Real-Time Position Tracking**
- **Current Position**: Live display of your active trades
  - Side (LONG/SHORT) with color-coded indicators
  - Entry price and current market price
  - Real-time P&L calculation (profit/loss in dollars)
  - Take Profit (TP) and Stop Loss (SL) levels
  - Bars held counter
  - Strategy name that opened the position
- **Flat Status**: Clear indication when no position is active

### 🎯 **Signal Monitoring**
- **Recent Signals**: Last 10 trading signals from all strategies
  - Strategy name (RegimeAdaptive, IntradayDip, Confluence, etc.)
  - Signal direction (LONG/SHORT)
  - TP and SL distances
  - Execution status (EXECUTED, BLOCKED, PENDING)
  - Timestamp for each signal
- **Strategy Coverage**: Monitors all 7 strategies:
  - RegimeAdaptive (FAST)
  - IntradayDip (FAST)
  - Confluence (STANDARD)
  - MLPhysics (STANDARD)
  - DynamicEngine (STANDARD)
  - ORB (LOOSE)
  - ICTModel (LOOSE)

### 🌍 **Market Context**
- **Session Detection**: Automatically identifies current trading session
  - ASIA (18:00-03:00 ET)
  - LONDON (03:00-08:00 ET)
  - NY_AM (08:00-12:00 ET)
  - NY_PM (12:00-17:00 ET)
- **Current Price**: Live MES futures price updates
- **Market Bias**: Current directional bias (LONG/SHORT/NEUTRAL)
- **Volatility Regime**: Market volatility state
- **Symbol Info**: Contract being traded (e.g., CON.F.US.MES.Z25)
- **Account Info**: Account ID and daily P&L tracking

### 🛡️ **Filter Status Dashboard**
Real-time status of all 6 defense filters:
- **Rejection Filter**: Trade direction bias filter
- **HTF FVG Filter**: Higher timeframe Fair Value Gap blocker
- **Chop Filter**: Range-bound market detector
- **Extension Filter**: Overextension/exhaustion blocker
- **Structure Blocker**: Equal highs/lows weak level filter
- **Bank Level Filter**: $12.50 bank level bias tracker

Each filter shows:
- ✓ PASS (green) - Filter allowing trades
- ✗ BLOCK (red) - Filter blocking trades
- IDLE (dim) - Filter not yet evaluated

### 📝 **Event Log**
Live scrolling log of bot activity:
- System events (startup, authentication, configuration)
- Trade executions (orders placed, filled, closed)
- Signal generation and rejection reasons
- Filter blocks with detailed explanations
- Break-even adjustments
- API connection status
- Error messages and warnings

### 🎨 **Visual Design**
- **Color-coded UI**: Green for LONG, red for SHORT, yellow for signals
- **Live updates**: Refreshes automatically (1-2 times per second)
- **Clean layout**: Professional panels with clear information hierarchy
- **Status indicators**: Visual feedback for all bot states
- **Responsive**: Works in any terminal size (recommended 120x40 minimum)

---

## Quickstart Guide

### Prerequisites

1. **Python packages** (should already be installed with your bot):
   ```bash
   pip install rich requests pandas
   ```

2. **Configuration**: Your `config.py` must have valid credentials:
   - `USERNAME` - Your TopstepX username
   - `API_KEY` - Your API key
   - (Optional) `ACCOUNT_ID` - Will auto-select if not set

3. **Bot running**: For signal monitoring, have `julie001.py` running in another terminal

### Running the Monitor

**Option 1: Direct execution**
```bash
python monitor_ui.py
```

**Option 2: As executable (Linux/Mac)**
```bash
./monitor_ui.py
```

**Option 3: Background monitoring**
```bash
# Run in background (Linux/Mac)
nohup python monitor_ui.py > /dev/null 2>&1 &

# Or use screen/tmux
screen -S julie-monitor
python monitor_ui.py
# Press Ctrl+A then D to detach
```

### What You'll See

```
╭─────────────────────────────────────────────────────────╮
│      JULIE - Advanced MES Futures Trading Bot          │
│    Session: NY_AM | Symbol: MES | 2024-12-14 10:30:15  │
╰─────────────────────────────────────────────────────────╯

╭─ Current Position ────────────────────────────────────╮
│ Status  │ Side  │ Entry   │ Current │ P&L    │ Bars  │
│ ACTIVE  │ LONG  │ 5875.25 │ 5878.50 │ +$16.25│   3   │
│ RegimeAdaptive                                        │
╰───────────────────────────────────────────────────────╯

╭─ Recent Signals ──────────────────────────────────────╮
│ 10:29:45 │ RegimeAdaptive │ LONG  │ TP: 6.0 │ EXECUTED│
│ 10:28:30 │ IntradayDip    │ SHORT │ TP: 5.0 │ BLOCKED │
╰───────────────────────────────────────────────────────╯

╭─ Filter Status ───────╮  ╭─ Market Context ──────────╮
│ Rejection    │ PASS ✓ │  │ Session:   NY_AM          │
│ HTF FVG      │ PASS ✓ │  │ Price:     5878.50        │
│ Chop         │ BLOCK ✗│  │ Bias:      LONG           │
│ Extension    │ PASS ✓ │  │ Volatility: NORMAL        │
│ Structure    │ PASS ✓ │  │ Account:   abc123...      │
│ Bank Level   │ PASS ✓ │  │ Daily P&L: +$125.50       │
╰────────────────────────╯  ╰───────────────────────────╯

╭─ Event Log ───────────────────────────────────────────╮
│ [10:29:45] TRADE: ✓ RegimeAdaptive LONG executed     │
│ [10:29:44] FILTER: All filters passed                │
│ [10:28:30] FILTER: Blocked by Chop filter            │
│ [10:25:00] SYSTEM: Listening for market data...      │
╰───────────────────────────────────────────────────────╯
```

### Stopping the Monitor

Press **Ctrl+C** to gracefully shut down the monitor.

---

## How It Works

### Independent Operation
The monitor runs **completely independently** from your main trading bot:
- ✅ **No modifications** to `julie001.py` required
- ✅ **Zero interference** with bot operations
- ✅ **Safe to stop/start** anytime without affecting trades
- ✅ **Multiple instances** can run simultaneously if desired

### Data Sources

1. **TopstepX API** (Direct connection)
   - Fetches live positions every 2 seconds
   - Gets current market price every 3 seconds
   - Uses same credentials as your bot
   - Independent rate limiting (won't conflict with bot)

2. **Log File Monitoring** (`topstep_live_bot.log`)
   - Tails the log file in real-time
   - Extracts signal executions and filter blocks
   - Parses trade events (orders, closes, break-even)
   - Updates every 0.5 seconds for near-instant feedback

### Architecture

```
julie001.py (Your Bot)          monitor_ui.py (This Monitor)
       │                                 │
       │ writes logs                     │
       ├──────────────────►──────────────┤
       │                                 │
       │                        ┌────────▼────────┐
       │                        │  Log Monitor    │
       │                        │  (tail parser)  │
       │                        └────────┬────────┘
       │                                 │
       │                        ┌────────▼────────┐
       │                        │  Terminal UI    │
       │                        │  (rich library) │
       │                        └────────┬────────┘
       │                                 │
       ▼                                 ▼
TopstepX API ◄──────────────────────────┤
  (positions)                    API Monitor
  (prices)                       (REST client)
```

---

## Troubleshooting

### "Failed to authenticate"
- Check your `config.py` has valid `USERNAME` and `API_KEY`
- Ensure your API key hasn't expired
- Verify network connectivity

### "Waiting for log file"
- The bot hasn't started yet or hasn't written logs
- Check the log file path: `topstep_live_bot.log` in same directory
- Start `julie001.py` first, then the monitor

### "No active accounts found"
- Your account may be disabled/inactive on TopstepX
- Check account status in TopstepX dashboard
- Hardcode `ACCOUNT_ID` in `config.py` if you know it

### Position shows "Unknown" strategy
- This is normal - the monitor can't determine strategy without log parsing
- If the bot just started, historical context isn't available
- The strategy name will appear for new trades

### Signals not appearing
- Make sure `julie001.py` is running and generating signals
- Check that the log file is being written to
- Verify log level is INFO or higher

### UI is sluggish/choppy
- Reduce refresh rate in `monitor_ui.py` (line 412): `ui.start(refresh_rate=0.5)`
- Close other terminal programs consuming resources
- Use a modern terminal emulator (iTerm2, Windows Terminal, etc.)

---

## Advanced Usage

### Custom Refresh Rates
Edit `monitor_ui.py` line 412:
```python
ui.start(refresh_rate=2.0)  # Update twice per second (default: 1.0)
```

### Monitor Multiple Accounts
Run separate instances with different `config.py` files:
```bash
# Terminal 1
python monitor_ui.py

# Terminal 2 (different config)
CONFIG_FILE=config_account2.py python monitor_ui.py
```

### Integration with Alerts
The monitor can be extended to send alerts:
```python
# Add to monitor_ui.py
if position['side'] is not None and pnl > 100:
    send_notification(f"Profit target reached: ${pnl}")
```

### Remote Monitoring
Run the monitor over SSH:
```bash
ssh user@trading-server
cd /path/to/JULIE
python monitor_ui.py
```

---

## Technical Details

### Dependencies
- `rich` - Terminal UI framework with live updates
- `requests` - HTTP client for TopstepX API
- `pandas` - Data processing (inherited from bot)
- `zoneinfo` - Timezone handling (US/Eastern) - built into Python 3.9+

### Performance
- CPU usage: ~1-3% (mostly from terminal rendering)
- Memory: ~50-100 MB
- Network: ~1-2 KB/s (API polling)
- Disk I/O: Minimal (log file reading)

### File Structure
```
JULIE/
├── julie001.py              # Main trading bot (UNCHANGED)
├── config.py                # Shared configuration
├── terminal_ui.py           # UI framework/layout engine
├── monitor_ui.py            # Monitoring application (THIS)
├── topstep_live_bot.log     # Log file (auto-created)
└── TERMINAL_UI_README.md    # This documentation
```

---

## Safety & Risk

### Safety Features
- ✅ **Read-only operation**: Monitor never places orders or modifies positions
- ✅ **Isolated API client**: Separate from bot's API client
- ✅ **No file writes**: Doesn't modify any bot files
- ✅ **Graceful errors**: Handles API failures without crashing

### Limitations
- Cannot control the bot (stop/start trades)
- Historical data limited to current session
- P&L calculation is estimated (based on current price vs entry)
- Some strategy details only available through log parsing

---

## Support

### Questions?
- Check the main JULIE bot documentation
- Review `config.py` for configuration options
- Examine `terminal_ui.py` for UI customization

### Want to contribute?
The monitor is designed to be extended:
- Add new panels to `terminal_ui.py`
- Parse additional log patterns in `monitor_ui.py`
- Integrate notifications, alerts, or webhooks
- Export data to CSV/database for analysis

---

**Enjoy your real-time trading dashboard!** 🚀

*The JULIE Terminal UI - Because traders deserve beautiful tools.*
