# JULIE GUI Monitor - Modern Trading Dashboard

A modern, sleek graphical interface for monitoring your JULIE trading bot with **multi-account support**.

## 🚀 New Features

### Multi-Account Support
- **Select Individual Account**: Choose one account from the dropdown
- **Monitor All Accounts**: Check "All Accounts" to watch all your trading accounts simultaneously
- **Real-time Switching**: Change accounts on-the-fly without restarting

### Modern GUI Interface
- **Dark Theme**: Professional dark mode design
- **Color-Coded Data**: Green for LONG, Red for SHORT, Yellow for signals
- **Tabular Display**: Clean, organized tables for positions and signals
- **Live Updates**: Automatic refresh every 2 seconds

## 📊 What It Shows

### Positions Panel
For each account being monitored:
- Account ID
- Position side (LONG/SHORT/FLAT)
- Entry price and current price
- Real-time P&L in dollars
- Take Profit and Stop Loss levels
- Strategy that opened the position

### Signals Panel
Recent trading signals from all strategies:
- Timestamp
- Account ID
- Strategy name
- Signal direction (LONG/SHORT)
- TP and SL distances
- Status (EXECUTED/BLOCKED/PENDING)

### Market Context Panel
Current market information:
- Trading session (ASIA/LONDON/NY_AM/NY_PM)
- Current symbol (e.g., MESZ25)
- Live price updates
- Market bias (LONG/SHORT/NEUTRAL)
- Volatility regime

### Filter Status Panel
Real-time status of all 6 defense filters:
- ✓ PASS (green) - Filter allowing trades
- ✗ BLOCK (red) - Filter blocking trades
- IDLE (gray) - Filter not evaluated yet

### Event Log
Scrolling log of all trading activity:
- System events (startup, auth)
- Trade executions
- Signal generation
- Filter blocks
- Errors and warnings

## 🎯 Quickstart

### Run the GUI
```bash
python gui_monitor.py
```

### Select Accounts

**Option 1: Single Account**
1. Select account from dropdown
2. Monitor that one account

**Option 2: All Accounts**
1. Check the "All Accounts" checkbox
2. See positions and signals from all accounts in one view

### Requirements

**Python Packages:**
```bash
pip install tkinter  # Usually included with Python
pip install requests pytz
```

**Configuration:**
- Your `config.py` must have valid credentials:
  - `USERNAME` - TopstepX username
  - `API_KEY` - Your API key

## 🎨 UI Layout

```
┌─────────────────────────────────────────────────────────────┐
│  JULIE - MES Futures Trading Dashboard   [Accounts: ▼] [✓All]│
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─ Current Positions ─────────┐  ┌─ Market Context ─────┐│
│  │ Account│Side │Entry │P&L    │  │ Session:  NY_AM      ││
│  │ abc123 │LONG │5875  │+$16.25│  │ Symbol:   MESZ25     ││
│  │        │     │      │       │  │ Price:    5878.50    ││
│  └────────────────────────────┘  │ Bias:     LONG       ││
│                                   └──────────────────────┘│
│  ┌─ Recent Signals ─────────────┐  ┌─ Filter Status ─────┐│
│  │ Time  │Strategy │Side│Status │  │ Rejection   PASS ✓  ││
│  │10:30  │Regime   │LONG│EXEC   │  │ HTF FVG     PASS ✓  ││
│  │10:29  │IntDay   │SHRT│BLOCK  │  │ Chop        BLOCK ✗ ││
│  └────────────────────────────────┘  └──────────────────────┘│
│                                                             │
│  ┌─ Event Log ───────────────────────────────────────────┐│
│  │ [10:30:15] TRADE: RegimeAdaptive LONG executed       ││
│  │ [10:29:45] FILTER: Blocked by Chop filter            ││
│  │ [10:28:30] SYSTEM: Monitoring all 3 accounts         ││
│  └───────────────────────────────────────────────────────┘│
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 🔧 How It Works

### Multi-Account Architecture

1. **Authentication**: Logs in once with your API key
2. **Account Discovery**: Fetches all active accounts from TopstepX
3. **Contract Mapping**: Maps each account to the correct MES contract
4. **Parallel Monitoring**: Updates all selected accounts every 2 seconds
5. **Unified Display**: Shows all data in one clean interface

### Data Flow

```
TopstepX API
     │
     ├─ /api/Account/search ──→ [Account List]
     │                                 │
     ├─ /api/Contract/search ──→ [Contract IDs per account]
     │                                 │
     ├─ /api/Position/search ──→ [Positions per account]
     │                                 │
     └─ /api/History/retrieveBars ──→ [Live prices]
                                       │
                                       ▼
                              ┌─────────────────┐
                              │   GUI Monitor   │
                              │  - Positions    │
                              │  - Signals      │
                              │  - Filters      │
                              │  - Events       │
                              └─────────────────┘
```

## 📋 Features Comparison

| Feature | Terminal UI (`monitor_ui.py`) | GUI Monitor (`gui_monitor.py`) |
|---------|-------------------------------|--------------------------------|
| **Interface** | Terminal/CLI | Graphical Window |
| **Accounts** | Single | Multiple / All |
| **Styling** | Text-based (rich) | Modern Dark Theme |
| **Positions** | Single view | Multi-account table |
| **Signals** | Log style | Tabular |
| **Filters** | List | Status panel |
| **Updates** | Real-time | Real-time |
| **Mouse Control** | No | Yes |

## ⚙️ Advanced Usage

### Custom Colors

Edit the `colors` dictionary in `gui_monitor.py`:
```python
self.colors = {
    'bg_dark': '#1e1e1e',      # Background
    'green': '#00ff00',         # LONG positions
    'red': '#ff4444',          # SHORT positions
    # ... customize as needed
}
```

### Refresh Rate

Change the monitoring interval (default: 2 seconds):
```python
# In monitor_loop function
if now - last_check > 2.0:  # Change to 1.0 for 1 second, etc.
```

### Window Size

Adjust the window dimensions:
```python
self.root.geometry("1400x900")  # Width x Height
```

## 🐛 Troubleshooting

### "No accounts found"
- Check your `config.py` has valid `USERNAME` and `API_KEY`
- Verify your TopstepX account is active
- Check network connectivity

### GUI doesn't open
- Ensure tkinter is installed: `python -m tkinter`
- On Linux: `sudo apt-get install python3-tk`
- On Mac: tkinter comes with Python

### Accounts dropdown is empty
- Authentication may have failed
- Check the Event Log for error messages
- Verify API credentials in `config.py`

### Positions not updating
- Ensure you have selected an account
- Check "All Accounts" if you want to see all
- Verify the bot (`julie001.py`) is running

## 🆚 When to Use GUI vs Terminal

**Use GUI Monitor when:**
- ✅ You have multiple trading accounts
- ✅ You prefer graphical interfaces
- ✅ You want mouse-clickable controls
- ✅ You need to monitor all accounts at once

**Use Terminal Monitor when:**
- ✅ You only have one account
- ✅ You prefer lightweight terminal tools
- ✅ You're SSH'd into a remote server
- ✅ You want minimal resource usage

## 📦 What's Included

**New Files:**
- `gui_monitor.py` - Modern GUI trading dashboard
- `GUI_MONITOR_README.md` - This documentation

**Still Available:**
- `terminal_ui.py` - Terminal UI framework
- `monitor_ui.py` - Terminal monitor (single account)
- `TERMINAL_UI_README.md` - Terminal UI docs

**Both work independently!** Choose whichever fits your workflow.

---

## 🎯 Quick Reference

**Start GUI:**
```bash
python gui_monitor.py
```

**Select account:** Use the dropdown in top-right
**Monitor all accounts:** Check the "All Accounts" box
**Stop monitoring:** Close the window (X button)

**Keyboard Shortcuts:**
- None currently (mouse-driven interface)

---

**Enjoy your modern trading dashboard!** 📈

*The JULIE GUI Monitor - Professional tools for professional traders.*
