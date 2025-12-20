# JULIE Tkinter UI

A modern, professional trading dashboard for the JULIE algorithmic trading bot, featuring a sleek login interface and real-time monitoring dashboard.

## Features

### Login Page
- Professional dark-themed design
- Account selection dropdown
- Secure authentication
- "Forgot password" functionality (placeholder)

### Main Dashboard

#### Signal Monitor & Market Context
- Real-time MES Futures price display
- Large, easy-to-read price and percentage change
- Session detection (ASIA, LONDON, NY_AM, NY_PM)

#### Strategy List
Display of all 9 trading strategies with real-time status:
1. **Regime Adaptive** - SMA crossover with 320 context-specific parameters
2. **Intraday Dip** - Mean-reversion strategy
3. **Confluence** - ICT-based price action system
4. **ICT Model (Silver Bullet)** - NY AM session setup hunter
5. **ORB Strategy** - Opening Range Breakout
6. **ML Physics** - Session-specific neural network models
7. **Dynamic Engine 1** - 235 indicator-based sub-strategies
8. **Dynamic Engine 2** - 167 price-action sub-strategies
9. **SMT Divergence** - ES/NQ divergence detection

Each strategy shows:
- Current status (WAITING, PENDING SIGNAL, EXECUTED, PASS)
- Execution price for active signals
- Color-coded status indicators

#### Active Positions
- Account ID display
- Position side (LONG/SHORT) with color coding
- Entry price
- Real-time P&L calculation
- Contract size

#### Filter Status Dashboard
Visual grid display of all filters with status indicators:

**Filters:**
- Rejection (❌) - Bias establishment
- Chop Filter (🌊) - Consolidation detection
- News (📰) - Economic event blocking
- Volatility (🎯) - Regime classification
- Correlation (🔄) - SMT analysis
- Depth (📊) - Market depth
- Time (🕐) - Session timing
- System (⚙️) - Circuit breaker status
- Extension (📈) - Exhaustion detection
- Trend (📉) - Multi-timeframe validation
- Impulse (⚡) - Impulsive candle detection
- HTF FVG (🎚️) - Higher timeframe FVG tracking

Status colors:
- **Green [PASS]** - Filter allows trading
- **Red [BLOCK]** - Filter blocking signals
- **Green [SAFE]** - Monitoring, no restrictions

#### Live Event Log
- Real-time scrolling log of all bot activities
- System messages
- Signal generation events
- Filter status changes
- Trade executions
- Position updates
- Timestamps for all events

## Installation

### Prerequisites

1. **Python 3.11+**
2. **Tkinter** (Python GUI library)
   - **Ubuntu/Debian:**
     ```bash
     sudo apt-get install python3-tk
     ```
   - **Fedora/RHEL:**
     ```bash
     sudo dnf install python3-tkinter
     ```
   - **macOS:** Comes with Python
   - **Windows:** Comes with Python

3. **Python Dependencies:**
   ```bash
   pip install requests
   ```

## Usage

### Quick Start

1. **Using the launcher script (recommended):**
   ```bash
   python launch_ui.py
   ```

2. **Direct launch:**
   ```bash
   python julie_tkinter_ui.py
   ```

### Configuration

The UI automatically integrates with your existing JULIE bot configuration:

1. **Account Selection:** Choose your account from the dropdown on the login page
2. **API Integration:** The UI will use credentials from `config.py`
3. **Log Monitoring:** Automatically monitors `topstep_live_bot.log` for events

### Login

1. Select your account from the dropdown (ACCT-001, ACCT-002, etc.)
2. Click the green "LOGIN" button
3. The dashboard will load and begin monitoring

### Dashboard Navigation

- **Strategy List:** Shows all strategies and their current status
- **Active Positions:** Displays your current market positions with real-time P&L
- **Filter Dashboard:** Grid view of all filter statuses at a glance
- **Event Log:** Scrolling log of all bot activities (right panel)

## Integration with JULIE Bot

The UI integrates with the existing JULIE infrastructure:

### Data Sources

1. **API Monitor (`julie_ui.py`):**
   - Fetches positions from TopstepX API
   - Retrieves market data
   - Updates account information

2. **Log Monitor (`julie_ui.py`):**
   - Parses `topstep_live_bot.log` for events
   - Extracts signal executions
   - Tracks filter status changes
   - Monitors trade lifecycle

### Real-time Updates

- **Positions:** Updated every 2 seconds
- **Market Price:** Updated every 3 seconds
- **Event Log:** Updated every 0.5 seconds
- **UI Refresh:** Smooth, non-blocking updates

## Mock Mode

If the bot monitoring modules are not available, the UI runs in mock mode with simulated data. This is useful for:
- Testing the UI design
- Demonstration purposes
- Development without API access

## Design

The UI features a professional dark theme inspired by modern trading platforms:

- **Background:** Deep navy (#0a0e1a)
- **Panels:** Dark gray (#141824)
- **Accents:**
  - Green (#4ade80) - Long positions, gains, pass status
  - Red (#f87171) - Short positions, losses, block status
  - Yellow (#fbbf24) - Pending/waiting status
- **Typography:** Helvetica with size hierarchy for readability

## Keyboard Shortcuts

- **Ctrl+C:** Exit the application (when in terminal)
- **Alt+F4:** Close window (platform dependent)

## Troubleshooting

### "ModuleNotFoundError: No module named 'tkinter'"
Install tkinter using the instructions in the Prerequisites section.

### "Authentication failed"
Check that your `config.py` has valid `USERNAME` and `API_KEY` values.

### "No log file found"
The UI will wait for the bot to create `topstep_live_bot.log`. Start your trading bot first.

### UI is slow or unresponsive
- Check your internet connection (API calls may be slow)
- Reduce update frequency in the code if needed
- Ensure the bot log file isn't too large

## Advanced Configuration

### Customizing Update Rates

Edit `julie_tkinter_ui.py` and modify the timing values in `start_real_monitoring()`:

```python
# Check position every 2 seconds
if now - last_position_check > 2.0:

# Check price every 3 seconds
if now - last_price_check > 3.0:

# Check log file every 0.5 seconds
if now - last_log_check > 0.5:
```

### Customizing Colors

Modify the `self.colors` dictionary in the `JulieUI.__init__()` method:

```python
self.colors = {
    'bg_primary': '#0a0e1a',      # Main dark background
    'bg_secondary': '#141824',     # Slightly lighter panels
    'accent_green': '#4ade80',     # Success/long green
    'accent_red': '#f87171',       # Danger/short red
    # ... etc
}
```

### Adding Custom Filters

To add more filters to the dashboard, edit the `filters` list in `create_filter_dashboard_section()`:

```python
filters = [
    ("Filter Name", "STATUS", "🎯"),
    # ... add more
]
```

## Architecture

```
julie_tkinter_ui.py (Main UI Application)
├── Login Page
│   ├── Account Selection
│   └── Authentication Handler
└── Dashboard
    ├── Market Context Display
    ├── Strategy List Monitor
    ├── Active Positions Display
    ├── Filter Status Grid
    └── Live Event Log
        └── Monitoring Threads
            ├── API Monitor (julie_ui.py)
            │   ├── Position Fetching
            │   └── Market Data
            └── Log Monitor (julie_ui.py)
                └── Event Parsing
```

## Files

- **`julie_tkinter_ui.py`** - Main UI application
- **`launch_ui.py`** - Launcher script with dependency checking
- **`TKINTER_UI_README.md`** - This file

## Requirements

- Python 3.11+
- tkinter (python3-tk)
- requests
- config.py (from JULIE bot)
- julie_ui.py (from JULIE bot)

## License

Part of the JULIE trading bot system.

## Support

For issues or questions:
1. Check the Troubleshooting section
2. Review the main JULIE README.md
3. Check your bot logs for errors

---

**Note:** This UI is designed for monitoring and visualization. All trading logic remains in the main JULIE bot. The UI is read-only and does not execute trades directly.
