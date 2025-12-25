# Julie: Advanced MES Futures Trading System

![Version](https://img.shields.io/badge/version-2.0.0-blue.svg) ![Market](https://img.shields.io/badge/Market-MES%20Futures-green.svg) ![Platform](https://img.shields.io/badge/Platform-TopstepX-orange.svg) ![Python](https://img.shields.io/badge/Python-3.11+-green.svg) ![License](https://img.shields.io/badge/License-Proprietary-red.svg)

**Julie** is a modern, high-frequency, session-specialized algorithmic trading bot built to execute autonomously on the **ProjectX Gateway (TopstepX)**. Unlike traditional bots that use a single logic set, Julie functions as an orchestrator for a "Team of Rivals"—a portfolio of **9 distinct strategy classes** that compete to find the best entry, all governed by a central "Defense Layer" of dynamic filters and blockers.

## 🚀 Quick Start

### Prerequisites
- **Python 3.11 or higher**
- **Internet connection** (for API access)
- **TopstepX account** with valid credentials

### Installation

#### macOS
```bash
# 1. Install Python 3.11+ (if not already installed)
brew install python@3.11

# 2. Clone the repository
git clone <repository-url>
cd JULIE

# 3. Install required Python packages
pip3 install requests pandas numpy

# 4. Configure credentials
# Edit config.py and add your TopstepX credentials:
# CONFIG["USERNAME"] = "your_username"
# CONFIG["API_KEY"] = "your_api_key"
```

#### Windows
```cmd
# 1. Install Python 3.11+ from python.org (if not already installed)
# Download from: https://www.python.org/downloads/

# 2. Clone the repository
git clone <repository-url>
cd JULIE

# 3. Install required Python packages
pip install requests pandas numpy

# 4. Configure credentials
# Edit config.py and add your TopstepX credentials:
# CONFIG["USERNAME"] = "your_username"
# CONFIG["API_KEY"] = "your_api_key"
```

### Running JULIE

#### Option 1: Trading Bot (Headless)
Run the core trading bot without UI:

**macOS/Linux:**
```bash
python3 julie001.py
```

**Windows:**
```cmd
python julie001.py
```

#### Option 2: Tkinter UI Dashboard
Run the modern Tkinter-based trading dashboard:

**macOS:**
```bash
# Install tkinter (comes with Python on macOS)
python3 launch_ui.py
```

**Windows:**
```cmd
# Install tkinter (comes with Python on Windows)
python launch_ui.py
```

The UI provides:
- Real-time market data display
- Live strategy status monitoring
- Active position tracking with P&L
- Filter status dashboard
- Live event log

See [TKINTER_UI_README.md](TKINTER_UI_README.md) for detailed UI documentation.

---

## 📖 Table of Contents
- [Quick Start](#-quick-start)
- [System Architecture](#1-system-architecture)
- [Modern Features](#modern-features)
- [Strategy Biography (Deep Dive)](#2-strategy-biography-deep-dive)
- [Defense Layer: Filters & Blockers](#3-defense-layer-filters--blockers)
- [Risk Management System](#4-risk-management-system)
- [LLM Integration & Optimization](#5-llm-integration--optimization)
- [Session Management & Quarterly Theory](#6-session-management--quarterly-theory)
- [SMT Divergence Analysis](#7-smt-divergence-analysis)
- [Dynamic Parameters System](#8-dynamic-parameters-system)

---

## Modern Features

### Async Architecture
Julie v2.0 features a fully asynchronous architecture for improved performance:
- **AsyncMarketDataManager**: Non-blocking real-time market data streaming via SignalR
- **Concurrent Task Execution**: Heartbeat monitoring, position syncing, and strategy execution run in parallel
- **Event-Driven Design**: Efficient event handling and logging system

### Professional Trading Dashboard
- **Tkinter UI**: Modern dark-themed professional dashboard
- **Real-time Monitoring**: Live strategy status, positions, and filter states
- **Event Log**: Complete audit trail of all trading activities
- **Multi-Account Support**: Easy account switching via dropdown

### Enhanced Components
- **Yahoo VIX Integration**: Real-time VIX data for volatility-based strategies
- **Gemini AI Optimizer**: Continuous LLM-powered parameter optimization
- **Event Logger**: Comprehensive structured logging system
- **Circuit Breaker**: Advanced safety mechanisms with automatic recovery

---

## 1. System Architecture

### Core Philosophy
* **Micro-Regime Specialization:** The market is not treated as a monolith. Julie fragments the trading year into **320 specific time contexts** (e.g., *"Q1 Week 3 Tuesday London Session"*) and applies unique risk parameters to each.
* **Defensive Priority:** A trade signal is only an "application." The central filter system acts as a strict "underwriter," rejecting any application that violates market structure, volatility limits, or institutional bias constraints.
* **Dynamic Risk:** Stop Losses (SL) and Take Profits (TP) breathe with the market, expanding during high volatility and contracting during chop, determined by real-time **Shannon Entropy** and **GARCH** volatility models.
* **LLM-Powered Optimization:** Parameters are continuously refined using Google's Gemini AI to analyze performance and suggest improvements.

### The 320 Hierarchical Threshold System

Julie uses a sophisticated time-bucketing system that creates unique trading contexts:

```
Format: YearlyQ_MonthlyQ_DayOfWeek_Session
Example: Q1_W1_MON_ASIA, Q3_W3_FRI_NY_PM
```

| Component | Values | Count |
|:---|:---|:---|
| **Yearly Quarter** | Q1, Q2, Q3, Q4 | 4 |
| **Monthly Week** | W1, W2, W3, W4 | 4 |
| **Day of Week** | MON, TUE, WED, THU, FRI | 5 |
| **Session** | ASIA, LONDON, NY_AM, NY_PM | 4 |
| **Total Combinations** | 4 × 4 × 5 × 4 | **320** |

Each of these 320 contexts has its own optimized thresholds for:
- Stop Loss / Take Profit
- Chop detection thresholds
- Extension percentiles
- Volatility scaling factors

---

## 2. Strategy Biography (Deep Dive)

Julie runs **9 primary strategy engines** simultaneously. Each strategy generates signals independently, but all must pass through the Defense Layer.

<details>
<summary><strong>A. Regime Adaptive Strategy (The Core)</strong></summary>

*This is the system's flagship strategy. It relies on historical probabilities specific to the current moment in time.*

* **Logic:**
    * **Trend:** Uses `SMA20` vs `SMA200` crossover + Low Volatility regime to identify trend.
    * **Trigger:** Enters on pullbacks (Long) or rallies (Short) that exhibit a "Range Spike" (candle range > 20-period average).
    * **Signal Reversion (The "Fade"):** The system maintains a database of **150 specific time combos** where standard logic historically fails (Win Rate < 35%). In these specific windows (e.g., *Q1 Week 4 Friday NY PM*), the bot **inverts** the signal (Longs become Shorts), effectively fading the trap.

| Feature | Details |
| :--- | :--- |
| **Total Iterations** | **320 Unique Micro-Regimes** (4 Quarters × 4 Weeks × 5 Days × 4 Sessions) |
| **Risk Parameters** | **320 Fixed Pairs** (Hardcoded SL/TP for every window) |

**Examples of Optimized Parameters:**
| Time Context | Strategy Type | Stop Loss | Take Profit |
| :--- | :--- | :--- | :--- |
| `Q1_W1_FRI_NY_AM` | High Volatility | **8.68 pts** | **14.46 pts** |
| `Q4_W4_THU_ASIA` | Scalp | **0.78 pts** | **1.29 pts** |
| `Q3_W1_TUE_LONDON` | Balanced | **2.50 pts** | **3.00 pts** |

</details>

<details>
<summary><strong>B. Intraday Dip Strategy</strong></summary>

*A mean-reversion strategy designed to catch overextended moves relative to the daily open.*

* **Logic:** Tracks the **09:30 ET Open** price.
    * **Long Signal:** Price down **≥ 1.0%** from open + Z-Score **< -0.5** (Oversold) + Volatility Spike.
    * **Short Signal:** Price up **≥ 1.25%** from open + Z-Score **> 1.0** (Overbought) + Volatility Spike.
* **Risk:** **Dynamic SL/TP** (Infinite combinations calculated by the `OptimizedTPEngine`).

</details>

<details>
<summary><strong>C. Confluence Strategy (ICT)</strong></summary>

*A strict price-action model focusing on liquidity sweeps.*

* **Logic:** Tracks previous session High/Low. Waits for a "Sweep" (wick past the level) followed by a close back inside the range.
* **Confirmation:** The sweep must occur while price is inside a **Higher Time Frame FVG** AND near a **$12.50 Bank Level** (e.g., 4012.50, 4025.00).
* **Risk (Fixed):**
    * **TP:** 5.0 Points
    * **SL:** 2.0 Points

</details>

<details>
<summary><strong>D. ICT Model Strategy (Silver Bullet)</strong></summary>

*Hunts for specific "Silver Bullet" setups during the NY AM Session (09:30–11:00 ET).*

* **Bias:** Determines bias by comparing current price to the **10:00 AM Open** (Above = Bullish, Below = Bearish).
* **Manipulation:** Waits for price to sweep a key liquidity level (Previous Day Low or Bullish 5m FVG Low).
* **Trigger:** Enters when a **1-minute Inversion FVG** (a Bearish FVG that gets broken upward) is confirmed.
* **Risk:** **Dynamic** (Calculated by Engine).

</details>

<details>
<summary><strong>E. ORB Strategy (Opening Range Breakout)</strong></summary>

*Opening Range Breakout logic with retest confirmation.*

* **Logic:** Defines the High/Low of the first 15 minutes of the NY session (**09:30–09:45 ET**).
* **Filter:** If range > **15 points**, strategy disables (avoids chop).
* **Trigger:** Requires a **retest** of the range midpoint (50% level) after 09:45, followed by a break of the High (Long only).
* **Risk:** **Dynamic**.

</details>

<details>
<summary><strong>F. ML Physics Strategy</strong></summary>

*Uses 4 session-specific Neural Network models trained on velocity and "physics" features (Z-Scores of Price, ATR, Volume).*

**Features Used:**
- Price velocity and acceleration
- ATR-normalized movements
- Volume momentum
- Z-Score deviations

| Session | Stop Loss | Take Profit |
| :--- | :--- | :--- |
| **Asia** | 4.0 pts | 6.0 pts |
| **London** | 4.0 pts | 4.0 pts |
| **NY AM** | 10.0 pts | 4.0 pts |
| **NY PM** | 10.0 pts | 8.0 pts |

</details>

<details>
<summary><strong>G. Dynamic Engine 1</strong></summary>

*A massive library of indicator-based conditions wrapped into a single strategy engine.*

* **Total Sub-Strategies:** **235** distinct indicator-based strategies
* **Indicators Used:** RSI, MACD, Bollinger Bands, Stochastic, ADX, CCI, Williams %R, and more
* **Risk:** Each sub-strategy has its own unique SL/TP parameters defined within the engine
* **Selection:** Sub-strategies are selected based on current market regime and historical performance

</details>

<details>
<summary><strong>H. Dynamic Engine 2</strong></summary>

*A price-action focused engine with hardcoded pattern recognition.*

* **Total Sub-Strategies:** **167** price-action based strategies
* **Patterns:** Engulfing candles, pin bars, inside bars, outside bars, doji patterns
* **Structure:** Swing highs/lows, trend breaks, consolidation breakouts
* **Risk:** Unique parameters per sub-strategy
* **Features:**
    * Fade-edge trades with minimum SL floor
    * Positive R:R enforcement
    * Session-specific parameter optimization

</details>

<details>
<summary><strong>I. SMT Divergence Strategy</strong></summary>

*Smart Money Technique divergence detection between correlated instruments.*

* **Logic:** Monitors ES (E-mini S&P 500) and NQ (E-mini Nasdaq) for divergences
* **Bullish SMT:** NQ makes lower low while ES makes higher low (or holds)
* **Bearish SMT:** NQ makes higher high while ES makes lower high (or holds)
* **Confirmation:** Requires FVG confluence and session timing alignment
* **Use Case:** Institutional order flow detection

</details>

---

## 3. Defense Layer: Filters & Blockers

This is the most critical component. A signal from any strategy above **MUST** pass all relevant filters to be executed. Julie employs **10 filters** and **2 blockers**.

### Primary Filters

#### 1. Rejection Filter (Bias Establishment)
Tracks key levels and establishes directional bias based on price reactions.

**Tracked Levels:**
- Previous Day PM High/Low
- Previous Session High/Low
- Midnight ORB (Opening Range Breakout) High/Low

**Logic:**
| Event | Result |
|:---|:---|
| Price sweeps Low + closes back above | **Long Bias** established → Shorts BLOCKED |
| Price sweeps High + closes back below | **Short Bias** established → Longs BLOCKED |

**Key Features:**
- **1-Candle Close Confirmation:** Requires a full candle close beyond level to establish bias
- **Continuation Logic:** After rejection, allows continuation trades in bias direction
- **Breakout Threshold:** Minimum 1.0 points beyond level required
- **Bias Persistence:** Bias remains until opposing rejection occurs

#### 2. Chop Filter (320 Hierarchical Thresholds)
Detects consolidation/ranging markets using time-specific thresholds.

**States:**
| State | Description |
|:---|:---|
| `NORMAL` | Market trending, trades allowed |
| `IN_CHOP` | Consolidation detected, ALL trades blocked |
| `BREAKOUT_LONG/SHORT` | Initial breakout detected, awaiting confirmation |
| `CONFIRMED_LONG/SHORT` | Breakout confirmed, directional trades allowed |
| `FAILED_LONG/SHORT` | False breakout, reverts to chop |

**Features:**
- **320 Unique Thresholds:** One for each time context (e.g., `Q1_W1_MON_ASIA: 2.25 pts`)
- **Structure Validation:** Confirms breakouts with HH/HL (uptrend) or LH/LL (downtrend) patterns
- **"Fade the Range" Logic:** Identifies mean-reversion opportunities at range extremes
- **Volatility Scaling ("Accordion Effect"):** Thresholds expand/contract with volatility
- **Time Decay Tracking:** "The longer the base, the higher in space" - tracks consolidation duration

#### 3. Extension Filter (Exhaustion Detection)
Compares current range to historical percentiles to detect overextension.

**320 Hierarchical Thresholds Include:**
| Threshold Type | Description |
|:---|:---|
| `session_extended` | Session range exceeds normal extension |
| `session_extreme` | Session range at extreme extension |
| `daily_extended` | Daily range exceeds normal extension |
| `daily_extreme` | Daily range at extreme extension |

**States:**
| State | Action |
|:---|:---|
| `NORMAL` | All trades allowed |
| `EXTENDED_UP` | Longs blocked (upside exhausted) |
| `EXTENDED_DOWN` | Shorts blocked (downside exhausted) |
| `EXTREME_UP` | Longs blocked, fade shorts considered |
| `EXTREME_DOWN` | Shorts blocked, fade longs considered |

#### 4. Volatility Filter
Classifies market volatility regime and adjusts trading behavior.

**Regimes:**
| Regime | ATR Condition | Action |
|:---|:---|:---|
| `ULTRA_LOW` | ATR < 0.5 pts | Skip all trades |
| `LOW` | ATR < 1.0 pts | Increase SL (1.5x), reduce size (0.67x) |
| `NORMAL` | ATR 1.0-3.0 pts | Standard parameters |
| `HIGH` | ATR 3.0-5.0 pts | Tighter stops, larger targets |
| `EXTREME` | ATR > 5.0 pts | Reduced position size, wider stops |

#### 5. Trend Filter
Validates trade direction against multi-timeframe trend.

**Components:**
- **SMA Alignment:** 20/50/200 period moving average relationships
- **Higher Timeframe Bias:** 1H and 4H trend direction
- **Trend Strength:** ADX-based trend strength measurement

**Logic:**
- Longs blocked when all timeframes show bearish alignment
- Shorts blocked when all timeframes show bullish alignment

#### 6. Impulse Filter
Detects impulsive moves that may indicate continuation or exhaustion.

**Detection Criteria:**
- Candle range > 2x ATR
- Volume > 1.5x average
- Close near high/low of candle (>70% body)

**Action:**
- After bullish impulse: Shorts blocked for N bars
- After bearish impulse: Longs blocked for N bars

#### 7. HTF FVG Filter (Higher Timeframe Fair Value Gaps)
Scans 1-Hour and 4-Hour charts for Fair Value Gaps.

**Features:**
- **Memory:** Remembers FVG zones for up to **141 bars**
- **Resistance Logic:** Blocks longs directly below bearish 4H FVG
- **Support Logic:** Blocks shorts directly above bullish 4H FVG
- **Mitigation Tracking:** Removes FVGs once price fills the gap

#### 8. Bank Level Quarter Filter
Tracks institutional $12.50 bank levels relative to key reference points.

**Bank Levels:** $12.50 increments (e.g., 4012.50, 4025.00, 4037.50)

**Reference Points:**
- Previous Session High/Low
- Previous PM High/Low
- Midnight ORB High/Low

**Confirmation:** Requires **2-candle confirmation** at bank level before establishing bias

#### 9. Memory S/R Filter (Support/Resistance Memory)
Tracks historical support and resistance levels with touch count.

**Features:**
- Remembers levels where price reversed multiple times
- Weights levels by recency and touch count
- Blocks trades directly into strong S/R without confirmation

#### 10. News Filter
Blocks trading around high-impact economic events.

**Events Tracked:**
- FOMC announcements
- NFP (Non-Farm Payrolls)
- CPI/PPI releases
- GDP announcements

**Buffer Periods:**
- Pre-news: 15-30 minutes before event
- Post-news: 5-15 minutes after event (configurable)

---

### Blockers

#### 1. Dynamic Structure Blocker
Identifies local market structure to prevent trading into weak levels.

**Detection:**
- **Swing Highs/Lows:** Uses fractal logic (N bars left/right)
- **Equal Highs/Lows:** Detects liquidity pools (stops clustered at same level)

**Blocking Logic:**
| Structure | Action |
|:---|:---|
| **Weak Highs** (Equal Highs) | Shorts blocked (liquidity grab likely) |
| **Weak Lows** (Equal Lows) | Longs blocked (liquidity grab likely) |
| **Strong High** (Clear rejection) | Shorts allowed after confirmation |
| **Strong Low** (Clear rejection) | Longs allowed after confirmation |

#### 2. Directional Loss Blocker
Tracks consecutive losses by direction and temporarily blocks that direction.

**Logic:**
- Tracks win/loss by direction (Long/Short) per session
- After **N consecutive losses** in one direction, that direction is blocked
- Block persists until:
    - New session begins
    - Opposing direction wins
    - Cooldown period expires

**Purpose:** Prevents "revenge trading" in a direction the market is clearly not favoring

---

## 4. Risk Management System

### Risk Engine
The central risk management system that governs all position sizing and exposure.

| Feature | Description |
|:---|:---|
| **Kelly Criterion Sizing** | Optimal position size based on win rate and risk/reward |
| **Max Daily Loss** | Hard stop on daily losses (e.g., $500) |
| **Max Position Size** | Caps maximum contracts per trade |
| **Correlation Adjustment** | Reduces size when multiple correlated positions open |
| **Volatility Scaling** | Adjusts size inversely to current volatility |

### Circuit Breaker
Emergency stop system that halts all trading under specific conditions.

**Trigger Conditions:**
| Trigger | Threshold | Action |
|:---|:---|:---|
| **Daily Loss Limit** | -$500 | Halt all trading for day |
| **Consecutive Losses** | 5 losses in a row | 30-minute cooldown |
| **Drawdown %** | -3% of account | Reduce position size 50% |
| **Win Rate Collapse** | <25% over 20 trades | Review and reduce activity |

**Recovery:**
- Automatic reset at new trading day (6 PM ET)
- Manual override available for exceptional circumstances

### Dynamic SL/TP Engine
Calculates optimal stop loss and take profit using statistical models.

**Models Used:**
| Model | Purpose |
|:---|:---|
| **Shannon Entropy** | Measures market noise/randomness |
| **GARCH Volatility** | Predicts future volatility |
| **ATR Multiplier** | Base SL/TP on current range |

**Adjustments:**
| Condition | SL Multiplier | TP Multiplier |
|:---|:---|:---|
| High Entropy (Chop) | 0.95x | 0.95x |
| Low Entropy (Trend) | 1.0x | 1.35x |
| High Volatility | 1.25x | 1.5x |
| Low Volatility | 1.5x | 1.0x |

### Trade Management Features

| Feature | Description |
|:---|:---|
| **Break-Even** | Moves stop to entry + 1 tick at **40%** of TP target |
| **Trailing Stop** | Trails by ATR once **60%** of TP reached |
| **Early Exit** | Closes trades not profitable within N bars (prevents "zombie trades") |
| **Partial Profits** | Takes 50% off at first target, lets rest run |

---

## 5. LLM Integration & Optimization

### Gemini Optimizer
Julie integrates with **Google's Gemini AI** for continuous parameter optimization.

**Capabilities:**
| Feature | Description |
|:---|:---|
| **Performance Analysis** | Analyzes trade history to identify patterns |
| **Parameter Suggestions** | Recommends SL/TP adjustments based on recent performance |
| **Regime Detection** | Identifies when market regime has shifted |
| **Strategy Ranking** | Ranks strategies by current effectiveness |

**Optimization Cycle:**
1. **Data Collection:** Aggregates last N trades with full context
2. **Analysis Request:** Sends structured data to Gemini API
3. **Response Parsing:** Extracts parameter recommendations
4. **Validation:** Backtests suggestions against recent data
5. **Application:** Applies validated changes to live parameters

**Example Prompt Structure:**
```
Given the following trade data for Q1_W2_TUE_NY_AM:
- Win Rate: 42%
- Average Win: 3.2 pts
- Average Loss: 2.8 pts
- Current SL: 2.5 pts
- Current TP: 4.0 pts

Suggest optimized SL/TP parameters to improve expectancy.
```

**Safety Guards:**
- Maximum parameter change per cycle: ±20%
- Minimum sample size: 20 trades
- Confidence threshold: 70% before applying changes

---

## 6. Session Management & Quarterly Theory

### Session Manager
Manages trading sessions and applies Quarterly Theory concepts.

**Trading Sessions (All times ET):**
| Session | Start | End | Characteristics |
|:---|:---|:---|:---|
| **Asia** | 6:00 PM | 2:00 AM | Low volatility, range-bound |
| **London** | 2:00 AM | 5:00 AM | Trend initiation, breakouts |
| **NY AM** | 9:30 AM | 12:00 PM | Highest volatility, reversals |
| **NY PM** | 12:00 PM | 4:00 PM | Trend continuation or reversal |

### Quarterly Theory Integration
Julie incorporates ICT's Quarterly Theory for macro timing.

**Quarterly Shifts:**
| Quarter | Dates | Typical Behavior |
|:---|:---|:---|
| **Q1** | Jan-Mar | Accumulation, trend establishment |
| **Q2** | Apr-Jun | Distribution or continuation |
| **Q3** | Jul-Sep | Reversal or consolidation |
| **Q4** | Oct-Dec | Final move, volatility increase |

**Weekly Power of 3:**
- **Monday:** Accumulation/manipulation
- **Tuesday-Wednesday:** Distribution/expansion
- **Thursday-Friday:** Reversal or continuation

**Session Power of 3:**
- **Asia:** Accumulation
- **London:** Manipulation
- **NY:** Distribution

---

## 7. SMT Divergence Analysis

### SMT Analyzer
Detects Smart Money Technique divergences between correlated instruments.

**Correlated Pairs Monitored:**
| Primary | Secondary | Correlation |
|:---|:---|:---|
| ES (S&P 500) | NQ (Nasdaq) | High positive |
| ES | YM (Dow) | High positive |
| ES | RTY (Russell) | Moderate positive |

**Divergence Types:**
| Type | ES Action | NQ Action | Signal |
|:---|:---|:---|:---|
| **Bullish SMT** | Higher Low | Lower Low | Long ES |
| **Bearish SMT** | Lower High | Higher High | Short ES |
| **Hidden Bullish** | Lower Low | Higher Low | Strong Long |
| **Hidden Bearish** | Higher High | Lower High | Strong Short |

**Confirmation Requirements:**
- Divergence must occur at key level (session high/low, bank level)
- Volume confirmation preferred
- FVG confluence increases probability

---

## 8. Dynamic Parameters System

### The 320 Parameter Matrix
Every time context has its own set of optimized parameters stored in `dynamic_sltp_params.py`.

**Parameters per Context:**
```python
{
    "Q1_W1_MON_ASIA": {
        "sl": 2.50,
        "tp": 4.00,
        "chop_threshold": 2.25,
        "extension_session": 0.85,
        "extension_daily": 0.90,
        "volatility_multiplier": 1.0,
        "max_trades": 3,
        "min_rr": 1.5
    },
    # ... 319 more contexts
}
```

### Parameter Categories

| Category | Description |
|:---|:---|
| **SL/TP** | Stop Loss and Take Profit in points |
| **Chop Threshold** | Minimum range to consider market "trending" |
| **Extension Percentiles** | Session/daily range exhaustion levels |
| **Volatility Multiplier** | Scaling factor for current volatility regime |
| **Max Trades** | Maximum trades allowed in this context |
| **Min R:R** | Minimum risk/reward ratio required |

### Configuration File (config.py)

**Core Settings:**
```python
# Risk Settings
MAX_DAILY_LOSS = 500
MAX_POSITION_SIZE = 5
DEFAULT_SL = 3.0
DEFAULT_TP = 5.0

# Session Settings
ASIA_START = "18:00"
LONDON_START = "02:00"
NY_AM_START = "09:30"
NY_PM_START = "12:00"

# Filter Settings
CHOP_LOOKBACK = 20
FVG_MEMORY_BARS = 141
REJECTION_CONFIRM_BARS = 1

# LLM Settings
GEMINI_API_KEY = "..."
OPTIMIZATION_INTERVAL = 86400  # Daily
```

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         JULIE SYSTEM                            │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   STRATEGY LAYER                         │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────────┐   │   │
│  │  │ Regime  │ │   ICT   │ │   ORB   │ │ ML Physics  │   │   │
│  │  └────┬────┘ └────┬────┘ └────┬────┘ └──────┬──────┘   │   │
│  │  ┌────┴────┐ ┌────┴────┐ ┌────┴────┐ ┌──────┴──────┐   │   │
│  │  │Dyn Eng 1│ │Dyn Eng 2│ │  SMT    │ │Intraday Dip │   │   │
│  │  └────┬────┘ └────┬────┘ └────┬────┘ └──────┬──────┘   │   │
│  └───────┼──────────┼──────────┼──────────────┼───────────┘   │
│          │          │          │              │               │
│          ▼          ▼          ▼              ▼               │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   DEFENSE LAYER                          │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │   │
│  │  │Rejection │ │   Chop   │ │Extension │ │Volatility│   │   │
│  │  │  Filter  │ │  Filter  │ │  Filter  │ │  Filter  │   │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘   │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │   │
│  │  │  Trend   │ │ Impulse  │ │ HTF FVG  │ │Bank Level│   │   │
│  │  │  Filter  │ │  Filter  │ │  Filter  │ │  Filter  │   │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘   │   │
│  │  ┌──────────┐ ┌──────────┐ ┌───────────────────────┐   │   │
│  │  │Memory SR │ │  News    │ │ Structure │ Dir Loss  │   │   │
│  │  │  Filter  │ │  Filter  │ │  Blocker  │ Blocker   │   │   │
│  │  └──────────┘ └──────────┘ └───────────────────────┘   │   │
│  └─────────────────────────┬───────────────────────────────┘   │
│                            │                                    │
│                            ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   RISK LAYER                             │   │
│  │  ┌────────────┐  ┌─────────────┐  ┌─────────────────┐   │   │
│  │  │Risk Engine │  │Circuit Break│  │Dynamic SL/TP    │   │   │
│  │  │            │  │             │  │    Engine       │   │   │
│  │  └────────────┘  └─────────────┘  └─────────────────┘   │   │
│  └─────────────────────────┬───────────────────────────────┘   │
│                            │                                    │
│                            ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   EXECUTION LAYER                        │   │
│  │           ProjectX Gateway (TopstepX API)                │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   OPTIMIZATION LAYER                     │   │
│  │  ┌────────────────┐      ┌────────────────────────┐     │   │
│  │  │Session Manager │      │   Gemini Optimizer     │     │   │
│  │  │(Quarterly Theory)     │   (LLM Integration)    │     │   │
│  │  └────────────────┘      └────────────────────────┘     │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
JULIE/
├── julie001.py                      # Main entry point (asyncio-based)
├── config.py                        # Configuration settings
├── dynamic_sltp_params.py           # 320 hierarchical parameters
├── regime_sltp_params.py            # Regime-specific parameters
│
├── UI & Launching
│   ├── launch_ui.py                 # UI launcher with dependency checks
│   ├── julie_tkinter_ui.py          # Modern Tkinter dashboard (v2.0)
│   ├── julie_ui.py                  # API/log monitoring utilities
│   ├── account_selector.py          # Account selection interface
│   └── TKINTER_UI_README.md         # UI documentation
│
├── Core Async Components
│   ├── async_market_stream.py       # AsyncIO market data manager
│   ├── async_tasks.py               # Background task management
│   ├── client.py                    # ProjectX API client
│   └── event_logger.py              # Structured event logging
│
├── Strategies
│   ├── regime_strategy.py           # Regime Adaptive Strategy
│   ├── intraday_dip_strategy.py     # Mean-reversion strategy
│   ├── confluence_strategy.py       # ICT Confluence
│   ├── ict_model_strategy.py        # Silver Bullet
│   ├── orb_strategy.py              # Opening Range Breakout
│   ├── ml_physics_strategy.py       # Neural Network (4 session models)
│   ├── dynamic_engine_strategy.py   # 235 indicator sub-strategies
│   ├── dynamic_engine2_strategy.py  # 167 price-action sub-strategies
│   ├── smt_strategy.py              # SMT Divergence
│   ├── vixmeanreversion.py          # VIX reversion strategy
│   ├── strategy_base.py             # Base strategy class
│   ├── dynamic_signal_engine.py     # Dynamic Engine 1 implementation
│   └── dynamic_signal_engine2.py    # Dynamic Engine 2 implementation
│
├── Filters
│   ├── rejection_filter.py          # Bias establishment filter
│   ├── chop_filter.py               # Consolidation detection
│   ├── extension_filter.py          # Exhaustion detection
│   ├── volatility_filter.py         # Regime classification
│   ├── trend_filter.py              # Multi-timeframe trend
│   ├── impulse_filter.py            # Momentum detection
│   ├── htf_fvg_filter.py            # Higher timeframe FVG
│   ├── bank_level_quarter_filter.py # Institutional levels
│   ├── memory_sr_filter.py          # Historical S/R levels
│   └── news_filter.py               # Economic event blocking
│
├── Blockers
│   ├── dynamic_structure_blocker.py # Structure-based blocking
│   └── directional_loss_blocker.py  # Consecutive loss prevention
│
├── Risk Management
│   ├── risk_engine.py               # Position sizing & TP calculation
│   ├── circuit_breaker.py           # Emergency stop system
│   └── param_scaler.py              # Parameter scaling utilities
│
├── Analysis & Optimization
│   ├── smt_analyzer.py              # SMT divergence detection
│   ├── session_manager.py           # Session & quarterly theory
│   ├── gemini_optimizer.py          # LLM-powered optimization
│   ├── yahoo_vix_client.py          # VIX data integration
│   └── dynamic_chop.py              # Dynamic chop analysis
│
└── Resources
    ├── README.md                    # This file
    ├── ASYNCIO_UPGRADE_SUMMARY.md   # Async migration notes
    ├── logo.gif                     # UI logo asset
    └── *.csv                        # Historical data files
```

---

## Quick Reference

### Filter Summary Table

| Filter | Purpose | Blocks |
|:---|:---|:---|
| Rejection | Bias establishment | Opposite direction after sweep |
| Chop | Consolidation detection | All trades during chop |
| Extension | Exhaustion detection | Direction of exhaustion |
| Volatility | Regime classification | Ultra-low vol trades |
| Trend | Direction validation | Counter-trend trades |
| Impulse | Momentum detection | Counter-impulse trades |
| HTF FVG | Resistance/Support zones | Trades into unfilled FVGs |
| Bank Level | Institutional levels | Trades against confirmed bias |
| Memory S/R | Historical levels | Trades into strong S/R |
| News | Event risk | All trades around news |

### Strategy Quick Reference

| Strategy | Style | Time Focus | Risk Type |
|:---|:---|:---|:---|
| Regime | Trend/Fade | All sessions | 320 Fixed |
| Intraday Dip | Mean Reversion | NY | Dynamic |
| Confluence | Sweep & Reject | All | Fixed |
| ICT Model | Silver Bullet | NY AM | Dynamic |
| ORB | Breakout | NY Open | Dynamic |
| ML Physics | Neural Net | All | Session-based |
| Dynamic Engine 1 | Indicator | All | Per-strategy |
| Dynamic Engine 2 | Price Action | All | Per-strategy |
| SMT | Divergence | All | Dynamic |

---

## Configuration

### API Credentials
Edit `config.py` to configure your TopstepX credentials:

```python
CONFIG = {
    "USERNAME": "your_topstepx_username",
    "API_KEY": "your_topstepx_api_key",
    "ACCOUNT_ID": None,  # Auto-fetched or set via env var JULIE_ACCOUNT_ID
    # ... other settings
}
```

### Risk Settings
Key risk parameters in `config.py`:

```python
CONFIG = {
    "MAX_DAILY_LOSS": 1000.0,  # Maximum daily loss in dollars
    "RISK": {
        "POINT_VALUE": 5.0,      # MES = $5 per point
        "FEES_PER_SIDE": 2.50,   # Commission per side
        "MIN_NET_PROFIT": 10.0,  # Minimum profit threshold
        "CONTRACTS": 1           # Position size
    }
}
```

### Environment Variables
- `JULIE_ACCOUNT_ID`: Override account ID selection

---

## Troubleshooting

### Common Issues

#### "ModuleNotFoundError: No module named 'X'"
**Solution:** Install missing dependencies:
```bash
# macOS/Linux
pip3 install requests pandas numpy

# Windows
pip install requests pandas numpy
```

#### "Authentication failed" or "401 Unauthorized"
**Solution:**
1. Verify your credentials in `config.py`
2. Ensure your TopstepX account is active
3. Check that your API key hasn't expired

#### UI won't start
**Solution:**
```bash
# macOS/Linux
sudo apt-get install python3-tk  # Ubuntu/Debian
brew install python-tk@3.11      # macOS

# Windows - tkinter comes with Python installer
# Reinstall Python and ensure "tcl/tk" is checked
```

#### Bot stops trading unexpectedly
**Solution:** Check the logs for:
- Circuit breaker triggers (`topstep_live_bot.log`)
- Daily loss limits reached
- Connection issues with TopstepX API
- Missing market data

#### Python version issues
**Solution:**
```bash
# Check your Python version
python3 --version  # Should be 3.11 or higher

# Install Python 3.11+ if needed
# macOS:
brew install python@3.11

# Windows:
# Download from https://www.python.org/downloads/
```

### Logs and Debugging
- **Trading Log**: `topstep_live_bot.log` - Contains all bot activities
- **Event Log**: Visible in Tkinter UI dashboard
- **Debug Mode**: Set `logging.DEBUG` in `julie001.py` for verbose output

### Performance Optimization
1. **Reduce Update Frequency**: Edit timing values in `julie_tkinter_ui.py`
2. **Disable Unused Strategies**: Comment out strategy imports in `julie001.py`
3. **Limit Historical Data**: Reduce lookback periods in filters

---

## System Requirements

### Minimum Requirements
- **CPU**: 2+ cores
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 500MB for application + logs
- **Network**: Stable internet connection (low latency preferred)
- **OS**:
  - macOS 10.15+
  - Windows 10/11
  - Linux (Ubuntu 20.04+, Debian 10+)

### Recommended Requirements
- **CPU**: 4+ cores
- **RAM**: 16GB
- **Network**: <50ms latency to TopstepX servers
- **Display**: 1920x1080 or higher (for UI)

---

## Advanced Topics

### Running on a VPS
For 24/7 operation, consider deploying on a VPS:

```bash
# Using screen to keep bot running
screen -S julie
python3 julie001.py
# Detach: Ctrl+A, then D
# Reattach: screen -r julie
```

### Multiple Instances
To run multiple accounts simultaneously:
1. Create separate directories for each instance
2. Configure different `ACCOUNT_ID` in each `config.py`
3. Run each instance in its own terminal/screen session

### Monitoring and Alerts
Consider setting up:
- **Log monitoring**: Use `tail -f topstep_live_bot.log`
- **Email alerts**: Integrate with Gmail API for trade notifications
- **SMS alerts**: Use Twilio API for critical events

---

## Changelog

### v2.0.0 (2025)
- ✨ Added full asyncio architecture with `async_market_stream.py` and `async_tasks.py`
- ✨ New modern Tkinter UI dashboard (`julie_tkinter_ui.py`)
- ✨ Yahoo VIX integration for volatility strategies
- ✨ Enhanced event logging system
- ✨ Improved error handling and circuit breaker logic
- 🔧 Refactored signal discovery and execution architecture
- 🐛 Fixed time variable errors and async bugs

### v1.0.0 (2023-2024)
- Initial release with 9 strategy engines
- 320 hierarchical threshold system
- Defense layer with 10 filters and 2 blockers
- Gemini AI optimizer integration
- Dynamic SL/TP engine

---

## Support & Contributing

### Getting Help
1. Check this README and [TKINTER_UI_README.md](TKINTER_UI_README.md)
2. Review logs in `topstep_live_bot.log`
3. Check [ASYNCIO_UPGRADE_SUMMARY.md](ASYNCIO_UPGRADE_SUMMARY.md) for async architecture details

### Security
- **Never commit credentials**: Keep `config.py` with real credentials in `.gitignore`
- **API Key Safety**: Store API keys in environment variables for production
- **Audit Logs**: Review `topstep_live_bot.log` regularly

---

*Julie v2.0.0 - Built for precision, optimized for survival, powered by async.*
