# Julie: Advanced MES Futures Trading System

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg) ![Market](https://img.shields.io/badge/Market-MES%20Futures-green.svg) ![Platform](https://img.shields.io/badge/Platform-TopstepX-orange.svg) ![License](https://img.shields.io/badge/License-Proprietary-red.svg)

**Julie** is a high-frequency, session-specialized algorithmic trading bot built to execute autonomously on the **ProjectX Gateway (TopstepX)**. Unlike traditional bots that use a single logic set, Julie functions as an orchestrator for a "Team of Rivals"—a portfolio of **7 distinct strategy classes** that compete to find the best entry, all governed by a central "Defense Layer" of dynamic filters.

---

## 📖 Table of Contents
- [1. System Architecture](#1-system-architecture)
- [2. Strategy Biography (Deep Dive)](#2-strategy-biography-deep-dive)
- [3. Defense Layer: Filters & Risk](#3-defense-layer-filters--risk)

---

## 1. System Architecture

### Core Philosophy
* **Micro-Regime Specialization:** The market is not treated as a monolith. Julie fragments the trading year into **320 specific time contexts** (e.g., *"Q1 Week 3 Tuesday London Session"*) and applies unique risk parameters to each.
* **Defensive Priority:** A trade signal is only an "application." The central filter system acts as a strict "underwriter," rejecting any application that violates market structure, volatility limits, or institutional bias constraints.
* **Dynamic Risk:** Stop Losses (SL) and Take Profits (TP) breathe with the market, expanding during high volatility and contracting during chop, determined by real-time **Shannon Entropy** and **GARCH** volatility models.

---

## 2. Strategy Biography (Deep Dive)

Julie runs **7 primary strategy engines** simultaneously. Click the arrows below to expand the logic for each strategy.

<details>
<summary><strong> A. The Regime Adaptive Strategy (The Core)</strong></summary>

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
<summary><strong> B. Intraday Dip Strategy</strong></summary>

*A mean-reversion strategy designed to catch overextended moves relative to the daily open.*

* **Logic:** Tracks the **09:30 ET Open** price.
    * **Long Signal:** Price down **≥ 1.0%** from open + Z-Score **< -0.5** (Oversold) + Volatility Spike.
    * **Short Signal:** Price up **≥ 1.25%** from open + Z-Score **> 1.0** (Overbought) + Volatility Spike.
* **Risk:** **Dynamic SL/TP** (Infinite combinations calculated by the `OptimizedTPEngine`).

</details>

<details>
<summary><strong> C. Confluence Strategy (ICT)</strong></summary>

*A strict price-action model focusing on liquidity sweeps.*

* **Logic:** Tracks previous session High/Low. Waits for a "Sweep" (wick past the level) followed by a close back inside the range.
* **Confirmation:** The sweep must occur while price is inside a **Higher Time Frame FVG** AND near a **$12.50 Bank Level** (e.g., 4012.50, 4025.00).
* **Risk (Fixed):**
    * **TP:** 5.0 Points
    * **SL:** 2.0 Points

</details>

<details>
<summary><strong> D. ICT Model Strategy (Silver Bullet)</strong></summary>

*Hunts for specific "Silver Bullet" setups during the NY AM Session (09:30–11:00 ET).*

* **Bias:** Determines bias by comparing current price to the **10:00 AM Open** (Above = Bullish, Below = Bearish).
* **Manipulation:** Waits for price to sweep a key liquidity level (Previous Day Low or Bullish 5m FVG Low).
* **Trigger:** Enters when a **1-minute Inversion FVG** (a Bearish FVG that gets broken upward) is confirmed.
* **Risk:** **Dynamic** (Calculated by Engine).

</details>

<details>
<summary><strong> E. ORB Strategy</strong></summary>

*Opening Range Breakout logic.*

* **Logic:** Defines the High/Low of the first 15 minutes of the NY session (**09:30–09:45 ET**).
* **Filter:** If range > **15 points**, strategy disables (avoids chop).
* **Trigger:** Requires a **retest** of the range midpoint (50% level) after 09:45, followed by a break of the High (Long only).
* **Risk:** **Dynamic**.

</details>

<details>
<summary><strong> F. ML Physics Strategy</strong></summary>

*Uses 4 session-specific Neural Network models trained on velocity and "physics" features (Z-Scores of Price, ATR, Volume).*

| Session | Stop Loss | Take Profit |
| :--- | :--- | :--- |
| **Asia** | 4.0 pts | 6.0 pts |
| **London** | 4.0 pts | 4.0 pts |
| **NY AM** | 10.0 pts | 4.0 pts |
| **NY PM** | 10.0 pts | 8.0 pts |

</details>

<details>
<summary><strong>⚙️ G. Dynamic Engines 1 & 2</strong></summary>

*Massive libraries of hardcoded conditions wrapped into single strategies.*

* **Engine 1:** **235** distinct indicator-based strategies.
* **Engine 2:** **167** price-action based strategies.
* **Risk:** Each sub-strategy has its own unique parameters defined within the engine files.

</details>

---

## 3. Defense Layer: Filters & Risk

This is the most critical component. A signal from any strategy above **MUST** pass all relevant filters to be executed.

### The Filter Stack

1.  **Rejection Filter (Bias):**
    * Tracks Highs/Lows of Prev Day PM, Prev Session, and Midnight ORB.
    * **Logic:**
        * **Long Bias:** If price sweeps a Low and closes back above, it establishes a **Long Bias**, blocking all Short signals.
        * **Short Bias:** If price sweeps a High and closes back below, it establishes a **Short Bias**, blocking all Long signals.

2.  **HTF FVG Filter (Memory):**
    * Scans 1-Hour and 4-Hour charts for Fair Value Gaps.
    * **Logic:** It remembers these zones for up to 141 bars. If a Long signal is generated directly below a Bearish 4H FVG (Resistance), the trade is **BLOCKED**.

3.  **Chop Filter (320 Thresholds):**
    * Uses the same 320 time-buckets as the Regime strategy.
    * **Logic:** Checks the 20-bar range. If `Current Range < Historical Chop Threshold` (e.g., < 2.25 pts in Asia), the market is flagged as "Choppy" and **ALL** trades are blocked until a breakout occurs.

4.  **Extension Filter (Exhaustion):**
    * Compares current session range to historical 90th percentile data.
    * **Logic:** If the session range is statistically "Extended" (e.g., > 124 pts in Q1 Mon Asia) and price is near the high, **Longs are blocked** (upside exhausted).

5.  **Dynamic Structure Blocker:**
    * Identifies local Swing Highs/Lows (Fractals).
    * **Logic:** Blocks Shorts entering directly into "Weak Highs" (Equal Highs) and Longs entering into "Weak Lows" (Equal Lows).

6.  **Bank Level Quarter Filter:**
    * Tracks $12.50 bank levels relative to Previous Session/PM/ORB.
    * **Logic:** Blocks trades that contradict a confirmed bank level rejection bias (requires 2-candle confirmation).

### Risk Management Features

| Feature | Description |
| :--- | :--- |
| **Dynamic SL/TP Engine** | Calculates risk using **Shannon Entropy** (market noise) and **GARCH** volatility. High Entropy (Chop) reduces targets (0.95x); High Trend expands them (1.35x). |
| **Volatility Regime Sizing** | **Ultra-Low Vol:** Skips trades.<br>**Low Vol:** Increases SL (1.5x) but reduces Size (0.67x) to prevent noise stops. |
| **Break-Even** | Automatically moves Stop to Entry + 1 tick once the trade is **40%** toward its Take Profit target. |
| **Early Exit** | Forcibly closes trades that don't turn green within a set number of bars (e.g., 5 bars for Confluence) to prevent "zombie trades." |

---
