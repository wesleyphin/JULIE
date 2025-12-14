Julie: The Definitive Documentation
This comprehensive manual details the architecture, strategy logic, risk management systems, and operational procedures for Julie, an advanced algorithmic trading system designed for the E-mini S&P 500 (ES) futures market.

1. General Biography & System Architecture
Julie is a high-frequency, session-specialized trading bot built to execute autonomously on the ProjectX Gateway (TopstepX). Unlike traditional bots that use a single logic set, Julie functions as an orchestrator for a "Team of Rivals"—a portfolio of 7 distinct strategy classes that compete to find the best entry, all governed by a central "Defense Layer" of filters.

Core Philosophy
Micro-Regime Specialization: The market is not treated as a monolith. Julie fragments the trading year into 320 specific time contexts (e.g., "Q1 Week 3 Tuesday London Session") and applies unique risk parameters to each.

Defensive Priority: A trade signal is only an "application." The central filter system acts as a strict "underwriter," rejecting any application that violates market structure, volatility limits, or institutional bias constraints.

Dynamic Risk: Stop Losses (SL) and Take Profits (TP) breathe with the market, expanding during high volatility and contracting during chop, determined by real-time Shannon Entropy and GARCH volatility models.

2. In-Depth Strategy Biography
Julie runs 7 primary strategy engines simultaneously. Below is the detailed breakdown of every strategy, its iterations, and its specific risk parameters.

A. The Regime Adaptive Strategy (The Core)
This is the system's flagship strategy. It does not use generic indicators but relies on historical probabilities specific to the current moment in time.

Logic:

Trend: Uses SMA20 vs SMA200 crossover + Low Volatility regime to identify trend.

Trigger: Enters on pullbacks (Long) or rallies (Short) that exhibit a "Range Spike" (candle range > 20-period average).

Signal Reversion (The "Fade"): The system maintains a database of 150 specific time combos where standard logic historically fails (WR < 35%). In these specific windows (e.g., Q1 Week 4 Friday NY PM), the bot inverts the signal (Longs become Shorts), effectively fading the trap.

Iterations & Combinations
Total Iterations: 320 Unique Micro-Regimes.

Calculation: 4 Yearly Quarters × 4 Monthly Weeks × 5 Days of Week × 4 Sessions = 320.

SL/TP Combinations: 320 Fixed Pairs.

Every single one of the 320 time buckets has its own hardcoded SL and TP optimized for that specific window.

Example 1 (High Vol): Q1_W1_FRI_NY_AM → SL: 8.68 pts / TP: 14.46 pts.

Example 2 (Scalp): Q4_W4_THU_ASIA → SL: 0.78 pts / TP: 1.29 pts.

B. Intraday Dip Strategy
A mean-reversion strategy designed to catch overextended moves relative to the daily open.

Logic: Tracks the 09:30 ET Open price.

Long: Price down ≥ 1.0% from open + Z-Score < -0.5 (Oversold) + Volatility Spike.

Short: Price up ≥ 1.25% from open + Z-Score > 1.0 (Overbought) + Volatility Spike.

SL/TP: Dynamic (Infinite combinations). Uses the OptimizedTPEngine to calculate targets based on live volatility.

C. Confluence Strategy (ICT)
A strict price-action model focusing on liquidity sweeps.

Logic: Tracks previous session High/Low. It waits for a "Sweep" (wick past the level) followed by a close back inside the range.

Confirmation: The sweep must occur while price is inside a Higher Time Frame Fair Value Gap (FVG) AND near a $12.50 Bank Level (e.g., 4012.50).

SL/TP: Fixed.

TP: 5.0 Points.

SL: 2.0 Points.

D. ICT Model Strategy (Silver Bullet)
Hunts for specific "Silver Bullet" setups during the NY AM Session (09:30–11:00 ET).

Logic:

Determines bias by comparing current price to the 10:00 AM Open.

Waits for a "Manipulation" move that sweeps Previous Day Low (PDL) or an FVG.

Trigger: Enters when a 1-minute Inversion FVG (a Bearish FVG that gets broken upward) is confirmed.

SL/TP: Dynamic (Calculated by Engine).

E. ORB Strategy (Opening Range Breakout)
Logic: Defines the High/Low of the first 15 minutes of the NY session (09:30–09:45 ET).

Filter: If range > 15 points, strategy disables (avoids chop).

Trigger: Requires a retest of the range midpoint (50% level) after 09:45, followed by a break of the High (Long only).

SL/TP: Dynamic.

F. ML Physics Strategy
Uses 4 session-specific Neural Network models trained on velocity and "physics" features (Z-Scores of Price, ATR, Volume).

Iterations: 4 Session Models (Asia, London, NY_AM, NY_PM).

SL/TP Combos: 4 distinct sets defined in config.py:

G. Dynamic Engines 1 & 2
These are massive libraries of hardcoded conditions wrapped into single strategies.

Engine 1: 235 distinct indicator-based strategies.

Engine 2: 167 price-action based strategies.

SL/TP: Each sub-strategy has its own parameters defined within the engine files.

3. Filters & Risk Management Deep Dive
This is the "Defense Layer." A signal from any strategy above MUST pass all relevant filters to be executed.

A. The Filter Stack
Rejection Filter (Bias):

Tracks Highs/Lows of Prev Day PM, Prev Session, and Midnight ORB.

Logic: If price sweeps a Low and closes back above, it establishes a Long Bias. The filter will then BLOCK any Short signals until the bias flips.

HTF FVG Filter (Memory):

Scans 1-Hour and 4-Hour charts for Fair Value Gaps.

Logic: It remembers these zones for up to 141 bars. If a Long signal is generated directly below a Bearish 4H FVG (Resistance), the trade is BLOCKED.

Chop Filter (320 Thresholds):

Uses the same 320 time-buckets as the Regime strategy.

Logic: Checks the 20-bar range. If Current Range < Historical Chop Threshold (e.g., < 2.25 pts in Asia), the market is flagged as "Choppy" and ALL trades are blocked until a breakout occurs.

Extension Filter (Exhaustion):

Compares current session range to historical 90th percentile data.

Logic: If the session range is statistically "Extended" (e.g., > 124 pts in Q1 Mon Asia) and price is near the high, Longs are blocked (upside exhausted).

Dynamic Structure Blocker:

Identifies local Swing Highs/Lows (Fractals).

Logic: Blocks Shorts entering directly into "Weak Highs" (Equal Highs) and Longs entering into "Weak Lows" (Equal Lows) to prevent being trapped in a liquidity sweep.

B. Risk Management Features
Dynamic SL/TP Engine (Entropy Model):

For strategies without fixed params, this engine calculates risk based on Shannon Entropy (market noise) and GARCH volatility.

High Entropy (Chop): Targets are reduced (e.g., 0.95x multiplier).

High Trend: Targets are expanded (e.g., 1.35x multiplier).

Volatility Regime Sizing:

Classifies volatility into 4 tiers: Ultra-Low, Low, Normal, High.

Ultra-Low: Skips all trades.

Low Vol: Increases Stop Loss distance (1.5x) but reduces Position Size (0.67x) to prevent getting stopped out by noise.

Active Management:

Break-Even: Automatically moves Stop Loss to Entry + 1 tick once the trade is 40% toward its Take Profit target.

Early Exit: Forcibly closes trades that don't turn green within a set number of bars (e.g., 5 bars for Confluence), preventing "zombie trades".

4. Usage Guide: How to Run Julie
Prerequisites (Both Platforms)
ProjectX Account: You must have a valid TopstepX username and API Key.

Python 3.8+: Installed on your system.

Dependencies: Install required libraries.

Bash

pip install requests pandas numpy pytz joblib
Configuration:

Open config.py in a text editor.

Enter your USERNAME and API_KEY inside the quotes.

(Optional) Set ACCOUNT_ID if you want to skip the selection menu.

Windows Execution
There are two ways to run the bot on Windows.

Option A: The Batch File (Easiest)

Locate the file named Rose.bat in the folder.

Double-click Rose.bat.

A command prompt will open, automatically launch the bot, and display the startup sequence.

Option B: Command Line

Press Win + R, type cmd, and press Enter.

Navigate to the bot's folder:

DOS

cd path\to\julie_folder
Run the bot:

DOS

python julie001.py
Apple (macOS) Execution
Open the Terminal app (Command + Space, type "Terminal").

Navigate to the folder where you saved the files:

Bash

cd /path/to/julie_folder
Run the bot:

Bash

python3 julie001.py
(Note: macOS usually requires python3 instead of python).

Startup Sequence
Once running (on either platform):

Authentication: The bot will log in to TopstepX.

Account Selection: If you didn't hardcode an ID, a list of accounts will appear. Type the number (e.g., 1) and hit Enter.

Backfill: The bot will say "Performing one-time backfill...". It is downloading 500 minutes of data to reconstruct the "Rejection Filter" and "ORB" lines. This takes a few seconds.

Live Mode: You will see "Heartbeat" messages (e.g., 💓 Heartbeat... Price: 4055.25) every 30 seconds. This means the bot is live and hunting for trades.

Stopping the Bot: Press Ctrl + C in the terminal/command window to safely shut down the process. Note that this does not close open trades; it only stops the bot from managing them or taking new ones.
