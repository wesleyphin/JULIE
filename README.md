# Julie: Advanced MES Futures Trading System (TopstepX / ProjectX Gateway)

Julie is an automated, session-specialized futures trading system built for MES
(and ES for training/backtests) on the TopstepX ProjectX Gateway. It runs a portfolio
of strategies, routes every candidate through a defense layer of filters and blockers,
and places fixed OCO brackets at entry. This README is intentionally detailed and
mirrors the current repository behavior.

## Metadata

| Item | Value |
| --- | --- |
| Platform | TopstepX ProjectX Gateway (REST + SignalR) |
| Primary Contract | MES (ES supported for training/backtests) |
| Core Timeframe | 1m (some ML sessions use 3m) |
| Language | Python 3.11+ |
| UI | Tkinter dashboard (optional) |
| Backtest | backtest_mes_et.py / backtest_mes_et_ui.py |
| Optimizer | Gemini (optional, rate-limited) |
| Units | Points (tick_size = 0.25 for MES) |

## Table of Contents

- Quick Start
- System Architecture
- Modern Features
- Session Schedule
- Strategy Layer
- Strategy Biographies (Deep Dive)
- Defense Layer (Filters and Blockers)
- Risk, SL/TP, and Viability
- Dynamic Parameters System
- ML Physics (Models and Guardrails)
- ASIA Calibrations and Smooth Trend Asia
- Continuation and Rescue Flow
- News and Holiday Filtering
- Gemini Optimizer (Optional)
- Session Manager and Quarterly Theory
- Backtesting
- File Layout
- Troubleshooting
- Safety and Notes

## Quick Start

### Prerequisites
- Python 3.11+
- TopstepX account and API key
- Internet connection

### Install

```bash
pip install -r freeze.txt
```

Minimal install:

```bash
pip install requests pandas numpy
```

### Configure credentials

Edit `config.py`:

```python
CONFIG["USERNAME"] = "your_username"
CONFIG["API_KEY"] = "your_api_key"
```

### Run the live bot

```bash
python julie001.py
```

### Run the UI (Tkinter)

```bash
python launch_ui.py
```

### Run a backtest

```bash
python backtest_mes_et.py
```

Optional backtest UI:

```bash
python backtest_mes_et_ui.py
```

## System Architecture

Julie is structured as a layered pipeline: strategies generate candidates, filters decide
eligibility, risk and SL/TP assign brackets, and the execution layer submits orders.

```
                           JULIE SYSTEM

   Market Data (SignalR) + Historical Bars (REST)
                         |
                         v
               Feature/Context Builder
     (session, vol regime, levels, indicators, ATR, VWAP)
                         |
                         v
                   Strategy Layer
      (signals from multiple engines, per-session)
                         |
                         v
               Defense Layer (Filters)
   (chop, extension, trend, impulse, HTF FVG, news, etc)
                         |
                         v
                Blockers and Guardrails
      (structure, directional loss, penalty box, ML guards)
                         |
                         v
              Risk and SL/TP Assignment
          (fixed brackets or dynamic parameters)
                         |
                         v
                Execution (TopstepX API)
                         |
                         v
                   Trade Management
              (logging, state, backtest)
                         |
                         v
              Optional Optimizer (Gemini)
```

Key runtime modules:
- `julie001.py` orchestrates live execution and filter order.
- `async_market_stream.py` and `async_tasks.py` manage data/heartbeat/HTF updates.
- `event_logger.py` and `topstep_live_bot.log` provide structured logs.
- `filter_arbitrator.py` can arbitrate between legacy and upgraded filter systems.

## Modern Features

Julie includes several modern runtime features that mirror the original design goals:

| Feature | Purpose | Primary Files |
| --- | --- | --- |
| Async market streaming | Non-blocking SignalR market data | async_market_stream.py |
| Background tasks | Heartbeat, position sync, HTF structure | async_tasks.py |
| Tkinter UI | Live status, logs, and health | launch_ui.py, julie_tkinter_ui.py |
| Event logger | Structured log entries for audits | event_logger.py |
| VIX integration | VIX-based strategy support | yahoo_vix_client.py, vixmeanreversion.py |
| Gemini optimizer | Runtime tuning of select parameters | gemini_optimizer.py |
| Filter arbitrator | Legacy vs upgraded filter arbitration | filter_arbitrator.py |

## Session Schedule

Sessions are defined in `CONFIG["SESSIONS"]` (ET):

| Session | Hours (ET) | Notes |
| --- | --- | --- |
| ASIA | 18:00-02:59 | Lower volatility / smoother trends |
| LONDON | 03:00-07:59 | Trend initiation / breakouts |
| NY_AM | 08:00-11:59 | Highest volatility / reversals |
| NY_PM | 12:00-16:59 | Continuation or rotation |

(Exact hours are defined in `config.py` and can be adjusted.)

## Strategy Layer

Julie runs a portfolio of strategy engines. Some are session-specific, others are global.
UI grouping is defined by `LOG_STRATEGY_ALIASES` in `julie001.py`.

### Strategy Quick Reference

| Strategy Family | Style | Sessions | SL/TP Source | UI Group | File |
| --- | --- | --- | --- | --- | --- |
| RegimeAdaptive | Trend + fade | All | Regime params or dynamic | RegimeAdaptive | regime_strategy.py |
| IntradayDip | Mean reversion | NY | Dynamic | IntradayDip | intraday_dip_strategy.py |
| Confluence (ICT) | Sweep + reject | All | Fixed | Confluence | confluence_strategy.py |
| ICT Model | Silver Bullet | NY_AM | Dynamic | ICT Model | ict_model_strategy.py |
| Breakout Strategy Group | Breakouts + acceptance | NY / LDN | Dynamic | Breakout Strategy | orb_strategy.py, impulse_breakout_strategy.py, value_area_breakout_strategy.py |
| Auction Reversion | Value area fade | NY/LDN | Dynamic | RegimeAdaptive | auction_reversion_strategy.py |
| Liquidity Sweep | Sweep + reclaim | All | Dynamic | SMT Divergence | liquidity_sweep_strategy.py |
| Smooth Trend Asia | Pullback trend | ASIA | Dynamic | RegimeAdaptive | smooth_trend_asia_strategy.py |
| ML Physics | ML directional | All | Session/regime | ML Physics | ml_physics_strategy.py |
| Dynamic Engine 1 | Indicator library | All | Fixed brackets | DynamicEngine1 | dynamic_engine_strategy.py |
| SMT Divergence | Inter-market | All | Dynamic | SMT Divergence | smt_strategy.py |
| VIX Mean Reversion | Volatility fade | NY | Fixed | VIX | vixmeanreversion.py |
| Continuation (Fractal Sweep) | Continuation rescue | NY/LDN | Dynamic | Continuation | continuation_strategy.py |
| Dynamic Engine 2 (experimental) | Price action library | All | Fixed brackets | DynamicEngine1 (sub) | dynamic_engine2_strategy.py |

UI grouping highlights:
- RegimeAdaptive group includes Auction Reversion and Smooth Trend Asia as sub-strategies.
- Breakout Strategy group includes ORB, Impulse Breakout, and Value Area Breakout.
- SMT Divergence group includes Liquidity Sweep as a sub-strategy.
- Dynamic Engine 2 is present in the repo but not wired by default.

## Strategy Biographies (Deep Dive)

### RegimeAdaptive (core)
- Uses a 320-context time hierarchy (Quarter x Week-of-month x Day-of-week x Session).
- Trend bias uses SMA20 vs SMA200 plus volatility regime.
- Range spike trigger: candle range exceeds rolling average.
- Signal inversion for known failure windows (fade logic) when historical win rate is poor.
- SL/TP from `regime_sltp_params.py` if present, otherwise dynamic parameters with floors.

Time bucket format:

| Component | Values | Count |
| --- | --- | --- |
| Yearly Quarter | Q1, Q2, Q3, Q4 | 4 |
| Week of Month | W1, W2, W3, W4 | 4 |
| Day of Week | MON-FRI | 5 |
| Session | ASIA, LONDON, NY_AM, NY_PM | 4 |
| Total | 4 x 4 x 5 x 4 | 320 |

### IntradayDip
- Mean reversion anchored to the NY session open (09:30 ET).
- Looks for large deviations with Z-score confirmation and volatility context.

### Confluence (ICT sweep/reject)
- Sweeps prior session highs/lows and confirms rejection.
- Uses hourly body-gap FVG alignment and bank-level proximity.
- Fixed bracket by design for this strategy.

### ICT Model (Silver Bullet)
- NY AM only.
- Requires sweep of a key liquidity level plus inversion FVG confirmation.

### Breakout Strategy Group
- ORB: uses the first 15 minutes of the NY session; retest of midpoint then breakout.
- Impulse Breakout: range expansion plus volume confirmation and ATR buffer.
- Value Area Breakout: requires acceptance beyond VAH/VAL for multiple bars.

### Auction Reversion
- Value-area fade strategy when ER is low (rotational regime).
- Skips high-vol regimes and requires minimum range.

### Liquidity Sweep
- Identifies sweep beyond pivots, then reclaim.
- Requires wick size, optional follow-through, and cooldown.
- Can be limited to low/normal regimes to avoid chasing impulses.

### Smooth Trend Asia
- Dedicated ASIA pullback strategy with strict trend qualification.
- EMA20/EMA50 alignment, ER/persistence checks, ATR ratio caps.
- Pullback touch + reclaim trigger (Trigger A) with max stop constraint.

### ML Physics
- Session-specific ML models with volatility splits.
- Labels can be barrier (TP before SL), ATR expansion, or direction.
- Guardrails apply by session/regime to avoid poor market states.

### Dynamic Engine 1
- Library of indicator sub-strategies.
- Uses fixed SL/TP framework and viability checks.

### SMT Divergence
- Divergence detection between ES and NQ (and related instruments).
- Seeks liquidity divergence and confirmation.

### VIX Mean Reversion
- 557 micro-segments (Quarter, Month, Week, Day, SessionID).
- Bollinger band mean reversion on VIX; generates MES long bias.
- Fixed SL/TP and size defined inside the strategy.

### Continuation (Fractal Sweep)
- Continuation entries used when primary signals are blocked.
- Guarded by allowlist + confirmation checks to avoid over-trading.

## Defense Layer (Filters and Blockers)

The defense layer decides if a candidate can trade. It includes filters (market state)
plus blockers (risk or structure rules).

### Filter Summary

| Filter | Purpose | Blocks |
| --- | --- | --- |
| Rejection Filter | Bias from sweeps/rejections | Counter-bias trades |
| Chop Filter | Consolidation detection | All trades during chop |
| Extension Filter | Session/daily exhaustion | Direction of extension |
| Volatility Filter | Regime classification | Ultra-low vol trades |
| Trend Filter | Multi-timeframe alignment | Counter-trend trades |
| Impulse Filter | Recent impulse detection | Trades against impulse |
| HTF FVG Filter | HTF imbalance zones | Trades into unfilled gaps |
| Bank Level Filter | Institutional levels | Trades against confirmed bias |
| Memory SR Filter | Historical S/R memory | Trades into strong S/R |
| News Filter | Event risk windows | Trades during blackout |

### Blockers

| Blocker | Purpose |
| --- | --- |
| Dynamic Structure Blocker | Local pivot/structure protection |
| Directional Loss Blocker | Blocks a direction after loss streaks |
| Penalty Box | Blocks trades near recent extremes |

### Chop Filter (state machine)
The chop filter uses 320 time-bucket thresholds and a state machine:

| State | Meaning |
| --- | --- |
| NORMAL | Trend or acceptable range |
| IN_CHOP | Consolidation detected |
| BREAKOUT_LONG/SHORT | Breakout detected, structure not confirmed |
| CONFIRMED_LONG/SHORT | HH/HL or LL/LH confirms breakout |
| FAILED_LONG/SHORT | Breakout failed (fade logic possible) |

Additional features:
- Volatility scaling (accordion effect) widens/narrows thresholds.
- Time-in-chop decay blocks fading after too long.
- Structure validation uses recent swings.

### Extension Filter
Checks session and daily range expansion vs time-bucket thresholds:
- Blocks trades in the exhausted direction.
- Supports "extended" and "extreme" thresholds.

### Volatility Regimes

| Regime | Interpretation | Typical Action |
| --- | --- | --- |
| ultra_low | Very quiet | Skip or tighten |
| low | Quiet | Guarded trades |
| normal | Baseline | Standard rules |
| high | Elevated | High-vol guards |

### Dynamic Chop Analyzer
`dynamic_chop.py` maintains dynamic thresholds and target feasibility checks:
- Calibrates 1m/15m/60m thresholds using percentiles of recent range.
- Adds an HTF breakout check to avoid blocking true expansions.
- Enforces room-to-target logic in rotational regimes.
- Supports a Gemini multiplier to adjust sensitivity.

### Bank Level and Quarterly Theory Filter
`bank_level_quarter_filter.py` tracks $12.50 grid levels and session quarters
inspired by Daye's Quarterly Theory. It establishes bias after multi-candle
confirmation and tracks key reference points (prior session highs/lows, midnight ORB).

## Risk, SL/TP, and Viability

### Units and OCO Constraints
- All distances are in points (tick_size = 0.25).
- Orders are sent with fixed OCO brackets and are not widened after entry.

### Fixed SL/TP Framework (OCO compatible)
When enabled, Julie assigns a bracket preset by regime/session:

| Bracket | SL | TP | Typical Use |
| --- | --- | --- | --- |
| ASIA_SMOOTH | 1.75 | 2.00 | Smooth ASIA trends |
| NORMAL_TREND | 2.25 | 2.75 | General trend |
| IMPULSE | 3.00 | 4.50 | High-volatility impulses |

Viability checks (when enabled):
- ATR floor (reject trades if ATR too low)
- Room-to-target (ensure enough runway for TP)
- Session overrides (ASIA uses smaller minimum room)

### Dynamic SL/TP (time-bucketed)
`dynamic_sltp_params.py` provides 320 time-context-specific parameters. These are used by
RegimeAdaptive, ML Physics, and other strategies when fixed brackets are disabled.

### Optimized TP Engine
`risk_engine.py` contains an `OptimizedTPEngine` that uses:
- Return volatility (std of returns)
- GARCH(1,1) volatility estimate
- Shannon entropy of returns
It scales TP around a 2.0-point base and is used by select strategies.

### Circuit Breaker
The circuit breaker halts trading when risk limits are hit:

| Trigger | Default | Action |
| --- | --- | --- |
| Max daily loss | Configurable | Block all trades |
| Max consecutive losses | Configurable | Block all trades |

### Risk Defaults
Key config items:
- `RISK`: point value, fees, min net profit (optional enforcement)
- `SLTP_MIN`: minimum SL/TP floors in points

Break-even support exists in the client and event logger and can be wired into live trade
management as needed.

## Dynamic Parameters System

Julie uses time-bucketed parameter sets for multiple subsystems. The canonical key format is:

```
YearlyQ_MonthlyQ_DayOfWeek_Session
Example: Q1_W2_TUE_NY_AM
```

Key uses:
- `dynamic_sltp_params.py` provides 320 time-context SL/TP parameter pairs.
- `chop_filter.py` uses 320 time-context chop/median/breakout thresholds.
- `extension_filter.py` uses time-context extension thresholds.

Parameter categories (by subsystem):

| Category | Typical Use |
| --- | --- |
| SL/TP multipliers | Time-specific risk sizing |
| Chop thresholds | Consolidation detection states |
| Extension thresholds | Session/daily exhaustion detection |
| Volatility scalars | Regime-aware threshold scaling |

## ML Physics (Models and Guardrails)

### Feature Set (training)
The ML pipeline uses features derived from:
- RSI, ADX, ATR, slope, volatility, range
- Returns (1, 5, 15 bars)
- Z-scores (price, ATR, volatility, range, volume)
- Relative volume (RVOL)
- Time-of-day cyclical encoding
- Trend flags and volatility flags

### Label Modes
- Barrier: TP before SL within horizon
- ATR: move exceeds ATR threshold within horizon
- Direction: simple future return direction

### Model Splits
- 2-way split (low/high volatility) for all sessions
- 3-way split (low/normal/high) for NY sessions
- Volatility regimes based on std of returns with per-session windows
  (NY uses longer windows for stability)

### Guardrails
- Walk-forward guard: requires positive folds to enable sessions
- Vol-regime confidence guard: adds confidence deltas per regime
- High-vol threshold bump and directional gates
- NY normal structure filter (ER + VWAP cross constraint)

### Training Commands

```bash
python ml_train_physics.py --csv ml_mes_et.csv --walk-forward --out-dir .
```

Automated LORO + walk-forward pipeline:

```bash
python ml_physics_auto_pipeline.py --manifest regimes.json --base-csv es_master.csv --out-dir .
```

Outputs:
- model_*.joblib
- ml_physics_thresholds.json
- ml_physics_metrics.json

## ASIA Calibrations and Smooth Trend Asia

### ASIA Viability Gate
ASIA trades require at least one of:
- ATR expansion (ATR5/ATR60 >= threshold)
- Compression then release (range expansion after low ATR percentile)
- Structural interaction (NY close, value area, or ASIA sweep)

### ASIA Calibrations
ASIA-specific overrides can loosen or tighten:
- Penalty box tolerance
- Target feasibility bounds
- Chop filter behavior

There are separate live and backtest blocks:
- `ASIA_CALIBRATIONS`
- `BACKTEST_ASIA_CALIBRATIONS`

### Smooth Trend Asia Strategy
Dedicated slow-trend strategy with:
- EMA alignment and slope
- ER and persistence thresholds
- Pullback touch and reclaim trigger
- Max stop constraint and cooldown

## Continuation and Rescue Flow

Julie supports continuation/rescue logic when primary signals are blocked.
Key controls:
- Backtest allowlist and confirmation blocks
- Live continuation guard and confirmations
- Entry modes logged as standard or rescued
- Rescues can be suppressed for ML Physics to avoid conflicts

## News and Holiday Filtering

### News Filter (ForexFactory)
`news_filter.py` dynamically pulls high-impact USD events and creates blackout windows.

Defaults (configurable):
- High-impact USD events only
- 20 minutes pre-event, 20 minutes post-event
- Daily market close/maintenance blackout at 16:55 ET for 70 minutes

### Holiday and Seasonal Context
The news filter also embeds holiday and seasonal behavior rules, including:

| Scenario | Effect |
| --- | --- |
| Thanksgiving Wed after 12:00 | Block trades |
| Thanksgiving Day | Market closed | 
| Thanksgiving Friday | Full disable |
| July 3 after 13:00 | Early close block |
| July 4 | Market closed |
| Labor Day Tuesday | Post-holiday volatility context |
| MLK / Presidents / Juneteenth Tuesday | Post-holiday context |
| Memorial Day | Holiday context |
| Easter Monday | Post-holiday context |
| Dec 20-23 | Seasonal context |
| Dec 24-31 | Holiday week context |
| Jan 2 | Early-year context |

These contexts are also supplied to Gemini for context-aware tuning.

## Gemini Optimizer (Optional)

Gemini can adjust select parameters and multipliers at runtime:
- Trend filter multipliers
- Dynamic chop multiplier
- Viability tuning (atr_floor, lookback, room-to-target factor)

Behavior:
- Only runs if `CONFIG["GEMINI"]["enabled"]` and API key are set
- Re-prompts on regime changes (ADX/chop, volatility regime, holiday context)
- Rate limited by `CONFIG["GEMINI"]["min_interval_minutes"]`
- Fixed bracket sizes remain unchanged when fixed SL/TP is enabled

## Session Manager and Quarterly Theory

### Bank Level Quarter Filter
`bank_level_quarter_filter.py` tracks $12.50 grid levels and session-specific time
quarters inspired by Daye's Quarterly Theory.

Quarter sizes by session:

| Session | Quarter Length |
| --- | --- |
| ASIA | 135 minutes |
| LONDON | 75 minutes |
| NY_AM | 60 minutes |
| NY_PM | 75 minutes |

Levels tracked:
- Previous PM high/low
- Previous session high/low
- Midnight ORB high/low

### Volume Profile
Value area levels (VAH/VAL/POC) are derived from recent volume profile windows and used
by Auction Reversion and Value Area Breakout.

## Backtesting

- Backtest runner: `backtest_mes_et.py`
- UI runner: `backtest_mes_et_ui.py`
- Auto contract selection per day:
  - `BACKTEST_SYMBOL_MODE = "auto_by_day"`
  - `BACKTEST_SYMBOL_AUTO_METHOD = "volume"` or `"rows"`
- Backtest-only overrides: `BACKTEST_*` keys in `config.py`
- Reports saved to `backtest_reports/*.json`

## File Layout

```
config.py
julie001.py
backtest_mes_et.py
backtest_mes_et_ui.py
fixed_sltp_framework.py
risk_engine.py
regime_strategy.py
intraday_dip_strategy.py
confluence_strategy.py
ict_model_strategy.py
orb_strategy.py
impulse_breakout_strategy.py
value_area_breakout_strategy.py
auction_reversion_strategy.py
liquidity_sweep_strategy.py
smooth_trend_asia_strategy.py
ml_physics_strategy.py
ml_train_physics.py
ml_physics_auto_pipeline.py
continuation_strategy.py
smt_strategy.py
vixmeanreversion.py
dynamic_engine_strategy.py
dynamic_engine2_strategy.py
dynamic_chop.py
volatility_filter.py
trend_filter.py
chop_filter.py
extension_filter.py
news_filter.py
```

## Troubleshooting

- No trades: check chop/extension/volatility filters and ASIA viability gate.
- ML models not loading: verify model_*.joblib files and thresholds JSON.
- Backtest differs from live: check BACKTEST_* overrides in config.
- Gemini slow startup: disable Gemini or increase min interval.
- News blackout unexpected: confirm holiday/news windows in `news_filter.py`.

## Safety and Notes

- This repository is for research and automation. Futures trading involves risk.
- Use paper trading and backtests before live deployment.
- Never commit credentials; keep real keys out of version control.
- All distances are in points (not ticks). Ensure external data or parameter files
  are normalized consistently.
