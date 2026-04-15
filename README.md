# Julie Filterless: Technical Architecture and Strategy Guide

This document explains the current Julie bot from a technical perspective:

- what the live bot is
- how the runtime is structured
- how market data becomes orders
- how the active strategy engines work
- how the sentiment monitor and emergency exits fit into live trading
- how the Kalshi gate interacts with execution and dashboard visibility
- how to bootstrap the local workspace and FinBERT dependencies
- what the important artifacts and files are

It is intentionally focused on the current filterless live stack, because that
is the path the bot is actually meant to run on.

## 1. System Overview

Julie is an automated MES futures trading system for TopstepX / ProjectX
Gateway. The codebase contains legacy strategies, research tooling, and many
historical experiments, but the current live architecture is centered on a
filterless execution stack.

In the current live mode, Julie trades a compact roster of four engines:

- DynamicEngine3
- RegimeAdaptive
- MLPhysics
- AetherFlow

Truth Social sentiment data is still collected via RSS and analyzed by FinBERT,
but only to power emergency exits on open positions — it is not a standalone
entry strategy.

The design goal is not "no controls." Filterless means the bot avoids the older
external strategy-filter stack and instead relies more directly on each active
engine's internal decision logic, gating, sizing, and bracket behavior.

At runtime, Julie combines:

- REST market/history access through ProjectX
- a ProjectX user stream for live account / position / trade updates
- a per-bar strategy evaluation loop
- an asynchronous sentiment polling and FinBERT inference service (for emergency exits)
- Kalshi crowd-probability gating during active settlement windows
- structured logging and persisted state
- a local dashboard bridge that turns bot state and logs into frontend JSON

## 2. Live Process Model

The preferred live entry path is the filterless workspace launcher:

- `LaunchFilterlessWorkspace.bat`
- `launch_filterless_workspace.py`

That launcher starts three cooperating processes:

1. the live trading bot
2. the filterless dashboard bridge
3. the local frontend server

### 2.1 Bot launcher

`launch_filterless_live.py` is a thin wrapper around `julie001.py`.

Its main job is to set the runtime environment into filterless mode before the
bot imports:

- `JULIE_FILTERLESS_ONLY=1`
- `JULIE_DISABLE_STRATEGY_FILTERS=1`

It also:

- forces UTF-8 stdio on Windows
- patches slow Windows platform detection issues
- acquires a singleton lock under `logs/filterless_live.lock`

### 2.2 Workspace launcher

`launch_filterless_workspace.py` exists to make the operator experience stable.

It handles:

- Python interpreter discovery
- workspace dependency bootstrap
- FinBERT bootstrap
- stale-process cleanup
- singleton locking at the workspace level
- starting the bot
- starting the dashboard bridge
- starting either a Vite dev server or a static server for the frontend
- optional browser launch
- PID tracking in `logs/filterless_workspace_pids.json`

This is important because the live stack is meant to be run as a small local
workspace, not as a single Python script in isolation.

## 3. Runtime Architecture

At a high level, the live stack looks like this:

```text
         ProjectX REST + ProjectX User Stream
                       |
                       v
                    client.py
                       |
                       v
                    julie001.py
                       |
    --------------------------------------------------------
    |                  |               |                   |
    v                  v               v                   v
  DE3 v4        RegimeAdaptive     MLPhysics          AetherFlow
    |                  |               |                   |
    -------------------- signal candidates ---------------------
                       |
                       v
     emergency exits + Kalshi crowd veto / sizing overlay
                       |
                       v
            execution / sizing / bracket logic
                       |
                       v
                ProjectX order placement
                       |
                       v
              logs + persisted runtime state
                       |
                       v
      tools/filterless_dashboard_bridge.py
                       |
                       v
      filterless_live_state.json for the frontend

 Sentiment polling path (runs asynchronously beside the bar loop):

   RSS feed -> services/sentiment_service.py -> bot_state.json -> dashboard/UI
```

The most important runtime files are:

- `julie001.py`: live orchestrator
- `client.py`: broker / data client
- `config.py`: runtime settings and artifact paths
- `event_logger.py`: structured event logging
- `bot_state.py`: persisted runtime state
- `process_singleton.py`: process locks
- `services/sentiment_service.py`: sentiment polling via RSS + FinBERT inference (emergency exits)
- `tools/filterless_dashboard_bridge.py`: dashboard JSON builder

## 4. What Filterless Changes

Inside `julie001.py`, filterless mode changes runtime behavior in several
important ways.

### 4.1 Strategy roster changes

The live roster is narrowed to:

- DynamicEngine3
- RegimeAdaptive
- MLPhysics
- AetherFlow

Older modules like VIX, MNQ-dependent logic, and the broader legacy strategy
mix are not part of the filterless live execution path. Truth Social sentiment
is still monitored but only feeds emergency exits, not entry signals.

### 4.2 Filter stack changes

The legacy external filter chain is disabled. This reduces coupling to the old
live architecture and avoids forcing the active engines through logic that was
designed for a much broader mixed portfolio.

### 4.3 DE3 version forcing

Filterless live forces DynamicEngine3 into DE3 v4 mode. That matters because
DE3 v4 uses its own learned runtime bundle instead of the simpler older DB-only
selection logic.

### 4.4 Gemini changes

Gemini optimization is disabled by default in filterless live. This keeps the
operator-facing runtime more deterministic and avoids unnecessary complexity in
the main execution path.

## 5. Data Flow and Execution Loop

The live bot in `julie001.py` is fundamentally an async trading loop with
background tasks.

### 5.1 Startup

At startup, the bot:

1. authenticates with ProjectX
2. selects the account
3. resolves the active contract symbol and contract ID
4. starts the ProjectX user stream
5. loads strategy engines, the sentiment runtime, and model artifacts
6. pulls historical bars for warmup
7. restores persisted state
8. enters the live loop

### 5.2 Historical warmup

Julie does not start from an empty chart. It pulls a deep block of recent bars
to initialize:

- resamplers
- volatility state
- trend / session context
- strategy-specific rolling features
- internal state used by the active engines

This prevents the first live bars from being evaluated with incomplete context.

### 5.3 Per-bar processing

Once running, the bot repeatedly:

1. fetches or updates market data
2. identifies whether a new bar has formed
3. updates session and risk context
4. checks circuit-breaker style exits, including sentiment emergency exits
5. asks each active strategy whether it has a candidate
6. evaluates candidate priority
7. applies live crowd gating / sizing overlays
8. sends valid candidates to execution

In filterless mode, the external strategy filter chain is removed, but the bot
still performs:

- session awareness
- signal sizing
- bracket calculation
- position / order state checks
- drawdown-aware logic
- trade state management

### 5.4 Execution

Execution happens through `client.py`, which places ProjectX orders and tracks:

- current account
- current contract
- local shadow position state
- active stop order IDs
- cached open orders

The bot places bracketed orders and then manages live position state using both:

- ProjectX REST calls
- ProjectX user-stream updates

### 5.5 State persistence

Runtime state is serialized to `bot_state.json`. This captures enough state for:

- dashboard continuity
- restart visibility
- session-aware context recovery
- operator inspection

The dashboard bridge then combines state + log parsing into a frontend-friendly
JSON snapshot.

## 6. Sessions and Time Context

Julie is heavily session-aware.

The standard session structure is:

- ASIA: 18:00 to 02:59 ET
- LONDON: 03:00 to 07:59 ET
- NY_AM: 08:00 to 11:59 ET
- NY_PM: 12:00 to 16:59 ET

The trading day is anchored at 18:00 ET, not midnight. That matters for:

- bot state rollover
- daily risk accounting
- session labeling
- context keys used by RegimeAdaptive and DE3

`bot_state.py` exposes `trading_day_start()` specifically for this reason.

## 6A. Sentiment Monitor and Emergency Exits

The sentiment service monitors Trump's Truth Social posts via an RSS mirror
(`trumpstruth.org`) and classifies them with FinBERT. It is **not** a
standalone entry strategy — it only powers emergency exits on open positions.

### 6A.1 Runtime model

The sentiment monitor runs outside the main per-bar loop as an independent
async task. That prevents RSS polling or model inference from stalling live
market processing.

- Posts are fetched from the `trumpstruth.org` RSS feed (no auth, no Cloudflare)
- FinBERT only loads when the service actually needs to classify a post
- The service writes a normalized `sentiment` block into `bot_state.json`
- After each poll cycle, the state is persisted immediately to keep the dashboard current

### 6A.2 FinBERT loading and quantization

Julie prefers a local FinBERT snapshot under `./models/finbert`.

The loader uses a platform-aware fallback chain:

1. `bitsandbytes` 8-bit loading where the runtime supports it
2. dynamic int8 quantization through PyTorch when `bitsandbytes` is not viable
3. standard precision only as the final fallback

This is important on local 16 GB machines because it reduces memory pressure
without making the live bot depend on one GPU-specific path.

### 6A.3 Sentiment classification

Each post is classified into one of three categories:

- **LONG** — sentiment score exceeds the pump threshold (strongly positive)
- **SHORT** — sentiment score drops below the negative threshold (strongly negative)
- **NEUTRAL** — sentiment score falls between thresholds (no market-moving bias)

The classification produces:

- `sentiment_score` in the `[-1.0, 1.0]` range
- `sentiment_label` (`positive`, `negative`, or `neutral`)
- `finbert_confidence`
- `trigger_side` (`LONG`, `SHORT`, or `NEUTRAL`)
- `trigger_reason`

Neutral posts are tracked and displayed but do not trigger any trading action.

### 6A.4 Emergency exit behavior

Before normal strategy evaluation proceeds, `julie001.py` checks the fresh
sentiment snapshot against the current shadow position:

- if Julie is `LONG` and sentiment flips below `EMERGENCY_EXIT_THRESHOLD`, the bot flattens immediately
- if Julie is `SHORT` and sentiment flips hard positive, the bot also flattens immediately

That emergency flatten path bypasses the normal SL/TP wait logic. It cancels
resting orders and sends a direct close order through `client.py`. Neutral
sentiment does not trigger an exit.

## 6B. Kalshi Crowd Gating and Dashboard Semantics

Kalshi is not just a passive dashboard add-on. In the current filterless stack
it does three separate jobs:

- crowd-based trade veto / sizing assistance
- open-position probability mapping against the active TP contract
- UI context for the current hourly settlement ladder

### 6B.1 Hour alignment

Kalshi hourly contracts roll on a `:05` boundary after each settlement hour.
The bot, bridge, and frontend now share the same active-hour calculation so the
dashboard does not drift one hour behind the actual gated contract.

### 6B.2 Hard-veto behavior

The gate is not purely cosmetic. During active settlement windows, extreme
contradictory probabilities hard-block trades even when the broader Kalshi
veto mode is otherwise soft.

That specifically protects against trades that try to fire with crowd odds such
as a `1%` probability against the intended path.

### 6B.3 UI probability semantics

The filterless dashboard converts the display side to match the open position:

- longs display the relevant `YES` path
- shorts display the relevant `NO` path

The strike ladder still anchors itself in the SPX contract space that Kalshi
quotes, but the operator-facing reference price is converted back into ES so it
matches the futures position the bot is actually managing.

## 6C. Local Setup and Bootstrap

The supported local bootstrap path is now:

1. run `LaunchFilterlessWorkspace.bat` or `Rose.bat`
2. let `setup_topstep2.ps1` create or refresh `.venv`
3. install `requirements.txt`
4. download or verify the local FinBERT snapshot
5. run import + FinBERT smoke checks

### 6C.1 What the bootstrap installs

The workspace setup path verifies:

- `transformers`
- `torch`
- `accelerate`
- the local `./models/finbert` snapshot

`bitsandbytes` remains optional and platform-specific. On platforms where it is
not available, Julie falls back to dynamic int8 quantization or standard
precision so setup does not fail just because one acceleration path is missing.

The sentiment service fetches posts from the `trumpstruth.org` RSS mirror, so
no Truth Social credentials or the `truthbrush` package are required.

## 7. DynamicEngine3

DynamicEngine3 is the most complex active engine in the live bot.

### 7.1 What DE3 actually is

DE3 is best understood as a strategy-of-strategies. It does not begin from a
single trading rule such as "buy pullbacks" or "fade extensions." Instead, it
begins from a catalog of already-defined strategy members and then asks a second
runtime layer which member, if any, deserves to trade now.

That makes DE3 a two-artifact system:

- a structural artifact: `dynamic_engine3_strategies_v2.json`
- a behavioral artifact: `artifacts/de3_v4_live/latest.json`

The structural artifact says what can exist. The behavioral artifact says how
the engine should choose among those possibilities.

This distinction matters more in DE3 than anywhere else in the bot. If the
structural artifact is skewed, the runtime can only optimize inside that skew.

### 7.2 The member database

The v2 member database is the raw inventory of tradable DE3 variants. Each row
represents a concrete member with attributes such as:

- timeframe, usually 5 minute or 15 minute
- session bucket
- directional side
- strategy family or setup label
- stop and target profile
- historical performance and ranking metadata

At a high level, each row is a sentence that looks like:

"In this session, on this timeframe, for this side, using this family, the
expected bracket and historical behavior look like this."

That means the member DB is not just a lookup table. It is the space of actions
DE3 is allowed to take.

The most important practical consequence is side coverage. If the DB has no
short members in a session, DE3 cannot truly become bearish there. The router,
lane selector, side prior, and entry models can rearrange ranking, but they
cannot invent absent inventory.

### 7.3 Candidate formation and feasibility

`dynamic_engine3_strategy.py` is the live wrapper around the DE3 family. It
maintains the resampled views that DE3 needs, mainly 5 minute and 15 minute
bars, and it turns current market state into a set of feasible candidates.

The rough order is:

1. load the member DB and runtime bundle
2. derive session, volatility, and structural context
3. build candidate rows from the member DB
4. compute execution payloads for those candidates
5. discard rows that are not feasible in the present market state

By the time DE3 v4 sees its candidate set, it is not evaluating every member in
the JSON blindly. It is evaluating the feasible subset that survived the base
runtime checks.

### 7.4 The v4 runtime as a decision stack

`de3_v4_runtime.py` is where DE3 becomes a genuine decision system rather than a
static ranker. The v4 runtime attaches a series of decision layers to the
feasible candidate pool:

- route decision
- lane selection
- entry-model evaluation
- execution-policy scoring
- bracket selection
- optional decision-side model application
- optional signal-size rules
- final drift gate

The wrapper exports a large amount of this state back into the final signal
payload. That is why DE3 signals carry fields such as:

- `de3_v4_route_decision`
- `de3_v4_route_confidence`
- `de3_v4_selected_lane`
- `de3_v4_selected_variant_id`
- `de3_v4_execution_policy_tier`
- `de3_v4_execution_quality_score`
- `de3_v4_entry_model_score`
- `de3_v4_bracket_mode`
- `decision_side_model_predicted_action`

This is deliberate. DE3 is meant to be inspectable after the fact.

### 7.5 Route decision and lane selection

The route stage answers the broadest DE3 question: what kind of choice should be
made in this bar context? In code terms, the router can either drive the choice
directly or hand off to the router-plus-lane stack, depending on bundle mode.

Conceptually, the router decides how the candidate universe should be read right
now. The lane stage then narrows that universe to a smaller behavior cluster.

Lanes are DE3's higher-level operating modes. They are not the same thing as a
single strategy member. A lane can be thought of as a runtime corridor that
groups compatible variants before the final winner is chosen.

This gives DE3 a hierarchy:

- first choose the broad operating corridor
- then rank candidates inside that corridor

That is why DE3 behaves more like a portfolio allocator than a normal signal
generator.

### 7.6 Entry model, execution policy, and decision-side model

Once a candidate is near the top, DE3 can still apply several learned or
artifact-configured overlays.

The entry model estimates whether the candidate should be allowed at all under
the current runtime state. The signal records whether the model was enabled,
what scope it applied under, what score it assigned, and what threshold it had
to clear.

The execution policy is a separate quality layer. It classifies the choice into
tiers, produces a runtime quality score, and can either:

- allow the candidate normally
- soft-pass it while downgrading trust
- hard-veto it if a hard limit is triggered

The decision-side model is yet another overlay. It is DE3's attempt to learn
when the current context looks more like a long, short, or no-trade
environment. Importantly, this does not automatically mean "side is solved."
Its real power depends on how the bundle applies it:

- as a soft prior, it nudges ranking
- as a hard override, it can force side choice more aggressively

Even then, it still lives inside the member inventory. If there are no credible
short candidates in the lane, the side model cannot manufacture one.

### 7.7 Profit gates, prune rules, and signal-size rules

The live DE3 wrapper also supports runtime risk shaping above and beyond simple
selection.

The pre-router profit gate can evaluate whether a lane, session, or
lane-session combination is statistically unhealthy. Its configuration supports
soft passes, catastrophic blocks, and separate sample-size / loss-probability /
EV requirements.

The pre-entry prune stage can then veto specific chosen entries after the v4
decision is already formed. This is useful for targeted cleanup when a variant
is structurally present but repeatedly undesirable in a narrow context.

Finally, signal-size rules can modify the final contract count after the winner
has already been selected. This is one of the reasons DE3 is not simply a
"which row wins?" engine. Position size itself can be part of the decision.

### 7.8 Bracket selection and execution payload

Once DE3 has a winner, the bracket module chooses the final stop and target
expression. `de3_v4_bracket_module.py` can either keep the canonical member
bracket or override it with a locally adapted bracket.

The chosen execution payload typically contains:

- final stop distance
- final target distance
- contract count
- policy risk multiplier
- gross profit estimate
- fee estimate
- net profit estimate

This is important because DE3's decision is not complete until the bracket is
resolved. In DE3, a member identity and a bracket identity are related but not
identical.

### 7.9 Final drift gate

Even after selection, sizing, and bracketing, DE3 can still refuse the trade.
The wrapper applies a drift gate that measures whether current price has moved
too far from the anchor condition the candidate expects.

This is one of the last defenses against stale signals. A candidate can be
structurally valid and top-ranked, but still fail because the market has drifted
too far away from the trade's intended entry geometry.

### 7.10 What DE3 is learning, and what it is not

DE3 learns less like a pure classifier and more like a routing policy over a
prebuilt library. Its intelligence lives in:

- how it scores context
- how it routes into lanes
- how it applies side priors
- how it adjusts bracket and size
- how it decides to abstain

What it is not doing is inventing new strategy families on the fly. If the
underlying inventory is poorly balanced, DE3's sophistication mainly improves
selection quality inside that biased inventory.

That makes DE3 the engine most sensitive to research hygiene. Better bundles
help, but better member inventory often helps more.

## 8. RegimeAdaptive

RegimeAdaptive is the most explicitly time-contextual engine in the live stack.

### 8.1 Core idea

RegimeAdaptive is a context-conditioned rule engine. It does not begin by
asking, "What does price do in general here?" It begins by asking, "What has
this very specific calendar-and-session context historically preferred?"

Its central abstraction is the context key:

- quarter of year
- week in month
- day of week
- trading session

Examples:

- `Q2_W1_TUE_NY_AM`
- `Q4_W3_FRI_ASIA`

This turns market timing into a lattice of contexts rather than a single global
environment.

### 8.2 The artifact is the strategy grammar

The live artifact at `artifacts/regimeadaptive_v19_live/latest.json` is not just
a parameter file. It is the grammar RegimeAdaptive uses to decide what type of
signal is valid for a given context.

The artifact can hold:

- rule catalog entries
- per-context combo policies
- per-group policies
- session defaults
- global defaults
- signal-gate configuration
- optional SL/TP and early-exit preferences

`regimeadaptive_artifact.py` parses this payload and normalizes its policies
into three verbs:

- `normal`
- `reversed`
- `skip`

Those three verbs are enough to radically change behavior by context.

### 8.3 Base indicator model

Under the artifact layer, the engine still has a concrete price-action model.
`regime_strategy.py` uses:

- fast and slow SMAs
- ATR
- range expansion
- rolling volatility
- session and hour context

From those, it determines whether the market is trending up or down, whether
range is large enough to matter, and whether the context is too volatile or time
blocked to trade.

This is why RegimeAdaptive should not be described as purely calendar-based. It
is calendar-conditioned, but still price- and volatility-aware at the bar level.

### 8.4 Rule families in detail

The artifact can choose among three rule types, each of which defines a
different interpretation of trend structure.

`pullback`

- Long version: uptrend, current close dips below the fast SMA by an ATR-scaled
  threshold, and the bar range is large enough to count as a meaningful pullback.
- Short version: downtrend, current close rallies above the fast SMA by an
  ATR-scaled threshold, with the same range-spike requirement.
- This is the default RegimeAdaptive pattern and behaves like a
  trend-with-retracement rule.

`continuation`

- Long version: market is already in an uptrend, price has recently touched or
  approached the fast SMA, and the current bar closes back above the SMA by a
  threshold.
- Short version: mirror image in a downtrend.
- This family is looking for resumed directional flow after shallow support or
  resistance interaction.

`breakout`

- Long version: uptrend plus a close above recent high by an ATR-scaled margin.
- Short version: downtrend plus a close below recent low by an ATR-scaled margin.
- This family is the least mean-reverting and the most directional of the three.

In other words, the artifact is not just selecting thresholds. It is selecting
the very geometry of the signal.

### 8.5 Equal-high and equal-low protection

RegimeAdaptive includes an equal-high / equal-low filter that matters most for
pullback logic. If the current bar is effectively printing into a repeated low
or repeated high, the engine can reject the signal.

The purpose is subtle but important: a "pullback" that is really just a repeated
failure zone can behave more like a continuation trap than a clean retracement.
The equal-level filter is a structural guard against that ambiguity.

### 8.6 Reversion and skip policy

The most distinctive RegimeAdaptive behavior is signal reversion.

Suppose a context key historically performs poorly when traded normally on the
long side. RegimeAdaptive does not have to choose between:

- keep trading it badly
- or disable it entirely

It can instead reverse the interpretation and trade the opposite side.

That is why the engine can emit both `original_signal` and `reverted` metadata.
The rule candidate may be detected as a long-style pullback, but the artifact
can explicitly say that this context should be faded instead.

This is one of the few engines in the stack where "bad historical context" can
be converted into a systematic contrarian rule rather than a dead zone.

### 8.7 Volatility and time guards

RegimeAdaptive is also opinionated about when not to trust its own rules.

The live runtime can block:

- high-volatility states in configured sessions
- blocked hours inside particular sessions
- low-quality trend states if low-vol trend confirmation is required
- weak range structure if the bar does not qualify as a range spike

This means the rule catalog does not operate in a vacuum. Every candidate still
passes through a live-quality screen before it becomes a trade.

### 8.8 The signal gate

The optional signal gate in `regimeadaptive_gate.py` is a second-stage model that
evaluates whether the chosen rule candidate looks tradable enough to keep.

Its runtime feature row includes:

- final side and original side
- whether the signal was reverted
- quarter, week, day, and session codes
- hour and minute encodings
- rule type code
- rule parameters such as SMA lengths and ATR multipliers
- strength and ATR-scaled geometric features
- return and volatility summaries

So the gate is not simply checking "is this a long?" It is checking the full
shape of the proposed signal inside its context.

### 8.9 Exit behavior

After signal formation, RegimeAdaptive resolves exits from either:

- artifact-provided SL/TP values
- optimized legacy parameters
- dynamic SL/TP fallback logic

It can also attach early-exit metadata. That lets the artifact influence not
just entry direction, but how impatient the engine should be after entry.

### 8.10 What makes RegimeAdaptive different

RegimeAdaptive is the least portfolio-like of the active strategies. Unlike DE3,
it is not choosing among a library of members. Unlike MLPhysics, it is not
primarily choosing among model outputs. It is taking a rule grammar and bending
that grammar around time context.

Its personality is:

- explicit
- interpretable
- calendar-aware
- reversible

It is the engine most suited to answering the question, "How should this setup
behave in this exact slice of the trading calendar?"

## 9. MLPhysics

MLPhysics is the heaviest predictive model in the current filterless roster.

### 9.1 Core idea

MLPhysics is a distribution-aware inference engine, not a simple directional
classifier.

Its job is not merely to say:

- go long
- go short
- do nothing

Its real job is closer to:

- estimate expected value
- estimate plausible favorable excursion
- estimate plausible adverse excursion
- choose a bracket shape
- estimate whether the resulting trade is worth taking

That is why the live replacement runtime is called `dist_bracket_ml`.

### 9.2 Bundle anatomy

The active inference run is loaded from a directory containing:

- `config.json`
- `artifact_index.json`
- `models/`

`config.json` describes runtime behavior. `artifact_index.json` maps sessions,
sides, model types, calibrators, and gate payloads to files in `models/`.

This layout is important because MLPhysics is really a bundle of cooperating
models, not a single file.

### 9.3 Session-side decomposition

MLPhysics is specialized simultaneously by session and by side. In effect, the
runtime can hold separate learned beliefs for:

- ASIA LONG
- ASIA SHORT
- LONDON LONG
- LONDON SHORT
- NY_AM LONG
- NY_AM SHORT
- NY_PM LONG
- NY_PM SHORT

This matters because MLPhysics assumes that both direction and bracket geometry
can be session-specific. A London short is not treated as the same statistical
object as a New York afternoon short.

### 9.4 What models exist inside the bundle

For a given session-side pair, the runtime can load multiple model families:

- EV models for expected value
- MFE quantile models for favorable excursion
- MAE quantile models for adverse excursion
- EV quantile models for uncertainty estimation
- hit models for bracket search
- gate classifiers and calibrators

This gives MLPhysics a more distributional view of a trade than the other live
engines. It is trying to estimate not just direction, but the shape of the
potential outcome distribution.

### 9.5 Per-side scoring flow

`dist_bracket_ml/dist_bracket_ml/inference.py` scores long and short
independently via `_score_side()`.

For each side, the runtime:

1. builds a feature row in the expected model-column order
2. predicts expected value
3. predicts MFE and MAE quantiles
4. converts those into candidate ATR-based stop and target distances
5. enforces minimum reward-to-risk constraints
6. estimates uncertainty
7. produces a side score

This is crucial. MLPhysics does not frame long versus short as a single
softmax-style contest. It asks whether each side is independently viable first,
then compares the survivors.

### 9.6 Bracket optimization

The most distinctive part of MLPhysics is its bracket search.

If hit-model grid search is enabled, the engine enumerates candidate ATR-space
brackets using configured `sl_atr` and `tp_atr` grids. For each bracket, it
predicts hit probability and computes a bracket EV:

- expected reward if target is hit
- expected loss if stop is hit
- trading cost adjustment

It then picks the best bracket from that candidate set.

This means MLPhysics is not merely inheriting a fixed bracket from training. It
can choose among multiple bracket shapes at runtime using learned hit
probabilities.

### 9.7 Scoring and uncertainty

After bracket resolution, MLPhysics computes a side score. The exact score mode
depends on bundle configuration, but typical forms are:

- EV divided by stop distance
- EV divided by uncertainty

It also tracks secondary quantities such as:

- `ev_pred`
- `tp_atr`
- `sl_atr`
- `rr`
- `p_hit`
- `ev_bracket`
- `mfe_spread`
- `mae_spread`

The returned signal is therefore both a decision and a diagnostic object.

### 9.8 Gate layer

After a side is chosen, `_gate_decision()` can still reject it.

The gate is a separate model family keyed by session and side. It builds its own
gate feature row from:

- the raw runtime feature row
- the chosen side
- base model outputs such as EV and bracket metrics

The gate then predicts `p_take`, optionally calibrates it, and compares it to a
threshold. If `p_take` is below threshold, the side is turned into `NONE`.

This is important philosophically. In MLPhysics, "tradeability" is a separate
problem from "directional edge."

### 9.9 The live wrapper around the bundle

`ml_physics_strategy.py` is the bridge between the bot and the replacement
runtime. It is responsible for:

- locating the correct run directory
- loading the bundle lazily
- enforcing history requirements
- aligning bars to the configured timeframe
- converting live data into the runtime's expected OHLCV frame
- applying runtime gate-threshold clamps and local policy logic

The wrapper also preserves a detailed `last_eval` payload so the operator can
see whether a bar failed because of missing history, timeframe alignment, gate
rejection, RR failure, or a harder model-side block.

### 9.10 Why MLPhysics feels different from the other engines

DE3 is a routing engine over member inventory. RegimeAdaptive is a context-aware
rule interpreter. AetherFlow is a compact two-stage setup-and-model engine.

MLPhysics is different from all of them because it thinks in distributions. It
tries to model how far price is likely to move in both favorable and adverse
directions, and it treats bracket design as part of inference instead of as a
fixed afterthought.

That is why it has the largest artifact footprint and the highest packaging
discipline requirements in the live stack.

## 10. AetherFlow

AetherFlow is the smallest and cleanest of the active filterless engines.

### 10.1 Core idea

AetherFlow is a compact regime-and-setup engine built on top of manifold
features. It is "small" only in artifact surface area. Conceptually it is one
of the cleaner engines in the stack because its pipeline is easy to describe:

1. derive manifold state from recent bars
2. score a small dictionary of setup families
3. choose the strongest setup if it clears family thresholds
4. pass that candidate through a learned probability model
5. attach setup-specific brackets and holding horizon

So AetherFlow is not a pure classifier. It is a two-stage engine:

- deterministic setup labeling
- probabilistic validation

### 10.2 Artifact set and live runtime

The live deployment is intentionally compact:

- a pickled model
- a thresholds JSON file
- a metrics JSON file

`aetherflow_strategy.py` loads those artifacts and then applies several live
restrictions from `config.py`, including:

- `min_bars = 320`
- `threshold_override = 0.55`
- `min_confidence = 0.55`
- `size = 5`
- `allowed_session_ids = [1, 2, 3]`
- `allowed_setup_families = ["compression_release", "transition_burst"]`
- `hazard_block_regimes = ["ROTATIONAL_TURBULENCE"]`

That means current live AetherFlow deliberately excludes ASIA and deliberately
trades only a subset of its full internal setup dictionary.

### 10.3 The manifold layer

AetherFlow's raw material comes from the manifold feature frame built in
`manifold_strategy_features.py` and `regime_manifold_engine.py`.

That manifold layer produces state variables such as:

- alignment
- smoothness
- stress
- dispersion
- risk multiplier
- side bias
- regime ID
- regime-specific allow flags

It also classifies each bar into one of four named regimes.

#### 10.3.1 TREND_GEODESIC

This is the clean-trend regime.

The manifold engine assigns `TREND_GEODESIC` when alignment is high, smoothness
is high, and dispersion is relatively low. In code terms the raw regime is
selected when alignment is at least about `0.62`, smoothness at least about
`0.60`, and dispersion no more than about `0.45`.

Interpretation:

- price motion is coherent
- direction is persistent rather than noisy
- path geometry is orderly

Allowed styles in this regime:

- trend
- breakout

Disallowed styles:

- mean reversion
- fade

This is the manifold's way of saying, "Directional continuation makes sense
here. Fighting the move does not."

#### 10.3.2 CHOP_SPIRAL

This is the rotational but still tradable regime.

The engine falls into `CHOP_SPIRAL` when stress is elevated or smoothness is too
low for trend classification, but the market has not become fully dispersed or
fully hazardous.

Interpretation:

- direction is unstable
- rotation dominates clean travel
- local reversions are more reliable than impulse continuation

Allowed styles:

- mean reversion
- fade

Disallowed styles:

- trend
- breakout

This regime is not "bad market" so much as "different market." It is tradable,
but not with the same logic as a geodesic trend.

#### 10.3.3 DISPERSED

This is the structurally weak regime.

The manifold assigns `DISPERSED` when dispersion is high and alignment is low,
roughly the state where price is moving but not along a coherent path.

Interpretation:

- energy is scattered
- direction does not organize
- signal persistence is weak

All style flags are turned off here and `no_trade` becomes true.

This regime is not just "choppy." It is more like fragmented motion without a
trustworthy local geometry.

#### 10.3.4 ROTATIONAL_TURBULENCE

This is the hard hazard regime.

The engine assigns `ROTATIONAL_TURBULENCE` when mean absolute rotational change
is high and stress is already elevated. In other words, the manifold is seeing
violent turning behavior rather than directional travel.

Interpretation:

- price is spinning, not flowing
- side persistence is poor
- local geometry is unstable enough to be dangerous

Like `DISPERSED`, all style flags are turned off and `no_trade` becomes true.
Additionally, current live AetherFlow blocks this regime explicitly through
`hazard_block_regimes`.

### 10.4 AetherFlow's derived feature layer

`aetherflow_features.py` does not stop with the raw manifold fields. It builds a
second layer of directional and structural features such as:

- `flow_fast` and `flow_slow`
- flow magnitudes and flow agreement
- flow curvature
- short- and medium-window pressure imbalance
- coherence
- compression, expansion, and extension scores
- transition energy
- novelty score
- regime-change flags
- short-window deltas of alignment, stress, dispersion, and coherence

This is the language AetherFlow uses to decide which setup family is present.

### 10.5 Setup families

AetherFlow has four setup families. These are not four models. They are four
market narratives encoded as deterministic score functions.

#### 10.5.1 Compression Release

`compression_release` is the setup that tries to catch directional expansion out
of a compressed state.

It requires a market with:

- meaningful compression
- at least some transition energy
- manageable stress
- nontrivial fast flow
- no rotational hazard

Its base score is a blend of compression, coherence, fast-flow magnitude, and
improving alignment. Long versus short is decided by directional bias, so the
same structural setup can point either way depending on the flow field.

This setup is best thought of as:

"The market has been compact, energy is starting to release, and a directional
break may now be worth following."

Default bracket profile:

- `sl_mult = 1.25`
- `tp_mult = 2.35`
- `horizon_bars = 20`

#### 10.5.2 Aligned Flow

`aligned_flow` is the cleanest trend-following setup in AetherFlow.

It requires:

- high coherence
- high alignment percentile
- high smoothness percentile
- low stress
- positive flow agreement
- strong slow-flow magnitude
- no rotational hazard

This is essentially the "the river is already flowing cleanly" setup. It wants
the manifold and the directional statistics to agree that the move is orderly.

Compared with `compression_release`, it is less about energy release and more
about joining an already well-formed flow.

Default bracket profile:

- `sl_mult = 1.10`
- `tp_mult = 2.00`
- `horizon_bars = 18`

#### 10.5.3 Exhaustion Reversal

`exhaustion_reversal` is the mean-reverting stress setup.

It looks for:

- large VWAP extension, beyond about `1.25` ATR
- falling coherence
- rising stress
- strong extension score
- weak flow agreement
- fast-flow curvature consistent with reversal
- regime state in `DISPERSED` or `CHOP_SPIRAL`

Long and short are asymmetric mirror cases:

- long exhaustion reversal wants negative extension first, then reversal pressure
- short exhaustion reversal wants positive extension first, then reversal pressure

This setup is not trying to catch a neat pullback in trend. It is trying to
catch the point where directional travel is becoming too stretched and too
unstable to continue cleanly.

Default bracket profile:

- `sl_mult = 1.00`
- `tp_mult = 1.65`
- `horizon_bars = 12`

Its tighter target profile reflects that reversal trades usually monetize faster
or fail faster than orderly continuation trades.

#### 10.5.4 Transition Burst

`transition_burst` is the regime-shift setup.

It requires:

- high transition energy
- either an actual regime change or high novelty
- strong fast-flow magnitude
- meaningful directional bias
- no rotational hazard

This setup is trying to capture the bar cluster where market structure is
changing fast enough that a new directional burst may emerge.

Compared with `compression_release`, which likes release out of compression,
`transition_burst` is more about abrupt reorganization. It is willing to trade
the moment the market stops looking like its recent self.

Default bracket profile:

- `sl_mult = 1.20`
- `tp_mult = 2.10`
- `horizon_bars = 16`

### 10.6 Setup competition and thresholds

All four setup families are scored on every eligible row, on both long and short
variants. AetherFlow then selects the highest-scoring setup key, but only if it
clears the family threshold:

- compression release: `0.26`
- aligned flow: `0.40`
- exhaustion reversal: `0.42`
- transition burst: `0.42`

If no family clears its threshold, `candidate_side` is zero and the engine
declares that there is no setup.

This is a key design choice. The model is not asked to decide whether any state
is tradable. The setup dictionary first decides whether a recognizable state
exists at all.

### 10.7 The learned model's role

Once a setup family and side have been selected, `aetherflow_strategy.py` feeds
the feature row into the learned model and gets a success probability.

The learned model is therefore answering a narrower question than the setup
dictionary:

"Given that this bar looks like setup X on side Y, how likely is this candidate
to succeed?"

This separation is what makes AetherFlow easy to reason about:

- deterministic setup identity
- probabilistic validation

### 10.8 Bracket and holding-horizon resolution

`resolve_setup_params()` converts the chosen setup's ATR-scaled defaults into
actual point distances.

The runtime:

- multiplies ATR by the setup's stop multiplier
- multiplies ATR by the setup's target multiplier
- clips stop distance into a sane minimum and maximum range
- clips target distance with both a hard cap and a minimum reward-to-risk floor

It also returns a setup-specific holding horizon. That horizon is part of the
setup definition, not just a generic live rule.

### 10.9 Why the live deployment only uses some regimes and setups

Current live config is intentionally selective.

Only sessions `1, 2, 3` are allowed, which correspond to:

- LONDON
- NY_AM
- NY_PM

Only two setup families are allowed live:

- `compression_release`
- `transition_burst`

And one regime is hard-blocked:

- `ROTATIONAL_TURBULENCE`

This tells you a lot about current trust in the strategy. Live AetherFlow is
being used as a directional structural engine, not as a full-spectrum manifold
research engine. The trend-like and regime-shift families are trusted; the
clean-flow and stress-reversal families exist in code, but are not currently
deployed.

### 10.10 What makes AetherFlow different

AetherFlow is the most compactly explainable active engine in the stack.

DE3 chooses among many members. RegimeAdaptive bends rules by calendar context.
MLPhysics estimates a full bracket distribution. AetherFlow instead says:

- identify the manifold regime
- identify the setup family
- confirm with a model
- trade with setup-specific geometry

That makes it the closest thing in this bot to a self-contained research paper
turned into runtime code.

## 11. Shared Runtime Services

The active strategies do not live alone. Several shared systems make the bot
work as a real trading application.

### 11.1 ProjectX client

`client.py` is responsible for:

- authentication
- account discovery
- contract lookup
- historical bar retrieval
- order placement
- open-order search
- position lookup
- stop-order tracking
- trade fill summary retrieval
- user stream integration

This file is the bridge between strategy logic and the broker.

### 11.2 Event logging

`event_logger.py` emits structured messages like:

- strategy signals
- filter checks
- trade placement
- trade close
- system-mode events

The dashboard bridge depends on these structured logs to reconstruct strategy
status for the frontend.

### 11.3 Bot state

`bot_state.py` provides:

- simple JSON serialization
- trading-day anchoring
- restart continuity for runtime status

This is deliberately lightweight and operator-friendly.

### 11.4 Dashboard bridge

`tools/filterless_dashboard_bridge.py` parses:

- `topstep_live_bot.log`
- `bot_state.json`
- `live_trade_factors.csv` when present

It turns that information into `filterless_live_state.json`, which the frontend
reads to render:

- active strategy statuses
- last signals
- last trades
- heartbeat / readiness state
- current position state

## 12. Order Lifecycle

From a technical perspective, the live trade lifecycle is:

1. a strategy emits a candidate
2. the bot determines execution priority
3. the bot converts the signal into bracket parameters
4. `client.py` places the ProjectX order
5. the bot records active-trade state
6. position sync and the user stream update runtime state
7. bracket-management or close logic updates the trade lifecycle
8. close events are logged and persisted

The system uses both a local shadow position model and broker-side reads because
live trading state can be noisy or delayed if it depends on only one source.

## 13. Repository Map

The most important technical files are:

### Live orchestration

- `launch_filterless_workspace.py`
- `launch_filterless_live.py`
- `julie001.py`
- `client.py`
- `config.py`
- `event_logger.py`
- `bot_state.py`
- `process_singleton.py`

### Active strategies

- `dynamic_engine3_strategy.py`
- `de3_v4_runtime.py`
- `de3_v4_router.py`
- `de3_v4_lane_selector.py`
- `de3_v4_bracket_module.py`
- `regime_strategy.py`
- `regimeadaptive_artifact.py`
- `regimeadaptive_gate.py`
- `ml_physics_strategy.py`
- `aetherflow_strategy.py`
- `aetherflow_features.py`

### Sentiment (emergency exits)

- `services/sentiment_service.py`

### Dashboard

- `tools/filterless_dashboard_bridge.py`
- `tools/filterless_static_server.py`
- `montecarlo/Backtest-Simulator-main/`

### Research / backtesting

- `backtest_mes_et.py`
- `backtest_mes_et_ui.py`
- `train_dynamic_engine3.py`
- `train_aetherflow.py`
- `tools/run_de3_backtest.py`

## 14. Notes on the Rest of the Repo

This repository still contains many other strategy modules and research
artifacts that are not part of the current filterless live roster.

Examples include:

- older breakout and VIX logic
- legacy mixed-stack filters
- DE3 training and hybrid-build scripts
- research reports
- historical model variants

Those files are useful for development and backtesting, but they are not the
best starting point for understanding how the current live bot works. The best
starting point is the filterless path described in this README.
