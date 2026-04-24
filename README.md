# Julie Filterless: Technical Architecture and Strategy Guide

This document explains the current Julie bot from a technical perspective:

- what the live bot is and how the runtime is structured
- how market data becomes orders
- how the active strategy engines work
- the four entry-path safety filters (loss-cluster protections) and when each fires
- the regime classifier (whipsaw / calm_trend / dead_tape / neutral) and its bracket / size / risk overrides
- the iter-11 risk layer (LossFactorGuard + regime-adaptive circuit breaker)
- how the ML overlay stack layers on top of the rule-based engines
- the AI-loop auto-improver (journal → analyzer → validator → applier → monitor)
- how the Kalshi gate interacts with execution and dashboard visibility
- how the sentiment monitor and emergency exits fit into live trading
- how to bootstrap the local workspace and FinBERT dependencies
- what the important artifacts and files are

It is intentionally focused on the current filterless live stack, because that
is the path the bot is actually meant to run on.

## Entry-Path Safety Filters

Every strategy signal passes through a fixed chain of safety filters
before the bot places an order. Four of those filters are
loss-cluster protections — each catches a different pathological
pattern in recent trade history. They run in series at all four
signal-birth hook sites in `julie001.py` using the same
`consensus-rescue` / `rescue-trigger` / `event-logger` plumbing, so a
strong cross-filter consensus can still override any individual
block.

All four filters are active by default when the bot is started via
the launcher. Each filter has an `ACTIVE` env flag that can be set
to `0` before launching to demote the filter to a no-op for that
session, plus tunable parameters that the AI-loop auto-adjuster can
move within documented bounds.

### Filter 1 — `DirectionalLossBlocker`

Counts **strictly-consecutive** losses on one side with no time
bound. After N consecutive same-side losses, that side is blocked
for a configurable cool-off window; a single winning trade on the
same side resets the counter. The filter also has a "bias reversal"
mode that flips allowed direction entirely after 4 consecutive same-
side losses for the rest of the current session quarter (Daye's
Quarterly Theory).

Module: `directional_loss_blocker.py`. Parameters:
`consecutive_loss_limit` (default 3), `block_minutes` (default 15),
`bias_reversal_limit` (default 4), `JULIE_DLB=1` to enable.

### Filter 2 — `CascadeLossBlocker`

Counts same-side losses **inside a sliding time window**, ignoring
intervening wins. Fires when at least `COUNT` losing trades land on
the same side within the last `WINDOW_MIN` minutes; every new
same-side entry is rejected for `COOLDOWN_MIN` measured from the
most recent qualifying loss (so each additional same-side loss
extends the cooldown). Catches rapid-fire loss clusters that
`DirectionalLossBlocker`'s strictly-consecutive count can miss when
wins sneak between losses.

Module: `cascade_loss_blocker.py`. Launcher defaults:

```
JULIE_CASCADE_BLOCKER_ACTIVE=1            # default 1 (ON)
JULIE_CASCADE_BLOCKER_COUNT=2             # bounds [2,4]
JULIE_CASCADE_BLOCKER_WINDOW_MIN=30       # bounds [10,60]
JULIE_CASCADE_BLOCKER_COOLDOWN_MIN=30     # bounds [10,60]
```

### Filter 3 — `AntiFlipBlocker`

Rejects a new **opposite-side** signal that fires near the price
where the last trade just stopped out. Side-asymmetric: a SHORT
stop-out blocks near-price LONG entries only; same-side SHORT
re-entries are unaffected. A stop-out is detected either by a
`stop` / `stop_gap` source tag (backtest tape) or by the exit price
being within 1.5 points of the trade's recorded SL (live broker
tag). Only stop-out losses populate the filter; winning exits, manual
closes, reversals, and horizon exits never record state.

Module: `anti_flip_blocker.py`. Launcher defaults:

```
JULIE_ANTI_FLIP_BLOCKER_ACTIVE=1          # default 1 (ON)
JULIE_ANTI_FLIP_WINDOW_MIN=30             # bounds [5,60]
JULIE_ANTI_FLIP_MAX_DIST_PTS=8.0          # bounds [2.0,15.0]
```

### Filter 4 — RL regime / MFE gates

Two policy guards sit between the RL trade-management policy and
the broker, rejecting `TIGHTEN_SL_25PCT` / `TIGHTEN_SL_50PCT`
actions before they reach the executor. The regime gate blocks
TIGHTEN actions in `calm_trend` regime (lets only `MOVE_SL_TO_BE`
through); the MFE-floor gate blocks any TIGHTEN whose current MFE is
less than a configurable fraction of the trade's TP distance.
Together they prevent the RL from ratcheting stops behind a trend
on every favorable bar and getting swept on mean-reversion
pullbacks.

Implemented in `julie001._apply_rl_management_action`. Launcher
defaults:

```
JULIE_RL_REGIME_GATE_ACTIVE=1             # default 1 (ON)
JULIE_RL_MIN_MFE_FRAC_FOR_TIGHTEN=0.50    # MFE/TP ratio floor
```

Rejected actions emit `[RL_LIVE] status=skipped` log lines with the
gate name and the relevant scalars so post-hoc audits can identify
exactly which guard fired.

### Filter state + persistence

`DirectionalLossBlocker`, `CascadeLossBlocker`, and
`AntiFlipBlocker` all implement the same persistence API
(`get_state` / `load_state`) and their state round-trips through
`bot_state.json` across restarts. Stop-out history, cooldown clocks,
and consecutive-loss counters survive bot restarts cleanly. The RL
executor gates are stateless (they consult the current trade's
regime label and live MFE at each call).

All four filters are part of the AI-loop auto-adjust whitelist.
Activation flags are marked `high_risk` so the applier will not
flip them without manual confirmation; the numeric tunables
(windows, distances, counts, MFE thresholds) are auto-adjustable
within their documented bounds subject to the backtest validator
gate. See the AI-Loop section below for details.

---

## Regime Classifier

A rolling close-based regime classifier labels the current session
into one of five states, then mutates configuration and signal
brackets accordingly. Enabled via `JULIE_REGIME_CLASSIFIER=1`
(launcher default).

### Regimes

| Regime | Trigger | Effect |
|---|---|---|
| `warmup` | Bootstrap state before enough closes have accumulated | No-op |
| `dead_tape` | `vol_bp < JULIE_REGIME_DEAD_VOL_BP` (default 1.5) — bottom-percentile volatility | Rewrites signal brackets to TP=3 / SL=5 / BE-trigger=3, forces size=1, disables BE-arm mechanism |
| `whipsaw` | `vol_bp > 3.5` AND `eff < EFF_LOW` (default 0.05) — violent chop | Raises Kalshi entry-block buffer to 0.25, tightens adaptive circuit breaker |
| `calm_trend` | `eff > EFF_HIGH` (default 0.12) — directional with moderate/low vol | Raises opposite-reversal confirmation requirement to 5 bars, loosens adaptive CB |
| `neutral` | Everything else | Restores all config keys to their pre-regime baselines |

Each transition respects a `TRANSITION_COOLDOWN_BARS` (default 30)
debounce so the classifier can't oscillate across a single noisy
print.

### Dead-tape bracket rewrite

On bottom-percentile-vol sessions, DE3's default TP=25 / SL=10 / BE-
at-10pt never activate because the day's max favorable excursion
only reaches 3-6 points. `apply_dead_tape_brackets` runs inside
`_signal_birth_hook` (called at every signal-birth site in
`julie001.py`) and rewrites `tp_dist` / `sl_dist` to scalp values
before the pct_overlay snapshot or any downstream BE / trail logic
sees them.

Three additional protections fire together on dead-tape:

1. **Size override.** The signal's `size` field is forced to 1
   regardless of strategy-internal sizing — a 3-point scalp risk
   should not be amplified by a 10-contract sizing decision that
   was calibrated against 25-point TPs.
2. **BE-arm disabled.** A 3pt BE trigger with a 5pt SL is close
   enough to current market that slippage at entry plus the BE-move
   can put the new stop on the wrong side of market, triggering a
   "crossed stop" force-flatten for a small loss. With BE-arm off,
   the trade runs to either the 3pt TP or the 5pt SL, no mid-trade
   management.
3. **No trail.** Same reasoning as BE-arm disable; the TP is tight
   enough that trailing produces force-flatten risk without reward.

Env tunables (all read at `regime_classifier.py` import time):

```
JULIE_REGIME_DEAD_VOL_BP=1.5          # vol-bp below which regime=dead_tape
JULIE_REGIME_DEAD_TP=3.0              # rewritten tp_dist on dead-tape signals
JULIE_REGIME_DEAD_SL=5.0              # rewritten sl_dist on dead-tape signals
JULIE_REGIME_DEAD_BE_TRIGGER=3.0      # reference BE trigger (recorded but BE-arm disabled)
```

### Regime-ML action-space decomposition (v5, shipped 2026-04-24)

The legacy `apply_dead_tape_brackets` coupled three actions — scalp bracket
rewrite, size→1, BE-arm off — behind a single vol_bp<1.5 threshold. v5
decomposes these into three independently-gated functions (`apply_scalp_brackets`,
`apply_size_reduction`, `apply_be_disable`) so each action can be driven by
its own ML model or fall back to the rule.

Training: HGB + LightGBM ensemble with cost-sensitive class weights,
40 features (vol_bp at 5 lookbacks, vol slopes, ATR, range/body stats,
momentum, volume-z, session timing, cross-strategy proxies).
Outcome-labeled over 15-minute forward windows on 2024-07 → 2026-01
training / 2026-01-27 → 2026-04-20 OOS holdout.

Ship-or-kill per model against strict gates:

| Model | Action | Verdict | OOS lift (April) |
|---|---|---|---|
| A — scalp brackets | rewrite TP/SL to 3/5 | **SHIP** @ thr=0.70 | +$3,916 PnL vs rule, DD $2,001 ≤ $2,442 |
| B — size reduction | force size=1 | KILL in combined | passed alone but regressed PnL -$11k when stacked with A |
| C — BE-arm disable | skip BE move-to-entry | KILL | lost at every threshold |

Combined A=ML, B=rule, C=rule on April OOS: **+$35,979 PnL vs rule-all
+$23,635 (+$12,343 lift, -$2,526 DD)**. Both combined gates pass.

Env flags in `launch_filterless_live.py`:

```
JULIE_REGIME_ML_BRACKETS=1    # Model A shipped (set to 0 to revert to rule)
JULIE_REGIME_ML_SIZE=0        # Model B kept on rule fallback
JULIE_REGIME_ML_BE=0          # Model C kept on rule fallback
```

When `JULIE_REGIME_ML_BRACKETS=1`, `apply_scalp_brackets` queries the
HGB+LightGBM ensemble at every signal-birth using a feature snapshot
built from the classifier's 520-bar OHLCV history (`record_bar` called
from julie001's per-bar hook). If the ensemble's dead_tape probability
≥ 0.70, brackets rewrite to 3/5; otherwise the rule's vol_bp<1.5 check
applies.

Artifacts: `artifacts/regime_ml_v5_brackets/model.pkl` (ensemble
payload) + `feature_order.json` (feature schema + threshold).
Training script: `scripts/ml_regime_v5_three_models.py`. OOS metrics:
`artifacts/regime_ml_v5_summary.json`.

Previously killed attempts preserved for audit:
`scripts/ml_regime_classifier.py` (v1 supervised-on-rules),
`ml_regime_classifier_v2.py` (outcome-labeled per-bar),
`ml_regime_classifier_v3.py` (confidence-thresholded hybrid),
`ml_regime_classifier_v4.py` (stacked improvements — identified that
coupled action space was blocking a real +$3,193 lift).

### 120-bar startup pre-warm

The classifier needs at least `WINDOW_BARS` (default 120) recent
closes before it can emit anything other than `warmup`. At startup,
`julie001.py` pre-warms the classifier by feeding it the last 120
historical bars from the bootstrap warmup dataset before the live
bar feed opens. Without pre-warm the classifier would spend the
first ~2 hours of live session in `warmup` regardless of tape
conditions, missing dead_tape / whipsaw / calm_trend detection on
the most important early part of the session.

A matching 120-bar pre-warm runs for the `LossFactorGuard` bar
cache at the same bootstrap point, so the iter-11 risk layer also
starts the live session with enough context to score the first
signals.

Every pre-warmed classifier emits a startup log line including the
current regime it classified the bootstrap window into:

```
Regime classifier pre-warmed with 120 historical bars |
  current regime=dead_tape vol=1.25bp eff=0.027
```

### Adaptive circuit-breaker per regime

When `JULIE_REGIME_ADAPTIVE_CB=1`, the classifier mutates the
circuit-breaker's daily loss cap and consecutive-loss limit on each
regime transition. Tight caps on whipsaw (default $250 / 4 consec),
medium on neutral ($350 / 5 consec), loose on calm_trend ($500 / 7
consec). Dead-tape inherits the neutral caps — the scalp brackets
already constrain per-trade exposure.

---

## Iter-11 Risk Layer

The iter-11 risk layer wraps the loss-cluster filters with two
additional protections: a per-strategy big-loss classifier (Filter
G) and a `LossFactorGuard` that tracks loss velocity across the
session.

### LossFactorGuard

Tracks per-bar loss factor — a rolling ratio of recent drawdown to
recent gains — and vetoes new entries when the current factor
exceeds a regime-dependent threshold. Module: `loss_factor_guard.py`.
Integration hooks in `julie001.py`:

- `init_guard` — one-time setup with regime-specific thresholds.
- `notify_trend_day` — updates the trend-day context used in the
  guard's regime classifier.
- `notify_bar` — per-bar update of the rolling loss cache.
- `notify_trade_closed` — records the trade's realized outcome so
  the rolling window reflects genuine loss velocity rather than
  noise.
- `should_veto_entry` — the entry-path query; returns a
  `(vetoed, reason)` tuple read at every signal-birth site.

A 120-bar startup pre-warm populates the guard's bar cache before
the live feed opens (same bootstrap path as the regime classifier's
pre-warm) so the guard can score signals from the first live bar.

### Filter G per-cell threshold overrides

Filter G's base threshold (default 0.35) is modified at runtime by
three multipliers, applied in sequence:

1. **Regime multiplier** — per-market-regime mapping. Whipsaw 0.60×
   (more aggressive veto on chop), calm_trend 1.05× (slightly more
   lenient on directional tapes), neutral / warmup 1.0×.
2. **Session multiplier** — per-strategy cumulative-day-PnL
   adaptation. When the strategy is already up $100+, loosen to 1.25×
   (let winners run); when down $200+, tighten to 0.80× (catch
   losers).
3. **Per-cell multiplier** — new as of 2026-04-23. Loaded from
   `ai_loop_data/triathlon/filterg_threshold_overrides.json` (produced
   by `scripts/idea1_filterg_per_cell_calibrate.py`). Per
   (strategy × regime × time-bucket) cell. Cells classified as
   *bleeding* on pre-April 2026 data (avg PnL/trade < −$2, n ≥ 20)
   receive a 0.75× multiplier (tighter = more aggressive veto).
   Cells classified as *strong* (avg PnL/trade > +$5, n ≥ 20)
   receive 1.15× (more lenient). Neutral / unrated cells get 1.0
   (no change).

All three multiplicative effects stack, and the final effective
threshold is floored at 0.25 so Filter G never fires on less than
25% P(big_loss) regardless of regime + cell.

**Per-cell calibration caveat.** The per-cell table was derived from
retrospective PnL outcomes on fired trades (not from directly-observed
Filter G block correctness, which would require SHADOW_GATE logs at
larger scale than currently available). The OOS backtest used an
**optimistic-bound proxy** that assumes tightened thresholds
retroactively remove the worst-outcome trades in bleeding cells. That
proxy showed a clean win (+$3.5k PnL, −$1.2k MaxDD, +3.7pp WR), but
the real live effect will be smaller than the proxy because Filter G
doesn't have perfect retroactive foresight. The direction is correct;
the magnitude should be monitored live and re-validated after 1-2
months of real data via the same script.

Disable with `export JULIE_FILTERG_PER_CELL_ACTIVE=0` before
launching; the flag is also in the AI-loop auto-adjust whitelist
(marked `high_risk` so the applier won't flip it without manual
confirmation).

### Filter G — per-strategy big-loss classifiers

Four GBT classifiers (one per live strategy: DE3, AetherFlow,
RegimeAdaptive, MLPhysics) score each signal's probability of
producing a large loss. Entries are vetoed when `P(big_loss) >=
threshold` (default 0.35). Artifacts:

| Classifier | Artifact |
|---|---|
| Machine Learning G — DE3 | `artifacts/signal_gate_2025/model_de3.joblib` |
| Machine Learning G — AetherFlow | `artifacts/signal_gate_2025/model_aetherflow.joblib` |
| Machine Learning G — RegimeAdaptive | `artifacts/signal_gate_2025/model_regimeadaptive.joblib` |
| Machine Learning G — MLPhysics | `artifacts/signal_gate_2025/model_mlphysics.joblib` |

The classifiers emit a `[SHADOW_GATE_2025]` log line for every
candidate signal with the computed probability and the live veto
decision, so the calibration set accumulates data even when
upstream filters (Kalshi overlay, filter-check chain) reject the
signal before Filter G gets consulted.

### Launcher wiring

Iter-11's risk layer is enabled via the default launcher
environment (`launch_filterless_live.py`). Relevant flags (all set
to their iter-11 defaults via `os.environ.setdefault`):

```
JULIE_CB=1                              # circuit breaker
JULIE_DLB=1                             # directional loss blocker
JULIE_DD_SCALE=1                        # dynamic drawdown scaling
JULIE_REGIME_CLASSIFIER=1               # regime classifier
JULIE_REGIME_ADAPTIVE_CB=1              # per-regime CB thresholds
JULIE_LOSS_FACTOR_GUARD=1               # LossFactorGuard
JULIE_CB_MAX_DAILY_LOSS=350             # daily-loss cap
JULIE_CB_MAX_CONSEC_LOSSES=5            # consecutive-loss cap
JULIE_CB_MAX_TRAILING_DD=0              # trailing DD cap (0 = off)
JULIE_SIGNAL_GATE_2025=1                # Filter G primary veto
```

---

## Triathlon Engine

A per-cell performance tracker with live feedback into signal sizing
and priority. Every signal the bot generates is assigned a **cell** —
the triple of (strategy × regime × time-bucket) — and every cell is
scored across three leagues. Each cell's best percentile across the
three leagues determines its **medal**, and the medal maps to a
multiplier applied to the signal's size and a delta applied to the
signal's priority on the rescue queue.

The engine is enabled via `JULIE_TRIATHLON_ACTIVE=1` (launcher
default). It is a non-blocking overlay — if the ledger is unavailable
or any single recording call fails, the engine fails closed (acts as
unrated / neutral effects) and live trading continues unaffected.

### Cells

A cell is the triple `(strategy, regime, time_bucket)`:

- **strategy** — the signal's engine: `DynamicEngine3`, `AetherFlow`,
  `RegimeAdaptive`, or `MLPhysics`.
- **regime** — from the regime classifier: `whipsaw`, `calm_trend`,
  `dead_tape`, `neutral`, or `warmup`.
- **time_bucket** — NY hour-of-day bucket: `pre_open` (04:00-09:30),
  `morning` (09:30-12:00), `lunch` (12:00-14:00), `afternoon`
  (14:00-16:00), `post_close` (16:00-17:00), `overnight` (17:00-04:00).

The full cell space has up to 4 × 5 × 6 = 120 cells, though in
practice many combinations never fire (e.g. MLPhysics during
`warmup`).

### Three leagues (all higher = better)

- **Purity** — win rate on fired trades in the cell (fraction of
  trades with `pnl_dollars > 0`). Measures the cell's hit rate.
- **Cash** — per-contract normalized average PnL in dollars. Measures
  the cell's dollar edge without penalizing cells that trade smaller
  sizes (e.g. under the dead-tape size=1 override).
- **Velocity** — `1 / median_bars_held` for **winning** trades only.
  Measures how quickly the edge materializes. Losers are excluded
  because losses hitting SL fast would bias every cell toward fake
  velocity.

Cells need at least `MIN_SAMPLES` (default 20) **effective** fired
trades to be rated — effective in the time-decay sense below. Below
the threshold a cell is marked `unrated` and receives neutral medal
effects.

### Time-decay weighting

When scoring cells for medal assignment, each trade carries a weight
that decays exponentially with age:

```
weight(trade, reference) = exp(-ln(2) × age_days / HALFLIFE_DAYS)
```

A trade that is `HALFLIFE_DAYS` old counts 50% as much as a trade
right at the reference point. The default half-life is **120 days**
(set via `JULIE_TRIATHLON_HALFLIFE_DAYS` at launcher load). The
metrics (Purity / Cash / Velocity) are then computed as weighted
aggregates rather than unweighted averages, so a cell's medal
reflects recent behavior more than 12-month-old behavior without
discarding historical data entirely.

The half-life was picked by sweeping `{30, 60, 90, 120}` on the
April 2026 holdout with pre-April as train. 120d produced a clean
win on all three ship-gate metrics versus both the no-medal-effects
baseline (+$678 PnL, WR flat, DD slightly lower) and the unweighted
medal sizing baseline (+$475 PnL, WR flat, DD $222 lower). Shorter
half-lives under-fit: 30d left only 2 cells with enough effective
sample size to rate. Longer half-lives (>120) approach unweighted
behavior.

The effective sample size of a cell is `sum(weights)` across its
trades. A cell with 200 trades from a year ago has a much smaller
effective-n than 100 trades from the last month — a genuinely stale
cell will fall below `MIN_SAMPLES` on effective-n and go unrated even
if its raw count is high. This is the intended safety mechanism
against acting on decayed evidence.

Disable time-decay by setting `JULIE_TRIATHLON_HALFLIFE_DAYS=0`
before launching — that restores unweighted scoring exactly as it
shipped originally. The half-life is also part of the AI-loop
auto-adjust whitelist (bounds 0–365 days, step 30) so the applier
can tune it within bounds as more live data accumulates.

### Medal assignment + effects

Each rated cell's rank in each league is converted to a percentile
(rank / n_rated). The cell's best percentile across the three
leagues determines its medal:

| Medal | Rule | Priority delta | Size multiplier |
|---|---|---:|---:|
| **gold** | top 20% in at least one league | +1 | ×1.50 |
| **silver** | top 50% in at least one league | 0 | ×1.00 |
| **bronze** | top 80% in at least one league | −1 | ×1.00 |
| **probation** | bottom 20% in **every** league simultaneously | −2 | ×0.50 |
| **unrated** | under min-samples threshold | 0 | ×1.00 |

Priority-delta semantics: a POSITIVE priority_delta promotes the signal
in the rescue queue (sorts earlier). `_live_signal_sort_key` in
`julie001.py` subtracts the delta from the base priority so gold's +1
produces a smaller effective sort priority = sorts first; probation's
−2 pushes it later. Signals without a delta (Triathlon off, or cell
unrated) add 0 → no change from the pre-Triathlon sort behavior.

**Honest caveat — the OOS evidence for these effects is weak.** Two
backtests on April 2026 data showed:

- **Size effects** (`scripts/backtest_triathlon_oos.py`, 644-trade
  April holdout): train-period medal ranking did not reliably predict
  April per-cell edge — silver and bronze out-traded gold on avg
  $/trade in the holdout; the +$267 PnL lift was within sample noise;
  max DD got worse. Spearman (train-rank → April-value) was 0.27–0.54.
- **Priority effects** (`scripts/backtest_triathlon_priority.py`, full
  rescue-queue candidate stream from the April replay): only 5 bars
  across 20 days had genuinely-competing candidates, and 0 / 5 flipped
  under the priority-adjusted sort (alphabetical tie-break dominated,
  or both candidates shared the same medal).

The effects were briefly neutralized after those backtests, then
re-enabled on operator instruction. The rationale for keeping them on
despite weak OOS signal: the April 2026 tape is a small, one-month
sample where DE3 dominates and probation is empty; as live data
accumulates across more regimes, the effects may show clearer
transfer. Watch the Triathlon dashboard tab's recent-signals stream
for unexpected behavior, and re-run the backtests after 3+ months of
fresh live data. Flip back to neutral by setting all priority_deltas
to 0 and size_mults to 1.0 in `MEDAL_EFFECTS` (`triathlon/medals.py`).

Priority delta is added to the existing priority score the rescue
queue reads (FAST=2, NORMAL=1, LOOSE=0). Size multiplier is applied
to the signal's `size` field at signal birth (rounded to nearest
integer, floor 1).

The "best percentile" rule lets a cell with a standout league still
get credit even if the other leagues are middling — e.g. a scalp
cell with high velocity but modest cash can still be gold. Probation
is intentionally hard to earn: a cell must be in the bottom 20% in
every dimension simultaneously, not just one.

### Ledger (sqlite)

`ai_loop_data/triathlon/ledger.db` stores four tables:

| Table | Role |
|---|---|
| `signals` | One row per signal (fired or blocked) with strategy, regime, time-bucket, entry price, SL/TP distances, status, and block filter/reason if rejected |
| `outcomes` | One row per resolved signal: realized PnL, exit source, bars held, and a `counterfactual` flag distinguishing real trades from simulated-for-blocked-signals |
| `standings` | Per-cell league scores + medal, append-only; one row per (cell_key, scored_at) so historical snapshots are preserved |
| `current_medals` | The latest per-cell medal assignments — read by the live runtime hot path |
| `retrain_queue` | Queued retrain requests produced when a strategy's aggregate purity drops across two consecutive scoring runs |

The ledger uses WAL mode so the dashboard and analyzer can read
concurrently with the bot writing.

### Counterfactual walk-forward

Every blocked signal carries its intended entry price plus the SL
and TP distances it would have used. The counterfactual resolver
(`python3 -m triathlon resolve-cf`) walks forward from each blocked
signal's timestamp through the price parquet (Layer 0 of the AI
Loop) and determines whether the trade would have hit TP, hit SL,
or timed out. The resulting simulated PnL is written to the
`outcomes` table with `counterfactual=True`. League scoring
excludes counterfactual outcomes from the cash metric by default
(keeping it tied to real realized PnL), but the dashboard can fold
them in to visualize what blocks cost.

### Seeding from 2025 + 2026 historical trades

At setup, the engine was seeded from the same historical bundle the
backtest-consensus journals use: twelve 2025 monthly replays plus the
2026 Jan-Apr multi-source bundle. The seed classifies each trade's
regime by replaying its entry timestamp through the same
`RegimeClassifier._classify` logic the live bot uses, so historical
cells map onto the current taxonomy.

Run the seed with:

```bash
python3 -m triathlon seed       # from 2025 + 2026 historical trades
python3 -m triathlon rescore    # compute standings + assign medals
python3 -m triathlon medals     # print current standings
python3 -m triathlon status     # health summary
```

### Retrain hook

After each rescoring run, `retrain_hook.queue_retrains()` compares
each strategy's aggregate weighted-average purity across the two
most recent standings snapshots. If a strategy drops more than
`PURITY_DROP_THRESHOLD` (default 0.08 absolute), a retrain request
is queued in the `retrain_queue` table with the drop amount and
sample size as its reason.

The queue is a signal, not an action — the engine does **not**
actually retrain models autonomously. A queued entry is visible in
the dashboard, available via `python3 -m triathlon queue-retrains`,
and intended to be consumed by a human operator (or an authorized
scheduled job in the future) who runs the actual retraining.

### Runtime hook (`tools/triathlon_runtime.py`)

A thin adapter the bot imports at startup. Public API:

- `is_active()` — gate reading `JULIE_TRIATHLON_ACTIVE`. All other
  calls short-circuit to safe defaults when false.
- `lookup_signal_effects(strategy, regime, time_bucket)` — hot-path
  point lookup into the cached medal map. Returns
  `{medal, size_mult, priority_delta, cell_key}`.
- `record_signal(...)` — persist a fired / blocked signal row.
  Called from `_signal_birth_hook` at every signal-birth site.
- `record_outcome(signal_id, pnl_dollars, ...)` — persist the
  realized outcome when a trade closes.
- `refresh_medals(force=False)` — reload the cached medal map,
  throttled to a 5-minute refresh interval.

The adapter lazy-opens the sqlite ledger on first use and fails
permanently-closed if that open raises — so a missing DB, a bad
install, or any other startup fault can never break live trading.

### Dashboard tab

The Monte Carlo dashboard's Filterless live view shows a live
Triathlon Engine section that polls `/triathlon_state.json` every
10 seconds. The tab renders the medal tally strip, a filterable
per-cell table (sorted by medal, then cash descending), the last
50 live signals with their block reasons, and the open retrain
queue. The state JSON is refreshed by `python3 -m triathlon export`
(auto-called by `rescore`).

### AI-Loop integration

`JULIE_TRIATHLON_ACTIVE` is in the `AUTO_ADJUSTABLE_PARAMS`
whitelist, flagged `high_risk` so the applier will refuse to flip
it without manual confirmation. The other Triathlon parameters
(medal cutoffs, shaping weights) are compile-time constants in
`triathlon/medals.py` and `triathlon/leagues.py` — they are
deliberately outside the auto-adjust whitelist because changing
them retroactively recategorizes every historical signal.

### Files

| File | Role |
|---|---|
| `triathlon/__init__.py` | Package constants (cell-key helpers, time-bucket boundaries) |
| `triathlon/ledger.py` | SQLite schema + signal/outcome dataclasses + transaction helpers |
| `triathlon/leagues.py` | Per-league metric computation + ranking |
| `triathlon/medals.py` | Medal cutoffs, effects, rescoring, live lookup |
| `triathlon/counterfactual.py` | Forward-walk blocked signals to produce simulated outcomes |
| `triathlon/seed.py` | Bootstrap from historical `closed_trades.json` bundles |
| `triathlon/retrain_hook.py` | Purity-drop detection + retrain queue |
| `triathlon/export.py` | Dashboard JSON exporter |
| `triathlon/__main__.py` | CLI (`python3 -m triathlon seed/rescore/resolve-cf/medals/status/export/queue-retrains`) |
| `tools/triathlon_runtime.py` | Thin adapter for `julie001.py` |
| `montecarlo/.../components/TriathlonTab.tsx` | React tab for the dashboard |
| `ai_loop_data/triathlon/ledger.db` | The live ledger |
| `montecarlo/.../public/triathlon_state.json` | Dashboard-polled state snapshot |

---

## AI-Loop Auto-Improver

A self-contained, safety-gated "AI loop" runs once per weekday during
the CME futures maintenance window and is also invokable manually. It
consumes the bot's own logs, compares them to long-horizon backtest
consensus and to the price-regime parquet, proposes whitelisted
configuration changes, validates each one with a backtest, and only
then applies it — with a complete audit trail and an auto-revert
monitor watching the next several sessions.

### Five-layer stack (`tools/ai_loop/`)

| Layer | Module | What it does |
|---|---|---|
| 0 | `price_parquet_updater.py` | Parses `[INFO] Bar: ... \| Price: X` lines from the live log and every replay log, appends new bars to `ai_loop_data/live_prices.parquet` (manifest-tracked, idempotent) |
| 1 | `journal.py` | Writes today's human-readable markdown journal + structured JSON sidecar: PnL, per-layer block rates, counterfactuals, pattern flags — now also attaches `price_context` (intraday range, trend, per-session range/vol) |
| 2 | `analyzer.py` | Reads last N daily journals + optional long-tape backtest priors + the price parquet, emits whitelisted param-change recommendations and prior-based structural advisories |
| 3 | `validator.py` | Fast-backtests each candidate change against the most recent replay's closed_trades; only passes ones that show ≥5% lift and ≤1.2× DD inflation; passes advisories through as `status=info` |
| 4 | `applier.py` | Applies `status=green` recommendations via env var or joblib payload edit, git-commits with `[AUTO_APPLIED]`, records to audit log |
| 5 | `monitor.py` | Nightly drift check: if live PnL drops >2σ below backtest forecast over 5 sessions, auto-reverts the most recent change |

### Backtest-consensus journals (`tools/ai_loop/backtest_journal.py`)

A companion to the daily journal that reads one-or-many
`closed_trades.json` files from replay directories and emits the same
markdown + structured JSON shape as the daily journal, but aggregated
across an arbitrary tape rather than a single session. The output gives
the analyzer a long-horizon "prior" to compare today's session against,
plus per-day price-regime context joined from the live-prices parquet.
Multiple labels (e.g. `2025_full`, `2026_full`) can be loaded
simultaneously via the analyzer's `--backtest-journal` flag.

### Logs → parquet → analyzer pipeline

Layer 0 (`price_parquet_updater.py`) extracts every per-minute bar from
the live log and from every replay log it can find under
`backtest_reports/`, dedupes by `bar_ts`, and writes them to
`ai_loop_data/live_prices.parquet`. A small manifest tracks log
size+mtime so reruns are idempotent and only scan files that have grown.

`price_context.py` exposes `load_prices()`, `day_context(date)`, and
`window_context(start, end)` helpers that produce per-day open/high/low/
close, intraday range, trend direction, bar-to-bar volatility, and a
per-session breakdown. The daily and consensus journals attach a
`price_context` block to their output, and the analyzer passes the
loaded DataFrame through to every rule via the `prices` keyword
argument so rules can correlate journal outcomes with price regime.

### Safety rails (`tools/ai_loop/config.py`)

Every auto-apply must pass an explicit chain of checks before the
applier touches anything:

- The parameter must appear in the `AUTO_ADJUSTABLE_PARAMS` whitelist.
  Anything outside the whitelist (side disable, session window,
  sub-strategy retirement) can be surfaced as an *advisory* but cannot
  be auto-applied.
- The proposed value must fall inside the bounds declared in the
  whitelist entry.
- The single-apply step size must be no larger than `max_step_delta`.
- The same parameter cannot be re-applied within `COOLDOWN_DAYS` of the
  last successful change to it (default 7 days).
- The total number of applies in any single day across all parameters
  must not exceed `MAX_APPLIES_PER_DAY` (default 2).
- A global kill switch (`JULIE_FREEZE_AUTO_CONFIG=1`) short-circuits
  the entire applier and can be flipped without restarting the bot.
- A live-PnL stop-loss freezes auto-applies for a week if drawdown over
  the last 48 hours exceeds `STOP_LOSS_48H_DOLLARS` (default $500).
- A backtest-validator gate runs every candidate change against the
  most recent replay's closed trades; only ones that show
  `BACKTEST_MIN_LIFT_PCT` lift and inflate max drawdown by at most
  `BACKTEST_MAX_DD_INFLATE` are green-lit.
- Parameters marked `high_risk: True` (e.g. activation flags for the
  RL policy or the cascade blocker) are refused even when validated —
  they require manual confirmation.

Every apply attempt — green-lit, skipped, or rejected — appends a
record to `ai_loop_data/audit.jsonl` with the commit SHA (when applied),
parameter, current value, proposed value, validator result, and the
applier's verdict.

### Scheduling (`tools/ai_loop/schedule/`)

A `launchd` plist fires the orchestrator weekdays at 14:30 PT (17:30 ET)
— thirty minutes into the CME futures maintenance window, when the bot
has written its final session state but the market is closed so there
is no race with live trades. The same orchestrator can be invoked
manually:

```bash
bash tools/ai_loop/schedule/install.sh       # install + load
bash tools/ai_loop/schedule/uninstall.sh     # remove
tail -f ai_loop_data/run_daily.log           # observe
python3 -m tools.ai_loop.run_daily --dry-run # run manually without applying
```

### Key files

| File | Role |
|---|---|
| `tools/ai_loop/run_daily.py` | Orchestrator that runs all five layers in sequence |
| `tools/ai_loop/price_parquet_updater.py` | Layer-0 log → parquet extractor |
| `tools/ai_loop/price_context.py` | Parquet loader plus per-day / per-window regime helpers |
| `tools/ai_loop/journal.py` | Layer-1 daily journal writer (markdown + JSON, with price-context) |
| `tools/ai_loop/backtest_journal.py` | Long-tape consensus journal builder |
| `tools/ai_loop/analyzer.py` | Layer-2 recommendation engine; rule modules consume journals + priors + prices |
| `tools/ai_loop/validator.py` | Layer-3 backtest-gated validator; advisories pass as `status=info` |
| `tools/ai_loop/applier.py` | Layer-4 auto-apply with git audit trail |
| `tools/ai_loop/monitor.py` | Layer-5 drift monitor + auto-revert |
| `tools/ai_loop/config.py` | Whitelist + safety constants |
| `tools/ai_loop/state.py` | Persistent state (cooldowns, freeze flag, PnL history) |
| `tools/ai_loop/schedule/com.julie.ai_loop.plist` | `launchd` weekday 14:30 PT trigger |
| `ai_loop_data/live_prices.parquet` | Logs-extracted per-minute bar history |
| `ai_loop_data/journals/` | Daily and consensus journals (markdown + JSON) |
| `ai_loop_data/recommendations/` | Analyzer output per run |
| `ai_loop_data/validated/` | Validator output per run |
| `ai_loop_data/applied/` | Applier output per run |
| `ai_loop_data/audit.jsonl` | Append-only audit log of every apply attempt |

---

## Machine Learning Overlay Stack — Live Layers

The bot runs a layered ensemble of machine-learning models on top of
the rule-based strategy engines. Each layer observes a specific
decision point and either vetoes, modifies, or augments what the
rule-based engine wanted to do. Layers fall into three families:

1. **Per-strategy big-loss classifiers** (Filter G) that veto a
   signal before any order is placed when the predicted probability
   of a large loss is high.
2. **Overlay layers** that modify the trade after a signal has passed
   the entry filters: choosing the fill style, predicting the trade's
   ability to reach a level, or scoring the prediction-market crowd's
   view.
3. **A reinforcement-learning policy** that takes per-bar
   trade-management actions on every active position.

All layers are live by default when launched via
`launch_filterless_live.py`. The launcher uses `os.environ.setdefault`
for every flag so per-session overrides are honored, and any individual
layer can be demoted to shadow-only (it still logs every decision but
takes no live action) by exporting its env flag set to `0` before
launching.

### Machine Learning G — per-strategy big-loss veto (live since prior releases)

| Classifier | Role | Artifact |
|---|---|---|
| **Machine Learning G: DE3** | Veto DE3 signals with high predicted big-loss probability | `model_de3.joblib` |
| **Machine Learning G: AetherFlow** | Veto AF signals with high predicted big-loss probability | `model_aetherflow.joblib` |
| **Machine Learning G: RegimeAdaptive** | Veto RA signals with high predicted big-loss probability | `model_regimeadaptive.joblib` |
| **Machine Learning G: MLPhysics** | Veto MLPhysics signals with high predicted big-loss probability | `model_mlphysics.joblib` |

### Overlay stack — five learned layers on top of the rule-based engines

| Layer | Decision point it models | Artifact |
|---|---|---|
| **LFO** (Level Fill Optimizer) | Should this signal fill IMMEDIATE or WAIT for a bank level? | `model_lfo.joblib` |
| **PCT overlay** (% level bias) | On a fresh touch of a %-from-open level, is the forward move breakout or pivot? | `model_pct_overlay.joblib` |
| **Pivot trail** (SL ratchet) | Will this confirmed swing pivot actually hold its trail level? | `model_pivot_trail.joblib` |
| **Kalshi entry gate** | Past the rule's support-score filter, is this trade still worth taking? | `model_kalshi_gate.joblib` |
| **Kalshi TP gate** | Does the Kalshi crowd believe this trade can reach its take-profit price? | `model_kalshi_tp_gate.joblib` |

### RL trade-management policy (Path 3, live)

A PPO-trained reinforcement-learning agent replaces the rule-based
stack of trade-management heuristics (break-even, profit milestone,
tiered take, pivot trail) with one joint policy that picks a per-bar
action from the full trade context. The policy actively steers stops
in production; activation is controlled by
`JULIE_ML_RL_MGMT_ACTIVE` (launcher default `1`).

**Action space.** The canonical policy has four discrete actions
(`HOLD`, `MOVE_SL_TO_BE`, `TIGHTEN_SL_25PCT`, `TIGHTEN_SL_50PCT`).
Every action the policy can emit has a wired live broker path.
A larger seven-action variant (`TAKE_PARTIAL_50PCT`,
`TAKE_PARTIAL_FULL`, `REVERSE`) is kept as a reference artifact
under `model_rl_management_v2_for_future_partial_close.zip` for the
day the partial-close broker path is built; until then those actions
have no live execution and would silently no-op if the canonical
policy could emit them.

**Observation layout (212 dims).** The agent sees:

- 150 dims — 30 bars × 5 OHLCV features (bar tape)
- 7 dims — trade-state scalars (pnl, bars held, mfe, mae, side sign, current SL distance, current TP distance)
- 4 dims — regime one-hot
- 6 dims — session one-hot
- 3 dims — Kalshi aligned probabilities (entry, SL, TP)
- 2 dims — running peak / trough PnL
- 32 dims — bar-sequence encoder embedding (Path 1)
- 8 dims — cross-market feature vector (Path 4)

The encoder embedding and the cross-market vector are snapshotted once
at trade entry and reused on every subsequent bar; their per-bar cost
is dominated by the cache lookup.

**Executor safety.** `_apply_rl_management_action` in `julie001.py`
translates the policy's chosen action into a broker call and applies
several layered guards before the action reaches the wire:

1. **Ratchet check.** A proposed new SL must be tighter than the
   current SL; the RL is never permitted to loosen the stop.
2. **Market-breach check.** A proposed new SL must sit on the
   protective side of current market by at least one tick. Without
   this guard, a `MOVE_SL_TO_BE` on a trade that is slightly
   underwater would place the new SL on the wrong side of market and
   the broker would force-fill the exit immediately.
3. **Regime gate.** In the `calm_trend` regime the executor rejects
   `TIGHTEN_SL_25` and `TIGHTEN_SL_50` and lets only `MOVE_SL_TO_BE`
   through; the trend is intact and chasing the SL invites a sweep
   on a normal pullback.
4. **MFE-floor gate.** Any TIGHTEN action is rejected when the trade's
   maximum favorable excursion is less than a configurable fraction of
   its TP distance (default 0.50). The trade must have moved a
   meaningful share of the way to TP before SL tightening is permitted.

Each rejected action emits a `status=skipped` log line with the
relevant scalars so audits can identify exactly why an action was
blocked. The trade retains its existing SL.

**Reward shaping.** The training environment (`rl/trade_env.py`)
includes optional shaping signals that bias the policy away from the
same failure modes the executor blocks:

- A per-step penalty for TIGHTEN actions in the `calm_trend` regime.
- A per-step penalty for TIGHTEN actions when MFE / TP is below a
  threshold.
- A terminal opportunity-cost penalty equal to a fraction of the gap
  between the realized PnL and the best PnL an unmodified hold would
  have produced (capped at the original TP, so the policy is never
  penalized for failing to hit unrealistic targets). This teaches the
  policy directly that locking in a small win when a large win was
  available is a worse outcome than waiting.

All shaping weights are CLI flags on `rl/train_ppo.py`; setting them
to zero recovers the unshaped reward for backwards compatibility.

**Trainer:** `rl/train_ppo.py` (use `--extended-obs --sl-only` to
reproduce the canonical training configuration; add the
`--calm-trend-tighten-penalty`, `--low-mfe-tighten-penalty`, and
`--opportunity-cost-weight` flags to apply shaping).
**Env:** `rl/trade_env.py` (`extended_obs`, `n_actions=4`, plus the
shaping kwargs).
**Inference:** `rl/inference.py` auto-detects obs_dim and action space
at load and caches per-trade observation augmentation via `trade_id`.
**Comparison:** `rl/compare_policies.py --model-v2 …` and
`scripts/ab_rl_v3_vs_v4.py` for head-to-head scoring of any two
policies on the validation split.

### Bar-sequence encoder (Path 1, live)

A self-supervised dilated-CNN encoder ingests the most recent 60 bars
of OHLCV and produces a 32-dim embedding intended to summarize the
recent price tape in a way other learned layers can consume cheaply.
The encoder is trained as an auxiliary task on next-bar-direction
prediction (three-way up / flat / down) with a small return-regression
component.

**Artifact:** `artifacts/signal_gate_2025/bar_encoder.pt` (PyTorch
checkpoint).
**Trainer:** `rl/bar_encoder.py`.
**Integration:** `ml_overlay_shadow.get_bar_embedding(bars_df)` returns
the 32-dim embedding for the most recent window. The embedding is
snapshotted once at trade entry and cached for subsequent reads inside
the same trade.

The encoder is an active input into the LFO and RL layers. The LFO
scoring path auto-computes the embedding when the loaded payload's
`uses_bar_encoder` flag is set; the RL observation includes the
embedding as part of its 212-dim vector.

Other classifier overlays (Kalshi TP, Pivot Trail, PCT overlay) do
not currently consume the encoder — they are local pattern
classifiers where the sequence context did not produce measurable
lift in retrain testing. The encoder is therefore a targeted input
for decision-type layers rather than a universal augmentation.

### Cross-market features (Path 4, live)

A stable-schema feature extractor that joins MES bars with several
external series:

```
mnq_ret_5min_pct          MNQ short-term return
mnq_ret_30min_pct         MNQ medium-term return
mes_mnq_corr_30bar        30-bar rolling MES↔MNQ correlation
mes_mnq_divergence_pct    spread between MES and MNQ 30-min returns
vix_level                 absolute VIX close
vix_regime_code           0=calm / 1=normal / 2=high / 3=extreme
vix_rate_of_change_5d     VIX 5-day pct change
dxy_level                 USD index (currently stubbed at 100)
```

The cross-market data is sourced from continuous non-adjusted
front-month MNQ 1-min bars (Databento) and VIX daily bars (Yahoo).
Both series are cached as parquet files under `data/`. Roll-day
protection clips MES↔MNQ divergence and the MNQ return features so
front-month switches do not produce artifact jumps that would
destabilize a downstream model.

**Module:** `rl/cross_market.py`.
**Integration:**
`ml_overlay_shadow.get_cross_market_features(ts_et, mes_bars=...)`.
**Live consumers:** the LFO model (as eight additional numeric
features in its input) and the RL policy (as eight scaled dims
appended to the 212-dim observation). The extractor is singleton-loaded
on first call so per-lookup cost is dominated by a small dataframe
slice rather than a parquet read.

All nine are live by default when launched via `launch_filterless_live.py`.
Full details in **section 11 (Machine Learning Overlay Stack)** below.

Env toggles — **all six ML decision layers default ON when launched via `launch_filterless_live.py`**
(the launcher sets each with `os.environ.setdefault` so they flip to active
automatically at startup). Any layer can be opted back OUT of active steering
by exporting the var as `0` in the shell before running the launcher:

```
# default behavior when you run LaunchFilterlessWorkspace.bat or
# launch_filterless_live.py — all six ML layers are ACTIVE (drive decisions):
JULIE_ML_LFO_ACTIVE=1           # LFO fill decisions by v2 ML (64 features)
JULIE_ML_PCT_ACTIVE=1           # PCT overlay bias by ML
JULIE_ML_PIVOT_TRAIL_ACTIVE=1   # skip pivot ratchets when ML says "pivot won't hold"
JULIE_ML_KALSHI_ACTIVE=1        # block signals when entry-gate ML predicts loss
JULIE_ML_KALSHI_TP_ACTIVE=1     # block signals when TP-gate ML predicts loss
JULIE_ML_KALSHI_TP_PNL_THR=0    # TP-gate threshold (block when pred_pnl <= $X)
JULIE_ML_RL_MGMT_ACTIVE=1       # RL v3 actively steers stops (as of 2026-04-22)

# to demote a layer back to shadow-only (logs but doesn't steer):
export JULIE_ML_RL_MGMT_ACTIVE=0   # e.g. to revert RL to shadow after the
                                    # 2026-04-22 flip
```

If you import `julie001` directly without the launcher (rare — only happens in
test scripts), all six layers default to **shadow mode** (log only, rule
steers) — the fail-safe default.

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

## 11. Machine Learning Overlay Stack

The four engines described in sections 7-10 are rule-based. In 2026-04, a
parallel ML overlay stack was added on top of them: four learned models that
observe the same state the rule-based logic observes and emit a second opinion
for every decision point that matters to P&L. Default behavior is shadow mode
(log both decisions; rule drives). Each layer can be flipped to active
steering via a dedicated env var.

### 11.1 Design principles

These are the invariants the ML overlay stack follows:

- **Shadow-first, always.** Every new model ships logging-only. It writes a
  `[SHADOW_*]` event alongside the rule's decision so we can audit agreement
  and disagreement on live bars before committing to any behavior change.
- **One env kill-switch per layer.** Live activation is a single environment
  variable, so the operator can promote layers independently or roll back
  instantly with no code change.
- **The rule still runs.** Even in active ML mode, the rule's output is
  computed and logged. That preserves a continuous reference series for
  A/B monitoring and makes regression detection trivial.
- **All models are gradient-boosted trees (sklearn GBT).** Same deployment
  path, same `joblib` artifact shape, same feature-encoding contract. This
  keeps the inference layer (`ml_overlay_shadow.py`) tiny.
- **Training is fully reproducible from the public parquet bars.** Each
  trainer lives in `scripts/signal_gate/` and walks either
  `es_master_outrights.parquet` (for market-structure models) or the
  historical `backtest_reports/full_live_replay/` directories (for models
  that need real-trade outcomes as labels).

### 11.2 Inference module

`ml_overlay_shadow.py` is the single load-and-score module. On bot startup,
`init_ml_overlays()` eagerly loads the four joblib payloads from
`artifacts/signal_gate_2025/` and returns `(lfo_loaded, pct_loaded,
pivot_loaded, kalshi_loaded)`. If a model is missing, its scoring function
returns `None` and the bot falls back to rule behavior silently — the overlay
layer is fail-safe by construction.

Four public scorers exist, one per layer:

- `score_lfo(signal, bar_features, dist_to_bank_below, dist_to_bank_above, …)`
- `score_pct_overlay(state)`
- `score_pivot_trail(pivot_type, bar OHLC, market-state, anchor, session, tape, et_hour)`
- `score_kalshi(signal, bar_features, regime, et_hour_frac, role)`

Four matching `is_<layer>_live_active()` helpers read the kill-switch env
vars (`JULIE_ML_LFO_ACTIVE`, `JULIE_ML_PCT_ACTIVE`,
`JULIE_ML_PIVOT_TRAIL_ACTIVE`, `JULIE_ML_KALSHI_ACTIVE`).

### 11.3 Layer 1 — Level Fill Optimizer ML

**Rule it shadows.** `level_fill_optimizer.py` decides whether a DE3 / AF /
RA signal should fill IMMEDIATE (at market on the next bar) or WAIT for a
bank-level pullback. The rule uses hand-tuned distance thresholds
(0.75 pts / 2.50 pts) plus priority hints.

**What the ML learns.** Binary classifier: "on this signal, does WAIT beat
IMMEDIATE on realized P&L?"

**Training set.** Real DE3+RA+AF trades pulled from the canonical monthly
replays. For each trade, the trainer simulates BOTH fill modes on the
actual parquet bars around `signal_bar = entry_time - 1 min`:

- IMMEDIATE → fill at next bar open
- WAIT → walk forward 3 bars; fill at the nearest $12.50 bank level if
  touched in favorable direction; else time out at the 3rd bar's close

Each trade gets a label: 1 if WAIT's realized P&L > IMMEDIATE's, else 0.
Training labels are therefore grounded in the actual bar tape the trade
ran through, not in synthetic price paths.

**Features.** *v1 (25 features):* Bar-shape features at signal time
(range, body, upper/lower wicks), distances to the nearest bank levels
above and below, ATR14, original stop/take distances, regime one-hot,
session bucket, ET hour.

*v2 — current canonical (64 features):* all of the v1 features, plus
the 32-dim bar-sequence encoder embedding (`enc_00 … enc_31`) and the
8-dim cross-market feature vector (MNQ returns / correlation / VIX
level / VIX regime code / etc.). Retrained via
`scripts/signal_gate/retrain_with_encoder.py`. Promotion from v1 to v2
was gated by a rolling-origin A/B harness
(`scripts/signal_gate/rolling_origin_ab.py`) that requires the new
model to beat the incumbent on the AUC metric across multiple
chronological splits before the joblib is renamed to canonical.

**What the layer contributes.** The live rule decides WAIT vs IMMEDIATE on
a small set of geometric rules that doesn't learn from outcomes. The ML
replacement folds in bar-shape, regime, and session context — so in chop
regimes it's more likely to pick IMMEDIATE (don't give a reversion tape a
chance to drift past the bank), and in clean trends it's more likely to
pick WAIT (let the pullback come). v2 additionally adjusts based on
MNQ–MES divergence and VIX regime. In shadow mode it just logs both
decisions; in active mode it directly steers the fill choice.

**Live hook.** `julie001.py` LFO decision site. Shadow log example:

```
[SHADOW_LFO] rule=WAIT ml=IMMEDIATE p_wait=0.320 thr=0.400 strat=DE3 side=LONG
```

**Trainer.** `scripts/signal_gate/train_lfo_ml.py`

### 11.4 Layer 2 — Percentage Level Overlay ML

**Rule it shadows.** `pct_level_overlay.py` assigns a bias
(`breakout_lean` / `pivot_lean` / `neutral` / `chop`) to every %-from-session-open
level touch (±1%, ±2%, ±3%). Rule uses a hand-tuned confluence score blended
from ATR bucket, range bucket, hour edge table, and per-level base edge.

**What the ML learns.** Binary classifier: "on a fresh touch of this level,
is the forward 60-min outcome BREAKOUT (extends 0.10% past level) vs PIVOT
(retraces 0.15% back)?"

**Training set.** Fresh-touch events generated by walking
`es_master_outrights.parquet` chronologically from 2020 forward. Each event
gets a label by looking forward 60 min and recording which condition fired
first (breakout extension past the level vs retrace back through it).
Labels come from actual forward-looking bars, so the model is training on
what actually happened after every level touch in the historical tape —
no synthetic paths.

**Features.** `pct_from_open`, signed and absolute level, `level_distance_pct`,
`atr_pct_30bar`, `range_pct_at_touch`, per-hour edge, minutes-since-open,
distance to running session hi/lo, the rule's own confluence score (teacher
leak so ML can re-weight it), plus tier / ATR bucket / range bucket / hour
bucket / direction categoricals.

**What the layer contributes.** The rule's confluence score is built from a
small hand-tuned basket of features. The ML adds features the rule doesn't
use — running-hi and running-lo distance in particular turned out to matter
a lot more than the rule assumed — and re-weights the existing ones against
outcomes. Overlay output is a bias (breakout/pivot/neutral) used by the
bot's existing sizing/direction logic; the overlay does not place its own
trades. In active mode the ML-derived bias replaces the rule's bias at
each fresh touch; in shadow mode both are logged and the rule's value is
still what the rest of the bot uses.

**Live hook.** `julie001.py` fresh-touch evaluation site. Shadow log example:

```
[SHADOW_PCT] rule=breakout_lean ml=pivot_lean p_breakout=0.432 lvl=+2.00% dist=0.015%
```

**Trainer.** `scripts/signal_gate/train_pct_overlay_ml.py`

### 11.5 Layer 3 — Pivot Trail ML

**Rule it shadows.** `_compute_pivot_trail_sl` ratchets the SL to one
bank-level behind a confirmed swing pivot (Reading B, fallback Reading C)
every time `_detect_pivot_high` / `_detect_pivot_low` fires, gated only on
`min_profit_pts ≥ 12.5`. There's no predictive signal in the rule — it
trails on **every** confirmed pivot, regardless of whether that pivot is
structurally meaningful or just noise.

**What the ML learns.** Binary classifier: "given this confirmed pivot,
will price respect the Reading-B trail level for the next 20 bars, or
will it get violated quickly?"

**Training set.** Every confirmed pivot found by walking the full 2020+
parquet. Label = "held" only if the forward 20 bars never had **two
consecutive closes** through the trail threshold. The close-based
confirmation is important: single wicks poke through nearby levels all
the time on minute data, so a label that treats any single wick as a
violation becomes degenerate. Requiring two consecutive closes through
the level filters intrabar noise and keeps the label focused on real
structural breaks — which is what actually causes a trailed stop to
close a trade.

Stratified downsample keeps the dataset balanced and training tractable.

**Features.** Pivot bar shape (range, body, upper/lower wick %), pivot
prominence (height vs the surrounding window mean), ATR14, 30-bar range,
20-bar trend %, distance from the 20-bar high and low, 5-bar and 20-bar
velocities, distance to the nearest 12.5-pt bank, pivot type (HIGH/LOW),
session bucket, tape classification (uptrend/downtrend/chop), ET hour.

**What the layer contributes.** The rule ratchets on every confirmed
pivot, which means every single time the 5-bar detector fires the stop
moves — regardless of whether the pivot is a real structural level or
just a local wiggle. The ML turns this into a **selective** ratchet: on
high-confidence pivots the ratchet fires as before; on low-confidence
pivots (the majority) the ratchet is skipped so the original, wider SL
stays in place and the trade keeps running. In shadow mode both
decisions are logged; in active mode the ML's skip overrides the rule's
ratchet.

**Live hook.** `julie001.py` pivot-trail evaluation site. Shadow log
example:

```
[SHADOW_PIVOT] type=HIGH px=5249.00 sl=5224.75 rule=RATCHET ml_p_hold=0.412 ml=SKIP
```

**Live mode behavior** (`JULIE_ML_PIVOT_TRAIL_ACTIVE=1`): the rule's
candidate SL is computed, but the ratchet is only applied when ML
confidence clears threshold. Low-confidence pivots get skipped → the
original SL stays in place → trade continues.

**Label-design history.** First two attempts produced degenerate datasets
because the violation test was done against bar lows/highs, so single
wicks would trip the label. Switching to close-based confirmation
(current design) was the fix. Details in commit `e00da42`.

**Trainer.** `scripts/signal_gate/train_pivot_trail_ml.py`

### 11.6 Layer 4 — Kalshi Entry Gate ML

**Rule it shadows.** `_apply_kalshi_trade_overlay_to_signal` computes a
`support_score` (blend of entry_probability, probe_probability,
momentum_retention) and blocks signals where `support_score < threshold`
(0.45-0.55 depending on role). Single feature, single threshold.

**What the ML learns.** Dual-head binary classifier + pnl regressor: "given
Kalshi features + market-state context, (a) is this trade profitable? and
(b) what's the expected pnl_dollars?"

**Training set.** PASS events matched to real closed trades across a
curated list of canonical monthly + outrageous-event-day replays. The
allowlist deliberately **excludes** experimental-variant replays (altered
filter configs, iteration sweeps, baseline comparisons) whose different
filter stacks would cause the same Kalshi features to map to different
outcomes. That label pollution was the biggest single threat to Kalshi
ML quality and is addressed by allowlisting.

All matched rows are `DynamicEngine3` because the live Kalshi gate only
applies to DE3 (AetherFlow and RegimeAdaptive don't go through the Kalshi
entry overlay). Match key: exact `(strategy, side, round(entry_price, 2))`.

**Features.** Kalshi-native features read from the KALSHI_ENTRY_VIEW log
(entry_probability, probe_probability, momentum_delta, momentum_retention,
support_score, threshold, probe_distance_pts) plus ET hour fraction plus
DE3 substrategy features (tier, is_rev, is_5min) plus market-state
computed from the parquet at signal time (ATR14, 30-bar range,
20-bar trend %, distance from 20-bar hi/lo, 5-bar velocity, distance to
nearest bank) plus regime one-hot (whipsaw / calm_trend / neutral,
computed from a 120-bar close window matching `regime_classifier.py`).

**Iteration history.** Three training rounds:

- **v1** used a flat 50+ replay directory set and random 5-fold CV. Both
  choices were wrong: the extra replays were experimental variants with
  different filter configs (label pollution), and random-shuffle CV leaks
  future information into training folds on time-series data. The
  validation numbers looked great but the model had no real edge.
- **v2** added the replay allowlist and replaced random CV with
  rolling-origin (train on oldest N%, test on next chunk, step forward).
  This is an honest time-series evaluation. It revealed that after
  removing label pollution, the feature set from v1 had essentially no
  generalizable signal.
- **v3** (shipped) added regime computed directly from parquet bars (v2
  parsed `Regime transition:` log lines but the canonical replays don't
  write them, so v2's regime was always "warmup" — effectively a dead
  feature), added DE3 substrategy features, and added a parallel
  regressor that predicts `pnl_dollars` directly. The binary label
  collapses a $5 scratch and a $200 winner into the same class; the
  regressor is what lets the model exploit that asymmetry.

**What the layer contributes.** Past the rule's flat threshold, the ML
uses market-state context (regime, volatility, recent velocity) and
substrategy identity to re-evaluate whether a PASS-ed trade is actually
worth taking. It's a second-stage filter on top of the rule. In shadow
mode both decisions are logged; in active mode, a low ML score on a
rule-PASS flips the decision to block.

**Live hook.** `julie001.py` Kalshi entry-view site. Shadow log example:

```
[SHADOW_KALSHI] rule=PASS ml_p_win=0.624 ml_pred_pnl=+31.2 ml=PASS strat=DynamicEngine3 side=LONG regime=neutral support=0.578 thr=0.450
```

**Live mode behavior** (`JULIE_ML_KALSHI_ACTIVE=1`): when the rule would
PASS but ML `p_win < 0.50`, the signal is blocked
(`kalshi_entry_blocked=True`).

**Trainer.** `scripts/signal_gate/train_kalshi_ml.py`

### 11.6a Layer 4b — Kalshi Take-Profit Gate ML

**What it models.** The Kalshi entry gate (11.6) reads aligned probability
at the ES entry price plus a short 5-point forward probe. The take-profit
gate asks a complementary question that the rule overlay does not ask
directly: *what does the Kalshi crowd say about the take-profit price
itself?* If the ladder assigns a low aligned probability at the TP strike,
the market is essentially pricing the trade's goal as unlikely — a useful
second-pass filter on top of the entry gate.

**What the ML learns.** Binary classifier: "given the Kalshi ladder's
reading at TP-price plus market-state context, will this trade reach its
take-profit?"

**Training set.** The same canonical replay allowlist as the entry-gate
trainer, but with a different feature-extraction pipeline:

1. Parse `tp_dist` from the DE3 sub_strategy tag (e.g. `..._SL10_TP25`
   → tp_dist=25)
2. Compute `tp_price = entry_price ± tp_dist` (by side)
3. For the trade's entry timestamp, pick the next Kalshi settlement event
   (12-16 ET gating hours) and pull the full strike ladder for that event
   from the daily `kxinxu_hourly_2025.parquet` snapshot
4. Interpolate aligned probability at `tp_price` across the two nearest
   strikes — mirrors `_interpolated_aligned_probability` in
   `kalshi_trade_overlay.py` so the training reading matches the live
   reading exactly
5. Label = HIT_TP (1 if trade's `exit_source` is `take` or `take_gap`;
   else 0)

This is a different labeling philosophy from the entry-gate ML. That one
uses `pnl_dollars > 0` (any profitable outcome). This one explicitly
asks whether the trade reached its programmed target — a cleaner match
for "was Kalshi right about the TP being reachable?"

**Features.**

- **TP-strike relationship** (the core of this model): `tp_aligned_prob`,
  `tp_dist_pts`, `tp_prob_edge` (= tp_aligned_prob − 0.50),
  `tp_vs_entry_prob_delta`
- **Ladder shape around TP**: `nearest_strike_dist` (interpolation error),
  `nearest_strike_oi` (liquidity proxy), `nearest_strike_volume`,
  `ladder_slope_near_tp` (local probability derivative in ±10pt window)
- **Time-to-event**: `minutes_to_settlement` (trades closer to the
  settlement hour exhibit stronger pinning behavior)
- **Entry anchor**: `entry_aligned_prob` for scale
- **Market state**: ATR14, 30-bar range, 20-bar trend %, 5-bar velocity
- **Substrategy**: tier, is_rev, is_5min
- **Regime** one-hot (whipsaw / calm_trend / neutral) computed from the
  same 120-bar close window as the regime classifier

**What the layer contributes.** The rule overlay's `support_score` is
built from entry_probability and a short 5pt probe — it never queries
the ladder at the actual take-profit price. This model fills that gap:
past the entry filter, it looks at whether the crowd is actually pricing
settlement at the TP strike as likely. In active mode it can block signals
whose TP looks structurally unreachable according to Kalshi, even when
the entry-side reading was acceptable.

It runs in parallel with the entry-gate ML, not in series — both shadow
logs fire on the same PASS events, and in active mode either can veto
independently.

**Live hook.** `julie001.py` Kalshi entry-view site (immediately after
the entry-gate ML). Shadow log example:

```
[SHADOW_KALSHI_TP] rule=PASS ml_p_hit_tp=0.412 ml=BLOCK tp_px=5274.00 tp_dist=25.0 tp_prob=0.326 entry_prob=0.512 strat=DynamicEngine3 side=LONG regime=neutral
```

**Live mode behavior** (`JULIE_ML_KALSHI_TP_ACTIVE=1`): when the rule
would PASS but ML `p_hit_tp < 0.50`, the signal is blocked
(`kalshi_entry_blocked=True`, reason tagged as `ml_kalshi_tp`).

**Trainer.** `scripts/signal_gate/train_kalshi_tp_ml.py`

### 11.7 Machine Learning G (per-strategy big-loss veto, shipped earlier)

Before the 2026-04 overlay stack, a family of per-strategy ML classifiers
was already live in `signal_gate_2025.py`. Those models — collectively
called **Machine Learning G** — evaluate whether a generated signal
(DE3 / AetherFlow / RegimeAdaptive / MLPhysics) looks like a
high-probability-big-loss before the bot places the trade.

Each strategy has its own dedicated ML classifier (not a single shared
model):

```
artifacts/signal_gate_2025/
  model_de3.joblib              — Machine Learning G for DE3
  model_aetherflow.joblib       — Machine Learning G for AetherFlow
  model_regimeadaptive.joblib   — Machine Learning G for RegimeAdaptive
  model_mlphysics.joblib        — Machine Learning G for MLPhysics
```

Unlike the overlay stack (which defaults to shadow mode until the launcher
activates each layer), Machine Learning G has been **live** since it
shipped — it directly vetoes signals whose predicted loss probability
exceeds the per-strategy threshold. It runs independently from the
overlay stack described above; the two systems don't share code paths or
features.

Activation: the launcher sets `JULIE_SIGNAL_GATE_2025=1` by default; all
four Machine Learning G classifiers load at bot startup and evaluate
every candidate signal. Revert path: `export JULIE_SIGNAL_GATE_2025=0`.

### 11.8 The five-layer stack, end to end

A single live signal passes through all five overlays plus Machine Learning G
in this order:

```
  Strategy candidate (DE3 / AF / RA / MLPhysics)
            │
            ▼
    Machine Learning G (signal_gate_2025.py)  ─── live; blocks P(big loss) > thr
            │
            ▼
    Kalshi entry overlay (rule)   ─── computes support_score vs threshold
            │
            ▼   (if rule PASS, both Kalshi ML layers score in parallel)
    SHADOW_KALSHI (entry-gate ML)     ─── logs p_win; blocks if LIVE
    SHADOW_KALSHI_TP (TP-gate ML)     ─── logs p_hit_tp; blocks if LIVE
            │
            ▼
    LFO decision                  ─── rule or ML chooses WAIT/IMMEDIATE
            │  SHADOW_LFO
            ▼
    PCT overlay bias applied      ─── rule or ML biases size/direction
            │  SHADOW_PCT
            ▼
    Kalshi size multiplier (up to 3x)  ─── live; multiplies contract size
            │                               when crowd aligns
            ▼
    Execution (client.py → ProjectX bracket order)
            │
            ▼
    Management loop (per-bar):
      ├─ pivot-trail detector fires
      │     │  SHADOW_PIVOT ML scores p_hold
      │     ▼
      │   Rule or ML ratchets SL to Reading-B
      └─ Kalshi TP-trail, regime transitions, emergency exits, etc.
```

### 11.9 How the layers compose

Running all five overlays simultaneously is intentional — each targets a
different decision point and they compose rather than compete:

- **Machine Learning G** (live, prior art) — per-strategy big-loss ML
  classifier that filters out signals the strategy shouldn't have emitted
  in the first place.
- **Kalshi entry-gate ML** re-evaluates the rule's PASS decisions using
  entry-side features (entry_probability + 5pt probe) and market-state
  context the rule doesn't see.
- **Kalshi TP-gate ML** independently asks whether the crowd is actually
  pricing the take-profit as reachable. Runs in parallel with the entry
  gate, not in series — either can veto.
- **LFO ML** chooses WAIT vs IMMEDIATE on signals that survive both
  Kalshi gates.
- **PCT overlay ML** biases the bot's sizing/direction logic when the
  signal happens at a %-level touch.
- **Pivot ML** selectively suppresses unnecessary SL ratchets during
  position management.

Because each layer is independent and shadow-gated, rolling out (or
rolling back) one layer doesn't affect the behavior of any of the others.
The bot's rule-based path always remains intact as a fallback.

### 11.10 Training artifacts

All five overlay models plus the four Machine Learning G classifiers and
their training parquets live in a single directory:

```
artifacts/signal_gate_2025/
  model_lfo.joblib                        — Level Fill Optimizer
  model_pct_overlay.joblib                — PCT level bias
  model_pivot_trail.joblib                — pivot trail hold gate
  model_kalshi_gate.joblib                — Kalshi entry gate (clf + reg)
  model_kalshi_tp_gate.joblib             — Kalshi TP gate (clf + reg)
  model_de3.joblib                        — Machine Learning G for DE3
  model_aetherflow.joblib                 — Machine Learning G for AetherFlow
  model_regimeadaptive.joblib             — Machine Learning G for RegimeAdaptive
  model_mlphysics.joblib                  — Machine Learning G for MLPhysics
  lfo_training_data.parquet               — LFO training set
  pct_overlay_training_data.parquet       — PCT overlay training set
  pivot_trail_training_data.parquet       — pivot hold labels
  kalshi_training_data.parquet            — Kalshi entry-gate training set
  kalshi_tp_training_data.parquet         — Kalshi TP-gate training set
```

All five overlay trainers live in `scripts/signal_gate/` and are fully
deterministic given the bars + replay logs + Kalshi parquet. To regenerate
any model from scratch:

```bash
python3 scripts/signal_gate/train_lfo_ml.py
python3 scripts/signal_gate/train_pct_overlay_ml.py
python3 scripts/signal_gate/train_pivot_trail_ml.py
python3 scripts/signal_gate/train_kalshi_ml.py
python3 scripts/signal_gate/train_kalshi_tp_ml.py
```

### 11.11 Observation and rollout plan

Shadow-mode logs are tailed in production via `topstep_live_bot.log`. Key
lines to grep:

```
grep '[SHADOW_LFO]'         topstep_live_bot.log   # LFO rule-vs-ML disagreement
grep '[SHADOW_PCT]'         topstep_live_bot.log   # PCT bias disagreement
grep '[SHADOW_PIVOT]'       topstep_live_bot.log   # Pivot ratchet disagreement
grep '[SHADOW_KALSHI]'      topstep_live_bot.log   # Kalshi entry-gate disagreement
grep '[SHADOW_KALSHI_TP]'   topstep_live_bot.log   # Kalshi TP-gate disagreement
```

Recommended rollout order (one layer at a time, one week shadow minimum
before activation):

1. **LFO** first. Choosing a fill mode can't create trades the bot wouldn't
   otherwise take, so the downside is bounded to "might have gotten a
   marginally worse fill on some signals." Safe first-mover.
2. **Pivot trail** second. Pivot events are rare (roughly ≤1 per trading
   day on average), so disagreement frequency is low and each disagreement
   can be inspected directly. Bounded downside: at worst, a skipped
   ratchet leaves the trade on its original SL, which is what would have
   happened if the pivot simply hadn't been detected.
3. **PCT overlay** third. The overlay biases size/direction on an existing
   signal rather than creating trades, but disagreements are more
   frequent than pivot, so more shadow observation time is warranted.
4. **Kalshi TP gate** fourth. A veto-only layer; the worst that happens
   is a trade gets blocked that the rule would have passed. No new trades
   created. The TP-gate has the cleaner label (HIT_TP vs not) of the two
   Kalshi ML layers, so it's the natural first Kalshi activation.
5. **Kalshi entry gate** last. Also a veto-only layer, but its
   disagreement with the rule's support_score threshold is subtle and
   benefits from the most shadow observation before activation.

Revert is always `export JULIE_ML_<LAYER>_ACTIVE=0` + bot restart; nothing
else needs to change.

## 12. Shared Runtime Services

The active strategies do not live alone. Several shared systems make the bot
work as a real trading application.

### 12.1 ProjectX client

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

### 12.2 Event logging

`event_logger.py` emits structured messages like:

- strategy signals
- filter checks
- trade placement
- trade close
- system-mode events

The dashboard bridge depends on these structured logs to reconstruct strategy
status for the frontend.

### 12.3 Bot state

`bot_state.py` provides:

- simple JSON serialization
- trading-day anchoring
- restart continuity for runtime status

This is deliberately lightweight and operator-friendly.

### 12.4 Dashboard bridge

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

## 13. Order Lifecycle

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

## 14. Repository Map

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
- `tools/run_full_live_replay.py` — runs the actual live loop against replayed bars (emits KALSHI_ENTRY_VIEW + SHADOW_* logs for ML overlay analysis)
- `tools/run_topstep_replay.py` — pulls fresh Topstep historical bars and replays through the backtest engine

### ML overlay stack (see Section 11)

- `ml_overlay_shadow.py` — runtime model loader + scorer for all four overlays
- `signal_gate_2025.py` — Machine Learning G (per-strategy big-loss veto; live, not shadow)
- `artifacts/signal_gate_2025/` — model artifacts (joblib) + training data (parquet)
- `scripts/signal_gate/train_lfo_ml.py`
- `scripts/signal_gate/train_pct_overlay_ml.py`
- `scripts/signal_gate/train_pivot_trail_ml.py`
- `scripts/signal_gate/train_kalshi_ml.py`

## 15. Notes on the Rest of the Repo

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
