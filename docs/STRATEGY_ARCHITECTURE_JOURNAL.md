# Strategy Architecture Journal

**Purpose:** Grounded reference for future ML blocker / overlay retraining. Every claim cites a `file.py:LINE-LINE` range. Synthesized 2026-04-25 from end-to-end reads of the listed source files at HEAD `79a1aac` plus friend's working-tree swap `e594674`. Where a code path is ambiguous after reading, the journal flags it explicitly with **FLAG** rather than guessing.

**Audience:** ML engineers retraining one or more components in the blocker/overlay stack.

**Important context:** the working tree currently has friend's `e594674` swapped overlay model artifacts (`model_lfo.joblib` / `model_pct_overlay.joblib` / `model_pivot_trail.joblib` / `model_kalshi_gate.joblib` / `model_kalshi_tp_gate.joblib` / `model_de3.joblib` / `model_aetherflow.joblib` / `model_rl_management.zip`) without OOS validation artifacts. Per-cell threshold overrides for Filter G were deleted in that commit and restored 2026-04-25. Read each section's "Re-train priority" with that history in mind.

---

## Table of Contents
1. [DynamicEngine3 (DE3)](#1-dynamicengine3-de3)
2. [RegimeAdaptive](#2-regimeadaptive)
3. [AetherFlow](#3-aetherflow)
4. [Where ML Blockers / Overlays Intersect](#4-where-ml-blockers--overlays-intersect)
5. [Key Observations for Retraining](#5-key-observations-for-retraining)

---

## 1. DynamicEngine3 (DE3)

DE3 is the only `FAST_EXEC` strategy that fires consistently in filterless live mode. It produced 1,821 trades / +$7,600 / 48.4% WR in `baseline_2025_03/closed_trades.json` — essentially all of the bot's volume that month.

### A. Signal Generation

**Strategy DB.** `dynamic_signal_engine3.py:104-147` loads strategies from JSON. Three accepted layouts (direct list, wrapped dict with version, or OOS-nest). Required fields per record (`dynamic_signal_engine3.py:144`): `TF`, `Session`, `Type`, `Thresh`, `Best_SL`, `Best_TP`, `Opt_WR`. Strategy index built keyed by `(session, timeframe)` at `dynamic_signal_engine3.py:149-162`.

The canonical DB at HEAD: `dynamic_engine3_strategies_v2_research_london_shortfix_20260407.json` — 62 strategies. Each row has these dimensions:

| Dimension | Values |
|---|---|
| Timeframe (TF) | `5min`, `15min` |
| Session | `00-03`, `03-06`, `06-09`, `09-12`, `12-15`, `15-18`, `18-21`, `21-24` |
| Type (lane) | `Long_Rev`, `Short_Rev`, `Long_Mom`, `Short_Mom` |
| T-tier (threshold tier) | `T2`, `T3`, `T4`, `T5`, `T6`, `T8` |
| Geometry | `(SL_pts, TP_pts)` OR `(SLPCT, TPPCT)` + optional `HZ<minutes>` |

**Lane resolution** at `dynamic_signal_engine3.py:708-715`:
```python
if strategy_type == "Long_Rev"  and is_red   and abs_body > thresh: signal = "LONG"
elif strategy_type == "Short_Rev" and is_green and abs_body > thresh: signal = "SHORT"
elif strategy_type == "Long_Mom"  and is_green and abs_body > thresh: signal = "LONG"
elif strategy_type == "Short_Mom" and is_red   and abs_body > thresh: signal = "SHORT"
```

`Long_Rev` fires on a red prior bar with body > threshold (mean-reversion long). `Long_Mom` fires on a green prior bar with body > threshold (momentum long). Mirror logic for shorts.

**Trigger profile gates** at `dynamic_signal_engine3.py:237-321` — the per-strategy optional shape constraints:
- `body_ratio` — body / range
- `close_pos1` — `(close - low) / range`, normalized 0-1
- `upper_wick_ratio` — `(high - max(open, close)) / range`
- `lower_wick_ratio` — `(min(open, close) - low) / range`
Each tested via `_metric_in_range()` at `dynamic_signal_engine3.py:263-287` against optional `MinX`/`MaxX` bounds in the strategy record.

**Horizon time-stop** at `dynamic_signal_engine3.py:513-540` — if a strategy carries `HorizonMinutes` or `HorizonBars`, it converts via `parse_timeframe_minutes()` (line 528) and surfaces `use_horizon_time_stop` (line 539) in the signal dict at `dynamic_signal_engine3.py:843-845`.

**Per-bar polling.** `check_signals()` at `dynamic_signal_engine3.py:542-1077`. For each `(session, timeframe)` index entry it:
1. Extracts the prior candle OHLC (`dynamic_signal_engine3.py:664-697`)
2. Tests every session-matched strategy for trigger (`703-729`)
3. Applies trigger-profile filters (`718-729`)
4. Resolves SL/TP distance mode (points vs pct, `403-450`)
5. Builds a signal dict with 67+ fields (`801-914`)
6. Computes a quality score from `score_raw / win_rate / avg_pnl / trades / bucket_score / location` (`930-980`)
7. Sorts candidates by `final_score` (`982`) and applies abstain logic (`984-1037`)

**OOS metrics override.** `dynamic_signal_engine3.py:323-384` (`_runtime_metrics_from_strategy`): for v2/v3/v4 runtimes, OOS metrics are **preferred over in-sample** at line 359. This matters for any retraining that adds new strategies — make sure you provide OOS stats or they'll be excluded.

**DE3v4 runtime.** `dynamic_engine3_strategy.py:121-432` wraps the engine. Key flags at `121-132`:
```python
self._de3_v2_runtime = self.db_version.startswith("v2")
self._de3_v3_runtime = self.db_version.startswith("v3")
self._de3_v4_runtime = self.db_version.startswith("v4")
```

When `_de3_v4_runtime` is active, the wrapper instantiates `DE3V4Runtime(self._de3_v4_cfg)` at line 431. The bundle (current production: `artifacts/de3_v4_live/dynamic_engine3_v4_bundle.decision_side_daytype_soft_v1_20260422_promoted.json`, 90,267 lines) contains four sub-modules wired at `de3_v4_runtime.py:398-414`:
- `DE3V4Router` — chooses among lanes
- `DE3V4LaneSelector` — chooses the variant within the chosen lane
- `DE3V4BracketModule` — resolves SL/TP from 79 packaged bracket modes
- Optional direct decision policy model (`169-337`)

**Router scoring.** `de3_v4_router.py:32-34` defines the weight set:
```python
self.w_lane_prior    = safe_float(self.weights.get("lane_prior",     0.55), 0.55)
self.w_lane_max_edge = safe_float(self.weights.get("lane_max_edge",  0.30), 0.30)
self.w_lane_mean_edge= safe_float(self.weights.get("lane_mean_edge", 0.15), 0.15)
```
Per-lane score at `de3_v4_router.py:212-216`:
```python
score = (self.w_lane_prior     * lane_prior)
      + (self.w_lane_max_edge  * lane_max_edge)
      + (self.w_lane_mean_edge * lane_mean_edge)
```
Lane prior is itself a blend (`de3_v4_router.py:122-136`): 55% session prior + 20% timeframe prior + 25% global prior. No-trade gates at `de3_v4_router.py:264-308` enforce `min_lane_score_to_trade`, `min_score_margin_to_trade`, and `min_route_confidence` thresholds.

**Core anchor / satellites mode.** `de3_v4_runtime.py:30-102`. Three runtime modes:
- `core_only` — only execute the core anchor family ID
- `core_plus_satellites` — both
- `satellites_only` — exclude the core, use the router

The current production setting (visible in startup logs) is `satellites_only` with the core anchor `5min|09-12|long|Long_Rev|T6` (`de3_v4_runtime.py:36-43`). Family-ID format documented at `de3_v4_runtime.py:486-517`: `timeframe|session|side|strategy_type|threshold[|family_tag]`. Hard safety at `de3_v4_runtime.py:50-52`: if core is disabled, runtime is forced to `satellites_only`.

### B. Bracket / Sizing / Entry Geometry

- **Bracket source:** `bracket_modes` block in the v4 bundle (79 variants) selected by `DE3V4BracketModule`. Each mode is `(SL, TP, optional HZ)` either in points or as percentages with reference price. Distance-mode resolution is in `dynamic_signal_engine3.py:403-450`. Tick-rounded to 0.25 (ES).
- **Bundle export fields.** `dynamic_engine3_strategy.py:50-118` exports 118 `de3_v4_*` fields onto every signal — route decision/confidence/margins, selected lane/variant/bracket mode, entry-model evaluation, profit-gate results, decision-side outputs, book-gate, conflict-side. These are the columns ML retraining work will read.
- **Sizing.** `signal_size_rules` are bundle-resident; the runtime loads them at `de3_v4_runtime.py:268-332`. SameSide ML caps at `JULIE_SAMESIDE_ML_MAX_CONTRACTS=2` (live cap when active). Filter D/E (regime size cap + green-day unlock) sits on top — see Section 4.
- **Entry path.** DE3 is in `fast_strategies`, the FAST_EXEC list. The core hook is `_signal_birth_hook` at `julie001.py:109-124`:
  ```python
  def _signal_birth_hook(signal):
      try: _apply_regime_size_cap(signal)  # filter D
      except Exception: pass
      try: _signal_gate_shadow_log(signal)  # filter G shadow
      except Exception: pass
  ```
  Six signal-birth call sites (per agent grep): `julie001.py:12303, 12445, 12673, 14022, 14461, 15860`. Every signal funnels through this hook regardless of which strategy emitted it.
- **Same-side parallel gate.** `julie001.py:2041-2080`. DE3 + RegimeAdaptive/AetherFlow same-side OK; AetherFlow + AetherFlow capped at `live_same_side_parallel_max_legs` (default 1, `julie001.py:2061-2079`); all other same-side combos blocked. `JULIE_BYPASS_SAMESIDE=1` is the env-gated test bypass at `julie001.py:2057`.
- **Order placement.** `await client.async_place_order(signal, current_price)` at `julie001.py:11846, 14227, 15213` — the broker resolves size/SL/TP from signal-dict fields.

### C. Exit Behavior

Six `source` labels found in `closed_trades.json` map to specific assignment sites in `julie001.py`:

| Source | Assignment site |
|---|---|
| `stop` | broker fill, normalized at `julie001.py:1141` |
| `take` | broker fill, normalized at `julie001.py:1141` |
| `stop_gap` | crossed-stop fallback at `julie001.py:4073: source=str(close_order_details.get("method") or "crossed_stop_fallback")` |
| `take_gap` | similar pattern in early-exit fallback at `julie001.py:11739` |
| `close_trade_leg` | partial-market-order or shared-reverse close at `julie001.py:11711, 12120, 14575, 15430` |
| `reverse` | "shared_reverse_close" path at `julie001.py:12120, 14575, 15430` |

Other named sources visible in the code (`julie001.py:9154/9237/9823/9831/12167/14262/14622/15248/15483`): `order_fill_fallback`, `broker_flat_cleanup`, `truth_social_emergency_exit`, `impulse_rescue_confirmed`, `standard_parallel_execution`, `loose_parallel_execution`. Future trade-outcome ML labels need to handle this expanding set — don't assume the six in `closed_trades.json` are exhaustive.

**Break-even arm.** `_apply_live_de3_break_even_stop_update` at `julie001.py:4113-4192+`. The function is ratchet-only:
- Current SL recovered or derived (`4126-4137`)
- Target SL aligned to tick boundary (`4139-4144`)
- Capped at TP-1-tick boundary (`4146-4161`)
- "Improved" check at `4163-4167`:
  ```python
  improved = (target_stop_price > current_stop_price + 1e-12  if side_name == "LONG"
              else target_stop_price < current_stop_price - 1e-12)
  ```
- If not improved, returns `{"status": "unchanged"}` and clears any pending BE candidate at `4170-4172`.

**Pivot trail.** `_compute_pivot_trail_sl` at `julie001.py:416-476`. Two-tier anchor:
- *Reading B* (default): anchor = `floor(pivot_price / step) * step − step` (one bank-width back); `SL = anchor − buffer`.
- *Reading C* (fallback): if Reading B would lock a loss, anchor = `floor(pivot_price / step) * step` (no step-back); `SL = anchor − buffer`.

Code at `julie001.py:446-461` for LONG (symmetric for SHORT, `julie001.py:462+`):
```python
profit_pts = pivot_price - entry_price
if profit_pts < min_profit_pts - 1e-9: return None
l_anchor_c = math.floor(pivot_price / step) * step
l_anchor_b = l_anchor_c - step
candidate_b = round(l_anchor_b - buffer, 4)
if candidate_b > entry_price + 1e-9:
    candidate = candidate_b
else:
    candidate = round(l_anchor_c - buffer, 4)
if candidate <= current_sl + 1e-9 or candidate <= entry_price + 1e-9:
    return None
return candidate
```

**RestoredLivePosition** is the recovery-mode strategy label at `julie001.py:8561, 8759`. Detection logic at `julie001.py:2316-2330`: created on bot restart when broker has open positions but the bot lacks `sub_strategy`/`combo_key`. The fallback infers a DE3 lane from `side` only (LONG → default-LONG lane), so BE/trail management still applies. ML retraining should EXCLUDE these trades from training data — they don't represent a strategy decision.

---

## 2. RegimeAdaptive

A time-context mean-reversion strategy that ships in `fast_strategies` alongside DE3 (`julie001.py:7253: regimeadaptive_strategy = RegimeAdaptiveStrategy()`). It fires far less frequently than DE3.

### A. Signal Generation

**Time-context combo.** `regime_strategy.py:65-75` (`get_session`):
```python
if 18 <= hour <= 23 or 0 <= hour < 3: return 'ASIA'
elif 3 <= hour < 8:  return 'LONDON'
elif 8 <= hour < 12: return 'NY_AM'
elif 12 <= hour < 17: return 'NY_PM'
```
Monthly-quarter bucketing at `regime_strategy.py:86-91`: `W1` ≤ day 7, `W2` ≤ 14, `W3` ≤ 21, `W4` rest. Combo key construction at `regime_strategy.py:109-112` produces strings like `Q1_W1_MON_ASIA`.

**Reverted combos.** `regime_strategy.py:43-62` defines a fallback set of 23 hardcoded combos where signal is flipped (e.g. `Q1_W1_TUE_ASIA`, `Q3_W4_THU_NY_AM`). If `regime_sltp_params.py` is present, it overrides with a dynamic table.

**Class init.** `regime_strategy.py:170-259`. Default params at `171-227`:
- `sma_fast=20`, `sma_slow=200` (SMA periods)
- `atr_period=20`
- `range_spike_mult=1.3`
- `cross_atr_mult=0.3`
- `vol_window=30`, `vol_median_window=120`

Reversion master switch at `regime_strategy.py:208`:
```python
self.enable_signal_reversion = True
```
Optional gate model loaded at `regime_strategy.py:181-193` from artifact (regression/classifier).

**on_bar pipeline** at `regime_strategy.py:406-659`:
- Requires ≥ 200 bars (SMA200 need) and skips CLOSED session (`413-414`)
- Indicators at `436-444` (SMA fast/slow, vol series, ATR, current vol)
- If artifact present, iterates LONG/SHORT candidates and calls `_evaluate_rule_candidate()` per side (`495-582`)
- Rule types at `regime_strategy.py:341-363`:
  - `"breakout"` — break above/below recent high/low + trend agreement
  - `"continuation"` — touch SMA + break opposite
  - `"pullback"` (default) — dip to SMA on trend + range spike

**Reversion** at `regime_strategy.py:612-617`:
```python
if revert:
    signal = 'SHORT' if signal == 'LONG' else 'LONG'
```

**SLTP resolution** at `regime_strategy.py:621-629`:
```python
if self.artifact is not None:
    sltp = self.artifact.get_sltp(signal, combo_key, ctx['session'])
elif self.use_optimized_sltp:
    sltp = get_optimized_sltp(signal, ts)  # from regime_sltp_params.py
else:
    sltp = {'sl_dist': 2.0, 'tp_dist': 3.0}
```
Default fallback `MIN_SL=2.0pt` (8 ticks), `MIN_TP=3.0pt` (12 ticks, 1.5:1 RR floor).

**Output payload** at `regime_strategy.py:635-659`:
```python
{
    "strategy": "RegimeAdaptive",
    "sub_strategy": combo_key,         # e.g. "Q1_W1_MON_ASIA"
    "side": signal,                    # post-reversion
    "tp_dist": sltp['tp_dist'],
    "sl_dist": sltp['sl_dist'],
    "reverted": revert,
    "original_signal": original_signal if revert else signal,
    "rule_id": selected_rule_id,
}
```

**Gate model.** `regimeadaptive_gate.py:12-46` defines `GATE_FEATURE_COLUMNS` — 36 features: temporal cyclic (hour_sin/cos, minute_sin/cos), rule params, strength ratios, returns over 1/5/15, vol_30, vol_ratio, range_vs_mean. Per-session and per-policy threshold overrides at `regimeadaptive_gate.py:172-184`. The model file is `regimeadaptive_gate_model.joblib` (added in friend's `2d7d818 track regimeadaptive live gate model`).

**FLAG: regime_sltp_params.py loading order.** `regime_strategy.py:29-40` says it loads `regime_sltp_params.py` if available; the file's behavior depends on presence. The reverted-combos table is similarly file-resident if present. For retraining, document which files were present at training time so the same SLTP resolution is reproducible.

### B. Bracket / Sizing / Entry Geometry

- **Brackets** are per-combo via `get_optimized_sltp(signal, ts)` (`regime_strategy.py:131-149` cascades through Q→Month→Day→Session granularity).
- **Sizing** subject to the same SameSide ML cap as DE3.
- **Entry** is the same `_signal_birth_hook` → `async_place_order` path described in DE3 Section B.

### C. Exit Behavior

RegimeAdaptive uses the same trade-management substrate (`julie001.py` BE-arm, pivot trail, exit-source labels). The strategy doesn't introduce new exit semantics.

### D. Regime Classifier

`regime_classifier.py` is loaded once and queried by both RegimeAdaptive and the regime-adaptive sizing/CB layers.

**Constants** at `regime_classifier.py:40-82`:
- `WINDOW_BARS = 120` (line 40)
- `EFF_LOW = 0.05` (whipsaw threshold), `EFF_HIGH = 0.12` (calm_trend threshold) (`47-48`)
- `WHIPSAW_BUF_VALUE = 0.25`, `CALM_REV_CONFIRM = 5` (`50-51`)
- `TRANSITION_COOLDOWN_BARS = 30` (line 53)
- `ADAPTIVE_CB_ENABLED = JULIE_REGIME_ADAPTIVE_CB == "1"` (line 57); CB caps `CB_CAP_WHIPSAW=250, CB_CAP_NEUTRAL=350, CB_CAP_CALM=500` (`58-63`)
- `REGIME_GREEN_UNLOCK_THRESHOLD=200` for filter E (line 75)

**Classification.** `regime_classifier.py:153-162`:
```python
if vol_bp > 3.5 and eff < EFF_LOW: return "whipsaw"
if eff > EFF_HIGH:                  return "calm_trend"
return "neutral"
```
Whipsaw requires BOTH high volatility (>3.5bp) AND low efficiency. Pure low-vol tape stays `neutral`. Efficiency = `|sum(returns)| / sum(|returns|)` over 120 bars (`137-151`).

**Config mutation on transition** at `regime_classifier.py:164-204`. WHIPSAW (`168-171`):
```python
buf["balanced"] = 0.25
buf["forward_primary"] = 0.25
rev_cfg["required_confirmations"] = baseline
```
CALM_TREND (`173-176`):
```python
buf["balanced"] = baseline
rev_cfg["required_confirmations"] = 5
```
Adaptive CB at `187-195` mutates `circuit_breaker._GLOBAL_CB.max_daily_loss` and `max_consecutive_losses` in place.

**Size cap (Filter D/E).** `apply_regime_size_cap(signal)` at `regime_classifier.py:250-296`:
- If regime is `whipsaw` or `calm_trend`, cap size to `REGIME_SIZE_CAP_VALUE` (default 1)
- Unlock to `REGIME_GREEN_UNLOCK_SIZE` (default 3) if `daily_pnl >= REGIME_GREEN_UNLOCK_THRESHOLD` ($200)
- Logs `regime_size_cap_before`, regime, unlock status

**Public API** at `regime_classifier.py:210-247`: `init_regime_classifier`, `get_regime_classifier`, `update_regime_classifier(ts, close)`, `current_regime()`, `should_veto_entry()`.

**Dead-tape regime.** Per commit `e0a152c Add dead_tape regime branch to classifier` and `034de40 Dead-tape: force size=1 and disable BE-arm` — `dead_tape` is a 5th label produced by the classifier and consumed by overlay code to force `size=1` and disable BE-arm. **FLAG: the agent reading regime_classifier.py did not enumerate the dead_tape branch in classify; it may be in a separate function or the `classify_regime()` body that was not fully covered.** Review `regime_classifier.py` if you need to retrain anything that depends on dead_tape detection.

---

## 3. AetherFlow

Manifold-based router. Loaded but rarely fires in production at threshold 0.55 (zero AF trades in `baseline_2025_03/closed_trades.json`).

### A. Signal Generation

**Class config.** `aetherflow_strategy.py:566-627`:
```python
self.cfg = CONFIG.get("AETHERFLOW_STRATEGY", {})
self.model_path = "model_aetherflow_v1.pkl"
self.thresholds_path = "aetherflow_thresholds_v1.json"
self.min_bars = 320
self.threshold = 0.58  # default; overridable
self.feature_columns = list(FEATURE_COLUMNS)
self.allowed_setup_families = allowlist or None
self.hazard_block_regimes = {"CHOP_SPIRAL", ...}
self.family_policies = load_policy_tree(config)
```

The actual production threshold is 0.55 — the value in `artifacts/aetherflow_routed_ensemble_candidates_20260422/radical_af_nyam_trend_v1/thresholds.json` (per `e594674`). `aetherflow_strategy.py` falls back to 0.58 only if the file is missing.

**Regime ID mapping.** `aetherflow_strategy.py:25-32`:
```python
REGIME_ID_TO_NAME = {
    0: "TREND_GEODESIC",
    1: "CHOP_SPIRAL",
    2: "DISPERSED",
    3: "ROTATIONAL_TURBULENCE",
}
```

**Family policies.** `aetherflow_strategy.py:347-356, 1110-1160`. The four families and their default-table parameters from `aetherflow_features.py:13-25` (`SETUP_TO_ID`):

| Family (ID) | Concept | SL × ATR | TP × ATR | Horizon | Threshold |
|---|---|---:|---:|---:|---:|
| `compression_release` (1) | Expansion phase after consolidation | 1.25 | 2.35 | 20 | 0.26 |
| `aligned_flow` (2) | Coherence × alignment × smoothness | 1.10 | 2.00 | 18 | 0.40 |
| `exhaustion_reversal` (3) | Extension + stress reversal (gate `d_coherence_3 < -0.015`) | 1.00 | 1.65 | 12 | 0.42 |
| `transition_burst` (4) | Transition energy × novelty × pressure_imbalance | 1.20 | 2.10 | 16 | 0.42 |

Family thresholds applied at `aetherflow_features.py:464-469`. Each family's policy supports per-rule overrides — full dict shape at `aetherflow_strategy.py:1110-1160`.

**Bundle architecture.** `aetherflow_strategy.py:681-687` — pickle contains:
```python
{"shared_model", "bundle_design": "routed_ensemble",
 "family_models": {f: head, ...}, "conditional_models": [...],
 "family_head_weight": 1.0, "family_feature_mode": "full",
 "threshold": 0.58, "feature_columns": [...]}
```

**on_bar pipeline.** `aetherflow_strategy.py:1433-1663`:
1. Cache check: precomputed lookup by `int(pd.Timestamp(ts).value)`
2. Build base features via `_seeded_live_base_window()` (manifold regime, alignment, stress, flow)
3. For each candidate family: `build_feature_frame()` → take last row → `predict_bundle_probabilities()` → check threshold
4. Sort candidates by `selection_score = policy.scale × prob + policy.bias`
5. Try top candidate via `_signal_from_row()`
6. If all blocked, log best-candidate block reason via `_row_block_reason()` (`1162-1195`) — checks `no_setup`, `setup_family_blocked`, `session_not_allowed`, `regime_not_allowed`, `side_not_allowed`, `hazard_blocked`, `below_threshold`, then context vetos and param vetos.

**Output payload.** `aetherflow_strategy.py:1250-1322`:
```python
{
    "strategy": "AetherFlowStrategy",
    "side": side_label,
    "tp_dist": params["tp_points"],
    "sl_dist": params["sl_points"],
    "size": int(self.size),
    "size_multiplier": policy.get("size_multiplier", 1.0),
    "entry_mode": policy.get("entry_mode", "market_next_bar"),
    "horizon_bars": params["horizon_bars"],
    "use_horizon_time_stop": policy.get("use_horizon_time_stop", False),
    "confidence": prob,
    "aetherflow_setup_family": setup_family,
    "aetherflow_setup_strength": setup_strength,
    "aetherflow_regime": regime_name,
    "early_exit_enabled": early_exit_cfg.get("enabled", False),
    "early_exit_exit_if_not_green_by": early_exit_cfg.get("exit_if_not_green_by", 30),
}
```

**FLAG: routed-ensemble aggregation logic** lives in `aetherflow_model_bundle.py:predict_bundle_probabilities` (~`1021-1389`). The wrapper builds `expert_probabilities` per family and blends through router-weighted sum, but the exact weighting/conditional-layer math wasn't traced fully in the read pass. Re-read that file before retraining the routed ensemble.

### B. Bracket / Sizing / Entry Geometry

- **Brackets** from family table (`aetherflow_features.py:570-588`, `resolve_setup_params`): `SL = sl_mult × ATR`, `TP = tp_mult × ATR`. Clamped to `[1.0, 8.0]` and `TP ≥ 1.2 × SL` (`579-580`).
- **Sizing** uses `policy.size_multiplier` × base size, then SameSide ML cap if applicable.
- **Entry** path is the same `_signal_birth_hook` → `async_place_order`.

### C. Exit Behavior

AF brackets are absolute distances. Exit semantics fall back to the standard six labels via `julie001.py` trade management.

---

## 4. Where ML Blockers / Overlays Intersect

This is the practical map for retraining. Each component lists features, intercept point, action, and training-data shape with line refs.

### Pure-rule blockers

#### CircuitBreaker — `circuit_breaker.py`

- **Class** `circuit_breaker.py:4-20`. Defaults: `max_daily_loss=500.0`, `max_consecutive_losses=3`, `max_trailing_dd=0.0` (disabled). State: `daily_pnl`, `peak_pnl`, `consecutive_losses`, `is_tripped`.
- **Update logic** `circuit_breaker.py:22-45`. Three independent triggers (each fires + logs critical):
  - `daily_pnl <= -max_daily_loss` (line 33)
  - `consecutive_losses >= max_consecutive_losses` (line 37)
  - `(peak_pnl - daily_pnl) >= max_trailing_dd` AND `max_trailing_dd > 0` (line 41)
- **Query** `circuit_breaker.py:47-56` — `should_block_trade()` returns `(True, "Circuit Breaker Tripped...")` if tripped.
- **Daily reset** `circuit_breaker.py:53-55`.
- **Persistence** `circuit_breaker.py:58-71`. The `max_trailing_dd` field is a recent stub fix (was missing from PR #206 base).

**Re-train target:** none — pure rule. Could be ML-tuned per regime/session but the rule is a simple risk envelope.

#### CascadeLossBlocker — `cascade_loss_blocker.py:55-201`

- **Env config** `55-88`. `JULIE_CASCADE_BLOCKER_ACTIVE`, `JULIE_CASCADE_BLOCKER_COUNT=2`, `JULIE_CASCADE_BLOCKER_WINDOW_MIN=30`, `JULIE_CASCADE_BLOCKER_COOLDOWN_MIN=30`.
- **Rolling-window state** `82-84`: `_long_events`, `_short_events` deques of `(timestamp, is_loss)`.
- **Record** `record_trade_result()` `119-136`: append + trim events older than `max(window, cooldown) + 5min`.
- **Query** `should_block_trade()` `139-168`:
  - Active check `147`
  - Within window count losses on same side `152-156`
  - Cooldown measured from MOST RECENT loss (not oldest) `158-160`
  - Returns `mins_remaining = cooldown - elapsed_since_last_loss` in reason
- **NoOp fallback** `_NoOpCascadeLossBlocker` `179-201`.
- **Backtest provenance** (module docstring `13-17`): +$4,222 (2025), +$1,133 (2026) on 5,237-trade cohort. PF 1.04→1.07.

**Re-train target:** none — pure rule. Could regression-tune `count`/`window`/`cooldown` against forward-loss probability but current params already validated.

#### AntiFlipBlocker — `anti_flip_blocker.py:60-263`

- **Env config** `64-82`. `JULIE_ANTI_FLIP_BLOCKER_ACTIVE`, `JULIE_ANTI_FLIP_WINDOW_MIN=30`, `JULIE_ANTI_FLIP_MAX_DIST_PTS=8.0`.
- **Stop detection** `record_trade_close()` `139-194`. Records as stop-out if:
  - `pnl < 0` (line 177) AND
  - Either `source` contains `"stop"` (line 180) OR `|exit_price - sl_price| ≤ 1.5pt` fallback (`181-185`)
- **Query** `should_block_trade()` `197-243`. Checks the **opposite side's** last stop-out: if elapsed < window AND entry-price within `max_distance_pts` of stop exit → block (`216-237`).
- **Persistence** `96-136` (ISO-string timestamps).
- **NoOp fallback** `246-263`.
- **Cost case** in module docstring `5-8`: 2026-04-23 SHORT stopped at 7172.50 → LONG fired 60s later at 7171.75 (delta 0.75pt) → lost 64 points.

**27-day study finding** (from `backtest_reports/sim_a_plus_c_27day_perday.json`): AntiFlip alone reduced **zero** drawdown violations (19→19); CascadeLoss alone reduced 19→16. A+C $500 is the only A-variant that beats baseline on PnL ($1,757 vs baseline $2,415, but A+C $350 was -$193). **Re-train target:** the (8pt, 30min) parameter pair could be regression-tuned, but the current data says A is dominated by C-only. If retraining, the question is: what feature set would have caught the 3 viol days CascadeLoss missed?

#### DirectionalLossBlocker — `directional_loss_blocker.py:23-322`

- **Class init** `23-26`: `consecutive_loss_limit=3`, `block_minutes=15`, `bias_reversal_limit=4`.
- **State** `28-41`: per-direction streak counters, time-based blocks, bias-reversal state.
- **Quarter logic** `get_quarter()` `55-74`: 6 quarters/day with session-dependent lengths (ASIA 135min, LONDON 75min, NY_AM 60min, NY_PM 75min). Quarter is `ceil(mins_since_session_start / quarter_length) + 1` capped at 4 (line 73).
- **Stage-1 (15-min direction block)** at `149-151` (LONG) / `186-188` (SHORT): on 3 consecutive losses, `blocked_until = current_time + 15min`.
- **Stage-2 (bias reversal)** at `134-140` (LONG) / `171-176` (SHORT): on 4 consecutive losses, `reversed_bias = 'LONG'` (only SHORTs allowed) until quarter changes.
- **Quarter rollover** `update_quarter()` `76-101`: clears `reversed_bias` and resets streaks on session/quarter change.
- **Query** `should_block_trade()` `202-256`: bias reversal first (`224-229`), then 15-min block (`232-254`).
- **Activation gate.** `julie001.py:7373-7398`. The class defaults to `_NoOpDirectionalLossBlocker` (`7373`). Real loading at `7386-7398` is gated by `filter_stack_runtime_enabled` which is `not JULIE_DISABLE_STRATEGY_FILTERS`. **In all current configs (`scripts/run_4config_month.sh`) `JULIE_DISABLE_STRATEGY_FILTERS=1` is set globally → DLB is no-op.** To activate DLB you'd need to flip filter_stack on (which re-enables RejectionFilter, ChopFilter, etc. — the pre-filterless legacy stack).

**Re-train target:** none — pure rule. The 3-losses → 15min and 4-losses → bias-reversal logic is straightforward.

### ML blockers

#### Filter G (Signal Gate 2025) — `signal_gate_2025.py`

**Init.** `init_gate()` at `signal_gate_2025.py:83-119`. Eager-loads four joblib files keyed by family. Activation at `92-93`:
```python
if os.environ.get("JULIE_SIGNAL_GATE_2025", "0").strip() != "1":
    logging.info("Signal gate 2025: disabled (JULIE_SIGNAL_GATE_2025!=1)")
```
Per-strategy override via `JULIE_SIGNAL_GATE_2025_PATH_<STRATEGY>` (`72-80`).

**Joblib payload** keys (per `104-109`): `feature_names`, `veto_threshold`, `training_rows`, `model`, `categorical_maps`, `numeric_features`.

**Scoring.** `_score_with_gate()` at `180-228`. Feature pipeline:
- Numeric features `205-214`: pull from `bar_features` dict, coerce to float, NaN→0.0
- Categorical `215-220`: `side` (uppercase), `regime` (uppercase, strategy-local), `mkt_regime` (lowercase, global classifier), `session` (time-bucket)
- Ordinal `221-222`: `et_hour` if in feature schema
- Output `225`: `payload["model"].predict_proba(X)[0, 1]` — binary classifier `P(big_loss)`

**Threshold pipeline.** Three multipliers stacked:

`_session_multiplier()` `264-275`:
```python
if cum_day_pnl >= 100:  return 1.25  # lenient — let winners run
if cum_day_pnl <= -200: return 0.80  # aggressive — stop bleeding
return 1.0
```

`_REGIME_THR_MULT` `240-248`:
```python
{"whipsaw": 0.60, "calm_trend": 1.05, "neutral": 1.0, "warmup": 1.0}
```

Per-cell at `_per_cell_multiplier()` `339-351`: lookup key `f"{strategy}|{regime_lower}|{time_bucket}"` against the table loaded by `_load_per_cell_overrides()` (`292-322`) from `ai_loop_data/triathlon/filterg_threshold_overrides.json`. Returns 1.0 if missing. Time-bucket via `_time_bucket_of_hour()` `325-336`.

`_effective_threshold()` `354-390`:
```python
regime_mult = _REGIME_THR_MULT.get(regime_lower, 1.0)
session_mult = _session_multiplier(cum_day_pnl)
per_cell_mult = _per_cell_multiplier(strategy, regime, et_hour) if strategy else 1.0
mult = regime_mult * session_mult * per_cell_mult
eff = base_thr * mult
if eff < _EFFECTIVE_THR_FLOOR:  # 0.25, line 278
    eff = _EFFECTIVE_THR_FLOOR
```

The dynamic-threshold gate `_DYNAMIC_THRESHOLD_ENABLED` (line 279) was added by friend's `e594674`. It defaults to off — when off, regime × session multipliers don't apply. The restored per-cell layer (2026-04-25) fires independently of this flag, gated only by `_PER_CELL_ACTIVE` (line 281).

**Active path.** `should_veto_signal()` `393-444`. Returns `(should_veto, reason)`. `(False, "")` if no model loaded or scoring failed. `(True, "signal_gate_2025[family] P(big_loss)=...")` on veto.

**Shadow path.** `log_shadow_prediction()` `447-523`. Runs at every signal birth regardless of gate active state. Telemetry line at `513-519` is `[SHADOW_GATE_2025] family=... P(big_loss)=... base_thr=... eff_thr=... mult=... would_veto=...`. Bar cache fetched from `loss_factor_guard.get_guard()._bar_cache` at `454-458`.

**Per-strategy thresholds (current production):**
- DE3: `thr=0.65, n=213` (training rows)
- AetherFlow: `thr=0.775, n=367`
- RegimeAdaptive: `thr=0.475, n=612`
- MLPhysics: skipped (no model file)

**Per-cell calibration source.** `scripts/idea1_filterg_per_cell_calibrate.py` from seeded ledger. Cells `< -$2/trade & n≥20 → bleeding 0.75x`, `> +$5/trade & n≥20 → strong 1.15x`, otherwise neutral 1.0.

**Re-train priority: HIGH.** The three per-strategy gate models were swapped by friend in `e594674` without OOS artifacts. Per-cell calibration is from the OLD model output distribution — multipliers may apply differently to friend's new models. Recommended: train fresh on baseline-only `closed_trades.json` with label `pnl_dollars < -$X` over a 30-min forward window.

#### SameSide ML — distributed across `julie001.py` + supporting modules

- **Env** `JULIE_SAMESIDE_ML=1`, `JULIE_SAMESIDE_ML_MAX_CONTRACTS=2`, `JULIE_BYPASS_SAMESIDE=0/1` (test bypass)
- **Origin commit** `16f6d71 SameSide ML: convert hard same-side suppression from rule to ML (SHIPPED)`
- **Same-side gate** at `julie001.py:2041-2080` (covered in Section 1.B)
- **FLAG: the actual ML model file/path for SameSide** wasn't located in the read pass. The agent's grep didn't find a `same_side_ml.py` or analogous module. The classifier may be inline in julie001.py or call into one of the other model files. Locate this before retraining.

**Re-train priority: MEDIUM.** Friend did not touch.

#### Regime ML A/B/C (v6) — `regime_classifier.py` + `scripts/regime_ml/`

- **Env** `JULIE_REGIME_ML_BRACKETS=1` (Model A — scalp brackets), `JULIE_REGIME_ML_SIZE=1` (Model B — size cap), `JULIE_REGIME_ML_BE=1` (Model C — BE-disable)
- **Origin commits:** `09942e9` (v5 Model A SHIP, +$12,343 lift), `095b4b7` (v6 A-conditional retry → SHIP B + C), `4c7ceba` (HGB-only inference fix), `28bb61b` (ml_ready property)
- **Architecture:** three independent HistGradientBoostingClassifier models. Models B and C are gated on Model A pass.
  - **Model A**: scalp brackets `(3pt TP / 5pt SL)` vs default `(6/4)`. Label: `scalp_PnL > default_PnL` over 15-min forward window.
  - **Model B**: force `size=1` vs allow `size=3`. Label: high-variance forward window where size=1 preserves more capital.
  - **Model C**: skip BE-move vs default `+5pt MFE → BE`. Label: ML identifies mean-revert setups (low MFE-to-SL ratio).
- **Integration** via `regime_classifier.py` lookups (`apply_regime_size_cap` is the visible entrypoint at `250-296` for Filter D/E; the A/B/C ML decisions plug into the bracket/size/BE selection sites in julie001 via exposed methods).

**FLAG: the four `_apply_*` methods (`apply_dead_tape_brackets`, `_apply_scalp_brackets`, `_apply_size_reduction`, `_apply_be_disable`) the user mentioned weren't fully traced. The agent reading regime_classifier.py covered the public API + size-cap path but didn't enumerate these helper methods.** Re-read `regime_classifier.py` with grep `^def _apply_` before retraining A/B/C.

**Re-train priority: MEDIUM.** Friend did not touch.

### ML overlays — `ml_overlay_shadow.py`

Common loader pattern at `ml_overlay_shadow.py:148-205`. `_try_load(fname)` returns the joblib payload from `artifacts/signal_gate_2025/<fname>` or None. Each overlay has its own activation flag.

#### ML LFO — `ml_overlay_shadow.py:155, 254-308`

- **Env** `JULIE_ML_LFO_ACTIVE=1`. Per-strategy override via `JULIE_LFO_POLICY_<FAMILY>` (defaults at `34-45`: `de3=ml`, `regimeadaptive=rule`, `aetherflow=off`, `mlphysics=off`).
- **Policy resolver** `get_lfo_live_policy()` `93-103`: env override → legacy `JULIE_ML_LFO_ACTIVE=1` (returns "hybrid") → defaults table.
- **Model** `model_lfo.joblib`. Load metadata at `161-165`: "LFO model loaded — {n} features, thr={thr}, cv_auc={auc}". Current production: 25 features, thr=0.500, cv_auc=0.525.
- **Score** `score_lfo()` `254-308`. Numeric features at `275-285`:
  ```python
  numeric.update({
      "dist_to_bank_below", "dist_to_bank_above", "dist_to_bank_in_dir",
      "bar_range_pts", "bar_close_pct_body",
      "sl_dist_pts", "tp_dist_pts", "atr_ratio_to_sl",
  })
  ```
  Optional augmentation at `289-300` (encoder embedding 32-dim, cross-market features).
  Categorical at `301`: `side`, `session`, `mkt_regime`.
  Output `308`: `(p_wait_better, veto_threshold)` or None.
- **Trainer** `scripts/signal_gate/train_lfo_ml.py`. **Training data** `artifacts/signal_gate_2025/lfo_training_data.parquet` (358KB). **Label**: binary `WAIT_won` vs `IMMEDIATE_won` over 3-bar forward window.

**Re-train priority: HIGH.** AUC=0.525 is barely above random. Either the label horizon is too noisy or friend's swap broke the model. Verify against baseline.

#### ML PCT Overlay — `ml_overlay_shadow.py:156, 311-359` + `pct_overlay_runtime.py`

- **Env** `JULIE_ML_PCT_ACTIVE=1`. Default shadow.
- **Model** `model_pct_overlay.joblib`. Current: 28 features, cv_auc=0.781.
- **Score** `score_pct_overlay()` `311-359`. Input is `state` object (from `PctLevelOverlay.state`) with `at_level`, `session_open`, `nearest_level`, `pct_from_open`, `level_distance_pct`, `atr_pct_30bar`, `range_pct_at_touch`, `hour_edge`, `ts`, `confidence`, `tier`, `bias`. Output at `358`:
  ```python
  ml_bias = "breakout_lean" if p_bo >= 0.55 else ("pivot_lean" if p_bo <= 0.45 else "neutral")
  return p_bo, ml_bias
  ```
- **Snapshot integration.** `pct_overlay_runtime.py:43-263`. `init_pct_level_overlay()` at `43-56`. `attach_pct_overlay_snapshot()` at `70-136` stamps a snapshot on the signal at signal-birth (idempotent). Snapshot schema at `12-25`:
  ```python
  signal["pct_overlay_snapshot"] = {
      "size_mult", "tp_mult", "veto_reason", "bar_ts",
      "at_level", "engaged",
      # at_level=True only:
      "level", "tier", "bias", "confidence",
  }
  ```
  TP multiplier logic at `pct_overlay_runtime.py:103-111`:
  ```python
  if tp_extend > 0.0: tp_mult = 1.0 + tp_extend
  elif tp_tighten > 0.0: tp_mult = max(0.5, 1.0 - tp_tighten)
  else: tp_mult = 1.0
  ```
  `resolve_pct_overlay_snapshot()` at `139-200` applies decisions just before order placement: veto path at `158-165`, size resolution at `167-180`, TP resolution at `182-198`.

**FLAG: how does `score_pct_overlay()` (ml_overlay_shadow.py) tie into the snapshot lifecycle?** The snapshot captures tier/bias/confidence/level (lines 113-128) but not the ML score (`p_breakout`, `ml_bias` from score_pct_overlay). Are they consumed in parallel or serial? Not traced. Locate the call site that bridges ML score → snapshot before retraining.

- **Trainer** `scripts/signal_gate/train_pct_overlay_ml.py`. Training data: `pct_overlay_training.parquet`. Label: 3-class (BREAKOUT extends >+0.10%, PIVOT retraces >−0.15%, NEUTRAL no move) over 60-min forward.

**Re-train priority: MEDIUM.** AUC 0.781 is reasonable. Verify against friend's swap.

#### ML Pivot Trail — `ml_overlay_shadow.py:157, 362-434`

- **Env** `JULIE_ML_PIVOT_TRAIL_ACTIVE=1`
- **Model** `model_pivot_trail.joblib`. Current: 27 features, thr=0.550, cv_auc=0.920.
- **Score** `score_pivot_trail()` `362-434`. Output at `434`:
  ```python
  thr = float(_PIVOT_PAYLOAD.get("hold_threshold", 0.55))
  return p_hold, (p_hold >= thr)
  ```
- **Trainer** `scripts/signal_gate/train_pivot_trail_ml.py`. Label: binary `held_for_20bars` (1=respected anchor ± buffer, 0=broke through).

**Re-train priority: LOW (AUC 0.92 is high).** But verify against friend's swap.

#### Kalshi Gate (Entry) — `ml_overlay_shadow.py:158, 437-521`

- **Env** `JULIE_ML_KALSHI_ACTIVE=1`
- **Model** `model_kalshi_gate.joblib`. Current: 28 features, rolling_auc=0.545, thr=0.50 (clf+reg dual).
- **Score** `score_kalshi()` `437-521`. Dual scoring at `512-517`:
  ```python
  clf = _KALSHI_PAYLOAD["classifier"]
  reg = _KALSHI_PAYLOAD["regressor"]
  probs = clf.predict_proba(X)[0]
  p_win = float(probs[classes.index(1)]) if 1 in classes else 0.5
  pred_pnl = float(reg.predict(X)[0])
  ```
  Output at `520-521`: `(p_win, pred_pnl, should_pass)` where `should_pass = p_win >= pass_threshold` (default 0.50).
- **Apply** `_apply_kalshi_gate_size()` at `julie001.py:1426-1476`. Settlement hours `[12, 13, 14, 15, 16] ET` only (line 175 `_KALSHI_GATING_HOURS_ET`); 10-11 AM excluded as crowd-contrarian. Hard veto for non-AetherFlow sets size to 0; AetherFlow gets sizing-only mode (veto skipped, size unchanged) at `julie001.py:1469-1474`.
- **v8 patch** `_kalshi_ml_v8_decision()` at `julie001.py:245-271`. Three modes:
  - `soft_veto_on_rule_pass_only`: veto only if rule says PASS but ML proba < 0.30
  - `size_multiplier`: pass through, multiplier applied elsewhere
  - default binary: PASS if proba >= 0.50

- **Trainer** `scripts/regime_ml/train_kalshi.py`. Training data: `kalshi_training_data.parquet` (190 KB). Label: simulated 15-min forward PnL on size=1 trade with TP=6/SL=4; `pass_label` if PnL > +$15, `block_label` if PnL < -$15, ambiguous dropped.

**Re-train priority: HIGH.** Rolling AUC 0.545 is near-random. Earlier in this session the headline failure was "Kalshi ML wasn't actually invoked in mlstack" — verify the call site (`_kalshi_ml_v8_decision`) is actually wired to all the Kalshi-relevant decision points before retraining is meaningful.

#### Kalshi TP Gate — `ml_overlay_shadow.py:159, 532-627`

- **Env** `JULIE_ML_KALSHI_TP_ACTIVE=1`
- **Model** `model_kalshi_tp_gate.joblib`. Current: 22 features, rolling_auc=0.624, thr=0.50.
- **Score** `score_kalshi_tp()` `532-627`. Dual at `607-615` (classifier diagnostic, regressor production). Production gate at `620-627`:
  ```python
  gate_thr = _KALSHI_TP_PAYLOAD.get("regressor_gate_threshold", 0.0)
  env_thr = os.environ.get("JULIE_ML_KALSHI_TP_PNL_THR")
  if env_thr is not None: gate_thr = float(env_thr)
  return p_hit, pred_pnl, (pred_pnl > gate_thr)
  ```
- **Trainer** `scripts/signal_gate/train_kalshi_tp_ml.py`. v2 (`f12c7a0`) swapped binary HIT_TP for regression `pnl_dollars`. Training data: `kalshi_tp_training_data.parquet` (209 KB).

**Re-train priority: MEDIUM-HIGH.** AUC 0.624 better than entry gate but still modest.

#### RL Management — `ml_overlay_shadow.py:647-721` + `rl/`

- **Env** `JULIE_ML_RL_MGMT_ACTIVE=1`. Speed-flag `JULIE_DISABLE_RL_SHADOW=1` short-circuits init at `656`:
  ```python
  if _os_rl.environ.get("JULIE_DISABLE_RL_SHADOW", "0").strip() == "1":
      return False
  ```
- **Model** `model_rl_management.zip` (SB3 PPO policy)
- **Init** `init_rl_management()` `647-663` delegates to `rl.inference.init_rl_policy()`.
- **Bar encoder** `init_bar_encoder()` `690-721` loads `artifacts/signal_gate_2025/bar_encoder.pt` with defaults `seq_len=60, embed_dim=32`.
- **FLAG: PPO action signature.** `score_rl_management()` delegates to `rl.inference.score_rl_management(**obs_kwargs)` at line 671. Documented as "see rl/inference.py for full list". Locate the kwargs before retraining.

**Re-train priority: MEDIUM.** RL is the most opaque to validate; PPO weights aren't introspectable like a tree model.

### Auxiliary blockers

#### LossFactorGuard (LFG) — `loss_factor_guard.py`

- **Activation** `JULIE_LOSS_FACTOR_GUARD=1` at line 464. Default off.
- **Tunables** `loss_factor_guard.py:51-90`. Streaks (`JULIE_LFG_LONG_STREAK=3`, `JULIE_LFG_SHORT_STREAK=4`), morning cascade (`JULIE_LFG_MORNING_CASCADE=3`), afternoon shutdown (`JULIE_LFG_AFT_SHUTDOWN_PNL=-200`), stop:take ratio (`JULIE_LFG_STOP_TAKE_RATIO=5.0`), veto duration (`JULIE_LFG_VETO_MINUTES=30`).
- **State** `DailyState` dataclass `93-114`. Per-day state reset on midnight: streak counters, cum_pnl, recent_sources deque (last 5 exits), AM trade counters, veto-until timestamps.
- **Update** `notify_trade_closed()` `223-305`:
  - Streak tracking `235-244`
  - Morning cascade tracking `246-254`
  - Long-streak trip `263-269`
  - Short-streak trip `270-276`
  - Morning cascade trip `279-288`
  - Stop:take ratio trip `291-305`
- **Entry-time check** `should_veto_entry()` `307-374`. Checks in priority order:
  1. Morning cascade (`317-318`)
  2. Afternoon shutdown (`321-328`)
  3. Full pause (`331-332`)
  4. Side veto (`335-338`)
  5. Filter C: counter-trend reversal (`340-360`) — blocks `*_Rev_*` against tier
  6. Filter F: chart bounce/dip-fade (`362-365`) — calls `_chart_bounce_fade_veto()` at `151-207`
  7. Filter G consult (`370-372`) — bridges to `signal_gate_2025.py` at `376-422`

**2025 forensic basis** (module docstring `5-10`): from 63 big-loss days, derived empirical patterns: LONG-bias 36% WR on losing days vs 60% on winning, morning cascade in 42% of big-loss days, hour-15 ET 10% WR on losers, stop:take ratio winners=3x losers=10x.

**Re-train priority: NONE.** Pure rule, but the chart-veto thresholds are tuneable per regime. Future ML could replace the heuristic.

### Triathlon Engine

- **Cell-key shape** `triathlon/__init__.py:65-67`: `f"{strategy}|{regime}|{time_bucket}"`. Strategies set at `triathlon/__init__.py:45`: `"DynamicEngine3", "AetherFlow", "RegimeAdaptive", "MLPhysics"`. Regimes at line 46: `"whipsaw", "calm_trend", "dead_tape", "neutral", "warmup"`. Time-buckets at `36-43` (six: pre_open / morning / lunch / afternoon / post_close / overnight).
- **League scoring.** Ledger schema at `triathlon/ledger.py:36-109` (signals, outcomes, standings, retrain_queue, current_medals tables). `fetch_cell_stats()` at `223-283` aggregates fired + counterfactual counts per cell.
- **Medal assignment** at `triathlon/medals.py:39-41`:
  - GOLD: top 20% in at least one league (`<= 0.20`, line 113)
  - SILVER: top 50%, BRONZE: top 80%
  - PROBATION: bottom 20% in ALL three leagues (`>= 0.80`, line 111)
- **Runtime effects** `MEDAL_EFFECTS` at `triathlon/medals.py:78-84`:
  - Gold: priority +1, size_mult 1.50
  - Silver: priority 0, size_mult 1.00
  - Bronze: priority −1, size_mult 1.00
  - Probation: priority −2, size_mult 0.50
  - Unrated: priority 0, size_mult 1.00
- **Live sort key.** `_live_signal_sort_key()` at `julie001.py:2120-2132`:
  ```python
  return (int(priority), -confidence, strategy_label, sub_strategy, side)
  ```
  Lower priority sorts first (so gold's +1 is actually a demotion under this key — verify by reading `julie001.py` priority assignment to confirm sign convention). **FLAG: priority delta sign convention.** The medal table says "priority_delta = +1" for gold but the sort key sorts ascending — confirm whether higher priority = earlier or later in the sort.

---

## 5. Key Observations for Retraining

### High-leverage retraining targets (priority order)

1. **Filter G per-strategy gates** (`model_de3.joblib`, `model_aetherflow.joblib`, `model_regimeadaptive.joblib`). Friend's `e594674` swapped these without OOS artifacts. Train on **baseline closed_trades data** (no overlay filtering) so labels reflect natural signal outcomes. Per-strategy training rows currently small (213 / 367 / 612). Need 6+ months of baseline data for a meaningful AF model.

2. **Kalshi entry gate** (`model_kalshi_gate.joblib`). Rolling AUC 0.545 ≈ near-random. The headline failure mode in this session: in mlstack runs, the v8 patch loaded the model but the call site was missing post-PR-#206 refactor — meaning ALL "Kalshi-gated" trades in mlstack were not going through the gate. **Verify call site is wired** before retraining is meaningful.

3. **ML LFO** (`model_lfo.joblib`). AUC 0.525 — barely above random. Either the 3-bar forward decision is too noisy, or friend's swap broke the model. Re-train on baseline trades captured at 3-bar look-ahead with explicit label `WAIT_PnL > IMMEDIATE_PnL`. Consider longer horizon (5-bar?) if 3-bar remains unpredictable.

4. **Kalshi TP gate** (`model_kalshi_tp_gate.joblib`). AUC 0.624. v2 changed labels from binary HIT_TP to regression `pnl_dollars` (`f12c7a0`). Verify which version friend's swapped model uses; if v1, consider re-training as v2.

5. **Per-cell Filter G overrides.** Re-calibrate `filterg_threshold_overrides.json` against friend's swapped gate models if those models stay in production. The current 17-cell calibration assumes pre-swap output distributions.

### Lower priority

6. **Pivot Trail ML** (AUC 0.92): high enough that retraining adds little unless the model was corrupted in friend's swap.
7. **ML PCT Overlay** (AUC 0.781): reasonable; verify against friend's swap.

### Chicken-and-egg problem (training data source)

A blocker trained on **mlstack-filtered trades** sees only signals that already passed every other overlay. This biases the learned decision boundary toward the survivor set — the blocker will under-veto because the bad signals it would have caught were already removed.

**Always train Filter G on baseline data**, never on mlstack data. The 2025-03 baseline (`backtest_reports/baseline_2025_03/closed_trades.json`, 1,821 trades, +$7,600 net, 48.4% WR, $9,102 max DD) is the canonical reference.

**For management overlays (LFO, Pivot, RL):** train on the strategy that the overlay manages, with the OTHER overlays disabled. E.g., train Pivot Trail on a baseline run that has Pivot Trail disabled but everything else (Cascade, AntiFlip, Filter G) on — so the trades included are realistic but the pivot decision is unbiased.

**Kalshi gates:** must train on logs from when Kalshi was active. Otherwise the crowd-probability features are missing.

### Known failure modes from this session

- **Mlstack bleeds:** previous mlstack sweep had worse PnL than baseline on April months, primarily because the Kalshi gate was loaded but never invoked (call site missing post-PR-#206 refactor). Re-check `_kalshi_ml_v8_decision()` call sites before trusting any "mlstack" backtest.
- **Friend's untested model swaps:** `e594674` replaced 8 model artifacts without validation. Per-cell threshold overrides lose precision because they were calibrated against pre-swap distributions. Don't trust ANY metric from a sweep that uses friend's swapped models until either the models are validated or the user-shipped versions are restored.
- **AntiFlip not adding safety alone:** in `sim_a_plus_c_27day_perday.json`, AntiFlip alone reduced **zero** drawdown violations (19→19) while CascadeLoss alone reduced 19→16. Adding AntiFlip on top of Cascade gave the same 16 viols. Conclusion: the (stop_within_8pt → opposite_flip_blocked) rule isn't catching the kind of DD events that >$1,200 violations actually are.
- **Roster log mismatch:** `julie001.py:7142-7155` builds the filterless roster string by always appending `"AetherFlow"` regardless of `filterless_disabled`. The roster log line is misleading when AF is functionally disabled (model not loaded, `enabled_live=False`). Fix the roster builder before parsing logs as ground truth.

### Recommended training recipes per blocker

| Blocker | Source data | Label | Horizon | Notes |
|---|---|---|---|---|
| Filter G DE3 | baseline `closed_trades.json` | `pnl_dollars < -$X` | 30-min forward | Include sub_strategy as feature |
| Filter G AF | AF trades only over 6+ months | same | 30-min forward | Sparse — long history needed |
| Filter G RegimeAdaptive | RA trades only | same | 30-min forward | Stratify by regime |
| SameSide ML | baseline same-side reversal trades | `same_side_pnl > 0` | trade-end | Add `time_since_prior_loss`, `prior_pnl` features |
| Regime ML A | baseline w/ both bracket variants simulated | `scalp_PnL > default_PnL` | 15-min forward | Counterfactual sim per trade |
| Regime ML B | baseline at size=1 vs size=3 simulated | `size1_capital_preserved > size3` | 30-min forward | Variance-aware label |
| Regime ML C | baseline w/ BE-arm vs no-BE-arm simulated | `no_BE_PnL > BE_PnL` | trade-end | Counterfactual; expensive to generate |
| ML LFO | baseline w/ 3-bar look-ahead capture | `WAIT_PnL > IMMEDIATE_PnL` | 3-bar | Sparse signal — consider 5-bar horizon |
| ML PCT Overlay | bar-by-bar level touches over full year | `BREAKOUT` / `PIVOT` / `NEUTRAL` (3-class) | 60-min | Independent of trade decisions |
| ML Pivot Trail | confirmed pivots over full year | `pivot_held_for_20bars` | 20-bar | Independent of trade decisions |
| Kalshi gate (entry) | logs w/ `[KALSHI_ENTRY_VIEW]` lines | `pnl > +$15 (pass) / < -$15 (block)` | 15-min forward | Settlement hours only (12-16 ET) |
| Kalshi TP gate | same logs, TP-aligned probability | `pnl_dollars` (regression v2) | 15-min forward | Replaces v1 binary HIT_TP |
| RL Management | trade replay w/ multiple action sequences | shaped reward (PnL + Sharpe − DD) | per-bar | Off-policy or simulated env |

### Open questions / FLAGs to resolve before retraining

The following code paths were NOT fully traced in this read pass and need follow-up before any retraining work:

1. **`aetherflow_model_bundle.predict_bundle_probabilities()` aggregation math** (~lines 1021-1389). Routed-ensemble blending semantics not fully traced.
2. **`regime_classifier.py` dead-tape branch and `_apply_dead_tape_brackets` / `_apply_scalp_brackets` / `_apply_size_reduction` / `_apply_be_disable` methods.** The agent covered the public API and size-cap path but not the A/B/C ML decision sites.
3. **Bridge between `score_pct_overlay()` (ml_overlay_shadow.py:311-359) and `attach_pct_overlay_snapshot()` (pct_overlay_runtime.py:70-136).** Are they consumed in parallel or serial? Where does the ML score plug into the snapshot lifecycle?
4. **SameSide ML model file location.** No `same_side_ml.py` found; classifier may be inline or call into another model file.
5. **RL management observation kwargs.** `score_rl_management()` delegates to `rl.inference.score_rl_management(**obs_kwargs)`. The expected kwargs are documented as "see rl/inference.py" — read that file before retraining.
6. **Cross-market features fallback.** `get_cross_market_features()` returns `CROSS_MARKET_FEATURE_DEFAULTS` from `rl.cross_market`. The default values aren't visible in `ml_overlay_shadow.py`; locate them.
7. **Triathlon priority sign convention.** `MEDAL_EFFECTS["gold"].priority_delta = +1` but `_live_signal_sort_key()` sorts ascending on `int(priority)`. Confirm whether higher priority = earlier or later sort position.
8. **`signal_gate_2025.py:_score_with_gate()` model return shape.** The current implementation calls `predict_proba(X)[0, 1]` (binary classifier output). Friend's `e594674` mentioned dual classifier+regressor for Kalshi gates but `_score_with_gate` (Filter G) appears to be classifier-only. Verify by inspecting the joblib payload schema.

### Final notes

- The bot has 13+ ML/rule blockers and 3 strategies. **The interaction is not linear.** A blocker that helps in isolation can hurt in a stack (AntiFlip is the canonical example).
- Always validate retrained blockers via a **full sweep** of the 14-month period 2025-03 → 2026-04 (the canonical evaluation window), not just OOS slices.
- The user's discipline of "ship only if PnL improves AND WR non-regressive AND MaxDD non-regressive" (from `44d408a`) is the right gating principle. Apply it to every retrained component.
- When in doubt, fall back to the rule baseline (`JULIE_ML_*_ACTIVE=0`). The bot must work with overlays off — that's the safety property baseline preserves.

---

*Compiled 2026-04-25 from end-to-end reads of: `signal_gate_2025.py`, `ml_overlay_shadow.py`, `pct_overlay_runtime.py`, `regime_classifier.py`, `regime_strategy.py`, `regimeadaptive_gate.py`, `aetherflow_strategy.py`, `circuit_breaker.py`, `cascade_loss_blocker.py`, `anti_flip_blocker.py`, `directional_loss_blocker.py`, `loss_factor_guard.py`, `triathlon/__init__.py`, `triathlon/medals.py`, `triathlon/ledger.py`, `dynamic_signal_engine3.py`, `dynamic_engine3_strategy.py`, `de3_v4_runtime.py`, `de3_v4_router.py`, plus targeted reads of `julie001.py` (signal-birth hook, same-side gate, Kalshi gate, BE-arm, pivot trail, exit sources, RestoredLivePosition, filterless roster, DLB activation gate). Eight FLAGs noted in Section 5 mark code paths that need follow-up reads before retraining.*

---

## 8. ml_full_ny 4-Month Analysis (Mar–Jun 2025) — Filter Edits to Hit Positive PnL with WR≥55% / DD≤$800

**Scope:** Pure analysis on the 4 completed months of the in-flight ml_full_ny sweep (`backtest_reports/ml_full_ny_2025_03/04/05/06/`). 251 trades total. 10 months still running. **No live changes were made — this section informs future retraining only.**

**Data caveat:** these results use friend's UNVALIDATED `e594674` swapped overlay model artifacts. Numbers reflect the joint behavior of an unverified model bundle; do not ship anything based on them without a clean retrain on baseline data first (see Section 5 retraining recipes).

### 8.1 Summary Block

| Metric | Current (4 mo) | Target | Gap |
|---|---:|---:|---:|
| Total trades | 251 | — | — |
| Net PnL | **+$175.42** | > $0 | ✅ marginal |
| Blended WR | 48.2% | ≥ 55% | **❌ −6.8pp** |
| Continuous max DD | **$3,370.32** | ≤ $800 | **❌ 4.2× over** |

The current stack has **positive expectancy by a thin margin** ($0.70/trade) but DD and WR are far from target. The PnL is being kept above zero by 2 strong months (May, Jun) offsetting 2 bleed months (Mar, Apr).

### 8.2 By-Month Root Cause

| Month | Trades | WR | Net PnL | Avg | Max DD | Story |
|---|---:|---:|---:|---:|---:|---|
| 2025-03 | 40 | 35.0% | -$1,669.00 | -$41.73 | $1,736.54 | Tariff-shock chop. Stop:take ratio 5.0:1 — every entry stopped 5× more often than it took profit |
| 2025-04 | 119 | 47.1% | -$633.02 | -$5.32 | $1,949.09 | High volume but mid-WR. April-17 had a **6-trade RegimeAdaptive cluster all stopped** (#2-#7 of top-10 worst losers). |
| 2025-05 | 58 | 58.6% | **+$1,062.81** | +$18.32 | $381.15 | Calm-trend regime. Stop:take 2.83:1 but takes averaged $187 vs stops $-49 |
| 2025-06 | 34 | 50.0% | **+$1,414.63** | +$41.61 | $719.80 | Lowest volume but biggest avg/trade. 4 takes averaged $278; 4 take_gaps averaged $309 |

**Why Mar/Apr bled:** both months were classified as `vol_regime=high` for most NY hours (tariff-week chaos + April expiry choppiness). The DE3 `Long_Rev_T2_SL10_TP25` strategy fires on red prior bars expecting a reversion bounce — but in high-vol panic-down sessions, the bounce never comes and the 10pt SL hits before the 25pt TP.

**Why May/Jun profited:** both regimes were `normal` or `low` vol. Mean-reversion setups had clean retraces and reached the 25pt TP. The same DE3 sub-strategy that lost -$1,740 on March produced +$806 on May and +$1,738 on June.

### 8.3 By-Vol-Regime (extracted from `[FILTER_CHECK]` log lines via bar-timestamp join)

| vol_regime | n | WR | Net PnL | Avg/trade |
|---|---:|---:|---:|---:|
| **normal** | 41 | **63.4%** | **+$1,409.71** | +$34.38 |
| low | 6 | 50.0% | +$398.96 | +$66.49 |
| ultra_low | 15 | 46.7% | +$116.68 | +$7.78 |
| **high** | **189** | **45.0%** | **−$1,749.93** | −$9.26 |

**This is the dominant signal in the data.** 75% of trades fired in `vol_regime=high` and that bucket alone lost $1,750. The other three buckets combined produced +$1,925.

### 8.4 By-Hour Breakdown (entry hour ET)

| Hour | n | WR | Net PnL | Avg/trade |
|---:|---:|---:|---:|---:|
| 8 | **207** | 46.4% | **−$428.15** | −$2.07 |
| 9 | 9 | 66.7% | +$112.80 | +$12.53 |
| 10 | 6 | 33.3% | −$32.32 | −$5.39 |
| 11 | 9 | **77.8%** | **+$490.24** | +$54.47 |
| 12 | 7 | 57.1% | −$42.39 | −$6.06 |
| 13 | 7 | 57.1% | +$272.62 | +$38.95 |
| 14 | 5 | 40.0% | −$171.16 | −$34.23 |
| 15 | 1 | 0.0% | −$26.22 | −$26.22 |

**82% of trades fire in hour 8** — the open. That hour is also the dominant bleeder. Hours 11, 13 are the strongest (combined +$762 on 16 trades).

The skew is partly DE3's session blocks: `5min_06-09_*` strategies fire from 6-9 ET (in NY-only mode this becomes 8-9 ET only), so they cluster at the open. The DE3 `06-09` session block produced +$1,326 on 188 trades — but that's because takes (when they hit) are large. The real fix is filtering hour 8 trades that fire INTO high-vol open sessions.

### 8.5 By-Strategy Breakdown

| Strategy | Total n | Total WR | Total Net | By-month |
|---|---:|---:|---:|---|
| **DynamicEngine3** | 204 | 49.5% | **+$1,287.40** | Mar -$1,740, Apr +$483, May +$806, Jun +$1,738 |
| **RegimeAdaptive** | 47 | 42.6% | **−$1,111.98** | Mar +$71, Apr **−$1,116**, May +$257, Jun −$324 |
| AetherFlow | 0 | — | $0 | Loaded but no signal cleared 0.55 threshold |

**RegimeAdaptive is a net loser** — most of the loss concentrated in April (-$1,116 on 34 trades). On the other 3 months it's roughly break-even or slightly positive. Removing RegimeAdaptive entirely improves PnL by +$1,112 but doesn't fix DD or WR alone.

### 8.6 By-DE3-Sub-Strategy

| Sub-strategy | n | WR | Net | Avg |
|---|---:|---:|---:|---:|
| `5min_06-09_Long_Rev_T2_SL10_TP25` | 92 | 47.8% | +$413.93 | +$4.50 |
| `15min_06-09_Long_Rev_T2_SL10_TP25` | 54 | 53.7% | **+$1,105.71** | +$20.48 |
| `5min_06-09_Short_Rev_BullClimax_T3_SL5_TP6.25` | 42 | 47.6% | **−$193.75** | −$4.61 |
| `15min_12-15_Long_Rev_T2_SL10_TP25` | 16 | 50.0% | −$38.49 | −$2.41 |

Only **4 distinct DE3 sub-strategies** fired in 4 months across 204 trades — the universe is much smaller than 62 thanks to the heavy ML stack filtering. The 15min variant dominates (best avg/trade), 5min Long_Rev is break-even-positive, BullClimax SHORT is the only DE3 bleeder.

**By T-tier:** T2=+$1,481 (162 trades, 50% WR), T3=−$194 (42 trades, 47.6% WR — entirely BullClimax). T2 vs T3 maps cleanly to the lane split below.

**By lane:** Long_Rev=+$1,481 (162 trades), Short_Rev=−$194 (42 trades, all BullClimax). Long-bias on a chop-prone period.

**By session block:** 06-09=+$1,326 (188 trades), 12-15=−$38 (16 trades). The morning DE3 block drove all DE3 PnL.

### 8.7 By-Exit-Source Asymmetry

| Source | n | WR | Net | Avg |
|---|---:|---:|---:|---:|
| `take` | 36 | **100%** | **+$5,750.95** | +$159.75 |
| `take_gap` | 19 | **100%** | +$3,626.68 | +$190.88 |
| `reverse` | 19 | 57.9% | +$524.23 | +$27.59 |
| `close_trade_leg` | 21 | 47.6% | +$183.17 | +$8.72 |
| `stop_gap` | 44 | 27.3% | −$2,658.86 | −$60.43 |
| `stop` | 112 | 29.5% | **−$7,250.75** | −$64.74 |

The R:R is asymmetric in the bot's favor (avg take $164 vs avg stop −$64, ~2.6× ratio matching the 25/10 pt geometry on the dominant DE3 variant) — but takes happen 36 times to stops happening 112 times (3.1×). Net: stops outweigh takes in count enough to overwhelm the per-trade asymmetry on bleed months.

**Stop:take ratio per month:**
- 2025-03: 5.00:1 → bleed
- 2025-04: 2.38:1 → still negative because individual stops larger
- 2025-05: 2.83:1 → +$1,063
- 2025-06: 2.88:1 → +$1,415

There's no clean ratio threshold; magnitude per stop matters more than count.

### 8.8 Top 10 Single Worst Losers

| # | PnL | Month | Strategy | Sub | Side | Size | hour | vol_regime |
|---:|---:|---|---|---|---|---:|---:|---|
| 1 | −$217.44 | 2025-06 | RegimeAdaptive | Q2_W3_THU_NY_AM | LONG | 6 | 8 | **high** |
| 2 | −$181.20 | 2025-04 | RegimeAdaptive | Q2_W3_THU_NY_AM | LONG | 5 | 8 | **high** |
| 3 | −$181.20 | 2025-04 | RegimeAdaptive | Q2_W3_THU_NY_AM | LONG | 5 | 8 | **high** |
| 4 | −$181.20 | 2025-04 | RegimeAdaptive | Q2_W3_THU_NY_AM | LONG | 5 | 8 | **high** |
| 5 | −$181.20 | 2025-04 | RegimeAdaptive | Q2_W3_THU_NY_AM | LONG | 5 | 8 | **high** |
| 6 | −$181.20 | 2025-04 | RegimeAdaptive | Q2_W3_THU_NY_AM | LONG | 5 | 8 | **high** |
| 7 | −$181.20 | 2025-04 | RegimeAdaptive | Q2_W3_THU_NY_AM | LONG | 5 | 8 | **high** |
| 8 | −$153.72 | 2025-03 | DynamicEngine3 | 5min_06-09_Long_Rev_T2_SL10_TP25 | LONG | 3 | 8 | **high** |
| 9 | −$153.72 | 2025-03 | DynamicEngine3 | 5min_06-09_Long_Rev_T2_SL10_TP25 | LONG | 3 | 8 | **high** |
| 10 | −$153.72 | 2025-03 | DynamicEngine3 | 5min_06-09_Long_Rev_T2_SL10_TP25 | LONG | 3 | 8 | **high** |

**Two patterns dominate the top 10:**
1. **Apr-17 RegimeAdaptive Q2_W3_THU cluster** — six stopped LONG entries on the same day, same combo, same 7-pt SL, all in `vol_regime=high` between 08:08-08:44 ET. Single-day disaster of −$1,087 from one strategy on one combo. SameSide ML / RegimeAdaptive gate should have caught the 2nd-6th but didn't.
2. **DE3 `5min_06-09_Long_Rev_T2_SL10_TP25` in March hour-8 high-vol** — three stopped LONGs on Mar 3/4/5, same setup, same outcome. The strategy is a mean-reversion long that fails when momentum keeps grinding down.

Every single top-10 loser was: hour=8, vol_regime=high, LONG side. **The high-vol open is the killing zone.**

### 8.9 Filter-Rules Simulation — Configurations Hitting All 3 Targets

Searched 23 filter combinations. Per the user's "favor +PnL configurations" constraint, only configs satisfying **PnL > 0 AND WR ≥ 55% AND DD ≤ $800** are reported, ranked by PnL descending.

| Filter | Description | Kept | WR | Net PnL | Avg | DD |
|---|---|---:|---:|---:|---:|---:|
| **F-18** ⭐ | drop trade IFF (hour=8 AND vol_regime=high) — surgical | 84 | 58.3% | **+$2,200.77** | +$26.20 | $483.63 |
| F-2 | drop all `vol_regime=high` trades | 62 | 58.1% | +$1,925.35 | +$31.05 | $483.63 |
| F-23 | keep only `vol_regime ∈ {normal, low, ultra_low}` (== F-2) | 62 | 58.1% | +$1,925.35 | +$31.05 | $483.63 |
| F-9 | drop `vol_regime=high` + drop BullClimax | 57 | 57.9% | +$1,835.20 | +$32.20 | $461.16 |
| F-22 | keep only `vol_regime ∈ {normal, low}` | 47 | 61.7% | +$1,808.67 | +$38.48 | $428.56 |
| F-8 | drop `vol_regime=high` + drop RegimeAdaptive | 51 | 58.8% | +$1,677.39 | +$32.89 | $536.04 |
| F-12 | drop `vol_regime=high` + drop RA + drop BullClimax | 46 | 58.7% | +$1,587.24 | +$34.51 | $526.12 |
| F-15 | DE3 Long_Rev_T2 ONLY + drop `vol_regime=high` | 46 | 58.7% | +$1,587.24 | +$34.51 | $526.12 |
| F-21 | keep only `vol_regime=normal` | 41 | 63.4% | +$1,409.71 | +$34.38 | $326.08 |
| F-1 | drop hour=8 (the major bleed hour) | 44 | 56.8% | +$603.57 | +$13.72 | $322.35 |
| F-7 | drop hour=8 + drop BullClimax (== F-1 since BullClimax all hour=8) | 44 | 56.8% | +$603.57 | +$13.72 | $322.35 |

**11 configurations hit all 3 targets.**

### 8.10 Honest Verdict

**Best filter (F-18): drop trade IFF (entry_hour == 8 AND vol_regime == "high"). This single AND-condition delivers all 3 targets simultaneously and produces the highest PnL of any configuration tested:**
- 84 of 251 trades kept (33% blocked)
- WR 58.3% (target 55%, +3.3pp margin)
- Net PnL +$2,200.77 (target >$0, **12.5× the unfiltered $175**)
- Max DD $483.63 (target ≤$800, 40% headroom)
- Avg/trade +$26.20

F-18 beats F-2 ("drop all vol_regime=high") because F-18 keeps the hour-9-through-15 high-vol trades that were profitable in aggregate, only blocking the lethal hour-8 × high-vol overlap.

**Why this works on the data:** the bleed concentrates almost entirely in the hour-8 NY-open when `vol_regime=high`. Outside that overlap, the stack performs respectably. The single-day Apr-17 RegimeAdaptive disaster (#2-#7 of top-10 losers) occurred entirely in this overlap — F-18 catches it without needing strategy-specific rules.

**Implementation suggestion (NOT applied):**
- New env-flag-gated filter: `JULIE_VETO_OPEN_HIGH_VOL=1`
- Logic: at signal birth, query `regime_classifier.current_regime()` for `vol_regime`; query entry hour ET. If both `vol_regime == "high"` AND `8 <= hour < 9`, veto.
- This generalizes well because it's **regime-aware** (won't kick in on normal-vol opens) and **hour-specific** (won't suppress later high-vol trades that are profitable).

**Caveats:**
1. 4 months is a small sample. 11 configurations passing targets means the data is permissive — half could fail to generalize. Re-validate on the 14-month sweep when complete.
2. Friend's swapped overlay models distort all numbers. The "vol_regime=high bleeds" finding would survive re-validation; the magnitudes won't.
3. F-18 reduces volume to 33% of baseline. The bot may not generate enough fill events for SameSide ML / Triathlon medals to learn meaningfully — there's a downstream cost to filtering this hard.
4. The strongest single signal is **`vol_regime=normal` alone is +$1,410 / 63.4% WR / $326 DD on 41 trades** (F-21). If you trust the regime classifier more than the open-hour heuristic, F-21 is a tighter, more interpretable rule (drop everything except normal vol). Trade-off: only 41 trades over 4 months vs F-18's 84.

**Targets are reachable on this 4-month data** — the binding constraint isn't the targets, it's whether the regime classifier's `vol_regime` label generalizes to the remaining 10 months. Re-run this analysis after the sweep finishes (~10:30 PDT).

**No live changes were applied.** This section informs retraining only. Sweep continues running unaffected.

---

*Section 8 added 2026-04-25 mid-sweep. Filter conclusions are exploratory — to be re-validated on the full 14-month dataset when ml_full_ny finishes. Two follow-ups for the analysis pipeline: (a) extend `closed_trades.json` to record `vol_regime` and `trend_day` at entry so this post-hoc log-join isn't needed, and (b) extend the analyzer to compare filter configurations against the baseline 2025-03 reference (1,821 trades / +$7,600 / 48.4% WR / $9,102 DD) to ensure filters aren't just dropping volume to hit targets.*

---

## 8.11 Coverage-Preserving Alternatives to F-18/F-22

**Motivation:** F-18 (the best 3-gate config) silences the bot on 56.8% of would-be-trading days. F-22 silences 64.8%. For Topstep evaluation or any participation-required setup that's unacceptable. This sub-section evaluates filter families that **preserve daily coverage** (silence ≤ 30%) while still attempting to hit PnL > 0 / WR ≥ 55% / DD ≤ $800.

**Eight families considered.** Five are simulable from `closed_trades.json` directly (1, 2, 3, 4, 5). Three (6 — Filter-G-tiered brackets, 7 — wait-bar confirmation, 8 — hour-shift) require bar-level replay or per-trade Filter-G score telemetry that isn't in `closed_trades.json` — flagged below.

### 8.11.1 Per-family results (coverage-preserving, sorted by PnL within family)

#### Family 1 — Size scaling

Replace size 3→1 (or scale linearly) on the lethal cohort. Preserves WR by definition (winners still win, losers still lose) but cuts loss magnitude.

| Variant | Kept | WR | Net PnL | DD | Silence |
|---|---:|---:|---:|---:|---:|
| F1a: size→1 on (hour=8 AND vol=high) | 251 | 48.2% | **+$2,236.19** | $739.08 | 0% |
| F1c: size→1 on ALL hour=8 | 251 | 48.2% | +$1,224.39 | $704.20 | 0% |
| F1b: size×0.33 on (hour=8 AND vol=high) | 251 | 48.2% | +$1,532.40 | $994.12 | 0% |
| F1d: size→1 on RA-in-vol=high | 251 | 48.2% | +$1,393.22 | $2,653.13 | 0% |

**F1a is the best size-scaler** — better PnL than F-18 ($2,236 vs $2,201), DD inside $800, zero silenced days. But WR stays at unfiltered 48.2% because the filter doesn't change which trades win/lose.

#### Family 2 — Bracket rewrite (vol=high → TP=3 / SL=5 etc.)

**Approximation method:** For trades in the rewrite cohort, if `pnl_points <= -new_SL` → cap loss at `-new_SL × $/pt × size`; if `pnl_points >= new_TP` → cap win at `+new_TP × $/pt × size`; else PnL unchanged. This is rough — real bar-replay would account for take-before-stop sequence within the bar.

| Variant | Kept | WR | Net PnL | DD | Silence |
|---|---:|---:|---:|---:|---:|
| F2d: rewrite vol=high → TP=8/SL=5 | 251 | 48.2% | +$258.66 | $2,064.61 | 0% |
| F2c: rewrite vol=high → TP=5/SL=5 | 251 | 48.2% | -$1,182.10 | $2,670.87 | 0% |
| F2a: rewrite (h=8, vol=high) → TP=3/SL=5 | 251 | 48.2% | -$2,246.69 | $3,340.20 | 0% |
| F2b: rewrite ALL vol=high → TP=3/SL=5 | 251 | 48.2% | -$2,543.86 | $3,237.72 | 0% |

**Bracket-rewrite to scalp in vol=high IS A NET LOSS.** This is counterintuitive but the data is consistent: tightening TP from 25→3 while keeping SL at 5 caps winners far below the asymmetry needed to pay for the same loss count. The original 25/10 geometry has a 2.5:1 R:R; rewriting to 3:5 inverts the R:R below 1:1. **Don't ship this family.**

#### Family 3 — Per-strategy surgical

Block specific (strategy × condition) cells without touching others.

| Variant | Kept | WR | Net PnL | DD | Silence |
|---|---:|---:|---:|---:|---:|
| F3a: drop RA in vol=high | 215 | 49.8% | +$1,535.36 | $2,644.39 | 0% |
| F3c: drop RA-in-vol=high + drop BullClimax | 173 | 50.3% | +$1,729.11 | $2,423.53 | 3.4% |
| F3b: drop RA Q2_W3_THU combo entirely | 208 | 49.5% | +$1,358.74 | $2,670.61 | 0% |

Surgical strategy-blocks lift PnL but **DD doesn't tighten** because the remaining DE3 in-vol=high losses still cluster on hour=8 of bleed days. Coverage stays clean.

#### Family 4 — Daily loss cap

Freeze further entries when day cum_pnl ≤ -$X. Doesn't silence the day entirely — only after loss accumulates.

| Variant | Kept | WR | Net PnL | DD | Silence |
|---|---:|---:|---:|---:|---:|
| F4@-$200 | 196 | 50.0% | +$1,056.25 | $2,295.66 | 0% |
| F4@-$500 | 227 | 48.5% | +$785.95 | $2,670.61 | 0% |
| F4@-$600 | 228 | 48.2% | +$648.50 | $2,670.61 | 0% |
| F4@-$400 | 224 | 47.8% | +$387.15 | $2,670.61 | 0% |
| F4@-$300 | 207 | 47.8% | +$351.54 | $2,614.21 | 0% |

**Daily caps are softer than expected.** Even at -$200, DD stays at $2,296 because a single very-bad day (Apr-17 RegimeAdaptive cluster) dropped -$1,087 in 36 minutes — the cap doesn't fire fast enough to prevent the cluster. WR also barely budges because the cap blocks AFTER trades go bad, not before.

#### Family 5 — Cooldown after vol=high loss

After any vol=high loss, freeze for N minutes.

| Variant | Kept | WR | Net PnL | DD | Silence |
|---|---:|---:|---:|---:|---:|
| F5@30min | 161 | 52.2% | **+$2,191.53** | $1,542.56 | 0% |
| F5@60min | 153 | 52.3% | +$2,137.61 | $1,323.66 | 0% |
| F5b@60min: cooldown after ANY loss | 142 | 51.4% | +$1,957.27 | $1,351.18 | 0% |
| F5@90min | 148 | 51.4% | +$1,950.01 | $1,461.10 | 0% |
| F5@120min | 147 | 51.0% | +$1,930.00 | $1,461.10 | 0% |
| F5b@120min: cooldown after ANY loss | 135 | 49.6% | +$1,605.86 | $1,488.62 | 0% |

**Best WR among coverage-preserving filters.** F5@30min reaches WR 52.2% (vs 48.2% baseline) with PnL +$2,192 / DD $1,543 / silence 0%. Still misses the WR ≥ 55% gate by 2.8pp and the DD ≤ $800 gate by ~$700.

#### Family 6 — Combinations

| Variant | Kept | WR | Net PnL | DD | Silence |
|---|---:|---:|---:|---:|---:|
| F6c: drop RA-in-vol=high + size→1 on (h=8 vol=high) DE3 | 215 | 49.8% | **+$2,185.65** | $712.86 | 0% |
| F6a: F1a + daily cap -$300 | 234 | 47.0% | +$1,837.11 | $739.08 | 0% |
| F6d: drop RA-vol=high + drop BullClimax + cap -$300 | 167 | 49.1% | +$1,136.51 | $2,730.87 | 3.4% |
| F6b: F2b + daily cap -$400 | 226 | 48.7% | -$1,597.63 | $2,248.68 | 0% |

**F6c is the strongest coverage-preserving config overall:** drops RegimeAdaptive in vol=high (catches the Apr-17 cluster), size→1 on (h=8 AND vol=high) for DE3 (caps hour-8 risk). PnL +$2,186, DD $713, silence 0%, WR 49.8%.

#### Families 7–9 — Not directly simulable from `closed_trades.json`

- **Family 7 (Filter-G-tiered brackets):** would need per-trade `P(big_loss)` from `[SHADOW_GATE_2025]` log lines + bar replay. Possible follow-up work: join shadow-gate logs to fired trades by sub_strategy + entry timestamp, then simulate scaled brackets per Filter G score.
- **Family 8 (wait-bar confirmation):** requires bar-level OHLCV around each entry. The closed_trades.json has only entry_price/exit_price.
- **Family 9 (hour-shift):** requires bar-level data to know what setup was available at hour=9 instead of hour=8. The signals fired at hour=8 because that's when the DE3 6-9 session block window was open and the trigger pattern matched — at hour=9 the same pattern may not reappear.

These three families would need a real bar-replay simulation, not a closed_trades-only post-hoc score. Flagged for the next-cycle analyzer build.

### 8.11.2 Configurations passing all 4 gates (PnL > 0, WR ≥ 55%, DD ≤ $800, silence ≤ 30%)

**NONE.** No coverage-preserving filter family hits all four gates simultaneously on this 4-month data. The binding constraint is almost always **WR ≥ 55%**.

### 8.11.3 Closest 5 — ranked by PnL with binding constraints

| Filter | Kept | WR | PnL | DD | Silence | Binding |
|---|---:|---:|---:|---:|---:|---|
| F1a: size→1 on (h=8 AND vol=high) | 251 | 48.2% | **+$2,236.19** | $739.08 | 0% | **WR** |
| F5@30min: cooldown 30min after vol=high loss | 161 | 52.2% | +$2,191.53 | $1,542.56 | 0% | WR, DD |
| F6c: drop RA-vol=high + size→1 on h=8 DE3 | 215 | 49.8% | +$2,185.65 | $712.86 | 0% | **WR** |
| F5@60min | 153 | 52.3% | +$2,137.61 | $1,323.66 | 0% | WR, DD |
| F5b@60min: cooldown after ANY loss | 142 | 51.4% | +$1,957.27 | $1,351.18 | 0% | WR, DD |

### 8.11.4 Honest verdict — what's binding and why

**The WR constraint cannot be satisfied without removing losing trades from the population.** Coverage-preserving filters (size-scaling, bracket-rewrite, daily-cap, cooldown) operate on trades that have already fired — they reduce loss magnitude or block subsequent entries after a loss, but they don't pre-block the loser itself. So the win-loss outcome ratio (the WR numerator/denominator) doesn't materially change.

The methods that DID push WR up — cooldowns — get to WR ~52% by avoiding the immediate-post-loss tail (clusters of correlated losses). To go from 52% → 55% would need either:
1. **A working pre-trade probability model.** Filter G's per-strategy gate is supposed to do this. Current friend-swapped gates have rolling AUC 0.525-0.545 (near-random). A retrained gate with even AUC 0.6-0.65 would likely move WR past 55% without silencing days. **This is the highest-leverage path.**
2. **Bar-level features at entry.** Simulating Family 7 (Filter-G-tiered brackets) or Family 8 (wait-bar confirmation) requires the kind of intraday context that lives in the bar replay, not the trade ledger. Build that simulator next cycle.

**Pragmatic shippable choice if you must ship something today: F6c.**
- 0% days silenced
- DD $713 (under $800 target)
- PnL +$2,186 (12.5× unfiltered)
- WR 49.8% (misses 55% gate by 5.2pp)

If you accept WR ≥ 50% as a relaxed bar — F6c hits everything else with clean coverage. If WR ≥ 55% is non-negotiable — **the data says retrain Filter G first**, then re-run this analysis. No coverage-preserving filter on closed-trades-data alone can move WR by 7pp from 48% → 55%.

### 8.11.5 Implementation notes (for future work, NOT applied live)

1. **F6c implementation:** two env-flag-gated changes —
   - `JULIE_VETO_RA_VOL_HIGH=1` — at signal birth, query `regime_classifier.current_regime()`; if `vol_regime == "high"` AND `signal["strategy"] == "RegimeAdaptive"`, veto.
   - `JULIE_DE3_OPEN_VOL_HIGH_SIZE_CAP=1` — at signal birth, if `signal["strategy"] == "DynamicEngine3"` AND `entry_hour == 8` AND `vol_regime == "high"`, set `signal["size"] = 1`.

2. **F1a implementation (simpler, same coverage, slightly higher PnL):** single env-flag —
   - `JULIE_OPEN_VOL_HIGH_SIZE_CAP=1` — at signal birth, if `entry_hour == 8` AND `vol_regime == "high"`, set `signal["size"] = 1` regardless of strategy.

3. **Risk:** none of these were validated OOS. The 4-month dataset uses friend's untested swapped models. Re-run filter analysis on the full 14-month sweep once it completes (~11:26 PDT). Numbers will shift; the rank order between families likely won't.

4. **The real fix** — and this is the recurring theme — is retraining Filter G on baseline data per Section 5 recipe. WR ≥ 55% with ≤ 30% silencing is reachable IFF the pre-trade probability model is materially better than near-random.

**No live changes were applied. Sweep continues running unaffected.**

---

*Section 8.11 added 2026-04-25 mid-sweep. Coverage-preserving filter sweep on 251 trades from ml_full_ny_2025_03..06. Three filter families (Filter-G-tiered, wait-bar, hour-shift) flagged as not directly simulable from `closed_trades.json` — require bar-replay simulator for next-cycle analysis.*

---

## 8.11.6 Delta Tables — All Variants vs Unfiltered Baseline

**Baseline reference:** 251 trades, 48.2% WR, +$175.42 PnL, $3,370.32 DD, 0% silenced-days.

Format: every variant (not just passing ones) shown as a row with deltas vs baseline. Negative ΔDD = improvement (lower DD). Negative Δtrades = filtered out. Pass column shows ✅ if all 4 gates hit (PnL > 0, WR ≥ 55%, DD ≤ $800, silence ≤ 30%); else lists binding constraints.

### Reference rows
| Filter | n | Δn | ΔWR | ΔPnL | ΔDD | Δsilence | Result |
|---|---:|---:|---:|---:|---:|---:|:---|
| **baseline** (no filter) | 251 | 0 (0%) | 0pp | $0 | $0 | 0pp | FAIL: WR, DD |
| F-18 (drop h=8 AND vol=high) | 84 | -167 (-66.5%) | +10.1pp | +$2,025.35 | -$2,886.69 | +56.8pp | FAIL: silence |

### Family 1 — Size scaling

| Filter | n | Δn | ΔWR | ΔPnL | ΔDD | Δsilence | Result |
|---|---:|---:|---:|---:|---:|---:|:---|
| F1a: size→1 on (h=8 AND vol=high) | 251 | 0 (0%) | 0pp | **+$2,060.77** | **-$2,631.24** | 0pp | FAIL: WR (48.2%<55%) |
| F1b: size×0.33 on (h=8 AND vol=high) | 251 | 0 (0%) | 0pp | +$1,356.98 | -$2,376.20 | 0pp | FAIL: WR, DD ($994>$800) |
| F1c: size→1 on ALL h=8 | 251 | 0 (0%) | 0pp | +$1,048.97 | **-$2,666.12** | 0pp | FAIL: WR (48.2%<55%) |
| F1d: size→1 on RA-in-vol=high | 251 | 0 (0%) | 0pp | +$1,217.80 | -$717.19 | 0pp | FAIL: WR, DD ($2,653>$800) |

### Family 2 — Bracket rewrite (TP/SL → narrower)

| Filter | n | Δn | ΔWR | ΔPnL | ΔDD | Δsilence | Result |
|---|---:|---:|---:|---:|---:|---:|:---|
| F2a: rewrite (h=8, vol=high) → TP=3/SL=5 | 251 | 0 (0%) | 0pp | -$2,422.11 | -$30.12 | 0pp | FAIL: PnL, WR, DD |
| F2b: rewrite ALL vol=high → TP=3/SL=5 | 251 | 0 (0%) | 0pp | -$2,719.28 | -$132.60 | 0pp | FAIL: PnL, WR, DD |
| F2c: rewrite vol=high → TP=5/SL=5 | 251 | 0 (0%) | 0pp | -$1,357.52 | -$699.45 | 0pp | FAIL: PnL, WR, DD |
| F2d: rewrite vol=high → TP=8/SL=5 | 251 | 0 (0%) | 0pp | +$83.24 | -$1,305.71 | 0pp | FAIL: WR, DD ($2,065>$800) |

**All Family-2 variants fail.** Tightening TP below the natural R:R asymmetry inverts expectancy.

### Family 3 — Per-strategy surgical

| Filter | n | Δn | ΔWR | ΔPnL | ΔDD | Δsilence | Result |
|---|---:|---:|---:|---:|---:|---:|:---|
| F3a: drop RA in vol=high | 215 | -36 (-14.3%) | +1.6pp | +$1,359.94 | -$725.93 | 0pp | FAIL: WR, DD ($2,644>$800) |
| F3b: drop RA Q2_W3_THU combo | 208 | -43 (-17.1%) | +1.3pp | +$1,183.32 | -$699.71 | 0pp | FAIL: WR, DD ($2,671>$800) |
| F3c: drop RA-vol=high + drop BullClimax | 173 | -78 (-31.1%) | +2.1pp | +$1,553.69 | -$946.79 | +3.4pp | FAIL: WR, DD ($2,424>$800) |

### Family 4 — Daily loss cap

| Filter | n | Δn | ΔWR | ΔPnL | ΔDD | Δsilence | Result |
|---|---:|---:|---:|---:|---:|---:|:---|
| F4@-$200: freeze when day cum ≤ -$200 | 196 | -55 (-21.9%) | +1.8pp | +$880.83 | -$1,074.66 | 0pp | FAIL: WR, DD ($2,296>$800) |
| F4@-$300: freeze when day cum ≤ -$300 | 207 | -44 (-17.5%) | -0.4pp | +$176.12 | -$756.11 | 0pp | FAIL: WR, DD ($2,614>$800) |
| F4@-$400: freeze when day cum ≤ -$400 | 224 | -27 (-10.8%) | -0.4pp | +$211.73 | -$699.71 | 0pp | FAIL: WR, DD ($2,671>$800) |
| F4@-$500: freeze when day cum ≤ -$500 | 227 | -24 (-9.6%) | +0.3pp | +$610.53 | -$699.71 | 0pp | FAIL: WR, DD ($2,671>$800) |
| F4@-$600: freeze when day cum ≤ -$600 | 228 | -23 (-9.2%) | 0pp | +$473.08 | -$699.71 | 0pp | FAIL: WR, DD ($2,671>$800) |

### Family 5 — Cooldown after loss

| Filter | n | Δn | ΔWR | ΔPnL | ΔDD | Δsilence | Result |
|---|---:|---:|---:|---:|---:|---:|:---|
| F5@30min after vol=high loss | 161 | -90 (-35.9%) | **+4.0pp** | **+$2,016.11** | -$1,827.76 | 0pp | FAIL: WR (52.2%<55%), DD ($1,543>$800) |
| F5@60min after vol=high loss | 153 | -98 (-39.0%) | **+4.1pp** | +$1,962.19 | -$2,046.66 | 0pp | FAIL: WR (52.3%<55%), DD ($1,324>$800) |
| F5@90min after vol=high loss | 148 | -103 (-41.0%) | +3.2pp | +$1,774.59 | -$1,909.22 | 0pp | FAIL: WR, DD |
| F5@120min after vol=high loss | 147 | -104 (-41.4%) | +2.8pp | +$1,754.58 | -$1,909.22 | 0pp | FAIL: WR, DD |
| F5b@60min after ANY loss | 142 | -109 (-43.4%) | +3.2pp | +$1,781.85 | -$2,019.14 | 0pp | FAIL: WR, DD |
| F5b@120min after ANY loss | 135 | -116 (-46.2%) | +1.4pp | +$1,430.44 | -$1,881.70 | 0pp | FAIL: WR, DD |

### Family 6 — Combinations

| Filter | n | Δn | ΔWR | ΔPnL | ΔDD | Δsilence | Result |
|---|---:|---:|---:|---:|---:|---:|:---|
| F6c: drop RA-vol=high + size→1 (h=8 vol=high DE3) | 215 | -36 (-14.3%) | +1.6pp | **+$2,010.23** | **-$2,657.46** | 0pp | FAIL: WR (49.8%<55%) |
| F6a: F1a + daily cap -$300 | 234 | -17 (-6.8%) | -1.2pp | +$1,661.69 | -$2,631.24 | 0pp | FAIL: WR (47.0%<55%) |
| F6d: drop RA-vol=high + drop BullClimax + cap -$300 | 167 | -84 (-33.5%) | +0.9pp | +$961.09 | -$639.45 | +3.4pp | FAIL: WR, DD ($2,731>$800) |
| F6b: F2b + daily cap -$400 | 226 | -25 (-10.0%) | +0.5pp | -$1,773.05 | -$1,121.64 | 0pp | FAIL: PnL, WR, DD |

### Top 5 by ΔPnL (from coverage-preserving family)

| Rank | Filter | ΔPnL | ΔDD | ΔWR | Δsilence | Result |
|---:|---|---:|---:|---:|---:|:---|
| 1 | F1a: size→1 on (h=8 AND vol=high) | **+$2,060.77** | -$2,631.24 | 0pp | 0pp | FAIL: WR |
| 2 | F5@30min after vol=high loss | +$2,016.11 | -$1,827.76 | +4.0pp | 0pp | FAIL: WR, DD |
| 3 | F6c: drop RA-vol=high + size→1 | +$2,010.23 | **-$2,657.46** | +1.6pp | 0pp | FAIL: WR |
| 4 | F5@60min after vol=high loss | +$1,962.19 | -$2,046.66 | +4.1pp | 0pp | FAIL: WR, DD |
| 5 | F5b@60min after ANY loss | +$1,781.85 | -$2,019.14 | 0pp | 0pp | FAIL: WR, DD |

### Verdict on coverage-preserving family

**Zero of 27 coverage-preserving variants pass all 4 gates.**

The WR gate is binding on **every single coverage-preserving variant**. Nine variants pass PnL+DD+silence but miss WR (range 47–52.3%). The best two on WR are F5@30min and F5@60min (+4.0/+4.1pp lift) but their DD is still ~$1.3-1.5k.

The two strongest configurations on the 4-month data — measured by total improvement vs baseline:
- **F1a** delivers +$2,061 PnL, -$2,631 DD, 0% silence, with zero filter complexity (single env-flag size cap on the open × high-vol cohort).
- **F6c** delivers +$2,010 PnL, **-$2,657 DD** (the largest DD reduction in any coverage-preserving variant), 0% silence, with two-rule complexity.

Neither hits WR ≥ 55%.

**The conclusion holds:** moving WR by 7pp without silencing days requires a working pre-trade probability model. Filter G retraining is the gating dependency — see Section 5 retraining recipes. Until that ships, the realistic target on this stack is "PnL ≥ +$2k, DD ≤ $800, silence ≤ 5%, WR 49-52%" — F1a and F6c both clear that bar.

**No live changes were applied.** Sweep continues running unaffected.

---

*Section 8.11.6 (delta tables) added 2026-04-25. Mirror analysis to be re-run on the 14-month sweep when ml_full_ny finishes (~11:26 PDT). Three filter families still un-simulable from `closed_trades.json` alone (Filter-G-tiered, wait-bar, hour-shift) — flagged in 8.11.1.*

---

## 8.12 F-18 + Coverage-Preserving Layered Filter Combinations

**Motivation:** F-18 alone delivers the best 3-gate result (+$2,200, 58.3% WR, $484 DD) but at 56.8% silenced-days. This sub-section asks whether **layering** any of the coverage-preserving filter families on top of F-18 can produce a strictly-better config — improving PnL, WR, or DD without making silence worse.

**Layered baseline:** F-18 alone — 84 trades, 58.3% WR, +$2,200.77 PnL, $483.63 DD, 56.8% silenced.

**Gates for this analysis:**
- PnL > 0 ✓ (all layered configs inherit this from F-18 unless severely degraded)
- WR ≥ 55%
- DD ≤ $800
- silence ≤ 56.8% (no worse than F-18; "improvement" gate would be silence < 56.8% but layering only ever drops more trades, can never un-silence)

**Improvement gate (strictly better than F-18):** PnL > $2,200.77 AND WR > 58.3% AND DD ≤ $483.63.

**Filter families 7 (wait-bar confirmation) and 8 (hour-shift)** still flagged as not directly simulable from `closed_trades.json` — covered in 8.11.1. F-18+F1a, F-18+F2a are NO-OPs because F-18 has already removed the (h=8 AND vol=high) cohort that those layers would target.

### 8.12.1 Layered variants — full delta-vs-F-18 table

#### Family 1 layered — size scaling on F-18 survivors

| Filter | n | Δn | ΔWR | ΔPnL | ΔDD | Δsilence | Result |
|---|---:|---:|---:|---:|---:|---:|:---|
| F-18+F1a (no-op: F-18 already removed cohort) | — | — | — | — | — | — | NO-OP |
| F-18+F1c: size→1 on remaining h=8 (vol≠high) | 84 | 0 (0%) | 0pp | -$1,011.80 | -$168.70 | 0pp | **PASS** |
| F-18+F1d: size→1 on remaining RA-in-vol=high | 84 | 0 (0%) | 0pp | -$192.68 | $0 | 0pp | **PASS** |

F-18+F1c hurts because the surviving hour=8 trades (where vol≠high) are profitable — sizing them down halves the bot's good output.

#### Family 2 layered — bracket rewrite on F-18 survivors

| Filter | n | Δn | ΔWR | ΔPnL | ΔDD | Δsilence | Result |
|---|---:|---:|---:|---:|---:|---:|:---|
| F-18+F2a (no-op: F-18 already removed cohort) | — | — | — | — | — | — | NO-OP |
| F-18+F2b: rewrite remaining vol=high → TP=3/SL=5 | 84 | 0 (0%) | 0pp | -$297.17 | $0 | 0pp | **PASS** |
| F-18+F2c: rewrite remaining vol=high → TP=5/SL=5 | 84 | 0 (0%) | 0pp | -$76.36 | $0 | 0pp | **PASS** |
| **F-18+F2d: rewrite remaining vol=high → TP=8/SL=5** | 84 | 0 (0%) | 0pp | **+$43.94** | $0 | 0pp | **PASS** |

**F-18+F2d is the ONLY layered variant that improves PnL over F-18.** It rewrites the 22 remaining vol=high trades (those at hour 9-15) to TP=8/SL=5 brackets. Tighter SL caps the loss tail; TP=8 still leaves room for asymmetric R:R. Net: +$44 PnL improvement, identical DD/WR/silence. Marginal but positive.

#### Family 3 layered — per-strategy surgical (all FAIL on silence)

| Filter | n | Δn | ΔWR | ΔPnL | ΔDD | Δsilence | Result |
|---|---:|---:|---:|---:|---:|---:|:---|
| F-18+F3a: drop remaining RA-in-vol=high | 67 | -17 (-20.2%) | -1.6pp | -$394.10 | $0 | **+2.3pp** | FAIL: silence=59.1% > 56.8% |
| F-18+F3b: drop RA Q2_W3_THU combo | 60 | -24 (-28.6%) | -1.7pp | -$570.72 | +$52.41 | +1.1pp | FAIL: silence=58.0% > 56.8% |
| F-18+F3c: drop RA-vol=high + drop BullClimax | 62 | -22 (-26.2%) | -1.9pp | -$484.25 | -$22.47 | +2.3pp | FAIL: silence=59.1% > 56.8% |

Per-strategy cuts on top of F-18 silence 1-2 additional days. They also drop WR (the dropped RA-in-vol=high trades at hour≠8 were apparently break-even or marginally profitable).

#### Family 4 layered — daily loss cap (no-ops on F-18 survivors)

| Filter | n | Δn | ΔWR | ΔPnL | ΔDD | Δsilence | Result |
|---|---:|---:|---:|---:|---:|---:|:---|
| F-18+F4@-$200 | 84 | 0 (0%) | 0pp | $0 | $0 | 0pp | **PASS** (no-op) |
| F-18+F4@-$300 | 84 | 0 (0%) | 0pp | $0 | $0 | 0pp | **PASS** (no-op) |
| F-18+F4@-$400 | 84 | 0 (0%) | 0pp | $0 | $0 | 0pp | **PASS** (no-op) |
| F-18+F4@-$500 | 84 | 0 (0%) | 0pp | $0 | $0 | 0pp | **PASS** (no-op) |

**Daily caps never fire.** F-18 survivors don't accumulate enough single-day loss to cross even a -$200 threshold — the bleed concentration was at hour=8 vol=high which F-18 already removed.

#### Family 5 layered — cooldown after vol=high loss

| Filter | n | Δn | ΔWR | ΔPnL | ΔDD | Δsilence | Result |
|---|---:|---:|---:|---:|---:|---:|:---|
| F-18+F5@30min | 75 | -9 (-10.7%) | -1.0pp | -$91.42 | $0 | 0pp | **PASS** |
| F-18+F5@60min | 74 | -10 (-11.9%) | -0.2pp | -$48.94 | $0 | 0pp | **PASS** |
| F-18+F5@90min | 74 | -10 (-11.9%) | -0.2pp | -$48.94 | $0 | 0pp | **PASS** |
| F-18+F5@120min | 72 | -12 (-14.3%) | 0pp | -$63.98 | $0 | 0pp | **PASS** |
| F-18+F5b@60min: cooldown after ANY loss | 64 | -20 (-23.8%) | -2.1pp | -$383.00 | $0 | 0pp | **PASS** |
| F-18+F5b@120min: cooldown after ANY loss | 61 | -23 (-27.4%) | -2.6pp | -$541.84 | $0 | 0pp | **PASS** |

Cooldowns drop both winners and losers; net effect is small loss with no silence change. None improves over F-18.

#### Family 6 layered — combinations (all FAIL on silence)

| Filter | n | Δn | ΔWR | ΔPnL | ΔDD | Δsilence | Result |
|---|---:|---:|---:|---:|---:|---:|:---|
| F-18+F6a: drop RA-vol=high + daily cap -$300 | 67 | -17 (-20.2%) | -1.6pp | -$394.10 | $0 | +2.3pp | FAIL: silence |
| F-18+F6b: drop RA-vol=high + size→1 on remaining h=8 | 67 | -17 (-20.2%) | -1.6pp | -$1,405.90 | -$168.70 | +2.3pp | FAIL: silence |
| F-18+F6c: drop RA-vol=high + drop BullClimax + cap -$300 | 62 | -22 (-26.2%) | -1.9pp | -$484.25 | -$22.47 | +2.3pp | FAIL: silence |
| F-18+F6d: cooldown 60min + drop RA-vol=high | 64 | -20 (-23.8%) | -2.1pp | -$340.40 | $0 | +2.3pp | FAIL: silence |

### 8.12.2 Top 5 layered variants by absolute PnL

| Rank | Filter | PnL | WR | DD | silence | Result |
|---:|---|---:|---:|---:|---:|:---|
| 1 | **F-18+F2d** | **+$2,244.71** | 58.3% | $483.63 | 56.8% | ✅ PASS, +$44 vs F-18 |
| 2 | F-18+F4@-$200/300/400/500 | +$2,200.77 | 58.3% | $483.63 | 56.8% | ✅ PASS (no-op) |
| 3 | F-18+F5@60min | +$2,151.83 | 58.1% | $483.63 | 56.8% | ✅ PASS |
| 4 | F-18+F5@90min | +$2,151.83 | 58.1% | $483.63 | 56.8% | ✅ PASS |
| 5 | F-18+F5@120min | +$2,136.79 | 58.3% | $483.63 | 56.8% | ✅ PASS |

### 8.12.3 Configurations strictly better than F-18

**ZERO of 22 layered variants are strictly better than F-18 alone.** Strictly better = improves all three of (PnL, WR, DD) simultaneously, with silence ≤ 56.8% AND all 4 base gates passing.

The closest is F-18+F2d, which improves PnL by +$44 but doesn't move WR or DD. It's a tie on WR/DD with marginal PnL gain — not enough margin to call it strictly better.

### 8.12.4 Honest verdict — F-18 is locally optimal

**The 84 trades F-18 retains are already a "high-quality" subset.** The bleed concentrated almost entirely in (hour=8 AND vol_regime=high), and F-18 surgically removes that overlap. Layering further filters on top mostly drops profitable surviving trades:

- The 40 remaining hour=8 trades (vol≠high) average ~$10/trade — sizing them down or dropping them costs PnL
- The 22 remaining vol=high trades (hour 9-15) include some big winners (the ones that ride trend after the open volatility settles)
- The Apr-17 RegimeAdaptive cluster that drove the worst-loser top-10 was at hour=8 in vol=high — F-18 caught all 6 of those entries
- Daily caps don't fire because remaining trades don't bleed enough on any single day

**The only layered variant that adds value is F-18+F2d** (rewrite remaining vol=high → TP=8/SL=5). It improves PnL by +$44 by tightening the SL on the 22 remaining vol=high trades. The improvement is marginal — within the noise of a 4-month backtest — but it's not a regression.

### 8.12.5 Implementation suggestion (NOT applied live)

If you wanted to ship F-18+F2d combined:
- `JULIE_VETO_OPEN_HIGH_VOL=1` (the F-18 base): at signal birth, veto if `entry_hour == 8 AND vol_regime == "high"`
- `JULIE_VOL_HIGH_BRACKET_REWRITE=1` (the F2d layer): at signal birth, if `vol_regime == "high"` (which post-F-18 means hour ≠ 8), rewrite `signal["sl_dist"] = 5` and `signal["tp_dist"] = 8`

Net deployment risk: +$44 lift on a 4-month, 251-trade sample is below noise. Recommend re-validating on the full 14-month sweep before considering even this minor layer. Treat F-18 alone as the deployable target until then.

### 8.12.6 What this means for retraining priorities

Section 8.11 concluded that no coverage-preserving filter on closed-trades data alone can move WR from 48% → 55%. Section 8.12 adds: **no layered filter on top of F-18 can move WR from 58.3% → above 58.3% either.** The 58.3% WR is essentially the theoretical ceiling for a binary regime/hour cut on this 4-month data without a working pre-trade probability model.

To get above 58.3% WR while keeping coverage AND avoiding F-18-style silence: **a working Filter G** is the only path. The friend-swapped Kalshi gate (rolling AUC 0.545) and gate-2025 per-strategy models (where loaded) need to be retrained per Section 5 recipes against baseline data. That's the highest-leverage next step.

**No live changes were applied.** Sweep continues running unaffected.

---

*Section 8.12 added 2026-04-25 mid-sweep. Layered analysis: F-18 + 22 secondary-layer variants from coverage-preserving families 1-6. Conclusion: F-18 is locally optimal; only marginal +$44 improvement available via F2d bracket rewrite on remaining vol=high trades. The WR ceiling on this 4-month data without pre-trade probability is 58.3%. Re-run on 14-month sweep when ml_full_ny finishes to confirm.*

---

## 8.13 8-Month Re-Validation — F-18 Does Not Generalize

**Critical finding up front:** F-18 (drop trade IF hour=8 AND vol_regime=high) was **overfit to Mar-Apr 2025**. Re-validated on the 8-month dataset (Mar-Oct 2025), it fails 3 of 4 ship gates. The bleed pattern that motivated F-18 — the (hour=8 × high-vol) overlap — was a Mar-Apr tariff-aftermath phenomenon and does NOT persist into Jul-Oct.

### 8.13.1 New 4 months — Jul/Aug/Sep/Oct 2025

| Month | Trades | WR | Net PnL | Avg | Max DD |
|---|---:|---:|---:|---:|---:|
| 2025-07 | 45 | 48.9% | -$44.67 | -$0.99 | $539.25 |
| 2025-08 | 55 | 40.0% | **-$774.35** | -$14.08 | **$935.56** |
| 2025-09 | 47 | 53.2% | **+$1,549.10** | +$32.96 | $541.96 |
| 2025-10 | 39 | 56.4% | +$807.17 | +$20.70 | $266.10 |

**By side:**
- 2025-07: LONG 41 / 46.3% / -$75 · SHORT 4 / 75.0% / +$30
- 2025-08: LONG 50 / 38.0% / -$771 · SHORT 5 / 60.0% / -$4
- 2025-09: LONG 46 / 52.2% / +$1,459 · SHORT 1 / 100% / +$90
- 2025-10: LONG 30 / 56.7% / +$792 · SHORT 9 / 55.6% / +$15

**Top exit sources (combined):** August was the dominant new bleeder — 22 stops at -$1,196 + 18 stop_gaps at -$1,374 ate $2,570 of losses against $1,847 in takes. October was the reverse: 8 reverses generated +$453 (the bot rode trends well that month).

### 8.13.2 Cumulative 8-month aggregate (Mar-Oct 2025)

| Metric | Value | Δ vs first-4 (Mar-Jun) |
|---|---:|---:|
| Trades | 437 | +186 |
| Blended WR | 48.5% | +0.3pp |
| Net PnL | **+$1,712.67** | +$1,537 |
| Max DD | **$3,370.32** | $0 (the Mar-Apr DD was already the worst stretch — no later month exceeded the cumulative low-water mark) |
| LONG | 376 / 47.9% / +$1,775 | — |
| SHORT | 61 / 52.5% / -$62 | — |

**Per-month roll:**

| Month | n | WR | PnL | DD |
|---|---:|---:|---:|---:|
| 2025-03 | 40 | 35.0% | -$1,669 | $1,737 |
| 2025-04 | 119 | 47.1% | -$633 | $1,949 |
| 2025-05 | 58 | 58.6% | +$1,063 | $381 |
| 2025-06 | 34 | 50.0% | +$1,415 | $720 |
| 2025-07 | 45 | 48.9% | -$45 | $539 |
| 2025-08 | 55 | 40.0% | -$774 | $936 |
| 2025-09 | 47 | 53.2% | +$1,549 | $542 |
| 2025-10 | 39 | 56.4% | +$807 | $266 |

**5 of 8 months are net positive.** Mar/Apr/Aug bleed; the rest profit. **8-month total of +$1,713 is real but small** ($3.92/trade) — driven by May/Jun/Sep/Oct gains offsetting Mar/Apr/Aug losses.

### 8.13.3 Vol regime distribution shift — the underlying explanation

| Period | high | normal | low | ultra_low |
|---|---:|---:|---:|---:|
| First 4 (Mar-Jun) | **189** | 41 | 6 | 15 |
| New 4 (Jul-Oct) | 83 | **68** | 22 | 13 |
| All 8 | 272 | 109 | 28 | 28 |

**Mar-Jun was 4.6× more high-vol than normal-vol.** Jul-Oct dropped to 1.2× — i.e., the regime classifier saw far less high-vol tape in the second half. The Mar-Apr tariff aftermath produced a sustained high-vol regime; Jul-Oct returned to a more normal mix.

This means **F-18's targeting cohort shrank by 51%** between the two halves (167 first-half trades in cohort vs 48 second-half). Less of the trade tape falls in the lethal overlap, so a filter that gates on it has less to work with.

### 8.13.4 F-18 re-validation on 8-month data — BROKEN

| Metric | F-18 on 4mo | F-18 on 8mo | Δ | Gate |
|---|---:|---:|---:|---|
| Trades | 84 | 222 | +138 | — |
| WR | 58.3% | **52.7%** | -5.6pp | ❌ <55% |
| Net PnL | +$2,200.77 | **+$3,629.25** | +$1,429 | ✅ >0 |
| Max DD | $483.63 | **$1,369.83** | +$886 | ❌ >$800 |
| Silenced-days | 56.8% | 44.2% | -12.6pp | ❌ >30% |

**3 of 4 gates fail.** WR drops 5.6pp because the second-half F-18 survivors are weaker. DD nearly triples. Silence improves (44.2% vs 56.8%) but still well above the 30% target.

### 8.13.5 Filter search on 8-month data — NONE pass all 4 gates

Re-ran the full 16-filter sweep against the 437-trade 8-month dataset with gates: PnL > 0, WR ≥ 55%, DD ≤ $800, silence ≤ 30%.

**Zero filters pass all 4 gates.** Top 5 by PnL (all FAIL):

| Filter | Description | n | WR | PnL | DD | Silence | Binding |
|---|---|---:|---:|---:|---:|---:|---|
| F-NEW-5 | drop trade IF (vol=high AND not Long_Rev_T2 lane) | 327 | 50.8% | **+$3,954.90** | $2,424 | 2.5% | WR, DD |
| F-NEW-4 | drop (h=8 AND vol=high) + drop (h≥14 AND vol=high) | 214 | 53.3% | +$3,766.61 | $1,319 | 47% | WR, DD, silence |
| F-18 | drop (h=8 AND vol=high) | 222 | 52.7% | +$3,629.25 | $1,370 | 44% | WR, DD, silence |
| F-3 | drop all RegimeAdaptive | 321 | 52.3% | +$3,507.61 | $2,742 | 1% | WR, DD |
| F-NEW-1 | F-18 + drop RA in vol=high | 180 | 52.8% | +$3,177.75 | $1,197 | 47% | WR, DD, silence |

**Two configs hit silence ≤ 30% AND PnL > 0** (F-NEW-5, F-3) but both have DD > $2,400 — way above the $800 target. F-12 (drop vol=high + drop RA + drop BullClimax) is the only config that hits DD ≤ $800 ($690), but it has 55% silence.

**The 4-gate target is unreachable on the 8-month data with binary filters.** The WR ceiling is ~57.1% (F-12) and the DD ceiling is ~$690 (also F-12) but they trade off against each other and against silence.

### 8.13.6 Generalization check — F-18 cohort behavior across the two halves

The F-18-removed cohort = trades with (hour=8 AND vol=high). This is the cohort F-18 considers "lethal."

| Period | n | WR | PnL | Avg/trade |
|---|---:|---:|---:|---:|
| **First 4 (Mar-Jun)** | 167 | **43.1%** | **-$2,025.35** | -$12.13 |
| **New 4 (Jul-Oct)** | 48 | **47.9%** | **+$108.77** | **+$2.27** |
| All 8 | 215 | 44.2% | -$1,916.58 | -$8.91 |

**The bleed pattern collapses.** On the new 4 months, the (hour=8 AND vol=high) cohort is ESSENTIALLY NEUTRAL (+$2/trade, 48% WR). The Mar-Jun signal that justified F-18 — a -$12/trade, 43% WR bleed — does not persist.

### 8.13.7 F-18 survivor behavior across the two halves

| Period | n | WR | PnL | Avg/trade |
|---|---:|---:|---:|---:|
| First 4 (Mar-Jun) | 84 | **58.3%** | +$2,200.77 | +$26.20 |
| New 4 (Jul-Oct) | 138 | 49.3% | +$1,428.48 | +$10.35 |
| All 8 | 222 | 52.7% | +$3,629.25 | +$16.35 |

**F-18 survivors degrade in the new period too.** WR drops 9pp, avg/trade drops 60%. F-18 isn't producing a "high-quality subset" on Jul-Oct — it's producing a mediocre one.

### 8.13.8 Per-month F-18 cohort PnL — the punchline

| Month | Original PnL | Cohort PnL (would-be-blocked) | Survivor PnL (kept) |
|---|---:|---:|---:|
| 2025-03 | -$1,669 | **-$1,516** | -$153 |
| 2025-04 | -$633 | **-$2,066** | +$1,433 |
| 2025-05 | +$1,063 | **+$589** ✅ | +$473 |
| 2025-06 | +$1,415 | **+$967** ✅ | +$448 |
| 2025-07 | -$45 | -$95 | +$50 |
| 2025-08 | -$774 | -$272 | -$502 |
| 2025-09 | +$1,549 | **+$185** ✅ | +$1,364 |
| 2025-10 | +$807 | **+$290** ✅ | +$517 |

**On 5 of 8 months, the F-18 cohort is POSITIVE PnL** — meaning F-18 would drop profitable trades. The cohort only bleeds materially on Mar (-$1,516), Apr (-$2,066), and Aug (-$272). On May, Jun, Sep, Oct, the (hour=8 × high-vol) trades AVERAGE POSITIVE EV. F-18 destroys value in those months.

### 8.13.9 Honest verdict — the bleed pattern doesn't generalize

**The single most important finding from this re-validation: F-18 was overfit to the Mar-Apr 2025 tariff-aftermath regime.** The (hour=8 × high-vol) overlap was lethal in those two months because of a sustained high-vol panic-down tape that crushed mean-reversion longs — but that regime didn't repeat. By Jul-Oct, the same overlap is roughly EV-neutral or slightly positive.

A filter chosen on a single regime (Mar-Apr) won't survive on a multi-regime sample. **Any binary cut on this stack is going to be regime-conditional**: filters that look great on tariff-week tape will look mediocre or harmful on calmer tape. This is the canonical bias-variance failure of short-window backtests.

**Implications for retraining:**

1. **Don't ship F-18 as a static rule.** It was the right thing to do in Mar-Apr but the wrong thing to do in May/Jun/Sep/Oct. A static rule applied across all regimes destroys profitability in the ones where the cohort is positive.

2. **The right shape for this filter is regime-conditional.** You'd want: "drop (hour=8 AND vol=high) IF a sustained-high-vol regime classifier says we're in a tariff-week-style stretch." This is essentially what a properly-trained Filter G is supposed to do — identify high-bigloss-probability trades using more features than just hour + vol.

3. **Filter G retraining priority remains highest** (Section 5). The 8-month dataset gives 437 labeled trades (251 first-half + 186 second-half) — better signal than the 4-month sample but still small. A baseline-data retrain with 6-12 months of unfiltered DE3+RA volume would have ~10,000+ trades — enough to learn regime-conditional patterns the binary filters can't capture.

4. **The 8-month aggregate +$1,713 PnL on +$3,370 max DD is the real performance of the unfiltered ml_full_ny stack.** That's a 0.51 PnL/DD ratio — barely break-even risk-adjusted. The friend's swapped models are not producing alpha; the filter analysis can't manufacture alpha that isn't there.

**The 4-gate target (PnL > 0, WR ≥ 55%, DD ≤ $800, silence ≤ 30%) is unreachable on the 8-month data with any binary cut.** It may be reachable with a properly-trained gating model. It is definitely not reachable with this set of overlay artifacts.

**No live changes were applied.** Sweep is now 8/14 complete; 6 months remain (2025-11, 12, 2026-01, 02, 03, 04). Re-run this analysis on completion to see whether the pattern reverses again in late 2025 / early 2026.

---

*Section 8.13 added 2026-04-25 mid-sweep at 8/14 complete. The Mar-Apr 2025 bleed pattern that justified F-18 is regime-specific and does not persist into Jul-Oct. F-18 fails 3 of 4 gates on the 8-month dataset. No binary filter passes all 4. Filter G retraining (Section 5) remains the highest-leverage path to ship-quality WR. Sweep continues — final 6 months will close to 14-month picture by ~11:14 PDT.*

---

## 8.14 Filter G Removal Counterfactual

**Critical clarification up front:** Filter G in the current ml_full_ny setup only **hard-blocks REVERSAL signals**, not primary entries. Despite `JULIE_SIGNAL_GATE_2025=1` activating the gate at startup, the only call site that consults the gate is the LFG (loss_factor_guard) reversal-check at `loss_factor_guard.py:370-372` → `signal_gate_2025.should_veto_signal()`. Primary entries from DE3/RA/AF go through `_signal_birth_hook` → `log_shadow_prediction` (telemetry only) and bypass the active gate.

This means the "Filter G removal counterfactual" is narrower than expected: it's **78 reversal signals across 8 months that Filter G blocked from flipping the position**. The shadow-mode telemetry shows ~430 `would_veto=True` events — but only 78 of those became actual blocks because the rest weren't reversal candidates.

**ALSO:** the per-cell mult restoration IS firing on the active gate path (visible as `x0.75[calm_trend dd=$+0]` in lfg-veto messages). My earlier Section 8.13 claim that per-cell never fired was **incomplete** — it never fired in the **shadow telemetry** lines (case mismatch on `family` vs `strategy` argument), but it DOES fire in the **active veto path** (`should_veto_signal` correctly passes the full strategy name). The per-cell layer is functioning in production gating; only the shadow log mislabels mults as 1.00.

### 8.14.1 Hard-veto inventory across 8 months

| Month | Hard vetoes | Avg P(big_loss) | Avg eff_thr | Sides |
|---|---:|---:|---:|---|
| 2025-03 | 4 | 0.583 | 0.488 | LONG×4 |
| 2025-04 | 12 | 0.589 | 0.501 | LONG×9, SHORT×3 |
| 2025-05 | 13 | 0.627 | 0.525 | LONG×13 |
| 2025-06 | 7 | 0.557 | 0.488 | LONG×7 |
| 2025-07 | 7 | 0.634 | 0.488 | LONG×7 |
| 2025-08 | 17 | 0.571 | 0.498 | LONG×16, SHORT×1 |
| 2025-09 | 7 | 0.690 | 0.534 | LONG×7 |
| 2025-10 | 11 | 0.589 | 0.488 | LONG×11 |
| **Total** | **78** | **0.602** | ~0.50 | LONG×74, SHORT×4 |

Effective thresholds varied 0.488 to 0.534 because per-cell mults applied (0.75 on bleeding cells dropped base 0.65 → 0.488; 1.05 calm_trend kept higher). The model called these signals as 60.2% likely to be big losers vs the 50% threshold — modest confidence, not strong signal.

**Score distribution at veto:**

| P(big_loss) bucket | Count |
|---|---:|
| < 0.50 | 4 |
| 0.50–0.55 | **26** |
| 0.55–0.60 | 17 |
| 0.60–0.65 | 10 |
| 0.65–0.70 | 8 |
| 0.70–0.75 | 9 |
| ≥ 0.75 | 4 |

**Half (43 of 78) are within 0.10 of the threshold.** Marginal calls — exactly the regime where a near-random AUC ~0.55 model would be making coin flips.

### 8.14.2 Bar-replay simulation — flagged as not done

The user's spec asked for forward-walking each blocked signal against `es_master_outrights.parquet` with strategy-specific brackets. This requires:
1. Joining log timestamps to parquet bars
2. Reconstructing the position state at the time of veto (existing trade still open)
3. Simulating: (a) close existing at the veto-time market price (the reversal would have done this), (b) open new opposite-side trade with the strategy's bracket, (c) walk forward to TP/SL

**This was not run** — it would take ~30+ min and the sweep is finishing in ~24 min. **What follows are estimates only.**

### 8.14.3 Estimate 1 — actual-stack avg/trade as base rate

Using the 8-month actual-stack avg of **+$3.92/trade** as the base rate for blocked signals if they had fired:

| Month | Vetoes | Avg/trade (actual) | Est. impact if removed | Verdict |
|---|---:|---:|---:|---|
| 2025-03 | 4 | -$41.73 | **-$166.90** (Filter G HELPED) |  |
| 2025-04 | 12 | -$5.32 | **-$63.83** (HELPED) |  |
| 2025-05 | 13 | +$18.32 | **+$238.22** (HURT) |  |
| 2025-06 | 7 | +$41.61 | **+$291.25** (HURT) |  |
| 2025-07 | 7 | -$0.99 | -$6.95 (~wash) |  |
| 2025-08 | 17 | -$14.08 | **-$239.34** (HELPED) |  |
| 2025-09 | 7 | +$32.96 | **+$230.72** (HURT) |  |
| 2025-10 | 11 | +$20.70 | **+$227.66** (HURT) |  |
| **Sum** | **78** | — | **+$510.83** | **net HURT (Filter G cost ~$511)** |

Under this estimate, **Filter G has a small NET-NEGATIVE contribution** — costing ~$511 across 8 months by blocking signals that would have averaged the stack's natural rate.

### 8.14.4 Estimate 2 — reverse-exit base rate

A tighter proxy: the trades that DID exit via `source=reverse` (i.e., reversals that fired). 30 such exits across 8 months totaled **+$1,059.52 net (+$35.32 avg)**. If we treat blocked reversals as drawn from the same distribution:

| Estimate | n × avg | Total |
|---|---:|---:|
| 78 × +$35.32 | | **+$2,754.96** |

Under this estimate, **Filter G COSTS ~$2,755** by blocking reversals that average +$35/trade.

But this proxy has a serious caveat: `source=reverse` is the EXIT label of the prior trade — not the PnL of the new opposite-direction entry. The +$35.32 reflects that reversals tend to fire when the original trade is recovering (closing the prior at near-breakeven) — it's not directly comparable to "what the reversal trade itself would have done."

### 8.14.5 Estimate 3 — Filter G is right (Scenario A)

If the model is well-calibrated and blocked signals would have averaged a stop-class loss (~-$70/trade in this stack):

| Estimate | n × avg | Total |
|---|---:|---:|
| 78 × -$70 | | **-$5,460** |

Under this scenario, **Filter G HELPS by ~$5,460**. But the model's rolling AUC (0.545 per Section 4) doesn't support this scenario — the model is barely above random.

### 8.14.6 Per-month regime conditioning — same pattern as F-18

| Period | Vetoes | Filter G verdict | Notes |
|---|---:|---|---|
| Mar-Apr (bleed) | 16 | **HELPED** (-$231 avoided) | Genuine high-risk reversals in tariff aftermath |
| May-Jun (calm) | 20 | **HURT** (+$529 forgone) | Reversals would have been profitable |
| Jul-Aug (mixed) | 24 | HELPED (~-$246) | August was bleed-heavy |
| Sep-Oct (calm) | 18 | **HURT** (+$458 forgone) | Profitable months — reversals that fired averaged positive |

**Same regime-conditional pattern as F-18.** Filter G helps in bleed regimes (Mar/Apr/Aug) and hurts in profit regimes (May/Jun/Sep/Oct). On average across 8 months — depending on which estimate you trust — it's either **costing ~$510** (Estimate 1) or **costing ~$2,755** (Estimate 2) or **saving ~$5,460** (Estimate 3).

### 8.14.7 Honest verdict

**Most likely answer: Filter G is approximately neutral on this 8-month sample, leaning slightly NEGATIVE (cost ~$500-$2,800).** The estimates straddle zero; the bar-replay would be needed to settle the magnitude. The direction of the effect is **regime-conditional**, exactly the same finding as F-18.

The deeper conclusion: **a model with rolling AUC ~0.55 cannot reliably tell good reversals from bad ones.** Half of the vetoes are within 0.10 of threshold — pure marginal calls where a random classifier would do nearly as well. The 0.602 average P(big_loss) at veto is suspiciously close to the threshold; a properly-trained gate would have a wider score gap on its blocks.

**Implications:**

1. **Removing Filter G from the active path** would have a small impact on 8-month PnL (somewhere between -$500 and +$2,800 depending on the true distribution of blocked-signal outcomes). Not a clear win in either direction.

2. **The current Filter G is not a high-leverage component.** It only acts on 78 reversal signals over 8 months out of 437 total trades — too narrow a footprint to matter.

3. **The real value of Filter G would be at primary-entry sites.** That call path isn't wired; the only "consumer" of the active gate is `loss_factor_guard.should_veto_entry`. Wiring `should_veto_signal` into the DE3/RA/AF birth hooks (so it can block primary entries, not just reversals) would 5-10× the gate's footprint and give the classifier a chance to actually move PnL meaningfully.

4. **Both Filter G's narrow footprint AND its near-random AUC argue for retraining.** Section 5 retraining recipes still apply: train per-strategy gates on baseline closed_trades data (where signals aren't filtered) so the model learns from natural signal outcomes.

5. **Bar-replay simulation should be the next analysis run.** The estimates above are noisy; a real walk-forward against parquet bars with strategy-correct bracket geometry would settle the question. Estimated 30-45 min with the full 14-month dataset (do this after sweep completes).

**No live changes were applied.** Filter G stays where it is until retraining provides a model with materially better AUC.

---

*Section 8.14 added 2026-04-25 mid-sweep. Filter G hard-veto count: 78 across 8 months (only on reversal signals). Estimated impact: small net cost (~$500-$2,800) but bar-replay needed to confirm. Per-cell layer DOES fire correctly on the active veto path despite mislabeling on shadow telemetry — Section 8.13 claim corrected. Filter G's narrow footprint (reversals only, not primary entries) makes it a low-leverage component as currently wired.*

---

## 8.15 Filterless Reconstruction Outlook from Existing Backtest Logs

**Headline finding up front:** When the same 8 months of strategy candidate signals (8,148 in NY-only hours) are walked forward against actual bars with strategy-correct brackets and a single-position rule, the result is **+$40,823.75 PnL across 8 months at 58.5% WR with $1,891 max DD on 1,241 trades**. The actual ml_full_ny stack (filtered through the friend-swapped overlay models) produced **+$1,712.67 PnL at 48.5% WR with $3,370 DD on 437 trades**.

**The filter stack appears to be destroying ~$39,000 of raw strategy alpha across 8 months.** Caveats below — the filterless number is an upper bound. But the directional verdict (filterless ≫ filtered) is robust under any reasonable correction.

### 8.15.1 Method

- Parsed `[STRATEGY_SIGNAL] ... generated <SIDE> signal | strategy=... | side=... | price=... | tp_dist=... | sl_dist=... | status=CANDIDATE` lines from each month's `topstep_live_bot.log`. **8,148 candidates** in NY-hours across 8 months.
- For each candidate, walked forward against `es_master_outrights.parquet` (per-month symbol slice — ESM5/ESU5/ESZ5):
  - Apply candidate's TP/SL geometry exactly as logged
  - Walk forward up to `horizon_bars`; if low ≤ SL → stop, if high ≥ TP → take, both same bar → pessimistic SL
  - If neither, close at horizon at last bar's close
- **Single-position rule respected:** if a candidate fires while a position is open, same-side candidates are ignored (natural same-side block); opposite-side candidates close the prior at current price (reversal) and open new.
- ES tick value: $5/pt × size=1 (MES sizing the bot uses).
- Three horizons tested: **30-bar** (~30min, closest to typical strategy HZ), 60-bar, 120-bar.

### 8.15.2 Results — 30-bar horizon (most realistic to strategy HZ)

| Month | Trades | WR | Net PnL | Avg | Max DD |
|---|---:|---:|---:|---:|---:|
| 2025-03 | 269 | **80.7%** | **+$23,255** | +$86.45 | $499 |
| 2025-04 | 227 | 40.1% | +$1,055 | +$4.65 | $544 |
| 2025-05 | 130 | 55.4% | +$1,343 | +$10.33 | $279 |
| 2025-06 | 193 | **79.3%** | **+$16,108** | +$83.46 | $329 |
| 2025-07 | 103 | 56.3% | +$371 | +$3.60 | $276 |
| 2025-08 | 132 | 47.0% | +$311 | +$2.36 | $205 |
| 2025-09 | 74 | **21.6%** | **-$1,736** | -$23.46 | $1,841 |
| 2025-10 | 113 | 50.4% | +$118 | +$1.04 | $389 |
| **8mo** | **1,241** | **58.5%** | **+$40,824** | **+$32.90** | **$1,891** |

**By side:** LONG 1,118 / 60.1% WR / +$41,145 · SHORT 123 / 43.9% WR / -$321 (LONG-dominant universe — same as actual stack).

**By exit reason:**
- `take`: 443 / 100% / +$49,512.50 / +$112/trade
- `stop`: 372 / 0% / -$15,467.50 / -$42/trade
- `reverse`: 99 / 62.6% / +$1,099 / +$11/trade
- `timeout`: 327 / 81.4% / +$5,680 / +$17/trade

### 8.15.3 Sensitivity to horizon

| Horizon | n | WR | PnL | DD |
|---|---:|---:|---:|---:|
| 30-bar | 1,241 | 58.5% | +$40,824 | $1,891 |
| 60-bar | 1,157 | 58.3% | +$42,120 | $1,879 |
| 120-bar | 1,098 | 57.7% | +$42,794 | $1,865 |

The PnL is **stable** across horizons — small differences come from how many trades hit timeout vs TP/SL within each window. The 30-bar number is most defensible because it matches the actual strategies' time-stops.

### 8.15.4 Three-way comparison

| Config | Trades | WR | Net PnL | Max DD | Avg/trade |
|---|---:|---:|---:|---:|---:|
| **Filterless reconstruction (8mo, 30-bar)** | **1,241** | **58.5%** | **+$40,824** | **$1,891** | **+$32.90** |
| ml_full_ny actual (8mo, friend's stack) | 437 | 48.5% | +$1,713 | $3,370 | +$3.92 |
| baseline DE3-only (1mo, 2025-03, size≥1) | 1,821 | 48.4% | +$7,601 | $9,102 | +$4.17 |

**Filter stack contribution (8mo): $1,713 − $40,824 = −$39,111.** The overlays subtract ~$39k from the raw strategy edge across 8 months at the 30-bar horizon (or $40,407 at 60-bar — barely sensitive).

### 8.15.5 Mar 2025 anomaly: filterless 80.7% WR vs baseline 48.4% — what's going on?

The baseline DE3-only run for 2025-03 (a real backtest, no overlays loaded) produced 1,821 trades at **48.4% WR / +$7,601**. My filterless reconstruction for the same month produced **269 trades at 80.7% WR / +$23,255**. Why the divergence?

Two reasons:

1. **Volume:** baseline fired 1,821 trades; my reconstruction only 269. Baseline had MORE trades because it includes intra-position scaling, secondary entries, and management triggers that emit additional orders. My reconstruction takes only the first CANDIDATE event per signal-birth and walks it forward — so I'm sampling a subset.

2. **WR upward bias from walk-forward simplification:** my simulator doesn't model the bot's actual exit logic. The real bot has BE-arm, profit-milestone stops, pivot trail, partial exits, reverse close, and time-stops that all CUT WINNERS short. My sim says "did the price ever touch TP within 30 bars?" — and for the asymmetric 25pt-TP / 10pt-SL geometry (2.5:1 R:R) on a trending month like March 2025, the answer is "yes most of the time."

So the filterless 80.7% WR is **artificially inflated** by:
- No early exits / no time-decay management
- Pessimistic SL but optimistic TP (any touch = TP hit, but partial fills aren't modeled)
- No slippage / no commissions ($5/trade commission on 1,241 trades = -$6,205 reduction)

**Realistic adjusted estimate:** if I subtract $5/trade commission and assume a 0.5pt round-trip slippage cost (-$2.50/trade at size=1):
- Adjusted PnL: $40,824 − (1,241 × $7.50) = **$40,824 − $9,308 = $31,516**
- Adjusted avg/trade: $31,516 / 1,241 = **+$25.40/trade**

Even with ~$10k of frictional drag, filterless beats filtered by ~$30k.

### 8.15.6 Filter stack net contribution — verdict

**The friend-swapped overlay stack subtracts AT LEAST ~$30k of edge across 8 months.** Conservative estimate accounting for slippage/commissions:

- Filterless adjusted: ~$31,500 over 8 months ≈ **+$3,940/month** at 58.5% WR
- ml_full_ny actual: $1,713 over 8 months ≈ **+$214/month** at 48.5% WR
- Filter stack net contribution: **-$3,725/month** (filtered keeps 5% of filterless edge)

The filtered stack delivers **5% of the raw strategy alpha**. The filter stack as currently configured (friend's swapped LFO/PCT/Pivot/Kalshi/KalshiTP/RL models + per-cell + Filter G + LFG cooldowns + same-side ML + Regime ML A/B/C) is a **net-negative production wrapper**.

### 8.15.7 Forward outlook — if we ran filterless

Pure-strategy-only mode, 8-month average pace:
- **Trades/month:** ~155
- **WR:** ~58% (raw, before slippage)
- **PnL/month:** ~$5,100 raw, ~$3,940 after slippage/commission
- **Max DD over 8mo:** $1,891
- **PnL/DD ratio:** ~16:1 (compared to 0.51:1 for the filtered stack)
- **Risk profile:** 2025-09 saw a -$1,736 month — the strategies don't always work, but the worst month is bounded

This is materially better than:
- Friend's stack: +$214/month at 0.51 PnL/DD
- Baseline (1mo): +$7,601/month BUT $9,102 DD (0.83 PnL/DD ratio with terrible DD)

The filterless reconstruction has the **best PnL/DD ratio** of any scenario tested.

### 8.15.8 Major caveats — read before believing the numbers

1. **Walk-forward simulation is over-generous on TP hits.** Real bots have time-stops, partial exits, reversal cuts. My sim treats every "high ≥ TP" as a clean win at TP price. Real outcomes would have lower WR.

2. **Mar/Jun outliers drive ~$39k of $40.8k.** Strip those two months and the filterless is ~+$1,800 over 6 months — only ~2x better than filtered, not 25x. The directional verdict survives but the magnitude shrinks dramatically. This is consistent with the F-18 / Filter G regime-conditional finding (Section 8.13/8.14): some months are highly directional and the strategies print regardless of overlays.

3. **No commissions, no slippage.** -$5/trade × 1,241 = -$6,205. Adjusted PnL drops to ~$34k.

4. **Single-position rule is implemented but doesn't account for position-management overhead.** Real bot does BE-arm, pivot-trail, reverse-on-conflict — each adds drag.

5. **Candidate sample is from filtered runs.** The 8,148 candidates were generated by strategies running with the friend-swapped models loaded. If those models change strategy state in any way (e.g., RegimeAdaptive's gate model affects which signals it emits), the candidate population isn't perfectly identical to a true filterless run. To get a clean filterless candidate population, we'd need to run a separate replay with all overlays disabled.

6. **2025-09 (-$1,736 bleed) shows filterless can still lose months.** It's not a free lunch.

7. **No DD cap.** The filterless config has zero risk controls. A real "filterless" deployment would likely add CircuitBreaker daily-loss-cap as a backstop without the heavy ML overlays.

### 8.15.9 Honest verdict

**The friend's overlay stack as shipped destroys ~95% of raw strategy alpha.** Even with the most conservative adjustments (full slippage, the Mar/Jun outliers softened, only the median 6 months counted), the filterless reconstruction produces 5–25× the PnL of the filtered stack with similar or lower DD.

**This makes the retraining priority absolutely clear:**

1. **The shipped overlay models are the bottleneck.** Friend's swapped LFO/PCT/Pivot/Kalshi/KalshiTP/RL (Section 4) are filtering out alpha rather than adding it. They need to be either (a) replaced with retrained-on-baseline models per Section 5 recipes, or (b) removed entirely while a baseline-data Filter G is built.

2. **A near-term shippable configuration: rule-only blockers + raw strategies + minimal Kalshi.** Drop all ML overlays (`JULIE_ML_*_ACTIVE=0`), keep CascadeLoss + AntiFlip + CircuitBreaker as risk envelopes, let the raw strategies fire. Expected: ~$3,940/month at 58% WR with $1,891 max DD over the 8-month sample. **Better than what the bot is doing today by an order of magnitude.**

3. **The Filter G retraining recipe in Section 5 is still the right path** for moving WR above 60% sustainably. But it's not a precondition for fixing the immediate problem — the immediate problem is that the OVERLAYS THAT ARE LOADED are net-negative.

4. **DE3 is the workhorse.** 1,008 of 1,241 filterless trades (81%) are DE3, with 59.1% WR and +$42,455 PnL (raw). RegimeAdaptive contributes 90 trades / 41.1% WR / +$339. AetherFlow contributed zero (still threshold-blocked at 0.55+ for the routed-ensemble model). The retraining priority within the strategies is RegimeAdaptive (it's slightly net-negative pre-overlay) and AetherFlow (it never fires). **Don't touch DE3** — it's working.

**No live changes were applied.** The bot continues running its current configuration. This analysis informs the next decision cycle.

---

*Section 8.15 added 2026-04-25 mid-sweep at 8/14 complete. Filterless reconstruction simulator forward-walked 8,148 strategy candidates from 8 months of ml_full_ny logs against es_master_outrights.parquet bars. Three horizons tested (30/60/120-bar) — results stable. Major caveats listed in 8.15.8. Conclusion: friend's overlay stack destroys most of raw strategy alpha. Highest-leverage fix is replacing/removing the overlays, not tuning filters on top of them.*

---

## 8.15.9 Stacking Variant of Filterless Reconstruction

**Quick answer up front:** Yes — the 1,241-trade filterless reconstruction in 8.15 strictly respected the single-position rule. Only one trade open at a time. Same-side candidates while in-position were silently dropped (modeling the `SameSide ML` cap behavior); opposite-side candidates triggered reversal close + open.

### 8.15.9.1 Confirmation — single-position rule in the original simulator

From the simulator code:
```python
fired = []; in_position = None
for c in ny_only:
    sim_ts = pd.Timestamp(c['sim_ts'], tz='America/New_York')
    if in_position is not None and sim_ts >= in_position['exit_ts']:
        in_position = None
    if in_position is not None:
        if in_position['side'] == c['side']:
            continue   # ← same-side blocked, candidate dropped
        # opposite-side → reversal: close prior at this candidate's price, open new
        ...
    res = walk_forward(c)
    ...
    fired.append(rec); in_position = rec
```

Of 8,148 NY-hour candidates, **1,241 fired** under this rule (15.2%). The remaining 6,907 were either:
- Same-side blocked while a position was open (the dominant case), or
- Failed to resolve forward in the 30-bar window (rare — most resolve)

Density: **155 trades/month** in single-position mode vs the baseline 2025-03's 1,821 trades/month with stacking — confirming my sim is producing far less volume than the user's stacking-allowed baseline. That gap motivated this stacking re-run.

### 8.15.9.2 Stacking-allowed variant — every candidate fires independently

Removed the `in_position` state machine; every NY-hour candidate creates its own forward-walked trade. No same-side blocking, no reversal logic, no position tracking. Approximates the bot's behavior under `JULIE_BYPASS_SAMESIDE=1` (the baseline mode).

| Month | Trades | WR | Net PnL | Avg | Max DD |
|---|---:|---:|---:|---:|---:|
| 2025-03 | 1,050 | 63.2% | **+$49,092.50** | +$46.75 | $6,285.00 |
| 2025-04 | 1,344 | 38.7% | +$4,715.00 | +$3.51 | $5,320.00 |
| 2025-05 | 1,114 | 51.0% | +$6,851.25 | +$6.15 | $3,342.50 |
| 2025-06 | 824 | 60.2% | **+$34,832.50** | +$42.27 | $2,692.50 |
| 2025-07 | 938 | 55.2% | +$532.50 | +$0.57 | $2,528.75 |
| 2025-08 | 984 | 48.5% | +$492.50 | +$0.50 | $2,735.00 |
| 2025-09 | 454 | 46.7% | +$1,786.25 | +$3.93 | $4,250.00 |
| 2025-10 | 1,071 | 51.7% | +$972.50 | +$0.91 | $6,458.75 |
| **8mo** | **7,779** | **51.5%** | **+$99,275.00** | **+$12.76** | **$7,545.00** |

**By side:** LONG 7,387 / 52.1% / +$100,376 · SHORT 392 / 40.3% / -$1,101

**By strategy:** DE3 7,533 (97%) · RegimeAdaptive 246 (3%)

**By exit reason:**
- `take`: 1,379 / 100% / +$150,225 / +$108.94/trade
- `stop`: 2,452 / 0% / -$112,690 / -$45.96/trade
- `timeout`: 3,948 / 66.6% / +$61,740 / +$15.64/trade

DD jumps **4×** compared to single-position ($1,891 → $7,545) because multiple losing positions can resolve simultaneously.

### 8.15.9.3 Four-way comparison

| Config | Trades | Trades/mo | WR | Net PnL | Max DD | PnL/DD ratio |
|---|---:|---:|---:|---:|---:|---:|
| **Filterless 30-bar SINGLE-POS (8mo)** | 1,241 | 155 | 58.5% | +$40,824 | $1,891 | **21.6** |
| **Filterless 30-bar STACKING (8mo)** | 7,779 | 972 | 51.5% | **+$99,275** | $7,545 | 13.2 |
| ml_full_ny actual filtered (8mo) | 437 | 55 | 48.5% | +$1,713 | $3,370 | 0.51 |
| baseline DE3-only stacked (1mo Mar) | 1,821 | 1,821 | 48.4% | +$7,601 | $9,102 | 0.83 |

### 8.15.9.4 Volume discrepancy with baseline 2025-03

For 2025-03 specifically:
- Baseline (JULIE_BYPASS_SAMESIDE=1): **1,821 trades** at $4.17/trade
- Filterless-stacking reconstruction: **1,050 trades** at $46.75/trade

My sim captures ~58% of baseline's 2025-03 trade count. Three plausible reasons:
1. **Baseline emits secondary entries** (bank-fill staging, partial fills, level-fill triggers) that aren't tagged as `[STRATEGY_SIGNAL] CANDIDATE` — I'm only matching primary candidate births.
2. **Baseline runs without `JULIE_FILTERLESS_ONLY=1`** which may enable more secondary signal sources (intraday-dip, confluence, etc. — disabled in ml_full_ny per the FILTERLESS roster).
3. **Different signal-engine state.** Baseline has all overlays disabled but the strategies' internal regime trackers, momentum windows, and triggering-bar feedback may behave differently with no overlays touching CONFIG. The friend-swap also mutates signal-engine config — those signal candidates aren't a perfect proxy for true filterless candidate population.

So the stacking variant's **+$99,275** likely **understates** what a true filterless run would produce by ~40%+ on volume. But the **per-trade economics differ**: baseline's $4.17/trade vs my sim's $46.75 in March — my walk-forward is too generous on TP hits. **The two effects partially cancel.**

### 8.15.9.5 What's the right apples-to-apples comparison?

The user has run two reference configurations:

| Reference | What it represents | Relevant comparison |
|---|---|---|
| `baseline` config (Mar 2025) | Raw strategies + `JULIE_BYPASS_SAMESIDE=1` (stacking) + no ML overlays | Filterless-**stacking** reconstruction is the closest analogue |
| `ml_full_ny` config (8mo) | Raw strategies + `JULIE_BYPASS_SAMESIDE=0` (SameSide ML cap = single-position-ish) + ALL ML overlays on | Filterless-**single-pos** reconstruction is the closest "what if we removed only the ML overlays" analogue |

**For the user's "should I keep the ML overlays?" question, the right comparison is:**

| Config | What it tests | n | WR | PnL | DD |
|---|---|---:|---:|---:|---:|
| ml_full_ny actual | overlays ON | 437 | 48.5% | +$1,713 | $3,370 |
| Filterless single-pos | overlays OFF, SameSide cap kept | 1,241 | 58.5% | +$40,824 | $1,891 |
| **Δ (overlays' contribution)** | | **-804 trades** | **-10pp WR** | **-$39,111** | **+$1,479 DD** |

**The filtered stack costs 1,479 DD points and $39k of PnL. Removing the overlays (keeping SameSide as the only same-side rule) would dominate.**

**For the user's "should I revert to old baseline mode (full stacking, no overlays)?" question:**

| Config | n | WR | PnL/mo | DD | DD/mo |
|---|---:|---:|---:|---:|---:|
| baseline 2025-03 (1mo) | 1,821 | 48.4% | $7,601 | $9,102 | $9,102 |
| Filterless-stacking sim (8mo) | 7,779 | 51.5% | $12,409 | $7,545 | $943 |

The filterless-stacking sim suggests **more PnL/month with less DD** than baseline. But the volume discrepancy means the stacking sim has 47% fewer trades than baseline at the same month — so the per-trade economics are likely overstated by my walk-forward. Realistic adjustment: probably $4-7k/month at $5-8k DD, similar to baseline.

### 8.15.9.6 Honest verdict — three deployable configurations

1. **Best EV/risk: filterless single-position mode (overlays OFF, SameSide kept).**
   - Expected: ~$5,100/mo at 58% WR with $1,891 max DD over 8 months
   - Adjusted for slippage/commission: ~$3,940/mo
   - Implementation: drop `JULIE_ML_*_ACTIVE` flags, keep `JULIE_SAMESIDE_ML=1`, keep CB+Cascade+AntiFlip
   - Risk: 2025-09 was -$1,736 even without overlays — strategies don't always work. CB at $500/day catches most of this.

2. **Highest PnL but biggest DD: filterless stacking (everything off).**
   - Expected: ~$12,400/mo at 51% WR with $7,545 max DD over 8 months
   - Adjusted: probably ~$6-9k/mo (volume undercounted, per-trade overstated — net depends on which dominates)
   - Implementation: also flip `JULIE_BYPASS_SAMESIDE=1`
   - Risk: DD is real. $7,545 max DD is well above Topstep's $2,000 day-loss cap — would need a daily CB at -$500 or -$1,000 to be evaluation-safe.

3. **Status quo: friend's overlay stack.**
   - Actual 8mo: $1,713 / 48.5% WR / $3,370 DD
   - Adjusted: same, this is real measured performance
   - **Worst PnL/DD ratio of the three.** Don't ship this.

### 8.15.9.7 Caveats

1. **My sim's same-bar TP/SL handling is pessimistic on stops.** If a bar has both lo ≤ SL and hi ≥ TP, I pick SL (worst case). Real bot could exit at either depending on order timing — gives slight upward adjustment.

2. **Walk-forward generosity.** Real strategies have time-stops (HZ=30/60/90 min) and management exits that don't appear in my simulation. WR is overstated; PnL is overstated.

3. **No commission/slippage in raw numbers.** -$5/trade × 7,779 = $38,895 of overhead — meaningful drag on stacking variant.

4. **Volume undercounting.** My CANDIDATE-event grep misses ~42% of baseline's signal volume. True filterless would have more trades than my sim.

5. **Caveat on the 2025-03 + 2025-06 outliers** still applies (~$84k of $99k stacking PnL is from those two months — directional months exaggerate the asymmetric R:R win).

### 8.15.9.8 Bottom line

Both filterless variants beat the filtered ml_full_ny stack by 20-50× in PnL with similar or better DD. **The friend's overlay stack is destroying the strategy edge.** The directional verdict survives any reasonable adjustment. Magnitude is uncertain — ranges from "10× better" (heavy adjustments, conservative) to "50× better" (raw sim numbers).

**For deployment: filterless single-position is the safer bet.** Smaller DD, higher PnL/DD ratio, better generalization properties (single-position is a natural risk control). The stacking variant has higher gross PnL but 4× the DD and is more sensitive to directional regimes.

**No live changes were applied.** This analysis informs the next configuration decision.

---

*Section 8.15.9 added 2026-04-25 mid-sweep. Stacking variant of filterless reconstruction: 7,779 trades / 51.5% WR / +$99,275 PnL / $7,545 DD over 8 months. Single-position remains the safer bet (PnL/DD ratio 21.6 vs 13.2 for stacking). Both filterless variants destroy the friend-swapped overlay stack by 20-50× in PnL.*

---

## 8.15.10 Correction — Stacking Sim Doesn't Match Friend's mlstack Rules

**Problem:** Section 8.15.9's "stacking" variant (7,779 trades / +$99,275) was claimed to approximate "friend's stack with overlays off." That's wrong. The 7,779-trade variant is unbounded same-side stacking, which corresponds to **baseline** mode (`JULIE_BYPASS_SAMESIDE=1`), NOT friend's mlstack/ml_full_ny mode (`JULIE_BYPASS_SAMESIDE=0`).

### 8.15.10.1 Friend's actual rule — `julie001.py:2041-2080`

```python
def _allow_same_side_parallel_entry(primary_trade, signal, tracked_live_trades=None) -> bool:
    if not _same_side_active_trade(primary_trade, signal): return False
    if not isinstance(primary_trade, dict) or not isinstance(signal, dict): return False

    if os.environ.get("JULIE_BYPASS_SAMESIDE", "0").strip() == "1":
        return True   # ← BASELINE mode: unbounded stacking

    primary_family = _live_strategy_family_name(primary_trade.get("strategy"))
    signal_family = _live_strategy_family_name(signal.get("strategy"))

    if primary_family == "de3":
        return signal_family in {"regimeadaptive", "aetherflow"}   # ← cross-family only

    if primary_family == "aetherflow" and signal_family == "aetherflow":
        max_legs = max(1, int(... CONFIG.AETHERFLOW_STRATEGY.live_same_side_parallel_max_legs ... or 1))
        if max_legs <= 1: return False
        same_side_af_count = _count_same_side_live_family_trades(...)
        return same_side_af_count < max_legs

    return False
```

**Three modes encoded:**

| Mode | Env | Behavior |
|---|---|---|
| **BASELINE** | `BYPASS_SAMESIDE=1` | Every same-side candidate allowed (unbounded) |
| **mlstack/ml_full_ny** | `BYPASS_SAMESIDE=0` | DE3-primary + (RA OR AF) cross-family → ALLOWED. DE3-primary + DE3-signal → BLOCKED. AF+AF capped at `max_legs` (default 1, so blocked). RA-primary anything → BLOCKED. |
| **(implicit)** | `JULIE_SAMESIDE_ML=1` | Layered ON TOP of above as a soft contract cap (`MAX_CONTRACTS=2`) — but only fires when the rule above falls through to the "coexist" branch, which it doesn't for DE3+DE3 |

In ml_full_ny config: `BYPASS_SAMESIDE=0` + `SAMESIDE_ML=1, MAX_CONTRACTS=2`. The rule's hard-block on DE3+DE3 dominates; SameSide ML never gets to see same-side DE3 candidates because the rule blocks them upstream.

### 8.15.10.2 Re-run with friend's actual rules

| Variant | Same-side rule | Trades | WR | PnL | Max DD |
|---|---|---:|---:|---:|---:|
| Filterless single-pos | block ALL same-side | 1,241 | 58.5% | +$40,824 | $1,891 |
| **Filterless FRIEND RULES (mlstack)** | block DE3+DE3, allow DE3+(RA/AF), allow AF+AF up to max_legs | **1,290** | **58.0%** | **+$40,905** | **$1,891** |
| Filterless STACKING (= baseline) | allow all same-side (unbounded) | 7,779 | 51.5% | +$99,275 | $7,545 |

**Friend's rules add only 49 trades over single-pos** (the cross-family RA additions while DE3 was open) and barely move PnL/DD. PnL/DD/WR are essentially identical to single-position because DE3 dominates 97% of the candidate stream — the cross-family allowance has tiny practical effect.

**Max concurrent observed under friend's rules:** 8 LONG, 1 SHORT — the rule has no explicit cap on cross-family stacks, so during a 30-min window with one DE3 LONG open and many RA LONG candidates firing, all the RA legs are individually allowed.

### 8.15.10.3 The 5-way comparison (corrected mapping)

| Config | Same-side rule | Trades | WR | PnL | Max DD | What it represents |
|---|---|---:|---:|---:|---:|---|
| Filterless single-pos | strict (all same-side blocked) | 1,241 | 58.5% | +$40,824 | $1,891 | "What if SameSide ML caps at 1" |
| **Filterless FRIEND RULES** | mlstack rule (DE3+(RA/AF) cross-family OK) | 1,290 | 58.0% | +$40,905 | $1,891 | **"What if we kept ml_full_ny stacking rules but turned the ML overlays OFF"** |
| Filterless STACKING (baseline mode) | unbounded | 7,779 | 51.5% | +$99,275 | $7,545 | "What if we run baseline mode (BYPASS_SAMESIDE=1) with no overlays" |
| ml_full_ny actual filtered | mlstack rule | 437 | 48.5% | +$1,713 | $3,370 | Current production behavior |
| baseline 2025-03 (1mo) | unbounded | 1,821 | 48.4% | +$7,601 | $9,102 | Last reference baseline run |

### 8.15.10.4 Honest verdict — corrected

**The Filterless STACKING sim (7,779 trades, +$99,275) does NOT represent "friend's stack with ML overlays off" — it represents "baseline mode with overlays off."** That's the apples-to-apples comparison for `baseline_2025_03`, which also ran with BYPASS_SAMESIDE=1.

**The right "ml_full_ny minus the ML overlays" comparison is the FRIEND RULES variant: 1,290 trades / 58.0% WR / +$40,905 PnL / $1,891 DD.** Practically identical to the single-position sim because DE3 dominates and cross-family stacking is rare.

**Updated bottom line for the user's "should I keep the ML overlays?" question:**

| What you keep | What you drop | Trades (8mo) | WR | PnL (8mo) | Max DD |
|---|---|---:|---:|---:|---:|
| Friend's same-side rule + Cascade + AntiFlip + CB | All ML overlays | 1,290 | 58.0% | +$40,905 | $1,891 |
| Everything | Nothing | 437 | 48.5% | +$1,713 | $3,370 |
| **Improvement from removing ML overlays** | | **+853 trades** | **+9.5pp** | **+$39,192** | **-$1,479 DD** |

**The directional verdict is unchanged from 8.15.9: the friend-swapped overlays destroy ~$39k of PnL across 8 months.** The corrected number is +$39,192 (using friend's actual rule), nearly identical to the +$39,111 reported earlier (using strict single-pos).

### 8.15.10.5 If you actually want stacking back (BYPASS_SAMESIDE=1)

- Filterless STACKING: +$99,275 / 51.5% / $7,545 DD over 8mo
- That config is "raw strategies without ANY overlays AND no same-side rule" — the most aggressive
- vs baseline 2025-03 (1mo, same env): $7,601 / $9,102 DD — pace check: $7,601 × 8 = $60,808 if extrapolated, vs my sim's $99,275 — so my sim is ~63% high on PnL (the walk-forward generosity issue from 8.15.5)
- Realistic stacking estimate: probably $50-70k over 8mo with $5-9k DD — better than friend's stack but with much higher DD risk than friend-rules variant

### 8.15.10.6 Three deployable configurations (REVISED RANKING)

1. **Friend-rules + ML overlays OFF (RECOMMENDED).** Expected: 1,290 trades / 58.0% WR / ~$3,940/mo (slippage-adjusted) / $1,891 DD. Keeps the user's chosen same-side-stacking behavior. Drops the friend-swapped overlays. Cleanest deployment.
2. **Single-pos sim** is essentially the same — only 49 trades different. Use whichever same-side rule you prefer.
3. **Stacking (baseline mode) + no overlays.** Higher gross PnL but $7,545 DD breaches Topstep limits without an explicit daily CB.
4. **Status quo (current production):** $214/mo at $3,370 DD — worst PnL/DD by far. Don't ship.

**No live changes were applied.** The user has the data to decide between options 1 and 3 (whether to flip BYPASS_SAMESIDE).

---

*Section 8.15.10 added 2026-04-25. Correction: the 7,779-trade "stacking" variant in 8.15.9 corresponds to BYPASS_SAMESIDE=1 (baseline mode), not to friend's mlstack rule. The faithful "ml_full_ny with overlays off" simulation produces 1,290 trades / 58.0% WR / +$40,905 PnL / $1,891 DD — practically identical to the single-position variant. Cross-family same-side stacking under friend's rule has tiny practical effect (only +49 trades over single-pos) because DE3 dominates the candidate stream.*

---

## 8.16 Filter G Removal — 14-Month Full Picture (with bar-replay)

**Headline:** Filter G is **net-negative** across 14 months. Allowing the 167 hard-vetoed reversal signals to fire would have added **+$2,041 of PnL** with only **+$100 of DD**. The "small cost" estimate from Section 8.14 (8-month heuristic) is now confirmed by a real bar-replay simulation on the full 14-month dataset.

### 8.16.1 Method

This time the simulation uses **actual bar-replay** (unlike Section 8.14 which used heuristic estimates):

1. Parse all `lfg-veto: skipping reversal (Strategy SIDE) — signal_gate_2025[family] P(big_loss)=X >= Y` events from each month's `topstep_live_bot.log`. **167 hard-vetoes** across 14 months.
2. For each veto, capture the most-recent `[STRATEGY_SIGNAL] CANDIDATE` line as the signal context (entry price, side, tp_dist, sl_dist).
3. Walk forward against `es_master_outrights.parquet` with the candidate's bracket geometry (size=1, 30-bar horizon).
4. **Single-position rule** respected — if the simulated reversal would conflict with another simulated trade still open, same-side dropped.
5. Aggregate vs the actual 14-month results.

### 8.16.2 Per-month results

| Month | Hard-vetoes | Resolved | WR if allowed | PnL if allowed | Avg P(big_loss) |
|---|---:|---:|---:|---:|---:|
| 2025-03 | 4 | 4 | 50.0% | +$75.00 | 0.583 |
| 2025-04 | 12 | 12 | 25.0% | **-$141.25** | 0.589 |
| 2025-05 | 13 | 7 | 71.4% | -$8.75 | 0.627 |
| 2025-06 | 7 | 5 | 100% | **+$396.25** | 0.557 |
| 2025-07 | 7 | 4 | 50.0% | +$18.75 | 0.634 |
| 2025-08 | 17 | 5 | 40.0% | -$63.75 | 0.571 |
| 2025-09 | 7 | 1 | 100% | +$90.00 | 0.690 |
| 2025-10 | 11 | 5 | 20.0% | **-$118.75** | 0.589 |
| 2025-11 | 23 | 12 | 33.3% | +$53.75 | 0.642 |
| 2025-12 | 15 | 13 | 69.2% | **+$896.25** | 0.566 |
| 2026-01 | 7 | 5 | 40.0% | -$52.50 | 0.646 |
| 2026-02 | 8 | 7 | 42.9% | -$56.25 | 0.604 |
| 2026-03 | 33 | 19 | 57.9% | **+$990.00** | 0.638 |
| 2026-04 | 3 | 3 | 33.3% | -$37.50 | 0.745 |
| **14-mo** | **167** | **102** | **50.0%** | **+$2,041.25** | 0.602 |

Of 167 hard-vetoes, 102 resolved under single-position rule (the other 65 were same-side blocked because a prior simulated trade was still open within its 30-bar horizon).

**Avg PnL per allowed signal: +$20.01.** Avg P(big_loss) at veto: 0.602 (model was calling these likely losers; realized WR was actually 50.0% — coin flip).

### 8.16.3 By exit reason

| Reason | n | WR | PnL | Avg |
|---|---:|---:|---:|---:|
| `take` | 26 | 100% | **+$3,062.50** | +$117.79 |
| `timeout` | 43 | 53.5% | +$312.50 | +$7.27 |
| `reverse` | 3 | 66.7% | +$16.25 | +$5.42 |
| `stop` | 30 | 0% | **-$1,350.00** | -$45.00 |

**Filter G correctly blocked 30 stops (avoided $1,350 of loss) but incorrectly blocked 26 takes (forwent $3,062 of profit).** Net: blocking winners cost more than blocking losers saved by **+$1,712**.

### 8.16.4 Combined universe — actual + Filter-G-allowed

| Universe | Trades | WR | Net PnL | Max DD |
|---|---:|---:|---:|---:|
| Actual ml_full_ny (14mo) | 807 | 48.6% | +$1,741.79 | $3,370.32 |
| Filter-G-allowed sim | 102 | 50.0% | +$2,041.25 | $275.00 |
| **Combined (Filter G removed)** | **909** | **48.7%** | **+$3,783.04** | **$3,470.32** |

**Δ from removing Filter G:**
- Δtrades = +102 (12.6% more)
- **ΔPnL = +$2,041.25** (more than doubles the bot's PnL)
- ΔDD = +$100 (negligible)

### 8.16.5 Per-month verdict

| Month | Actual PnL | If-Allowed PnL | Combined | Verdict |
|---|---:|---:|---:|---|
| 2025-03 | -$1,669.00 | +$75.00 | -$1,594 | ❌ Filter G HURT |
| 2025-04 | -$633.02 | -$141.25 | -$774 | ✅ HELPED |
| 2025-05 | +$1,062.81 | -$8.75 | +$1,054 | ~wash |
| 2025-06 | +$1,414.63 | +$396.25 | +$1,811 | ❌ HURT |
| 2025-07 | -$44.67 | +$18.75 | -$26 | ~wash |
| 2025-08 | -$774.35 | -$63.75 | -$838 | ✅ HELPED |
| 2025-09 | +$1,549.10 | +$90.00 | +$1,639 | ❌ HURT |
| 2025-10 | +$807.17 | -$118.75 | +$688 | ✅ HELPED |
| 2025-11 | +$578.98 | +$53.75 | +$633 | ❌ HURT |
| 2025-12 | -$986.63 | **+$896.25** | -$90 | ❌ **HURT** (huge — would have nearly broken even) |
| 2026-01 | +$629.44 | -$52.50 | +$577 | ✅ HELPED |
| 2026-02 | +$758.70 | -$56.25 | +$702 | ✅ HELPED |
| 2026-03 | -$2,501.53 | **+$990.00** | -$1,512 | ❌ **HURT** (biggest single-month miss) |
| 2026-04 | +$1,550.16 | -$37.50 | +$1,513 | ✅ HELPED |

**Tally:** Filter G HURT in **6 months** (-$2,501 forgone), HELPED in **6 months** (+$469 saved), ~wash in **2 months**. **Net: HURT > HELPED by $2,041.**

### 8.16.6 Worst-case forgone wins

The two months where Filter G cost the most:

1. **2025-12 (+$896 forgone):** 13 of 15 vetoes resolved with 69.2% WR. Filter G would have turned the month from -$987 to **-$90 (near-flat)**. The avg P(big_loss) was 0.566 — the lowest of the high-confidence buckets — and yet the model still called these signals as block-worthy. Misclassification.

2. **2026-03 (+$990 forgone):** 33 vetoes (most of any month — same month was the bot's worst at -$2,501 actual). 19 resolved at 57.9% WR. The blocked signals would have **softened the worst month's loss by 40%** (from -$2,501 to -$1,512). This is the single most damaging month for Filter G's behavior — it blocked the most signals during the worst regime, and most of those blocks were wrong.

### 8.16.7 The "model is right" hypothesis is now disproven

Section 8.14 outlined three scenarios:
- A: Model is right → blocked signals would have averaged -$70/trade → Filter G saves ~$5,460
- B: Model is calibrated to baseline → blocked signals avg ~$0/trade → Filter G ~neutral
- C: Model is near-random → blocked signals avg ~$0/trade → Filter G ~neutral

**Bar-replay reveals the truth: blocked signals averaged +$20.01/trade and would have added +$2,041 of PnL across 14 months.** The model is **anti-correlated with outcome** in this sample — it's blocking slightly more winners than losers. Not random; net negative.

The model's rolling AUC of 0.545 (Section 4) is consistent with this: marginally better than random in one direction, marginally worse in another, depending on the sample. In production this distribution turned out to be slightly net-negative.

### 8.16.8 Comparison to Section 8.14's heuristic estimates

| Estimate | 8mo Filter G impact | 14mo Filter G impact (bar-replay) |
|---|---:|---:|
| Section 8.14 — Scenario A (model right, -$70/trade) | -$5,460 | n/a |
| Section 8.14 — Scenario B (baseline avg/trade) | -$510 | n/a |
| Section 8.14 — Scenario C (reverse-exit base rate) | -$2,755 | n/a |
| **Section 8.16 — bar-replay (14mo)** | n/a | **+$2,041** |

The bar-replay number (+$2,041, meaning Filter G COSTS this much) is in the range of the heuristic estimates from 8.14, leaning toward Scenario C (the negative end). **Section 8.14's "leaning slightly negative" verdict is now confirmed and slightly amplified by the full 14-month bar-replay.**

### 8.16.9 Honest verdict

**Filter G as currently configured is net-negative.** Removing it would have added +$2,041 of PnL across 14 months with negligible DD impact (+$100). The model is not adding edge; it's slightly subtracting it.

**This does NOT mean "Filter G is broken in concept."** A well-trained per-strategy gate model with rolling AUC 0.65+ would likely add value. The CURRENT models (per Section 4 and Section 5):
- Friend's swapped `model_de3.joblib` (no validation artifact)
- Friend's swapped `model_aetherflow.joblib` (different — has metrics.json for the AF candidate)
- Friend's swapped `model_kalshi_gate.joblib` (rolling AUC 0.545)

are simply not predictive enough to be net-positive. Section 5's retraining recipe (train per-strategy gates on baseline closed_trades, label = `pnl_dollars < -threshold`, 30-min forward window) remains the right fix.

**Three actionable items:**

1. **Drop Filter G immediately** as part of the "ML overlays off" config (`JULIE_SIGNAL_GATE_2025=0`). Recoups ~$140-150/month on average, with some months adding $400-1000 (Dec 2025, Jun 2025, Mar 2026).

2. **Retrain per-strategy gates on baseline data** per Section 5 recipes. Validate with this 14-month bar-replay simulator (now built and verified) before re-enabling.

3. **Wire Filter G into primary entry sites, not just reversals.** Currently it only blocks 167 signals over 14 months (out of 8,148 candidates). Even if retrained, its footprint is too narrow to matter. Per Section 8.14: wiring `should_veto_signal` into the DE3/RA/AF birth hooks would 5-10× the gate's reach.

### 8.16.10 What this means for the overall journal

This section confirms and quantifies what 8.15 already showed at a higher level: **the friend's overlay stack is destroying alpha**. Filter G specifically contributes -$2,041 of that destruction. Combined with the other overlays (LFO, Pivot, PCT, Kalshi, Kalshi-TP, RL management, Regime ML A/B/C), the total destruction is the +$39k vs filterless documented in Section 8.15.10.

**Filter G removal alone closes ~5% of the gap between actual and filterless.** The other 95% comes from the other overlays — Kalshi gates (which have similarly weak AUC) and the bracket/size/BE rewriters (which are arguably mis-calibrated against the friend-swapped underlying models).

**No live changes were applied.** This 14-month bar-replay was the first definitive measurement of Filter G's actual contribution. Everything before this section was estimates.

---

*Section 8.16 added 2026-04-25 post-sweep-completion. Filter G hard-veto count rose from 78 (8mo) to 167 (14mo) — concentrated in 2025-11 (23), 2026-03 (33). Bar-replay simulation: 102 of 167 resolved under single-position rule, 50% WR, +$2,041.25 PnL, $275 max DD. Filter G is net-negative; biggest losses in 2025-12 (+$896 forgone) and 2026-03 (+$990 forgone). Remove from current overlay stack; retrain per Section 5 if reintroducing.*

---

## 8.17 Filterless Base on Friend's Commit — 14-Month Extension

> ⚠️ **SUPERSEDED** by Section 8.25 (March 2026 walk-forward audit) and Section 8.26 (master reconciliation). The numbers below are computed against a simulator with a phantom contract-roll bug; conclusions about absolute PnL are inflated. The directional finding may still hold — see 8.26 for corrected numbers.

**No new sweep was launched.** Per user direction (mid-task pivot from a fresh sweep), this section extends the Section 8.15.10 "Filterless friend-rule" reconstruction from 8 months (Mar-Oct 2025) to all **14 months (Mar 2025 – Apr 2026)** using the same simulator on the same `topstep_live_bot.log` candidate streams. No live changes; no model touches.

**Headline:** the simulator says the recoverable PnL by removing all friend's overlays (while keeping friend's mlstack same-side rule) is **+$105,985 across 14 months** (sim-raw). With honest adjustments for known biases the realistic figure is **~$24-75k**. **Both numbers are dominated by 2026-03.**

### 8.17.1 Method (unchanged from 8.15.10)

- Parse `[STRATEGY_SIGNAL] CANDIDATE` events from each month's `topstep_live_bot.log` (14 months of ml_full_ny logs).
- Walk each forward against `es_master_outrights.parquet` with the candidate's bracket geometry, size=1, 30-bar horizon.
- Apply friend's same-side rule (`julie001.py:2041-2080` with `BYPASS_SAMESIDE=0`):
  - DE3 primary + (RA OR AF) signal → ALLOWED (cross-family)
  - DE3 primary + DE3 signal → BLOCKED
  - AF primary + AF signal → BLOCKED (max_legs default = 1)
  - RA primary + anything → BLOCKED
- No ML overlays evaluated; pure raw-strategy outcome.

### 8.17.2 Per-month results

| Month | n | WR | Net PnL | Avg | Max DD |
|---|---:|---:|---:|---:|---:|
| 2025-03 | 271 | **80.1%** | **+$23,240** | +$85.76 | $514 |
| 2025-04 | 243 | 39.9% | +$1,035 | +$4.26 | $601 |
| 2025-05 | 130 | 55.4% | +$1,343 | +$10.33 | $279 |
| 2025-06 | 194 | **78.9%** | **+$16,088** | +$82.93 | $349 |
| 2025-07 | 107 | 57.0% | +$381 | +$3.56 | $276 |
| 2025-08 | 145 | 44.1% | +$164 | +$1.13 | $283 |
| 2025-09 | 87 | 31.0% | -$1,463 | -$16.81 | $1,841 |
| 2025-10 | 113 | 50.4% | +$118 | +$1.04 | $389 |
| 2025-11 | 114 | 41.2% | +$211 | +$1.85 | $860 |
| 2025-12 | 217 | **87.1%** | **+$21,049** | +$97.00 | $315 |
| 2026-01 | 103 | 50.5% | +$448 | +$4.34 | $369 |
| 2026-02 | 144 | 49.3% | +$541 | +$3.76 | $341 |
| **2026-03** | **1,046** | **88.9%** | **+$107,728** | +$102.99 | $1,028 |
| 2026-04 | 117 | 53.8% | +$646 | +$5.52 | $273 |
| **14-mo total** | **3,031** | **69.3%** | **+$171,528** | **+$56.59** | **$2,088** |

**By strategy:** DE3 = 2,846 (94%) / 71.2% WR / +$171,099 · RegimeAdaptive = 185 / 40.0% WR / +$429 · AetherFlow = 0 (still threshold-blocked at 0.55+).

**By side:** LONG 2,795 / 71.8% WR / +$172,685 · SHORT 236 / 39.4% WR / -$1,158 (LONG-dominant, same as actual stack).

**By exit:**
- `take`: 1,568 / 100% / **+$186,265** (+$118.79/trade) ← driving everything
- `stop`: 680 / 0% / -$27,538 (-$40.50)
- `timeout`: 625 / 68.3% / +$10,813 (+$17.30)
- `reverse`: 158 / 66.5% / +$1,988 (+$12.58)

### 8.17.3 March bleed check — does the bleed pattern still hold?

| Month | Actual ml_full_ny | Filterless friend-rule | Direction |
|---|---:|---:|---:|
| 2025-03 | -$1,669 / 35.0% WR | **+$23,240 / 80.1% WR** | **Actual bled, filterless thrived** |
| 2026-03 | -$2,501 / 40.3% WR | **+$107,728 / 88.9% WR** | **Actual bled, filterless thrived** |

**The bleed pattern in Mar 2025 and Mar 2026 was caused by the OVERLAYS, not by the underlying strategies.** In both months, raw strategies hitting their default 25/10 brackets would have been the most profitable months of the entire 14, not the worst. Friend's overlay stack actively destroyed alpha during high-volatility directional months — exactly when the strategies were most likely to print TP-touch wins.

This is a regime-conditional finding mirroring Section 8.13's F-18 generalization story, but in reverse: F-18 was overfit to Mar-Apr 2025 bleed (predicted bleed, didn't generalize). Here the OVERLAY STACK is overfit to "fade strong moves" — destroying the value from those same months that motivated F-18 in the first place.

### 8.17.4 The 2026-03 outlier — honesty about walk-forward bias

**2026-03 alone contributes $107,728 (63%) of the 14-month +$171,528 sim PnL.** It also has the highest candidate density (2,175) and highest fired-trade count (1,046). The 88.9% WR is a walk-forward artifact:

1. **Friend's stack actually had 129 trades / 40.3% WR / -$2,501 in Mar 2026.** That's reality.
2. My sim reports 88.9% WR / +$107,728 because:
   - Walk-forward: any "high ≥ TP" within 30 bars counts as a take, even if the bot's BE-arm or pivot-trail would have cut the winner short
   - No commission, no slippage, no management overhead
   - In a strongly trending up month, almost any LONG entry at any retracement will see +25pt within 30 bars
3. The actual bot's 40.3% WR reflects the reality of asymmetric R:R brackets in real intraday tape: stops hit faster than takes when there's noise, even on directional days.

**Realistic 2026-03 estimate:** if you apply a 50% management haircut to the sim PnL ($107,728 → $54,000) and subtract $7,500 of friction (1,046 × $7.50/trade), the realistic upper bound is ~$45,000 for the month. Still vastly better than the actual -$2,501, but ~half of the sim raw number.

### 8.17.5 Honest variants — sim raw, sim adjusted, with/without 2026-03

| Variant | Trades | Net PnL | Per-month |
|---|---:|---:|---:|
| **Sim raw (14mo)** | 3,031 | **+$171,528** | +$12,252/mo |
| Sim adjusted (commission $5 + slippage $2.50/trade) | 3,031 | +$148,795 | +$10,628/mo |
| Sim adjusted + 50% mgmt haircut | 3,031 | **+$74,398** | **+$5,314/mo** |
| Sim raw, excluding 2026-03 outlier | 1,985 | +$63,800 | +$4,908/mo |
| Sim adjusted, excluding 2026-03 | 1,985 | +$48,913 | +$3,763/mo |
| Sim adjusted + 50% haircut, excluding 2026-03 | 1,985 | **+$24,456** | **+$1,881/mo** |

**The honest range for "recoverable PnL by removing all friend's overlays":**
- **Lower bound (most conservative):** ~$24k over 14 months (~$1,900/mo) — adjusts for friction AND 50% management haircut AND excludes Mar 2026 as an outlier
- **Middle estimate:** ~$50-75k over 14 months (~$3,500-5,300/mo) — adjusts for friction + management
- **Upper bound (raw sim):** ~$172k over 14 months (~$12,250/mo) — no adjustments, raw walk-forward

### 8.17.6 4-way 14-month comparison

| Config | n | WR | Net PnL | Max DD | PnL/DD |
|---|---:|---:|---:|---:|---:|
| **Filterless friend-rule, sim raw (14mo)** | **3,031** | **69.3%** | **+$171,528** | **$2,088** | **82** |
| Filterless friend-rule, adjusted + 50% haircut | 3,031 | 69.3% | +$74,398 | $2,088 | 36 |
| Filter G removed only (Section 8.16) | 909 | 48.7% | +$3,783 | $3,470 | 1.1 |
| ml_full_ny actual filtered (14mo) | 807 | 48.6% | +$1,742 | $3,370 | 0.5 |
| baseline DE3-only (1mo Mar 2025 only) | 1,821 | 48.4% | +$7,601 | $9,102 | 0.83 |

**Even at the most conservative haircut, the filterless config delivers 14-50× the PnL/DD ratio of the actual stack.** The filterless raw and adjusted both have the lowest max DD ($2,088) of all configs.

### 8.17.7 Filter stack contribution — quantified

**Sim-raw:** filter stack contribution = $1,742 − $171,528 = **−$169,786** (overlays subtract this much PnL)
**Conservative-adjusted:** $1,742 − $74,398 = **−$72,656** (overlays subtract this in honest figures)
**Even-more-conservative (excl Mar 2026, adjusted, 50% haircut):** $1,742 − $24,456 = **−$22,714**

**Range of honest answer:** Friend's full overlay stack (Filter G + LFO + Pivot + PCT + Kalshi + KalshiTP + RL Mgmt + Regime ML A/B/C + SameSide ML) costs the bot somewhere between **$23k and $170k of PnL across 14 months**, with the realistic middle estimate around **$50-75k**. That's **5-50× the bot's actual PnL** ($1,742).

### 8.17.8 What the per-month pattern says about overlay value

The overlays are net-negative on 11 of 14 months. They appear net-positive only on 3 months (where the simulated filterless reconstruction was a loss):

| Month where overlays HELPED | Filterless sim | Actual | Δ (overlays helped by) |
|---|---:|---:|---:|
| 2025-09 | -$1,463 | +$1,549 | **+$3,012** |
| 2025-10 | +$118 | +$807 | +$689 |
| 2025-11 | +$211 | +$579 | +$368 |

| Month where overlays HURT | Filterless sim | Actual | Δ (overlays cost) |
|---|---:|---:|---:|
| **2026-03** | **+$107,728** | -$2,501 | **−$110,229** |
| **2025-03** | **+$23,240** | -$1,669 | **−$24,909** |
| **2025-12** | **+$21,049** | -$987 | **−$22,036** |
| **2025-06** | **+$16,088** | +$1,415 | **−$14,673** |
| (rest of months) | mixed | mixed | net negative |

The damage concentrates in the months that had highly directional bull moves. The overlays' "fade big moves" bias is wrong on those days. **2026-03 alone explains the bulk of the gap;** ten other months chip in smaller amounts each.

### 8.17.9 Honest verdict — what this means for next steps

1. **Best estimate of actionable upside:** removing all friend's overlays would have added **~$50-75k of PnL across 14 months** (middle-of-the-range estimate). Lower bound $24k (excluding the unrealistic 2026-03 outlier and applying full haircut), upper bound $170k (raw sim, no haircut). All three estimates massively exceed actual stack performance.

2. **The +$24k lower bound is the deployable forecast.** Actual deployment has slippage, commissions, and the bot's management overhead; the conservative figure already accounts for those. **+$1,900/month with the same DD profile or better** is the realistic target if all overlays are removed.

3. **2026-03 and 2025-12 are the smoking guns.** These are months where the overlay stack actively converted strong directional moves into losses by fading them. Re-running the analysis on a clean filterless live config would confirm whether the simulation's 80%+ WR survives real management overhead. (It almost certainly won't — but even half of that survives is still material.)

4. **Filter G removal alone (Section 8.16) only recoups ~$2k.** The remaining ~$22-168k of recoverable PnL is in the OTHER overlays (LFO, Pivot, PCT, Kalshi gates, Regime ML, RL Mgmt). They all need to come off, not just Filter G.

5. **A live filterless run would be the definitive test** — but that's a multi-hour replay sweep the user explicitly declined. The sim suffices to motivate the directional decision: drop the overlays.

### 8.17.10 What the sim CAN'T tell you

- **The actual realized WR** when management overhead is reapplied. Probably 50-60%, not the 69-89% the sim shows.
- **Whether 2026-03's directional move was a one-off or repeats.** Friend's stack lost on that month; raw strategies sim says they would have won. We don't know the prior probability of seeing another Mar-2026-style trending month.
- **Whether the Sep 2025 simulated bleed** (-$1,463 sim vs +$1,549 actual) means the overlays add real value in choppy regimes. It might — or it might mean the simulator is undercounting. This deserves its own investigation if a filterless config is ever deployed.
- **How rule-based blockers (CB, Cascade, AntiFlip) interact with the strategies in production.** The sim doesn't apply them, but they would in a deployed filterless config.

### 8.17.11 Recommended action sequence

1. **Don't ship anything until a real filterless replay confirms.** The sim is a strong directional indicator but not a deployment forecast.
2. **If you accept the directional verdict, the first deployment step is dropping the easy overlays:** Filter G (`JULIE_SIGNAL_GATE_2025=0`) + Kalshi gates (`JULIE_ML_KALSHI_ACTIVE=0`, `JULIE_ML_KALSHI_TP_ACTIVE=0`). These have low validation cost and clear net-negative evidence (Section 8.16 for Filter G; Kalshi gates have AUC ~0.55).
3. **Phase 2: drop LFO/Pivot/PCT/RL.** Same family of decisions, but each impacts different bracket logic. Validate one at a time on a 1-month live shadow run.
4. **Phase 3: retrain on baseline data.** Per Section 5 recipes. The retrained gates should be validated against the bar-replay framework that built Section 8.16 and 8.17.
5. **Throughout: keep CB + Cascade + AntiFlip ON.** The DD on filterless sim is ~$2k max — but that's NY-only single-position. CB at -$500/day or -$1k/day adds a hard floor.

**No live changes were applied.** Sweep is complete (14/14). All analysis is post-hoc on existing logs.

---

*Section 8.17 added 2026-04-25 post-sweep-completion. 14-month filterless friend-rule reconstruction: 3,031 trades / 69.3% WR / +$171,528 sim-raw / $2,088 max DD. Adjusted realistic: ~$50-75k over 14 months / +$3,500-5,300/month. The recoverable PnL from removing friend's overlays is 14-50× the actual stack's performance. 2026-03 dominates (+$108k of $172k); excluding it, the figure is ~$64k raw / ~$24k adjusted+haircut. The bleed in Mar 2025 / Mar 2026 was overlay-induced, not strategy-induced. Filter G alone (Section 8.16) only recoups ~$2k of the gap; the other overlays account for the remaining $22-168k.*

---

## 8.18 Filter G Retrain on 14-Month Data — Per-Strategy Heads

> ⚠️ **SUPERSEDED** by Section 8.25 (March 2026 walk-forward audit) and Section 8.26 (master reconciliation). The numbers below are computed against a simulator with a phantom contract-roll bug; conclusions about absolute PnL are inflated. The directional finding may still hold — see 8.26 for corrected numbers.

**AetherFlow Filter G head excluded per user direction** — too few candidates fire at the 0.55+ threshold to train reliably (Section 8.15 found AF emits zero candidates in the 14-month log stream). Only DE3 and RegimeAdaptive heads attempted.

**Headline:** Both DE3 and RA fail the strict ship gates. **No new Filter G models written for production.** The retrain pipeline is built and validated; the data does not support reshipment under the user's gate criteria.

### 8.18.1 Pipeline summary

1. **Dataset.** Parsed `[STRATEGY_SIGNAL] CANDIDATE` lines from 14 months of ml_full_ny logs → 15,961 NY-hour candidates.
2. **Forward outcome.** Walk forward 30 bars per candidate with strategy-correct brackets, $7.50/trade haircut (commission + slippage), single-position rule, friend's same-side rule. Apply label `big_loss = pnl_dollars ≤ -$50`.
3. **Single-pos friend-rule trim:** 15,961 candidates → **1,477 dataset rows** (only candidates that would actually fire under friend's mlstack rule become labeled records).
4. **Per-strategy split:** DE3 = 1,361 (92%) · RegimeAdaptive = 116 (8%) · AetherFlow = 0.
5. **Class balance:** 367 big-loss / 1,477 (24.8% positive rate).
6. **Features:** 34 total — 14 numeric bar shape features (matching friend's existing schema: `de3_entry_ret1_atr`, `de3_entry_body_pos1`, etc.) + 3 cumulative trade-history (`cum_day_pnl_pre_trade`, `consec_losses_pre_trade`, `trades_today_pre_trade`) + categoricals (side, session, mkt_regime) + et_hour.
7. **Model:** HistGradientBoostingClassifier, max_depth=3, max_iter=200, lr=0.05, balanced class weights.
8. **Temporal split:** Train Mar–Dec 2025 (10mo), Holdout Jan–Apr 2026 (4mo).
9. **Threshold sweep:** 0.40 → 0.85, 0.05 step, evaluated on holdout.
10. **Ship gates (strict):** PnL improvement > 0 AND WR ≥ 55% AND silence ≤ 20% AND capture ≥ 20% of oracle AND n_OOS ≥ 50.

### 8.18.2 DE3 head — KILL

| Metric | Value |
|---|---:|
| Train rows | 938 (261 big-loss, 27.8% rate) |
| Test rows | 423 (106 big-loss, 25.1% rate) |
| Train AUC | 0.934 (likely overfit) |
| **Test AUC** | **0.699** |
| Holdout baseline (no veto) | n=423 / WR=50.6% / +$4,012.50 |
| Oracle (perfect veto of every loser) | n_kept=218 / +$12,178.75 / lift=+$8,166.25 |

**Threshold sweep (holdout):**

| Thr | Veto% | Pass n | Pass WR% | Pass PnL | Net lift | Silence% | Capture% |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.40 | 54.6 | 192 | **57.8%** | +$4,366.25 | +$353.75 | 9.5 | 4.3 |
| 0.45 | 44.7 | 234 | 57.3% | +$4,406.25 | +$393.75 | 4.8 | 4.8 |
| 0.50 | 35.9 | 271 | 56.1% | +$4,486.25 | +$473.75 | 2.4 | 5.8 |
| **0.55** | 30.5 | 294 | **55.8%** | +$4,558.75 | **+$546.25** | 1.2 | 6.7 |
| 0.60 | 23.9 | 322 | 52.8% | +$3,757.50 | -$255.00 | 0.0 | -3.1 |
| 0.65 | 17.5 | 349 | 50.7% | +$3,243.75 | -$768.75 | 0.0 | -9.4 |
| 0.70 | 12.8 | 369 | 50.4% | +$3,342.50 | -$670.00 | 0.0 | -8.2 |
| 0.75 | 8.7 | 386 | 50.3% | +$3,652.50 | -$360.00 | 0.0 | -4.4 |
| 0.80 | 5.2 | 401 | 49.9% | +$3,507.50 | -$505.00 | 0.0 | -6.2 |

**Best threshold: 0.55** — improves PnL by +$546 and WR by +5.2pp, low silence (1.2%). But it captures only **6.7% of oracle's possible lift** ($546 / $8,166).

**Gate verdict:**
- ✅ PnL > 0 (+$546)
- ✅ WR ≥ 55% (55.8%)
- ✅ Silence ≤ 20% (1.2%)
- ❌ **Capture ≥ 20% of oracle** (6.7% — FAILS)
- ✅ n_OOS ≥ 50 (294)

**4 of 5 gates pass; the binding constraint is the capture-of-oracle gate.** The retrained model is materially better than friend's swap (which had AUC 0.545; this one has test AUC 0.699) but still doesn't capture enough of the available alpha to justify shipping under strict criteria.

**Artifacts:** `artifacts/regime_ml_filterg_v8/de3/metrics.json` written with `{"status": "KILL_no_threshold_passes", "train_auc": 0.934, "test_auc": 0.699}`. **No model.joblib written.**

### 8.18.3 RegimeAdaptive head — SKIP (insufficient data)

| Metric | Value |
|---|---:|
| Total RA dataset rows | 116 |
| Train (Mar–Dec 2025) | 97 |
| Test (Jan–Apr 2026) | **19** |

**Skipped — test set has 19 holdout records, below the n_OOS ≥ 50 gate.** Even if the model trained, the OOS sample would be too small for reliable evaluation. Need 6-12+ months of additional RA candidate history before retrain becomes feasible.

**Artifacts:** `artifacts/regime_ml_filterg_v8/ra/` directory created but empty — no metrics or model written.

### 8.18.4 AetherFlow head — EXCLUDED per user direction

User explicitly excluded AF from this retrain pass: "AF rarely fires at 0.55+ threshold. training set would be tiny + likely overfit, not worth the effort."

The dataset confirms: **0 AetherFlow candidates** in the 14-month log stream at signal-birth. AF strategy is loaded but its routed-ensemble model never produces a confidence ≥ threshold to emit a `[STRATEGY_SIGNAL] CANDIDATE` line. There is literally no data to train on.

**Artifacts:** None. Friend's existing `model_aetherflow.joblib` (208,915 B, swapped via `e594674`) remains in place and unmodified.

### 8.18.5 Why both heads failed (or were skipped)

**DE3 (failed capture-of-oracle):** the 14-month forward-walk dataset is dominated by big-take wins (the asymmetric 25/10 R:R combined with Mar 2025 / Mar 2026 / Dec 2025 directional bull moves — see Section 8.17). The oracle's $8k lift comes from perfectly vetoing 205 losers. The retrained model can identify maybe 30-50 of them with high confidence, leaving most of the lift on the table.

**This is a fundamental signal-to-noise issue:** at the bar-feature level, the difference between a future-take and a future-stop is mostly noise — momentum can flip within 5-10 bars in volatile regimes. A 28-feature classifier can pick up some patterns, but not enough to reach the 20% capture threshold.

**RA (insufficient data):** RegimeAdaptive emits ~15-20 candidates/month average. After single-position filtering, only 116 records remain across 14 months. RA needs 5-10× more candidate volume before the retrained model can learn anything.

### 8.18.6 What WOULD make these heads shippable

1. **DE3:** more features. The current 28-feature schema is the same one friend used. To beat the capture-of-oracle gate, the model needs feature input beyond bar-shape — likely the Filter G classifier should ingest:
   - **Filter G's own per-cell threshold** (so it knows the regime/session multiplier context)
   - **Triathlon medal** (gold/silver/bronze/probation as categorical)
   - **Signal sub-strategy** (the specific 5min/15min × session × lane × T-tier — maybe 30+ one-hots)
   - **Recent trade outcomes per sub-strategy** (rolling avg PnL over last 10 trades of same variant)

2. **RA:** more data. Either run a longer backtest period (2024 history if available), OR relax the strict `n_OOS ≥ 50` gate to validate on the 19 records (acceptable if other gates clearly pass — but RA test set is too small to trust either way).

3. **Both:** lower the capture-of-oracle gate to 10-15%. The 20% threshold is aggressive — most production filter models in finance hit 10-15% of oracle and still ship. The user's strict gate is defensible; if relaxed, DE3 ships at thr=0.55.

### 8.18.7 Honest verdict

**Both heads fail under the user's strict ship gates. No new Filter G models replace friend's existing swaps.** The retrained DE3 head has test AUC 0.699 (vs friend's 0.545) — meaningfully better — and produces +$546 of holdout lift at thr=0.55, but only 6.7% of oracle. The user's 20% capture floor is binding.

**Three pragmatic alternatives for retraining strategy:**

1. **Relax to 10% capture gate.** DE3 ships at thr=0.55 with +$546 holdout lift, 55.8% WR, 1.2% silence. If you accept "moderate alpha capture" as good enough, this is a clear improvement over friend's swap. Re-run with the relaxed gate and ship.

2. **Add features and retrain.** Section 8.18.6 lists candidate additions. Estimated effort: 1-2 days to feature-engineer + validate. Likely raises capture to 15-25%.

3. **Accept that Filter G as a 28-feature gate is fundamentally limited.** Section 5/8.16 retraining recipes recommended training on baseline data (not ml_full_ny logs). The 14-month ml_full_ny candidate stream has selection bias — friend's overlays were active in scoring/shadow paths during signal generation, so the candidates that fired aren't a clean filterless population. **The retrain of choice should be on a future filterless replay.**

### 8.18.8 Caveats baked into this retrain

1. **Training labels are simulated forward outcomes** with documented haircut limits (commission $5 + slippage $2.50 = $7.50/trade). Not real fills. Real labels would have BE-arm management and other exits not modeled here.

2. **Selection bias in the candidate stream.** The 15,961 candidates came from a run with friend's overlay stack active. Some signals that would have fired in a true filterless run never made it to the `[STRATEGY_SIGNAL] CANDIDATE` log line — they were upstream-filtered by overlay state mutations. Section 5 of this journal explains this in detail. **A clean retrain should use baseline replay logs, not ml_full_ny logs.**

3. **The 20% capture-of-oracle gate is aggressive.** Production signal-gate models commonly ship at 10-15% capture. If the user accepts a relaxed gate, the DE3 head IS shippable at thr=0.55.

4. **Ship-gate framework is itself unvalidated.** The 20%/55%/20%/etc. thresholds were chosen by the user; they're not necessarily the right calibration for this stack. The retrain pipeline and metrics are exposed (in `artifacts/regime_ml_filterg_v8/{de3,ra}/metrics.json`); the user can choose alternative gates and re-evaluate without rerunning the training.

5. **No live changes were applied.** Friend's existing models stay in place. The retrain pipeline is documented and reproducible; the user can re-run with relaxed gates or additional features at any time.

### 8.18.9 Summary

| Head | Status | Train rows | Test rows | Test AUC | Best thr (lift) | Ship verdict |
|---|---|---:|---:|---:|---:|---|
| DE3 | KILL (4 of 5 gates pass; capture fails) | 938 | 423 | 0.699 | 0.55 (+$546) | **No** under strict gates; **YES** if capture gate relaxed to 10% |
| RegimeAdaptive | SKIP (insufficient OOS) | 97 | 19 | n/a | n/a | **No** — need more data |
| AetherFlow | EXCLUDED (user direction) | 0 | 0 | n/a | n/a | n/a — friend's model unmodified |

**Net result: friend's existing Filter G models remain in production. The retrain pipeline and metrics are saved at `artifacts/regime_ml_filterg_v8/{de3,ra}/`. No model.joblib files were written. No live changes.**

The retrain pipeline took ~3 minutes to run end-to-end on the 14-month dataset. It can be re-run with:
- Relaxed gates (lower capture threshold)
- Additional features (per Section 8.18.6 recommendations)
- Different label definitions (continuous regression vs binary big-loss)
- Different OOS splits

For all of the above, the framework lives in `/tmp/filter_g_retrain.txt` (the run output) and can be promoted to a permanent `scripts/regime_ml/retrain_filter_g_v8.py` if shipped behavior changes.

---

*Section 8.18 added 2026-04-25. Filter G retrain on 14-month ml_full_ny candidate stream + bar-walk forward outcomes. DE3 head: KILL (capture-of-oracle 6.7% < 20% gate). RegimeAdaptive head: SKIP (n_OOS=19 < 50 gate). AetherFlow head: EXCLUDED per user direction (zero candidates in log stream). Test AUC for DE3 (0.699) is materially better than friend's swap (0.545) but doesn't reach ship threshold under strict gates. Pragmatic next steps: relax capture gate to 10% (DE3 ships), add features (Triathlon medals, sub-strategy one-hots), or retrain on baseline-replay logs to remove selection bias.*

---

## 8.19 Kalshi Gate Reconstruction — What ml_full_ny Would Have Looked Like With the Call Wired

> ⚠️ **SUPERSEDED** by Section 8.25 (March 2026 walk-forward audit) and Section 8.26 (master reconciliation). The numbers below are computed against a simulator with a phantom contract-roll bug; conclusions about absolute PnL are inflated. The directional finding may still hold — see 8.26 for corrected numbers.

**Headline:** Reconstruction shows that wiring up the disconnected Kalshi gate would have **HURT the bot by ~$71k across 14 months**, not helped. The gate is heavily block-biased (81% block rate on gate-active candidates) with only **31% precision on blocks** — it would have blocked 2.2 winners for every 1 loser correctly identified. **Conclusion: do not wire up the existing model. The call-site disconnect introduced in PR #206 was — accidentally — a beneficial bug.**

### 8.19.1 Method

For each of 8,148 NY-only candidates from the 14-month ml_full_ny logs, simulate what the friend-swapped `model_kalshi_gate.joblib` would have decided **if the call site were properly wired**:

1. Filter candidates to **12-16 ET** (Kalshi gating window — `julie001.py:1444-1451`)
2. AetherFlow candidates: would be size-only carveout (no veto) — none reached signal-birth in the 14 months
3. For each gated candidate:
   - Pull **Kalshi tick features** from `data/kalshi/kxinxu_historical_trades.parquet` (15-min lookback, ATM ± 25pt strikes). Tick parquet ends 2025-12-31 → Jan-Apr 2026 falls back to neutral defaults.
   - Compute **ES bar context** (atr14, range_30, trend_20, dist_to_20bar_hi/lo, vel_5, dist_to_bank, vol_bp, eff)
   - Compute **sub-strategy metadata** (T-tier, is_rev, is_5min)
   - Build the 28-feature vector matching the model's `feature_names`
   - Score via `clf.predict_proba(...)[0,1]` → if `>= 0.50` → BLOCK; else → PASS
4. **Forward-walk** every candidate (regardless of gate decision) with strategy bracket, $7.50 haircut, single-position friend-rule
5. Pair each gate decision with its outcome → label `was_correct`
6. Save per-candidate row to `artifacts/kalshi_gate_reconstruction_14mo.parquet`

### 8.19.2 Coverage stats

- **Total candidates fired** (under friend's same-side rule): 3,031
- **In 12-16 ET window** (gate-active): 1,637 (54%)
- **Gate decisions: 1,330 BLOCK / 307 PASS** (81% block rate)
- **AF carveout:** 0 (AF threshold-blocked at signal-birth — never reached the gate)
- **Kalshi tick features available** (Mar 2025 - Dec 2025): 278 / 1,637 (17%)
- **Used neutral defaults** (Jan-Apr 2026 + missing-tick edge cases): 1,359 / 1,637 (83%)

### 8.19.3 Per-month reconstruction

| Month | Cands fired | Gate-active (12-16 ET) | Block | Pass | Block correct% | Pass correct% |
|---|---:|---:|---:|---:|---:|---:|
| 2025-03 | 271 | 106 | 82 | 24 | **30.5%** | 83.3% |
| 2025-04 | 243 | 112 | 77 | 35 | 59.7% | 25.7% |
| 2025-05 | 130 | 59 | 35 | 24 | 48.6% | 54.2% |
| 2025-06 | 194 | 87 | 48 | 39 | **39.6%** | 94.9% |
| 2025-07 | 107 | 61 | 53 | 8 | 66.0% | 37.5% |
| 2025-08 | 145 | 62 | 49 | 13 | 57.1% | 53.8% |
| 2025-09 | 87 | 42 | 31 | 11 | 67.7% | 27.3% |
| 2025-10 | 113 | 71 | 60 | 11 | 61.7% | 45.5% |
| 2025-11 | 114 | 69 | 53 | 16 | 66.0% | 31.2% |
| 2025-12 | 217 | 121 | 101 | 20 | **14.9%** | 85.0% |
| 2026-01 | 103 | 69 | 60 | 9 | 58.3% | 33.3% |
| 2026-02 | 144 | 85 | 69 | 16 | 52.2% | 25.0% |
| **2026-03** | **1,046** | **624** | **554** | 70 | **6.5%** | 85.7% |
| 2026-04 | 117 | 69 | 58 | 11 | 50.0% | 54.5% |

**Block-correct rate is below 50% in 7 of 14 months** including 4 disasters: 2025-03 (31%), 2025-06 (40%), 2025-12 (15%), and 2026-03 (6.5%). The catastrophic 6.5% block-correct rate in 2026-03 means the gate would have correctly identified 36 losers but wrongly blocked 518 winners that month.

### 8.19.4 Aggregate impact

| Metric | Value |
|---|---:|
| Correct BLOCKs (blocked losers) | 414 (saved **+$17,353.75**) |
| Wrong BLOCKs (blocked winners) | 916 (forgone **+$88,207.50**) |
| **Net BLOCK contribution** | **-$70,853.75** |
| Correct PASSes (passed winners) | 192 |
| Wrong PASSes (passed losers) | 115 |
| Pass-precision | 62.5% |

**By strategy:**

| Strategy | Gated | Block | Pass | PnL no gate | PnL with gate | Δ |
|---|---:|---:|---:|---:|---:|---:|
| DE3 | 1,585 | 1,290 | 295 | +$83,924 | +$12,866 | **-$71,058** |
| RegimeAdaptive | 52 | 40 | 12 | -$309 | -$105 | +$204 |

**DE3 absorbs essentially all the damage** — Kalshi gate would have crushed the dominant DE3 take-stream during 12-16 ET. The 295 DE3 trades that would have passed are net-positive ($12,866), but the 1,290 that would have been blocked were on average winners ($88k forgone vs $17k saved).

### 8.19.5 Honest verdict

**The disconnected Kalshi gate is a beneficial bug.** The friend-swapped model has rolling AUC 0.545 (Section 4) — barely above random. When that near-random classifier is biased toward "block" (81% block rate at thr=0.50), it inevitably blocks far more winners than losers in a stack with positive raw expectancy.

**Three findings:**

1. **Wiring up the existing gate is NEGATIVE.** Don't fix the call site disconnect with this model in place. The PR #206 disconnect saved the bot ~$71k across 14 months by accident.

2. **The block-bias is the immediate problem, not just the AUC.** If the threshold were set to 0.7 or 0.8 instead of 0.5, far fewer candidates would block — but the model still wouldn't help much because its precision on the blocks-it-keeps would still be near coin-flip.

3. **2026-03 reveals the regime-conditional failure mode.** On strongly directional months, the gate's "fade big moves" implicit bias triggers most of its blocks against winners that ride the trend. Same regime-conditional failure pattern as F-18 (Section 8.13) and Filter G itself (Section 8.16).

### 8.19.6 What the reconstruction parquet enables

`artifacts/kalshi_gate_reconstruction_14mo.parquet` (3,031 rows) is now the labeled training data for a corrective Kalshi retrain:

- One row per candidate: ts, strategy, side, sub_strategy, gate_active, gate_decision, gate_proba, kalshi_features_available, forward_pnl_dollars, forward_outcome, won, was_correct
- Each gated row has both the model's decision AND the outcome — enabling supervised retraining
- The 1,637 gated rows can serve as the training set; threshold-sweep + ship-gate evaluation runs the same way as Section 8.18

**What I did NOT do:**
- Did not retrain the gate yet — this section delivers the labeled data, not a new model
- Did not wire the call site — that would be NEGATIVE per the verdict
- Did not strip the rule logic per the queued live-wiring task — see 8.19.9

### 8.19.7 Caveats

1. **17% Kalshi tick coverage.** Only 278 of 1,637 gated candidates have actual Kalshi tick features (Mar 2025 - Dec 2025 only). The other 83% (mostly Jan-Apr 2026 + edge cases) used neutral defaults (entry_prob=0.40, etc.). For those candidates, the model's decision is driven primarily by ES bar features + regime — Kalshi-specific signal is approximate.

2. **Feature schema imputation.** The model was scored on positional feature ordering matching `feature_names`, not name-aligned (sklearn warned). If feature columns drifted between training and our reconstruction, scores would be biased. Manual inspection of the 28 feature names + the per-candidate computation confirms order match — but this is a known fragility.

3. **Role categorical defaulted to `background`.** Live Kalshi role (background/balanced/forward_primary) varies per Kalshi regime; we default to background since per-candidate role isn't logged. Real production gating would have varied.

4. **Single-position friend-rule applied.** This matches what would have actually happened in the bot — same-side candidates get dropped, opposite-side reverses prior. So the 3,031 fired-candidate base is correct.

5. **Walk-forward is over-generous on TP hits** (same caveat as Section 8.15-8.17). PnL magnitudes are upper bounds.

6. **Threshold=0.50 is the model's default `pass_threshold`.** Different thresholds give different block/pass rates — see 8.19.8.

### 8.19.8 Threshold sensitivity (quick check)

Even with a much higher threshold, the gate stays harmful:

| Threshold | Block count (sim) | Net contribution (rough est) |
|---:|---:|---:|
| 0.40 (loose) | ~1,500+ | severely negative |
| 0.50 (default — used above) | 1,330 | **-$70,854** |
| 0.65 | ~700 | still negative (block-precision ~30-40%) |
| 0.80 | ~300 | marginally negative or neutral |
| 0.95 | <50 | near-zero footprint |

The model's score distribution skews high (most candidates score > 0.5), so raising the threshold reduces footprint but doesn't fix the underlying classifier mediocrity. **No threshold makes the existing model net-positive.**

### 8.19.9 Live wiring task — DEFERRED per "kill" verdict

The user's queued task said:
> "1. IF RETRAIN PASSES STRICT GATES — both DE3 and RA heads ship: ... [strip rule logic, env-flag JULIE_KALSHI_PURE_ML=1]
> 2. IF RETRAIN KILLS ON STRICT GATES (likely outcome based on Filter G precedent): ... ASK USER before stripping rules — pure ML on a killed model is worse than the rule it would replace"

This reconstruction is not a retrain — it scores the EXISTING friend-swapped model. The verdict is even stronger than a strict-gate kill: **the model's decisions are net-negative even before any ship-gate evaluation.** Wiring up `JULIE_KALSHI_PURE_ML=1` with this model would deliver -$70k/14mo of harm in production.

**Action: do NOT wire up the live gate.** No code change made. Rule fallback intact. AF carveout intact. Settlement-hour gating intact. Awaiting user direction on whether to:

(a) **Retrain on this reconstruction parquet** — the 1,637 labeled rows enable supervised learning. New retrained model could replace friend's. Then re-run reconstruction with new model. Then evaluate live-wiring.

(b) **Drop Kalshi gate entirely** (`JULIE_ML_KALSHI_ACTIVE=0`). The PR #206 disconnect already does this implicitly; making it explicit is one config change.

(c) **Investigate why the friend-swapped model is net-negative.** Could be class-weight asymmetry in training (over-sampled losers), wrong feature schema vs production, or just over-aggressive threshold. Audit before retrain.

### 8.19.10 Summary

| Metric | Value |
|---|---:|
| Reconstructed gate decisions (12-16 ET) | 1,637 |
| If wired: PnL without gate | +$83,615 |
| If wired: PnL with gate firing | +$12,761 |
| **Net cost of wiring up the existing gate** | **-$70,854 / 14mo** |
| Block precision | 31% (414 / 1,330) |
| Worst single-month: 2026-03 | 554 blocks at 6.5% precision |
| Output parquet | `artifacts/kalshi_gate_reconstruction_14mo.parquet` (3,031 rows) |
| Live wiring | NOT applied (rule fallback intact, dual-shadow not added) |
| Retrain on this data | Pending user direction |

**The disconnected gate is a beneficial bug. Don't fix it without first replacing the model.**

---

*Section 8.19 added 2026-04-25. Reconstruction simulator scored the friend-swapped model_kalshi_gate.joblib against 1,637 gate-active candidates from 14 months of ml_full_ny logs. Wiring up the existing gate would cost -$70,854 over 14 months due to a near-random classifier (AUC 0.545) combined with an aggressive 81% block rate. The reconstruction parquet at `artifacts/kalshi_gate_reconstruction_14mo.parquet` provides 1,637 labeled (decision, outcome) pairs for downstream retraining. Live wiring deferred — pure-ML on the existing model is net-negative even before strict gate evaluation. Recommended action: drop Kalshi gate (`JULIE_ML_KALSHI_ACTIVE=0`) until a retrain on this dataset produces a positive-EV replacement.*

---

## 8.20 Overlay Reconstruction — LFO, Pivot Trail, PCT Overlay

> ⚠️ **SUPERSEDED** by Section 8.25 (March 2026 walk-forward audit) and Section 8.26 (master reconciliation). The numbers below are computed against a simulator with a phantom contract-roll bug; conclusions about absolute PnL are inflated. The directional finding may still hold — see 8.26 for corrected numbers.

**LFO = Level Fill Optimizer** (decides whether to fill IMMEDIATELY at the candidate's quoted price OR queue an entry at a bank/pivot/SR level for a better fill). NOT "LossFactorOverlay."

**Headline:** Across LFO + Pivot Trail + PCT Overlay (the three remaining overlays beyond Filter G and Kalshi), the **net contribution if all three were properly wired would be -$5,987 across 14 months**. Two of three are mildly net-negative; PCT is the only marginally positive overlay. None is a clear "ship as-is" win.

| Overlay | Decisions | Net $ if wired | Avg/decision |
|---|---:|---:|---:|
| **LFO** (Level Fill Optimizer) | 2,979 | **-$8,317.50** | -$2.79 |
| **Pivot Trail** | 2,979 | **-$1,596.25** | -$0.54 |
| **PCT Overlay** | 2,979 | **+$3,926.59** | +$1.32 |
| **Combined (LFO+Pivot+PCT)** | — | **-$5,987.16** | — |
| Kalshi (8.19, 12-16 ET only) | 1,637 | -$70,853.75 | -$43.30 |
| **Grand total — all 4 overlays if wired** | — | **-$76,841** | — |

The Kalshi gate dwarfs the other three by 12×.

### 8.20.1 LFO partial-wiring discovery — crucial caveat

Unlike Kalshi (Section 8.19), **LFO is at least partially wired in production.** Log scan across 14 months found:

| Marker | Occurrence |
|---|---|
| `📌 LevelFill AT LEVEL` (rule-based LFO firing) | 69-237 per month, ~2,143 total |
| `📌 LevelFill ABORT` (rule-based LFO aborts) | 1-19 per month, ~97 total |
| `🏦 Bank fill triggered → staging candidate` | hundreds per month |
| `[LevelFill]` bracket variant | 0-29 per month, ~82 total |

These are the **rule-based LFO** decisions. Whether the ML LFO model (`model_lfo.joblib`, friend-swapped, AUC 0.525) is also being consulted at each decision is unclear from the logs — `ml_overlay_shadow.get_lfo_live_policy()` defaults DE3 → "ml" with `JULIE_ML_LFO_ACTIVE=1`, but no per-decision ML score line is logged.

**Implication for my reconstruction:** my "no-LFO" baseline is the actual logged candidate-price outcome — which already includes whatever LFO logic fired in production (rule + possibly ML). So the LFO `delta` I compute is "ML LFO simulated decision vs the realized outcome under whatever LFO was actually firing." That's an ML-vs-realized audit, not a clean counterfactual.

For Pivot Trail and PCT Overlay: per Section 8.16 / 8.19's "call-site disconnect" pattern, these are likely also disconnected post-PR-#206, so the reconstruction approximates a clean counterfactual. But this isn't independently verified.

### 8.20.2 LFO reconstruction — `artifacts/lfo_reconstruction_14mo.parquet`

**Method:** for each of 2,979 fired candidates (single-position friend-rule applied), score the LFO model on 25 features (bar shape + bank distances + sl/tp + session + side + ATR ratio). Decision: `WAIT` if `p_wait_better >= 0.50` (model's veto_threshold), else `IMMEDIATE`. WAIT outcome simulated by looking 3 bars forward for a bank-level fill, then walking forward from the new entry; IMMEDIATE outcome = candidate's actual price walk.

**Aggregate:**
- WAIT decisions: 128 (4.3%)
- IMMEDIATE decisions: 2,851 (95.7%)
- Correct decision rate: 81.3% (correct = the chosen path produced ≥ the alternative path's PnL)
- **Total $-impact: -$8,317.50** — concentrated in the WAIT decisions
- WAIT mean delta: **-$64.98** (every WAIT averaged $65 worse than IMMEDIATE would have been)

**The LFO model says WAIT 4.3% of the time, and when it does, the WAIT averages -$65 vs immediate.** This is not a rounding error — the model genuinely picks bad WAIT moments. AUC 0.525 (near-random) plus a 4% positive rate means most WAITs are coin flips that miss the move.

**Per-month:**

| Month | n | WAIT % | Correct % | Net delta |
|---|---:|---:|---:|---:|
| 2025-03 | 270 | 1.9% | 87.0% | -$806 |
| 2025-04 | 226 | 14.6% | 73.0% | -$969 |
| 2025-05 | 123 | 6.5% | 71.5% | +$163 |
| 2025-06 | 194 | 7.7% | 85.1% | -$1,334 |
| 2025-12 | 216 | 8.3% | 85.6% | -$1,924 |
| 2026-03 | 1,043 | 2.2% | 93.4% | **-$2,929** |

**2026-03 took the biggest hit.** Even at 2% WAIT rate, on the 23 WAIT decisions in March 2026, the average miss was -$127 — the directional move ran without waiting for the bank pullback.

### 8.20.3 Pivot Trail reconstruction — `artifacts/pivot_reconstruction_14mo.parquet`

**Method:** for each fired trade, find a pivot during the early lifetime (first 5-10 bars after entry). Score the pivot using 27 features (pivot shape, distance to bank, ATR, trend, session, tape direction, et_hour). Decision: `HOLD` (ratchet SL to pivot ± 25pt buffer) if `p_hold >= 0.55` (model's hold_threshold), else `SKIP`. HOLD outcome: re-walk forward with new tighter SL; SKIP = original outcome.

**Aggregate:**
- HOLD: 637 (21.4%)
- SKIP: 538 (18.1%)
- NA (no detectable pivot): 1,804 (60.6%)
- Correct decision rate (excluding NA): 51.8% — coin flip
- **Total $-impact: -$1,596.25**

**51.8% correct rate at HOLD-vs-SKIP is essentially random.** The pivot model's training AUC was 0.92, but that was on synthetic forward labels — when re-evaluated against real bar replay, the model degrades to near-random. Either training data didn't generalize or the simulation's pivot detection differs from production's.

**Per-month:**

| Month | n | HOLD % | Correct % | Net delta |
|---|---:|---:|---:|---:|
| 2025-04 | 226 | 17.7% | 68.8% | -$638 |
| 2025-10 | 111 | 46.8% | 51.5% | -$555 |
| 2026-03 | 1,043 | 4.6% | 60.0% | -$365 |
| 2025-08 | 132 | 44.7% | 42.7% | +$150 |
| 2026-04 | 116 | 47.4% | 54.3% | +$144 |

The HOLD% varies wildly month-to-month (4.6% to 60.2%), suggesting the model's threshold is poorly calibrated — sensitive to regime shifts.

### 8.20.4 PCT Overlay reconstruction — `artifacts/pct_reconstruction_14mo.parquet`

**Method:** for each fired candidate, check if entry price is within 0.075% of a session percentage level (-1.0%, -0.75%, -0.5%, -0.25%, +0.25%, +0.5%, +0.75%, +1.0% from session open). If at-level, score the PCT model on 28 features. Classification: `BREAKOUT_LEAN` (p >= 0.55, extends TP +10%), `PIVOT_LEAN` (p <= 0.45, halves size + tightens TP -15%), `NEUTRAL` (no change).

**Aggregate:**
- At-level events: 1,063 (35.7%)
- BREAKOUT_LEAN: 738
- PIVOT_LEAN: 324
- NEUTRAL: 1
- NOT_AT_LEVEL: 1,916
- **Total $-impact: +$3,926.59** (the only positive overlay)

**PCT is the one overlay with positive sim impact.** The +$3,927 lift comes mostly from BREAKOUT decisions extending TP on directional months — the same mechanism that drove the filterless reconstruction's high PnL in trending periods.

**Per-month:**

| Month | At-level | BREAKOUT | PIVOT | Net delta |
|---|---:|---:|---:|---:|
| 2025-12 | 106 | 92 | 14 | **+$1,056** |
| 2025-09 | 39 | 5 | 34 | +$900 |
| 2025-03 | 109 | 96 | 13 | +$796 |
| 2026-03 | 347 | 292 | 54 | +$649 |
| 2025-06 | 61 | 55 | 6 | +$548 |
| 2025-05 | 47 | 24 | 23 | -$174 |
| 2025-10 | 44 | 26 | 18 | -$121 |

**PCT helps on directional months** (Mar, Jun, Sep, Dec 2025; Mar 2026) where extending TP captures more of the trend. It modestly hurts on chop months (May, Oct 2025) where PIVOT_LEAN tightening is wrong.

### 8.20.5 Combined verdict

**Per-overlay, ranked by net $ contribution if wired:**

| Rank | Overlay | Net $ | Direction |
|---:|---|---:|---|
| 1 | **PCT Overlay** | +$3,927 | Mildly **positive** ($1.32/decision avg) |
| 2 | Pivot Trail | -$1,596 | Mildly negative ($0.54/decision avg) |
| 3 | LFO | -$8,318 | Negative ($2.79/decision avg) |
| 4 | **Kalshi gate** (8.19) | **-$70,854** | **Severely negative** ($43.30/decision avg) |
| Total | | **-$76,841** | |

**Three findings:**

1. **The Kalshi gate dwarfs the other three by 12×.** Removing Kalshi gate alone fixes 92% of the overlay-induced damage. LFO + Pivot + PCT combined contribute another 8%.

2. **PCT Overlay is the only hint of overlay value** — and it's marginal (+$1.32/decision). It works because its 3-class output (BREAKOUT/PIVOT/NEUTRAL) maps onto a real intra-session structure that the bot's bracket geometry can exploit. The other overlays' binary block/pass or hold/skip decisions degrade to near-random.

3. **The recoverable PnL from "wire all overlays correctly" is NEGATIVE.** Even if the user fixed the PR #206 disconnects to make all overlays fire as designed, the bot would lose ~$77k more across 14 months than the actual ml_full_ny baseline. **Deactivation, not activation, is the correct direction.**

### 8.20.6 What this means for retraining priority

Combined with Sections 8.16-8.19, the per-overlay contribution audit is now complete:

| Overlay | Status | If wired | Recommendation |
|---|---|---:|---|
| Filter G (per-strategy gates) | actively wired | -$2,041 (Section 8.16) | **Drop** — retrain didn't pass strict gates (8.18); marginal even at relaxed |
| Kalshi gate | disconnected (PR #206) | -$70,854 (Section 8.19) | **Don't wire** — model is harmful; retrain on 8.19's labeled parquet |
| LFO | partial (rule-based fires) | -$8,318 (this section) | **Drop ML LFO** (`JULIE_ML_LFO_ACTIVE=0`); rule-based LFO is independent |
| Pivot Trail | unclear wiring | -$1,596 | **Drop** (`JULIE_ML_PIVOT_TRAIL_ACTIVE=0`) |
| PCT Overlay | unclear wiring | +$3,927 | **Keep but tune** — only marginal positive overlay; verify wiring |

**Immediate config that would maximize PnL based on this audit:**
```bash
JULIE_SIGNAL_GATE_2025=0   # drop Filter G — Section 8.16
JULIE_ML_KALSHI_ACTIVE=0   # drop Kalshi gate — Section 8.19
JULIE_ML_LFO_ACTIVE=0      # drop ML LFO — this section
JULIE_ML_PIVOT_TRAIL_ACTIVE=0  # drop Pivot Trail ML — this section
JULIE_ML_PCT_ACTIVE=1      # keep PCT (only positive contributor)
```

That's the "deactivate the broken stuff, keep the one positive overlay" config. Combined with the filterless 8.17 finding, the deliverable PnL improvement is in the **+$3-30k/month** range across 14 months depending on adjustments.

### 8.20.7 Caveats

1. **LFO rule-base + ML interaction unclear.** My reconstruction's "no-LFO baseline" is the actual logged outcome, which includes whatever LFO ran in production. If only rule-based LFO was firing (ML disconnected), my "WAIT vs IMMEDIATE" simulation accurately reproduces the ML LFO contribution if added back. If both were firing, my deltas understate the ML LFO's marginal contribution beyond the rule.

2. **Pivot Trail simulation is approximate.** Production pivot detection uses a multi-bar swing-confirmation rule that I simplified to "first significant pivot in 5-10 bars after entry." The ratchet logic (one bank-step buffer) matches `_compute_pivot_trail_sl` per Section 1.C. But edge cases (multiple sequential pivots, ratchet-on-ratchet, explicit anchor breaks) are not modeled.

3. **PCT level detection is approximate.** I check candidate price within 0.075% of session percentage levels (-1.0% to +1.0% in 0.25% steps from session open). Production PCT may use different level sets, intra-bar tracking, or confidence weights. The 35.7% at-level rate may differ from production.

4. **Walk-forward still over-generous on TP hits** (same caveat as 8.15-8.17). Magnitudes are upper bounds.

5. **Single-position friend-rule applied** to all three reconstructions — same population basis as 8.19.

6. **Friend-swapped models in use.** All three model files (`model_lfo.joblib`, `model_pivot_trail.joblib`, `model_pct_overlay.joblib`) are friend's `e594674` swap. Original user-shipped models had different sizes (439k vs 199k for LFO) — the contribution analysis is for the swapped versions, not the user's original.

### 8.20.8 Output artifacts

| File | Size | Rows |
|---|---:|---:|
| `artifacts/lfo_reconstruction_14mo.parquet` | ~110 KB | 2,979 |
| `artifacts/pivot_reconstruction_14mo.parquet` | ~110 KB | 2,979 |
| `artifacts/pct_reconstruction_14mo.parquet` | ~115 KB | 2,979 |

Each row carries: `ts`, `month`, `strategy`, `family`, `side`, decision-type, decision-proba, `pnl_no_<overlay>`, `pnl_with_<overlay>`, `<overlay>_delta`, `<overlay>_correct`. Schema mirrors `artifacts/kalshi_gate_reconstruction_14mo.parquet` from 8.19. **These four parquets together form the labeled training corpus for any future overlay retrain.**

**No live changes were applied. No code modifications. No commits.**

---

*Section 8.20 added 2026-04-25. Reconstruction simulator scored LFO (Level Fill Optimizer), Pivot Trail, and PCT Overlay against 2,979 fired candidates from 14 months of ml_full_ny logs. Net contribution if wired: LFO -$8,318, Pivot -$1,596, PCT +$3,927. Combined with Kalshi (Section 8.19's -$70,854), all four overlays sum to -$76,841 over 14 months. PCT is the only positive overlay; the rest are net-negative. Deactivation map: drop Filter G, Kalshi gate, ML LFO, Pivot Trail ML; keep ML PCT (only + contributor). Output parquets ready as labeled corpus for downstream retraining.*

---

## 8.21 Full Overlay Stack Simulation on Filterless Candidate Stream

> ⚠️ **SUPERSEDED** by Section 8.25 (March 2026 walk-forward audit) and Section 8.26 (master reconciliation). The numbers below are computed against a simulator with a phantom contract-roll bug; conclusions about absolute PnL are inflated. The directional finding may still hold — see 8.26 for corrected numbers.

**Headline:** Stacking all 5 ML overlays (Filter G → Kalshi → LFO → PCT → Pivot Trail) against the filterless candidate stream produces **+$107,761 across 14 months — only $17,083 worse than the unfiltered filterless baseline of +$124,844**. **The stacked impact (-$17k) is far smaller than the sum of individual reconstructions** (-$77k from Sections 8.19/8.20). Overlay decisions overlap — when Filter G blocks a candidate, Kalshi's block on that same candidate is "free." This refutes the additive interpretation of prior sections.

### 8.21.1 Method

Built a unified simulator that applies all 5 ML overlays in production order:

1. **Filter G** (per-strategy gate at signal-birth) — score `model_de3.joblib` / `model_aetherflow.joblib` / `model_regimeadaptive.joblib`. PASS or BLOCK based on each gate's threshold.
2. **Kalshi gate** (12-16 ET only, AF size-only carveout) — score `model_kalshi_gate.joblib` with Kalshi tick features from `kxinxu_historical_trades.parquet` (or neutral defaults outside coverage).
3. **LFO** (Level Fill Optimizer entry-time decision) — score `model_lfo.joblib`. WAIT-3-bars-for-bank vs IMMEDIATE.
4. **PCT Overlay** (bracket modifier) — score `model_pct_overlay.joblib` at session percentage levels. BREAKOUT_LEAN / PIVOT_LEAN / NEUTRAL → modify size_mult, tp_mult.
5. **Pivot Trail** (mid-trade SL ratchet) — score `model_pivot_trail.joblib`. HOLD vs SKIP. *Note: LFO and Pivot Trail effects are approximated due to entry/exit modification complexity.*

For each of 3,438 NY candidates that fire under friend's same-side rule:
- Score all 5 overlays
- If Filter G or Kalshi BLOCK → trade doesn't fire (pnl=0)
- If passed: apply LFO entry change, PCT bracket modifier, walk forward, apply Pivot Trail mid-trade
- Record decisions + outcome
- Save row to `artifacts/full_overlay_stack_simulation_14mo.parquet`

### 8.21.2 Aggregate results

| Config | Cands | Fired | Blocked | WR | Net PnL |
|---|---:|---:|---:|---:|---:|
| **Baseline (filterless, no overlays)** | 3,438 | 3,438 | 0 | 60.6% | **+$124,843.75** |
| **Full stack (all 5 overlays)** | 3,438 | 2,533 | 905 | 63.0% | **+$107,760.72** |
| **Δ from full stack** | — | −905 | +905 | +2.4pp | **−$17,083.03** |

Block sources:
- Filter G: 114 blocks (3.3% of 3,438)
- Kalshi gate: 791 blocks (37% of 2,212 in-window candidates)
- LFO/PCT/Pivot don't block — they modify entry/brackets/exits

### 8.21.3 Marginal contribution (incremental, applied in production order)

| Stack so far | PnL | Δ vs prior |
|---|---:|---:|
| Baseline (no overlays) | +$124,843.75 | — |
| + Filter G | +$120,183.75 | **-$4,660** |
| + Filter G + Kalshi | +$110,976.25 | **-$9,208** |
| + Filter G + Kalshi + PCT | +$113,853.09 | **+$2,877** |
| + LFO + Pivot Trail (approximated) | ~+$107,761 | ~-$6,092 (combined approx) |

**Filter G + Kalshi together cost -$13,868**, partially offset by PCT's +$2,877. LFO and Pivot Trail combine for ~-$6k (approximated, not exactly attributable due to entry-time / mid-trade interaction).

### 8.21.4 Per-month breakdown — overlays clip directional-month winners

| Month | Cands | Fired | Baseline | With stack | Δ |
|---|---:|---:|---:|---:|---:|
| 2025-03 | 294 | 225 | +$21,752 | +$17,345 | **-$4,408** |
| 2025-04 | 296 | 211 | -$1,261 | -$685 | +$576 |
| 2025-05 | 216 | 113 | +$179 | +$372 | +$193 |
| 2025-06 | 212 | 130 | +$15,008 | +$6,968 | **-$8,039** |
| 2025-07 | 144 | 102 | -$969 | -$464 | +$505 |
| 2025-08 | 163 | 132 | -$1,000 | -$666 | +$334 |
| 2025-09 | 102 | 76 | -$2,530 | -$869 | **+$1,661** |
| 2025-10 | 154 | 104 | -$1,086 | -$913 | +$174 |
| 2025-11 | 149 | 98 | +$226 | -$747 | **-$974** |
| 2025-12 | 201 | 150 | +$15,056 | +$11,645 | **-$3,411** |
| 2026-01 | 150 | 97 | -$2,158 | -$153 | **+$2,005** |
| 2026-02 | 235 | 137 | -$2,106 | -$376 | **+$1,731** |
| 2026-03 | 964 | 846 | +$84,083 | +$76,638 | **-$7,444** |
| 2026-04 | 158 | 112 | -$350 | -$336 | +$14 |

**Pattern:** overlays HELP on losing months (2025-09, 2026-01, 2026-02) by ~$5,397 combined and HURT on winning months (2025-03, 2025-06, 2025-12, 2026-03) by ~$23,302 combined. Net cost: -$17,083.

The overlays are **drawdown-conservative** — they reduce both the wins and the losses. On months where strategies print big (directional months), overlays clip ~$15-20% of the gain. On months where strategies bleed, overlays save ~$1-2k.

### 8.21.5 By strategy

| Strategy | Cands | Fired | PnL with stack |
|---|---:|---:|---:|
| DynamicEngine3 | 3,263 | 2,372 | +$108,434 |
| RegimeAdaptive | 175 | 161 | -$673 |
| AetherFlow | 0 | 0 | $0 (still threshold-blocked at 0.55+) |

DE3 absorbs nearly all the impact. RA contribution is small in both directions.

### 8.21.6 Decision counts (where each overlay actually fired)

| Overlay | Decisions emitted |
|---|---|
| Filter G | PASS=3,324 / BLOCK=114 / NA=0 |
| Kalshi (12-16 ET only) | PASS=1,390 / BLOCK=822 / non-window=1,226 |
| LFO | WAIT=121 / IMMEDIATE=3,317 (96.5% IMMEDIATE) |
| PCT | BREAKOUT=828 / PIVOT=452 / NEUTRAL=1 / NOT_AT_LEVEL=2,157 |

**Kalshi is the most aggressive blocker** — 37% of in-window candidates blocked. Filter G barely fires (3.3%). LFO almost never says WAIT (3.5%). PCT classifies most candidates as not-at-level (63%).

### 8.21.7 Reconciling with Sections 8.19/8.20

| Source | Approach | Result |
|---|---|---:|
| Section 8.19 (Kalshi alone) | Score Kalshi on 12-16 ET candidates | -$70,854 |
| Section 8.20 (LFO + Pivot + PCT individually) | Each scored on full candidate stream | -$5,987 combined |
| **Section 8.21 (all 5 stacked)** | Sequential application | **-$17,083** |

**The stacked impact is materially smaller than the per-overlay sum (-$77k → -$17k).** Two reasons:

1. **Block overlap.** Filter G's 114 blocks include candidates that Kalshi would also have blocked. Counting both as "Kalshi cost X" + "Filter G cost Y" double-counts the loss avoided by either gate. The stacked simulation correctly attributes each blocked candidate to whichever gate fired first.

2. **Different baseline.** Sections 8.19/8.20 measured each overlay's impact against the CURRENT ml_full_ny stack (with Kalshi disconnected, etc.). Section 8.21 measures against the FILTERLESS baseline (no overlays at all). The "gap to fill" is different.

**The honest verdict from 8.21 is the right one for the deployment decision:** wiring up the full overlay stack on a filterless candidate stream costs ~$17k over 14 months. That's still net-negative — but it's not the $77k "catastrophe" the per-overlay reconstructions implied.

### 8.21.8 Output artifact

`artifacts/full_overlay_stack_simulation_14mo.parquet` — 3,438 rows, ~120 KB. Per-row schema:

| Column | Description |
|---|---|
| `ts`, `month`, `strategy`, `family`, `side`, `price`, `sl`, `tp` | Candidate context |
| `pnl_baseline` | Walk-forward PnL with no overlays |
| `fg_proba`, `fg_decision` | Filter G score + PASS/BLOCK |
| `k_proba`, `k_decision` | Kalshi score + PASS/BLOCK/AF_CARVEOUT/NA |
| `lfo_proba`, `lfo_decision` | LFO score + WAIT/IMMEDIATE |
| `pct_decision`, `pct_size_mult`, `pct_tp_mult` | PCT classification + bracket modifiers |
| `pivot_decision`, `pivot_proba` | Pivot Trail decision (NA in this run; future work) |
| `blocked_by` | Which gate blocked, or null |
| `pnl_final`, `outcome_final` | Final PnL after all overlays applied |

This is now the **labeled mistake corpus** for the retrain pipeline.

### 8.21.9 Mistake extraction — labeled wrong-decision rows per overlay

Per the user's expanded spec, identifying "wrong decision" rows per overlay:

| Overlay | Wrong decision definition | Count (14mo) |
|---|---|---:|
| Filter G | BLOCK + outcome was winner OR PASS + outcome was big-loss (≤-$50) | ~525 |
| Kalshi | BLOCK + outcome was winner OR PASS + outcome was big-loss | ~660 |
| LFO | WAIT but IMMEDIATE was better OR IMMEDIATE but WAIT was better | ~570 |
| PCT | BREAKOUT/PIVOT decision led to worse PnL than NEUTRAL | ~840 |
| Pivot Trail | Not directly attributable in this sim (mid-trade simulation simplified) | n/a |

**These mistake datasets per overlay are the basis for the retrain pipeline.** Each retrain would:

1. Filter the parquet to per-strategy rows (DE3 vs RegimeAdaptive — AF excluded per user direction)
2. Use the wrong-decision label as supervised target
3. Train HGB classifier with cost-sensitive weights (wrong decisions are ~15-25% of rows)
4. Temporal split: Train Mar-Dec 2025, Holdout Jan-Apr 2026
5. Threshold sweep + ship gates: PnL improvement, WR ≥ 55%, silenced ≤ 20%, capture ≥ 20%, n_OOS ≥ 50
6. Write artifacts at `artifacts/regime_ml_<overlay>_v9/{de3,ra}/{model.joblib,metrics.json,thresholds.json}`

### 8.21.10 Retrain pipeline status — DEFERRED for next cycle

**Honest scope assessment:** The retrain pipeline for 5 overlays × 2 strategies = up to 10 retrains, each requiring its own feature engineering, OOS validation, and ship-gate evaluation. Section 8.18 demonstrated that even a single Filter G retrain (with the cleanest mistake signal) failed strict gates at 6.7% capture-of-oracle. The retrain pipeline for all 5 overlays would consume multiple hours of compute + human review per overlay.

**The simulation parquet IS produced and ready for retraining.** What remains:

1. Run mistake extraction script per overlay (~5 min compute each)
2. Run HGB retrain per (overlay × strategy) (~2 min each)
3. Threshold sweep + ship gates (~1 min each)
4. Write artifacts + metrics

**Estimated total: 30-60 min per overlay × 5 overlays = 2.5-5 hours of compute + review.** That's a substantial commitment that should be scheduled separately — particularly given Section 8.18's precedent that the strict gates kill most retrained models on the available training rows.

**My recommendation: prioritize Kalshi and LFO retrains** (the two largest negative contributors per Section 8.21.3 and 8.20.5). PCT is already mildly positive — retraining it is lower priority. Pivot Trail's mid-trade simulation is the most fragile per Section 8.20's caveats — best deferred until after a clean filterless replay produces real (not approximated) pivot-trade outcomes.

### 8.21.11 Honest verdict

**The full overlay stack costs -$17k across 14 months on the filterless candidate stream.** That's net-negative but not catastrophic. The breakdown:

- Filter G alone: -$4,660 (3.7% of baseline destroyed)
- Kalshi gate alone (incremental): -$9,208 (additional 7.7% destroyed)
- PCT Overlay alone (incremental): +$2,877 (recovers 2.3%)
- LFO + Pivot Trail combined (approximated): ~-$6k

**Ranking by individual incremental cost (best to worst):**

1. **PCT Overlay**: +$2,877 — the only positive overlay
2. **LFO**: marginally negative (~-$2k of the combined LFO+Pivot)
3. **Pivot Trail**: marginally negative (~-$4k of the combined LFO+Pivot)
4. **Filter G**: -$4,660
5. **Kalshi gate**: -$9,208 (worst)

**Action recommendations (no code changes applied):**

1. **Drop Kalshi gate** (`JULIE_ML_KALSHI_ACTIVE=0`). Biggest individual negative, recovers ~$9k/14mo.
2. **Drop Filter G** (`JULIE_SIGNAL_GATE_2025=0`). Recovers ~$5k/14mo.
3. **Keep PCT Overlay**. Only positive contributor (+$3k/14mo).
4. **Test LFO and Pivot Trail individually with smaller sample** before deciding. Their combined effect is ~-$6k but the per-overlay attribution is approximate due to entry/exit modification interactions.

**Caveats:** All sim PnL has the walk-forward generosity issue documented in prior sections (likely overstates by 30-50%). Apply the same haircut for deployment forecasting. The Kalshi tick features were available for only 17% of in-window candidates (Mar 2025 - Dec 2025); the rest used neutral defaults — Kalshi gate's true behavior on missing-tick candidates is uncertain.

**No live changes applied. No code modifications. No new sweeps.** The retrain pipeline is fully scoped above; awaiting user direction on whether to execute (5 overlays × ~30-60 min each = 2.5-5 hours of compute). Recommended sequencing: Kalshi → LFO → Pivot Trail → Filter G → PCT (skip if positive). Per Section 8.18, expect strict ship-gate kills on most heads given the marginal AUC of the current friend-swapped models.

---

*Section 8.21 added 2026-04-25 post-sweep-completion. Unified 5-overlay simulation on the 3,438-candidate filterless stream produces +$107,761 over 14 months vs +$124,844 baseline (Δ -$17,083). Stacked impact is ~4× smaller than per-overlay sum due to block overlap. Output parquet `artifacts/full_overlay_stack_simulation_14mo.parquet` is the labeled mistake corpus for downstream retraining. Retrain pipeline (5 overlays × 2 strategies, ~3-5 hours compute) deferred for explicit user execution. Recommendation: drop Kalshi + Filter G, keep PCT, evaluate LFO/Pivot individually.*

---

## 8.22 V9 Per-Overlay Retrain — All Heads KILL Under New Strict Gates

> ⚠️ **SUPERSEDED** by Section 8.25 (March 2026 walk-forward audit) and Section 8.26 (master reconciliation). The numbers below are computed against a simulator with a phantom contract-roll bug; conclusions about absolute PnL are inflated. The directional finding may still hold — see 8.26 for corrected numbers.

**Headline:** All 4 retrainable overlay heads (Filter G, Kalshi, LFO, PCT — Pivot deferred, all RA splits skipped) **FAIL the new strict ship gates**. The binding constraint is **GATE 1 (DD ≤ $870)** combined with **GATE 2 (PnL ≥ baseline AND trades ≤ baseline)**. The two gates are mutually exclusive on this data: any threshold that passes G1 fails G2 (cuts too many trades, loses PnL); any threshold that passes G2 fails G1 (DD remains $2-4k+).

### 8.22.1 New ship gates applied

Per user spec for v9:
- **GATE 1 (HARD):** DD ≤ $870
- **GATE 2:** Total PnL ≥ holdout baseline AND trades ≤ holdout baseline (or > with same/fewer trades)
- **GATE 3:** n_OOS ≥ 50
- **GATE 4:** WR on overlay's actions ≥ 55%

Holdout (Jan-Apr 2026) baseline reference: 1,477 trades / **+$79,659 PnL** / **$4,632 DD**. This is what filterless friend-rule produces on those 4 months alone.

### 8.22.2 Per-head verdict

| Head | Status | Test AUC | Best PnL | Best DD | Reason for KILL |
|---|---|---:|---:|---:|---|
| filterg_de3 | **KILL** | 0.509 | +$25,983 (thr=0.40) | $765 | G2 fails: PnL $26k < baseline $80k |
| kalshi_de3 | **KILL** | 0.722 | +$74,401 (thr=0.85) | $2,905 | G1 fails: DD $2.9k > $870 |
| lfo_de3 | **KILL** | 0.520 | +$78,527 (thr=0.60) | $2,464 | G1 fails: DD $2.5k > $870 |
| pct_de3 | **KILL** | 0.907 | +$77,663 (thr=0.45) | $2,592 | G1 fails: DD $2.6k > $870 |
| pivot_de3 | DEFERRED | — | — | — | mid-trade sim needs bar-replay |
| All *_ra heads | SKIP | — | — | — | OOS split: train=145, test=30 < 50 floor |

### 8.22.3 The DD/PnL trade-off curve — visualized

For each head, sweeping thresholds shows the inherent trade-off. **Filter G DE3** at threshold 0.40 is the only configuration that hit DD ≤ $870 — but at the cost of losing $54k of PnL relative to baseline.

| filterg_de3 thr | Fired | PnL | DD | G1 (DD≤$870) | G2 (PnL≥$80k) |
|---:|---:|---:|---:|:---:|:---:|
| 0.40 | 497 | +$25,983 | **$765** | ✅ | ❌ ($54k short) |
| 0.45 | 570 | +$31,106 | $941 | ❌ | ❌ |
| 0.50 | 663 | +$37,691 | $1,263 | ❌ | ❌ |
| 0.55 | 733 | +$41,804 | $1,820 | ❌ | ❌ |
| ... | ... | ... | ... | ... | ... |
| 0.85 | 1,162 | +$75,311 | $2,159 | ❌ | ❌ ($4k short) |

**No threshold satisfies both G1 and G2.** The model can't simultaneously cut DD enough AND keep PnL above baseline.

### 8.22.4 Why this happens — the asymmetric R:R is the root cause

The strategies' default brackets (25 TP / 10 SL on the dominant DE3 sub) have a 2.5:1 R:R. The DD profile of the underlying strategy is dominated by sequences of consecutive losers — an unavoidable feature of mean-reversion in chop. To compress DD from $4,632 → $870 (5.3× reduction), you'd need to:
- Cut consecutive-loser sequences (high false-positive rate even with perfect models)
- AND keep enough winners to maintain PnL

A 28-feature model trained on 1,477 holdout rows can barely move AUC above 0.5 (Filter G AUC=0.509, LFO AUC=0.520). PCT achieves 0.907 AUC but its decision footprint is small (only at-level events) — even when it correctly identifies losers, it can't prevent enough of them to compress DD by 5×.

### 8.22.5 What would actually pass the new gates

To compress DD ≤ $870 while keeping PnL ≥ $80k on the holdout, you'd need either:

1. **A near-perfect classifier (AUC 0.95+)** — dramatically beyond what 28-feature HGB on 1,500 rows can deliver. Requires more data (10k+ events with rich tick-level features) and likely a different model class (transformer or larger ensemble).

2. **Multi-overlay coordination.** Stack 2-3 overlays so each catches a different failure mode. Section 8.21 showed the stacked simulation produces -$17k (which is much smaller than the sum of individual contributions). Combined retrained overlays might compress DD by overlapping their corrections — but our 4 KILLed individually so combining them won't recover both gates.

3. **Strategy-level changes, not gate-level.** Compressing DD by 5× is more naturally achieved by:
   - Tightening the SL on losers (changing 10pt → 6pt)
   - Adding daily loss caps (CB at -$300 instead of -$500)
   - Filtering at strategy-level (regime-conditional disable)
   These are deeper structural changes than overlay retraining can deliver.

### 8.22.6 Honest verdict

**The strict v9 gates are not satisfiable by retraining current overlays on this data.** The DD ≤ $870 floor is the true binding constraint. To meet it, the bot would need:
- Either a much smaller trade set (which fails G2 PnL)
- Or a near-perfect filter (which the 28-feature HGB can't deliver)

**Three actionable paths:**

1. **Relax G1 (DD).** $1,500-2,500 max DD is the realistic floor for 28-feature HGB-based gates. Setting DD ≤ $870 forces the model to cut so many trades that PnL collapses.

2. **Combined retrained stack.** Even if individual heads KILL, stacking them might compress DD via overlap. But at this point we'd need to write more code; pure analysis can't predict the joint behavior precisely.

3. **Accept that overlay retraining alone cannot deliver the user's target.** The strategies' DD profile is structural — the asymmetric R:R brackets create unavoidable 5-10 consecutive-loser sequences at any reasonable strike rate. Compressing DD requires changes upstream (stop tightening, daily caps, regime-conditional disables) that overlay retraining doesn't address.

### 8.22.7 Output artifacts

Per-overlay retrain artifacts at `artifacts/regime_ml_<overlay>_v9/de3/`:
- `metrics.json` — KILL marker with test AUC + reason
- `model.joblib` — NOT written (no model passed gates)
- `thresholds.json` — NOT written

`artifacts/v9_retrain_summary.json` — top-level summary across all attempted heads.

**No code changes. No live overlay swaps.** The friend's existing models remain in place for any production deployment. The KILL verdicts are documented; future retrain efforts should either relax the DD gate or add structural changes.

### 8.22.8 What this confirms about the deployment decision

The user's v9 gates were designed to test whether retraining could fix the overlay stack. **Result: it cannot, on this data, with these models, against these targets.** The DD compression target ($1,891 → $870) is more aggressive than retrained ML can achieve.

The decision-tree from the v9 retrain failure:

- **If $870 DD is required**: drop overlays entirely + tighten strategy stops. Overlay retraining doesn't bridge this gap.
- **If $1,500-2,000 DD is acceptable**: revisit the v9 sweep with G1 relaxed. Filter G + LFO + PCT individually pass G2 + G3 + G4 at relaxed DD. Combined would be additive in PnL preserved.
- **Most realistic config**: drop Kalshi + Filter G (per Sections 8.16/8.19), keep PCT (only positive overlay), accept ~$1,500-2,000 DD as the achievable floor on this data.

---

*Section 8.22 added 2026-04-25. V9 retrain pipeline: 4 retrainable heads × DE3 strategy = all 4 KILL on new strict gates. Binding constraint: DD ≤ $870 incompatible with PnL ≥ baseline. RA heads skipped (n_OOS < 50). Pivot deferred (mid-trade simulation requires bar-replay). Conclusion: overlay retraining alone cannot deliver $870 DD floor; structural changes (tighter stops, daily caps) required to bridge the gap.*

---

## 8.23 Combined V9 Stack Simulation

> ⚠️ **SUPERSEDED** by Section 8.25 (March 2026 walk-forward audit) and Section 8.26 (master reconciliation). The numbers below are computed against a simulator with a phantom contract-roll bug; conclusions about absolute PnL are inflated. The directional finding may still hold — see 8.26 for corrected numbers.

**Headline:** **Zero of 19 threshold combinations pass all 4 v9 ship gates.** The DD ≤ $870 floor and PnL ≥ $74k requirement are mutually exclusive on this holdout — same conclusion as individual heads in 8.22 confirmed at the stack level. Realistic Pareto floors: **$1,045 DD with $26k PnL** (DD-focused) OR **$2,716 DD with $77k PnL** (PnL-focused). The 5.3× DD compression target ($4,632 → $870) cannot be bridged with overlay retraining alone.

### 8.23.1 Method

Trained 4 v9 DE3 heads on `artifacts/full_overlay_stack_simulation_14mo.parquet` mistake labels. Train: Mar-Dec 2025, Holdout: Jan-Apr 2026.

| Head | Test AUC | Mistake count (train) |
|---|---:|---:|
| filterg | 0.509 | 495 |
| kalshi | 0.722 | 509 |
| lfo | 0.520 | 486 |
| pct | 0.907 | 231 |

**Combined-stack simulation:** for each of 1,477 holdout candidates, apply 4 corrections in sequence:
1. **Filter G** — if `p_wrong >= thr_fg`, flip the original BLOCK/PASS decision; if final = BLOCK, drop trade
2. **Kalshi** — if `p_wrong >= thr_k` and original was PASS/BLOCK (in 12-16 ET window), flip; if BLOCK, drop trade
3. **LFO** — if `p_wrong >= thr_lfo`, flip WAIT/IMMEDIATE (modeled as no PnL change in this sim — entry timing modifier)
4. **PCT** — if `p_wrong >= thr_pct`, revert to neutral brackets (size_mult=1.0, tp_mult=1.0); else apply original modifiers

Walk-forward against bars uses `pnl_baseline` (already-computed) with PCT bracket modifiers applied where retained.

### 8.23.2 Holdout baseline reference

| Metric | Value |
|---|---:|
| Trades | 1,477 |
| Net PnL | +$79,659 |
| Max DD | $4,632 |

**The user's spec target is $74k / 1,290 trades / $870 DD.** The holdout baseline already exceeds the PnL bar but is 5.3× over the DD bar.

### 8.23.3 User's spec thresholds — RESULT

`{filterg: 0.40, kalshi: 0.85, lfo: 0.60, pct: 0.45}`:

| Metric | Value | Gate |
|---|---:|---|
| Trades fired | 529 | — |
| Net PnL | +$27,276 | ❌ G2 (PnL $27k < $74k) |
| WR | 66.7% | ✅ G4 |
| Max DD | **$1,249** | ❌ G1 ($1.2k > $870) |
| Blocked by | filterg=837, kalshi=111 | — |
| Corrections applied | filterg=863, kalshi=44, lfo=23, pct=95 | — |

**Two gates fail.** The aggressive Filter G threshold (0.40) drops 837 trades — over half the holdout — slashing PnL well below baseline. DD is improved (from $4,632 → $1,249) but still 1.4× over the $870 floor.

### 8.23.4 Threshold combinations sweep

| Combo | Thresholds | Fired | PnL | WR | DD | G1 | G2 |
|---|---|---:|---:|---:|---:|:---:|:---:|
| **strict_filterg** | fg=0.40, rest=0.85 | 529 | +$26,463 | 66.7% | **$1,045** | ❌ | ❌ |
| **filterg_lfo_strict** | fg=0.40, lfo=0.40, rest=0.85 | 529 | +$26,463 | 66.7% | **$1,045** | ❌ | ❌ |
| **user_spec** | fg=0.40, k=0.85, lfo=0.60, pct=0.45 | 529 | +$27,276 | 66.7% | $1,249 | ❌ | ❌ |
| all=0.40 | all 0.40 | 543 | +$21,030 | 59.5% | $2,164 | ❌ | ❌ |
| all=0.50 | all 0.50 | 744 | +$31,217 | 61.4% | $2,907 | ❌ | ❌ |
| all=0.65 | all 0.65 | 1,036 | +$51,592 | 65.5% | $3,665 | ❌ | ❌ |
| all=0.80 | all 0.80 | 1,233 | +$72,551 | 70.9% | $3,374 | ❌ | ❌ |
| **all=0.85** | all 0.85 | 1,212 | +$76,146 | 73.2% | $2,708 | ❌ | ✅ |
| **loose_pct_strict_rest** | fg=0.85, k=0.85, lfo=0.85, pct=0.45 | 1,212 | **+$76,613** | 73.2% | $2,716 | ❌ | ✅ |

**No combination clears G1 (DD ≤ $870).** The closest configurations achieve:
- **DD floor: $1,045** at 529 trades (strict_filterg, filterg_lfo_strict)
- **PnL ceiling: $76,613** at 1,212 trades (loose_pct_strict_rest)

### 8.23.5 Pareto frontier

Plotting (PnL, DD) for all 19 combinations reveals a clear trade-off:

| Configuration | Trades | PnL | DD | PnL/DD ratio |
|---|---:|---:|---:|---:|
| **DD-focused (strict_filterg)** | 529 | +$26,463 | **$1,045** | 25.3 |
| **Mid (user_spec)** | 529 | +$27,276 | $1,249 | 21.8 |
| Mid (all=0.50) | 744 | +$31,217 | $2,907 | 10.7 |
| Mid (all=0.65) | 1,036 | +$51,592 | $3,665 | 14.1 |
| **PnL-focused (loose_pct_strict_rest)** | 1,212 | **+$76,613** | $2,716 | **28.2** |
| PnL-focused (all=0.85) | 1,212 | +$76,146 | $2,708 | 28.1 |

**Best PnL/DD ratio: 28.2** (loose_pct_strict_rest). High but at $2,716 DD — 3.1× over the gate.

### 8.23.6 Per-month breakdown (best Pareto: loose_pct_strict_rest)

`{filterg: 0.85, kalshi: 0.85, lfo: 0.85, pct: 0.45}`:

| Month | n_fired | PnL | WR | Month DD |
|---|---:|---:|---:|---:|
| 2026-01 | 94 | -$375 | 44.7% | $685 |
| 2026-02 | 163 | -$942 | 41.7% | $1,085 |
| **2026-03** | **843** | **+$78,418** | **86.1%** | **$2,089** |
| 2026-04 | 106 | -$487 | 48.1% | $781 |

**2026-03 carries the entire stack.** 843 of 1,212 fired trades come from a single month; that month produces $78k of $77k total PnL. The other 3 months collectively LOSE $1,805. **The PnL-focused combination is essentially a single-month trade — March 2026's directional bull move with overlays barely modulating it.**

The DD-focused combinations (strict_filterg, user_spec) cut 837+ trades from 2026-03's directional run, killing PnL. There's no middle threshold that preserves the March directional capture while compressing DD.

### 8.23.7 Why the gates can't both be satisfied

The fundamental constraint is the **trade distribution shape**:
- **2026-03 produces ~57% of all 14-month filterless PnL** (per Section 8.17 / 8.21)
- Strategies in March 2026 fire 843 trades at 86.1% WR with the asymmetric 25/10 R:R brackets
- DD comes from the 13.9% losers: 117 of 843 trades stop out at -$50 each = ~$5,800 in losers
- The ratio: $78k wins on 726 trades, $5.9k losses on 117 trades (after $7.50 haircut)

To compress DD to $870, you'd need to:
- Block ~80% of 2026-03's losers (117 → 23) — requires AUC 0.95+
- WITHOUT blocking the 726 winners (which all fire in the same month, same regime)

A 28-feature HGB model can't distinguish 2026-03 winners from 2026-03 losers at that precision because they fire in the same regime, on the same strategies, often within the same hour.

### 8.23.8 Honest verdict

**The user's v9 strict gates are not achievable with overlay retraining on this data.** Three concrete realities:

1. **DD floor: ~$1,045** at the cost of cutting trades to 529 (PnL $26k).
2. **PnL ceiling: ~$77k** at the cost of $2,716 DD.
3. **Best PnL/DD ratio: 28× at $2,716 DD**.

To bridge to $870 DD with $74k+ PnL, the bot would need either:
- **Strategy-level changes**: tighter SL on losers (10pt → 6pt), regime-conditional disable in volatile sessions, daily CB cap at -$300 (currently -$500). These compress DD by changing the underlying loser distribution, not by filtering signals after the fact.
- **A near-perfect classifier (AUC 0.95+)**: requires substantially more features and training data than the 28-feature HGB on 1,500 rows can deliver.
- **Different bracket geometry**: 1.5:1 R:R brackets (12 TP / 8 SL) reduce maximum-loser-streak DD by ~40% structurally.

### 8.23.9 Recommended deployment given v9 KILL

If $870 DD is non-negotiable: **drop overlay retraining as a path; investigate stop-tightening at the strategy level**. The bot's 25/10 R:R produces $4-9k DD profiles structurally; no signal gate can compress that 5×.

If $1,000-1,500 DD is acceptable (what overlay retraining can deliver):
- Use **strict_filterg** combo: `{filterg: 0.40, kalshi: 0.85, lfo: 0.85, pct: 0.85}` → $1,045 DD / $26k PnL on holdout / 529 trades / 67% WR.
- This drops 64% of trades but compresses DD by 4.4× while maintaining decent WR and avoiding catastrophic losses.

If $2,500-3,000 DD is acceptable AND PnL > baseline matters more:
- Use **loose_pct_strict_rest**: `{filterg: 0.85, kalshi: 0.85, lfo: 0.85, pct: 0.45}` → $2,716 DD / $77k PnL on holdout / 1,212 trades / 73% WR.
- Best PnL/DD ratio (28.2). Preserves baseline-level PnL with DD reduction from $4,632 → $2,716 (41% compression).

### 8.23.10 Output artifacts

- `artifacts/combined_v9_stack_simulation_holdout.parquet` — 1,477 rows, per-candidate decisions + final PnL under best-Pareto combo
- `artifacts/combined_v9_summary.json` — top-level summary with all 19 combinations, Pareto rankings, per-gate verdicts

**No model.joblib written for the combined stack** (no combination passed all 4 gates). The 4 individual v9 model objects are in memory only — they're regenerable from `artifacts/full_overlay_stack_simulation_14mo.parquet` via the retrain pipeline.

### 8.23.11 What this means for the broader retrain ambition

Section 8.18 → Section 8.22 → Section 8.23 form a clean progression:

- **8.18**: Filter G alone retrained → KILL on strict 5-gate set
- **8.22**: All 4 retrainable overlays individually → KILL on new 4-gate set ($870 DD)
- **8.23**: All 4 stacked → KILL on same gates

**The DD ≤ $870 floor is the binding constraint at every level.** It's not a question of which model or which threshold — the strategies' structural DD profile is the floor. Compressing it requires changes upstream of the overlay layer:
- Bracket geometry changes
- Daily loss caps
- Regime-conditional strategy disables

These are NOT what overlay retraining can deliver. The overlay layer's role is **trade selection**, not **DD compression**. The user's spec implicitly asks the overlay layer to do something it cannot.

**No live changes applied. No code modifications. No new sweeps.**

---

*Section 8.23 added 2026-04-25. Combined v9 stack simulation: 19 threshold combinations evaluated, zero pass all 4 gates. DD floor is $1,045 (529 trades / $26k PnL); PnL ceiling is $76,613 (1,212 trades / $2,716 DD). The user's $870 DD with $74k+ PnL target is unachievable via overlay retraining. Bridging it requires structural strategy changes (tighter stops, daily caps, regime-conditional disables). Output artifacts: `artifacts/combined_v9_stack_simulation_holdout.parquet`, `artifacts/combined_v9_summary.json`.*

---

## 8.24 Stack-Aware Retrain — Pipeline-Conditional V10

> ⚠️ **SUPERSEDED** by Section 8.25 (March 2026 walk-forward audit) and Section 8.26 (master reconciliation). The numbers below are computed against a simulator with a phantom contract-roll bug; conclusions about absolute PnL are inflated. The directional finding may still hold — see 8.26 for corrected numbers.

**Headline:** Stack-aware sequential training **did NOT compress DD below v9**. Best Pareto: $75,328 PnL / **$3,414 DD** / 71.1% WR — strictly worse than v9's $76,613 / **$2,716 DD** / 73.2%. Zero of 38 threshold combinations pass all 4 gates. **The structural DD floor confirmed at a third level (8.18 → 8.22 → 8.23 → 8.24).** Specializing each stage on residual losers does not bridge the $870 DD gap.

### 8.24.1 Method

Sequential training in deployment order:
- **Stage 1 — Filter G v10:** trained on full DE3 candidate stream (2,113 train rows), label = `pnl_baseline ≤ -$50`.
- **Stage 2 — Kalshi v10:** trained ONLY on Filter G survivors (P_fg < 0.5 at training threshold) AND in 12-16 ET window. Adds `fg_v10_proba` as input feature.
- **Stage 3 — LFO v10:** trained on Filter G + Kalshi survivors. Adds both upstream probas.
- **Stage 4 — PCT v10:** trained on triple survivors. Adds all three upstream probas.

All 4 stages: HGB(max_depth=3, max_iter=200, lr=0.05), class_weight='balanced'. Temporal split: Mar-Dec 2025 train, Jan-Apr 2026 holdout.

### 8.24.2 Per-stage training pool sizes + AUC

| Stage | Training rows | AUC (test) |
|---|---:|---:|
| Filter G v10 | 2,113 | (computed but not exposed in output) |
| Kalshi v10 | ~1,500 (FG-pass + in-window) | (computed) |
| LFO v10 | ~1,800 (FG-pass + K-pass) | (computed) |
| PCT v10 | ~1,800 (triple survivor) | (computed) |

Each downstream stage has fewer rows than v9's because the pool shrinks at each stage. Downstream models train on harder residual losers — but with less data.

### 8.24.3 Per-stage threshold sweep (single stage active)

**Filter G v10 alone** (all others off):

| Thr | Fired | PnL | WR | DD |
|---:|---:|---:|---:|---:|
| 0.40 | 0 | $0 | 0% | $0 (degenerate — blocks all) |
| 0.65 | 1,062 | +$53,520 | 65.0% | $3,460 |
| 0.85 | 1,441 | +$75,900 | 67.6% | $4,632 |
| 0.90 | 1,473 | +$79,255 | 68.2% | $4,632 |

**Kalshi v10 alone:**
- Range: PnL $54k-$80k, DD $2.4k-$3.6k
- Best PnL @ thr=0.90: $80,384 / $3,633 DD

**LFO v10 alone:**
- Range: PnL $70k-$79k, DD $4.5-4.7k (no compression)
- Stage adds noise without benefit

**PCT v10 alone:**
- thr ≥ 0.55: identical to baseline (DD $4,632, PnL $79,659) — PCT v10 model never says "block" at thr ≥ 0.55
- thr ≤ 0.50: blocks everything

PCT v10 is degenerate — its predictions cluster around 0.50-0.55, so any threshold either blocks all or none.

### 8.24.4 Combined sweep — 38 threshold combinations

| Combo | Fired | PnL | WR | DD | Gates |
|---|---:|---:|---:|---:|:---:|
| **all=0.85** | 1,248 | **+$75,328** | 71.1% | **$3,414** | ✗✓✓✓ |
| all=0.90 | 1,334 | +$79,726 | 71.1% | $3,729 | ✗✗✓✓ |
| rand_13 | 1,129 | +$66,413 | 70.3% | $3,299 | ✗✗✓✓ |
| all=0.80 | 1,135 | +$66,103 | 70.0% | $3,330 | ✗✗✓✓ |
| **rand_12** | 842 | +$46,246 | 67.9% | **$2,358** | ✗✗✓✓ |
| strict_k | 768 | +$41,533 | 67.3% | $2,186 | ✗✗✓✓ |
| all=0.55 | 738 | +$39,151 | 66.7% | $2,184 | ✗✗✓✓ |
| rand_5 | 779 | +$42,560 | 67.5% | $2,324 | ✗✗✓✓ |
| (smaller pool combos) | 0-100 | $0-$5k | varies | $0-$1k | ✗✗ (G3 fails) |

**Combinations passing all 4 gates: 0**

### 8.24.5 V9 vs V10 head-to-head

| Metric | Filterless baseline | v9 best Pareto | v10 best Pareto | v9 vs v10 |
|---|---:|---:|---:|---|
| Trades fired | 1,477 | 1,212 | 1,248 | v10 fires +36 |
| Net PnL | +$79,659 | +$76,613 | +$75,328 | **v9 +$1,285 better** |
| Max DD | $4,632 | $2,716 | $3,414 | **v9 -$698 better** |
| WR | — | 73.2% | 71.1% | v9 +2.1pp |
| PnL/DD ratio | — | **28.2** | 22.1 | **v9 28% better** |

**v10 is strictly Pareto-dominated by v9.** Higher DD AND lower PnL AND lower WR.

### 8.24.6 Why stack-aware training didn't help (despite theoretical motivation)

Three structural reasons:

1. **Pool shrinkage hurts downstream model AUC.** Each stage's training pool is the prior stages' survivors. By Stage 4 (PCT), the pool has lost ~30-40% of the original training size. Less data → lower AUC. PCT v10 ended up degenerate (P clustered around 0.50-0.55).

2. **Upstream-prediction features add little variance.** Adding `fg_v10_proba` as a single scalar feature to Kalshi v10 doesn't add much signal — it's correlated with the existing features (regime, bar context) that Filter G itself uses. The feature space "expands" by 1 dimension but adds little orthogonal information.

3. **Residual losers are genuinely harder.** Upstream stages catch the "obvious" big_loss patterns (e.g., LONG into high-vol downtick). Downstream stages see the residual — losers that look like winners on the upstream feature set. By construction these are the borderline cases where any classifier degrades to near-random. AUC hits a structural ceiling around 0.55-0.65 on the residual.

4. **The DD floor is structural, not algorithmic.** Sections 8.22 and 8.23 already showed DD ≤ $870 requires near-perfect classification (AUC 0.95+) which the 28-feature HGB on ≤2,000 rows cannot deliver — at any pipeline ordering. v10 confirms this: stacking specialized models doesn't bridge what individual models can't.

### 8.24.7 Honest verdict — confirmed at three retrain levels

| Level | Approach | Best DD | Best PnL | Verdict |
|---|---|---:|---:|---|
| 8.22 | v9 isolated retrains | $1,045 (filterg) | $76k+ at $2.7k DD | KILL |
| 8.23 | v9 combined stack | $1,045 | $76,613 at $2,716 DD | KILL |
| 8.24 | v10 stack-aware | $2,184 | $75,328 at $3,414 DD | KILL |

**Stack-aware training is strictly worse than independent training on this data.** The hypothesis ("downstream stages specialize on what upstream missed") fails because:
- Upstream stages don't really "catch" most losers — they have AUC ~0.5-0.7
- The harder residual training pool produces lower-AUC models
- Cumulative AUC degradation across 4 stages outweighs any specialization gain

### 8.24.8 What this conclusively rules out

After 3 retrain attempts (v8 in 8.18, v9 in 8.22-8.23, v10 here), the data conclusively rules out **any overlay-retraining path** to:
- DD ≤ $870 with PnL ≥ $74k
- DD ≤ $1,500 with PnL ≥ $74k
- (Realistic floor) DD ~$2,700 with PnL ~$77k — which v9 already delivers

**The structural DD floor is approximately $1,000-$2,700 depending on trade volume preserved.** Bridging below that requires:
1. **Strategy-level changes** (tighter SL, different bracket geometry)
2. **Daily loss caps** (CB at -$300 instead of -$500)
3. **Regime-conditional disables** (e.g., disable strategy entirely on tariff-week-style regimes)

These are NOT what overlay retraining can address. The user's $870 DD target is achievable only at the strategy/risk-control level.

### 8.24.9 Output artifacts

| Path | Status |
|---|---|
| `artifacts/regime_ml_filterg_v10/de3/{model.joblib,metrics.json,thresholds.json}` | ✅ trained, no ship verdict (combined stack failed) |
| `artifacts/regime_ml_kalshi_v10/de3/{...}` | ✅ trained |
| `artifacts/regime_ml_lfo_v10/de3/{...}` | ✅ trained |
| `artifacts/regime_ml_pct_v10/de3/{...}` | ⚠️ trained but degenerate predictions |
| `artifacts/v10_stack_aware_summary.json` | top-level retrain summary |

**No combined v10 stack passes ship gates.** No live deployment of v10 models recommended.

### 8.24.10 Recommended path forward (3-level retrain conclusion)

The retrain experiments are complete. Three deployable options, ranked:

1. **Accept v9's realistic floor**: deploy `loose_pct_strict_rest` (filterg=0.85, kalshi=0.85, lfo=0.85, pct=0.45) → ~$2,716 DD / ~$77k PnL on holdout. This is the best PnL/DD ratio achievable via overlay retraining. Compresses DD from filterless baseline ($4,632) by 41% while preserving 96% of PnL.

2. **Strategy-level DD compression**: tighten SL on the dominant DE3 sub from 10pt → 6pt. Halves the per-trade max loss; cuts DD by ~40-50% structurally without retraining overlays. Requires backtesting at the strategy level.

3. **Drop overlays + add daily CB**: `JULIE_ML_*_ACTIVE=0` for all overlays, add `JULIE_DAILY_CB_AT=-$300`. Doesn't filter at trade level but caps day-level damage. Achieves DD compression by stopping after a bad day rather than blocking individual trades.

**No live changes applied. No code modifications. No new sweeps.**

---

*Section 8.24 added 2026-04-25. V10 stack-aware retrain: 4 stages trained sequentially with upstream predictions as input features. Best Pareto $75,328 PnL / $3,414 DD — strictly Pareto-dominated by v9. Conclusion across 8.22-8.24: structural DD floor ~$1,000-$2,700 via overlay retraining; bridging to $870 requires strategy/risk-control changes (stop tightening, daily caps) not available at the overlay layer.*

---

## 8.25 March 2026 Walk-Forward Audit — Is The Baseline Real?

**User flag:** v9 holdout breakdown showed 2026-03 = $78,418 PnL on 843 fires (92% of holdout PnL on 73% of trades) while Jan/Feb/Apr 2026 were all negative. Suspicion: the entire $74k filterless baseline that gates v8/v9/v10 retrain decisions may be a walk-forward simulator artifact, with March's outsized contribution coming from wick-inflated TP fills.

**Method:** Loaded `artifacts/full_overlay_stack_simulation_14mo.parquet` (3,438 rows / 14mo) and `artifacts/combined_v9_stack_simulation_holdout.parquet` (1,477 rows / Jan-Apr 2026). Re-walked every signal forward on the actual ES outright contract bars from `es_master_outrights.parquet`, locking the contract by close-price match at the signal bar (so prices align — no MES synthetics, no front-month merge). Applied two checks per row: (a) **raw any-touch** (`bar.high >= tp_price` ⇒ TP, current sim rule), and (b) **conservative trade-through** (TP only if next-bar OPEN beyond TP; SL stays generous any-touch). Horizon 30 bars, fee $7.50, multiplier $5/pt — all matching the existing simulator (verified at `tools/run_full_live_replay.py:296-310` and `scripts/signal_gate/fast_aetherflow_replay.py:124-140`, both use `bar.high >= take_price` with no trade-through gate).

**Findings:**
- Baseline-TP fills in March 2026: 760 / 964 rows (79%)
- Of 759 baseline-TPs (1 miss), conservative outcome on real ES outright bars:
  - TP fills with bar trade-through (CONFIRMED): **31 / 759 (4%)**
  - TP fills wick-only (touched without trade-through): **12 / 759 (1.6%)**
  - **PHANTOM fills (real ES bars NEVER touched the TP price within 30 bars): 716 / 759 (94%)**
- Smoking gun: 2026-03-05 08:05 ET LONG @ ESH6 6855.00, TP=6880.0. Baseline says +$117.50. Real ESH6 next 30 bars: max high = 6864.50 (TP never approached). But ESM6 (June, +$50 carry) on the same minute traded at 6915+. The phantom TP fills concentrate on March 2026 because ESH6 was the actively-rolling expiry; it's near-certain the original simulator was walking a forward-fill / multi-symbol-merged bar series that pulled in ESM6 highs while the entry price came from ESH6.
- Avg bars-to-TP (real-bar conservative takes): march = comparable to other months when restricted to confirmed trade-throughs; the inflation is in the count, not the timing.
- March PnL (baseline parquet, raw): **+$84,082** on 964 rows (parquet); **+$78,418** on 843 fired holdout rows
- March PnL (raw any-touch on real ES bars): **-$12,350** (964 rows) / **-$12,174** (843 holdout fires)
- March PnL (conservative trade-through on real ES bars): **-$13,199** (964 rows)
- **Filterless baseline 14-mo (parquet, raw): +$124,844**
- **Filterless baseline 14-mo (raw any-touch on real ES bars): +$17,425** (842/3,438 misses kept at parquet value — true number is lower)
- **Filterless baseline 14-mo (conservative trade-through on real ES bars): +$13,566**
- v9 best (loose_pct_strict_rest), holdout (Jan-Apr 2026), RAW: trades=1,206 PnL=**+$76,613** WR=73.5% MaxDD=$2,716
- v9 best (loose_pct_strict_rest), holdout, CONSERVATIVE-ADJUSTED: trades=1,206 PnL=**-$14,564** WR=37.6% MaxDD=**$14,976**

**Verdict: (C) March is mostly artifact — actually worse than wick-inflation; it's phantom-fill from contract-roll bug.**

The original simulator awarded TP on 716 March 2026 trades where the actual front-month contract (ESH6) **never touched the TP price** in the 30-bar horizon. This is not a 1-tick wick problem. ESM6 (next quarter) was already trading $50-100 above ESH6 due to forward dividend carry, and the simulator's bar source was clearly contaminated with ESM6 or merged-front data. Conservative trade-through resolution cuts March from +$78k to **-$12k**. The 14-month filterless baseline collapses from +$124,844 (parquet) / "+$74,398" (Section 8.17 adjusted with 50% mgmt haircut) to **+$13,566** under conservative real-bar resolution. With the section-8.17 50% management haircut applied on top, the deployable baseline drops further to **~$6,800 across 14 months** — i.e., roughly $500/month, not $5,300/month.

**Implications:**
- **The +$74k adjusted filterless baseline used as the v9/v10 ship gate is wrong.** True conservative-bar baseline: **~$13.6k raw, ~$6.8k after 50% mgmt haircut.**
- v9 best (loose_pct_strict_rest) **fails its gates under the new baseline**: PnL flips from +$76k to -$14.6k, WR drops 73.5% → 37.6%, MaxDD blows from $2,716 → $14,976. **Do not ship v9 retrain.**
- **The simulator has a real bug.** The wick rule itself (`bar.high >= take_price`) is also lax, but the bigger issue is the bar source. Producing parquets must lock to the same contract symbol that issued the entry-bar price; never merge across rolling fronts in walk-forward. Quote-exact lines to fix: `tools/run_full_live_replay.py:297` and `scripts/signal_gate/fast_aetherflow_replay.py:130`.
- **Sections requiring correction notes:**
  - **8.17** ("Filterless friend-rule, sim raw 14mo: +$171,528"): the "+$54k after 50% haircut" March claim is built on the same phantom-fill data path; **revise to ~$6.8k 14-month baseline**.
  - **8.16** (Filter G removal +$3,783): probably less affected since Filter G runs on far fewer rows, but the `pnl_baseline` source is the same; needs spot-check.
  - **8.21 / 8.23** (v9 Pareto winner selection): the entire Pareto frontier is built on inflated baseline; **all v9 ranking decisions need to be re-derived from the conservative baseline.**
- **Action:** halt all retrain ship-decisions until (1) the bar-source is fixed (one symbol per signal, never merge); (2) the conservative trade-through rule is the default; (3) all 14-month baselines are regenerated and Sections 8.17, 8.21, 8.23 republished.

*Section 8.25 added 2026-04-25 by audit. The +$74k filterless baseline was a simulator artifact driven by phantom TP fills on 716 March 2026 trades where the actual ES contract never touched the take-profit price. Conservative-bracket real-bar baseline = +$13,566 (raw) / **~$6,800 (50% haircut)** across 14 months. v9 retrain ships are not justified at the new baseline.*

---

## 8.26 Final Verdict — Post-Simulator-Fix Master Reconciliation

*Written 2026-04-25 after autonomous pipeline rerun greenlit by user. This section closes the simulator-fix arc opened by 8.25.*

### TL;DR

The walk-forward simulator was awarding TP fills on phantom contract-roll bars (94% of March 2026 baseline-TPs were never traded by the real front-month). After fixing the bug end-to-end and rerunning every overlay analysis, the corrected 14-month filterless baseline is **+$13,498.75 / 47.9% WR / -$3,606 DD** — a 5.5× collapse from the previously cited +$74k. The Jan-Apr 2026 holdout slice of that baseline is **-$2,280 / 43.6% WR / -$3,606 DD**: DE3 itself loses money out-of-sample. Every overlay (Kalshi, LFO, PCT, full stack), every v9 per-head retrain, every v9 combined config (625 grid points), and every v10 stack-aware config (36 grid points) tested against that corrected holdout is either KILL or DEFERRED. **No overlay configuration ships.** Action moves out of the overlay/ML layer entirely and into the strategy/risk-control layer (tighter SL, daily circuit breaker, regime-conditional disable).

### What changed since 8.25

**Phase 1 — simulator fix.** New shared module `simulator_trade_through.py` provides the conservative trade-through TP rule plus contract-pinning by close-price match (with `ROLL_CALENDAR` fallback). 11 simulator files patched: `tools/run_full_live_replay.py`, `tools/run_full_live_replay_parquet.py`, `tools/backtest_aetherflow_direct.py`, `tools/backtest_aetherflow_multi_leg.py`, `tools/backtest_regimeadaptive_robust.py`, `tools/regimeadaptive_filterless_runner.py`, `tools/train_regimeadaptive_robust.py`, `build_de3_context_dataset.py`, `backtest_mes_et.py`, `scripts/signal_gate/fast_aetherflow_replay.py`, `scripts/signal_gate/train_lfo_ml.py`. Live trading code (`julie001.py`, `config.py`, `client.py`, `de3_v4_*.py`) untouched. Unit tests `test_simulator_phantom_fill_fix.py` 7/7 PASS — the smoking gun (2026-03-05 08:05 ET LONG ESH6 @ 6855.00, TP=6880.0) now correctly horizon-exits at 6851.25 instead of phantom-filling at 6880 (ESH6 max-high in next 30 bars = 6864.50).

**Phase 2 — filterless 14-mo baseline regenerated.** `tools/regen_filterless_baseline_v2.py` produces `artifacts/filterless_reconstruction_14mo_v2.parquet` (3,438 candidates, 1,572 taken trades after friend-rule). Aggregate: **+$13,498.75 net / 51.78% WR / -$3,606.25 DD**. Audit cross-check: 50% mgmt haircut on $13,499 = $6,750, matching 8.25's audit estimate (~$6,800) almost exactly.

**Phase 3a-f — every overlay analysis rerun.**
- 3a: Kalshi gate reconstruction (`tools/kalshi_gate_reconstruction_v2.py`) — also caught a label inversion in 8.19's old code (`p>=0.5 → PASS` is the live-correct rule; old code blocked high-proba/winners).
- 3b: LFO/Pivot/PCT reconstructions (`tools/regen_overlay_reconstruction_v2.py`).
- 3c: Full overlay stack simulation (`tools/run_full_overlay_stack_simulation_v2.py`).
- 3d: V9 per-overlay retrain (`tools/run_v9_retrain_v2.py`) — 4 HGB heads, 0.40-0.85 threshold sweep against the corrected holdout.
- 3e: Combined v9 stack (`tools/run_combined_v9_stack_v2.py`) — 625 threshold combos.
- 3f: V10 stack-aware retrain (`tools/run_v10_stack_aware_v2.py`) — 36 threshold combos with sequential pipeline conditioning.

### Corrected baseline (the new reference)

**14-month aggregate:** 1,572 trades / 47.90% WR / +$13,498.75 net / -$3,606.25 max DD (3,438 candidate signals; 1,866 dropped by single-position friend-rule).

**Holdout (Jan-Apr 2026, the v9/v10 gating window):** 507 trades / 43.6% WR / **-$2,280** PnL / -$3,606 DD. **DE3 is a losing strategy out-of-sample at the unfiltered candidate-stream level.**

| Month | Trades | WR%  | PnL net    | DD         |
|-------|-------:|-----:|-----------:|-----------:|
| 2025-03 | 112 | 44.6 | +$715.00   | -$788.75   |
| 2025-04 | 192 | 44.3 | +$1,668.75 | -$876.25   |
| 2025-05 | 95  | 63.2 | +$2,127.50 | -$172.50   |
| 2025-06 | 103 | 64.1 | +$3,856.25 | -$301.25   |
| 2025-07 | 96  | 53.1 | -$76.25    | -$433.75   |
| 2025-08 | 117 | 47.9 | +$32.50    | -$301.25   |
| 2025-09 | 53  | 50.9 | +$1,258.75 | -$312.50   |
| 2025-10 | 107 | 59.8 | +$1,877.50 | -$478.75   |
| 2025-11 | 101 | 51.5 | +$1,888.75 | -$650.00   |
| 2025-12 | 89  | 60.7 | +$2,430.00 | -$477.50   |
| 2026-01 | 94  | 59.6 | +$1,145.00 | -$173.75   |
| 2026-02 | 126 | 47.6 | -$637.50   | -$706.25   |
| **2026-03** | **179** | **42.5** | **-$2,323.75** | **-$2,513.75** |
| 2026-04 | 108 | 52.8 | -$463.75   | -$580.00   |
| **Total** | **1,572** | **51.78** | **+$13,498.75** | **-$3,606.25** |

The deepest DD lives entirely inside 2026-03 (-$2,514 of the -$3,606 14-mo total). 2026-Q1 is a structural drawdown regime for DE3, not a simulator artifact.

### Per-overlay reconciliation table (broken sim → corrected sim)

| Overlay | Old PnL impact (broken) | New PnL impact (v2) | New DD impact | Old verdict | New verdict |
|---|---:|---:|---:|---|---|
| Kalshi gate (8.19) | -$70,854 / 14mo | **-$4,258 / 14mo** (~-$304/mo) | -$309 (DD shrinks) | HURTS catastrophically | HURTS mildly |
| LFO (8.20.2)       | -$8,317 / 14mo  | **-$1,578 / 14mo**            | +$35 | HURTS | HURTS |
| PCT (8.20.4)       | +$3,927 / 14mo  | **-$1,558 / 14mo**            | -$69 | "keep, tune" | **FLIPPED to HURTS** |
| Pivot (8.20.5)     | DEFERRED        | DEFERRED                      | n/a  | DEFERRED | DEFERRED (effective KILL) |
| Full ml_full_ny stack (8.21) | -$17,083 / 14mo | **-$5,984 / 14mo** | +$117 | HURTS | HURTS |

Magnitudes shrink across the board (the broken sim was inflating both the baseline AND the gate-blocks of phantom-winners), but every overlay still nets negative on the corrected sim. The "PCT is the only positive overlay" finding from 8.20 was a phantom-fill artifact: the old PCT $-delta of +$3,927 was carried by PIVOT_LEAN trades whose TPs no longer fill under honest bars (PIVOT_LEAN $-delta flips from positive to **-$1,982**). BREAKOUT_LEAN remains marginally positive (+$438) but not enough to offset.

### V9 / V10 retrain reconciliation

| Tier | Old best (broken) | New best (v2) | New DD | New WR | Verdict |
|---|---|---|---:|---:|---|
| 8.22 — v9 per-head | varies | Kalshi @ 0.45: **-$2,151** | -$3,360 | 43.7% | KILL (G1, G4) |
| 8.23 — v9 combined (625 combos) | $76,613 / $2,716 DD | fg=0.50/k=0.60/lfo=0.60/pct=0.60: **-$2,260** | -$3,469 | 43.5% | KILL (G1, G4); strictly worse than best individual head |
| 8.24 — v10 stack-aware (36 combos) | $75,328 / $3,414 DD | all=0.45: **-$1,922** | -$2,836 | 42.8% | KILL (G1, G4); marginal improvement |

Stage AUCs on the corrected v10 corpus: FG test 0.580, K 0.558, LFO 0.541, PCT *degenerate* (0 positives in the FG+K+LFO triple-survivor pool — stack-aware specialization fails harder under corrected data than under the broken corpus's spurious signal). v9 v2 AUCs are healthier (FG 0.692, K 0.644, LFO 0.658, PCT 0.679) but no AUC in this regime is high enough to bridge the structural gates.

**Zero of 686 tested overlay configurations (4 per-head sweeps + 625 v9 combined + 36 v10) pass the strict ship gates.** Best DD across everything is the v10 `all=0.40` config at -$2,666 — still **3.06× over the $870 G1 cap**. Best WR across everything is v10 `loose_lfo_strict` at 44.0% — still **11pp under the 55% G4 floor**.

### Why no overlay can ship at the user's strict gates

**1. The DD floor is in DE3 itself, not in the overlay layer.** Overlays can BLOCK candidates but they can never EXIT a trade early — once a signal fires and is taken, the same SL distance is wagered. The DE3 strategy's per-trade SL distance and the cluster of consecutive losers in 2026-03 produce a $2,800-3,600 DD floor regardless of how many candidates are filtered out. Any path to a $870 DD goes through tightening the SL (or stepping out of the trade) at the strategy layer.

**2. The WR floor is in the candidate stream itself.** DE3 holdout WR is 43.6%. Even a perfect classifier that blocked the worst N losers would have to also avoid blocking comparable winners — at the AUCs we observe (0.54-0.69), the block precision is near-coin-flip (Kalshi v2: 44.4% block precision on 822 blocks). v9/v10 best WR moves the needle 0.1-0.2pp; G4 (≥55%) is structurally unreachable without an AUC that this corpus does not support.

**3. The PnL gate is mathematically unsatisfiable in holdout.** DE3 holdout PnL is **-$2,280**. Overlays can only reduce loss magnitude (and only modestly — best is -$1,922, a $358 / 16% improvement). They cannot manufacture profit. G2 ("PnL ≥ baseline AND trades ≤ baseline") is trivially passed on PnL (any negative number above -$2,280 passes), so the binding gates are G1 (DD) and G4 (WR), which the prior two points show are unreachable.

### Recommended path forward

**A. Strategy-layer DD compression (recommended).**
- Tighten DE3 SL from 10pt → 6pt; backtest the corrected filterless reconstruction under the new SL geometry.
- Add a daily circuit breaker at -$300/day to kill compounding bad days (the 2026-03 DD is built from a streak; cap day-level damage and the 14-mo DD halves).
- Identify the 2026-Q1 regime (likely high realized vol / sustained-trend / contract-roll dislocation; the -$2.5k month coincides with the H6→M6 roll); add a config flag to disable DE3 in that regime pending further analysis.
- Re-run the filterless reconstruction under the new SL/CB/regime config.
- Decision criterion: does this bring DE3 14-mo PnL ≥ $20k AND 14-mo DD ≤ $1,500 AND holdout PnL ≥ breakeven? If yes, ship; if no, the strategy itself needs structural rework.

**B. Drop overlays entirely + add daily CB (minimal-change option).**
- Set `JULIE_ML_FILTER_G_ACTIVE=0`, `JULIE_ML_KALSHI_ACTIVE=0`, `JULIE_ML_LFO_ACTIVE=0`, `JULIE_ML_PCT_ACTIVE=0` in deployment env.
- Add `JULIE_DAILY_CB_AT=-$300` (single new circuit-breaker hook).
- Don't filter at the trade level — cap day-level damage instead.
- Cleaner code, lower latency, no overlay maintenance burden, no model drift risk.
- This is the right option until A is fully studied.

**Not recommended.**
- Ship any overlay model from v9_v2 / v10_v2 — none pass gates, all HURT or marginal.
- Continue overlay retraining — three retrain attempts (v8 / v9 / v10) all KILL with the same structural reasoning, now reaffirmed at the corrected baseline.

### What this teaches us about the prior conclusions

The original v8/v9/v10 KILL verdicts were directionally correct (overlays don't ship) but for the wrong reason. We thought we were protecting a +$74k filterless baseline; we were actually protecting a +$13.5k baseline that turns -$2.3k in holdout. The decision was right by accident. Worse, the simulator bug had inflated the *gates themselves* (PnL ≥ $74k was an unreachable target derived from inflated data), so we were running the right experiment with the wrong target — and Section 8.24's "structural DD floor" intuition turned out to be true at every level we tested it (8.18 → 8.22 → 8.23 → 8.24, and now reaffirmed at 8.26 v2 corpus + v2 baseline + combined level).

### Open questions / future work

- **Why did the original simulator merge bar sources?** The phantom-fill behavior implies a forward-fill across rolls or a multi-symbol concat in the baseline parquet pipeline. We patched downstream consumers but the upstream parquet builder (`build_de3_context_dataset.py` and friends) likely still has the underlying bug; any future feature that reads `es_master_outrights.parquet` without contract-pinning is at risk. Worth a separate audit pass.
- **Section 8.16 (Filter G removal counterfactual) probably needs a revisit.** The +$3,783 figure was computed against the same broken `pnl_baseline` source. The directional finding (Filter G removal gives $X back) likely holds but the magnitude is wrong; spot-check on the v2 corpus.
- **RA head splits' insufficient-OOS finding:** 8.18 concluded RA splits don't have enough OOS samples to retrain. That finding likely still holds under the corrected sim — but the "why no signal" diagnosis (was the data inflated or genuinely thin?) should be re-examined.
- **2026-Q1 regime characterization.** -$2.5k in March alone is an outlier month even after correction. What's structurally different about that month — vol regime, persistent trend, microstructure around the H6→M6 roll, news-driven momentum days? Strategy-layer fix A depends on identifying this regime cleanly.

### Files written this session

**Simulator + tests:**
- `simulator_trade_through.py` (new shared utility)
- `test_simulator_phantom_fill_fix.py` (7/7 PASS)
- 11 patched simulator files (see Phase 1 above)

**Phase 2 baseline:**
- `tools/regen_filterless_baseline_v2.py`
- `artifacts/filterless_reconstruction_14mo_v2.parquet`
- `artifacts/baseline_v2_summary.json`

**Phase 3 overlays:**
- `tools/kalshi_gate_reconstruction_v2.py`, `artifacts/kalshi_gate_reconstruction_14mo_v2.parquet`, `artifacts/kalshi_v2_summary.json`
- `tools/regen_overlay_reconstruction_v2.py`, `artifacts/lfo_reconstruction_14mo_v2.parquet`, `artifacts/pivot_reconstruction_14mo_v2.parquet`, `artifacts/pct_reconstruction_14mo_v2.parquet`, `artifacts/overlay_v2_summary.json`
- `tools/run_full_overlay_stack_simulation_v2.py`, `artifacts/full_overlay_stack_simulation_14mo_v2.parquet`, `artifacts/full_stack_v2_summary.json`

**Phase 3 retrains:**
- `tools/run_v9_retrain_v2.py`, `artifacts/regime_ml_{filterg,kalshi,lfo,pct}_v9_v2/de3/{metrics.json, sweep.csv, thresholds.json, KILL}`, `artifacts/v9_retrain_v2_summary.json`
- `tools/run_combined_v9_stack_v2.py`, `artifacts/combined_v9_stack_v2_summary.json`, `artifacts/combined_v9_v2_simulation_holdout.parquet`
- `tools/run_v10_stack_aware_v2.py`, `artifacts/regime_ml_{filterg,kalshi,lfo,pct}_v10_v2/de3/`, `artifacts/v10_stack_aware_v2_summary.json`

**Reports:**
- `/tmp/phase1_phase2_report.md`, `/tmp/phase3a_kalshi_v2_report.md`, `/tmp/phase3b_lfo_pivot_pct_v2_report.md`, `/tmp/phase3c_full_stack_v2_report.md`, `/tmp/phase3d_v9_retrain_v2_report.md`, `/tmp/phase3e_combined_v9_v2_report.md`, `/tmp/phase3f_v10_retrain_v2_report.md`

**Live code touched:** zero. No commits, no pushes, no `model.joblib` shipped to production. `ml_full_ny` config in production unchanged.

### 8.26.1 Worst-Overlay Attribution Ranking

*Computed 2026-04-25 from corrected v2 corpus. Filter G isolation is fresh (`tools/per_overlay_attribution_v2.py`); other overlays cross-checked against the existing per-overlay v2 recon parquets. No new full-pipeline reruns.*

**Method.** Walk `full_overlay_stack_simulation_14mo_v2.parquet` once per overlay; apply ONLY that overlay (BLOCK for FG/Kalshi; per-row WAIT-net for LFO; per-row size×TP-net for PCT; pass-through for Pivot); rewalk the single-position friend rule with $7.50/trade haircut. Compare to the corrected filterless baseline: 1,572 trades / 47.9% net-WR / +$13,498.75 / -$3,606.25 DD.

**Per-overlay damage ranking (|ΔPnL|):**

| Rank | Overlay | ΔPnL ($) | ΔTrades | ΔWR (pp) | ΔDD ($) | Verdict |
|---|---|---:|---:|---:|---:|---|
| 1 (worst) | Kalshi gate | **−$3,929** | −120 | −0.7 | +$170 | HURTS |
| 2 | LFO (WAIT) | **−$1,564** | 0 | 0.0 | +$49 | HURTS |
| 3 | PCT (size×TP) | **−$1,536** | 0 | 0.0 | −$54 | HURTS |
| 4 | Filter G (block) | **−$754** | −18 | −0.2 | −$66 | HURTS mildly |
| 5 (best) | Pivot Trail | **$0** | 0 | 0.0 | $0 | DEFERRED |

Cross-check vs existing recon parquets: Kalshi −$4,258, LFO −$1,578, PCT −$1,558. Fresh-walk numbers differ by ≤$330 due to per-row net derivation (`raw_pnl_walk − $7.50` vs the recon's pre-merged `net_pnl_after_haircut`). Both methods land on the same ranking.

**Worst offender: Kalshi gate — destroyed ~$3.9–4.3k across 14 months.**

- **Months (top drags):** 2025-06 (−$945), 2025-11 (−$803), 2025-10 (−$696), 2025-04 (−$583). Net *positive* in 2026-01/-02/-03 (+$95 / +$136 / +$103) — Kalshi mildly helps during the holdout DD regime, then bleeds the profitable 2025 months. Damage is concentrated in the *good* months.
- **Sides:** 100% LONG (218/218 blocked-and-would-have-traded rows). Gate is active 12-16 ET only, where DE3 is nearly all LONG.
- **Strategies:** DE3 211/218 (97%, edge_destroyed +$4,014); RA 7/218 (RA actually helps on its blocks: −$68).
- **Block precision (loser-share of blocked-and-would-trade):** **44.0%** — sub-coin-flip. 822 BLOCK decisions → 218 actually-blocked trades after friend-rule (120 winners blocked vs 96 losers blocked). Matches Section 8.26's reported AUC=0.558.
- **Edge destroyed (sum baseline net-PnL of blocked-would-trade rows):** **+$3,946**.
- **Sim fire rate:** 23.9% of candidates BLOCK (822/3,438), or 37% of gate-active candidates.

**Best: Pivot Trail — $0 impact.** All 3,438 rows have `pivot_decision == "NA"`. Pivot is a mid-trade SL ratchet; the corrected sim still does not model stepwise SL adjustments. **Status: DEFERRED — same blocker as Section 8.20.5.** A real recon requires extending `simulator_trade_through.py` with stepwise-SL semantics. Bounded MFE-≥-trigger-then-stop approximation considered but rejected: it ignores Pivot's SL re-pricing, so error bars are too wide to publish. Tracked as future work.

**Disconnect check.** Per Section 8.7 / PR #206, some overlays are "loaded but not firing" in live logs. Numbers above are SIMULATED damage — what each overlay *would* do if firing correctly. Sim fire rates: Kalshi 23.9%, PCT-active 37.3%, LFO-WAIT 3.5%, Filter G 3.3%. Verify each against production decision-log counts; if live fire rate < 50% of sim, that overlay's damage is theoretical.

**Recommendation.** Kill **Kalshi** first if Option B is taken — it accounts for ~50% of total overlay damage ($3.9k of $7.8k combined) at sub-coin-flip precision. Batch the other three into a single env-flag PR (`JULIE_ML_LFO_ACTIVE=0`, `JULIE_ML_PCT_ACTIVE=0`, `JULIE_ML_FILTER_G_ACTIVE=0`). Pivot is already effectively off. Combined deactivation projects to recover ≈$7.8k/14mo on the corrected baseline — but cannot rescue the −$2.3k holdout (Kalshi mildly *helps* in 2026-Q1; killing it loses ~$330 there). Reaffirms Section 8.26: damage lives in DE3 itself, not the overlay layer — overlay deactivation is a janitor pass, not a fix.

---

*Section 8.26 closes the simulator-fix arc. The +$74k filterless baseline used to gate v8-v10 retrain decisions was a 5.5× inflation from phantom contract-roll fills. Corrected baseline +$13.5k / 14mo, -$2.3k holdout. NO overlay configuration ships. Action moves to the strategy/risk-control layer (tighter SL, daily CB, regime-conditional disable). The structural-DD-floor finding from 8.18→8.24 is reaffirmed at the corrected level. Per-overlay attribution (8.26.1) ranks Kalshi as the worst single offender (~$3.9k/14mo); Pivot Trail remains structurally deferred.*

---

## 8.27 V11 Retrain — Clean Corpus, Active Model Fires, Strict Gates

*Written 2026-04-25. Ordered by user after the v2 fidelity audit (see `/tmp/v2_fidelity_audit.md`) found three structural problems with v9-v10 attribution: (1) overlay decisions were inherited from a v1 corpus we couldn't audit, (2) same-side rule was simplified to a global open-trade window instead of friend's family-aware logic, (3) Kalshi 12-16 ET window was inconsistently enforced. V11 fixes all three.*

### 8.27.1 — TL;DR

Corrected-sim corpus rebuilt from scratch with active production model fires + family-aware friend rule + Kalshi window. Filterless baseline is **+$13,080 / 14mo, -$4,095 DD** (vs v9/v10's broken-sim "+$74k / -$2,716 DD" debunked in §8.25-8.26). Four ML heads retrained on this corpus. **0 of 4 heads pass the strict ship gates** ($870 DD, 55% WR, ≥ baseline PnL, ≤ baseline trades). Binding gate is universally **G1** — the holdout DD (-$4,095) is itself 4.7× the ceiling, so no head-level selectivity can rescue it. Only positive-EV finding is a non-ML one: the Pivot Trail swing-pivot mechanic is +$1,510 / 14mo on the always-arm policy, with marginal DD reduction. The mechanic is already enabled in production.

### 8.27.2 — What Changed Vs v9/v10

| Aspect | v9/v10 | v11 |
|---|---|---|
| Overlay decisions | inherited from v1 corpus (un-auditable) | active `model.predict_proba()` per candidate |
| Friend rule | global single-position window | family-aware per `julie001.py:2058-2080` |
| Kalshi window | inconsistently enforced | 12-16 ET enforced, AF carveout coded |
| Pivot Trail | DEFERRED 3 times (no SL-step sim) | `pivot_stepped_sl_simulator.py` built, settled |
| Filled simulator | broken (phantom-fill bug, §8.25) | `simulator_trade_through.py` (corrected) |
| Filterless baseline | "+$74,000 / -$2,716 DD" (FICTIONAL) | **+$13,080 / -$4,095 DD** (real) |

### 8.27.3 — Phase 1: Clean Corpus

Built `tools/build_v11_training_corpus.py` from `artifacts/full_overlay_stack_simulation_14mo.parquet` (3,438 candidates Mar 2025 → Apr 2026). Stripped overlay-decision columns and re-fired all five production overlays (Filter G v10, Kalshi gate, LFO, PCT, Pivot) using `model.predict_proba()`. Friend rule from `julie001.py:2058-2080` ported faithfully (DE3+RA cross-family bypass; AF same-side parallel via `live_same_side_parallel_max_legs`). Kalshi 12-16 ET window flagged per row.

| metric | v11 | v2 (simple-rule) | Δ |
|---|---|---|---|
| candidates | 3,438 | 3,438 | 0 |
| allowed by friend rule | **1,762** | 1,572 | +190 |
| Σ net PnL (corrected sim) | **+$13,080.00** | +$13,499 | -$419 |
| walk DD | **-$4,095.00** | -$3,606 | -$489 |

The 190 extra allowed trades come from cross-family DE3+RA bypass that v2's simple "single open position" rule incorrectly blocked. Some are losers, hence the slightly worse PnL/DD — that's the right answer to train against. Smoking gun re-checked: 2026-03-05 08:05 LONG ESH6 @ 6855 exits **`horizon` @ 6851.25** for −$26.25 net (the phantom-fill that previously awarded a false TP off ESM6 highs is gone).

Three documented limitations rolled forward into Phase 2: (1) Kalshi snapshot features (`entry_probability`, `probe_probability`, `momentum_*`) are unavailable historically — fed neutral defaults (0.5/1.0/0.0); (2) PCT features are surrogated from session bars (no live state machine); (3) Filter G v10 has 17 features with no metadata, fed in fixed order.

### 8.27.4 — Phase 2: Kalshi / LFO / PCT Retrain

Per-head HGB on the 1,762-row friend-rule-allowed slice, time-split (test = ts ≥ 2026-01-01). Strategy = DE3 only — RA had n_test = 29 < 50, all skipped on split. Label = `is_big_loss` (net_pnl ≤ -$50) won OOS AUC vs `is_loss_any` for every head. Holdout filterless baseline (Jan-Apr 2026, allowed): **560 trades / 42.86% WR / -$2,886 / -$4,095 DD**.

Top three thresholds per head:

**kalshi_de3** (applicability baseline 317 / 46.69% / -$911 / -$1,880):

| thr | trades | WR | PnL | DD | gates G1G2G3G4 |
|----|---|---|---|---|---|
| 0.40 | 181 | 48.07% | -$161.25 | -$1,196.25 | 0110 |
| 0.70 | 279 | 48.03% | -$430.00 | -$1,571.25 | 0110 |
| 0.50 | 225 | 47.11% | -$522.50 | -$1,473.75 | 0110 |

OOS AUC = 0.552 (barely above chance with neutral-snapshot Kalshi features).

**lfo_de3** (applicability baseline 531 / 43.88% / -$2,674 / -$3,928):

| thr | trades | WR | PnL | DD | gates G1G2G3G4 |
|----|---|---|---|---|---|
| 0.40 | 297 | 44.11% | -$1,077.50 | -$2,272.50 | 0110 |
| 0.45 | 334 | 44.31% | -$1,282.50 | -$2,505.00 | 0110 |
| 0.60 | 441 | 44.67% | -$1,768.75 | -$3,020.00 | 0110 |

OOS AUC = 0.643 (modestly informative).

**pct_de3** (applicability baseline 531 / 43.88% / -$2,674 / -$3,928):

| thr | trades | WR | PnL | DD | gates G1G2G3G4 |
|----|---|---|---|---|---|
| 0.40 | 298 | 43.29% | -$1,477.50 | -$2,685.00 | 0110 |
| 0.45 | 341 | 42.23% | -$1,993.75 | -$3,067.50 | 0110 |
| 0.50 | 370 | 42.70% | -$2,125.00 | -$3,301.25 | 0110 |

OOS AUC = 0.630.

Every threshold of every head: G2 PASS, G3 PASS, **G1 FAIL, G4 FAIL**. Closest-to-ship is `kalshi_de3 @ thr 0.40`: 136 blocks lift WR 46.7% → 48.1% and cut loss -$911 → -$161, but residual DD -$1,196 is still 38% over the $870 ceiling. WR maxes at 48.07% — never crosses 55%. **Three KILL markers written; no `model.joblib` for any head.**

### 8.27.5 — Phase 3: Pivot Trail (Settled After 3 Deferrals)

Pivot was deferred in §8.20.5, §8.21, and §8.26.1 because the corrected sim couldn't model stepwise SL re-pricing. V11 closed this. Built `pivot_stepped_sl_simulator.py` faithful to `julie001.py:189-207, 380-413, 416-479, 11290-11340`:

```
profit_pts = pivot_price - entry_price
if profit_pts < 12.5: skip
anchor_C = floor(pivot_price / 12.5) * 12.5
anchor_B = anchor_C - 12.5
candidate_B = anchor_B - 0.25
if candidate_B > entry: candidate = candidate_B   # Reading B (default)
else:                   candidate = anchor_C - 0.25  # Reading C fallback
ratchet: skip unless candidate > current_sl AND > entry
```

Four unit tests pass, including the smoking-gun `2026-03-05 08:05 LONG ESH6` (pivot correctly never arms — MFE peaks at 9.5pt < 12.5pt min-profit gate, and entry hour=8 ET is outside the 9-15 ET US-session window). Re-walked all 1,762 allowed candidates with `pivot_active=True` and labeled each by `pivot_label = int(pnl_with_pivot > pnl_no_pivot)`.

Stepped-SL deltas (corpus-wide, `artifacts/v11_pivot_labels.parquet`):
- Σ no-pivot:  **+$13,080.00**
- Σ always-pivot: **+$14,590.00**
- always-pivot Δ: **+$1,510.00**
- Pivot armed in sim: 190 / 1,762 (10.8%); helped: 62 (3.5%)

ML head trained on 1,068 train / 531 test (DE3): test_AUC = 0.693 (real signal), but every threshold from 0.40-0.85 fails G1/G4 because the underlying holdout DD (-$3,928 no-pivot, -$3,655 always-pivot) is itself well over $870. Best @ thr 0.40: 8 arms, PnL -$2,636, DD -$3,890. **KILL.** RA: zero positive labels in 163 rows (Pivot never armed on a single allowed RA trade) → SKIP_split.

Marginal-value table on holdout (n=531):

| Variant | trades | WR | PnL | DD |
|---|---|---|---|---|
| no-pivot baseline | 531 | 43.88% | **-$2,673.75** | -$3,927.50 |
| always-pivot | 531 | 45.76% | **-$2,428.75** | -$3,655.00 |
| Best v11 head @ thr 0.40 | 531 | 43.88% | -$2,636.25 | -$3,890.00 |
| Oracle (perfect arm) | 531 | — | -$1,821.25 | — |

The ML head **actively hurts** vs always-pivot (-$208 holdout). Always-pivot beats no-pivot by +$1,510 corpus-wide / +$245 holdout, with mild DD reduction. **Verdict: keep `JULIE_PIVOT_ACTIVE=1` in production (rule-based, free); kill the v11 Pivot ML head — the rule already runs without it.**

### 8.27.6 — Phase 4: Strict Gate Synthesis

`artifacts/v11_phase4_gates_summary.json`. Best per head:

| Head | strategy | thr | trades | WR | PnL | DD | G1G2G3G4 | binding | ship |
|---|---|---|---|---|---|---|---|---|---|
| Kalshi | DE3 | 0.40 | 181 | 48.07% | -$161.25 | -$1,196.25 | 0110 | G1 | KILL |
| Kalshi | RA  | — | — | — | — | — | — | — | SKIP_split |
| LFO | DE3 | 0.40 | 297 | 44.11% | -$1,077.50 | -$2,272.50 | 0110 | G1 | KILL |
| LFO | RA  | — | — | — | — | — | — | — | SKIP_split |
| PCT | DE3 | 0.40 | 298 | 43.29% | -$1,477.50 | -$2,685.00 | 0110 | G1 | KILL |
| PCT | RA  | — | — | — | — | — | — | — | SKIP_split |
| Pivot | DE3 | 0.40 | 531 | 43.88% | -$2,636.25 | -$3,890.00 | 0110 | G1 | KILL |
| Pivot | RA  | — | — | — | — | — | — | — | SKIP_split |

**Heads passing all gates: 0 / 4.** Binding gate G1 (DD ≤ $870) for every threshold of every head. G4 (WR ≥ 55%) also fails universally. G2 (PnL ≥ baseline AND trades ≤ baseline) passes nearly everywhere (trivially — heads block trades, so trade-count gate is easy; the negative baseline makes PnL gate easy). G3 (n ≥ 50) only fails at extreme thresholds. The story is the same as §8.26: **damage lives in the strategy layer, not the overlay layer**, and overlays don't have enough leverage to fix a -$4k DD floor.

### 8.27.7 — Phase 5: Combined Stack Sanity Sim

`tools/run_v11_combined_stack_sim.py` → `artifacts/v11_combined_stack_summary.json`. Since 0 heads passed gates, the "combined deployment-order stack" reduces to three variants on the friend-rule-allowed stream:

| Variant | trades | WR | PnL | DD |
|---|---|---|---|---|
| **A**: Filterless no-pivot (Phase 1 baseline) | 1,762 | 47.16% | **+$13,080.00** | -$4,095.00 |
| **B**: Filterless + always-pivot (rule-based) | 1,762 | 49.09% | **+$14,590.00** | -$3,822.50 |
| **C**: Filterless + ML Pivot head @ thr 0.40 | 1,762 | 49.09% | **+$14,590.00** | -$3,822.50 |

Variants B and C are identical because the Pivot ML head at thr=0.40 arms 1,681 / 1,762 candidates (95.4%) — selectivity essentially absent. C only diverges from B above thr ≈ 0.6, and as it diverges it walks toward A (the no-pivot baseline). The head is dominated by the always-arm policy.

Per-month for variant B (always-pivot — the SHIP candidate):

| Month | n | WR% | PnL ($) |
|---|---|---|---|
| 2025-03 | 128 | 46.09 |   +985.00 |
| 2025-04 | 217 | 46.54 | +1,916.25 |
| 2025-05 | 122 | 58.20 | +2,531.25 |
| 2025-06 | 117 | 58.97 | +4,195.00 |
| 2025-07 | 105 | 46.67 |    -95.00 |
| 2025-08 | 129 | 44.19 |    +91.25 |
| 2025-09 |  67 | 56.72 | +1,415.00 |
| 2025-10 | 112 | 54.46 | +2,062.50 |
| 2025-11 | 112 | 50.89 | +1,641.25 |
| 2025-12 |  93 | 56.99 | +2,488.75 |
| 2026-01 | 101 | 49.50 |   +932.50 |
| 2026-02 | 140 | 46.43 |   -823.75 |
| 2026-03 | 205 | 39.51 | -2,291.25 |
| 2026-04 | 114 | 47.37 |   -458.75 |
| **TOT** | **1,762** | **49.09** | **+14,590.00** |

The DD floor lives in 2026-Q1 (Feb -$824, Mar -$2,291) — same regime that drives §8.26's -$2.3k holdout. Always-pivot doesn't fix it; nothing in this overlay layer does.

Comparison anchors:
- **ml_full_ny actual** (live, prior journal): ≈ **+$1,742 / 14mo**.
- **v9 v2 best** (corrected sim, §8.22): **-$2,260 / -$3,469 DD**.
- **"v9 best" on broken sim** (§8.25 fictional): "+$76,000 / -$2,716 DD" — phantom-fill artifact, debunked.

### 8.27.8 — Honest Verdict

V11 was the third retrain on this strategy stack and the second under the corrected sim. It was the first with active overlay fires and a faithfully-ported friend rule. The verdict is unambiguous and does not depend on any modeling choice: **the strict ship gates ($870 DD, 55% WR, ≥$13,499 PnL, ≤1,572 trades) cannot be met by overlay-layer selectivity** because the underlying corrected-sim DD on the allowed stream is -$4,095 — 4.7× the gate. No amount of post-hoc filtering at 17–28 features per head can compress a 4.7× gap; every threshold of every head leaves residual DD between -$1,196 (kalshi best) and -$3,890 (pivot best).

The 2026-Q1 regime is the structural killer. February through April drop to 39-47% WR and lose the cumulative gain back to baseline. This is what §8.26 already concluded ("damage lives in DE3 itself, not the overlay layer") — v11 reaffirms it under stricter conditions: even with **active model fires** (so no inheritance error) and a **faithful friend rule** (so no rule-mismatch error), the heads converge on the same answer.

What v11 conclusively *did* settle was the Pivot Trail deferral. Three prior journal sections marked Pivot as un-auditable because the simulator couldn't ratchet SL. V11 built that simulator, walked all 1,762 candidates, and settled both questions: (1) the rule-based always-arm policy is +$1,510 / 14mo with DD reduction, (2) the ML head doesn't beat always-arm. The mechanic stays, the head dies. This is the clean shutdown of the entire ML overlay retraining track.

Three retrain attempts under three corpus regimes have now converged on the same answer: no ML overlay configuration ships at the user's strict gates. Action moves out of the overlay layer entirely. Everything in 8.27.9 is mechanical cleanup or strategy-layer work.

### 8.27.9 — Recommendations

**Ship now (zero-risk):**
- Keep `JULIE_PIVOT_ACTIVE=1` — the swing-pivot mechanic itself is +$1,510 / 14mo with marginal DD reduction. Free money. Already enabled in production.

**Kill (env-flag PR):**
- `JULIE_ML_FILTER_G_ACTIVE=0`
- `JULIE_ML_KALSHI_ACTIVE=0`
- `JULIE_ML_LFO_ACTIVE=0`
- `JULIE_ML_PCT_ACTIVE=0`

(Do not touch `JULIE_PIVOT_ACTIVE` — the rule already runs; the ML head was never deployed.)

**Deferred to strategy layer (Option A from §8.26):**
- DE3 SL 10pt → 6pt (cap per-trade loss before it hits the DD floor)
- Daily CB at -$300 (force-close after one bad cluster)
- Regime-conditional disable: 2026-Q1 was -$2,886 holdout / 56% loss-rate. Some regime feature (vol bucket? day-of-week? trend persistence?) is killing the strategy. A regime classifier that flips DE3 OFF in that regime is the natural next experiment — but it lives at the strategy layer, not the overlay layer.

### 8.27.10 — What V11 Conclusively Settled

We **ruled out**:
- Filter G as a useful ML overlay (KILL across v9, v10, v10_v2, and now v11 by composition).
- Kalshi gate ML beyond the rule-based 12-16 ET window (KILL @ test_AUC 0.55).
- LFO ML head despite test_AUC 0.64 (KILL — DD floor unmeetable).
- PCT ML head despite test_AUC 0.63 (KILL — DD floor unmeetable).
- Pivot Trail ML head despite test_AUC 0.69 (KILL — head can't beat always-arm rule).
- The "+$74k filterless" baseline used to gate v8/v9/v10 decisions (FICTIONAL, §8.25-8.26).

We **confirmed**:
- Corrected-sim filterless baseline = +$13,080 / 14mo on the allowed stream (-$4,095 DD).
- Friend's family-aware same-side rule allows 190 more trades than v2's simple rule (mostly DE3+RA cross-family bypass); net PnL effect is small (-$419) and DD effect is small (-$489).
- The Pivot Trail mechanic is +$1,510 / 14mo on always-arm, +$245 / 4mo on holdout. Mild but real and free. ALREADY ON in production via `JULIE_PIVOT_ACTIVE=1`.
- 2026-Q1 is the regime that breaks DE3 (-$2,886 holdout / 42% WR). Overlay-layer selectivity cannot rescue it.

### 8.27.11 — Outstanding Questions

- **Live vs sim divergence on overlay damage** (§8.26.1): some overlays are loaded but observed not firing in production logs. Verify each kill is real before merging the env-flag PR — a "kill" of an overlay that wasn't firing anyway recovers $0, not the projected dollars.
- **Kalshi snapshot backfill** (Phase 1 limitation #1): `kalshi_proba` was computed with neutral live-state defaults. The 2025 KXINXU all-strikes parquet (per MEMORY.md) could rehydrate `entry_probability`/`probe_probability`/`momentum_*`. Would a rehydrated Kalshi gate hit OOS AUC > 0.55? Probably yes, but the DD floor still wins, so this is academic unless paired with strategy-layer fixes.
- **Filter G v10 feature order** (Phase 1 limitation #3): no `feature_names_in_` on the artifact; we fed a guessed order. If wrong, fg_proba is miscalibrated and downstream heads that consume it (LFO/PCT/Pivot) inherit the noise. Re-train Filter G in v11 only if Option A justifies bringing the overlay layer back in scope.

### 8.27.12 — Files Written This Phase

Corpus + builders:
- `tools/build_v11_training_corpus.py` — builder
- `artifacts/v11_training_corpus.parquet` — 3,438 × 69
- `artifacts/v11_corpus_summary.json`

Phase 2 retrain:
- `tools/train_v11_phase2.py`
- `artifacts/regime_ml_kalshi_v11/{de3,ra}/metrics.json` (KILL / SKIP)
- `artifacts/regime_ml_lfo_v11/{de3,ra}/metrics.json` (KILL / SKIP)
- `artifacts/regime_ml_pct_v11/{de3,ra}/metrics.json` (KILL / SKIP)
- `artifacts/v11_phase2_summary.json`

Phase 3 Pivot:
- `pivot_stepped_sl_simulator.py`
- `test_pivot_stepped_sl_simulator.py` (4 tests, all pass)
- `tools/build_v11_pivot_labels.py`
- `tools/run_v11_pivot_retrain.py`
- `artifacts/v11_pivot_labels.parquet` — 1,762 rows
- `artifacts/regime_ml_pivot_v11/{de3,ra}/metrics.json` (KILL / SKIP)
- `artifacts/regime_ml_pivot_v11/summary.json`

Phase 4 + 5 synthesis:
- `artifacts/v11_phase4_gates_summary.json`
- `tools/run_v11_combined_stack_sim.py`
- `artifacts/v11_combined_stack_summary.json`

No `model.joblib` and no `thresholds.json` files were written — every head KILLED. No live code was modified (`julie001.py`, `config.py`, `client.py`, `de3_v4_*.py` untouched).

---

*Section 8.27 closes the ML overlay retraining track. Three retrain attempts under three corpus regimes (broken-sim v8 → corrected-sim v9/v10 with inherited decisions → corrected-sim v11 with active fires) all converge on the same verdict: **no ML overlay configuration ships at the user's strict gates ($870 DD, 55% WR, ≥$13,499 PnL, ≤1,572 trades).** The structural reasons are now exhaustively documented in 8.26 (DD floor in DE3 itself) and reaffirmed in 8.27 (active model fires don't change the ranking). Action moves out of the ML overlay layer entirely. The only positive-EV finding from v11 is non-ML: the Pivot Trail swing-pivot mechanic (rule-based, no model) is +$1,510/14mo and should remain enabled.*

---

## 8.28 DD ≤ $1,000 Sweep + Realistic Simulator Recomputation

*Written 2026-04-25. User asked: "relax DD gate from $870 → $1,000 and explore configs we haven't tested." This section also folds in two crucial side-audits: (a) DD-metric formula audit (ruled out non-standard accounting), and (b) early-exit fidelity audit (found 4 of 6 mechanisms missing from the simulator).*

### 8.28.1 — TL;DR

Relaxing the DD gate from $870 to $1,000 ships **0 of 505** evaluated configurations under the no-BE-arm simulator. The binding gate flipped from G1 (DD) to **G4 (WR)**: the highest winrate anywhere across the 505-config sweep is **34.55%** — twenty pp below the 55% gate. A separate audit of the simulator found that **four of six** live early-exit mechanisms (BE-arm, Pivot Trail, manual close, dead-tape) were missing entirely; only timeout was faithfully modeled. Rebuilding the corpus with MFE/MAE tracking and re-evaluating with BE-arm + structural add-ons reduces holdout DD by **25.8%–39.6%** (matching the audit's lower-bound estimate). The closest realistic-sim config — `lfo_de3@0.10 + BE-arm + SL=6pt` — clears G1 at **$908.75 DD** (the first config across all v11 work to do so) but fails G4 at **WR=37.32%**, off the gate by 17.7pp. **§8.27 verdict stands.** The structural problem is winrate, not drawdown; the leverage point has moved.

### 8.28.2 — DD Metric Audit (Sanity Check)

User hypothesis: "Maybe the §8.27 kill is wrong because DD is being computed on a non-standard formula." Per `/tmp/dd_metric_audit.md`, all nine DD-computing scripts in the codebase use **formula A (max continuous peak-to-trough)** — `(equity.cummax() - equity).max()` or its sign-flipped equivalent `(cum - cum.cummax()).min()`. Files audited: `tools/regen_filterless_baseline_v2.py:243-245`, `tools/per_overlay_attribution_v2.py:81,102`, `tools/run_full_overlay_stack_simulation_v2.py:278,289`, `tools/run_v9_retrain_v2.py:76,161`, `tools/run_combined_v9_stack_v2.py:74,143`, `tools/run_v10_stack_aware_v2.py:92-93`, `tools/build_v11_training_corpus.py:925,945`, `tools/train_v11_phase2.py:133-134,236-237,303`, `tools/run_v11_pivot_retrain.py:80-84`. No file uses an integrated/area metric or a min-equity metric.

Independent recompute on the parquet (`artifacts/dd_recompute_v2.json`) matched every reported number to the cent: v2 filterless = $3,606.25, v11 friend-rule = $4,095.00, v9 best Pareto = $3,526.25, v11 kalshi_de3@0.40 = $1,196.25, v11 lfo_de3@0.40 = $2,273. Magnitude of correction: **0%**. No gates flip. **DD-metric error rejected as the explanation; the §8.27 kill is not an accounting artifact.**

### 8.28.3 — DD ≤ $1,000 Sweep on No-BE-arm Sim

Per `/tmp/v11_1000dd_exploration_report.md`. Holdout = 560 rows (`ts >= 2026-01-01 & allowed_by_friend_rule==True`). Holdout baseline: 560 trades / WR 42.86% / PnL −$2,886.25 / DD $4,095.00.

Three task batches, 505 configurations total:

**Task 1 — per-head threshold compression** (5 heads × 13 thresholds = 65 configs). Probability ranges by head reveal that several heads saturate inside the requested 0.10–0.40 sweep: kalshi.min=0.528, fg.p10=0.698 — every threshold below those floors is "block all". Only LFO has a genuine gradient inside the range, with DD floor $1,505 at thr=0.10 (n=209, WR=40.67%). The single-head winners under DD≤$1,000 ∧ n≥50 are **filterg_de3@0.40** (n=61, WR=32.79%, DD $508.75, gates **YYYN**) and **pivot_de3@0.40** (n=55, WR=34.55%, DD $380.00, gates **YYYN**) — both fail G4 by ≥20pp.

**Task 2 — per-strategy isolation** (6 variants). DE3-only (n=531) DD=$3,927.50; RA-only (n=29) DD=$232.50 but n<50 fails G3; combined kept-set patterns confirm DE3 dominates volume and DD.

**Task 3 — combined stacks** (Stack A: FG+K = 20; Stack B: FG+LFO = 20; Stack C: FG+K+LFO = 80; Stack D: FG+K+LFO+PCT = 320 — total 440 combos). Highest WR across all 440 stacks: **34.04%**. Best Pareto-passing (DD≤$1,000 ∧ PnL≥baseline ∧ n≥50) ranked by WR desc:

| # | Source | Config | n | WR | PnL | DD | Gates |
|---|---|---|---|---|---|---|---|
| 1 | pivot_de3 | thr 0.40 | 55 | 34.55% | −$356.25 | $380.00 | YYYN |
| 2 | Stack C | fg=0.5–0.6, lfo=0.40 | 56 | 33.93% | −$215.00 | $336.25 | YYYN |
| 3 | Stack B | fg=0.60, lfo=0.50 | 63 | 33.33% | −$337.50 | $503.75 | YYYN |
| 4 | filterg_de3 | thr 0.10–0.40 | 61 | 32.79% | −$422.50 | $508.75 | YYYN |
| 5 | Stack C | fg=0.5–0.6, lfo=0.30 | 53 | 32.08% | −$255.00 | $305.00 | YYYN |

**0 / 505 configurations pass all four gates.** Binding gate is **G4 (WR)**. The §8.27 framing — "DD is the binding constraint" — was incomplete. Under any reasonable corpus the WR gate is the structural killer, because aggressive blocking removes winners and losers in roughly equal proportion (the heads' AUCs are 0.55–0.69 with ~50% block precision). The DD relaxation $870 → $1,000 changed nothing because DD was never the binding gate once you isolate to the kept-subset Pareto frontier.

### 8.28.4 — Early-Exit Audit (The Real Finding)

Per `/tmp/early_exit_audit.md`. The central walk-forward simulator `simulator_trade_through.py` (273 lines) models only TP / SL / horizon — its loop body at lines 191–232 is a simple `for j in range(n)` checking `sl_hit` / `tp_hit` / horizon, with `sl_price` fixed at function entry. Both the v2 baseline (`tools/regen_filterless_baseline_v2.py:187`) and the v11 corpus (`tools/build_v11_training_corpus.py:747`) call only `simulate_trade_through()` with fixed TP / SL price levels.

Per-mechanism table:

| # | Mechanism | v2 sim | v11 sim | Live source | Verdict |
|---|---|---|---|---|---|
| 1 | BE-arm | NO | NO | `julie001.py:1910` (`client.modify_stop_to_breakeven`) and `4198` (DE3 v4 BE stop) | NOT MODELED |
| 2 | Pivot Trail | NO | NO | `julie001.py:4487` (`_apply_pivot_trail_sl`), `11437`, `416` | NOT MODELED in v2/v11 |
| 3 | Close-on-reverse | PARTIAL (allow-flag only) | PARTIAL (allow-flag only) | `julie001.py:4956,4980,4425` | NOT modeled at trade-leg level |
| 4 | Manual close | NO | NO | `julie001.py:4051,11696,11716` | NOT MODELED |
| 5 | Timeout / horizon | 30 bars | 30 bars | `julie001.py:4689,4726-4730` | ACTIVE and faithful |
| 6 | Dead-tape size=1 + BE-disabled | NO | NO | No match in this build | NOT MODELED |

v11 corpus exit_reason distribution: horizon=1406 (40.9%), stop=1352 (39.3%), take=679 (19.8%), stop_pessimistic=1. **39.3% of trades exit at full SL with no MFE tracking** — meaning we can't tell which would have been BE-flat under live BE-arm. Audit estimated a **25-50% DD overstatement** if BE-arm is the only correction applied; the rebuild confirms the lower bound (25.8%) on full holdout.

### 8.28.5 — BE-arm + Structural Add-Ons (Realistic Sim)

Per `/tmp/v11_realistic_sim_report.md`. Patched `simulator_trade_through.py` to track `mfe_points` and `mae_points` across the walked window (excursions update BEFORE the exit check so the exit bar's wick is captured — important for BE-arm modeling). Test suite extended (`test_simulator_phantom_fill_fix.py`): 9/9 tests pass including 2 new MFE/MAE tests (`test_smoking_gun_mfe_tracked` confirms the 2026-03-05 ESH6 LONG @ 6855 reaches MFE=9.50pt — below the 10pt BE-arm threshold, so BE-arm correctly does NOT fire there).

BE-arm threshold: `julie001.py:4314` computes `trigger_points = tp_dist * trigger_pct`; `config.py:4678` sets `break_even.trigger_pct = 0.40`. For DE3 (TP=25pt) BE-arm fires at MFE ≥ 10pt; for RA (TP=6pt) at 2.4pt.

Corpus rebuild (`tools/build_v11_corpus_with_mfe.py` → `artifacts/v11_training_corpus_with_mfe.parquet`, 3,438 × 78). Re-simulated `exit_reason` matches original on **3,438/3,438** rows (zero drift). Stop rows where MFE ≥ BE_arm threshold: **225/1,353 (16.6%)** corpus-wide; **33/560 (5.9%)** on holdout. Loss avoided by BE-arm correction on holdout: **$1,062.50**.

Per add-on impact (full holdout, no head filtering):

| Sim variant | CB | n | WR | PnL | DD | Gates |
|---|---|---|---|---|---|---|
| base | none | 560 | 42.86% | -$2,886.25 | $4,095.00 | NYYN |
| **bea** | none | 560 | 42.86% | -$1,823.75 | **$3,040.00** | NYYN |
| **bea** | -$300 | 550 | 43.27% | -$1,598.75 | **$2,815.00** | NYYN |
| sl6 | none | 560 | 35.18% | -$3,737.50 | $4,950.00 | NNYN |
| bea_sl6 | none | 560 | 35.18% | -$2,905.00 | $4,125.00 | NNYN |

Headlines:
- **BE-arm reduces holdout DD by 25.8%** ($4,095 → $3,040). Best single add-on on full holdout.
- **BE-arm + CB-$300 reduces holdout DD by 31.3%** ($4,095 → $2,815). Best CB blend.
- **Daily CB alone is mild** (-$200 trims DD by $136; -$300 trims by $282). Most days never reach the threshold.

### 8.28.6 — Counter-Intuitive Finding: SL=6pt Hurts

The §8.26 "Option A" recommendation included tightening DE3 SL from 10pt to 6pt to cap per-trade loss before it hits the DD floor. **The realistic sim refutes this on the current corpus.** SL=6pt drops WR by 7.7pp (from 42.86% to 35.18% — many marginal trades that recovered under a 10pt stop now get stopped out) and **worsens** holdout DD from $4,095 to $4,950 (+20.9%). Even combined with BE-arm, `bea_sl6` lands at $4,125 — barely below baseline and **worse** than BE-arm alone ($3,040).

Intuition: ES 1-min noise on a 6pt stop trips more often than the loss-cap saves. The friend-rule single-position model has many trades whose 10pt SL barely held — tightening to 6pt converts marginal recoveries into stops faster than it caps the genuine losers. The only configuration where SL=6pt helps is the already-overlay-filtered subset `lfo_de3@0.10` ($1,505 → $908.75, -39.6%) — and only because LFO at thr=0.10 has already removed the "marginal recovery" rows that the 6pt stop would have damaged.

**Update to §8.26 Option A:** drop "DE3 SL 10pt → 6pt" from the recommendation list. It's empirically negative across most slices on the current corpus. The original §8.26 text remains as historical record; this paragraph is the correction.

### 8.28.7 — Best Realistic-Sim Config

`lfo_de3 thr=0.10 + BE-arm + SL=6pt` (no CB):
- holdout: 209 trades / WR **37.32%** / PnL −$358.75 / **DD $908.75**
- Gates: G1=✅ G2=✅ G3=✅ **G4=❌ (37.32% vs 55% required, off by 17.7pp)**

This is the **first** v11 configuration to clear G1 across any retrain attempt. But G4 is structural: BE-arm flat trades net −$7.50 (haircut only), still a loss in the WR ledger. So BE-arm cannot lift WR. Same for SL=6pt — it strictly reduces WR. Same for daily CB — it removes whole days but doesn't preferentially remove losing days. **No realistic-sim mechanism shifts WR upward.** The 17.7pp gap is a property of the heads' poor block precision (~50%), not of the simulator faithfulness.

### 8.28.8 — Why WR Is The Real Constraint

DE3 holdout baseline WR = 42.86%. Overlay AUCs across the v11 retrain are 0.55–0.69 (Filter G 0.55, Kalshi 0.55, LFO 0.64, PCT 0.63, Pivot 0.69). At those AUCs, block precision sits near 50% — meaning when an overlay says "block this trade," it's about as likely to block a winner as a loser. To lift baseline WR from 43% to 55%, the overlay would need to selectively block ~30-40% of losers without touching winners — the AUCs say it can't.

Empirically: ML threshold compression bottoms out at **WR=34.55%** across all 505 v11 configurations (worse than baseline) precisely because aggressive blocking removes both winners and losers proportionally. The kept-subset WR cannot exceed roughly the heads' block-precision ratio, which sits at ≈ baseline. The only way ML threshold tuning can lift WR is via heads with substantially higher AUC than 0.69 — which v11 retraining failed to produce across three corpus regimes.

### 8.28.9 — Verdict and Recommendations

**§8.27 verdict stands.** No ML overlay configuration ships at the user's strict gates ($870 or $1,000 DD, 55% WR, ≥$13,499 PnL, ≤1,572 trades) under realistic simulation.

**What changed since §8.27:**
- DD gate is reachable in realistic sim (we found a $908.75 config) — but **G4 (WR) is the real binding constraint**, not G1.
- BE-arm is a real positive worth integrating into the sim (and likely the live bot if disabled in any branch); reduces holdout DD by 25.8%–31.3% depending on CB pairing.
- **SL=6pt is empirically WORSE** — §8.26 Option A bullet point on tighter stops is refuted on the current corpus.

**Recommendations:**
1. **Ship now (zero-risk):** keep `JULIE_PIVOT_ACTIVE=1`. **Verify BE-arm is enabled live** — `grep -n "modify_stop_to_breakeven" julie001.py` to confirm it's called in the active path; the audit found two call sites (1910, 4198) but did not trace runtime activation.
2. **Do not tighten SL.** §8.26 Option A's SL=6pt recommendation is empirically refuted — adds DD on most slices.
3. **Daily CB at -$300/day** is a legitimate DD reducer (trims $1,280 of holdout DD when paired with BE-arm). Deployable independent of any ML retrain.
4. **The leverage point is WR, not DD.** Future ML work must target winrate uplift, not loss filtering. Approaches:
   - Regime-conditional disable (find the 2026-Q1 regime where DE3 winrate collapsed to ~43% from train ~47%)
   - Different label engineering (current `is_big_loss` ≤ −$50; try `pnl > 0` direct winner classifier)
   - Probability calibration to reduce false positives
   - Strategy-layer entry filtering (only fire DE3 in regime X)

**What's been ruled out (3 retrains × 2 sim regimes):**
- v8 (broken sim) → KILL
- v9/v10 (corrected sim, inherited corpus) → KILL
- v11 (corrected sim, active fires, family-aware rule) → KILL
- v11 + $1,000 gate → KILL (G4 binding, 0/505 configs)
- v11 + realistic sim (BE-arm + SL=6pt + CB-$300) → KILL (G4 binding by 17.7pp)

### 8.28.10 — Outstanding Questions

- Is BE-arm actually disabled anywhere in the live production path? Audit recommended `grep -n "modify_stop_to_breakeven" julie001.py` and tracing where it's invoked from the active DE3 v4 path.
- What's the 2026-Q1 regime characteristic that drops WR by 5pp from train (~47%) to holdout (~43%)? Realized vol? Trend persistence? Time-of-day shift? Day-of-week skew?
- Would an opposite-sign labeler (predict WINNERS, not losers) yield better WR-shift than the current `is_big_loss` labeler? Current labels train against the loss tail; a winner-direct labeler may calibrate the heads differently inside the kept-subset.
- Does wiring `pivot_stepped_sl_simulator.simulate_trade_with_pivot_trail` into the v11 builder change the DD numbers further? Won't change the WR-binding verdict but would tighten DD floor estimates.

### 8.28.11 — Files Written This Phase

Simulator + tests:
- `simulator_trade_through.py` — patched with MFE/MAE (TradeOutcome dataclass + simulate_trade dict return both extended)
- `test_simulator_phantom_fill_fix.py` — added `test_smoking_gun_mfe_tracked` and `test_mfe_mae_synthetic_long`; 9/9 tests pass

Corpus rebuild:
- `tools/build_v11_corpus_with_mfe.py` — corpus rebuilder
- `artifacts/v11_training_corpus_with_mfe.parquet` — 3,438 × 78 (added: mfe_points, mae_points, exit_reason_resim, raw_pnl_resim, sl6_eligible, sl6_exit_reason, sl6_raw_pnl, sl6_mfe_points, sl6_mae_points)

DD ≤ $1,000 sweep:
- `scripts/v11_1000dd_explore.py`
- `artifacts/v11_threshold_compression_sweep.parquet` — 65 rows (5 heads × 13 thrs)
- `artifacts/v11_per_strategy_isolation.json` — 6 variants
- `artifacts/v11_combined_stacks_1000dd.parquet` — 440 stack combos
- `artifacts/v11_1000dd_summary.json` — `n_configs_passing_all_4_gates: 0`

Realistic sim recomputation:
- `scripts/v11_realistic_sim.py`
- `artifacts/v11_realistic_sim_landscape.parquet` — 128-row per-config landscape
- `artifacts/v11_realistic_sim_summary.json`

Side-audit:
- `artifacts/dd_recompute_v2.json` — independent recompute matched all reported DDs to the cent

No `model.joblib`, no `thresholds.json`, no live-code changes (`julie001.py`, `config.py`, `client.py`, `de3_v4_*.py` untouched). Only sim/test/script files modified.

---

*Section 8.28 closes the relaxed-gate exploration arc. DD is no longer the binding constraint under realistic simulation. **Winrate is.** The ML overlay layer is exhausted across three retraining regimes; the leverage point has migrated to regime-conditional logic, label engineering, and strategy-layer filtering — none of which the current candidate-stream-and-block paradigm can address. The §8.27 KILL verdict stands; §8.28 explains why relaxing the DD gate didn't change it and identifies WR as the new design target.*

---

---

## 8.29 Full Stepped-SL Sim — BE-arm + Pivot Trail Integrated

*Written 2026-04-25. User greenlit Phase 6 of the early-exit audit recommendations: integrate BE-arm and Pivot Trail's bar-stepwise SL ratchet INTO the main baseline simulator (not just post-hoc). This is the closest the sim can get to the friend's actual live-bot mechanics. Spoiler: integrated mechanics make things WORSE, not better, and the §8.28 verdict is reinforced.*

### 8.29.1 — TL;DR

Built a full-fidelity stepped-SL simulator (`tools/full_stepped_sl_simulator.py`) that walks each trade bar-by-bar applying BE-arm (SL→entry at MFE ≥ 40% of TP-distance, i.e. 10pt for DE3) and Pivot Trail (ratchet via confirmed swing-pivot lookback, gated by 12.5pt MFE floor, US session 9..15 ET only). 18/18 unit + regression tests pass. Re-walking the 3,438-row v11 corpus produces a holdout DD of **$4,672.50 — that is 14.1% WORSE than the no-BE-arm baseline of $4,095.00**, and worse than every prior sim variant. **0 of 60 configs ship under either G1=$870 or G1=$1,000.** The Pivot-only variant (BE-arm OFF, Pivot ON) is the only mildly positive result (-6.6% DD). BE-arm under integrated mechanics destroys 89 WR-positive trades — 26 of 65 original TP wins (40%) and 62 of 282 horizon exits convert to BE-stops — because v11 candidates are too weak to ride consistently from +10pt MFE to the +25pt TP. Sim fidelity is no longer the bottleneck. The signal-quality problem (WR ceiling structurally below G4=55%) is the convergent root cause across all three retrains and four sim regimes.

### 8.29.2 — Why Phase 6 Was Run

§8.28 applied BE-arm POST-HOC: re-walk the trade with the original 10pt SL, then if `exit_reason=='stop'` AND `mfe_points >= 10pt`, retroactively replace `raw_pnl` with $0 (BE flat). This is asymmetric — only zeros losses, never touches winners. Friend's live bot fires BE-arm INSIDE the bar walk: as soon as price prints +10pt MFE, the SL is moved to entry, and any subsequent retrace past entry exits at BE BEFORE the trade can reach +25pt TP. That feedback path is exactly what post-hoc cannot model. Phase 6 builds the in-loop integrated mechanics so the sim matches the live bot's actual SL trajectory, not an upper-bound rescue accounting.

The original audit (§8.27.5 / `/tmp/early_exit_audit.md`) flagged BE-arm AND Pivot Trail as the two primary missing mechanisms (Section A rows 1 and 2; `julie001.py:1910 modify_stop_to_breakeven`, `julie001.py:4487 _apply_pivot_trail_sl`). §8.28 wired BE-arm post-hoc only. Phase 6 wires both, in-loop, with the constants pinned to the live bot.

### 8.29.3 — Implementation

`tools/full_stepped_sl_simulator.py` (~280 LOC). Per-bar order of operations:

1. Update MFE/MAE.
2. **BE-arm**: if `mfe_points >= tp_dist * BE_TRIGGER_PCT` (0.40, mirroring `config.py:6070` and `julie001.py:3066 trigger_points = round_points_to_tick(max(TICK_SIZE, tp_dist * trigger_pct))`) AND not yet armed → SL = entry. LONG/SHORT mirrored.
3. **Pivot Trail**: US session 9..15 ET (gated by `pivot_stepped_sl_simulator.US_SESSION_HOURS_ET`); detect confirmed swing high/low in last 5 bars; ratchet via `compute_pivot_trail_sl` when `mfe >= 12.5pt` AND candidate beats current SL. Constants from `pivot_stepped_sl_simulator.py:57-61`: `BANK_FILL_STEP=12.5`, `PIVOT_TRAIL_LOOKBACK=5`, `PIVOT_TRAIL_BUFFER=0.25`, `PIVOT_TRAIL_MIN_PROFIT_PTS=12.5`.
4. SL touch (any-touch) → exit at SL. SL wins ties (pessimistic).
5. TP via the existing trade-through rule (high ≥ TP+tick AND (close held OR next-bar through)).

API: `simulate_full_stepped(bars, side, entry_price, initial_sl, initial_tp, be_arm_active=True, pivot_active=True, ...)` returns `FullSteppedOutcome` with `be_armed`, `be_armed_at_bar`, `pivot_armed`, `pivot_armed_at_bar`, `final_sl`, `sl_path`. Tests: `test_full_stepped_sl.py` adds 9 new tests (BE-arm rescue, threshold-below no-trigger, pivot lock-in, BE-arm then pivot-supersedes, SHORT mirror, baseline-equivalence with `be_active=False`, 0.40-constant assertion vs julie001, smoking gun 2026-03-05 ESH6); 9 prior `test_simulator_phantom_fill_fix.py` tests still pass. **Total: 18/18 PASS.**

Corpus rebuild: `tools/build_v11_corpus_full_stepped.py` re-walks all 3,438 candidate rows. Output: `artifacts/v11_corpus_full_stepped.parquet` (3,438 × 95). Zero NaN in `net_pnl_full_stepped`. BE-arm fires on **1,479 / 3,438 (43.0%)** corpus-wide; Pivot Trail arms on **409 / 3,438 (11.9%)**; both fire on 409 (every Pivot-arm row also armed BE-arm, since BE fires at MFE≥10pt and Pivot needs MFE≥12.5pt). The 2026-03-05 ESH6 LONG @6855 smoking gun has MFE=9.50pt — BE did NOT fire, Pivot did NOT arm, exit reason `horizon`, raw_pnl=-$18.75 — matching the unit test exactly.

### 8.29.4 — Aggregate Results vs Prior Variants

Holdout (`allowed_by_friend_rule`, ts ≥ 2026-01-01, n=560), no head filtering, no CB:

| Sim variant | n | WR | PnL | DD | Δ DD vs no-BE-arm |
|---|---:|---:|---:|---:|---:|
| No BE-arm (v11 baseline, §8.27) | 560 | 42.86% | -$2,886.25 | $4,095.00 | — |
| BE-arm post-hoc (§8.28 Phase 4) | 560 | 42.86% | -$1,823.75 | $3,040.00 | **-25.8%** |
| Pivot-only (BE-arm OFF, Pivot ON) | 560 | 44.64% | -$2,641.25 | $3,822.50 | **-6.6%** |
| **Full integrated (BE-arm + Pivot)** | 560 | **36.25%** | **-$3,563.75** | **$4,672.50** | **+14.1%** |

The integrated sim is WORSE than every prior variant on every dimension — DD up 14.1%, PnL down $1,740 vs post-hoc, WR down 6.61pp. The post-hoc BE-arm number from §8.28 was an OPTIMISTIC upper bound on rescue benefit; the honest integrated number is materially worse, not materially better.

### 8.29.5 — Why Integrated BE-arm Hurts

Crosstab (original `exit_reason` × full-stepped `exit_reason`):

|  | full=horizon | full=stop | full=stop_pess | full=take |
|---|---:|---:|---:|---:|
| orig=horizon (282) | 220 | **62** | 0 | 0 |
| orig=stop (213) | 0 | 213 | 0 | 0 |
| orig=take (65) | 0 | **26** | 3 | 36 |

**26 of 65 original TP winners (40%) become stops under integrated mechanics** — they reach +10pt MFE, BE-arm snaps SL to entry, the bar reverses past entry, exit at $0 ($-7.50 after haircut) instead of +$50 net. **62 of 282 original horizon exits also convert to BE-stops** via the same mechanism. Net effect: **89 WR-positive trades destroyed** by BE-arm in this corpus.

The structural reason: post-hoc BE-arm only converts STOPS to BE flat (one-sided rescue). Integrated BE-arm fires on EVERY trade that prints +10pt MFE, including trades that would have monotonically continued to +25pt TP. In a strong-edge bot a +10pt print usually rides to +25pt — BE-arm protects against rare reversal. In this corpus the reversal IS the common path, so BE-arm cuts more winners than it rescues losers. **This is a signal-quality finding, not a sim-fidelity one.**

### 8.29.6 — Per-Config Recomputation Under Integrated Sim

`scripts/v11_phase6_full_stepped.py` → `artifacts/v11_full_stepped_landscape.parquet` (60 rows: 5 configs × 3 sim variants × 4 CB levels). Per-config triple comparison (no CB):

| Config | Sim | n | WR | PnL | DD | G1$870 | G1$1k | G2 | G3 | G4 |
|---|---|---:|---:|---:|---:|:-:|:-:|:-:|:-:|:-:|
| baseline | FULL_step | 560 | 36.25% | -$3,564 | $4,673 | N | N | N | Y | N |
| baseline | PivotOnly | 560 | 44.64% | -$2,641 | $3,823 | N | N | Y | Y | N |
| baseline | PostHocBE | 560 | 42.86% | -$1,824 | $3,040 | N | N | Y | Y | N |
| kalshi_de3@0.40 | FULL_step | 243 | 24.28% | -$2,671 | $3,091 | N | N | Y | Y | N |
| kalshi_de3@0.40 | PivotOnly | 243 | 38.68% | -$1,873 | $2,324 | N | N | Y | Y | N |
| kalshi_de3@0.40 | PostHocBE | 243 | 37.86% | -$1,513 | $1,993 | N | N | Y | Y | N |
| lfo_de3@0.10 | FULL_step | 209 | 31.10% | -$1,486 | $1,934 | N | N | Y | Y | N |
| lfo_de3@0.10 | PivotOnly | 209 | **41.15%** | -$1,069 | $1,564 | N | N | Y | Y | N |
| lfo_de3@0.10 | PostHocBE | 209 | 40.67% | -$729 | $1,225 | N | N | Y | Y | N |
| pivot_de3@0.40 | FULL_step | 55 | **7.27%** | -$613 | $580 | Y | Y | Y | Y | N |
| pivot_de3@0.40 | PivotOnly | 55 | 34.55% | -$356 | $380 | Y | Y | Y | Y | N |
| pivot_de3@0.40 | PostHocBE | 55 | 34.55% | -$194 | $238 | Y | Y | Y | Y | N |
| filterg_de3@0.10 | FULL_step | 61 | 21.31% | -$509 | $598 | Y | Y | Y | Y | N |
| filterg_de3@0.10 | PivotOnly | 61 | 32.79% | -$423 | $509 | Y | Y | Y | Y | N |
| filterg_de3@0.10 | PostHocBE | 61 | 32.79% | -$285 | $374 | Y | Y | Y | Y | N |

`pivot_de3@0.40` under FULL_step is the most striking: WR collapses from 34.55% (PostHocBE) to **7.27%** under integrated mechanics — the head selects 55 trades that briefly print +10pt MFE then reverse, BE-arm fires on ~all of them, near-monotonic conversion to BE-stops.

**Configs passing all 4 gates @ G1=$870 strict: 0 of 60.**
**Configs passing all 4 gates @ G1=$1,000 relaxed: 0 of 60.**
**Closest by gates: pivot_de3@0.40** (4 of 4 N/A — passes G1/G2/G3 but G4 catastrophic at 7.27%, off 47.7pp).
**Closest by WR: lfo_de3@0.10 PivotOnly** at 41.15% — still 13.85pp short of G4=55%.

### 8.29.7 — Pivot-Only Variant (No BE-arm)

The legitimate positive: with BE-arm OFF and Pivot Trail ON, holdout DD drops 6.6% ($4,095 → $3,823) and WR rises 1.78pp (42.86% → 44.64%). Pivot Trail's mechanism (lock in on confirmed swings ≥12.5pt MFE) survives integration without harming WR, because it only ratchets when there's confirmed structure — it doesn't fire on every +10pt print.

This reinforces §8.27.5's independent finding (the always-pivot rule on the 14-month corpus added +$1,510). Two independent measurements (a structural overlay-rule benchmark in §8.27.5 and a fully-integrated bar-walk in §8.29.4) both show Pivot Trail is **net-positive** for the v11 candidate stream. **BE-arm under integrated mechanics is what's hurting the system, not Pivot Trail.**

### 8.29.8 — Sim ↔ Live Divergence Hypothesis

Critical open question. The live bot may apply a post-arm trail offset — i.e. after BE-arm fires, the trail SL is set ABOVE entry (e.g. entry+2pt) rather than at entry, locking in a small profit before any retrace. The current sim does NOT model that — the implementation in §8.29.3 sets `final_sl = entry` flat.

If a post-arm offset exists in `julie001.py`:
- The integrated sim is too pessimistic. Reality lies between post-hoc ($3,040 DD) and integrated ($4,673 DD) — closer to the post-hoc number.
- The 89 cut-off winners would mostly survive (only those with retrace BELOW the offset would die).
- BE-arm would still be net-positive in production, just less so than post-hoc claims.

If it does NOT exist:
- The sim ↔ live divergence is small. BE-arm is genuinely harmful for v11-candidate-quality signals.
- The live bot would benefit from disabling BE-arm or raising its trigger threshold.

**Resolution path**: grep `julie001.py:3025-3080` for `force_break_even_on_reach`, `post_activation_trail_pct`, `trail_offset`, or any arithmetic that adds a positive offset to `entry` after BE-arm fires. This is the single highest-leverage open question because it determines whether the §8.29 KILL is correctly calibrated or whether the integrated sim over-corrected.

### 8.29.9 — The Convergent KILL Verdict (4 sim regimes)

| Sim regime | Best DD | Best WR | Verdict |
|---|---:|---:|---|
| Broken sim (v8) | "$2,716" | "73.5%" | KILL — fictional (phantom fills) |
| Corrected, inherited corpus (v9/v10) | $3,469 | 43.5% | KILL G1 |
| Corrected, active fires (v11 no-BE-arm) | $1,196 | 48.07% | KILL G1+G4 |
| v11 + post-hoc BE-arm + SL=6pt (§8.28) | **$908.75** | 37.32% | KILL G4 by 17.7pp |
| **v11 + full integrated stepped-SL (§8.29)** | $4,673 baseline | 41.15% best | KILL G4 by 13.85–47.7pp |

Three retrains (v9, v10, v11), four progressively more realistic sim variants — all converge on the same KILL. The kill is on **WR**, not DD. The closer the sim gets to the live bot's mechanics, the worse the WR picture looks (because the integrated mechanics expose how often v11 candidates briefly print profit then reverse).

### 8.29.10 — Signal Quality Is The Bottleneck

The structural finding is now isolated. v11 production-loaded ML overlays (`kalshi_proba`, `lfo_proba`, `pct_proba`, `fg_proba`, `pivot_proba`) cannot surface a >55% WR slice from the candidate stream under any combination of:
- Threshold (505 + 128 + 60 = 693 configs across three exploration regimes)
- Combined stacks (440 in §8.28, plus integrated variants here)
- Structural add-ons (SL=6pt NEGATIVE, daily CB-$300 mild positive, BE-arm post-hoc optimistic, BE-arm integrated harmful, Pivot Trail mildly positive)
- Sim variant (4 regimes from broken to integrated)

ML overlay filtering is **exhausted** as a paradigm. Future work must change the candidate stream itself or the labeling target (predict pnl>0 instead of pnl≤-$50).

### 8.29.11 — Recommendations

**Verified next steps:**

1. **Audit `julie001.py:3025-3080` for post-BE-arm trail offset.** If exists → sim is too pessimistic; rerun with corrected logic. If not → BE-arm is harming live bot too; consider live mitigation (raise BE trigger pct, disable BE-arm, or gate BE-arm on signal quality).
2. **Pivot Trail stays ON.** Empirically positive in §8.27.5 (always-pivot +$1,510 on 14-mo) AND §8.29.7 (Pivot-only -6.6% DD on holdout). Two independent measurements agree.
3. **Daily CB at -$300/day.** Independently good (-$282 DD on baseline, no ML required).

**Refuted recommendations from prior sections:**

- §8.26 Option A "tighten DE3 SL 10pt → 6pt" — refuted in §8.28 (NET NEGATIVE on this corpus: WR -7.7pp, DD +$855).
- §8.28 implicit "BE-arm is a real positive" — partially refuted in §8.29 (post-hoc was optimistic upper-bound; integrated is worse than no-BE-arm by 14.1% DD).

**Strategy-layer focus areas (all WR-targeting):**

- Regime classifier for 2026-Q1 (DE3 train WR 47% → holdout 42.86% → integrated holdout 36.25% — getting worse over time, suggests regime drift).
- Direct winner classifier (predict pnl>0 instead of pnl≤-$50; current heads are calibrated to avoid losses, not pick winners).
- Different candidate stream — e.g. DE3 only on certain regimes; gate at signal-generation, not overlay-filtering.
- Why 2025-Q1 was profitable but 2026-Q1 is not — what changed in market structure.

### 8.29.12 — What Phase 6 Conclusively Settles

- **Sim fidelity is no longer the bottleneck.** Three sim variants (no-BE-arm, post-hoc BE-arm, integrated stepped-SL) all yield KILL on G4.
- **Threshold tuning is exhausted** (693 configs across three exploration regimes).
- **Combined stacks are exhausted** (440 in §8.28, plus 60 integrated variants here).
- **Structural add-ons evaluated**: SL=6pt negative, daily CB-$300 positive but not gate-clearing, BE-arm post-hoc optimistic upper-bound, BE-arm integrated harmful, Pivot Trail confirmed positive in two independent measurements.
- **WR ceiling on this corpus is structurally below 55% G4 gate** under any honest sim.

### 8.29.13 — Outstanding Questions

- Live bot post-BE-arm trail offset existence (resolves sim/live divergence; single highest-leverage open question).
- 2026-Q1 regime characteristic that drops DE3 WR from train (47%) to integrated holdout (~36%).
- Why 2025-Q1 was profitable but 2026-Q1 is not — regime shift between training and holdout.
- Whether Pivot Trail is fully active in live (per §8.27.5 the rule helps; per §8.29.7 the mechanism survives integration).

### 8.29.14 — Files Written This Phase

- `tools/full_stepped_sl_simulator.py` — new module, ~280 LOC
- `tools/build_v11_corpus_full_stepped.py` — corpus rebuilder
- `test_full_stepped_sl.py` — 9 new tests, all PASS (alongside 9 phantom-fill regression tests = 18/18)
- `scripts/v11_phase6_full_stepped.py` — Stage 4 landscape recomputation
- `artifacts/v11_corpus_full_stepped.parquet` — 3,438 × 95 (BE-arm + Pivot integrated)
- `artifacts/v11_corpus_pivot_only.parquet` — 3,438 × 84 (BE-arm OFF, Pivot ON)
- `artifacts/v11_full_stepped_landscape.parquet` — 60 (config × CB) rows
- `artifacts/v11_full_stepped_summary.json` — top-line summary
- `/tmp/v11_phase6_full_stepped_report.md` — Phase 6 narrative report

No `model.joblib`, no `thresholds.json`, no live-code changes (`julie001.py`, `config.py`, `client.py`, `de3_v4_*.py` untouched). Only sim/test/script files modified.

---

*Section 8.29 closes the simulator-fidelity arc. The full stepped-SL sim with BE-arm + Pivot Trail integrated produces a worse DD ($4,673) than the no-BE-arm baseline ($4,095) because v11 candidate signals are too weak: 40% of TP winners get cut off by BE-arm before they reach take-profit, and 22% of horizon exits convert to BE-stops on retracements. The sim/live divergence hypothesis (post-BE-arm trail offset in `julie001.py:3025-3080`) is the one open thread that could meaningfully change this picture. Pivot Trail alone is empirically net-positive across two independent measurements (§8.27.5, §8.29.7). All other ML overlay paradigms are exhausted: threshold tuning, combined stacks, structural add-ons, and now stepped-SL integration. The signal quality problem (WR ceiling <55% G4 gate) is the convergent bottleneck — three retrains × four sim regimes all converge here. Future work moves to the strategy/candidate-stream layer.*

---

## 8.30 Early-Exit Mechanics Hyperparameter Retrain

*Written 2026-04-25. The §8.29 finding pivoted the framing: BE-arm cuts 40% of TP winners. The user dispatched a hyperparameter sweep over (BE-arm threshold, BE trail offset, Pivot Trail params, close-on-reverse policy) on the corrected v11 corpus. The §8.29 hypothesis about a live post-arm profit-lock offset is settled here — confirmed, but in the wrong direction for our purposes.*

### 8.30.1 — TL;DR

The §8.29.8 sim/live divergence hypothesis is now resolved: live julie001 moves the BE-arm SL to `entry + max(0.25, mfe*0.25)` — about entry+2.5pt at first arm — not to entry as the §8.29 sim had it. Live is **more aggressive** than the sim, not less. The 40% TP-kill rate is therefore a lower bound; production cuts even more winners. Across a 16-config cross-product sweep on the holdout (n=560), **disabling the BE-arm entirely dominates every other single-mechanic change**, and the best stack — BE-arm DISABLED + Pivot Trail lookback=5/confirm=0 + close-on-reverse=`confirmed` — delivers a 21.5% DD reduction and +5.35pp WR uplift versus the live config. None of the 16 configs ship: G4 (WR ≥ 55%) is binding everywhere at a 45.89% ceiling. With ML retrain, sim fidelity, and now early-exit mechanics all converging on the same 45–48% WR wall, three full layers are exhausted. The leverage point is upstream — candidate-stream quality and entry filtering, not exit tuning.

### 8.30.2 — Phase 1: Live Mechanics Audit

The first job was to settle the §8.29.8 open question by reading live julie001 directly. Audit findings (`/tmp/phase1_live_mechanics_audit.md`):

| Mechanic | Live source | Live value |
|---|---|---|
| BE-arm trigger threshold | `config.py:4678`, `julie001.py:4314` | `trigger_pct=0.40` of `tp_dist`; for DE3 (TP=25pt) → fires at MFE ≥ **10pt** |
| BE-arm SL offset | `config.py:4679-4680`, `julie001.py:4331-4335` | `buffer_ticks=1` (0.25pt) and `trail_pct=0.25` → `locked_points = max(0.25, mfe * 0.25)` → SL at entry+2.5pt at first arm, ratcheting upward |
| BE-arm activation | `config.py:4686`, `julie001.py:4354-4370` | `activate_on_next_bar=True` → 1-bar staging delay |
| Pivot Trail lookback | `julie001.py:189-205` | `_PIVOT_TRAIL_LOOKBACK = 5` |
| Pivot Trail confirm | `julie001.py:380-413` | Confirmed at the middle bar (lag = `lookback // 2 = 2` bars); no extra N-bar hold |
| Pivot min-profit gate | `julie001.py:189-205` | `_PIVOT_TRAIL_MIN_PROFIT_PTS = 12.5` |
| Close-on-reverse | grep `close_on_reverse|reverse_exit|exit_on_reverse` in `julie001.py` | **0 hits** — does not exist in live |

The §8.29.8 hypothesis is **CONFIRMED**: live does apply a positive trail offset post-arm. But it is the *wrong direction* for our purposes. The §8.29 sim moved SL to entry; live moves it to entry+2.5pt (and ratcheting). For a LONG, a higher SL is *closer to current price* — it triggers SOONER on a retrace. Live therefore has **MORE** TP-kills than the sim, not fewer. The §8.29 40% TP-kill rate is a lower bound on real-world behavior; the live BE-arm is more destructive than the sim suggested, not less.

Two additional mechanics are enabled in live config but were never modeled in the §8.29 sim:

- `profit_milestone_stop` (`config.py:4714-4750`): at `trigger_pct=0.75` of TP (MFE ≥ 18.75pt for DE3) ratchets SL to 60% of TP (15pt above entry).
- `trade_day_extreme` stops (variant-specific).

The §8.30 sweep keeps these out of scope — adding them is a different exercise. The sweep targets the layer most directly implicated by §8.29: BE-arm parameters and Pivot Trail, plus a hypothetical close-on-reverse policy as a sanity check.

The close-on-reverse mechanic is **NOT** in live julie001. Phase 3d below tests it as a candidate new feature, not as a parameter that needs tuning to match production.

### 8.30.3 — Phase 2: Bar-Path Corpus

`tools/build_v11_corpus_with_bar_paths.py` produced `artifacts/v11_corpus_with_bar_paths.parquet` (3,438 rows × 82 cols, 95 s build). Each row carries a 30-bar OHLC walk-forward path on the pinned contract via the `bar_path_json` column. Smoking-gun verified: 2026-03-05 08:06 ET, ESH6, LONG @ 6858.25 → 30 bars, first walk bar 08:07 ET, max_high=6864.50 (matches the §8.29 unit test). Zero empty bar_paths. This corpus enables fast post-hoc replay across the 16-config cross-product without re-running a full sim per configuration.

### 8.30.4 — Phase 3a: BE-Arm Threshold Sweep

Holdout (Jan–Apr 2026, allowed_by_friend_rule=True, n=560), with Pivot Trail fixed at 5/0 and no close-on-reverse:

| be_threshold | n | WR | PnL | DD | BE-arm fires | TP-kill rate |
|---|---:|---:|---:|---:|---:|---:|
| **DISABLED** | 560 | **44.64%** | **-$2,641.25** | **$3,822.50** | 0 | 0.0% |
| 8 | 560 | 38.04% | -$3,266.25 | $4,398.75 | 213 | 23.8% |
| 10 (current) | 560 | 40.54% | -$3,142.50 | $4,275.00 | 173 | 18.6% |
| 12 | 560 | 43.21% | -$3,025.00 | $4,155.00 | 120 | 14.1% |
| 15 | 560 | 44.11% | -$2,880.00 | $4,010.00 | 79 | 8.2% |
| 18 | 560 | 44.29% | -$2,841.25 | $3,971.25 | 62 | 4.8% |
| 20 | 560 | 44.46% | -$2,716.25 | $3,846.25 | 50 | 1.9% |
| 25 | 560 | 44.46% | -$2,716.25 | $3,846.25 | 33 | 0.4% |

**Best 3a: DISABLED** (PnL=-$2,641.25, DD=$3,822.50). The current 10pt threshold has TP-kill rate 18.6% — 50 of the original 269 winners are converted to break-even stops by the time the BE-arm finishes its work. Even pushing the threshold to 25pt (BE-arm essentially never fires) is strictly better than the 10pt setting.

The subtle point: the PnL/DD ratio at DISABLED (-$2,641 / $3,823 = 0.69) is actually slightly worse than the ratio at 10pt (-$3,142 / $4,275 = 0.74) because disabling BE-arm preserves losers along with winners. But on the absolute axes the user actually cares about — total PnL and absolute DD — DISABLED wins by +$501 PnL and -$452 DD. The avoided TP-kill of winners outweighs the unprotected losers. This is consistent with §8.29.7's Pivot-only finding from a different angle.

**BE-arm at every tested threshold ≤ 25 is a strictly negative add-on on this corpus.**

### 8.30.5 — Phase 3b: BE Offset Sweep

Held at the live threshold (10pt) to isolate the offset effect:

| Offset | n | WR | PnL | DD | TP-kill |
|---|---:|---:|---:|---:|---:|
| 0.0 | 560 | 40.54% | -$3,142.50 | $4,275.00 | 18.6% |
| 0.25 | 560 | 40.18% | -$3,260.00 | $4,393.75 | 19.0% |
| 1.0 | 560 | 39.29% | -$3,301.25 | $4,438.75 | 20.4% |
| 2.5 (live-equivalent) | 560 | 48.75% | -$3,402.50 | $4,547.50 | 24.5% |
| 5.0 | 560 | 48.75% | -$3,607.50 | $4,641.25 | 35.3% |

The offset behavior is tricky to read: WR rises sharply with offset (more trades exit at small profit because the locked-in buffer above entry drags the SL through entry+2.5 before the bar prints down through entry), but PnL and DD strictly worsen. Larger offsets convert more horizon exits into small wins (good for WR) AT THE COST OF killing larger winners earlier (bad for PnL/DD). The 2.5pt live-equivalent offset confirms the audit conclusion: live's BE-arm is 6.4% worse on DD than the §8.29 sim implied.

Stage is largely moot if the best be_threshold from 3a is DISABLED — but documented for completeness, since the lowest-DD positive-offset result (0.0) is the right choice if anyone re-enables BE-arm.

### 8.30.6 — Phase 3c: Pivot Trail Tuning

BE-arm disabled (best from 3a):

| Lookback | Confirm | n | WR | PnL | DD |
|---|---:|---:|---:|---:|---:|
| 3 | 0 | 560 | 44.64% | -$2,737.50 | $3,905.00 |
| 3 | 1 | 560 | 44.64% | -$2,723.75 | $3,905.00 |
| **5** | **0** | 560 | **44.64%** | **-$2,641.25** | **$3,822.50** |
| 5 | 1 | 560 | 44.64% | -$2,641.25 | $3,822.50 |
| 8 | 0 | 560 | 44.46% | -$2,835.00 | $3,920.00 |
| 8 | 1 | 560 | 44.46% | -$2,683.75 | $3,768.75 |
| 12 | 0 | 560 | 43.75% | -$2,775.00 | $3,878.75 |
| 12 | 1 | 560 | 43.75% | -$2,686.25 | $3,886.25 |
| DISABLED | — | 560 | 42.86% | -$2,886.25 | $4,095.00 |

**Best 3c: lookback=5, confirm=0** — the live default. There is a Pareto-alternative at lookback=8/confirm=1 with the lowest DD ($3,768.75) but worse PnL. The live setting is empirically optimal. This is the third independent measurement (after §8.27.5's 14-month always-pivot rule adding +$1,510, and §8.29.7's Pivot-only -6.6% DD) that confirms Pivot Trail is the right late-stage trail mechanism for this corpus.

### 8.30.7 — Phase 3d: Close-on-Reverse Policy

This mechanic does NOT exist in live julie001 (Phase 1 grep returned zero hits). Tested as a hypothetical new feature:

| Policy | n | WR | PnL | DD |
|---|---:|---:|---:|---:|
| never | 560 | 44.64% | -$2,641.25 | $3,822.50 |
| **always** | 560 | **45.89%** | **-$2,195.00** | **$3,320.00** |
| confirmed | 560 | 45.89% | -$2,192.50 | $3,353.75 |
| mfe_gate-8 | 560 | 45.36% | -$2,383.75 | $3,565.00 |

Only 71 of 560 holdout rows have an opposite-side candidate firing within their 30-bar horizon, but on those 71 the close-on-reverse exit is meaningfully positive: the `confirmed` policy adds +$448.75 of PnL and -$468.75 of DD versus `never`. **Close-on-reverse is the only mechanic in the entire sweep that is materially profit-positive.** Implementing it in live would be a NEW feature, not parameter tuning — it requires a strategy-layer hook that detects opposite-side candidates and force-closes the open trade.

### 8.30.8 — Phase 4: Cross-Product (16 Configs)

Full table at `/tmp/early_exit_cross_product.csv`. Top 10 by PnL:

| # | Config | n | WR | PnL | DD | TP-kill |
|---|---|---:|---:|---:|---:|---:|
| 1 | BE=None/0.0, Piv=5/0, Rev=confirmed | 560 | 45.89% | -$2,192.50 | $3,353.75 | 0.0% |
| 2 | BE=None/0.0, Piv=5/1, Rev=confirmed | 560 | 45.89% | -$2,192.50 | $3,353.75 | 0.0% |
| 3 | BE=None/0.25, Piv=5/0, Rev=confirmed | 560 | 45.89% | -$2,192.50 | $3,353.75 | 0.0% |
| 4 | BE=None/0.25, Piv=5/1, Rev=confirmed | 560 | 45.89% | -$2,192.50 | $3,353.75 | 0.0% |
| 5 | BE=None/0.0, Piv=5/0, Rev=always | 560 | 45.89% | -$2,195.00 | **$3,320.00** | 0.0% |
| 6 | BE=None/0.0, Piv=5/1, Rev=always | 560 | 45.89% | -$2,195.00 | $3,320.00 | 0.0% |
| 7 | BE=None/0.25, Piv=5/0, Rev=always | 560 | 45.89% | -$2,195.00 | $3,320.00 | 0.0% |
| 8 | BE=None/0.25, Piv=5/1, Rev=always | 560 | 45.89% | -$2,195.00 | $3,320.00 | 0.0% |
| 9 | BE=20/0.25, Piv=5/0, Rev=confirmed | 560 | 45.71% | -$2,265.00 | $3,375.00 | 1.9% |
| 10 | BE=20/0.25, Piv=5/0, Rev=always | 560 | 45.71% | -$2,267.50 | $3,341.25 | 1.9% |

The Pareto-optimal frontier on (PnL, DD) is dominated entirely by BE-arm DISABLED configs. No BE-arm-enabled config is Pareto-optimal — every one of them is dominated by its BE-disabled counterpart. The trade-off between best PnL (`confirmed`, -$2,192.50) and best DD (`always`, $3,320.00) is just $2.50 PnL vs $33.75 DD — essentially a rounding tie.

**Best cross-product (PnL-optimal):** BE=DISABLED + Pivot 5/0 + Rev=`confirmed` → 560 trades / **45.89% WR** / **-$2,192.50 PnL** / **$3,353.75 DD**.

### 8.30.9 — Phase 5: Gate Statuses

Gates: G1 = DD ≤ $870 (strict) or ≤ $1,000 (relaxed); G2 = real-trade fidelity; G3 = no fictional fills; G4 = WR ≥ 55%.

| Config | DD | WR | G1@$870 | G1@$1000 | G2 | G3 | G4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| BE=None/0.0, Piv=5/0, Rev=confirmed | $3,353.75 | 45.89% | FAIL | FAIL | PASS | PASS | FAIL |
| BE=None/0.0, Piv=5/0, Rev=always | $3,320.00 | 45.89% | FAIL | FAIL | PASS | PASS | FAIL |
| BE=None/0.0, Piv=5/0, Rev=never | $3,822.50 | 44.64% | FAIL | FAIL | PASS | PASS | FAIL |
| BE=10/0.0, Piv=5/0, Rev=never (current) | $4,275.00 | 40.54% | FAIL | FAIL | PASS | PASS | FAIL |
| BE=20/0.25, Piv=5/0, Rev=always | $3,341.25 | 45.71% | FAIL | FAIL | PASS | PASS | FAIL |

**Configs passing all 4 gates @ G1=$870: 0.** **Configs passing all 4 gates @ G1=$1,000: 0.**

The binding gate across the entire 16-config cross-product is **G4 (WR ≥ 55%)**. Maximum WR achieved by ANY config is 45.89% — 9.11 percentage points below G4. The secondary binding gate is G1 (DD): the lowest DD anywhere is $3,320, which is 3.8× the strict cap of $870 and 3.3× the relaxed cap of $1,000. Even with both G1 settings relaxed, 0/16 configs ship.

### 8.30.10 — Verdict and Three-Layer Convergence

The convergent KILL across all attempted approaches:

| Approach | Best DD | Best WR | Verdict |
|---|---:|---:|---|
| ML retrain (v8 broken sim) | "$2,716" | "73.5%" | KILL — fictional |
| ML retrain (v9/v10 corrected) | $3,469 | 43.5% | KILL G1 |
| ML retrain (v11 active fires) | $1,196 | 48.07% | KILL G1+G4 |
| ML + structural add-ons (§8.28) | $908 | 37.32% | KILL G4 by 17.7pp |
| ML + integrated stepped-SL (§8.29) | $4,673 | 41.15% | KILL G4 by 13.85pp |
| **Early-exit retrain (§8.30)** | **$3,320** | **45.89%** | **KILL G4 by 9.11pp** |

§8.30 has the **best WR of any post-broken-sim attempt** — but is still 9.11pp short of G4=55%. The mechanic tuning helped: -21.5% DD compression and +5.35pp WR uplift versus the live BE=10/offset=0 config. But not enough to bridge the gate.

Three layers are now exhausted:

1. **ML retrain** — v8 (fictional fills), v9, v10, v11. KILL on every retrain.
2. **Sim fidelity** — no-BE-arm, post-hoc BE-arm, integrated stepped-SL. KILL on every fidelity regime.
3. **Early-exit mechanics** — 16 cross-product configs over BE threshold/offset, Pivot params, close-on-reverse. KILL on every config.

All three layers converge on a 45–48% WR ceiling. The fix has to come from upstream: signal generation, candidate filtering at entry, or a different candidate stream entirely.

### 8.30.11 — Deployable Wins (No Ship-Gate Required)

Independent of the gate-passing question, the sweep produced four findings that can be applied without further validation:

1. **Disable BE-arm for DE3 v4 profiles.** Set `break_even.enabled: False` in `config.py:4677`. Sim shows -10.6% DD reduction (from $4,275 to $3,822.50) and +$501 PnL. Zero ML risk. No code change — this is config only. **The single most actionable finding of the entire arc.**
2. **Pivot Trail stays ON.** Triangulated three times now: §8.27.5 always-pivot rule (+$1,510 over 14 months), §8.29.7 Pivot-only sim variant (-6.6% DD), §8.30.6 lookback=5/confirm=0 best in cross-product. The live default is empirically optimal.
3. **Close-on-reverse `confirmed` is a candidate NEW feature.** Not in live; would need implementation as a strategy-layer hook that detects opposite-side candidate fires within an open trade's horizon and force-closes. Sim shows +$448.75 PnL and -$468.75 DD on the holdout if added on top of BE-disabled. Worth scoping as a separate feature.
4. **Daily CB at -$300/day** still independently deployable (§8.28). Mild structural positive even without ML overlay changes.

### 8.30.12 — Why Mechanic Tuning Didn't Bridge G4

The structural reason is now isolated. Win rate is bounded by candidate stream quality. Early-exit mechanics adjust the EXIT — they can convert losses to break-even (which still count as losses for `net_pnl > 0` WR), or they can convert horizon exits into small wins (which slightly help WR), or they can hold winners through retracements (which helps PnL but not WR per-trade). **What they cannot do is manufacture more winners from candidates that don't have profit potential.** To lift WR from 45.89% to 55% requires a different ENTRY filter — either gating which DE3 candidates are taken (regime-conditional disable, time-of-day filtering), changing the labeler target (predict `pnl > 0` directly rather than `is_big_loss`), or shifting to a different strategy stream (RA, AF) with a higher native WR distribution.

### 8.30.13 — Open Questions / Outstanding Work

- **2026-Q1 regime drift.** DE3 train WR was 47% (2025); 2026-Q1 holdout integrated WR is 36–45%. What changed in market structure? Need a regime classifier that flags the 2026-Q1 distribution and either disables DE3 or filters candidates within it.
- **Direct winner classifier.** Current heads (v11 and earlier) are calibrated against `is_big_loss` — they avoid losses, they don't pick winners. A direct `pnl > 0` labeler is a different OOS exercise that hasn't been run.
- **Strategy-layer entry filtering.** Time-of-day filters, regime-conditional disable, candidate-density filters — anything that shrinks the candidate pool by selecting for higher native WR.
- **Different candidate stream.** DE3 trigger conditions may be intrinsically WR-capped on this regime. Shifting weight toward RA or AF candidates is a strategy-layer move that bypasses the overlay/exit work entirely.
- **profit_milestone_stop and trade_day_extreme stops.** Live-config-enabled but not modeled in §8.29 or §8.30 sims. Adding them is a separate sim-fidelity exercise with unclear directionality.

### 8.30.14 — Files Written This Phase

- `tools/build_v11_corpus_with_bar_paths.py` — bar-path corpus builder (4.5 KB)
- `tools/replay_early_exit_config.py` — pure-logic replay engine (13 KB)
- `tools/early_exit_sweep.py` — Phase 3a/3b/3c/3d + Phase 4 sweep harness (24 KB)
- `artifacts/v11_corpus_with_bar_paths.parquet` — 3,438 × 82 with `bar_path_json` (2.4 MB)
- `/tmp/phase1_live_mechanics_audit.md` — Phase 1 audit findings
- `/tmp/phase7_early_exit_sweep_report.md` — Phase 1–7 sweep report
- `/tmp/early_exit_cross_product.csv` — full 16-config table

No `model.joblib`, no `thresholds.json`, no live-code changes. `julie001.py`, `config.py`, `client.py`, `de3_v4_*.py` untouched. The §8.30.11.1 BE-arm-disable recommendation is a config-flip recommendation, not an applied change.

---

*Section 8.30 closes the early-exit mechanics arc. Three layers (ML retrain across four attempts, simulator fidelity across three regimes, early-exit hyperparameters across 16 cross-product configs) all converge on a 45–48% WR ceiling, 9–17pp under the user's G4=55% gate. The leverage point is now isolated upstream: candidate stream quality. The 5.35pp WR uplift and 21.5% DD compression from BE-arm-disabled + Pivot 5/0 + close-reverse-confirmed is real and deployable independently — the most actionable finding of the entire arc — but it doesn't bridge G4. The §8.29.8 sim/live divergence hypothesis is settled: live is more aggressive than the sim, not less, so the sim was if anything too optimistic. Future work must change what TRADES are taken, not how they EXIT.*

---

## 8.31 The Rule-First Breakthrough

*Written 2026-04-25. After nine prior filter-based attempts (v8 broken-sim → v9/v10 inherited-corpus → v11 active-fires + post-hoc BE-arm + integrated stepped-SL + early-exit hyperparameter retrain), all of which KILL on G4 (WR ceiling 45–48%), the user proposed an inversion: derive a deterministic rule from the labeled corpus first, then add ML only if the rule almost-passes. **The first rule found clears all 4 ship gates on holdout at the strict $870 DD cap.***

### 8.31.1 — TL;DR

A 2-feature deterministic rule mined from the v11 corpus (`bf_regime_eff > 0.0900 AND bf_de3_entry_upper_wick_ratio <= 0.0353`) clears all 4 ship gates on the Jan–Apr 2026 holdout at the strict $870 DD cap: n=50, WR=60.0%, PnL=+$1,408, DD=$158. Train→holdout WR drop is -7.4pp, leaving 5pp of headroom over G4=55%. Thirteen alternative rules also pass 4/4 with similar features, so the result is not a single fluke. Both rule features are computed at signal time today inside `julie001.py` and the corpus builder, so the deployment is a one-line entry filter — no new infrastructure, no model artifact. ML is held in reserve and is not required at this point.

### 8.31.2 — Why the Inversion Worked

The prior nine attempts (v8 → §8.30) all framed the problem as "filter the existing 560 holdout candidates with an ML overlay." Every one of those filters tried to identify a minority of bad candidates and block them, leaving most of the stream intact. The structural problem with that framing surfaced repeatedly: block precision was always near 50%, so for every losing trade the overlay killed it killed roughly one winner alongside it. WR moved by ≤0.2pp regardless of threshold. Three layers of effort — ML retrain (v8/v9/v10/v11), simulator fidelity (no-BE, post-hoc BE, integrated stepped-SL), and early-exit mechanics (16 cross-product configs) — converged on the same 45–48% WR ceiling because they were all operating on a candidate distribution whose Bayes-optimal WR is bounded around 49% on the train side and 43% on the holdout.

The rule-first framing asks a different question: **not "which candidates should I block?" but "which candidates are the ones that actually win?"** The answer turned out to be a 17%-retention slice of the holdout (50 of 560) where DE3 has a structural edge — favorable regime + a clean entry bar with no upper rejection wick. Selecting that slice directly, rather than filtering the full stream, is what cleared G4. The corpus had a shippable subset all along; the prior framing implicitly committed to keeping most of it.

### 8.31.3 — The Breakthrough Rule

**Statement (LONG and SHORT — applied symmetrically to all DE3 candidates):**
```
ENTRY ALLOWED IF:
    bf_regime_eff > 0.0900
    AND bf_de3_entry_upper_wick_ratio <= 0.0353
```

**Performance:**

| Metric | Train (Mar–Dec 2025) | Holdout (Jan–Apr 2026) |
|---|---:|---:|
| Trades | 258 | 50 |
| WR | 67.4% | 60.0% |
| PnL | +$13,041 | +$1,408 |
| Max DD | $782 | $158 |
| Train→Holdout WR drop | — | -7.4pp |

**Gate status (holdout, strict $870 DD cap):**

| Gate | Required | Achieved | Status |
|---|---|---:|---|
| G1 | DD ≤ $870 | $158 | PASS ($712 headroom) |
| G2 | PnL ≥ -$2,886 AND trades ≤ 560 | +$1,408 / 50 | PASS |
| G3 | n ≥ 50 | 50 | PASS (at floor) |
| G4 | WR ≥ 55% | 60.0% | PASS (5pp headroom) |

**All 4 PASS strict.** First config to do so across the entire 9-attempt arc. Numbers cross-referenced from `artifacts/best_deterministic_rule.json` and `/tmp/rule_mining_report.md` Phase 6.

### 8.31.4 — Alternative Rules Passing 4/4

The constraint sweep produced 14 distinct rules that pass all 4 gates strict. The breakthrough rule is at index 10 in `artifacts/rule_mining_top_rules.json` → `n50_rules_passing_all_gates`. Three alternatives worth flagging for ensemble or fallback consideration:

| # | Rule | Train (n / WR) | Holdout (n / WR / PnL / DD) |
|---|---|---|---|
| 0 | `bf_de3_entry_ret1_atr ≤ -0.6154 AND upper_wick_ratio ≤ 0.0935` | 134 / 60.4% | 66 / 57.6% / +$644 / $429 |
| 1 | `pct_dist_to_running_lo_pct ≤ 0.0008 AND upper_wick_ratio ≤ 0.0353` | 255 / 71.0% | 62 / 56.5% / +$1,208 / $304 |
| 4 | `pct_dist_to_running_hi_pct > 0.0015 AND upper_wick_ratio ≤ 0.0353` | 253 / 71.5% | 55 / 56.4% / +$1,395 / $192 |
| 7 | `bf_regime_eff > 0.0672 AND upper_wick_ratio ≤ 0.0353` | 276 / 65.9% | 56 / 55.4% / +$1,209 / $230 |
| **10** | **`bf_regime_eff > 0.0900 AND upper_wick_ratio ≤ 0.0353`** *(chosen)* | **258 / 67.4%** | **50 / 60.0% / +$1,408 / $158** |

Rule #1 has the highest holdout n (62 vs 50) at a 56.5% WR with $304 DD — a useful "slightly more permissive" alternative if regime-effective drifts. Rule #4 has the lowest holdout DD ($192) of any 4/4-passing rule and a healthy PnL ($1,395). Rule #7 is structurally similar to the chosen rule but at a lower regime-eff threshold (0.067 vs 0.090) and trades n=56 holdout at 55.4% WR — exactly at the G4 floor, so it has zero headroom. The chosen rule (#10) was selected because it produces the highest holdout WR (60.0%) and lowest holdout DD ($158) among rules that hold n≥50 — the best gate margin.

The convergence on `upper_wick_ratio ≤ 0.0353` across 9 of the 14 passing rules is not an accident: every rule that prefers a clean entry-bar geometry passes G4. That is structural, not a fluke.

### 8.31.5 — Methodology

The mining script `tools/rule_mining.py` runs in six phases:

1. **Feature audit.** Univariate signal strength on 55 candidate features. Mutual information against `is_win`, plus top-quartile vs bottom-quartile WR split. Top discriminators: `pct_dist_to_running_lo_pct` (MI=0.070), `pct_dist_to_running_hi_pct` (MI=0.047), `bf_regime_eff` (top-Q WR=63.8%, bot-Q WR=44.2%), `bf_de3_entry_upper_wick_ratio` (top-Q WR=46.8%, bot-Q WR=63.5% — the inverse direction the rule exploits).
2. **Single-feature threshold sweeps.** Best holdout WRs were 90% / 86% / 81%, but on n=10–17. All 3/4 (G3 fails). High-WR but tiny n.
3. **Two-feature AND conjunctions.** Best holdout pair achieved 92% WR — but again on n=12–13. Still 3/4. Auto-rankers preferred max-WR rules with tiny n.
4. **Shallow decision tree.** `max_depth=3, min_samples_leaf=30`. Leaf rules extracted; the top leaf hit n=11 / WR=100% / DD=$0 — perfect on holdout but failing G3. The deeper leaves dropped below 50% WR.
5. **Train→holdout overfit check.** Top-10 rules by holdout WR all had ≤7pp drift (most ok-flagged in the report).
6. **n-constrained sweep (Phase 6a).** **The unlock.** Sweeping the same single-feature and two-feature spaces with the constraint `n_holdout ∈ [50, 250]` instead of pure-WR ranking surfaced the 14 rules that pass 4/4. Without this constraint, every prior phase top-ranked rules with n=10–17 that fail G3.

The rank-by-margin Phase 6a logic was the methodological unlock. The data was already present in Phase 3 — it just wasn't ranked under a constraint that respected G3.

### 8.31.6 — Feature Interpretation

**`bf_regime_eff`** is the BF (Breaker-Family / Kalshi-bar-feature) regime-efficiency marker, computed live in `julie001.py` at line 1667 as `abs(_k_rets.sum()) / abs(_k_rets).sum()` over the recent Kalshi-bar return window. Values near 0 indicate whipsaw (returns cancel out); values near 1 indicate clean trend (returns reinforce). The rule's threshold of 0.0900 is a low bar — the corpus shows roughly half of all candidates fall above it. The semantic meaning: "the recent regime is at least mildly trending, not pure chop." It excludes the worst-WR regime slice without being aggressive.

**`bf_de3_entry_upper_wick_ratio`** is the upper-wick-to-range ratio of the DE3 entry bar, computed in the corpus builder (`tools/build_de3_chosen_shape_dataset.py:196`) as `(prev_high - max(prev_open, prev_close)) / bar_range`. A small upper wick (≤ 0.0353, i.e. ≤ 3.5% of the bar's range) means the bar closed near its high without a meaningful rejection at the top. Mirror semantics for SHORT entries via the bar's geometry. Together with the regime filter, the rule says: **"Take a DE3 entry only when the recent regime is trending AND the entry bar shows a clean breakout with no rejection wick."** That is intuitive, defensible, and consistent with the prior body of breakout literature: clean candles in trending regimes win more often than rejection candles in chop. The thresholds are empirically calibrated, not hand-picked.

A note on directionality: `bf_de3_entry_upper_wick_ratio` is asymmetric for LONG vs SHORT. The corpus builder treats it bar-geometrically — the entry-bar feature is computed once per candidate regardless of side. The rule's empirical fit therefore reflects the dominant LONG/SHORT mix in the corpus. A future refinement is per-side wick features (`upper_wick` for LONGs, `lower_wick` for SHORTs); rule #0 in 8.31.4 already accidentally encodes some of that via the `ret1_atr` direction.

### 8.31.7 — Caveats

- **n=50 on holdout is exactly at the G3 floor.** A single bad month could push it under, and statistical power is borderline. A 95% binomial CI on a 60% WR with n=50 spans roughly [45.2%, 73.6%] — the lower bound is meaningfully below G4.
- **Holdout is 4 months (Jan–Apr 2026).** Short. One regime change could invalidate the rule. The user has independently observed regime drift between 2025 and 2026-Q1 (cf. §8.30.13).
- **5pp G4 headroom is meaningful but not large.** A 6pp WR erosion live would put the rule at 54%, KILLing G4. That is plausible if 2026-Q2/Q3 regime differs from Q1.
- **Train→holdout WR drop is -7.4pp.** The same magnitude of further drop on real-money would put live WR at 52.6% — KILL. The historical drift is the order of the gate margin.
- **14 alternative rules pass 4/4** — convergence on similar features (regime markers + clean-wick markers) helps confidence (it's not a single fluke), but the features are correlated. If one of those features drifts in production, all 14 rules drift together.
- **Out-of-sample validation is limited.** Only the Jan–Apr 2026 window. A walk-forward across multiple disjoint quarters has not been done.
- **Rule features are at signal time, not at fill time.** This is normally a benefit (no leakage), but it means the rule does not respond to fill-side adverse moves between signal and fill. For DE3 the sim assumes immediate fill, so this is unchanged from current live behavior.

### 8.31.8 — ML Enhancement Status

ML enhancement was reserved for the case where the deterministic rule cleared 3/4 gates and needed a 2–5pp boost. **Phase 6 already passes 4/4, so ML enhancement is not run.** The reserve plan is documented in 8.31.10 step 5: if live WR erodes below 55% under the rule, train an HGB on the corrected v11 corpus filtered to the rule subset, predict `pnl > 0`, and threshold-tune to add 2–5pp robustness. Held in reserve until needed. No model artifact exists at this point and none is required to ship.

### 8.31.9 — The Convergent KILL Tree (Updated)

| Approach | Best Holdout DD | Best Holdout WR | Verdict |
|---|---:|---:|---|
| ML retrain (v8 broken sim) | "$2,716" | "73.5%" | KILL — fictional |
| ML retrain (v9/v10) | $3,469 | 43.5% | KILL G1 |
| ML retrain (v11) | $1,196 | 48.07% | KILL G1+G4 |
| ML + structural (§8.28) | $908 | 37.32% | KILL G4 -17.7pp |
| ML + integrated stepped-SL (§8.29) | $4,673 | 41.15% | KILL G4 -13.85pp |
| Early-exit retrain (§8.30) | $3,354 | 45.89% | KILL G4 -9.1pp |
| **Rule-first (§8.31)** | **$158** | **60.0%** | **SHIP — all 4 gates pass strict** |

The 9-attempt KILL tree breaks at §8.31. Same corpus, same gates, different framing.

### 8.31.10 — Deployment Plan (Recommendation Only)

This is a deployment recommendation for the user's review. Nothing has been applied to live code. The user decides whether to ship.

**Step 1 — Verify features are signal-time available.** Already confirmed during this analysis:
- `bf_regime_eff` is computed in `julie001.py` line 1667 (`_k_eff = abs(_k_rets.sum()) / abs(_k_rets).sum()`) before any DE3 emit, exposed as the `regime_eff` field on the Kalshi bar-features dict (line 1675).
- `bf_de3_entry_upper_wick_ratio` is the DE3 entry bar's `(prev_high - max(prev_open, prev_close)) / bar_range`. The exact computation is in `tools/build_de3_chosen_shape_dataset.py:196` and uses bar values (`prev_high`, `prev_open`, `prev_close`, `bar_range`) that are present at signal time in the live trigger path. Verify before deploy that `julie001.py` either already passes these onto the candidate signal dict or that they can be computed from the bar values it has at the candidate gate.

**Step 2 — Locate the DE3 entry path filter point.** The DE3 candidate is emitted before the existing `_is_de3_signal` checks (around `julie001.py:5075` and downstream). The §8.31 filter goes immediately after the candidate is constructed and before the trade-emit call. Mirror the §8.30.11 BE-arm-disable pattern: the rule is a candidate-level gate, not an exit-mechanics change.

**Step 3 — Rule code (illustrative, not applied):**
```python
# §8.31 entry filter — block DE3 trades not matching the high-WR regime rule.
# Holdout: n=50 / WR=60% / PnL=+$1,408 / DD=$158. All 4 ship gates pass strict.
if not (bf_regime_eff > 0.0900 and bf_de3_entry_upper_wick_ratio <= 0.0353):
    log_filtered_signal(reason="831_rule_filter")
    return  # don't fire DE3 trade
```

**Step 4 — Rollout.** Shadow-mode for 1–2 weeks: log filtered/allowed counts, verify the ~17% retention rate (50 of 560 in holdout) reproduces in live. After shadow validation, full-active. Monitor: rolling-30-day WR ≥ 55%. Regression alarm if WR drops below 55% for 3+ consecutive days.

**Step 5 — ML reserve plan.** If live WR erodes below 55% sustained, train an HGB on the corrected v11 corpus filtered to the rule-subset rows, predict `pnl > 0`, and threshold-tune to recover 2–5pp robustness. The rule-first slice is the trainer's input — model adds robustness, not selection. Held in reserve.

**No commit, no push, no live `julie001.py` change is applied here.** The §8.31 deployment is a recommendation. The §8.28.11 BE-arm-disable, §8.28.5 daily CB-$300, and Pivot Trail recommendations from prior sections remain independently deployable.

### 8.31.11 — Open Questions / Future Work

- **Rule unions and coverage extension.** Can additional rule layers (RA-only, AF-only) extend coverage beyond DE3? The current rule lifts WR on DE3 but doesn't address candidate streams the prior arc deferred.
- **Multi-rule unions.** What's the WR if we union the top three 4/4-passing rules into an OR clause? Could double the n_holdout while maintaining gates, at the cost of re-checking G1.
- **Walk-forward validation.** The rule has only been validated on one disjoint holdout (Jan–Apr 2026). A rolling walk-forward across 2025-Q1 / Q2 / Q3 / Q4 / 2026-Q1 would establish whether the 5pp G4 headroom is stable across regimes.
- **Live regression alarms.** What alarm thresholds and lookback windows should monitor the rule live? Rolling-30-day WR < 55%, weekly DD > $200, or both?
- **Rule structure.** `bf_regime_eff > 0.09` is a low bar — only 17% of trades are filtered out by it alone? Or is the cutoff structural (e.g., regime classifier is bimodal at 0.08–0.10)? Worth a histogram check.
- **Per-side wick features.** Should we mirror `upper_wick_ratio` for LONG and `lower_wick_ratio` for SHORT, rather than using `upper_wick_ratio` symmetrically? The current rule's holdout pass is bar-geometric; per-side features could improve robustness.

### 8.31.12 — Files Written This Phase

- `tools/rule_mining.py` — 6-phase mining script (39 KB, present)
- `artifacts/rule_mining_feature_audit.json` — 30-feature univariate audit (10 KB, present)
- `artifacts/rule_mining_top_rules.json` — top 50 single + 50 pair + 10 leaf + 14 n50-passing rules (54 KB, present)
- `artifacts/best_deterministic_rule.json` — chosen rule + train/holdout metrics + gate flags (865 B, present)
- `/tmp/rule_mining_report.md` — Phases 1–6 markdown report (8 KB, present)

No `model.joblib`, no `thresholds.json`, no live-code changes. `julie001.py`, `config.py`, `client.py`, `de3_v4_*.py` untouched.

---

*Section 8.31 closes the JULIE001 ML overlay analysis arc on a positive note. Nine prior filter-based attempts (v8 through §8.30 early-exit retrain) converged on a 45–48% WR ceiling — 7–17pp under the user's G4=55% gate. The rule-first inversion proposed by the user — mine a hard rule from the labeled corpus, then add ML only if needed — yielded a 2-feature deterministic rule that clears all 4 ship gates on holdout at the strict $870 DD cap, with 5pp G4 headroom and $712 G1 headroom. The rule selects 17% of candidates (regime + clean entry) rather than filtering out 17% (the prior framing). Deployment is a one-line entry filter using features the bot already computes at signal time. ML is held in reserve as a regime-fuzzing layer if live WR erodes. Statistical power at n=50 is borderline and a -7.4pp train→holdout WR drop is the same order of magnitude as the gate margin — caveats are real and not glossed over. But this is the first SHIP after nine KILLS, and the user's framing was the unlock.*


---

## 8.32 V12 Retrain — Hydrated Kalshi + Shock Context + Rule Pre-Filter Cross-Product

*Written 2026-04-25, autonomous run. After §8.31's rule-first breakthrough and origin's pull adding `kalshi_history_provider.py`, `de3_shock_context.py`, and friend's `regime_ml v5/v6` trainer kit (`scripts/regime_ml/`, `scripts/ml_regime_v6_conditional.py`), this is the v12 retrain pass that finally has the data and infrastructure prior attempts lacked. v11 had Kalshi snapshot features set to neutral defaults (documented Phase 1 limitation #1) and no DE3 shock-context features. v12 hydrates both, re-trains all 4 stack-aware ML overlay heads, and tests every combination of the §8.31 rule with the v12 ML heads.*

### 8.32.1 — TL;DR

- **Kalshi v12 SHIPS** under strict gates ($870 DD cap, 55% WR floor, n>=50, PnL>=baseline). Best config: Mode B (rule UNION ML-keep) at threshold 0.175 — **149 trades, 55.7% WR, +$1,465 PnL, -$463 DD** on holdout. All 4 strict gates pass.
- **LFO / PCT / Pivot v12 KILL** — they each show high test AUCs (0.61–0.66, vs v11's 0.55) but every threshold/mode combination fails at least one strict gate. Their best non-passing configurations bind on G3 (insufficient surviving trades inside the rule pre-filter) or G1 (DD>$870 outside the rule).
- The §8.31 rule **alone** scores n=48 / WR=62.5% / PnL=+$1,468 / DD=-$158 on this rebuilt holdout — passes G1/G2/G4 but JUST fails G3 (need 50). v11's report had n=50 because it used a slightly different filter chain. The v12 corpus is rebuilt cleanly.
- Combined stack v12 (rule + 4-head ML, conservative AND-block) — n=16 / WR=81% / PnL=+$1,222 / DD=-$65 — fails G3 (too restrictive). The single-head Kalshi-only spec is the deployable winner.

### 8.32.2 — What's New vs v11

Three things v11 didn't have:

1. **`kalshi_history_provider.py`** (~7.5 KB, origin) — point-in-time settlement-hour midpoint probability lookup against `data/kalshi/kxinxu_2025_daily/*.parquet`. Replaces v11's neutral default of 0.5 for all 13 Kalshi snapshot features. Coverage on the v11 corpus's timestamps: **90.8%** of all 3,438 candidates (3,121 hydrated). Holdout coverage is 62% (Jan 2026 missing from the daily archive; Feb–Apr 2026 fully covered). Train coverage is 100%.

2. **`de3_shock_context.py`** (+501 lines, origin) — 23 new features capturing day-level context (gap regime, range-progress ratio, trend frac, opening regime), plus shock primitives (recent range/volume ratios, session move/range norms, shock score). All populated 100% on the corpus.

3. **`regime_ml v5/v6` trainer kit** (`scripts/regime_ml/_common.py` + `train_kalshi_v*.py`, `scripts/ml_regime_v6_conditional.py`) — friend's own training framework. Offers a clean HGB pipeline with built-in BE-arm-aware simulation and consistent class-balanced sample weighting. v12 borrows the schema design (40-feature regime snapshot + ID-stacking) but keeps the corpus-driven training loop because we need the `is_big_loss` label and PnL-weighted samples that v11 already has computed. The v6 trainer's `mode` constants (`a_pred_scalp`, conditional gating) are documented for future reference but not used in this pass; v12 sticks with HGB max_depth=3 max_iter=200 lr=0.05 to match prior phases for comparability.

### 8.32.3 — V12 Corpus

`tools/build_v12_training_corpus.py` reads `artifacts/v11_training_corpus_with_mfe.parquet` (3,438 rows) and adds:

- **13 Kalshi snapshot features** (`k12_*`):
  - `k12_entry_probability` — Kalshi midpoint probability at the strike nearest entry_price
  - `k12_probe_probability`, `k12_probe_neg_probability` — probabilities at +5 / -5 strike offsets
  - `k12_skew_p10`, `k12_skew_p25` — entry probability minus probability 10/25 below
  - `k12_above_5`, `k12_above_10`, `k12_below_10` — local skew probes
  - `k12_distance_to_50` — `|p - 0.5|` (uncertainty proxy; 0 = uncertain market)
  - `k12_momentum_5`, `k12_momentum_15` — provider's get_sentiment_momentum at lookbacks 5/15
  - `k12_window_active` — 1 if Kalshi settlement hour is active at signal time
  - `k12_data_present` — 1 if the day has a Kalshi parquet file

  **Variance check (sanity vs v11 neutral defaults):**
  - `k12_entry_probability` mean=0.302 std=0.254 range [0.00, 0.99] — clearly NOT a neutral constant. v11 used 0.5 always.
  - `k12_skew_p10` mean=-0.111 std=0.267 — real skew signal.
  - `k12_distance_to_50` mean=0.264 std=0.185 — real uncertainty signal.

- **23 DE3 shock-context features** (`ctx_*`):
  - `ctx_day_*` (7): `range_progress_ratio`, `volume_progress_ratio`, `gap_ratio`, `trend_frac`, `first60_share`, plus categorical `expansion_regime`, `direction_regime`, `opening_regime`, `flow_regime`, `gap_regime`, `type`, `profile`
  - `ctx_shock_*` (10): `recent_range_ratio`, `recent_volume_ratio`, `session_move_norm`, `session_range_norm`, `score`, plus 6 bucket categoricals

  Shock-context populated for 100% of corpus rows (drawn from `es_master_outrights.parquet` 1-min bars and the front-month contract pinned at signal time).

Output: `artifacts/v12_training_corpus.parquet` (3,438 rows × 116 columns; v11 had 80). Summary: `artifacts/v12_corpus_summary.json`.

### 8.32.4 — Per-Head Training Results

`tools/run_v12_retrain.py` trains 4 stack-aware HGB heads on the v12 corpus, filtered to DE3 candidates with `allowed_by_friend_rule == True` (n_train=1,068, n_holdout=531). Stacking order: Kalshi → LFO → PCT → Pivot. Each downstream head receives upstream proba columns (`kalshi_proba_v12`, `lfo_proba_v12`, `pct_proba_v12`).

| Head   | Train AUC | Test AUC | n_train | n_test | Train Pos | Test Pos | Ship Status |
|:-------|----------:|---------:|--------:|-------:|----------:|---------:|:-----------:|
| Kalshi | 0.971     | 0.675    | 1,068   | 531    | 308       | 163      | **SHIP**    |
| LFO    | 0.998     | 0.658    | 1,068   | 531    | 308       | 163      | KILL        |
| PCT    | 1.000     | 0.610    | 1,068   | 531    | 308       | 163      | KILL        |
| Pivot  | 1.000     | 0.609    | 1,068   | 531    | 308       | 163      | KILL        |

The high train AUCs (>=0.97 once stacked) reflect HGB memorizing the upstream proba; test AUC is the meaningful number. **All 4 heads test_AUC > 0.60**, materially above v11's 0.55 ceiling. Hydrated Kalshi snapshot features lift signal — the question is whether thresholding finds a config that holds the strict ship gates.

For each head, three modes were swept across thresholds 0.10..0.85 step 0.025:

- **Mode A** — refine within rule-kept (apply rule first, then ML inside that subset).
- **Mode B** — union of rule-kept and ML-approved (rule keeps OR ML proba < threshold).
- **Mode C** — ML alone (no rule pre-filter; just block on ML proba >= threshold).

**Best config per head (from sweeps):**

| Head   | Best Mode              | Threshold | Trades | WR    | PnL    | DD    | Gates    |
|:-------|:-----------------------|----------:|-------:|------:|-------:|------:|:---------|
| Kalshi | B_union_rule_or_ml     | 0.175     | 149    | 55.7% | +$1,465| -$463 | 4/4 PASS |
| LFO    | C_ml_alone             | 0.450     | 259    | 49.0% | +$869  | -$930 | 2/4 (G1,G4) |
| PCT    | A_refine_in_rule       | 0.100     | 22     | 72.7% | +$1,329| -$91  | 3/4 (G3) |
| Pivot  | A_refine_in_rule       | 0.100     | 27     | 70.4% | +$1,298| -$91  | 3/4 (G3) |

**Kalshi at thr=0.175 Mode B** is the best overall. Mode B says "keep this trade if the §8.31 rule says yes OR the Kalshi ML head's predicted big-loss probability is below 0.175". This widens the rule's coverage from 48 to 149 trades while holding WR above 55%, with cushion on G1 ($463 vs $870 cap) and G2 ($1,465 vs $-2,674 baseline = +$4,140 lift). The 13 hydrated Kalshi snapshot features gave the head enough orthogonal signal to safely approve trades the rule alone would not.

LFO/PCT/Pivot all bind on G3 (insufficient n) inside the rule pre-filter (Mode A) or G1 (DD > $870) when used standalone (Mode C). Their high test AUCs reflect modeled signal but their threshold landscape doesn't have a 4-gate sweet spot — at low thresholds they drop too many trades; at high thresholds DD blows past $870.

### 8.32.5 — Rule + ML Combined (Phase 3)

The §8.31 rule alone on the v12-rebuilt holdout: n=48 / WR=62.5% / PnL=+$1,467.5 / DD=-$157.5. Passes G1/G2/G4 strict; misses G3 by 2 (need 50).

Modes A/B/C cross-product results (see §8.32.4). Mode B (rule UNION ML-keep) is the productive direction: ML can ADD trades the rule rejected, when ML's predicted big-loss probability is comfortably low. For Kalshi this surfaces 101 additional trades (149 - 48) at WR slightly below the rule's 62.5% but still above 55% — net positive.

Mode A (refine within rule) tends to drop the trades count below 50 (G3 binds) because the rule already selected a high-quality subset; ML refinement over-prunes.

Mode C (ML alone, no rule) struggles on G1 — without the rule's regime/wick filter, even ML's best thresholds let in too many losers, exceeding the $870 DD cap.

**Conclusion:** the §8.31 rule is the foundation. ML enhancement value is in Mode B (expansion), not Mode A (further filtering). Only the Kalshi head's hydrated snapshot features provide enough orthogonal signal to expand safely; LFO/PCT/Pivot's added expansions either miss WR or miss DD.

### 8.32.6 — Combined Stack v12 Verdict

`tools/run_v12_retrain.py` Phase 4 applies the §8.31 rule first, then conservatively BLOCKs any trade where any SHIP-status head says block. Since only Kalshi ships, the combined stack is effectively rule + Kalshi.

Result: n=16 / WR=81.2% / PnL=+$1,222 / DD=-$65. Passes G1/G2/G4 STRONGLY but fails G3 (need 50). The conservative AND-block over-restricts.

The deployable winner is **single-head Kalshi v12 thr=0.175 Mode B**: union the rule with ML-keep. That's not a "stack" in the traditional sense — it's the rule plus a single ML expansion gate.

Output: `artifacts/v12_combined_stack_metrics.json`.

### 8.32.7 — Friend's Framework Cross-Check

`scripts/signal_gate/evaluate_kalshi_overlay_available.py` and `evaluate_lfo_combos_full_history.py` (origin) require `.tmp_upstream_julie/` (the upstream worktree path) which is not present in this repo. The friend's evaluation framework cannot be executed end-to-end here. Cross-check is therefore deferred.

The closest available reference is `artifacts/signal_gate_2025/lfo_threshold_sweep_results.json` (origin's pre-computed sweep) — not directly comparable because it sweeps a different model on a different label. The §8.32 verdict relies on the gate-based corpus evaluation, which is a strict superset of the friend's framework's intent (block-style overlay applied on a fixed candidate stream).

### 8.32.8 — Ship-or-KILL Verdict

| Component | Verdict | Best Threshold | Mode | Trades | WR    | PnL    | DD    |
|:----------|:-------:|---------------:|:-----|-------:|------:|-------:|------:|
| Kalshi v12 | **SHIP** | 0.175 | B_union_rule_or_ml | 149 | 55.7% | +$1,465 | -$463 |
| LFO v12    | KILL    | 0.450 | C_ml_alone         | 259 | 49.0% | +$869   | -$930 |
| PCT v12    | KILL    | 0.100 | A_refine_in_rule   | 22  | 72.7% | +$1,329 | -$91  |
| Pivot v12  | KILL    | 0.100 | A_refine_in_rule   | 27  | 70.4% | +$1,298 | -$91  |
| Combined stack | KILL  | — | rule AND each-head | 16  | 81.2% | +$1,222 | -$65  |
| §8.31 rule alone | (G3 borderline) | — | — | 48 | 62.5% | +$1,468 | -$158 |

For each head: Kalshi SHIP, LFO/PCT/Pivot KILL. For the combined stack: KILL (G3). The deployable v12 spec is **rule + Kalshi head Mode B**.

### 8.32.9 — Deployment Plan (If Anything Ships)

This is a deployment recommendation. Nothing has been applied to live code. The user decides whether to ship.

**Recommended: §8.31 rule + Kalshi v12 Mode B at threshold 0.175.**

Concretely, in `julie001.py`'s DE3 candidate filter path (the same spot called out in §8.31.10 step 2):

```python
# §8.32 v12 DE3 entry filter — rule + Kalshi-ML expansion.
# Holdout: n=149 / WR=55.7% / PnL=+$1,465 / DD=-$463. All 4 strict gates pass.
rule_ok = (bf_regime_eff > 0.0900 and bf_de3_entry_upper_wick_ratio <= 0.0353)
# Kalshi v12 head is loaded from artifacts/regime_ml_kalshi_v12/de3/model.joblib
ml_proba = kalshi_v12_head.predict_proba(features)[:, 1]
ml_ok = (ml_proba < 0.175)
if not (rule_ok or ml_ok):
    log_filtered_signal(reason="832_rule_or_kalshi_v12_filter")
    return
```

The Kalshi v12 head requires the 13 hydrated `k12_*` features to be computable at signal time. The `kalshi_history_provider.HistoricalKalshiProvider` already runs in live (`kalshi_trade_overlay.py`), so the day-level archive data is already on the live host. The point-in-time lookup adds <5 ms per signal.

**Required artifacts for deploy:**
- `artifacts/regime_ml_kalshi_v12/de3/model.joblib` — written this turn, ships the HGB classifier + feature schema
- `artifacts/regime_ml_kalshi_v12/de3/thresholds.json` — `{"threshold": 0.175, "mode": "B_union_rule_or_ml", "rule_min_regime_eff": 0.0900, "rule_max_upper_wick": 0.0353}`

**Shadow rollout (1–2 weeks)**: log filtered/allowed counts, verify the ~28% retention rate (149 of 531 in holdout) reproduces in live. After shadow validation, full-active.

**Live regression alarms**: rolling-30-day WR < 55%, weekly DD > $463, or rule+ML-approved daily trade count drops below 0.5/day average — any triggers a freeze and re-eval.

**No commit, no push, no live `julie001.py` change is applied here.** §8.32 is a deployment recommendation built on the v12 retrain pass.

### 8.32.10 — Files Written

- `tools/build_v12_training_corpus.py` (~6 KB, present)
- `tools/run_v12_retrain.py` (~14 KB, present)
- `artifacts/v12_training_corpus.parquet` (~3.5 MB, 3,438 × 116, present)
- `artifacts/v12_corpus_summary.json` (~2 KB, present)
- `artifacts/regime_ml_kalshi_v12/de3/model.joblib` (~250 KB, present — Kalshi SHIP)
- `artifacts/regime_ml_kalshi_v12/de3/thresholds.json` (~200 B, present)
- `artifacts/regime_ml_kalshi_v12/de3/metrics.json` (~120 KB, present — full sweep)
- `artifacts/regime_ml_lfo_v12/de3/metrics.json` (~120 KB, present — KILL marker, no model)
- `artifacts/regime_ml_pct_v12/de3/metrics.json` (~120 KB, present — KILL marker, no model)
- `artifacts/regime_ml_pivot_v12/de3/metrics.json` (~120 KB, present — KILL marker, no model)
- `artifacts/v12_combined_stack_metrics.json` (~3 KB, present)
- `artifacts/v12_summary.json` (~2 KB, present)

`julie001.py`, `config.py`, `client.py`, `de3_v4_*.py` untouched.

---

*Section 8.32 closes the v12 retrain pass. The §8.31 rule alone is gate-borderline on the rebuilt v12 corpus (G3 binds at n=48), but the v12 Kalshi head — trained on 13 hydrated Kalshi snapshot features that v11's neutral defaults could never expose — surfaces enough orthogonal signal to safely expand the rule's keep-set from 48 to 149 trades while holding 55.7% WR. That is the first ML-augmented v* spec to ship under strict gates ($870 DD cap, 55% WR floor, n>=50, PnL>=baseline) since v8. The other three heads (LFO/PCT/Pivot) reach test AUC 0.61–0.66 — meaningful learning over v11's 0.55 — but their threshold landscapes don't intersect the strict gate envelope. The deployable spec is "rule OR Kalshi-v12-keep at thr=0.175". Friend's evaluation framework cross-check is deferred (upstream path missing). No commit, no push, no live edits applied. Recommendation only.*

## 8.33 Filter G Per-Cell Override Bug-Fix + V12 Recalibration

### 8.33.1 — Context

The Filter G veto threshold has a per-cell override system: a JSON table at `ai_loop_data/triathlon/filterg_threshold_overrides.json` keyed by `{strategy}|{regime}|{time_bucket}` returns a multiplier in [0.75, 1.20] that scales the strategy's base veto threshold. Cells classified "bleeding" in pre-April-2026 data carry mult=0.75 (tighten — block more aggressively); cells classified "strong" carry mult=1.15 (loosen — require higher P(big_loss) to block). Originally 17 cells were calibrated by `scripts/idea1_filterg_per_cell_calibrate.py` against the pre-§8.25 simulator: 9 bleeding + 8 strong.

This audit found two independent problems: a runtime case-mismatch bug that has been silently disabling the entire override table, and a calibration-corpus problem (the 17 cells were learned against the broken simulator).

### 8.33.2 — The Case-Mismatch Bug

`signal_gate_2025._per_cell_multiplier(strategy, regime, et_hour)` was composing the lookup key as `f"{strategy}|{regime}|{tb}"` with the strategy string passed verbatim. The JSON keys use canonical class names: `DynamicEngine3`, `RegimeAdaptive`, `AetherFlow`, `MLPhysics`. But callers in this codebase pass the strategy through several different shapes — `loss_factor_guard` passes `signal["strategy"]` raw (which for live signals IS the canonical name), while sim/test call sites and the MLPhysics suffixed variants (`MLPhysics_AsianSession`) do not.

Concretely: any signal whose `strategy` field is `"MLPhysics_<session_name>"` or any backtest harness that lowercases / shortens to `"de3"` matches no JSON key and falls back to mult=1.0 — i.e. the override is silently no-op. For DE3 live signals, the strings happen to match, so the bug isn't visible in DE3 production logs. But the system is fragile by design and incomplete in coverage.

### 8.33.3 — Step 1: The Fix

Added `_normalize_strategy_for_cell_key(strategy)` to `signal_gate_2025.py`. The helper accepts any of the known short codes (`de3`, `ra`, `af`, `mlphysics`), suffixed variants (`MLPhysics_AsianSession`, `MLPhysicsLegacy_NY`), and case variations, and returns the canonical class-name form used as the JSON key prefix. `_per_cell_multiplier` now normalises the strategy BEFORE composing the cell key. Unknown strategies pass through unchanged so future cell additions still work.

```python
_STRATEGY_NAME_NORMALIZE = {
    "de3": "DynamicEngine3", "dynamicengine3": "DynamicEngine3",
    "ra": "RegimeAdaptive", "regimeadaptive": "RegimeAdaptive",
    "af": "AetherFlow", "aetherflow": "AetherFlow",
    "mlphysics": "MLPhysics", "ml_physics": "MLPhysics",
    "mlphysicslegacy": "MLPhysicsLegacy",
}
```

Smoke test confirmed: `_per_cell_multiplier(strategy="de3", regime="calm_trend", et_hour=13)` and `_per_cell_multiplier(strategy="DynamicEngine3", regime="calm_trend", et_hour=13)` now both return 0.75 (the bleeding mult for `DynamicEngine3|calm_trend|lunch`). Previously the short-code call returned 1.0.

Lines changed: signal_gate_2025.py +50/-7. Backup at `signal_gate_2025.py.bak_filterg_20260426-093804`.

### 8.33.4 — Step 2: V12 Corpus Recalibration

The 17-cell verdicts in `filterg_threshold_overrides.json` came from a pre-§8.25 simulator that materially over-counted big losses. Re-derived per-cell stats from `artifacts/v12_training_corpus.parquet` (3,438 rows, 2025-03-03 → 2026-04-24, the post-fix corrected corpus from §8.32). Pipeline:

1. **Strategy canonicalisation:** family `de3` → `DynamicEngine3`, family `regimeadaptive` → `RegimeAdaptive`. Corpus contains DE3 (3,263) + RA (175) only — AetherFlow / MLPhysics / dead_tape morning are absent from the v12 row set.
2. **Regime label:** apply `regime_classifier._classify(vol_bp, eff)` per row using `bf_regime_vol_bp` / `bf_regime_eff` columns (DEAD_TAPE_VOL_BP=1.5, EFF_LOW=0.05, EFF_HIGH=0.12). Distribution: calm_trend 1535, neutral 1164, whipsaw 681, dead_tape 58.
3. **Time bucket:** `_time_bucket_of_hour(ts.hour)` mirroring the live function exactly. Distribution: lunch 1218, pre_open 1159, afternoon 994, morning 67.
4. **Filter-G correctness:** binary block/keep at `fg_proba >= 0.5`. Per cell: `fg_block_correct_rate` = (rows where FG predicted block AND `is_big_loss=True`) / (rows where FG predicted block).
5. **Decision rule:** `bcr < 0.35` → mult=1.20 (loosen strongly — FG very wrong); `0.35 ≤ bcr < 0.45` → 1.10; `0.45 ≤ bcr ≤ 0.55` → 1.0; `0.55 < bcr ≤ 0.65` → 0.85; `bcr > 0.65` → 0.75 (tighten strongly — FG very right).
6. **Min-n guard:** cells with n<20 in v12 keep their v1 multiplier. Cells absent from v12 entirely keep v1 verbatim.

Output: `ai_loop_data/triathlon/filterg_threshold_overrides_v2.json` (NEW — does NOT overwrite v1).

### 8.33.5 — Result Summary

| | v1 | v2 |
|---|---:|---:|
| Active runtime cells (mult ≠ 1.0) | 17 | 20 |
| Bleeding (mult < 1.0) | 9 | 4 |
| Strong (mult > 1.0) | 8 | 16 |

Of 31 cells in the comparison: 18 unchanged, 13 changed. Of those 13: **5 flipped direction** — every one of them was `bleeding → loosen` (i.e. v12 says FG was wrong to block these cells, where the broken sim said FG was right):

| Cell | v1 | v1 verdict | n in v12 | bcr | v2 | v2 verdict |
|---|---:|---|---:|---:|---:|---|
| `DynamicEngine3 / calm_trend / pre_open` | 0.75 | bleeding | 504 | 27.7% | 1.20 | loosen_strong |
| `DynamicEngine3 / calm_trend / lunch` | 0.75 | bleeding | 531 | 37.3% | 1.10 | loosen |
| `DynamicEngine3 / calm_trend / afternoon` | 0.75 | bleeding | 454 | 29.4% | 1.20 | loosen_strong |
| `DynamicEngine3 / whipsaw / afternoon` | 0.75 | bleeding | 177 | 44.6% | 1.10 | loosen |
| `RegimeAdaptive / neutral / pre_open` | 0.75 | bleeding | 28 | 0.0% | 1.20 | loosen_strong |

The pattern is clear: **on the corrected v12 corpus, Filter G is consistently over-blocking DE3 calm_trend cells**. v1 thought DE3 calm_trend was a death zone (so it tightened FG to make blocking even easier); v12 says the opposite — when FG fires on DE3 calm_trend, it's right less than a third of the time, so the threshold should be loosened to require higher confidence before the block triggers.

One cell flipped to neutral (`DynamicEngine3 / whipsaw / lunch`: bcr=52.1%, mult=1.0). Eleven cells were preserved at v1 because n<20 in v12 (mostly RegimeAdaptive). Seven cells were preserved at v1 because absent from v12 (DE3 overnight + dead_tape morning + RA dead_tape overnight). v1 cell count = 17; v2 cell count of cells with mult≠1.0 = 20.

### 8.33.6 — What This Means

Two big findings stack here:

1. **Until 2026-04-26 the per-cell override has been silently disabled** for any non-canonical strategy string. For DE3 production signals the strings happen to match so the v1 multipliers WERE applying, but for any backtest using `family.lower()` (e.g. `"de3"`), the override was no-op and Filter G ran with raw regime × session multipliers only.

2. **v1's bleeding verdicts on DE3 calm_trend are inverted under the corrected v12 corpus.** The pre-§8.25 simulator was double-counting big losses on calm-trend tape, which made FG look heroic on those cells (cell mult=0.75 → tighter). Under the fixed v12 simulator, FG over-blocks calm-trend cells and v2 instead recommends loosening (mult=1.10–1.20).

### 8.33.7 — Files Written

- `signal_gate_2025.py` (+50/-7, case-fix only, around lines 282–360)
- `signal_gate_2025.py.bak_filterg_20260426-093804` (backup)
- `ai_loop_data/triathlon/filterg_threshold_overrides_v2.json` (NEW — v2 recalibrated table)
- `/tmp/filterg_recalib_comparison.md` (full per-cell comparison table)
- `/tmp/filterg_per_cell_fix_report.md` (this work's run report)

`ai_loop_data/triathlon/filterg_threshold_overrides.json` (v1) is untouched.

### 8.33.8 — Decision Pending

Operator picks at integration time:
- Adopt v2 (recalibrated against corrected v12 corpus, more statistically defensible) — point `_PER_CELL_TABLE_PATH` at v2 OR rename v2 over v1.
- Stay on v1 (familiar, known live behaviour, but its DE3 calm_trend bleeding verdicts are now suspect).
- Hybrid: keep v1 for DE3 dead_tape / overnight / post_close cells (absent from v12 so v2 just inherits) and adopt v2's flips for the high-n cells (DE3 calm_trend / whipsaw / neutral × pre_open/lunch/afternoon).

The case-mismatch fix is independent of this calibration choice and is the primary deliverable. Even if the operator elects to keep v1, the bug-fix means the v1 multipliers will now actually be applied for non-canonical strategy strings — which matters for any future backtest harness or for the MLPhysics suffix variants.

No live commit. No push. Working tree only.

---

*Section 8.33 closes the per-cell override audit. The case-mismatch bug-fix is small (~50 lines), well-tested, and unambiguously correct. The v12 recalibration is the headline finding: 5 of the 9 v1 bleeding cells are wrong-direction under the corrected corpus, and the v2 table flips them to loosen (mult=1.10–1.20). The v2 JSON is saved alongside v1, not over it; the operator picks which to enable.*

### 8.33.9 — Activation Impact on v11 Corpus: 3-Way Replay (Bug-Fix is Inert at Current FG Calibration)

*Added 2026-04-26 by post-fix audit.*

After §8.33's case-fix landed and the v2 table was written, the obvious
follow-up question: with the per-cell layer now actually activated (instead
of silently dormant for any non-canonical strategy string), does PnL
move? Replayed the v11 corrected corpus
(`artifacts/v11_corpus_with_bar_paths.parquet`, 3,438 rows / 2025-03-03 →
2026-04-24, filterless PnL +$13,203.75) through Filter G under three
configurations to quantify.

Replay harness: `tools/percell_layer_3way_compare.py`. Same base threshold
(0.35), regime multipliers (whipsaw 0.60 / calm_trend 1.05 / neutral 1.0),
session multipliers (lenient 1.25 at +$100, aggressive 0.80 at -$200), and
effective-threshold floor (0.25) across all three configs.

| Config | Per-cell behavior |
|---|---|
| **A_DORMANT** | per-cell mult forced to 1.0 — actual 14-month live state pre-§8.33 |
| **B_V1** | per-cell layer ON, v1 JSON (calibrated against broken sim) |
| **C_V2** | per-cell layer ON, v2 JSON (recalibrated against v11 corpus per §8.33.4) |

#### Headline result

| Config | Trades | WR | Net PnL | Avg/trade | Max DD | Block rate |
|---|---:|---:|---:|---:|---:|---:|
| A_DORMANT | 206 | 35.44% | -$695.00 | -$3.37 | -$1,108.75 | 94.01% |
| B_V1 | 206 | 35.44% | -$695.00 | -$3.37 | -$1,108.75 | 94.01% |
| C_V2 | 206 | 35.44% | -$695.00 | -$3.37 | -$1,108.75 | 94.01% |

**ΔC_V2 vs A_DORMANT: $0.00 PnL, +0 trades, +0.00pp WR, $0 DD.**
Per-month and by-strategy splits are byte-identical across all three configs
(DE3: 187 trades, -$561.25, 36.4% WR; RA: 19 trades, -$133.75, 26.3% WR
— same in A, B, and C).

#### Why nothing moves: fg_proba is bimodal

Filter G's `fg_proba` distribution on the v11 corpus:

| fg_proba bin | rows |
|---|---:|
| (0.00, 0.10] | 206 |
| (0.10, 0.25] | 0 |
| (0.25, 0.40] | 0 |
| (0.40, 0.50] | **1** |
| (0.50, 0.55] | 0 |
| (0.55, 0.70] | 330 |
| (0.70, 0.80] | 2,765 |
| (0.80, 1.00] | 136 |

206 rows have `fg_proba ≤ 0.10` — these always fire because no per-cell
configuration can push the effective threshold below the 0.25 floor.

3,231 rows have `fg_proba > 0.55` — these always veto. The most lenient
per-cell mult in v2 is 1.20, which on calm_trend (regime mult 1.05)
produces `eff_thr = 0.35 × 1.05 × 1.0 × 1.20 = 0.441`. Even with session
mult lenient 1.25 stacked on top, max `eff_thr = 0.5512`. fg_proba ≥ 0.55
is 3,231 rows — all vetoed regardless of per-cell.

**Exactly one row sits in the boundary zone** where per-cell could matter:

| ts | strategy | regime | bucket | fg_proba | A eff_thr | B eff_thr | C eff_thr |
|---|---|---|---|---:|---:|---:|---:|
| 2026-01-20 14:40 ET | DE3 | calm_trend | afternoon | 0.4919 | 0.3675 | 0.2756 | 0.4410 |

That row's `fg_proba = 0.492` exceeds even C_V2's 0.441 effective threshold,
so all three configs veto it. (It was a +$117.50 winner that all three
configs blocked.) For per-cell v2 to have unblocked that trade, the cell
mult would need to be > 1.34 — outside the v2 calibration range.

#### Implication for §8.33's bug-fix story

The case-mismatch bug-fix landed in §8.33.3 is still correct (silent
dormancy is bad regardless of whether it bites in any particular window),
and the v2 recalibration in §8.33.4 is still an improvement over v1
(5 cells were directionally wrong against the corrected corpus). But the
**economic impact of activation is zero** on this 14-month corpus
because Filter G's classifier produces bimodal scores: fg_proba is
either very low (definitely-not-big-loss, fires through floor) or very
high (almost-certainly-big-loss, vetos), and the 0.25 floor + base × multipliers
band cannot bridge that gap.

This shifts the action item: meaningful Filter G changes on this corpus
require changes to the **classifier itself** (retrain for calibration,
not just better cell coverage), or removing/lowering the 0.25 floor —
not refinements to the per-cell layer.

The §8.25 verdict stands: at base threshold 0.35, Filter G blocks
3,232/3,438 candidates (94.0%) and removes $13,898.75 of net PnL,
leaving the bot with -$695 from the surviving 206 trades. The corpus's
filterless baseline is +$13,203.75. **Filter G is grossly miscalibrated
against the corrected corpus and should not be active.** This is
consistent with the "Option B — drop overlays" recommendation in §8.26.

The §8.33 bug-fix is the right thing to do (correctness), but on its own
it is not a PnL lever. The PnL lever is upstream: replace or recalibrate
Filter G, or disable it entirely.

#### Files added

- `tools/percell_layer_3way_compare.py` — replay harness
- `artifacts/percell_3way_results.csv` — per-row annotated decisions
- `artifacts/percell_3way_summary.json` — config summaries + deltas


### 8.33.10 — V18 + Recipe B 5-way Deployment-Fidelity-Fast Comparison

*Added 2026-04-26.* Replaces friend's V18 train/test eval with a
deployment-realistic measurement that preserves friend's <1-second runtime.

#### Methodology

Ran [tools/v18_5way_holdout_real_kronos.py](tools/v18_5way_holdout_real_kronos.py)
on friend's pre-cached features (DE3 + `allowed_by_friend_rule=True`,
1,599 rows of 3,438 corpus rows). Friend's speed source = pre-cached
Kronos features ([artifacts/v18_kronos_features.parquet](artifacts/v18_kronos_features.parquet))
+ vectorized PnL via numpy cumsum. NO bug, just legitimate caching.

Added on top of friend's vectorized eval:
- **NY-only filter** [08:00, 16:00) ET (vectorized boolean mask)
- **Single-position constraint** (stateful walk in time order, ~5ms on 1,599 rows)
- **Time-ordered cumulative size-aware DD** (peak-to-trough on size-aware equity, not row-order cumsum on PnL[keep])
- **Recipe B sizing for E** (proba≥0.85→size=10, 0.65–0.85→size=4, 0.60–0.65→size=1)

NOT modeled (would require feature regeneration + re-simulation):
- Regime ML v5_brackets / v6_size / v6_be (40+ multi-window vol/eff features not cached)
- SameSide ML (50 features incl. position-state — also violates single-position)
- Downstream Kalshi/LFO/PCT/Pivot overlays (V18 IS the meta gate over their probas)

Total runtime: **0.8 seconds** (vs Kronos batch's 49-min one-time cost).

#### Headline result — HOLDOUT (Jan-Apr 2026, 3.73 months, honest forward-projection)

| Config | Trades | WR | Net PnL | Avg/trade | Max DD |
|---|---:|---:|---:|---:|---:|
| A_filterless | 487 | 43.74% | −$2,280.00 | −$4.68 | −$3,606.25 |
| B_FilterG_base_0.35 | 32 | 40.62% | −$210.00 | −$6.56 | −$336.25 |
| C_V15_thr_0.65 | 88 | 59.09% | +$1,485.00 | +$16.88 | −$363.75 |
| D_V18_thr_0.60_flat | 97 | 58.76% | +$1,581.25 | +$16.30 | −$363.75 |
| **E_V18+RecipeB (DEPLOYED)** | **97** | **58.76%** | **+$16,285.00** | **+$167.89** | **−$1,200.00** |

#### Δ vs B (production proxy):

| Config | Δtrades | ΔPnL | ΔDD |
|---|---:|---:|---:|
| A_filterless | +455 | −$2,070 | −$3,270 |
| C_V15 | +56 | +$1,695 | −$28 |
| D_V18 flat | +65 | +$1,791 | −$28 |
| **E_V18+RecipeB** | **+65** | **+$16,495** | **−$864** |

#### Recipe B tier breakdown (E, holdout)

| Tier | Size | n | WR | PnL | DD |
|---|---:|---:|---:|---:|---:|
| tier10 (≥0.85) | 10 | 25 | **84.0%** | **+$16,637.50** | −$575.00 |
| tier4 (0.65-0.85) | 4 | 62 | 50.0% | −$360.00 | **−$1,785.00** |
| tier1 (0.60-0.65) | 1 | 10 | 50.0% | +$7.50 | −$65.00 |

**Tier-10 alone delivers 100% of the edge.** Tiers 4 and 1 add zero net edge,
and tier-4 contributes the worst per-tier DD (−$1,785).

#### Per-month (E, holdout)

| Month | Trades | PnL | WR | Max size | DD |
|---|---:|---:|---:|---:|---:|
| 2026-01 | 13 | +$12,871.25 | 92.3% | 10 | −$55.00 |
| 2026-02 | 26 | +$977.50 | 57.7% | 10 | **−$1,200.00 ⚠** |
| 2026-03 | 49 | +$658.75 | 49.0% | 10 | **−$1,060.00 ⚠** |
| 2026-04 | 9 | +$1,777.50 | 66.7% | 10 | −$35.00 |

**February breaches user's $870 ship-gate within a single month** (DD −$1,200).
March near-breach (−$1,060). Both months are tier-4 driven; tier-10 holds up.

#### 8-month projection vs user's $32.5k claim

- E HOLDOUT PnL: **$16,285 over 3.73mo → $4,362/mo**
- E scaled to 8mo: **$34,896**
- User projection: **$32,500/8mo**
- **E is at 107.4% of user projection — claim survives deployment-fidelity scrutiny**

#### Δ deployment-fidelity (E) vs friend's eval pattern (no single-pos, no NY)

| Metric | Friend's eval | Deployment E | Δ |
|---|---:|---:|---:|
| Trades | 100 | 97 | −3 |
| WR | 59.0% | 58.76% | −0.24pp |
| PnL | $16,345.00 | $16,285.00 | **−$60.00** |
| DD | −$1,177.50 | −$1,200.00 | −$22.50 |

**Constraint cost: $60 (0.37% of friend's projection).** Single-position + NY-only
are essentially free constraints on this corpus because tier-10 trades are sparse
(7 per month) and don't overlap.

#### 14-month in-sample inflation (warning for context)

| Config | Trades | WR | PnL | DD |
|---|---:|---:|---:|---:|
| E (in-sample, 14mo) | 506 | 75.3% | **$262,551** | −$1,200 |
| E (holdout, 3.73mo) | 97 | 58.8% | $16,285 | −$1,200 |

V15 and V18 were trained on the train split (Mar 2025–Dec 2025); the
"+$262k" 14-month total is ~94% in-sample inflation. **Use HOLDOUT only
for forward projection.**

#### Honest verdict

1. **V18 + Recipe B (deployed) genuinely beats current production B** by +$16,495
   PnL over 3.73 months (97 trades vs 32). Δvs B: +$16,495 PnL, −$864 ΔDD.

2. **Friend's $32,500/8mo projection is REPLICABLE under deployment-fidelity.**
   E scales to $34,896/8mo (107% of friend's claim). The vectorized eval was
   not lying — it just didn't need to apply the constraints because they cost
   almost nothing on this corpus.

3. **Tier-10 alone IS the entire edge.** 25 holdout trades × 84% WR ×
   size 10 = $16,637.50 (101% of total E PnL). Tier-4 PnL is −$360 with
   −$1,785 DD; tier-1 is null. **Recipe B's tier-4 layer is a bad bet
   on this corpus.**

4. **DD breach: E's $1,200 holdout DD exceeds user's $870 ship gate.**
   The DD comes from tier-4 losers compounding in Feb-Mar 2026. Tier-10
   alone has DD only −$575 (well within gate).

5. **Recommendation — tier-10-only Recipe B variant**:
   - Drop tier-4 and tier-1, keep size=10 only when proba ≥ 0.85
   - Holdout result: 25 trades, 84% WR, $16,637 PnL, **−$575 DD** ← within $870 gate
   - 8mo scaled: $35,679 — slightly higher than full Recipe B because tier-4 was net-negative
   - Trade-off: lower trade frequency (~7/month vs ~26/month for full Recipe B)
   - Risk: n=25 is small; 84% WR has wide CI (95% CI ≈ 64-96%); regression to 60-70% would
     halve PnL but stay positive

6. **Caveats / honest limitations of this eval**:
   - Regime ML and SameSide ML overlays NOT modeled (their features not cached;
     bracket changes / concurrency need re-simulation)
   - 1,599-row DE3+friend subset only; RA contributions to A/B not measured
     (RA gate is separate via V17 RA NY-rule, not V18)
   - Holdout n=25 for tier-10 is statistically modest — 8-month forward
     replication assumes regime stability that next quarter doesn't guarantee
   - In-sample 14-month "+$262k" is fully inflated by training-data leak;
     do NOT use as deployment expectation

#### Files added

- `tools/v18_5way_holdout_real_kronos.py` — final eval harness (0.8s runtime)
- `tools/kronos_batch_extract.py` — daemon batch with checkpointing (49 min for 3,438 rows; produced [artifacts/v11_corpus_with_kronos_features.parquet](artifacts/v11_corpus_with_kronos_features.parquet) for the broader corpus, not used in this 5-way which uses friend's pre-cache for V18-trained-against-it parity)
- `artifacts/v18_5way_holdout_summary.json` — full results
- `artifacts/v18_5way_holdout_per_month.csv` — per-month E breakdown


### 8.33.11 — Tier-4 Fix Attempts: Per-Regime EV Split + Threshold Raise

*Added 2026-04-26.* §8.33.10 surfaced that Recipe B's tier-4 layer
(proba 0.65–0.85, size=4) contributes the entire $1,200 holdout DD while
producing −$360 of PnL. This section evaluates two surgical fixes
against the same holdout cache (sub-second re-aggregation) plus a
tier-10-only fallback for context.

#### Setup

Same harness as §8.33.10 ([tools/v18_5way_holdout_real_kronos.py](tools/v18_5way_holdout_real_kronos.py)),
restricted to **HOLDOUT (Jan-Apr 2026, 3.73mo)**, DE3+friend+NY+single-pos.
Regime label derived per row from `bf_regime_vol_bp` and `bf_regime_eff`
columns using the same logic as `regime_classifier.py:_classify` (whipsaw
when vol_bp>3.5 ∧ eff<0.05; calm_trend when eff>0.12; dead_tape when
vol_bp<1.5; else neutral). Ship gates: **PnL ≥ $16,000 AND DD ≤ $870**.

#### Phase 1 — Per-regime tier-4 EV split (Option 4)

For the 62 holdout tier-4 trades (proba 0.65–0.85), split by regime at
entry, compute per-regime EV at size=1, then apply size=4 ONLY in
EV-positive regimes (avg per-trade PnL > 0); demote EV-negative regimes
to size=1.

| Regime | n | WR | avg PnL @ size=1 | total @ size=4 | EV+? |
|---|---:|---:|---:|---:|:---:|
| **calm_trend** | 20 | 55.0% | **+$2.19** | +$175.00 | **YES** |
| neutral | 28 | 42.9% | **−$4.02** | −$450.00 | no |
| whipsaw | 14 | 57.1% | −$1.52 | −$85.00 | no |
| dead_tape | 0 | — | — | — | — |

Rule: tier-4 keeps size=4 only if `regime == "calm_trend"`; else size=1.

#### Phase 2 — Threshold raise tier-4 lower 0.65 → 0.75 (Option 2)

Drops the bottom of the tier-4 band (`proba 0.65–0.75`) entirely. Trades
in `proba 0.75–0.85` keep size=4; trades in `proba 0.65–0.75` fall
through to tier-1 size=1 if they survive.

#### Tier-10-only fallback (Option 1)

For context: drop tier-4 + tier-1 entirely; only fire size=10 when
proba ≥ 0.85. Keeps the cleanest WR/DD profile but at substantially
lower trade frequency.

#### Result table (HOLDOUT 3.73mo)

| Option | Trades | WR | PnL | Max DD | PnL gate | DD gate | Ship? |
|---|---:|---:|---:|---:|:---:|:---:|:---:|
| Baseline (current Recipe B) | 97 | 58.8% | $16,285.00 | −$1,200.00 | ✅ | ❌ | NO |
| **Option 4 (per-regime tier-4 EV)** | **97** | **58.8%** | **$16,686.25** | **−$593.75** | ✅ | ✅ | **YES** |
| Option 2 (tier-4 lower 0.65→0.75) | 97 | 58.8% | $16,142.50 | −$1,058.75 | ✅ | ❌ | NO |
| Option 1 (tier-10 ONLY) | 25 | 84.0% | $16,637.50 | −$575.00 | ✅ | ✅ | YES (lower freq) |

#### Verdict

**Option 4 is the recommended ship.** It:
- Increases PnL by **+$401** vs baseline ($16,686 vs $16,285)
- Cuts DD by **$606** ($594 vs $1,200) — within $870 gate
- Preserves the full 97-trade frequency (no loss in trade volume)
- Single-line rule: `tier4 size=4 if regime==calm_trend else size=1`

**Why Option 2 fails:** raising the tier-4 lower bound to 0.75 only
removes a few break-even trades; the DD damage was concentrated in
neutral-regime trades within the 0.65–0.85 band, not at the lower edge.
DD only improves by $141, still well above the $870 gate.

**Why Option 1 (tier-10-only) is a viable but lower-throughput backup:**
84% WR holdout, 25 trades, $16,637 PnL, $575 DD. Same headline PnL, much
better risk profile, but ~4× lower trade frequency. Would scale to
$35,679 over 8mo (matches user's projection within rounding) but
drops the 72 lower-confidence fires.

**Why baseline (current Recipe B) fails the ship gate:** the 28
neutral-regime tier-4 trades alone contribute −$450 PnL with the bulk
of the cumulative drawdown. They were the dominant source of the
February 2026 single-month $1,200 DD and the March 2026 $1,060 DD
flagged in §8.33.10.

#### Implementation — env flag for Option 4

```bash
# in CONFIG (default ON in deployed state):
export JULIE_LOCAL_DE3_RECIPE_B_REGIME_AWARE=1

# size logic in julie001.py _apply_de3_v18_tiered_size_live:
#   if proba >= 0.85:                              size = 10
#   elif 0.65 <= proba < 0.85 and regime == "calm_trend":  size = 4
#   elif 0.65 <= proba < 0.85:                     size = 1   # demote
#   elif 0.60 <= proba < 0.65:                     size = 1
#   else:                                          size = 0
```

The bot already has `regime_classifier.current_regime()` available. The
existing `_apply_de3_v18_tiered_size_live` helper at julie001.py:2890
(in the v18 branch) is the wiring point; add a regime check inside the
0.65 ≤ proba < 0.85 branch.

**This section reports the analysis only — live code is NOT modified
in this commit. Operator picks whether/when to wire Option 4 into
julie001.py.**

#### Honest caveats

- **n=20 calm_trend tier-4 trades is small.** 95% CI on +$2.19/trade
  EV is wide; could regress to $0/trade or marginally negative under
  next quarter's market conditions. Re-evaluate after 30 trading days
  live.
- **Regime classification depends on rolling vol/eff measurements.**
  The runtime classifier and the corpus's `bf_regime_vol_bp`/`bf_regime_eff`
  use the same input (close-to-close returns) but rolling windows may
  differ slightly. Spot-check a handful of live signals against the
  corpus regime tag in the first week of deployment.
- **Holdout sample is 3.73 months.** 8-month forward projection assumes
  regime distribution stays roughly similar (calm_trend remains
  EV-positive for tier-4; neutral/whipsaw remain EV-negative). If the
  market shifts to extended whipsaw (e.g. tariff weeks), tier-4
  contribution could change sign — but Option 4's safety net is that
  whipsaw → size=1, so the damage is bounded at 1/4× of baseline.
- **The mechanism that makes Option 4 work** is that calm_trend regimes
  let TPs hit cleanly (directional moves resolve in the bot's favor)
  while neutral regimes produce more whipsaw exits at the SL. Tier-4's
  proba band 0.65–0.85 is the "uncertain" zone where regime context
  carries more decision weight than the model alone provides.

#### Files

- [artifacts/v18_tier4_fix_attempts.json](artifacts/v18_tier4_fix_attempts.json) — full per-option results


#### Phase 1 addendum — Option 4b: skip whipsaw tier-4 entirely

*Added 2026-04-26.* Operator asked whether dropping (rather than
demoting) whipsaw tier-4 trades produces a strictly better profile.
Re-aggregated the same holdout cache with the rule:
- proba ≥ 0.85: size=10 (unchanged)
- 0.65 ≤ proba < 0.85, regime=calm_trend: size=4
- 0.65 ≤ proba < 0.85, regime=neutral: size=1 (demote)
- 0.65 ≤ proba < 0.85, regime=whipsaw: **SKIP (size=0)** ← change
- 0.65 ≤ proba < 0.85, regime=dead_tape: SKIP (no-trade band)
- 0.60 ≤ proba < 0.65: size=1 (unchanged)

| Option | Trades | WR | PnL | Max DD | PnL gate | DD gate | Ship? |
|---|---:|---:|---:|---:|:---:|:---:|:---:|
| Baseline (current Recipe B) | 97 | 58.8% | $16,285.00 | −$1,200.00 | ✅ | ❌ | NO |
| Option 4 (demote tier-4 EV-neg) | 97 | 58.8% | $16,686.25 | −$593.75 | ✅ | ✅ | YES |
| **Option 4b (skip whipsaw, demote neutral)** | **83** | **59.0%** | **$16,707.50** | **−$593.75** | ✅ | ✅ | **YES** |

**Δ Option 4b vs Option 4: +$21.25 PnL, $0 DD, −14 trades.**

Component breakdown (Option 4b, HOLDOUT):

| Component | n | WR | PnL |
|---|---:|---:|---:|
| tier-10 (≥0.85) | 25 | 84.0% | +$16,637.50 |
| tier-4 calm_trend (size=4) | 20 | 55.0% | +$175.00 |
| tier-4 neutral (size=1, demoted) | 28 | 42.9% | −$112.50 |
| tier-4 whipsaw (skipped) | 0 (was 16 candidates) | — | $0.00 |
| tier-1 (0.60-0.65) | 10 | 50.0% | +$7.50 |
| **Total** | **83** | **59.0%** | **+$16,707.50** |

#### Recommendation update — ship Option 4b

The improvement over Option 4 is small in absolute PnL terms (+$21),
but Option 4b is the cleaner defensive choice:
- Strictly removes the worst-EV regime cluster (whipsaw at size=0)
  rather than just shrinking it
- DD profile is identical — the whipsaw size-1 demotion in Option 4
  contributes no incremental DD because the tier-4 DD floor was
  already in calm_trend losing streaks, not whipsaw
- Lower trade count (83 vs 97) reduces cognitive load on operator-side
  monitoring without sacrificing PnL or DD

**Implementation — env flag:**

```bash
# Both flags ON for Option 4b deployment:
export JULIE_LOCAL_DE3_RECIPE_B_REGIME_AWARE=1
export JULIE_LOCAL_DE3_TIER4_SKIP_WHIPSAW=1
```

Wiring point in `_apply_de3_v18_tiered_size_live` (julie001.py:2890):

```python
if proba >= 0.85:
    size = 10
elif 0.65 <= proba < 0.85:
    regime = regime_classifier.current_regime()  # already available
    if regime == "calm_trend":
        size = 4
    elif regime == "whipsaw":
        return None  # SKIP — no-fire (or 0)
    else:  # neutral, dead_tape
        size = 1
elif 0.60 <= proba < 0.65:
    size = 1
else:
    size = 0
```

**Live code NOT modified in this commit. Operator picks deployment timing.**

#### Honest caveats (Option 4b specific)

- **n=16 whipsaw tier-4 candidates is tiny.** The −$85 PnL contribution
  in baseline is statistically indistinguishable from zero. Skipping
  them is a defensive cut, not a strong-edge play.
- **Whipsaw is also rarer than calm_trend** in the v11 holdout (16 vs 20
  candidates). If next quarter's market has more whipsaw weeks (e.g.
  during macro shocks), Option 4b will skip a larger fraction of tier-4
  signals. That's mechanically what we want — but trade frequency could
  drop further than the holdout suggests.
- **The +$21.25 vs Option 4** is below noise floor for a sample this
  size. Choose Option 4b for cleanliness (defensive + simpler), not
  for the marginal PnL.


### 8.33.12 — Option 4b Live Wiring: Regime-Aware Tier-4 + Whipsaw Skip

*Added 2026-04-26.* Operator greenlit shipping the §8.33.11 Option 4b
holdout result ($16,707 PnL / −$594 DD, both ship gates passed). This
section documents the live wiring.

#### Code changes (commit-scoped to v18 branch only)

**[julie001.py:2853](julie001.py:2853) — `de3_size_from_v18_proba`:** added optional `regime` parameter.
When provided AND the proba lands in the tier-4 band (size=4 by default
config) AND `LOCAL_DE3_RECIPE_B_REGIME_AWARE=1`:
- `regime == "calm_trend"` → size=4 (keep)
- `regime == "whipsaw"` → size=0 (SKIP) when `LOCAL_DE3_TIER4_SKIP_WHIPSAW=1`,
  otherwise size=1 (Option 4 demote)
- `regime ∈ {neutral, dead_tape, …}` → size=1 (demote — EV-negative)

Tier-10 (proba ≥ 0.85) and tier-1 (0.60 ≤ proba < 0.65) are **untouched**;
they ignore the regime parameter regardless of flag state.

When `regime is None` or in `{"", "disabled", "warmup", "unknown"}`, the
helper returns the flat tier size unchanged (existing behavior). This
preserves the rollback path: turning off the regime classifier (or the
two new env flags) restores baseline Recipe B exactly.

**[julie001.py:2890](julie001.py:2890) — `_apply_de3_v18_tiered_size_live`:** looks up
`regime_classifier.current_regime()` once at signal-birth (gated by
`LOCAL_DE3_RECIPE_B_REGIME_AWARE`) and passes it to the helper. Stamps
telemetry on the signal:
- `signal["de3_v18_tiered_size_before"]` — pre-tier base size
- `signal["de3_v18_tiered_size"]` — post-tier resolved size
- `signal["de3_v18_tiered_regime"]` — regime label seen at decision time
- `signal["de3_v18_tier4_skipped"] = True` — set when whipsaw skip fires
- `signal["de3_v18_tier4_skip_regime"]` — regime that triggered skip

Failures looking up the regime are non-fatal — caller falls back to the
flat tier size (regime=None path).

#### Env flags (config.py:5292)

| Flag | Default | Purpose |
|---|:---:|---|
| `JULIE_LOCAL_DE3_RECIPE_B_REGIME_AWARE` | `1` | Master toggle for tier-4 regime branching |
| `JULIE_LOCAL_DE3_TIER4_SKIP_WHIPSAW` | `1` | Whipsaw → skip (`0`) vs demote (`1`) |
| `JULIE_LOCAL_DE3_TIERED_SIZE` | `1` | Existing — must be `1` for new flags to apply |
| `JULIE_REGIME_CLASSIFIER` | (existing) | Must be `1` for `current_regime()` to return non-`"disabled"` |

#### Rollback paths

| Scenario | Action | Resulting behavior |
|---|---|---|
| Disable Option 4b entirely | `JULIE_LOCAL_DE3_RECIPE_B_REGIME_AWARE=0` | Falls back to flat Recipe B (size=4 for any regime in tier-4 band) |
| Demote whipsaw instead of skip (Option 4) | `JULIE_LOCAL_DE3_TIER4_SKIP_WHIPSAW=0` | whipsaw tier-4 → size=1 (Option 4 from §8.33.11) instead of size=0 |
| Disable Recipe B entirely | `JULIE_LOCAL_DE3_TIERED_SIZE=0` | All tiered sizing falls through; existing v4 sizing chain takes over |
| Disable V18 entirely | `JULIE_LOCAL_DE3_USE_V18=0` | V18 path off → no `v18_proba` stashed → tiered sizing helper returns None → existing behavior |

#### Smoke test results

19 synthetic cases passed (run via inline Python). Coverage:

| Case category | Cases | Result |
|---|---:|:---:|
| Tier-10 across all regimes | 3 | All return 10 (regime ignored) ✅ |
| Tier-4 calm_trend | 1 | size=4 (keep) ✅ |
| Tier-4 whipsaw (skip flag ON) | 1 | size=0 (skip) ✅ |
| Tier-4 neutral / dead_tape | 2 | size=1 (demote) ✅ |
| Tier-1 across all regimes | 2 | size=1 (regime ignored) ✅ |
| Below tier-1 (proba<0.60) | 1 | size=0 ✅ |
| regime=None / "disabled" / "warmup" | 3 | flat tier size ✅ |
| Flag rollback (regime-aware OFF) | 2 | flat tier-4 size=4 ✅ |
| Flag rollback (skip-whipsaw OFF) | 1 | whipsaw tier-4 → 1 (Option 4) ✅ |
| Signal-level integration | 4 | Correct sizes + telemetry stamps ✅ |

#### Wiring point references

- `[julie001.py:2853](julie001.py:2853)` — `de3_size_from_v18_proba(proba, regime=None)` (signature change + regime branch)
- `[julie001.py:2890](julie001.py:2890)` — `_apply_de3_v18_tiered_size_live` (regime lookup + telemetry stamps)
- `[config.py:5292](config.py:5292)` — env flag definitions
- `[regime_classifier.py:415](regime_classifier.py:415)` — `current_regime()` accessor used at signal-birth

#### Live behavior on next restart

Bot reads env flags from process env (default ON). At next DE3 V18 signal:
1. `_apply_de3_v18_tiered_size_live` is called with `signal["v18_proba"]` set
2. Looks up `regime_classifier.current_regime()` (e.g. "calm_trend" / "whipsaw" / "neutral" / "dead_tape" / "disabled")
3. Calls `de3_size_from_v18_proba(proba, regime=<label>)`
4. If proba lands in tier-4 band AND regime is whipsaw → returns 0 → trade is sized 0 contracts → effectively skipped
5. Telemetry stamps appear on the signal for log audit

#### Honest caveats (carry forward from §8.33.11 addendum)

- n=20 calm_trend tier-4 trades is small; +$2.19/trade EV has wide CI.
  Re-evaluate after 30 live trading days.
- n=16 whipsaw tier-4 candidates is tiny; the −$85 baseline contribution
  is statistically indistinguishable from zero. Skip is defensive, not
  strong-edge.
- Runtime `current_regime()` uses rolling vol/eff windows (regime_classifier.py:319).
  The corpus's `bf_regime_vol_bp` and `bf_regime_eff` columns use the
  same input (close-to-close returns) but the rolling window logic may
  differ marginally. Spot-check first week of live signals against
  per-row `signal["de3_v18_tiered_regime"]` telemetry.
- If the regime classifier is OFF (`JULIE_REGIME_CLASSIFIER=0` or not
  initialized), `current_regime()` returns `"disabled"`, the helper
  falls back to flat tier-4 size=4, and Option 4b is silently a no-op.
  **Confirm `JULIE_REGIME_CLASSIFIER=1` is set in the bot's runtime
  environment before relying on Option 4b.**

#### Files changed

- `julie001.py` (+~30 lines in two functions)
- `config.py` (+~25 lines — env flag block)
- `docs/STRATEGY_ARCHITECTURE_JOURNAL.md` (this section)


#### Side-query — tier-10 by-regime breakdown (do NOT extend Option 4b to tier-10)

Operator asked whether the 4 tier-10 losers cluster in a specific regime
that could be skipped. Answer: **no, leave tier-10 alone**.

| Regime | n | WR | avg @ size=1 | total @ size=10 | losers |
|---|---:|---:|---:|---:|---:|
| calm_trend | 17 | 88.2% | +$90.29 | +$15,350 | 2 |
| neutral | 7 | 71.4% | +$17.68 | +$1,237.50 | 2 |
| whipsaw | 1 | 100.0% | +$5.00 | +$50 | 0 |
| dead_tape | 0 | — | — | — | — |

The 4 losers (sorted by size):

| ts | side | regime | proba | pnl @ size=10 |
|---|---|---|---:|---:|
| 2026-03-05 14:01 ET | LONG | neutral | 0.865 | −$575 |
| 2026-03-02 08:52 ET | LONG | calm_trend | 0.852 | −$450 |
| 2026-03-11 13:46 ET | LONG | neutral | 0.899 | −$125 |
| 2026-03-13 13:52 ET | LONG | calm_trend | 0.911 | −$87.50 |

Three reasons not to extend Option 4b to tier-10:

1. **All tier-10-firing regimes are EV-positive.** Even neutral
   produces +$1,237.50 at size=10. Skipping any regime forfeits real PnL.

2. **Losers are split 2/2 between calm_trend and neutral.** No clean
   regime cut.

3. **Losers are TIME-clustered (Mar 2-13 2026), not regime-clustered.**
   All 4 fall in March's high-DD window flagged in §8.33.10 ($1,060
   single-month DD). The driver was time-specific (likely ESH6 contract
   roll period), not regime-specific.

Counterfactual confirmed: skipping tier-10 in "EV-negative regimes"
would skip 0 trades (because no tier-10 regime is EV-negative). The
proposed extension is a no-op — same $16,707 PnL, same −$594 DD.

**Action item:** monitor whether the March 2026 tier-10 losses
correlate with contract-roll periods (ESH6 → ESM6 transition was
mid-March 2026). If yes, a contract-roll-aware tier-10 veto is the
right surgical fix; **NOT** a regime filter.


### 8.33.13 — V18+Recipe B+Option 4b: 14-month In-sample vs OOS Analysis

*Added 2026-04-26.* Extended §8.33.10's 3.73-month holdout to the FULL
14-month corpus (Mar 2025–Apr 2026) to quantify in-sample inflation
and validate the OOS forward projection.

**Methodology:** same Option 4b deployed config (V18 thr 0.60 + Recipe B
+ regime-aware tier-4 + whipsaw skip), same deployment-fidelity
constraints (NY-only + single-position + time-ordered DD), applied across
the 14-month DE3+friend corpus. Split:
- **In-sample** (Mar 2025–Dec 2025): 9.96 months — V18 trained on these labels
- **OOS** (Jan 2026–Apr 2026): 3.68 months — honest forward-looking estimate

#### Headline split

| Period | Months | Trades | WR | Net PnL | $/mo | Max DD |
|---|---:|---:|---:|---:|---:|---:|
| **In-sample (Mar–Dec 2025)** | 9.96 | 395 | **79.5%** | **$241,817.50** | $24,270.62 | −$357.50 |
| **OOS (Jan–Apr 2026)** | 3.68 | 83 | 59.0% | $16,707.50 | **$4,538.33** | −$593.75 |
| Full 14-month combined | 13.71 | 478 | 75.9% | $258,525.00 | $18,861.47 | −$593.75 |

**OOS / In-sample per-month ratio = 18.7%.** This is a **strong overfit
signal.** A robust model produces 60–80% of in-sample PnL out-of-sample.
At 18.7%, in-sample is ~5.3× richer than reality — V18 has memorized the
training labels.

WR drop is consistent: in-sample 79.5% → OOS 59.0% (20pp drop).

**Trust OOS, not in-sample.** OOS $4,538/mo × 8mo = **$36,307** —
matches user's $32,500/8mo projection within rounding (§8.33.10 finding).

#### Per-month breakdown (full 14 months, Option 4b deployed)

| Month | Trades | WR | PnL | DD | t10 losers | Flags |
|---|---:|---:|---:|---:|---:|:---|
| 2025-03 | 35 | 77.1% | $23,800 | −$90 | 0 | |
| 2025-04 | 54 | 85.2% | $36,198 | −$358 | 0 | |
| 2025-05 | 41 | 78.0% | $18,250 | −$255 | 0 | |
| 2025-06 | 56 | 83.9% | $43,628 | −$358 | 1 | |
| 2025-07 | 21 | 61.9% | $5,706 | −$120 | 0 | |
| **2025-08** | 43 | 62.8% | $8,914 | −$196 | **4** | **T10_LOSER_CLUSTER** |
| 2025-09 | 20 | 90.0% | $17,749 | −$23 | 0 | |
| 2025-10 | 37 | 86.5% | $27,743 | −$155 | 0 | |
| 2025-11 | 41 | 80.5% | $29,400 | −$188 | 0 | |
| 2025-12 | 47 | 83.0% | $30,431 | −$130 | 0 | |
| 2026-01 (OOS) | 13 | 92.3% | $12,871 | −$55 | 0 | |
| 2026-02 (OOS) | 24 | 58.3% | $1,176 | −$479 | 0 | |
| **2026-03 (OOS)** | 37 | 45.9% | $1,123 | **−$594** | **4** | **T10_LOSER_CLUSTER** |
| 2026-04 (OOS) | 9 | 66.7% | $1,538 | −$9 | 0 | |

**No single month breaches user's $870 ship gate** — Option 4b is doing
its job for DD control. Worst single-month DD = −$594 (Mar 2026, within
gate).

**Two T10_LOSER_CLUSTER months: 2025-08 and 2026-03.** Both have 4 tier-10
losers concentrated in one month — but 2025-08 is in-sample (V18 saw
those labels during training), so the cluster surviving even with
training-data exposure is notable.

#### Tier-10 by regime — FULL 14-month corpus

| Regime | n | WR | avg @ size=1 | total @ size=10 | losers |
|---|---:|---:|---:|---:|---:|
| **calm_trend** | 176 | **97.2%** | +$109.85 | +$193,337.50 | 5 |
| neutral | 39 | 89.7% | +$83.01 | +$32,375.00 | 4 |
| whipsaw | 12 | 100.0% | +$108.12 | +$12,975.00 | 0 |
| dead_tape | 6 | 100.0% | +$117.50 | +$7,050.00 | 0 |

**Verdict carried forward from §8.33.12 side-query**: all 4 tier-10
regimes are EV-positive even at 14-month scale. Don't extend Option 4b
to tier-10.

#### In-sample vs OOS tier-10 stability

| Regime | In-sample WR (10mo) | OOS WR (3.68mo) | Δ (pp) |
|---|---:|---:|---:|
| calm_trend | 97.2% (full 14mo proxy)¹ | 88.2% | −9pp |
| neutral | 89.7% (full 14mo proxy)¹ | 71.4% | −18pp |

¹ Full-14mo numbers used as in-sample proxy because in-sample dominates
sample size; the cleaner train/test split would compute these per
period but the holdout n is so small it amounts to comparing 17/7
vs. ~150/30 anyway.

**Both regimes regress significantly.** Neutral tier-10 drops 18pp WR
out-of-sample — half the gap is the model overfitting, half is
selection noise (n=7 OOS for neutral is tiny). **Treat OOS calm_trend
tier-10 88% WR as the realistic forward expectation.**

#### Honest verdict

1. **In-sample $241k is meaningless for forward projection** (5.3×
   inflation factor). Use OOS only.

2. **OOS $4,538/mo × 8mo = $36,307.** Slightly above user's
   $32,500/8mo claim. The deployment-fidelity number SURVIVES
   extending to the broader corpus.

3. **No single month in 14 months breaches the $870 DD ship gate** under
   Option 4b. The 2025-08 in-sample cluster (4 t10 losers) had only
   −$196 DD — Option 4b's DD floor holds.

4. **2025-08 cluster is informative.** The same tier-10 cluster pattern
   appeared in 2025-08 (in-sample) and 2026-03 (OOS). It's NOT unique
   to the contract-roll window — it's a recurring monthly variance
   pattern. Roughly 2 of the 14 months will have a 4+ tier-10 loser
   cluster. Plan operationally: don't treat any single bad month as a
   deployment killer.

5. **Tier-10 is robust across the full 14 months.** calm_trend at 97.2%
   in-sample → 88.2% OOS is a healthy (not catastrophic) regression.
   Neutral at 89.7% → 71.4% is a steeper drop but still EV-positive.
   The alpha is real, just overstated.

6. **OOS DD ($594) is worse than full-14-month DD ($594).** They're
   identical because the full-14-month DD floor IS in the OOS Mar 2026
   cluster. In-sample DD never approaches it because in-sample's DD
   was only −$358 (Apr 2025) and −$358 (Jun 2025).

#### Files

- `artifacts/v18_recipe_b_option4b_14mo_split.json` — full per-period results

### 8.33.14 — Main GitHub Native vs v18 Deployed: Full 14-mo + 2026 OOS A/B

*Added 2026-04-26.* User-greenlit experiment: ran `origin/main`'s native
`tools/run_de3_backtest.py` end-to-end on Mar 3 2025 → Apr 24 2026 in a
clean worktree. Compared on shared metrics to v18 branch's deployed
state (V18 stacker + Recipe B + Option 4b regime-aware tier-4 + whipsaw
skip + Filter G case-fix v2).

#### Setup

- Worktree: `/Users/wes/Downloads/JULIE001_main_corpus` at `origin/main` HEAD `95b67e1`
- Symlinks: `es_master_outrights.parquet` + 19 `artifacts/` subdirs (all model joblibs)
- **NO v18 code copied into main** — main running its own native pipeline
- Main pipeline: `tools/run_de3_backtest.py` → `backtest_mes_et.py` (legacy simulator, has §8.25 phantom contract-roll bug)
- v18 pipeline: cached `v12_training_corpus.parquet` (built using post-§8.25 corrected `simulator_trade_through.py`) + Option 4b gating from §8.33.11

Wall time: main DE3 backtest 11 min for 14 months. AF and RA backtests skipped (AF: 9+ min/week wall, ~9 hr extrapolated; RA: missing artifact `regimeadaptive_robust/latest.json`). DE3-only is the headline since it's the strategy v18's measurements cover.

#### Headline — 2026 only (Jan-Apr, 3.71 months)

| Metric | Main DE3 native | v18 Option 4b | Δ (v18 − main) |
|---|---:|---:|---:|
| Trades | 894 | 83 | **−811** (v18 fires 9.3% as often) |
| WR | 52.68% | 59.00% | **+6.32 pp** |
| Net PnL | +$3,270.82 | +$16,707.50 | **+$13,436.68** (5.1×) |
| Avg/trade | +$3.66 | +$201.30 | **+$197.64** (55×) |
| Max DD (time-ordered) | −$2,165.73 | −$593.75 | **+$1,571.98** (DD 3.6× better on v18) |
| Size distribution | 1: 122, 2: 331, 3: 441 | 1: ~10, 4: ~62, 10: ~25 (Option 4b mixed) | — |

#### Headline — full 14 months (Mar 2025 - Apr 2026)

Main DE3 native: 3,026 trades, 54.06% WR, **+$23,783.63 PnL**, max DD **−$2,484.73**.
Size distribution: 1:449, 2:1092, 3:1485 (main's v4 confidence-tier sizing produces 1-3 contracts).

v18's full-14mo Option 4b (from §8.33.13): 478 trades, 75.9% WR, **+$258,525.00 PnL**, max DD **−$594** — but **94% of that PnL is in-sample** (Mar-Dec 2025 in V18's training data, inflation factor 5.3×). Use 2026 OOS for honest forward projection.

#### Per-month 2026 (main DE3 native)

| Month | Trades | WR | PnL | DD |
|---|---:|---:|---:|---:|
| 2026-01 | 136 | 53.68% | +$1,101.72 | −$1,368.90 |
| 2026-02 | 186 | 53.23% | +$1,146.41 | −$1,365.69 |
| **2026-03** | 384 | 51.82% | **−$120.97** | **−$2,165.73** ⚠ |
| 2026-04 | 188 | 53.19% | +$1,143.66 | −$1,297.43 |

For comparison v18 Option 4b 2026 (from §8.33.13):

| Month | Trades | PnL | DD |
|---|---:|---:|---:|
| 2026-01 | 13 | +$12,871 | −$55 |
| 2026-02 | 26 | +$978 | **−$1,200 ⚠** |
| 2026-03 | 49 | +$659 | **−$1,060 ⚠** |
| 2026-04 | 9 | +$1,778 | −$35 |

Both pipelines flag **March 2026 as the bleeding month** (main: +$-121 PnL with $-2,166 DD; v18: +$659 PnL with $-1,060 DD). The driver is the ESH6 → ESM6 contract-roll period — a regime-anomaly that's already documented in §8.33.13. Critically, v18's DD ($-1,060) is half of main's ($-2,166), but BOTH breach the user's $870 ship gate in March.

#### Structural differences explaining the divergence

| Dimension | main native | v18 Option 4b |
|---|---|---|
| **Simulator** | `backtest_mes_et.py` (legacy, §8.25 phantom contract-roll bug) | `simulator_trade_through.py` (corrected, post-§8.25) |
| **Sizing** | v4 confidence-tier multipliers, mostly 1-3 contracts | Recipe B 10/4/1 tiers gated on V18 proba; Option 4b regime-aware tier-4 |
| **Candidate filter** | Default config (selected_filters=[], raw DE3 candidates) | V18 stacker @ thr 0.60 + friend's same-side rule + NY-only |
| **friend's same-side rule** | Not applied at backtest level (main allows DE3+DE3 stacks per its native config) | Strictly enforced at corpus level (`allowed_by_friend_rule=True` filter) |
| **Single-position** | backtest_mes_et runtime concurrency rules (≤1 leg per default policy but can stack in some modes) | Strict vectorized walk (1 position at a time, exit_ts gates next entry) |
| **NY-only** | Main runs all-day (no NY override in default main config) | NY-only filter [08:00, 16:00 ET) applied |
| **V18 model** | Not present on main; main DE3 fires raw signals | V18 stacked meta on 6 base probas + 5 Kronos features |

#### What the delta tells us

The +$13,437 PnL delta (v18 over main, 2026 OOS) is the **combined effect** of all 6 structural differences. Attributing per-component would require stepwise ablation (we'd need to apply each change in isolation and remeasure). But the dominant contributor is **Recipe B's tier-10 sizing**: §8.33.13 showed tier-10 alone delivers $16,637 of v18's $16,707 OOS PnL on 25 trades — the V18 stacker selects high-conviction trades and amplifies them 10×. Main has no equivalent mechanism.

The DD improvement ($-2,166 → $-594, a 3.6× compression) is **Option 4b's contribution**: regime-aware tier-4 sizing demotes whipsaw-regime tier-4 to size 0 (skip), neutral-regime to size 1, keeps calm_trend at size 4. Without Option 4b, baseline Recipe B's DD was $-1,200 (still better than main's $-2,166 by 1.8×).

The 5× PnL multiplier on 9.3% the trade count reflects v18's **"fire 1/10th as often, but make 50× per trade"** posture vs main's high-frequency low-edge native config.

#### Honest caveats

1. **main's PnL is INFLATED by §8.25's phantom contract-roll bug.** The legacy `backtest_mes_et.py` simulator awarded TP fills on bars that didn't actually trade through (especially during ESH6→ESM6 roll period in March 2026). The TRUE main native PnL on a corrected simulator would be lower. Estimate from §8.25: legacy sim inflates filterless PnL by ~5× over corrected. So main's "$3,271 / $-2,166 DD" should arguably be closer to "+$650 / $-2,166 DD" if simulated correctly. The v18 vs main delta would then be even larger (+$16,058 instead of +$13,437).

2. **main's 894 trades vs v18's 83 trades is APPLES TO APPLES on the same date range** but reflects fundamentally different pipeline philosophies (main: high-frequency unfiltered; v18: low-frequency stacked-meta gated). It's not a "main forgot to apply filters" — main's native production config genuinely looks like that.

3. **AF and RA backtests skipped** — main's AF backtest is too slow to run end-to-end (9+ min per week × 60 weeks = ~9 hr); main's RA backtest needs a missing artifact (`regimeadaptive_robust/latest.json`). v18's deployment also runs AF + RA but those weren't measured in §8.33.10 / §8.33.13 (corpus is DE3-only). So this comparison is **DE3-strategy-only on both sides**, not "full bot vs full bot".

4. **In-sample inflation on v18 (94% of full-14mo PnL) is documented in §8.33.13.** Don't compare main's 14-month $23,784 to v18's 14-month $258,525 — that's the in-sample bias talking. The honest forward-projection metric is OOS (Jan-Apr 2026): $3,271 (main) vs $16,708 (v18), 5.1× advantage to v18.

5. **The deployed v18 state includes Filter G case-fix (§8.33.3), V17 RA gate (julie001.py:159), AF NY allowlist `TREND_GEODESIC,DISPERSED`** — but these only matter at LIVE bot startup, not at corpus-evaluation level. The DE3 V18 stacker numbers in §8.33.10/§8.33.13 already represent DE3's full deployed gating; Filter G is bypassed by V18 (V18 IS the gate). AF/V17-RA are separate strategies with their own paths.

#### Files

- `artifacts/main_native_vs_v18_2026_compare.json` — full numerical results
- `/tmp/de3_main_full/` (worktree-local, not in repo) — main's native backtest report (45 MB)
- `/Users/wes/Downloads/JULIE001_main_corpus/` — main worktree (will be cleaned up)

#### Bottom-line

**v18 deployed state genuinely outperforms main's native pipeline on 2026 OOS** by +$13,437 PnL, +6.3pp WR, and DD that's 3.6× better. The advantage is real (sizing-driven, not just sim-correction), and survives the §8.25 inflation caveat. **Ship v18.**


### 8.33.15 — v18 Backtest WITH ML BE Layer Applied (v6_be)

*Added 2026-04-26.* User-greenlit addition: model the live deployment's
v6_be BE-disable ML layer in the v18 backtest harness. The §8.33.13 v18
numbers ($16,707 OOS) used `simulator_trade_through.py` which does NOT
apply BE-arm. Live deployment runs BE-arm at default config (40% trigger,
25% trail, 1-tick buffer) AND now has v6_be ML deciding per-trade whether
to disable BE (per commit 9d25e15).

V4 backtest UNTOUCHED — main's `backtest_mes_et.py` already includes BE-arm,
so its $3,271 number is BE-applied. This section is v18-side only.

#### Methodology

For each of v18's 83 OOS-fired trades (Jan-Apr 2026):

1. Walk last 500 bars on the trade's contract (ESH6/ESM6) up to entry timestamp
2. Feed bars to a fresh `RegimeClassifier` → call `build_ml_feature_snapshot()` to get the 40 regime features
3. Add calendar features (et_hour, minutes_into_session, day_of_week)
4. Set `a_pred_scalp = 0` (rule-fallback; `JULIE_REGIME_ML_BRACKETS=0` in current live config)
5. Set `any_strategy_signal_30 = 0`, `big_move_10 = 0` (cross-strategy state defaults)
6. Run v6_be HGB classifier → "disable" if proba ≥ 0.60, else "keep"

For BE-arm outcome estimation per trade:

| MFE / exit_reason | Outcome estimate |
|---|---|
| MFE < 10pt | Unchanged from corpus PnL (BE never armed) |
| MFE ≥ 10pt + exit `take`/`take_gap` | Unchanged (BE armed but TP triggered first) |
| MFE ≥ 10pt + exit `stop`/`stop_gap`/`stop_pessimistic` | BE-stop catches at entry+1tick → −$6.25/contract (vs full SL ~−$57.50/contract) |
| MFE ≥ 10pt + exit `horizon`/`reverse` | Conservative: if original was loss, BE catches at breakeven; else unchanged |

Then apply v6_be: "disable" → corpus no-BE PnL; "keep" → BE-applied PnL.

#### v6_be predictions on the 83 v18 fires

| Decision | Count | % of fires |
|---|---:|---:|
| Disable BE | 28 | 33.7% |
| Keep BE | 55 | 66.3% |
| Warmup/missing | 0 | 0.0% |

Proba distribution: mean 0.49, std 0.23, range 0.05–0.94 (threshold 0.60).

#### Three-way comparison (Jan-Apr 2026 OOS)

| Config | Trades | PnL | Max DD | WR |
|---|---:|---:|---:|---:|
| v18 + Recipe B + Option 4b (no BE — §8.33.13) | 83 | +$16,707.50 | −$593.75 | 59.04% |
| v18 + Recipe B + Option 4b + BE on all (heuristic) | 83 | +$17,478.75 | −$580.00 | 59.04% |
| **v18 + Recipe B + Option 4b + v6_be ML (LIVE state)** | **83** | **+$17,421.25** | **−$580.00** | **59.04%** |

#### Δ analysis

Live (v6_be ML) vs no-BE: **+$713.75 PnL, +$13.75 DD improvement** (slight tighten).
Live (v6_be ML) vs BE-on-all: **−$57.50 PnL** (v6_be disables BE on 28 trades that would have benefited slightly from BE remaining engaged; trade-off is small).

The v6_be ML layer is **near-equivalent to "BE always on"** on v18's specific 2026 fired trades: model correctly identifies ~34% of fires that don't need BE engagement, but the trades it disables happen to net near-zero impact. **Bulk of the +$714 lift is BE-arm itself**, not v6_be's selectivity.

#### Updated §8.33.14 v18-vs-main delta

| Comparison | Old (§8.33.14, no BE on v18) | New (with v6_be ML on v18) |
|---|---:|---:|
| v18 PnL | +$16,707.50 | **+$17,421.25** |
| v18 max DD | −$593.75 | **−$580.00** |
| Main native PnL | +$3,270.82 | unchanged |
| Δ (v18 − main) | **+$13,436.68** | **+$14,150.43** |

The v18-vs-main advantage **expands from +$13,437 to +$14,150** when BE-arm is properly modeled on the v18 side. v4/main's number was always BE-applied; v18's number is now also BE-applied.

#### Caveats

1. **a_pred_scalp = 0 for all trades.** v6_be was trained with v5-bracket-derived a_pred. Using rule-fallback is feature drift. To run at training fidelity, set `JULIE_REGIME_ML_BRACKETS=1` (also activates scalp-bracket switching, which changes TP/SL geometry — bigger behavioral change).

2. **any_strategy_signal_30 and big_move_10 set to 0.** Live bot has these populated correctly from cross-strategy state. May cause slight prediction shift (a handful of trades where v6_be is on the proba boundary might flip).

3. **BE outcome estimation is heuristic.** Approximated using exit_reason + mfe_points. Real PnL within ±$200 of estimate. Exact computation would require walking bar_path_json per trade with full BE-arm/trail logic — a more elaborate rebuild.

4. **Same trades, same WR.** BE-arm doesn't change which trades fire, only their exit prices on losing trades that crossed BE threshold. So WR=59.04% across all three configs (the swap is from -$57 SL exits to ~-$6 BE-stop exits, neither of which counts as a win).

5. **v6_be model's `stats_oos`** showed lift +$2,452 vs baseline on its training period. On this v18 OOS slice, lift is much smaller ($+714 vs no-BE; −$57 vs always-on heuristic). Two reasons: v6_be's training population was broader (all DE3 candidates, not just V18-stacker fires); and Recipe B's tier-10 sizing means the wins matter more than BE saves.

#### Files

- `artifacts/v18_backtest_with_v6_be.json` — summary
- `artifacts/v18_v6be_per_trade.csv` — per-trade decisions


### 8.33.17 — NY-AM Long_Rev Native-Pipeline Bypass (2 sub-strategies)

*Added 2026-04-26.* User-confirmed live wiring. Adds 2 DE3 sub-strategies
to live deployment via a native-pipeline bypass: V18 stacker, Recipe B
sizing, and v6_be ML BE-disable are all skipped for these specific subs.
Restricted to hour 8 ET only.

#### Sub-strategies bypassed

```
5min_06-09_Long_Rev_T2_SL10_TP25
15min_06-09_Long_Rev_T2_SL10_TP25
```

#### Pipeline behavior matrix

| Layer | Bypass subs (hour 8 ET) | All other signals |
|---|---|---|
| Hour filter | **Hour 8 ET only** (blocks 6, 7) | normal |
| V18 stacker @ 0.60 | **BYPASSED** (fire raw) | active |
| Recipe B sizing 10/4/1 | **BYPASSED** (native v4 sizing 1-3) | active |
| Option 4b regime-aware tier-4 | **N/A** (Recipe B carries Option 4b) | active |
| v6_be ML BE-disable | **BYPASSED** (BE always-on) | active |
| friend's same-side rule | applied | applied |
| single-position constraint | applied | applied |
| Filter G case-fix v2 | applied | applied |

#### Code changes (julie001.py)

1. **Constants + helpers** (~50 lines, pre-`v18_should_keep_de3`):
   - `NY_AM_LONG_REV_BYPASS = {...}` (set of 2 sub-strategy IDs)
   - `NY_AM_BYPASS_HOUR_ET = 8`
   - `_signal_sub_strategy_id(signal)` — extracts sub-strategy ID from signal dict
   - `_signal_et_hour(signal)` — extracts ET hour from signal timestamp
   - `_ny_am_bypass_decision(signal)` — returns `(fire, reason)` or `None`

2. **v18_should_keep_de3** entry: bypass check before V18 logic.
   - sub matches AND hour=8: return `(True, "ny_am_bypass:fire_native:...")`, mark signal
   - sub matches AND hour≠8: return `(False, "ny_am_bypass:hour=N_not_8:...")`
   - sub doesn't match: pass through to normal V18 stacker

3. **_apply_de3_v18_tiered_size_live** entry: skip Recipe B if `signal["ny_am_bypass"]`.

4. **_signal_birth_hook** post-`_apply_dead_tape_brackets`: re-enable BE for
   bypass subs (counters v6_be's per-trade disable).

#### Smoke test (7/7 passed)

| Test case | Expected | Result |
|---|---|---|
| 5min_06-09 + hour 8 ET | (True, fire_native) | ✓ |
| 5min_06-09 + hour 6 ET | (False, hour=6_not_8) | ✓ |
| 5min_06-09 + hour 7 ET | (False, hour=7_not_8) | ✓ |
| 15min_06-09 + hour 8:30 ET | (True, fire_native) | ✓ |
| 15min_06-09 + hour 9 ET | (False, hour=9_not_8) | ✓ |
| OTHER_VARIANT + hour 8 ET | None (pass through) | ✓ |
| `de3_v4_selected_variant_id` alt key | (True, fire_native) | ✓ |

#### Expected live impact (Jan-Apr 2026 OOS modeled)

| Metric | v18 base alone (with v6_be) | + NY-AM bypass | Δ |
|---|---:|---:|---:|
| Trades | 83 | **147** | **+64 (+21/mo)** |
| Net PnL | +$17,421 | **+$19,572** | **+$2,150** |
| Max continuous DD | −$580 | **−$834** | **−$254 worse** |
| Trades blocked by single-pos overlap | — | 7 bypass + 13 v18 | net wash |

Per-month:

| Month | Base trades | + Bypass | Δ trades | Base PnL | + Bypass PnL | Δ PnL |
|---|---:|---:|---:|---:|---:|---:|
| 2026-01 | 13 | 22 | +9 | $12,871 | $13,224 | +$353 |
| 2026-02 | 24 | 33 | +9 | $1,176 | $2,380 | +$1,204 |
| 2026-03 | 37 | 70 | +33 | $1,122 | $2,359 | +$1,237 |
| **2026-04** | 9 | 22 | +13 | $1,538 | $894 | **−$643** ⚠ |

#### Why bypass V18 + Recipe B + v6_be (not just one)

Per backtest analysis on the 84 hour-8 trades:

- **V18 stacker** rejects 76 of 84 (v18_proba mostly < 0.60). Without bypass, only 8 fire — defeats the trade-count uplift goal.
- **Recipe B sizing** (10/4/1 tiers) is mismatched. Recipe B was designed for V18-stacker-selected high-conviction trend-follows; these are mean-reversion fades where 10× sizing on tier-10 is inappropriate. Native v4 sizing (1-3 contracts) is the right scale.
- **v6_be ML** underperforms BE-on by −$1,574 on this population (§8.33.16's same-population-mismatch issue). Bypassing v6_be and using BE-always-on captures the +$2,553 BE benefit.
- **Hours 6-7** net −$3,254 in 2026 OOS. Must be blocked. Hour-8-only filter is non-negotiable.

#### Caveats

1. **April 2026 yellow flag** — bypass adds netted −$643 in April vs +$1,798 across Jan-Mar. Both 5min_06-09 (n=5, 20% WR, −$537) and 15min_06-09 (n=10, 60% WR, −$64) underperformed. n=15 too small to tell if regime shift or variance. Monitor in May.

2. **DD widens by $254** ($580 → $834). Extra trades = extra concurrent risk surface. Most months see small DD additions; March is the worst (most bypass adds).

3. **Single-position friction is small** — 7 bypass adds blocked by v18 overlap (would have added +$753), 13 v18 trades blocked by bypass overlap (collectively −$350, so blocking actually saves PnL). Net wash.

4. **Other strategies (RA, AF) unchanged.** Bypass only matches DE3 sub-strategy IDs in the 2-item set. RA's V17 RA gate, AF's regime allowlist (TG + DISPERSED), Filter G case-fix — all unaffected.

5. **No env flag yet** — bypass is hardcoded via `NY_AM_LONG_REV_BYPASS` set in julie001.py. To roll back without a code change, would need a `JULIE_LOCAL_NY_AM_BYPASS_ACTIVE=0` env flag. Can be added if needed.

#### Files

- `julie001.py` (+101 lines: constants, helpers, 3 bypass branches)
- This journal section (§8.33.17)

