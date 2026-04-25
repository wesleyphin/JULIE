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
