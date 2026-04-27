export interface FilterlessBotRisk {
  daily_pnl?: number | null;
  circuit_consecutive_losses?: number | null;
  circuit_tripped?: boolean | null;
  long_consecutive_losses?: number | null;
  short_consecutive_losses?: number | null;
  long_blocked_until?: string | null;
  short_blocked_until?: string | null;
  reversed_bias?: string | null;
  hostile_day_active?: boolean | null;
  hostile_day_reason?: string | null;
}

export interface FilterlessPosition {
  strategy_id: string;
  strategy_label: string;
  side: 'LONG' | 'SHORT' | string;
  entry_price?: number | null;
  avg_price?: number | null;
  signal_entry_price?: number | null;
  current_price?: number | null;
  tp_price?: number | null;
  sl_price?: number | null;
  stop_price?: number | null;
  target_price?: number | null;
  size?: number | null;
  order_id?: string | null;
  entry_order_id?: string | number | null;
  opened_at?: string | null;
  rule_id?: string | null;
  combo_key?: string | null;
  early_exit_enabled?: boolean | null;
  gate_prob?: number | null;
  gate_threshold?: number | null;
  kalshi_gate_applied?: boolean | null;
  kalshi_gate_reason?: string | null;
  kalshi_gate_multiplier?: number | null;
  kalshi_trade_overlay_applied?: boolean | null;
  kalshi_trade_overlay_reason?: string | null;
  kalshi_trade_overlay_role?: string | null;
  kalshi_trade_overlay_mode?: string | null;
  kalshi_trade_overlay_forward_weight?: number | null;
  kalshi_curve_informative?: boolean | null;
  kalshi_entry_probability?: number | null;
  kalshi_entry_support_score?: number | null;
  kalshi_entry_threshold?: number | null;
  kalshi_tp_anchor_price?: number | null;
  kalshi_tp_anchor_probability?: number | null;
  kalshi_tp_trail_enabled?: boolean | null;
  entry_mode?: string | null;
  base_session?: string | null;
  current_session?: string | null;
  sub_strategy?: string | null;
  vol_regime?: string | null;
  open_pnl_points?: number | null;
  open_pnl_dollars?: number | null;
}

export interface FilterlessStrategyState {
  id: string;
  label: string;
  status: string;
  updated_at?: string | null;
  last_signal_time?: string | null;
  last_signal_side?: string | null;
  last_signal_price?: number | null;
  tp_dist?: number | null;
  sl_dist?: number | null;
  priority?: string | null;
  last_reason?: string | null;
  last_block_reason?: string | null;
  latest_activity?: string | null;
  latest_activity_time?: string | null;
  latest_activity_type?: string | null;
  latest_activity_severity?: string | null;
  last_trade_pnl?: number | null;
  last_trade_points?: number | null;
  last_trade_time?: string | null;
  last_trade_side?: string | null;
  last_trade_entry?: number | null;
  last_trade_exit?: number | null;
  sub_strategy?: string | null;
  combo_key?: string | null;
  rule_id?: string | null;
  early_exit_enabled?: boolean | null;
  gate_prob?: number | null;
  gate_threshold?: number | null;
  kalshi_gate_applied?: boolean | null;
  kalshi_gate_reason?: string | null;
  kalshi_gate_multiplier?: number | null;
  kalshi_trade_overlay_applied?: boolean | null;
  kalshi_trade_overlay_reason?: string | null;
  kalshi_trade_overlay_role?: string | null;
  kalshi_trade_overlay_mode?: string | null;
  kalshi_trade_overlay_forward_weight?: number | null;
  kalshi_curve_informative?: boolean | null;
  kalshi_entry_probability?: number | null;
  kalshi_entry_support_score?: number | null;
  kalshi_entry_threshold?: number | null;
  kalshi_tp_anchor_price?: number | null;
  kalshi_tp_anchor_probability?: number | null;
  kalshi_tp_trail_enabled?: boolean | null;
  entry_mode?: string | null;
  base_session?: string | null;
  current_session?: string | null;
  vol_regime?: string | null;
}

export interface FilterlessEvent {
  time?: string | null;
  event_type: string;
  severity: string;
  strategy_id?: string | null;
  strategy_label?: string | null;
  message: string;
  details?: Record<string, unknown>;
}

export interface FilterlessTrade {
  time?: string | null;
  strategy_id: string;
  strategy_label: string;
  side: string;
  entry_price?: number | null;
  exit_price?: number | null;
  pnl_points?: number | null;
  pnl_dollars?: number | null;
  pnl_dollars_gross?: number | null;
  pnl_dollars_net?: number | null;
  pnl_fee_dollars?: number | null;
  size?: number | null;
  result?: string | null;
  kalshi_gate_applied?: boolean | null;
  kalshi_gate_reason?: string | null;
  kalshi_gate_multiplier?: number | null;
  kalshi_trade_overlay_applied?: boolean | null;
  kalshi_trade_overlay_reason?: string | null;
  kalshi_trade_overlay_role?: string | null;
  kalshi_trade_overlay_mode?: string | null;
  kalshi_entry_probability?: number | null;
  kalshi_entry_support_score?: number | null;
  kalshi_entry_threshold?: number | null;
  kalshi_tp_trail_enabled?: boolean | null;
}

export interface FilterlessPricePoint {
  time: string;
  price: number | null;
}

export interface FilterlessOhlcBar {
  /** ISO-8601 string of the bar's start (NY-local). */
  t: string;
  /** Open. */
  o: number;
  /** High. */
  h: number;
  /** Low. */
  l: number;
  /** Close. */
  c: number;
  /** Volume in contracts. */
  v: number;
}

export interface FilterlessKalshiStrike {
  strike: number;
  probability: number;
  volume?: number | null;
  status?: string | null;
  result?: string | null;
}

export interface FilterlessKalshiDailyContract {
  et_hour: number;
  strike_count: number;
  settled?: boolean | null;
}

export interface FilterlessKalshiMetrics {
  enabled: boolean;
  healthy?: boolean;
  requested?: boolean | null;
  configured?: boolean | null;
  observer_only?: boolean | null;
  status_label?: string | null;
  status_reason?: string | null;
  updated_at?: string | null;
  source?: string | null;
  basis_offset?: number | null;
  probability_60m?: number | null;
  probability_reference_kind?: string | null;
  probability_reference_side?: string | null;
  probability_reference_es_price?: number | null;
  probability_contract_es_price?: number | null;
  probability_contract_spx_price?: number | null;
  probability_contract_probability?: number | null;
  probability_contract_outcome?: string | null;
  probability_contract_distance_es?: number | null;
  event_ticker?: string | null;
  es_reference_price?: number | null;
  spx_reference_price?: number | null;
  strikes: FilterlessKalshiStrike[];
  trade_gating_active?: boolean | null;
  trade_gating_hour?: number | null;
  daily_contracts?: FilterlessKalshiDailyContract[] | null;
}

export interface FilterlessSentimentMetrics {
  enabled: boolean;
  active?: boolean | null;
  healthy?: boolean | null;
  model_loaded?: boolean | null;
  quantized_8bit?: boolean | null;
  target_handle?: string | null;
  source?: string | null;
  last_poll_at?: string | null;
  last_analysis_at?: string | null;
  latest_post_id?: string | null;
  latest_post_created_at?: string | null;
  latest_post_url?: string | null;
  latest_post_text?: string | null;
  sentiment_label?: string | null;
  sentiment_score?: number | null;
  finbert_confidence?: number | null;
  trigger_side?: string | null;
  trigger_reason?: string | null;
  last_error?: string | null;
  metadata?: Record<string, unknown> | null;
}

export interface FilterlessPipelineV18Stacker {
  enabled: boolean;
  threshold: number;
  bundle_path?: string | null;
}

export interface FilterlessPipelineKronos {
  available: boolean;
  daemon_running: boolean;
  daemon_restarts: number;
  timeout_s: number;
}

export interface FilterlessPipelineRecipeB {
  enabled: boolean;
  tiers: Array<[number, number]>;
  regime_aware_tier4: boolean;
  skip_whipsaw_tier4: boolean;
}

export interface FilterlessPipelineRegimeML {
  classifier_enabled: boolean;
  be_disable_ml: boolean;
  scalp_brackets_ml: boolean;
  size_reduction_ml: boolean;
}

export interface FilterlessPipelineNYAMBypass {
  enabled: boolean;
  subs: string[];
  hour_et: number;
}

export interface FilterlessPipelineSameSideML {
  enabled: boolean;
  max_contracts: number;
}

export interface FilterlessPipelineAFAllowlist {
  enabled: boolean;
  allowed_regimes: string[];
}

export interface FilterlessPipelineTriathlon {
  enabled: boolean;
}

export interface FilterlessPipelineFilterG {
  enabled: boolean;
}

export interface FilterlessPipelineState {
  v18_stacker: FilterlessPipelineV18Stacker;
  kronos: FilterlessPipelineKronos;
  recipe_b: FilterlessPipelineRecipeB;
  regime_ml: FilterlessPipelineRegimeML;
  ny_am_bypass: FilterlessPipelineNYAMBypass;
  sameside_ml: FilterlessPipelineSameSideML;
  af_regime_allowlist: FilterlessPipelineAFAllowlist;
  triathlon: FilterlessPipelineTriathlon;
  filter_g: FilterlessPipelineFilterG;
}

export interface FilterlessLiveState {
  schema_version: number;
  generated_at: string;
  meta: {
    log_path: string;
    state_path: string;
    trade_factors_path: string;
  };
  bot: {
    status: string;
    session?: string | null;
    price?: number | null;
    trading_day_start?: string | null;
    last_bar_time?: string | null;
    last_heartbeat_time?: string | null;
    heartbeat_age_seconds?: number | null;
    session_connection_ok?: boolean | null;
    last_position_sync_time?: string | null;
    position_sync_status?: string | null;
    current_position?: FilterlessPosition | null;
    current_positions?: FilterlessPosition[] | null;
    price_history: FilterlessPricePoint[];
    price_history_ohlc?: FilterlessOhlcBar[] | null;
    risk: FilterlessBotRisk;
    warnings: string[];
  };
  strategies: FilterlessStrategyState[];
  events: FilterlessEvent[];
  trades: FilterlessTrade[];
  kalshi_metrics?: FilterlessKalshiMetrics | null;
  sentiment_metrics?: FilterlessSentimentMetrics | null;
  pipeline?: FilterlessPipelineState | null;
}
