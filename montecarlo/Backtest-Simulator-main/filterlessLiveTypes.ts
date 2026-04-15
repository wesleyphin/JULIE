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
  opened_at?: string | null;
  rule_id?: string | null;
  combo_key?: string | null;
  early_exit_enabled?: boolean | null;
  gate_prob?: number | null;
  gate_threshold?: number | null;
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
  result?: string | null;
}

export interface FilterlessPricePoint {
  time: string;
  price: number | null;
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
  event_ticker: string;
  settled: boolean;
  strike_count: number;
}

export interface FilterlessKalshiMetrics {
  enabled: boolean;
  healthy?: boolean;
  requested?: boolean | null;
  configured?: boolean | null;
  observer_only?: boolean | null;
  status_label?: string | null;
  status_reason?: string | null;
  source?: string | null;
  updated_at?: string | null;
  basis_offset?: number | null;
  probability_60m?: number | null;
  event_ticker?: string | null;
  spx_reference_price?: number | null;
  trade_gating_active?: boolean | null;
  trade_gating_hour?: number | null;
  strikes: FilterlessKalshiStrike[];
  daily_contracts?: FilterlessKalshiDailyContract[] | null;
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
    price_history: FilterlessPricePoint[];
    risk: FilterlessBotRisk;
    warnings: string[];
  };
  strategies: FilterlessStrategyState[];
  events: FilterlessEvent[];
  trades: FilterlessTrade[];
  kalshi_metrics?: FilterlessKalshiMetrics | null;
}
