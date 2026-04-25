import React, { startTransition, useEffect, useMemo, useRef, useState } from 'react';
import {
  Activity,
  ArrowDownRight,
  ArrowUpRight,
  Bot,
  BrainCircuit,
  Clock3,
  Radar,
  RefreshCw,
  ShieldAlert,
  Waves,
  Wifi,
  WifiOff,
} from 'lucide-react';
import {
  Area,
  AreaChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import StatsCard from './components/StatsCard';
import {
  FilterlessEvent,
  FilterlessKalshiMetrics,
  FilterlessKalshiStrike,
  FilterlessLiveState,
  FilterlessPosition,
  FilterlessSentimentMetrics,
  FilterlessStrategyState,
  FilterlessTrade,
} from './filterlessLiveTypes';

const REFRESH_MS = 3000;
const FEED_STALE_SECONDS = 90;
const FEED_OFFLINE_SECONDS = 300;
const DEFAULT_SENTIMENT_METRICS: FilterlessSentimentMetrics = {
  enabled: true,
  active: false,
  healthy: false,
  model_loaded: false,
  quantized_8bit: false,
  target_handle: 'realDonaldTrump',
  source: 'rss_finbert',
  last_poll_at: null,
  last_analysis_at: null,
  latest_post_id: null,
  latest_post_created_at: null,
  latest_post_url: null,
  latest_post_text: null,
  sentiment_label: null,
  sentiment_score: null,
  finbert_confidence: null,
  trigger_side: null,
  trigger_reason: null,
  last_error: null,
  metadata: {
    gemini_enabled: false,
    gemini_configured: false,
    gemini_model: 'gemini-3-pro-preview',
    gemini_used: false,
  },
};

const EMPTY_STATE: FilterlessLiveState = {
  schema_version: 1,
  generated_at: new Date().toISOString(),
  meta: {
    log_path: '',
    state_path: '',
    trade_factors_path: '',
  },
  bot: {
    status: 'offline',
    session: null,
    price: null,
    last_bar_time: null,
    last_heartbeat_time: null,
    heartbeat_age_seconds: null,
    session_connection_ok: null,
    last_position_sync_time: null,
    position_sync_status: null,
    current_position: null,
    current_positions: [],
    price_history: [],
    risk: {},
    warnings: [],
  },
  strategies: [],
  events: [],
  trades: [],
  kalshi_metrics: null,
  sentiment_metrics: DEFAULT_SENTIMENT_METRICS,
};

function formatMoney(value?: number | null): string {
  if (value == null || Number.isNaN(value)) return '--';
  return `${value >= 0 ? '$' : '-$'}${Math.abs(value).toFixed(2)}`;
}

function formatPoints(value?: number | null): string {
  if (value == null || Number.isNaN(value)) return '--';
  return `${value >= 0 ? '+' : ''}${value.toFixed(2)} pts`;
}

function formatPrice(value?: number | null): string {
  if (value == null || Number.isNaN(value)) return '--';
  return value.toFixed(2);
}

function formatLots(value?: number | null): string {
  if (value == null || Number.isNaN(value)) return '--';
  const rounded = Number.isInteger(value) ? value : Number(value.toFixed(2));
  const absRounded = Math.abs(rounded);
  return `${rounded} lot${absRounded === 1 ? '' : 's'}`;
}

function formatLotCount(value?: number | null): string {
  if (value == null || Number.isNaN(value)) return '--';
  return Number.isInteger(value) ? String(value) : value.toFixed(2);
}

function formatTimestamp(value?: string | null): string {
  if (!value) return '--';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return '--';
  return date.toLocaleString([], {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

function formatShortTime(value?: string | null): string {
  if (!value) return '--';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return '--';
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function formatRelativeTime(value?: string | null): string {
  if (!value) return 'No activity';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return 'No activity';
  const diffSeconds = Math.max(0, Math.round((Date.now() - date.getTime()) / 1000));
  if (diffSeconds < 60) return `${diffSeconds}s ago`;
  if (diffSeconds < 3600) return `${Math.round(diffSeconds / 60)}m ago`;
  return `${Math.round(diffSeconds / 3600)}h ago`;
}

function computeFeedStatus(status: string, generatedAt?: string | null): string {
  const date = generatedAt ? new Date(generatedAt) : null;
  if (!date || Number.isNaN(date.getTime())) return status;
  const ageSeconds = Math.max(0, Math.round((Date.now() - date.getTime()) / 1000));
  if (ageSeconds >= FEED_OFFLINE_SECONDS) return 'offline';
  if (ageSeconds >= FEED_STALE_SECONDS) return 'stale';
  return status;
}

function statusChipClasses(status: string): string {
  switch (status) {
    case 'online':
    case 'ready':
    case 'idle':
      return 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30';
    case 'candidate':
    case 'queued':
      return 'bg-sky-500/10 text-sky-400 border-sky-500/30';
    case 'in_trade':
      return 'bg-amber-500/10 text-amber-400 border-amber-500/30';
    case 'disabled':
      return 'bg-neutral-700/40 text-neutral-200 border-neutral-500/40';
    case 'blocked':
    case 'stale':
    case 'offline':
      return 'bg-rose-500/10 text-rose-400 border-rose-500/30';
    default:
      return 'bg-neutral-800 text-neutral-300 border-neutral-700';
  }
}

function eventTone(severity: string): string {
  switch (severity) {
    case 'error':
    case 'danger':
      return 'border-rose-500/30 bg-rose-500/5';
    case 'warning':
      return 'border-amber-500/30 bg-amber-500/5';
    case 'success':
      return 'border-emerald-500/30 bg-emerald-500/5';
    default:
      return 'border-neutral-800 bg-neutral-900/60';
  }
}

function tradeTone(trade: FilterlessTrade): string {
  const pnl = trade.pnl_dollars_net ?? trade.pnl_dollars ?? null;
  if (pnl != null && !Number.isNaN(pnl)) {
    if (pnl > 0) return 'bg-emerald-500/5';
    if (pnl < 0) return 'bg-rose-500/5';
  }

  const result = String(trade.result || '').toLowerCase();
  if (result.includes('win') || result.includes('target')) return 'bg-emerald-500/5';
  if (result.includes('loss') || result.includes('stop')) return 'bg-rose-500/5';
  return 'bg-neutral-950/40';
}

function tradeSideClasses(side?: string | null): string {
  if (side === 'LONG') return 'bg-emerald-500/10 text-emerald-300 border-emerald-500/30';
  if (side === 'SHORT') return 'bg-rose-500/10 text-rose-300 border-rose-500/30';
  return 'bg-neutral-800 text-neutral-300 border-neutral-700';
}

function strategyTone(strategyId: string): { shell: string; badge: string; accent: string } {
  switch (strategyId) {
    case 'dynamic_engine3':
      return {
        shell: 'border-sky-500/20 bg-gradient-to-b from-sky-500/10 via-surface to-surface',
        badge: 'bg-sky-500/10 text-sky-300 border-sky-500/30',
        accent: 'text-sky-300',
      };
    case 'regime_adaptive':
      return {
        shell: 'border-amber-500/20 bg-gradient-to-b from-amber-500/10 via-surface to-surface',
        badge: 'bg-amber-500/10 text-amber-200 border-amber-500/30',
        accent: 'text-amber-200',
      };
    case 'aetherflow':
      return {
        shell: 'border-fuchsia-500/20 bg-gradient-to-b from-fuchsia-500/10 via-surface to-surface',
        badge: 'bg-fuchsia-500/10 text-fuchsia-200 border-fuchsia-500/30',
        accent: 'text-fuchsia-200',
      };
    case 'ml_physics':
      return {
        shell: 'border-emerald-500/20 bg-gradient-to-b from-emerald-500/10 via-surface to-surface',
        badge: 'bg-emerald-500/10 text-emerald-300 border-emerald-500/30',
        accent: 'text-emerald-300',
      };
    default:
      return {
        shell: 'border-neutral-800 bg-surface',
        badge: 'bg-neutral-800 text-neutral-300 border-neutral-700',
        accent: 'text-neutral-200',
      };
  }
}

function formatToken(value?: string | null): string {
  if (!value) return '--';
  return value.replace(/_/g, ' ');
}

function formatBooleanLabel(value?: boolean | null, trueLabel = 'On', falseLabel = 'Off'): string {
  if (value == null) return '--';
  return value ? trueLabel : falseLabel;
}

function formatPercent(value?: number | null, digits = 1): string {
  if (value == null || Number.isNaN(value)) return '--';
  return `${(value * 100).toFixed(digits)}%`;
}

function formatStrikeLabel(value?: number | null): string {
  if (value == null || Number.isNaN(value)) return '--';
  return value.toFixed(0);
}

function formatCompactNumber(value?: number | null): string {
  if (value == null || Number.isNaN(value)) return '--';
  return new Intl.NumberFormat([], {
    notation: 'compact',
    maximumFractionDigits: value >= 1000 ? 1 : 0,
  }).format(value);
}

function metadataBoolean(metadata: Record<string, unknown> | null | undefined, key: string): boolean {
  return metadata?.[key] === true;
}

function metadataNumber(metadata: Record<string, unknown> | null | undefined, key: string): number | null {
  const value = metadata?.[key];
  return typeof value === 'number' && !Number.isNaN(value) ? value : null;
}

function metadataString(metadata: Record<string, unknown> | null | undefined, key: string): string | null {
  const value = metadata?.[key];
  return typeof value === 'string' && value.trim() ? value.trim() : null;
}

function sentimentScoreClasses(metrics?: FilterlessSentimentMetrics | null): string {
  const score = metrics?.sentiment_score;
  if (score == null || Number.isNaN(score)) return 'text-neutral-100';
  if (score >= 0.25) return 'text-emerald-300';
  if (score <= -0.25) return 'text-rose-300';
  return 'text-neutral-100';
}

function formatGateSummary(prob?: number | null, threshold?: number | null): string {
  if (prob == null && threshold == null) return '--';
  if (prob == null) return `min ${formatPercent(threshold)}`;
  if (threshold == null) return formatPercent(prob);
  return `${formatPercent(prob)} >= ${formatPercent(threshold)}`;
}

function kalshiModeLabel(metrics?: FilterlessKalshiMetrics | null): string {
  if (!metrics) return 'Unavailable';
  if (metrics.observer_only) return 'Observe only';
  if (metrics.enabled) return 'Trading + dashboard';
  if (metrics.configured) return 'Disabled';
  return 'Not configured';
}

function kalshiExecutionLabel(metrics?: FilterlessKalshiMetrics | null): string {
  if (!metrics) return '--';
  if (metrics.trade_gating_active) return 'Live gating';
  if (metrics.observer_only) return 'Observe-only';
  if (metrics.enabled) return 'Armed';
  if (metrics.configured) return 'Configured';
  return 'Unavailable';
}

function formatKalshiGateSummary(
  applied?: boolean | null,
  multiplier?: number | null,
  reason?: string | null,
): string {
  const parts: string[] = [];
  if (multiplier != null && !Number.isNaN(multiplier) && multiplier !== 1) {
    parts.push(`${multiplier.toFixed(multiplier % 1 === 0 ? 0 : 1)}x`);
  }
  const normalizedReason = String(reason || '').trim();
  if (normalizedReason) {
    parts.push(normalizedReason);
  } else if (applied) {
    parts.push('Applied');
  }
  return parts.length > 0 ? parts.join(' · ') : '--';
}

function kalshiStatusClasses(metrics?: FilterlessKalshiMetrics | null): string {
  if (metrics?.healthy && (metrics.strikes?.length || 0) > 0) {
    return 'bg-emerald-500/10 text-emerald-300 border-emerald-500/30';
  }
  if (metrics?.configured) {
    return 'bg-amber-500/10 text-amber-200 border-amber-500/30';
  }
  return 'bg-neutral-800 text-neutral-300 border-neutral-700';
}

interface KalshiHourlyContract {
  hour: number;
  label: string;
  eventTicker: string;
  available: boolean;
  settled: boolean;
  upcoming: boolean;
  strikeCount: number;
  isActive: boolean;
  tradeGating: boolean;
}

const KALSHI_SETTLEMENT_HOURS = [10, 11, 12, 13, 14, 15, 16] as const;
const KALSHI_GATING_HOURS: readonly number[] = [12, 13, 14, 15, 16];

function tzHourToLocalLabel(hour: number, timeZone: string): string {
  const now = new Date();
  const parts = new Intl.DateTimeFormat('en-US', {
    timeZone,
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
  }).formatToParts(now);
  const yy = parts.find((p) => p.type === 'year')!.value;
  const mm = parts.find((p) => p.type === 'month')!.value;
  const dd = parts.find((p) => p.type === 'day')!.value;
  const utcMs = new Date(now.toLocaleString('en-US', { timeZone: 'UTC' })).getTime();
  const tzMs = new Date(now.toLocaleString('en-US', { timeZone })).getTime();
  const tzOffsetHours = Math.round((tzMs - utcMs) / 3_600_000);
  const utcHour = hour - tzOffsetHours;
  const target = new Date(`${yy}-${mm}-${dd}T${String(utcHour).padStart(2, '0')}:00:00Z`);
  return target.toLocaleTimeString([], { hour: 'numeric', hour12: true });
}

function etHourToLocalLabel(etHour: number): string {
  return tzHourToLocalLabel(etHour, 'America/New_York');
}

function currentETMinuteOfDay(): number {
  const now = new Date();
  const parts = new Intl.DateTimeFormat('en-US', {
    timeZone: 'America/New_York',
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
  }).formatToParts(now);
  const hour = Number(parts.find((part) => part.type === 'hour')?.value || '0');
  const minute = Number(parts.find((part) => part.type === 'minute')?.value || '0');
  return (hour * 60) + minute;
}

function buildKalshiHourlyContracts(
  eventTicker?: string | null,
  strikes?: FilterlessKalshiStrike[],
  tradeGatingHour?: number | null,
  dailyContracts?: { et_hour: number; strike_count: number; settled?: boolean | null }[] | null,
): KalshiHourlyContract[] {
  const tickerPrefix = eventTicker?.replace(/H\d+$/, '') || null;
  const activeHourMatch = eventTicker?.match(/H(\d+)$/);
  const activeHourCode = activeHourMatch ? parseInt(activeHourMatch[1], 10) : null;
  const etMinuteOfDay = currentETMinuteOfDay();
  const renderableHours = new Set<number>();

  if (dailyContracts) {
    for (const dailyContract of dailyContracts) {
      if (dailyContract.strike_count > 0) {
        renderableHours.add(dailyContract.et_hour);
      }
    }
  }

  return KALSHI_SETTLEMENT_HOURS.map((hour) => {
    const hourCode = hour * 100;
    const ticker = tickerPrefix ? `${tickerPrefix}H${hourCode}` : `H${hourCode}`;
    const isActive = activeHourCode === hourCode;
    const isSettled = etMinuteOfDay >= ((hour * 60) + 5);
    const hasData = renderableHours.has(hour) || isActive;
    const isAvailable = !isSettled;
    const isUpcoming = !isSettled && !isAvailable && !hasData;
    const strikeCount = isActive ? (strikes?.length || 0) : 0;
    const tradeGating = tradeGatingHour === hour;

    return {
      hour,
      label: etHourToLocalLabel(hour),
      eventTicker: ticker,
      available: isAvailable,
      settled: isSettled,
      upcoming: isUpcoming,
      strikeCount,
      isActive,
      tradeGating,
    };
  });
}

const KALSHI_STRIKE_WINDOW_SIZE = 30;

function pickKalshiWindow(
  strikes: FilterlessKalshiStrike[],
  referencePrice?: number | null,
  windowSize = KALSHI_STRIKE_WINDOW_SIZE,
): FilterlessKalshiStrike[] {
  if (strikes.length <= windowSize) return strikes;
  if (referencePrice == null || Number.isNaN(referencePrice)) {
    const midpoint = Math.floor(strikes.length / 2);
    const start = Math.max(0, midpoint - Math.floor(windowSize / 2));
    return strikes.slice(start, start + windowSize);
  }

  let nearestIndex = 0;
  let bestDistance = Number.POSITIVE_INFINITY;
  strikes.forEach((strike, index) => {
    const distance = Math.abs(strike.strike - referencePrice);
    if (distance < bestDistance) {
      bestDistance = distance;
      nearestIndex = index;
    }
  });
  const halfWindow = Math.floor(windowSize / 2);
  const start = Math.max(0, nearestIndex - halfWindow);
  const end = Math.min(strikes.length, start + windowSize);
  return strikes.slice(Math.max(0, end - windowSize), end);
}

function getPositionEntryPrice(position: FilterlessPosition): number | null {
  return position.entry_price ?? position.avg_price ?? position.signal_entry_price ?? null;
}

function getPositionCurrentPrice(position: FilterlessPosition, fallbackCurrentPrice?: number | null): number | null {
  if (position.current_price != null && !Number.isNaN(position.current_price)) {
    return position.current_price;
  }
  const entry = getPositionEntryPrice(position);
  const openPoints = position.open_pnl_points;
  if (entry != null && openPoints != null && !Number.isNaN(openPoints)) {
    if (position.side === 'LONG') return entry + openPoints;
    if (position.side === 'SHORT') return entry - openPoints;
  }
  return fallbackCurrentPrice ?? null;
}

const Panel: React.FC<{ title: string; right?: React.ReactNode; children: React.ReactNode; className?: string }> = ({
  title,
  right,
  children,
  className,
}) => (
  <section className={`bg-surface rounded-xl border border-neutral-800 shadow-lg ${className || ''}`}>
    <div className="flex items-center justify-between border-b border-neutral-800 px-5 py-4">
      <h3 className="text-sm font-semibold tracking-wide text-neutral-200 uppercase">{title}</h3>
      {right}
    </div>
    <div className="p-5">{children}</div>
  </section>
);

const MonitorMetaItem: React.FC<{ label: string; value: React.ReactNode; className?: string }> = ({
  label,
  value,
  className,
}) => (
  <div className={`rounded-lg border border-neutral-700 bg-neutral-950/88 px-3 py-2.5 ${className || ''}`}>
    <p className="text-[10px] font-semibold uppercase tracking-[0.18em] text-neutral-500">{label}</p>
    <p className="mt-1 text-xs text-neutral-300">{value}</p>
  </div>
);

const MonitorMetric: React.FC<{
  label: string;
  value: React.ReactNode;
  hint?: React.ReactNode;
  valueClassName?: string;
  className?: string;
}> = ({
  label,
  value,
  hint,
  valueClassName,
  className,
}) => (
  <div className={`rounded-xl border border-neutral-700 bg-neutral-950/78 px-4 py-4 shadow-[inset_0_1px_0_rgba(255,255,255,0.03)] ${className || ''}`}>
    <p className="text-[10px] font-semibold uppercase tracking-[0.18em] text-neutral-500">{label}</p>
    <div className={`mt-2 text-lg font-semibold ${valueClassName || 'text-neutral-100'}`}>{value}</div>
    {hint ? <p className="mt-2 text-xs leading-5 text-neutral-500">{hint}</p> : null}
  </div>
);

const MonitorSection: React.FC<{
  title: string;
  subtitle?: React.ReactNode;
  right?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
}> = ({
  title,
  subtitle,
  right,
  children,
  className,
}) => (
  <div className={`rounded-xl border border-neutral-700 bg-neutral-950/82 p-4 shadow-[inset_0_1px_0_rgba(255,255,255,0.03)] ${className || ''}`}>
    <div className="flex flex-wrap items-start justify-between gap-3 border-b border-neutral-800 pb-3">
      <div>
        <p className="text-[10px] font-semibold uppercase tracking-[0.18em] text-neutral-500">{title}</p>
        {subtitle ? <p className="mt-1 text-sm text-neutral-300">{subtitle}</p> : null}
      </div>
      {right}
    </div>
    <div className="mt-4">{children}</div>
  </div>
);

const StrategyCard: React.FC<{ strategy: FilterlessStrategyState }> = ({ strategy }) => {
  const tone = strategyTone(strategy.id);
  const tradeColor =
    strategy.last_trade_pnl == null
      ? 'text-neutral-400'
      : strategy.last_trade_pnl >= 0
        ? 'text-emerald-400'
        : 'text-rose-400';
  const gateActive = strategy.gate_prob != null || strategy.gate_threshold != null;
  const contextLabel = strategy.combo_key || strategy.sub_strategy;
  const executionLabel = strategy.rule_id || strategy.entry_mode;
  const sessionLabel = strategy.current_session || strategy.base_session;
  const latestActivity = strategy.latest_activity || strategy.last_block_reason || strategy.last_reason || 'No recent strategy activity.';
  const latestActivityTime = strategy.latest_activity_time || strategy.updated_at;

  return (
    <div className={`rounded-xl border p-5 shadow-sm transition-colors hover:border-neutral-600 ${tone.shell}`}>
      <div className="flex items-start justify-between gap-4 mb-4">
        <div>
          <div className="flex items-center gap-2">
            <p className="text-lg font-semibold text-neutral-100">{strategy.label}</p>
            <span className={`rounded-full border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.18em] ${tone.badge}`}>
              Live
            </span>
          </div>
          <p className="text-xs text-neutral-500 mt-1">{formatRelativeTime(strategy.updated_at)}</p>
        </div>
        <span className={`rounded-full border px-3 py-1 text-xs font-semibold uppercase tracking-wide ${statusChipClasses(strategy.status)}`}>
          {strategy.status.replace('_', ' ')}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-2 mb-4 xl:grid-cols-4">
        <div className="rounded-lg border border-neutral-800/80 bg-neutral-950/60 px-3 py-2">
          <p className="text-[10px] font-semibold uppercase tracking-[0.18em] text-neutral-500">Context</p>
          <p className={`mt-1 text-xs font-medium ${tone.accent}`}>{formatToken(contextLabel)}</p>
        </div>
        <div className="rounded-lg border border-neutral-800/80 bg-neutral-950/60 px-3 py-2">
          <p className="text-[10px] font-semibold uppercase tracking-[0.18em] text-neutral-500">Rule</p>
          <p className="mt-1 text-xs font-medium text-neutral-200">{formatToken(executionLabel)}</p>
        </div>
        <div className="rounded-lg border border-neutral-800/80 bg-neutral-950/60 px-3 py-2">
          <p className="text-[10px] font-semibold uppercase tracking-[0.18em] text-neutral-500">Early Exit</p>
          <p className="mt-1 text-xs font-medium text-neutral-200">
            {formatBooleanLabel(strategy.early_exit_enabled, 'Enabled', 'Disabled')}
          </p>
        </div>
        <div
          className={`rounded-lg border px-3 py-2 ${
            gateActive ? 'border-amber-500/25 bg-amber-500/5' : 'border-neutral-800/80 bg-neutral-950/60'
          }`}
        >
          <p className="text-[10px] font-semibold uppercase tracking-[0.18em] text-neutral-500">Gate</p>
          <p className={`mt-1 text-xs font-medium ${gateActive ? 'text-amber-200' : 'text-neutral-200'}`}>
            {formatGateSummary(strategy.gate_prob, strategy.gate_threshold)}
          </p>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3 text-sm mb-4">
        <div>
          <p className="text-neutral-500">Side</p>
          <p className="text-neutral-100 font-medium">{strategy.last_signal_side || '--'}</p>
        </div>
        <div>
          <p className="text-neutral-500">Priority</p>
          <p className="text-neutral-100 font-medium">{strategy.priority || '--'}</p>
        </div>
        <div>
          <p className="text-neutral-500">Signal Price</p>
          <p className="text-neutral-100 font-medium">{formatPrice(strategy.last_signal_price)}</p>
        </div>
        <div>
          <p className="text-neutral-500">Bracket</p>
          <p className="text-neutral-100 font-medium">
            {strategy.tp_dist != null || strategy.sl_dist != null
              ? `${formatPrice(strategy.tp_dist)} / ${formatPrice(strategy.sl_dist)}`
              : '--'}
          </p>
        </div>
      </div>

      <div className="rounded-lg border border-neutral-800 bg-neutral-950/70 p-3 mb-4 min-h-[84px]">
        <div className="mb-2 flex items-center justify-between gap-3">
          <p className="text-xs font-semibold uppercase tracking-wide text-neutral-500">Latest Activity</p>
          <p className="text-[11px] text-neutral-500">{formatRelativeTime(latestActivityTime)}</p>
        </div>
        <p className="text-sm text-neutral-300 leading-6">{latestActivity}</p>
      </div>

      <div className="grid grid-cols-2 gap-3 text-sm">
        <div>
          <p className="text-neutral-500">Last Trade</p>
          <p className={`font-semibold ${tradeColor}`}>{formatMoney(strategy.last_trade_pnl)}</p>
          <p className="text-xs text-neutral-500 mt-1">{formatTimestamp(strategy.last_trade_time)}</p>
        </div>
        <div>
          <p className="text-neutral-500">Session</p>
          <p className="text-neutral-100 font-medium">{sessionLabel || '--'}</p>
          <p className="text-xs text-neutral-500 mt-1">
            {strategy.vol_regime ? `${formatToken(strategy.vol_regime)} regime` : formatToken(strategy.entry_mode)}
          </p>
        </div>
      </div>
    </div>
  );
};

function FilterlessLiveApp() {
  const [state, setState] = useState<FilterlessLiveState>(EMPTY_STATE);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedKalshiHour, setSelectedKalshiHour] = useState<number | null>(null);
  const inFlightRef = useRef(false);
  const abortRef = useRef<AbortController | null>(null);
  const lastGeneratedAtRef = useRef<string | null>(null);
  const lastGoodSentimentRef = useRef<FilterlessSentimentMetrics>(DEFAULT_SENTIMENT_METRICS);

  useEffect(() => {
    let cancelled = false;

    const loadState = async () => {
      if (inFlightRef.current) return;
      if (document.visibilityState === 'hidden') return;
      const controller = new AbortController();
      abortRef.current?.abort();
      abortRef.current = controller;
      inFlightRef.current = true;
      try {
        const response = await fetch(`/filterless_live_state.json?ts=${Date.now()}`, {
          cache: 'no-store',
          signal: controller.signal,
        });
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        const payload = (await response.json()) as FilterlessLiveState;
        if (payload.generated_at && payload.generated_at === lastGeneratedAtRef.current) {
          if (!cancelled) {
            startTransition(() => {
              setError(null);
              setLoading(false);
            });
          }
          return;
        }
        lastGeneratedAtRef.current = payload.generated_at || null;
        if (!cancelled) {
          startTransition(() => {
            setState(payload);
            setError(null);
            setLoading(false);
          });
        }
      } catch (err) {
        const isAbortError =
          (err instanceof DOMException && err.name === 'AbortError') ||
          (err instanceof Error && err.name === 'AbortError');
        const message = err instanceof Error ? err.message : String(err);
        if (!cancelled && !isAbortError) {
          startTransition(() => {
            setError(message);
            setLoading(false);
          });
        }
      } finally {
        inFlightRef.current = false;
      }
    };

    loadState();
    const timer = window.setInterval(loadState, REFRESH_MS);
    const handleVisibilityChange = () => {
      if (document.visibilityState === 'visible') {
        void loadState();
      }
    };
    document.addEventListener('visibilitychange', handleVisibilityChange);
    return () => {
      cancelled = true;
      abortRef.current?.abort();
      window.clearInterval(timer);
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, []);

  const tradingDayStartValue = state.bot.trading_day_start;
  const tradingDayStart = tradingDayStartValue ? new Date(tradingDayStartValue) : null;
  const intradayAnchorLabel = tradingDayStart && !Number.isNaN(tradingDayStart.getTime())
    ? `since ${formatTimestamp(tradingDayStart.toISOString())}`
    : 'current futures session';

  const priceData = useMemo(
    () => state.bot.price_history.map((point) => ({ ...point, label: formatShortTime(point.time) })),
    [state.bot.price_history],
  );

  const intradayPnlData = useMemo(() => {
    const sessionStartMs = tradingDayStart && !Number.isNaN(tradingDayStart.getTime()) ? tradingDayStart.getTime() : null;
    let cumulative = 0;
    return state.trades
      .slice()
      .reverse()
      .filter((trade) => {
        if (!trade.time) return false;
        const tradeTime = new Date(trade.time);
        if (Number.isNaN(tradeTime.getTime())) return false;
        if (sessionStartMs == null) return true;
        return tradeTime.getTime() >= sessionStartMs;
      })
      .map((trade) => {
        cumulative += trade.pnl_dollars || 0;
        return {
          time: trade.time || '',
          label: formatShortTime(trade.time),
          pnl: cumulative,
        };
      });
  }, [state.trades, tradingDayStartValue]);

  const effectiveBotStatus = computeFeedStatus(state.bot.status, state.generated_at);
  const feedWarnings = useMemo(() => {
    const warnings = [...state.bot.warnings];
    if (effectiveBotStatus !== state.bot.status) {
      warnings.unshift(`Dashboard feed has stopped refreshing. Last successful update was ${formatRelativeTime(state.generated_at)}.`);
    }
    return warnings;
  }, [effectiveBotStatus, state.bot.status, state.bot.warnings, state.generated_at]);
  const displayStrategies = useMemo(
    () =>
      state.strategies
        .filter((strategy) => strategy.id !== 'truth_social')
        .map((strategy) => (
        effectiveBotStatus === 'online' ? strategy : { ...strategy, status: effectiveBotStatus }
      )),
    [effectiveBotStatus, state.strategies],
  );
  const openPositions = useMemo(() => {
    if (effectiveBotStatus !== 'online') return [];
    const botPositions = Array.isArray(state.bot.current_positions)
      ? state.bot.current_positions.filter((position): position is FilterlessPosition => position != null)
      : [];
    if (botPositions.length > 0) {
      return botPositions;
    }
    return state.bot.current_position ? [state.bot.current_position] : [];
  }, [effectiveBotStatus, state.bot.current_position, state.bot.current_positions]);
  const primaryOpenPosition = openPositions[0] ?? null;
  const openPositionCount = openPositions.length;
  const openPositionSummarySide = useMemo(() => {
    if (openPositions.length === 0) return null;
    const normalizedSides = Array.from(
      new Set(
        openPositions
          .map((position) => String(position.side || '').toUpperCase())
          .filter((side) => side === 'LONG' || side === 'SHORT'),
      ),
    );
    if (normalizedSides.length === 1) {
      return normalizedSides[0];
    }
    return 'MIXED';
  }, [openPositions]);
  const totalOpenLots = useMemo(
    () => openPositions.reduce((total, position) => total + (position.size ?? 0), 0),
    [openPositions],
  );
  const totalOpenPnlDollars = useMemo(
    () => openPositions.reduce((total, position) => total + (position.open_pnl_dollars ?? 0), 0),
    [openPositions],
  );
  const heartbeatOk = effectiveBotStatus === 'online';
  const botStatusColor = heartbeatOk ? 'success' : effectiveBotStatus === 'stale' ? 'warning' : 'danger';
  const dailyPnlColor = (state.bot.risk.daily_pnl || 0) >= 0 ? 'success' : 'danger';
  const openPnlColor = totalOpenPnlDollars >= 0 ? 'success' : 'danger';
  const kalshiMetrics = state.kalshi_metrics ?? null;
  const kalshiVisible = kalshiMetrics != null;
  const sentimentMetrics = useMemo(() => {
    const incoming = state.sentiment_metrics;
    if (incoming && incoming.sentiment_score != null) {
      lastGoodSentimentRef.current = incoming;
      return incoming;
    }
    if (incoming) {
      return {
        ...lastGoodSentimentRef.current,
        ...incoming,
        metadata: {
          ...(lastGoodSentimentRef.current.metadata || {}),
          ...(incoming.metadata || {}),
        },
      };
    }
    return lastGoodSentimentRef.current;
  }, [state.sentiment_metrics]);
  const kalshiOpenSide = String(primaryOpenPosition?.side || '').toUpperCase();
  const kalshiPredictionLabel = kalshiOpenSide === 'SHORT' ? 'NO' : 'YES';
  const kalshiSpxReferencePrice =
    kalshiMetrics?.spx_reference_price ??
    ((state.bot.price != null && kalshiMetrics?.basis_offset != null)
      ? state.bot.price - kalshiMetrics.basis_offset
      : null);
  const kalshiEsReferencePrice =
    kalshiMetrics?.es_reference_price ??
    ((kalshiSpxReferencePrice != null && kalshiMetrics?.basis_offset != null)
      ? kalshiSpxReferencePrice + kalshiMetrics.basis_offset
      : state.bot.price ?? null);
  const kalshiStrikes = useMemo(
    () => {
      const rows = (kalshiMetrics?.strikes || [])
        .filter((row): row is FilterlessKalshiStrike => row != null && row.strike != null && row.probability != null)
        .sort((a, b) => a.strike - b.strike);
      return pickKalshiWindow(rows, kalshiSpxReferencePrice, KALSHI_STRIKE_WINDOW_SIZE);
    },
    [kalshiMetrics, kalshiSpxReferencePrice],
  );
  const nearestKalshiStrike = useMemo(() => {
    if (kalshiSpxReferencePrice == null || kalshiStrikes.length === 0) return null;
    return kalshiStrikes.reduce((best, strike) => (
      Math.abs(strike.strike - kalshiSpxReferencePrice) < Math.abs(best.strike - kalshiSpxReferencePrice) ? strike : best
    ));
  }, [kalshiSpxReferencePrice, kalshiStrikes]);
  const kalshiHourlyContracts = useMemo(
    () => buildKalshiHourlyContracts(kalshiMetrics?.event_ticker, kalshiStrikes, kalshiMetrics?.trade_gating_hour, kalshiMetrics?.daily_contracts || null),
    [kalshiMetrics?.event_ticker, kalshiMetrics?.trade_gating_hour, kalshiMetrics?.daily_contracts, kalshiStrikes],
  );
  const kalshiProbabilityCaption = useMemo(() => {
    const referenceKind = kalshiMetrics?.probability_reference_kind;
    const referenceEsPrice = kalshiMetrics?.probability_reference_es_price;
    const contractEsPrice = kalshiMetrics?.probability_contract_es_price;
    const contractOutcome = kalshiMetrics?.probability_contract_outcome;
    const referenceSide = kalshiMetrics?.probability_reference_side;

    if (referenceKind === 'open_position_target' && referenceEsPrice != null && contractEsPrice != null) {
      const tpLabel = referenceSide ? `${referenceSide} TP` : 'Open TP';
      const outcomeLabel = contractOutcome === 'below' ? 'NO' : 'YES';
      return `${tpLabel} ${formatPrice(referenceEsPrice)} -> ${outcomeLabel} @ ${formatPrice(contractEsPrice)} ES contract`;
    }
    if (referenceEsPrice != null) {
      return `Using current ES price ${formatPrice(referenceEsPrice)}`;
    }
    return kalshiMetrics?.healthy ? 'Event ladder is reachable.' : 'Waiting for a healthy snapshot.';
  }, [
    kalshiMetrics?.healthy,
    kalshiMetrics?.probability_contract_es_price,
    kalshiMetrics?.probability_contract_outcome,
    kalshiMetrics?.probability_reference_es_price,
    kalshiMetrics?.probability_reference_kind,
    kalshiMetrics?.probability_reference_side,
  ]);
  const kalshiDisplayedStrikes = useMemo(
    () => kalshiStrikes.map((strike) => ({
      ...strike,
      displayProbability: kalshiOpenSide === 'SHORT'
        ? Math.max(0, Math.min(1, 1 - strike.probability))
        : strike.probability,
    })),
    [kalshiOpenSide, kalshiStrikes],
  );
  const viewedKalshiContract = useMemo(() => {
    if (selectedKalshiHour != null) {
      return kalshiHourlyContracts.find((contract) => contract.hour === selectedKalshiHour) ?? null;
    }
    return kalshiHourlyContracts.find((contract) => contract.isActive) ?? null;
  }, [kalshiHourlyContracts, selectedKalshiHour]);
  const kalshiMode = kalshiModeLabel(kalshiMetrics);
  const kalshiAvailableContracts = useMemo(
    () => kalshiHourlyContracts.filter((contract) => contract.available).length,
    [kalshiHourlyContracts],
  );
  const sentimentExcerpt = useMemo(() => {
    const raw = String(sentimentMetrics?.latest_post_text || '').trim();
    if (!raw) return '';
    return raw.length > 800 ? `${raw.slice(0, 797)}...` : raw;
  }, [sentimentMetrics?.latest_post_text]);
  const sentimentMetadata = useMemo(
    () => ((sentimentMetrics?.metadata ?? null) as Record<string, unknown> | null),
    [sentimentMetrics?.metadata],
  );
  const geminiEnabled = metadataBoolean(sentimentMetadata, 'gemini_enabled');
  const geminiConfigured = metadataBoolean(sentimentMetadata, 'gemini_configured');
  const geminiUsed = metadataBoolean(sentimentMetadata, 'gemini_used');
  const geminiModel = metadataString(sentimentMetadata, 'gemini_model') || 'gemini-3-pro-preview';
  const geminiConfidence = metadataNumber(sentimentMetadata, 'gemini_confidence');
  const geminiScore = metadataNumber(sentimentMetadata, 'gemini_score');
  const geminiMarketImpact = metadataString(sentimentMetadata, 'gemini_market_impact');
  const geminiReasoning = metadataString(sentimentMetadata, 'gemini_reasoning');
  const geminiOverrideSource = metadataString(sentimentMetadata, 'override_source');
  const geminiOverrideActive = geminiOverrideSource === 'gemini_geopolitical';
  const geminiUsageLabel = geminiEnabled
    ? (geminiUsed ? (geminiOverrideActive ? 'Override' : 'Used') : 'Standby')
    : (geminiConfigured ? 'Configured' : 'Off');
  const geminiVerdict = useMemo(() => {
    if (!geminiEnabled || !geminiUsed) return null;
    if (geminiOverrideActive) {
      return 'Gemini reviewed this post, found material market impact, and overrode the base FinBERT read.';
    }
    if (geminiMarketImpact === 'low') {
      return 'Gemini reviewed this post and found low market impact, so JULIE kept the ES bias neutral.';
    }
    if (geminiMarketImpact === 'medium' || geminiMarketImpact === 'high') {
      return `Gemini reviewed this post and rated its market impact ${geminiMarketImpact.toUpperCase()}, but FinBERT remained the active signal.`;
    }
    return 'Gemini reviewed this post, but it did not override the primary FinBERT signal.';
  }, [geminiEnabled, geminiMarketImpact, geminiOverrideActive, geminiUsed]);
  const sentimentPanelStatus = sentimentMetrics.last_error
    ? 'blocked'
    : sentimentMetrics.trigger_side && sentimentMetrics.trigger_side !== 'NEUTRAL'
      ? 'candidate'
      : sentimentMetrics.enabled && sentimentMetrics.healthy
        ? 'ready'
        : 'idle';
  const sentimentPanelStatusLabel = sentimentMetrics.last_error
    ? 'Issue'
    : sentimentMetrics.trigger_side && sentimentMetrics.trigger_side !== 'NEUTRAL'
      ? `${sentimentMetrics.trigger_side} Watch`
      : sentimentMetrics.trigger_side === 'NEUTRAL'
        ? 'Neutral'
        : sentimentMetrics.enabled && sentimentMetrics.healthy
          ? 'Monitoring'
          : 'Idle';

  return (
    <div className="min-h-screen bg-background text-neutral-100 pb-16">
      <header className="sticky top-0 z-50 border-b border-neutral-800 bg-surface/70 backdrop-blur-sm">
        <div className="max-w-[1600px] mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Radar className="w-6 h-6 text-white" />
            <div>
              <h1 className="text-xl font-bold tracking-tight">Filterless Live Desk</h1>
              <p className="text-xs text-neutral-500">Dynamic Engine 3, RegimeAdaptive, ML Physics, AetherFlow</p>
            </div>
          </div>
          <div className="flex items-center gap-4 text-sm">
            <a href="/" className="text-neutral-400 hover:text-white transition-colors">Monte Carlo</a>
            <div className="flex items-center gap-2 text-neutral-400">
              <RefreshCw className="w-4 h-4" />
              <span>{formatRelativeTime(state.generated_at)}</span>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-[1600px] mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
        <div className="flex items-center gap-2 border-b border-neutral-800 pb-2">
          <Bot className="w-5 h-5 text-white" />
          <h2 className="text-xl font-bold">Live Overview</h2>
        </div>

        {error && (
          <div className="rounded-xl border border-rose-500/30 bg-rose-500/5 px-4 py-3 text-sm text-rose-300">
            Dashboard feed error: {error}
          </div>
        )}

        {loading && (
          <div className="rounded-xl border border-neutral-800 bg-surface px-4 py-3 text-sm text-neutral-400">
            Loading filterless dashboard state...
          </div>
        )}

        <div className="grid grid-cols-2 xl:grid-cols-6 gap-4">
          <StatsCard title="Bot Status" value={effectiveBotStatus.toUpperCase()} icon={heartbeatOk ? Wifi : WifiOff} color={botStatusColor} />
          <StatsCard title="Session" value={state.bot.session || '--'} icon={Clock3} />
          <StatsCard title="Market Price" value={formatPrice(state.bot.price)} icon={Activity} />
          <StatsCard title="Daily Realized" value={formatMoney(state.bot.risk.daily_pnl)} icon={Waves} color={dailyPnlColor} />
          <StatsCard
            title="Open Position"
            value={openPositionSummarySide || 'FLAT'}
            icon={
              openPositionSummarySide === 'LONG'
                ? ArrowUpRight
                : openPositionSummarySide === 'SHORT'
                  ? ArrowDownRight
                  : BrainCircuit
            }
            color={openPositionCount > 0 ? 'warning' : 'default'}
          />
          <StatsCard title="Open PnL" value={formatMoney(openPositionCount > 0 ? totalOpenPnlDollars : null)} icon={ShieldAlert} color={openPositionCount > 0 ? openPnlColor : 'default'} />
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-6">
          {displayStrategies.map((strategy) => (
            <StrategyCard key={strategy.id} strategy={strategy} />
          ))}
        </div>

        <div className="grid grid-cols-1 xl:grid-cols-3 gap-6 items-stretch">
          <Panel
            title="Live Positions"
            right={
              <span className={`rounded-full border px-3 py-1 text-xs font-semibold uppercase tracking-wide ${statusChipClasses(openPositionCount > 0 ? 'in_trade' : 'idle')}`}>
                {openPositionCount > 0 ? `${openPositionCount} Open · ${formatLots(totalOpenLots)}` : 'Flat'}
              </span>
            }
            className="xl:col-span-1"
          >
            {openPositionCount > 0 ? (
              <div className="max-h-[520px] space-y-3 overflow-y-auto pr-1">
                {openPositions.map((position, index) => {
                  const positionCurrentPrice = getPositionCurrentPrice(position, state.bot.price);
                  const pnlDollars = position.open_pnl_dollars ?? null;
                  const pnlPoints = position.open_pnl_points ?? null;
                  return (
                    <div key={`${position.strategy_id}-${position.entry_order_id || position.order_id || position.opened_at || index}`} className="rounded-lg border border-neutral-800 bg-neutral-950/72 px-4 py-3">
                      <div className="flex items-start justify-between gap-3">
                        <div className="min-w-0">
                          <p className="truncate text-sm font-medium text-neutral-100">{position.strategy_label}</p>
                          <p className="mt-1 truncate text-xs text-neutral-500">
                            {formatToken(position.combo_key || position.sub_strategy)} · {formatToken(position.rule_id || position.entry_mode)}
                          </p>
                        </div>
                        <div className="flex flex-wrap items-center justify-end gap-2">
                          <span className={`rounded-full border px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.18em] ${tradeSideClasses(position.side)}`}>
                            {position.side}
                          </span>
                          <span className="rounded-full border border-neutral-700 bg-neutral-900 px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.18em] text-neutral-300">
                            {formatLots(position.size)}
                          </span>
                        </div>
                      </div>

                      <div className="mt-3 grid grid-cols-2 gap-3 text-sm">
                        <div>
                          <p className="text-neutral-500">Entry</p>
                          <p className="font-medium text-neutral-100">{formatPrice(position.entry_price)}</p>
                        </div>
                        <div>
                          <p className="text-neutral-500">Current</p>
                          <p className="font-medium text-neutral-100">{formatPrice(positionCurrentPrice)}</p>
                        </div>
                        <div>
                          <p className="text-neutral-500">Stop</p>
                          <p className="font-medium text-neutral-100">{formatPrice(position.stop_price ?? position.sl_price)}</p>
                        </div>
                        <div>
                          <p className="text-neutral-500">Target</p>
                          <p className="font-medium text-neutral-100">{formatPrice(position.target_price ?? position.tp_price)}</p>
                        </div>
                        <div>
                          <p className="text-neutral-500">Open Points</p>
                          <p className={`font-semibold ${(pnlPoints || 0) >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                            {formatPoints(pnlPoints)}
                          </p>
                        </div>
                        <div>
                          <p className="text-neutral-500">Open Dollars</p>
                          <p className={`font-semibold ${(pnlDollars || 0) >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                            {formatMoney(pnlDollars)}
                          </p>
                        </div>
                      </div>

                      <div className="mt-3 flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] text-neutral-500">
                        <span>Gate {formatGateSummary(position.gate_prob, position.gate_threshold)}</span>
                        <span>Kalshi {formatKalshiGateSummary(position.kalshi_gate_applied, position.kalshi_gate_multiplier, position.kalshi_gate_reason)}</span>
                        <span>Early Exit {formatBooleanLabel(position.early_exit_enabled, 'On', 'Off')}</span>
                        <span>{formatTimestamp(position.opened_at)}</span>
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className="rounded-lg border border-dashed border-neutral-800 px-4 py-10 text-center text-sm text-neutral-500">
                No filterless positions are currently open.
              </div>
            )}
          </Panel>

          <Panel title="Price Tape" className="xl:col-span-2">
            {priceData.length > 1 ? (
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={priceData}>
                    <defs>
                      <linearGradient id="priceFade" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#ffffff" stopOpacity={0.18} />
                        <stop offset="100%" stopColor="#ffffff" stopOpacity={0.01} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#262626" vertical={false} />
                    <XAxis dataKey="label" stroke="#666666" tickLine={false} tick={{ fontSize: 12 }} />
                    <YAxis stroke="#666666" tickLine={false} tick={{ fontSize: 12 }} domain={['auto', 'auto']} />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#0a0a0a', border: '1px solid #262626', borderRadius: 12 }}
                      labelStyle={{ color: '#d4d4d4' }}
                      formatter={(value: number) => [`${value.toFixed(2)}`, 'Price']}
                    />
                    <Area
                      type="monotone"
                      dataKey="price"
                      stroke="#ffffff"
                      fill="url(#priceFade)"
                      strokeWidth={2}
                      dot={false}
                      isAnimationActive={false}
                    />
                    {primaryOpenPosition?.entry_price != null && openPositionCount === 1 && (
                      <Line
                        type="monotone"
                        dataKey={() => primaryOpenPosition.entry_price as number}
                        stroke="#737373"
                        strokeDasharray="5 5"
                        dot={false}
                        isAnimationActive={false}
                      />
                    )}
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <div className="rounded-lg border border-dashed border-neutral-800 px-4 py-10 text-center text-sm text-neutral-500">
                No bar history has been bridged yet.
              </div>
            )}
          </Panel>
        </div>

        <div className="grid grid-cols-1 xl:grid-cols-3 gap-6 items-stretch">
          <Panel title="Risk Monitor" className="xl:col-span-1">
            <div className="space-y-4 text-sm">
              <div className="flex items-center justify-between">
                <span className="text-neutral-500">Circuit Breaker</span>
                <span className={state.bot.risk.circuit_tripped ? 'text-rose-400 font-semibold' : 'text-emerald-400 font-semibold'}>
                  {state.bot.risk.circuit_tripped ? 'Tripped' : 'Clear'}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-neutral-500">Long Losses</span>
                <span className="text-neutral-100 font-medium">{state.bot.risk.long_consecutive_losses ?? 0}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-neutral-500">Short Losses</span>
                <span className="text-neutral-100 font-medium">{state.bot.risk.short_consecutive_losses ?? 0}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-neutral-500">Bias Reversal</span>
                <span className="text-neutral-100 font-medium">{state.bot.risk.reversed_bias || '--'}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-neutral-500">Hostile Day</span>
                <span className={state.bot.risk.hostile_day_active ? 'text-rose-400 font-semibold' : 'text-neutral-200 font-medium'}>
                  {state.bot.risk.hostile_day_active ? 'Active' : 'Clear'}
                </span>
              </div>
              <div className="rounded-lg border border-neutral-800 bg-neutral-950/70 px-3 py-3 text-neutral-400 leading-6 min-h-[92px]">
                {state.bot.risk.hostile_day_reason || 'No hostile-day escalation is active.'}
              </div>
            </div>
          </Panel>

          <Panel title="Session Realized PnL" right={<span className="text-xs text-neutral-500">{intradayAnchorLabel}</span>} className="xl:col-span-2">
            {intradayPnlData.length > 0 ? (
              <div className="h-72">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={intradayPnlData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#262626" vertical={false} />
                    <XAxis dataKey="label" stroke="#666666" tickLine={false} tick={{ fontSize: 12 }} />
                    <YAxis stroke="#666666" tickLine={false} tick={{ fontSize: 12 }} tickFormatter={(value) => `$${value}`} />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#0a0a0a', border: '1px solid #262626', borderRadius: 12 }}
                      labelStyle={{ color: '#d4d4d4' }}
                      formatter={(value: number) => [formatMoney(value), 'Closed PnL']}
                    />
                    <Line type="monotone" dataKey="pnl" stroke="#10b981" strokeWidth={2} dot={false} isAnimationActive={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <div className="rounded-lg border border-dashed border-neutral-800 px-4 py-10 text-center text-sm text-neutral-500">
                No closed filterless trades are available in the current futures session day.
              </div>
            )}
          </Panel>
        </div>

        {kalshiVisible && (
          <Panel
            title="Kalshi Hourly Contracts (KXINXU)"
            right={
              <div className="flex flex-wrap items-center justify-end gap-2">
                {kalshiMetrics?.trade_gating_active ? (
                  <span className="rounded-full border px-3 py-1 text-xs font-semibold uppercase tracking-wide bg-amber-500/10 text-amber-300 border-amber-500/30">
                    Trade Gating
                  </span>
                ) : (
                  <span className={`rounded-full border px-3 py-1 text-xs font-semibold uppercase tracking-wide ${kalshiStatusClasses(kalshiMetrics)}`}>
                    {kalshiMode}
                  </span>
                )}
                <span className="text-xs text-neutral-500">{formatTimestamp(kalshiMetrics?.updated_at)}</span>
              </div>
            }
          >
            <div className="space-y-5">
              <div className="grid grid-cols-2 gap-3 rounded-xl border border-neutral-800 bg-neutral-950/82 p-3 md:grid-cols-4">
                <MonitorMetaItem label="Mode" value={kalshiMode} />
                <MonitorMetaItem label="Source" value={kalshiMetrics?.source || 'snapshot'} />
                <MonitorMetaItem label="Live Contracts" value={`${kalshiAvailableContracts}/${kalshiHourlyContracts.length}`} />
                <MonitorMetaItem label="Execution" value={kalshiExecutionLabel(kalshiMetrics)} />
              </div>

              <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
                <MonitorMetric
                  label="Active Event"
                  value={kalshiMetrics?.event_ticker || '--'}
                  hint="Hourly KXINXU contract stream mirrored from the monitor feed."
                  valueClassName="text-sm font-semibold text-neutral-100"
                />
                <MonitorMetric
                  label={`60-Min ${kalshiPredictionLabel} Probability`}
                  value={formatPercent(kalshiMetrics?.probability_60m, 2)}
                  hint={kalshiProbabilityCaption}
                  valueClassName="text-lg font-semibold text-sky-300"
                />
                <MonitorMetric
                  label="ES Reference"
                  value={formatPrice(kalshiEsReferencePrice)}
                  hint={`SPX basis offset ${formatPrice(kalshiMetrics?.basis_offset)}`}
                  valueClassName="text-lg font-semibold text-neutral-100"
                />
                <MonitorMetric
                  label="Attention Window"
                  value={
                    kalshiMetrics?.trade_gating_active
                      ? `3x focus — ${etHourToLocalLabel(kalshiMetrics.trade_gating_hour!)}`
                      : kalshiExecutionLabel(kalshiMetrics)
                  }
                  hint={kalshiMetrics?.status_reason || 'Kalshi ladder status is waiting for a healthy snapshot.'}
                  valueClassName={kalshiMetrics?.trade_gating_active ? 'text-lg font-semibold text-amber-300' : 'text-lg font-semibold text-neutral-300'}
                  className={kalshiMetrics?.trade_gating_active ? 'border-amber-500/30 bg-amber-500/5' : ''}
                />
              </div>

              <div className="grid grid-cols-1 gap-5 xl:grid-cols-[1.2fr,0.95fr]">
                <MonitorSection
                  title="Hourly Schedule"
                  subtitle="10 AM - 4 PM ET, rendered in your local timezone."
                  right={
                    selectedKalshiHour != null ? (
                      <button
                        onClick={() => setSelectedKalshiHour(null)}
                        className="rounded-full border border-neutral-700 px-3 py-1 text-[10px] font-semibold uppercase tracking-[0.18em] text-neutral-300 transition-colors hover:border-sky-500/40 hover:text-sky-300"
                      >
                        Return To Active
                      </button>
                    ) : null
                  }
                >
                  <div className="grid grid-cols-2 gap-2 sm:grid-cols-4 xl:grid-cols-7">
                    {kalshiHourlyContracts.map((contract) => {
                      const isSelected = selectedKalshiHour === contract.hour;
                      const isViewed = isSelected || (selectedKalshiHour == null && contract.isActive);

                      let borderClass: string;
                      let bgClass: string;
                      if (isViewed) {
                        borderClass = 'border-sky-500/50';
                        bgClass = 'bg-sky-500/10';
                      } else if (contract.tradeGating) {
                        borderClass = 'border-amber-500/40';
                        bgClass = 'bg-amber-500/5';
                      } else if (contract.available) {
                        borderClass = 'border-emerald-500/30';
                        bgClass = 'bg-emerald-500/5';
                      } else if (contract.upcoming) {
                        borderClass = 'border-neutral-700';
                        bgClass = 'bg-neutral-900/40';
                      } else {
                        borderClass = 'border-neutral-800';
                        bgClass = 'bg-neutral-950/40';
                      }

                      let statusLabel: string;
                      let statusColor: string;
                      if (contract.tradeGating) {
                        statusLabel = 'Gating';
                        statusColor = 'text-amber-400';
                      } else if (contract.available) {
                        statusLabel = 'Live';
                        statusColor = 'text-emerald-400';
                      } else if (contract.upcoming) {
                        statusLabel = 'Upcoming';
                        statusColor = 'text-neutral-500';
                      } else {
                        statusLabel = 'Settled';
                        statusColor = 'text-neutral-600';
                      }

                      return (
                        <button
                          key={contract.hour}
                          onClick={() => setSelectedKalshiHour(isSelected ? null : contract.hour)}
                          className={`min-h-[92px] rounded-xl border px-3 py-3 text-left transition-all hover:border-neutral-500 ${borderClass} ${bgClass} ${
                            isViewed ? 'shadow-[0_0_0_1px_rgba(56,189,248,0.18)]' : ''
                          }`}
                        >
                          <p
                            className={`text-sm font-semibold ${
                              isViewed
                                ? 'text-sky-300'
                                : contract.tradeGating
                                  ? 'text-amber-300'
                                  : contract.available
                                    ? 'text-emerald-300'
                                    : 'text-neutral-400'
                            }`}
                          >
                            {contract.label}
                          </p>
                          <p className={`mt-2 text-[10px] font-semibold uppercase tracking-[0.18em] ${statusColor}`}>
                            {statusLabel}
                          </p>
                          <p className="mt-2 text-[11px] text-neutral-500">
                            {contract.strikeCount > 0 ? `${contract.strikeCount} strikes` : 'No ladder yet'}
                          </p>
                        </button>
                      );
                    })}
                  </div>

                  {viewedKalshiContract && (
                    <div className="mt-4 grid grid-cols-1 gap-4 rounded-xl border border-neutral-700 bg-neutral-950/90 px-4 py-4 md:grid-cols-[1.2fr,0.8fr]">
                      <div>
                        <p className="text-[10px] font-semibold uppercase tracking-[0.18em] text-neutral-500">
                          Selected Window
                        </p>
                        <p className="mt-1 text-sm font-semibold text-neutral-100">
                          {viewedKalshiContract.label} Contract
                        </p>
                        <p className="mt-1 text-xs leading-5 text-neutral-500">
                          {viewedKalshiContract.tradeGating
                            ? 'Kalshi is actively influencing execution during this settlement hour with veto / sizing logic.'
                            : viewedKalshiContract.available && KALSHI_GATING_HOURS.includes(viewedKalshiContract.hour)
                              ? `Live crowd contract for ${etHourToLocalLabel(viewedKalshiContract.hour)}. Gating arms automatically when that hour is active.`
                              : viewedKalshiContract.available && [10, 11].includes(viewedKalshiContract.hour)
                                ? 'Morning crowd contract is live, but execution stays ML-only because that window is contrarian.'
                                : viewedKalshiContract.upcoming
                                  ? 'This settlement hour has not opened yet.'
                                  : 'This hourly contract has already settled.'}
                        </p>
                      </div>
                      <div className="space-y-3">
                        <div className="flex items-center justify-between gap-3">
                          <span className="text-[10px] font-semibold uppercase tracking-[0.18em] text-neutral-500">Ticker</span>
                          <span className="text-xs text-neutral-300">{viewedKalshiContract.eventTicker}</span>
                        </div>
                        <div className="flex items-center justify-between gap-3">
                          <span className="text-[10px] font-semibold uppercase tracking-[0.18em] text-neutral-500">Status</span>
                          <span
                            className={`rounded-full border px-3 py-1 text-[10px] font-semibold uppercase tracking-[0.18em] ${
                              viewedKalshiContract.tradeGating
                                ? 'bg-amber-500/10 text-amber-300 border-amber-500/30'
                                : viewedKalshiContract.available
                                  ? 'bg-emerald-500/10 text-emerald-300 border-emerald-500/30'
                                  : viewedKalshiContract.upcoming
                                    ? 'bg-neutral-900 text-neutral-300 border-neutral-700'
                                    : 'bg-neutral-800 text-neutral-400 border-neutral-700'
                            }`}
                          >
                            {viewedKalshiContract.tradeGating
                              ? 'Gating'
                              : viewedKalshiContract.available
                                ? 'Available'
                                : viewedKalshiContract.upcoming
                                  ? 'Upcoming'
                                  : 'Settled'}
                          </span>
                        </div>
                        <div className="flex items-center justify-between gap-3">
                          <span className="text-[10px] font-semibold uppercase tracking-[0.18em] text-neutral-500">Local Time</span>
                          <span className="text-xs text-neutral-300">{etHourToLocalLabel(viewedKalshiContract.hour)}</span>
                        </div>
                      </div>
                    </div>
                  )}
                </MonitorSection>

                <MonitorSection
                  title="Strike Ladder"
                  subtitle={`${kalshiMetrics?.event_ticker || 'N/A'} on the ${kalshiPredictionLabel} side.`}
                  right={
                    nearestKalshiStrike ? (
                      <div className="rounded-full border border-sky-500/20 bg-sky-500/10 px-3 py-2 text-xs text-sky-200">
                        Nearest strike {formatStrikeLabel(nearestKalshiStrike.strike)}
                      </div>
                    ) : null
                  }
                >
                  <div className="grid grid-cols-1 gap-3 rounded-xl border border-neutral-800 bg-neutral-950/82 p-3 sm:grid-cols-3">
                    <MonitorMetaItem label="Reference SPX" value={formatPrice(kalshiSpxReferencePrice)} />
                    <MonitorMetaItem label="Window" value={`${kalshiDisplayedStrikes.length} rows`} />
                    <MonitorMetaItem label="Execution" value={kalshiExecutionLabel(kalshiMetrics)} />
                  </div>

                  {(selectedKalshiHour == null || viewedKalshiContract?.isActive) && kalshiStrikes.length > 0 ? (
                    <div className="mt-4 overflow-hidden rounded-xl border border-neutral-800">
                      <div className="max-h-[360px] overflow-auto">
                        <table className="min-w-full text-sm">
                          <thead className="sticky top-0 bg-neutral-950 text-left text-neutral-500 border-b border-neutral-800">
                            <tr>
                              <th className="py-3 pl-4 pr-4 font-medium">S&amp;P 500 Strike</th>
                              <th className="py-3 pr-4 font-medium">{kalshiPredictionLabel} Probability</th>
                              <th className="py-3 pr-4 font-medium">Volume</th>
                              <th className="py-3 pr-4 font-medium">Status</th>
                            </tr>
                          </thead>
                          <tbody>
                            {kalshiDisplayedStrikes.map((strike, index) => (
                              <tr
                                key={`${strike.strike}-${index}`}
                                className={`border-b border-neutral-900 last:border-b-0 ${
                                  nearestKalshiStrike && nearestKalshiStrike.strike === strike.strike
                                    ? 'bg-sky-500/5'
                                    : 'bg-neutral-950/40'
                                }`}
                              >
                                <td className="py-3 pl-4 pr-4 text-neutral-200">{formatStrikeLabel(strike.strike)}</td>
                                <td className="py-3 pr-4 font-medium text-sky-300">{formatPercent(strike.displayProbability, 2)}</td>
                                <td className="py-3 pr-4 text-neutral-300">{formatCompactNumber(strike.volume)}</td>
                                <td className="py-3 pr-4 text-neutral-400">{strike.status || '--'}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  ) : selectedKalshiHour != null && !viewedKalshiContract?.isActive ? (
                    <div className="mt-4 rounded-xl border border-dashed border-neutral-800 px-4 py-8 text-center text-sm text-neutral-500">
                      Strike data is only available for the active contract window. Return to the active hour to inspect the live ladder.
                    </div>
                  ) : (
                    <div className="mt-4 rounded-xl border border-dashed border-neutral-800 px-4 py-8 text-center text-sm text-neutral-500">
                      {kalshiMetrics?.status_reason || 'Kalshi strike ladder is not available yet. Contracts populate as each hourly window opens.'}
                    </div>
                  )}
                </MonitorSection>
              </div>
            </div>
          </Panel>
        )}

        <Panel
          title="Truth Social Monitor"
          right={
            <span className={`rounded-full border px-3 py-1 text-xs font-semibold uppercase tracking-wide ${statusChipClasses(sentimentPanelStatus)}`}>
              {sentimentPanelStatusLabel}
            </span>
          }
        >
          <div className="space-y-5">
            <div className="grid grid-cols-2 gap-3 rounded-xl border border-neutral-800 bg-neutral-950/82 p-3 md:grid-cols-4">
              <MonitorMetaItem label="Handle" value={`@${sentimentMetrics.target_handle || 'realDonaldTrump'}`} />
              <MonitorMetaItem label="Source" value={sentimentMetrics.source || 'rss_finbert'} />
              <MonitorMetaItem label="Mode" value="Observe-only" />
              <MonitorMetaItem label="Last Poll" value={formatRelativeTime(sentimentMetrics.last_poll_at)} />
            </div>

            <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
              <MonitorMetric
                label="Sentiment Score"
                value={sentimentMetrics.sentiment_score == null ? '--' : sentimentMetrics.sentiment_score.toFixed(2)}
                hint={sentimentMetrics.sentiment_label ? sentimentMetrics.sentiment_label.toUpperCase() : 'No classified sentiment yet'}
                valueClassName={sentimentScoreClasses(sentimentMetrics)}
              />
              <MonitorMetric
                label="FinBERT Confidence"
                value={formatPercent(sentimentMetrics.finbert_confidence, 1)}
                hint={sentimentMetrics.quantized_8bit ? '8-bit runtime path' : 'Full precision runtime path'}
                valueClassName="text-lg font-semibold text-neutral-100"
              />
              <MonitorMetric
                label="Bias / Watch"
                value={sentimentMetrics.trigger_side || (sentimentMetrics.sentiment_label || '--').toUpperCase()}
                hint={sentimentMetrics.trigger_reason || 'Dashboard context only; no entries or exits are driven from this feed here.'}
                valueClassName="text-lg font-semibold text-neutral-100"
              />
              <MonitorMetric
                label="Gemini Layer"
                value={geminiUsageLabel}
                hint={geminiModel}
                valueClassName={geminiUsed ? 'text-lg font-semibold text-sky-300' : 'text-lg font-semibold text-neutral-100'}
              />
            </div>

            <div className="grid grid-cols-1 gap-5 xl:grid-cols-[1.15fr,0.85fr]">
              <MonitorSection
                title="Latest Post"
                subtitle={formatRelativeTime(sentimentMetrics.latest_post_created_at || sentimentMetrics.last_analysis_at)}
                right={
                  sentimentMetrics.latest_post_url ? (
                    <a
                      className="rounded-full border border-neutral-700 px-3 py-1 text-[10px] font-semibold uppercase tracking-[0.18em] text-sky-300 transition-colors hover:border-sky-500/40 hover:text-sky-200"
                      href={sentimentMetrics.latest_post_url}
                      target="_blank"
                      rel="noreferrer"
                    >
                      Open Post
                    </a>
                  ) : null
                }
              >
                {sentimentExcerpt ? (
                  <>
                    <div className="rounded-xl border border-neutral-700 bg-neutral-950/90 px-5 py-5">
                      <p className="text-sm leading-7 text-neutral-100">
                        "{sentimentExcerpt}"
                      </p>
                    </div>
                    {sentimentMetrics.trigger_reason && (
                      <p className="mt-4 text-sm leading-6 text-neutral-400">
                        {sentimentMetrics.trigger_reason}
                      </p>
                    )}
                    {geminiVerdict && (
                      <div className="mt-4 rounded-xl border border-sky-500/20 bg-sky-500/5 px-4 py-3 text-sm leading-6 text-sky-100">
                        {geminiVerdict}
                      </div>
                    )}
                    {geminiReasoning && (
                      <p className="mt-3 text-xs leading-6 text-neutral-500">
                        Gemini reasoning: {geminiReasoning}
                      </p>
                    )}
                  </>
                ) : (
                  <div className="rounded-xl border border-dashed border-neutral-800 px-4 py-10 text-center text-sm text-neutral-500">
                    {sentimentMetrics.last_error
                      ? sentimentMetrics.last_error
                      : 'No post has been analyzed yet. Waiting for new Truth Social activity.'}
                  </div>
                )}
              </MonitorSection>

              <MonitorSection title="Monitor State">
                <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                  <div className="rounded-xl border border-neutral-700 bg-neutral-950/88 px-4 py-4">
                    <p className="text-[10px] font-semibold uppercase tracking-[0.18em] text-neutral-500">Model</p>
                    <p className="mt-2 text-sm font-semibold text-neutral-100">
                      {sentimentMetrics.quantized_8bit ? '8-bit FinBERT' : 'FinBERT'}
                    </p>
                    <p className="mt-1 text-xs text-neutral-500">
                      {sentimentMetrics.model_loaded ? 'Loaded and evaluating new posts.' : 'Waiting for model/runtime warmup.'}
                    </p>
                  </div>
                  <div className="rounded-xl border border-neutral-700 bg-neutral-950/88 px-4 py-4">
                    <p className="text-[10px] font-semibold uppercase tracking-[0.18em] text-neutral-500">Gemini</p>
                    <p className="mt-2 text-sm font-semibold text-neutral-100">{geminiUsageLabel}</p>
                    <p className="mt-1 text-xs text-neutral-500">{geminiModel}</p>
                  </div>
                </div>

                <div className="mt-4 grid grid-cols-2 gap-3 rounded-xl border border-neutral-800 bg-neutral-950/82 p-3 md:grid-cols-4">
                  <MonitorMetaItem label="Analysis" value={formatRelativeTime(sentimentMetrics.last_analysis_at)} />
                  <MonitorMetaItem label="Healthy" value={formatBooleanLabel(sentimentMetrics.healthy, 'Yes', 'No')} />
                  <MonitorMetaItem label="Gemini" value={geminiEnabled ? 'Enabled' : 'Disabled'} />
                  <MonitorMetaItem label="Execution" value="No trade impact" />
                  {geminiConfidence != null && (
                    <MonitorMetaItem label="Confidence" value={formatPercent(geminiConfidence, 1)} />
                  )}
                  {geminiScore != null && (
                    <MonitorMetaItem label="Score" value={geminiScore.toFixed(2)} />
                  )}
                  {geminiMarketImpact && (
                    <MonitorMetaItem label="Impact" value={geminiMarketImpact.toUpperCase()} />
                  )}
                </div>
              </MonitorSection>
            </div>
          </div>
        </Panel>

        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6 items-start">
          <Panel title="Recent Events" right={<span className="text-xs text-neutral-500">{state.events.length} items</span>}>
            <div className="space-y-3 max-h-[520px] overflow-y-auto pr-1">
              {state.events.length > 0 ? (
                state.events.map((event: FilterlessEvent, index) => (
                  <div key={`${event.event_type}-${event.time}-${index}`} className={`rounded-lg border px-4 py-3 ${eventTone(event.severity)}`}>
                    <div className="flex items-center justify-between gap-3 mb-2">
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-semibold uppercase tracking-wide text-neutral-400">{event.event_type}</span>
                        {event.strategy_label && <span className="text-xs text-neutral-500">{event.strategy_label}</span>}
                      </div>
                      <span className="text-xs text-neutral-500">{formatTimestamp(event.time)}</span>
                    </div>
                    <p className="text-sm text-neutral-200 leading-6">{event.message}</p>
                  </div>
                ))
              ) : (
                <div className="rounded-lg border border-dashed border-neutral-800 px-4 py-10 text-center text-sm text-neutral-500">
                  No filterless events have been bridged yet.
                </div>
              )}
            </div>
          </Panel>

          <Panel title="Trade Blotter" right={<span className="text-xs text-neutral-500">{state.trades.length} trades</span>}>
            {state.trades.length > 0 ? (
              <div className="max-h-[520px] overflow-y-auto pr-1">
                <div className="overflow-hidden rounded-lg border border-neutral-800">
                  <table className="min-w-full text-sm">
                    <thead className="sticky top-0 bg-neutral-950 text-left text-neutral-500 border-b border-neutral-800">
                      <tr>
                        <th className="py-3 pl-4 pr-4 font-medium">Time</th>
                        <th className="py-3 pr-4 font-medium">Strategy</th>
                        <th className="py-3 pr-4 font-medium">Side</th>
                        <th className="py-3 pr-4 font-medium">Lots</th>
                        <th className="py-3 pr-4 font-medium">Entry</th>
                        <th className="py-3 pr-4 font-medium">Exit</th>
                        <th className="py-3 pr-4 font-medium">Points</th>
                        <th className="py-3 pr-4 font-medium">PnL</th>
                      </tr>
                    </thead>
                    <tbody>
                      {state.trades.map((trade: FilterlessTrade, index) => {
                        const pnlDollars = trade.pnl_dollars_net ?? trade.pnl_dollars ?? null;
                        const rowTone = tradeTone(trade);
                        return (
                          <tr key={`${trade.time}-${index}`} className={`border-b border-neutral-900 last:border-b-0 ${rowTone}`}>
                            <td className="py-3 pl-4 pr-4 text-neutral-400">{formatTimestamp(trade.time)}</td>
                            <td className="py-3 pr-4 text-neutral-100">{trade.strategy_label}</td>
                            <td className="py-3 pr-4">
                              <span className={`rounded-full border px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.18em] ${tradeSideClasses(trade.side)}`}>
                                {trade.side || '--'}
                              </span>
                            </td>
                            <td className="py-3 pr-4 text-neutral-400">{formatLotCount(trade.size)}</td>
                            <td className="py-3 pr-4 text-neutral-300">{formatPrice(trade.entry_price)}</td>
                            <td className="py-3 pr-4 text-neutral-300">{formatPrice(trade.exit_price)}</td>
                            <td className="py-3 pr-4 text-neutral-400">{formatPoints(trade.pnl_points)}</td>
                            <td className={`py-3 pr-4 font-semibold ${pnlDollars == null ? 'text-neutral-300' : pnlDollars >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                              {formatMoney(pnlDollars)}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            ) : (
              <div className="rounded-lg border border-dashed border-neutral-800 px-4 py-10 text-center text-sm text-neutral-500">
                No recent closed filterless trades have been detected.
              </div>
            )}
          </Panel>
        </div>

        {feedWarnings.length > 0 && (
          <Panel title="Feed Notes" right={<span className="text-xs text-neutral-500">observability gaps</span>}>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {feedWarnings.map((warning, index) => (
                <div key={`${warning}-${index}`} className="rounded-lg border border-amber-500/20 bg-amber-500/5 px-4 py-3 text-sm text-amber-200">
                  {warning}
                </div>
              ))}
            </div>
          </Panel>
        )}
      </main>
    </div>
  );
}

export default FilterlessLiveApp;
