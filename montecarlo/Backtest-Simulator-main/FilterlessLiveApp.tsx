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
  FilterlessStrategyState,
  FilterlessTrade,
} from './filterlessLiveTypes';

const REFRESH_MS = 3000;
const FEED_STALE_SECONDS = 90;
const FEED_OFFLINE_SECONDS = 300;

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
    price_history: [],
    risk: {},
    warnings: [],
  },
  strategies: [],
  events: [],
  trades: [],
  kalshi_metrics: null,
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

function hasText(value?: string | null): boolean {
  return Boolean(value && value.trim());
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
  /** Contract is currently shown on Kalshi (within its 4-hour pre-settlement window) */
  available: boolean;
  /** Contract has already settled */
  settled: boolean;
  /** Contract is not yet within its availability window */
  upcoming: boolean;
  strikeCount: number;
  /** This is the contract the backend is currently tracking */
  isActive: boolean;
  /** Trade gating is active for this hour */
  tradeGating: boolean;
}

const KALSHI_SETTLEMENT_HOURS = [10, 11, 12, 13, 14, 15, 16] as const;
// Hours where trade gating (3x sizing) is active — 10-11 AM excluded due to contrarian crowd
const KALSHI_GATING_HOURS = [12, 13, 14, 15, 16] as const;

function formatHourLabel(hour: number): string {
  if (hour === 12) return '12 PM';
  if (hour > 12) return `${hour - 12} PM`;
  return `${hour} AM`;
}

/**
 * Convert a timezone-specific hour (0-23) to a label in the user's local timezone.
 */
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

  // Step 2: Compute the timezone's current UTC offset
  const utcMs = new Date(now.toLocaleString('en-US', { timeZone: 'UTC' })).getTime();
  const tzMs = new Date(now.toLocaleString('en-US', { timeZone })).getTime();
  const tzOffsetHours = Math.round((tzMs - utcMs) / 3_600_000);

  // Step 3: Build a UTC timestamp for "today at hour:00 in the given timezone"
  const utcHour = hour - tzOffsetHours;
  const target = new Date(`${yy}-${mm}-${dd}T${String(utcHour).padStart(2, '0')}:00:00Z`);

  // Step 4: Format in the browser's local timezone
  return target.toLocaleTimeString([], { hour: 'numeric', hour12: true });
}

/** Convert an ET hour to a label in the user's local timezone. */
function etHourToLocalLabel(etHour: number): string {
  return tzHourToLocalLabel(etHour, 'America/New_York');
}

/**
 * Get the current ET minute-of-day for availability calculations.
 */
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

/**
 * KXINXU contracts settle at each hour 10 AM - 4 PM ET (per CFTC filing:
 * "Time will be measured in Eastern Time (ET)" covering "traditional
 * market hours 9:30 AM - 4 PM ET").
 *
 * A contract is:
 *   - "available" if the settlement hour has not yet passed
 *   - "settled" once the settlement hour has passed
 *   - never "upcoming/not yet" once Kalshi has data for it (renderable)
 *
 * Trade gating activates during each settlement hour with 3x sizing.
 * The backend sends trade_gating_hour in ET matching these hours.
 *
 * UI labels are displayed in the viewer's local timezone.
 */
function buildKalshiHourlyContracts(
  eventTicker?: string | null,
  strikes?: FilterlessKalshiStrike[],
  tradeGatingHour?: number | null,
  dailyContracts?: { et_hour: number; strike_count: number; settled: boolean }[] | null,
): KalshiHourlyContract[] {
  const tickerPrefix = eventTicker?.replace(/H\d+$/, '') || null;
  const activeHourMatch = eventTicker?.match(/H(\d+)$/);
  const activeHourCode = activeHourMatch ? parseInt(activeHourMatch[1], 10) : null;

  const etMinuteOfDay = currentETMinuteOfDay();

  // Build a lookup of which hours have data from the backend backfill
  const renderableHours = new Set<number>();
  if (dailyContracts) {
    for (const dc of dailyContracts) {
      if (dc.strike_count > 0) {
        renderableHours.add(dc.et_hour);
      }
    }
  }

  return KALSHI_SETTLEMENT_HOURS.map((hour) => {
    // hour is an ET settlement hour (10, 11, 12, 13, 14, 15, 16)
    const hourCode = hour * 100;
    const ticker = tickerPrefix ? `${tickerPrefix}H${hourCode}` : `H${hourCode}`;
    const isActive = activeHourCode === hourCode;

    // The backend rolls to the next contract at :05 after settlement.
    const isSettled = etMinuteOfDay >= ((hour * 60) + 5);
    // Contract has data from Kalshi (renderable) — never show as "NOT YET"
    const hasData = renderableHours.has(hour) || isActive;
    // Available = not settled, or has data and not settled
    const isAvailable = !isSettled;
    // Only "upcoming" if no data AND before settlement — once renderable, skip this state
    const isUpcoming = !isSettled && !isAvailable && !hasData;

    const strikeCount = isActive ? (strikes?.length || 0) : 0;
    const tradeGating = tradeGatingHour === hour;

    // Display in user's local timezone (convert from ET)
    const localLabel = etHourToLocalLabel(hour);

    return {
      hour,
      label: localLabel,
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

function clampPercent(value: number): number {
  return Math.max(0, Math.min(100, value));
}

function positionMarkerPercent(position: FilterlessPosition, fallbackCurrentPrice?: number | null): number {
  const target = position.target_price ?? position.tp_price;
  const stop = position.stop_price ?? position.sl_price;
  const entry = getPositionEntryPrice(position);
  if (target == null || stop == null || entry == null) return 50;
  const currentPrice = getPositionCurrentPrice(position, fallbackCurrentPrice);
  if (currentPrice == null) return 50;

  if (position.side === 'LONG') {
    if (currentPrice <= entry) {
      const riskSpan = entry - stop;
      if (riskSpan <= 0) return 50;
      return clampPercent(((currentPrice - stop) / riskSpan) * 50);
    }
    const rewardSpan = target - entry;
    if (rewardSpan <= 0) return 50;
    return clampPercent(50 + ((currentPrice - entry) / rewardSpan) * 50);
  }

  if (position.side === 'SHORT') {
    if (currentPrice >= entry) {
      const riskSpan = stop - entry;
      if (riskSpan <= 0) return 50;
      return clampPercent(((stop - currentPrice) / riskSpan) * 50);
    }
    const rewardSpan = entry - target;
    if (rewardSpan <= 0) return 50;
    return clampPercent(50 + ((entry - currentPrice) / rewardSpan) * 50);
  }

  return 50;
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
  const details = [
    hasText(contextLabel) ? { label: 'Context', value: formatToken(contextLabel), accent: tone.accent } : null,
    hasText(executionLabel) ? { label: 'Rule', value: formatToken(executionLabel), accent: 'text-neutral-100' } : null,
    gateActive ? { label: 'Gate', value: formatGateSummary(strategy.gate_prob, strategy.gate_threshold), accent: 'text-amber-200' } : null,
    sessionLabel ? { label: 'Session', value: sessionLabel, accent: 'text-neutral-100' } : null,
    strategy.vol_regime ? { label: 'Regime', value: formatToken(strategy.vol_regime), accent: 'text-neutral-100' } : null,
    strategy.early_exit_enabled != null
      ? { label: 'Early Exit', value: formatBooleanLabel(strategy.early_exit_enabled, 'Enabled', 'Disabled'), accent: 'text-neutral-100' }
      : null,
  ].filter((item): item is { label: string; value: string; accent: string } => item !== null);
  const candidateSummary = [
    strategy.last_signal_side ? `Side ${strategy.last_signal_side}` : null,
    strategy.priority ? `Priority ${strategy.priority}` : null,
    strategy.last_signal_price != null ? `Price ${formatPrice(strategy.last_signal_price)}` : null,
    strategy.tp_dist != null || strategy.sl_dist != null
      ? `Bracket ${formatPrice(strategy.tp_dist)} / ${formatPrice(strategy.sl_dist)}`
      : null,
  ].filter((item): item is string => item !== null);

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

      {details.length > 0 ? (
        <div className="flex flex-wrap gap-2 mb-4">
          {details.map((detail) => (
            <div key={`${strategy.id}-${detail.label}`} className="rounded-full border border-neutral-800/80 bg-neutral-950/65 px-3 py-1.5">
              <span className="text-[10px] font-semibold uppercase tracking-[0.18em] text-neutral-500">{detail.label}</span>
              <span className={`ml-2 text-xs font-medium ${detail.accent}`}>{detail.value}</span>
            </div>
          ))}
        </div>
      ) : (
        <div className="rounded-lg border border-dashed border-neutral-800 bg-neutral-950/55 px-3 py-3 mb-4 text-sm text-neutral-500">
          Waiting for richer strategy metadata from the live engine.
        </div>
      )}

      <div className="rounded-lg border border-neutral-800 bg-neutral-950/70 p-3 mb-4 min-h-[84px]">
        <div className="mb-2 flex items-center justify-between gap-3">
          <p className="text-xs font-semibold uppercase tracking-wide text-neutral-500">Latest Activity</p>
          <p className="text-[11px] text-neutral-500">{formatRelativeTime(latestActivityTime)}</p>
        </div>
        <p className="text-sm text-neutral-300 leading-6">{latestActivity}</p>
      </div>

      <div className="grid grid-cols-1 gap-3 text-sm lg:grid-cols-2">
        <div>
          <p className="text-neutral-500">Last Candidate</p>
          <p className="text-neutral-100 font-medium">
            {candidateSummary.length > 0 ? candidateSummary.join(' · ') : 'No candidate has been logged yet.'}
          </p>
          <p className="text-xs text-neutral-500 mt-1">{formatTimestamp(strategy.last_signal_time)}</p>
        </div>
        <div>
          <p className="text-neutral-500">Last Trade</p>
          <p className={`font-semibold ${tradeColor}`}>{formatMoney(strategy.last_trade_pnl)}</p>
          <p className="text-xs text-neutral-500 mt-1">
            {strategy.last_trade_time ? formatTimestamp(strategy.last_trade_time) : 'No closed trade in this session yet.'}
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
      state.strategies.map((strategy) => (
        effectiveBotStatus === 'online' ? strategy : { ...strategy, status: effectiveBotStatus }
      )),
    [effectiveBotStatus, state.strategies],
  );
  const openPosition = effectiveBotStatus === 'online' ? state.bot.current_position : null;
  const openPositionCurrentPrice = useMemo(
    () => (openPosition ? getPositionCurrentPrice(openPosition, state.bot.price) : state.bot.price ?? null),
    [openPosition, state.bot.price],
  );
  const heartbeatOk = effectiveBotStatus === 'online';
  const botStatusColor = heartbeatOk ? 'success' : effectiveBotStatus === 'stale' ? 'warning' : 'danger';
  const dailyPnlColor = (state.bot.risk.daily_pnl || 0) >= 0 ? 'success' : 'danger';
  const openPnlColor = (openPosition?.open_pnl_dollars || 0) >= 0 ? 'success' : 'danger';
  const kalshiMetrics = state.kalshi_metrics ?? null;
  const kalshiVisible = kalshiMetrics != null;
  const kalshiOpenSide = String(openPosition?.side || '').toUpperCase();
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
    () => buildKalshiHourlyContracts(kalshiMetrics?.event_ticker, kalshiStrikes, kalshiMetrics?.trade_gating_hour, kalshiMetrics?.daily_contracts as any),
    [kalshiMetrics?.event_ticker, kalshiStrikes, kalshiMetrics?.trade_gating_hour, kalshiMetrics?.daily_contracts],
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
  // When selectedKalshiHour is set, show that contract's info; otherwise show the active one
  const viewedKalshiContract = useMemo(() => {
    if (selectedKalshiHour != null) {
      return kalshiHourlyContracts.find((c) => c.hour === selectedKalshiHour) ?? null;
    }
    return kalshiHourlyContracts.find((c) => c.isActive) ?? null;
  }, [selectedKalshiHour, kalshiHourlyContracts]);
  const warningCount = feedWarnings.length;
  const feedSummary = error
    ? `Dashboard feed error: ${error}`
    : effectiveBotStatus === 'online'
      ? 'Feed and heartbeat are updating normally.'
      : effectiveBotStatus === 'stale'
        ? 'Feed is stale. Recent state is still visible while the launcher catches up.'
        : 'Feed is offline. The launcher may still be starting, or the workspace needs attention.';
  const kalshiMode = kalshiModeLabel(kalshiMetrics);

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

        <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-6 gap-4">
          <StatsCard
            title="Bot Status"
            value={effectiveBotStatus.toUpperCase()}
            subValue={state.bot.last_heartbeat_time ? `Heartbeat ${formatRelativeTime(state.bot.last_heartbeat_time)}` : 'Waiting for heartbeat'}
            icon={heartbeatOk ? Wifi : WifiOff}
            color={botStatusColor}
          />
          <StatsCard
            title="Session"
            value={state.bot.session || '--'}
            subValue={state.bot.trading_day_start ? `Trading day ${formatTimestamp(state.bot.trading_day_start)}` : 'Current futures session'}
            icon={Clock3}
          />
          <StatsCard
            title="Market Price"
            value={formatPrice(state.bot.price)}
            subValue={state.bot.last_bar_time ? `Last bar ${formatRelativeTime(state.bot.last_bar_time)}` : 'Waiting for market tape'}
            icon={Activity}
          />
          <StatsCard
            title="Daily Realized"
            value={formatMoney(state.bot.risk.daily_pnl)}
            subValue={intradayAnchorLabel}
            icon={Waves}
            color={dailyPnlColor}
          />
          <StatsCard
            title="Open Position"
            value={openPosition ? openPosition.side : 'FLAT'}
            subValue={openPosition?.strategy_label || state.bot.position_sync_status || 'No active position'}
            icon={openPosition ? (openPosition.side === 'LONG' ? ArrowUpRight : ArrowDownRight) : BrainCircuit}
            color={openPosition ? 'warning' : 'default'}
          />
          <StatsCard
            title="Open PnL"
            value={formatMoney(openPosition?.open_pnl_dollars)}
            subValue={openPosition ? formatPoints(openPosition.open_pnl_points) : (state.bot.last_position_sync_time ? `Sync ${formatRelativeTime(state.bot.last_position_sync_time)}` : 'Awaiting broker sync')}
            icon={ShieldAlert}
            color={openPosition ? openPnlColor : 'default'}
          />
        </div>

        {/* System Pulse removed — Kalshi hourly contracts provide more actionable context */}

        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-6">
          {displayStrategies.map((strategy) => (
            <StrategyCard key={strategy.id} strategy={strategy} />
          ))}
        </div>

        <div className="grid grid-cols-1 xl:grid-cols-3 gap-6 items-stretch">
          <Panel
            title="Live Position"
            right={<span className={`rounded-full border px-3 py-1 text-xs font-semibold uppercase tracking-wide ${statusChipClasses(openPosition ? 'in_trade' : 'idle')}`}>{openPosition ? 'Active' : 'Flat'}</span>}
            className="xl:col-span-1"
          >
            {openPosition ? (
              <div className="space-y-5">
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <p className="text-neutral-500">Strategy</p>
                    <p className="text-neutral-100 font-medium">{openPosition.strategy_label}</p>
                    <p className="text-xs text-neutral-500 mt-1">
                      {formatToken(openPosition.combo_key || openPosition.sub_strategy)}
                    </p>
                  </div>
                  <div>
                    <p className="text-neutral-500">Side / Rule</p>
                    <p className="text-neutral-100 font-medium">
                      {openPosition.side} · {formatToken(openPosition.rule_id || openPosition.entry_mode)}
                    </p>
                    <p className="text-xs text-neutral-500 mt-1">
                      Early Exit {formatBooleanLabel(openPosition.early_exit_enabled, 'On', 'Off')}
                    </p>
                    <p className="text-xs text-neutral-500 mt-1">
                      Gate {formatGateSummary(openPosition.gate_prob, openPosition.gate_threshold)}
                    </p>
                  </div>
                  <div>
                    <p className="text-neutral-500">Entry</p>
                    <p className="text-neutral-100 font-medium">{formatPrice(getPositionEntryPrice(openPosition))}</p>
                  </div>
                  <div>
                    <p className="text-neutral-500">Current</p>
                    <p className="text-neutral-100 font-medium">{formatPrice(openPositionCurrentPrice)}</p>
                  </div>
                  <div>
                    <p className="text-neutral-500">Stop</p>
                    <p className="text-neutral-100 font-medium">{formatPrice(openPosition.stop_price ?? openPosition.sl_price)}</p>
                  </div>
                  <div>
                    <p className="text-neutral-500">Target</p>
                    <p className="text-neutral-100 font-medium">{formatPrice(openPosition.target_price ?? openPosition.tp_price)}</p>
                  </div>
                </div>

                <div>
                  <div className="flex items-center justify-between text-xs text-neutral-500 mb-2">
                    <span>{formatPrice(openPosition.stop_price ?? openPosition.sl_price)}</span>
                    <span>{formatPrice(getPositionEntryPrice(openPosition))}</span>
                    <span>{formatPrice(openPosition.target_price ?? openPosition.tp_price)}</span>
                  </div>
                  <div className="relative h-3 rounded-full bg-neutral-900 border border-neutral-800 overflow-hidden">
                    <div className="absolute inset-y-0 left-1/2 w-px bg-neutral-700" />
                    <div
                      className={`absolute top-0 h-full w-3 rounded-full ${
                        (openPosition.open_pnl_dollars || 0) >= 0
                          ? 'bg-emerald-300 shadow-[0_0_18px_rgba(52,211,153,0.35)]'
                          : 'bg-rose-300 shadow-[0_0_18px_rgba(251,113,133,0.35)]'
                      }`}
                      style={{ left: `calc(${positionMarkerPercent(openPosition, openPositionCurrentPrice)}% - 6px)` }}
                    />
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <p className="text-neutral-500">Open Points</p>
                    <p className={`font-semibold ${(openPosition.open_pnl_points || 0) >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                      {formatPoints(openPosition.open_pnl_points)}
                    </p>
                  </div>
                  <div>
                    <p className="text-neutral-500">Open Dollars</p>
                    <p className={`font-semibold ${(openPosition.open_pnl_dollars || 0) >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                      {formatMoney(openPosition.open_pnl_dollars)}
                    </p>
                  </div>
                </div>
              </div>
            ) : (
              <div className="rounded-lg border border-dashed border-neutral-800 px-4 py-10 text-center text-sm text-neutral-500">
                No filterless position is currently open.
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
                    {openPosition?.entry_price != null && (
                      <Line
                        type="monotone"
                        dataKey={() => openPosition.entry_price as number}
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
              <div className="flex items-center gap-2">
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
            {/* Summary row */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-5">
              <div className="rounded-lg border border-neutral-800 bg-neutral-950/60 px-3 py-3">
                <p className="text-[10px] font-semibold uppercase tracking-[0.18em] text-neutral-500">Active Event</p>
                <p className="mt-1 text-sm font-medium text-neutral-200">{kalshiMetrics?.event_ticker || '--'}</p>
                <p className="mt-1 text-xs text-neutral-500">{kalshiMetrics?.source || 'snapshot'}</p>
              </div>
              <div className="rounded-lg border border-neutral-800 bg-neutral-950/60 px-3 py-3">
                <p className="text-[10px] font-semibold uppercase tracking-[0.18em] text-neutral-500">60-Min {kalshiPredictionLabel} Probability</p>
                <p className="mt-1 text-lg font-semibold text-sky-300">{formatPercent(kalshiMetrics?.probability_60m, 2)}</p>
                <p className="mt-1 text-xs text-neutral-500">{kalshiProbabilityCaption}</p>
              </div>
              <div className="rounded-lg border border-neutral-800 bg-neutral-950/60 px-3 py-3">
                <p className="text-[10px] font-semibold uppercase tracking-[0.18em] text-neutral-500">ES Reference</p>
                <p className="mt-1 text-lg font-semibold text-neutral-100">{formatPrice(kalshiEsReferencePrice)}</p>
                <p className="mt-1 text-xs text-neutral-500">Converted from SPX using basis offset {formatPrice(kalshiMetrics?.basis_offset)}</p>
              </div>
              <div className={`rounded-lg border px-3 py-3 ${
                kalshiMetrics?.trade_gating_active
                  ? 'border-amber-500/30 bg-amber-500/5'
                  : 'border-neutral-800 bg-neutral-950/60'
              }`}>
                <p className="text-[10px] font-semibold uppercase tracking-[0.18em] text-neutral-500">Trade Gating</p>
                <p className={`mt-1 text-lg font-semibold ${kalshiMetrics?.trade_gating_active ? 'text-amber-300' : 'text-neutral-400'}`}>
                  {kalshiMetrics?.trade_gating_active
                    ? `Active 3x — ${etHourToLocalLabel(kalshiMetrics.trade_gating_hour!)}`
                    : 'Observe Only'}
                </p>
                <p className="mt-1 text-xs text-neutral-500">
                  {kalshiMetrics?.status_reason || 'Gating activates during 10 AM – 4 PM ET settlement hours with 3x sizing.'}
                </p>
              </div>
            </div>

            {/* Hourly contract schedule — clickable buttons */}
            <div className="mb-5">
              <div className="flex items-center justify-between mb-3">
                <p className="text-[10px] font-semibold uppercase tracking-[0.18em] text-neutral-500">Hourly Contract Schedule — 10 AM – 4 PM ET (Your Local Time)</p>
                {selectedKalshiHour != null && (
                  <button
                    onClick={() => setSelectedKalshiHour(null)}
                    className="text-[10px] text-sky-400 hover:text-sky-300 uppercase tracking-wide cursor-pointer"
                  >
                    Show Active
                  </button>
                )}
              </div>
              <div className="grid grid-cols-7 gap-2">
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
                    statusLabel = 'GATING';
                    statusColor = 'text-amber-400';
                  } else if (contract.available) {
                    statusLabel = 'AVAILABLE';
                    statusColor = 'text-emerald-400';
                  } else if (contract.upcoming) {
                    statusLabel = 'NOT YET';
                    statusColor = 'text-neutral-500';
                  } else {
                    statusLabel = 'SETTLED';
                    statusColor = 'text-neutral-600';
                  }

                  return (
                    <button
                      key={contract.hour}
                      onClick={() => setSelectedKalshiHour(isSelected ? null : contract.hour)}
                      className={`rounded-lg border px-3 py-3 text-center cursor-pointer transition-all hover:brightness-125 ${borderClass} ${bgClass} ${
                        isViewed ? 'ring-1 ring-sky-500/30' : ''
                      }`}
                    >
                      <p className={`text-sm font-bold ${
                        isViewed ? 'text-sky-300' : contract.tradeGating ? 'text-amber-300' : contract.available ? 'text-emerald-300' : 'text-neutral-500'
                      }`}>
                        {contract.label}
                      </p>
                      <p className={`mt-1 text-[10px] font-semibold uppercase tracking-wide ${statusColor}`}>
                        {statusLabel}
                      </p>
                      {contract.isActive && contract.strikeCount > 0 && (
                        <p className="mt-1 text-[10px] text-sky-400/70">{contract.strikeCount} strikes</p>
                      )}
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Contract detail for selected/active hour */}
            {viewedKalshiContract && (
              <div className="mb-4 rounded-lg border border-neutral-800 bg-neutral-950/50 px-4 py-3">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-xs font-semibold text-neutral-400">
                      {viewedKalshiContract.label} Contract &mdash; {viewedKalshiContract.eventTicker}
                    </p>
                    <p className="mt-1 text-xs text-neutral-500">
                      {viewedKalshiContract.tradeGating
                        ? 'Trade gating is active with 3x sizing for this settlement hour (70% accuracy).'
                        : viewedKalshiContract.available && (KALSHI_GATING_HOURS as readonly number[]).includes(viewedKalshiContract.hour)
                          ? `Tradable now. Settles at ${etHourToLocalLabel(viewedKalshiContract.hour)} with 3x trade gating.`
                          : viewedKalshiContract.available && [10, 11].includes(viewedKalshiContract.hour)
                            ? `Tradable now. Observe only — crowd unreliable in morning hours.`
                            : 'This contract has settled.'}
                    </p>
                  </div>
                  <span className={`rounded-full border px-3 py-1 text-[10px] font-semibold uppercase tracking-wide ${
                    viewedKalshiContract.tradeGating
                      ? 'bg-amber-500/10 text-amber-300 border-amber-500/30'
                      : viewedKalshiContract.available
                        ? 'bg-emerald-500/10 text-emerald-300 border-emerald-500/30'
                        : 'bg-neutral-800 text-neutral-400 border-neutral-700'
                  }`}>
                    {viewedKalshiContract.tradeGating ? 'Gating' : viewedKalshiContract.available ? 'Available' : viewedKalshiContract.upcoming ? 'Upcoming' : 'Settled'}
                  </span>
                </div>
              </div>
            )}

            {/* Strike ladder — shown when viewing the active contract */}
            {(selectedKalshiHour == null || viewedKalshiContract?.isActive) && kalshiStrikes.length > 0 ? (
              <div>
                <p className="text-[10px] font-semibold uppercase tracking-[0.18em] text-neutral-500 mb-3">
                  Strikes &mdash; {kalshiMetrics?.event_ticker || 'N/A'} &mdash; {kalshiPredictionLabel} Side
                </p>
                <div className="overflow-x-auto">
                  <table className="min-w-full text-sm">
                    <thead className="text-left text-neutral-500 border-b border-neutral-800">
                      <tr>
                        <th className="py-2 pr-4 font-medium">S&amp;P 500 Strike</th>
                        <th className="py-2 pr-4 font-medium">{kalshiPredictionLabel} Probability</th>
                        <th className="py-2 pr-4 font-medium">Volume</th>
                        <th className="py-2 pr-4 font-medium">Status</th>
                      </tr>
                    </thead>
                    <tbody>
                      {kalshiDisplayedStrikes.map((strike, index) => (
                        <tr
                          key={`${strike.strike}-${index}`}
                          className={`border-b border-neutral-900 last:border-b-0 ${
                            nearestKalshiStrike && nearestKalshiStrike.strike === strike.strike
                              ? 'bg-sky-500/5'
                              : ''
                          }`}
                        >
                          <td className="py-2 pr-4 text-neutral-200">{formatStrikeLabel(strike.strike)}</td>
                          <td className="py-2 pr-4 text-sky-300 font-medium">{formatPercent(strike.displayProbability, 2)}</td>
                          <td className="py-2 pr-4 text-neutral-300">{formatCompactNumber(strike.volume)}</td>
                          <td className="py-2 pr-4 text-neutral-400">{strike.status || '--'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            ) : selectedKalshiHour != null && !viewedKalshiContract?.isActive ? (
              <div className="rounded-lg border border-dashed border-neutral-800 px-4 py-6 text-center text-sm text-neutral-500">
                Strike data is only available for the currently active contract. Select the active hour or click &ldquo;Show Active&rdquo; to view strikes.
              </div>
            ) : (
              <div className="rounded-lg border border-dashed border-neutral-800 px-4 py-6 text-center text-sm text-neutral-500">
                {kalshiMetrics?.status_reason || 'Kalshi strike ladder is not available yet. Contracts become available as each hourly window opens.'}
              </div>
            )}
          </Panel>
        )}

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
              <div className="overflow-x-auto">
                <table className="min-w-full text-sm">
                  <thead className="text-left text-neutral-500 border-b border-neutral-800">
                    <tr>
                      <th className="py-2 pr-4 font-medium">Time</th>
                      <th className="py-2 pr-4 font-medium">Strategy</th>
                      <th className="py-2 pr-4 font-medium">Side</th>
                      <th className="py-2 pr-4 font-medium">Entry</th>
                      <th className="py-2 pr-4 font-medium">Exit</th>
                      <th className="py-2 pr-4 font-medium">PnL</th>
                    </tr>
                  </thead>
                  <tbody>
                    {state.trades.map((trade: FilterlessTrade, index) => (
                      <tr key={`${trade.time}-${index}`} className="border-b border-neutral-900 last:border-b-0">
                        <td className="py-3 pr-4 text-neutral-400">{formatTimestamp(trade.time)}</td>
                        <td className="py-3 pr-4 text-neutral-100">{trade.strategy_label}</td>
                        <td className="py-3 pr-4 text-neutral-200">{trade.side}</td>
                        <td className="py-3 pr-4 text-neutral-300">{formatPrice(trade.entry_price)}</td>
                        <td className="py-3 pr-4 text-neutral-300">{formatPrice(trade.exit_price)}</td>
                        <td className={`py-3 pr-4 font-semibold ${(trade.pnl_dollars || 0) >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                          {formatMoney(trade.pnl_dollars)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
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
