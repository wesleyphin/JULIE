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

function formatGateSummary(prob?: number | null, threshold?: number | null): string {
  if (prob == null && threshold == null) return '--';
  if (prob == null) return `min ${formatPercent(threshold)}`;
  if (threshold == null) return formatPercent(prob);
  return `${formatPercent(prob)} >= ${formatPercent(threshold)}`;
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
          <StatsCard title="Open Position" value={openPosition ? openPosition.side : 'FLAT'} icon={openPosition ? (openPosition.side === 'LONG' ? ArrowUpRight : ArrowDownRight) : BrainCircuit} color={openPosition ? 'warning' : 'default'} />
          <StatsCard title="Open PnL" value={formatMoney(openPosition?.open_pnl_dollars)} icon={ShieldAlert} color={openPosition ? openPnlColor : 'default'} />
        </div>

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
                    <p className="text-neutral-100 font-medium">{formatPrice(openPosition.entry_price)}</p>
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
