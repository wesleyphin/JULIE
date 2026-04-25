import React, { startTransition, useEffect, useMemo, useRef, useState } from 'react';
import type {
  FilterlessEvent,
  FilterlessKalshiMetrics,
  FilterlessLiveState,
  FilterlessPosition,
  FilterlessSentimentMetrics,
  FilterlessStrategyState,
  FilterlessTrade,
} from './filterlessLiveTypes';

const REFRESH_MS = 3000;
const FEED_STALE_SECONDS = 90;
const FEED_OFFLINE_SECONDS = 300;
const TAU = Math.PI * 2;

const COLORS = {
  purple: '#a855ff',
  violet: '#6d5dff',
  cyan: '#35f5ff',
  pink: '#ff3df2',
  amber: '#ffb347',
  lime: '#c7ff4a',
  red: '#ff3864',
  green: '#45ffc8',
  muted: '#9c8bbb',
  text: '#f4ecff',
  dim: '#5a4a72',
};

type ScreenId = 'overview' | 'aetherflow' | 'kalshi' | 'news' | 'strategies' | 'journal' | 'command';

type ManifoldRegime = 'TREND_GEODESIC' | 'CHOP_SPIRAL' | 'DISPERSED' | 'ROTATIONAL_TURBULENCE';
type BadgeTone = 'live' | 'watch' | 'block' | 'info';

interface AetherFeatures {
  pressure10: number;
  pressure30: number;
  flowFast: number;
  flowSlow: number;
  emaSpread: number;
  directionalBias: number;
  alignment: number;
  smoothness: number;
  dispersion: number;
  meanAbsDphi: number;
  stress: number;
  novelty: number;
  transitionEnergy: number;
  flowMagFast: number;
  burstPressure: number;
  R: number;
  regime: ManifoldRegime;
  riskMult: number;
  noTrade: boolean;
  trendPersistence: number;
  alignedFlow: number;
  transitionBurst: number;
  nypmTrendEdge: number;
  chopFadeEdge: number;
  price: number | null;
  spread: number;
  edge: number;
  truthScore: number;
  truthConfidence: number;
  headlineRisk: number;
  advisoryPressure: number;
  truthRiskWatch: boolean;
  foldDepth: number;
}

interface SceneState {
  yaw: number;
  pitch: number;
  zoom: number;
}

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
  metadata: null,
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
    trading_day_start: null,
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

const NAV: Array<{ id: ScreenId; label: string; code: string; title: string; subtitle: string }> = [
  { id: 'overview', label: 'Market Core', code: '01', title: 'MARKET CORE', subtitle: 'Live execution cockpit with entry, stop, target, news, and position context.' },
  { id: 'aetherflow', label: 'AetherFlow', code: '02', title: 'AETHERFLOW MANIFOLD', subtitle: 'Rotatable folded pressure surface with state shading, depth, and route features.' },
  { id: 'kalshi', label: 'Kalshi', code: '03', title: 'KALSHI MARKET ARRAY', subtitle: 'Hourly contract routing, book edge, spread, and consensus flow.' },
  { id: 'news', label: 'News', code: '04', title: 'NEWS AND TRUTH MONITOR', subtitle: 'Truth Social sentiment, macro tape, advisory risk, and calendar context.' },
  { id: 'strategies', label: 'Strategies', code: '05', title: 'STRATEGY STACK', subtitle: 'Filterless strategy modules mapped to current live state.' },
  { id: 'journal', label: 'Journal', code: '06', title: 'TRACE JOURNAL', subtitle: 'Decision log with trade levels, news fields, and manifold snapshots.' },
  { id: 'command', label: 'Command', code: '07', title: 'COMMAND MATRIX', subtitle: 'Runtime controls, guard rails, and operator actions.' },
];

const COCKPIT_CSS = `
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Orbitron:wght@500;700;800&family=Rajdhani:wght@400;500;600;700&display=swap');
:root {
  --bg: #000;
  --line: rgba(160, 90, 255, 0.23);
  --line-strong: rgba(190, 118, 255, 0.5);
  --text: #f4ecff;
  --muted: #9c8bbb;
  --dim: #5a4a72;
  --purple: #a855ff;
  --violet: #6d5dff;
  --cyan: #35f5ff;
  --pink: #ff3df2;
  --amber: #ffb347;
  --lime: #c7ff4a;
  --red: #ff3864;
  --green: #45ffc8;
  --shadow: 0 0 0 1px rgba(168, 85, 255, 0.07), 0 18px 60px rgba(0, 0, 0, 0.62);
  --display: "Orbitron", "Rajdhani", system-ui, sans-serif;
  --body: "Rajdhani", system-ui, sans-serif;
  --mono: "JetBrains Mono", ui-monospace, monospace;
}
body { background: #000; color: var(--text); font-family: var(--body); letter-spacing: 0; overflow-x: hidden; }
button { font: inherit; color: inherit; }
canvas { display: block; width: 100%; }
h1, h2, h3, p { margin: 0; }
.fl-cockpit * { box-sizing: border-box; }
.app { min-height: 100vh; display: grid; grid-template-columns: 248px minmax(0, 1fr); background: #000; }
.rail { min-width: 0; border-right: 1px solid var(--line); background: #050208; display: grid; grid-template-rows: auto 1fr auto; }
.brand { min-width: 0; padding: 18px 16px 16px; border-bottom: 1px solid var(--line); }
.brand h1 { font-family: var(--display); font-size: 13px; letter-spacing: 1.8px; color: var(--purple); text-shadow: 0 0 18px rgba(168, 85, 255, 0.9); }
.brand p { margin-top: 8px; color: var(--muted); font-size: 12px; line-height: 1.35; font-family: var(--mono); }
.brand .wire { height: 2px; margin-top: 15px; background: linear-gradient(90deg, var(--purple), transparent 70%); box-shadow: 0 0 18px rgba(168, 85, 255, 0.8); }
.nav { min-width: 0; padding: 12px; display: grid; align-content: start; gap: 6px; }
.nav button { min-width: 0; height: 40px; border: 1px solid transparent; background: transparent; border-radius: 0; display: grid; grid-template-columns: minmax(0, 1fr) auto; align-items: center; gap: 10px; padding: 0 9px; cursor: pointer; color: var(--muted); text-align: left; text-transform: uppercase; }
.nav button:hover, .nav button.active { color: var(--text); border-color: var(--line-strong); background: rgba(168, 85, 255, 0.075); box-shadow: inset 2px 0 0 var(--purple), 0 0 20px rgba(168, 85, 255, 0.14); }
.nav span { min-width: 0; font-weight: 700; letter-spacing: 0.8px; font-size: 12px; }
.nav small, .micro { font-family: var(--mono); font-size: 10px; color: var(--dim); }
.rail-bottom { min-width: 0; padding: 14px; border-top: 1px solid var(--line); display: grid; gap: 8px; }
.status-line { min-width: 0; display: grid; grid-template-columns: minmax(0, 1fr) auto; align-items: center; gap: 10px; font-family: var(--mono); font-size: 10px; color: var(--muted); }
.status-line strong { min-width: 0; color: var(--text); font-weight: 700; }
.dot { display: inline-block; width: 6px; height: 6px; margin-right: 6px; border-radius: 50%; background: var(--green); box-shadow: 0 0 14px var(--green); }
.dot.warn-dot { background: var(--amber); box-shadow: 0 0 14px var(--amber); }
.dot.down-dot { background: var(--red); box-shadow: 0 0 14px var(--red); }
.deck { min-width: 0; padding: 14px; display: grid; grid-template-rows: auto auto auto 1fr; gap: 10px; }
.top { min-width: 0; display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 14px; align-items: start; border-bottom: 1px solid var(--line); padding-bottom: 10px; }
.kicker { font-family: var(--mono); color: var(--muted); font-size: 10px; text-transform: uppercase; }
.top h2 { margin-top: 4px; font-family: var(--display); color: var(--text); font-size: clamp(19px, 2vw, 30px); letter-spacing: 1px; text-shadow: 0 0 24px rgba(168, 85, 255, 0.5); }
.top p { margin-top: 6px; color: var(--muted); font-family: var(--mono); font-size: 11px; }
.actions { min-width: 0; display: flex; justify-content: flex-end; flex-wrap: wrap; gap: 8px; }
.chip, .command, .badge { min-width: 0; border: 1px solid var(--line); background: rgba(168, 85, 255, 0.06); color: var(--text); height: 30px; padding: 0 10px; display: inline-flex; align-items: center; justify-content: center; gap: 7px; font-family: var(--mono); font-size: 10px; text-transform: uppercase; white-space: nowrap; }
.command { cursor: pointer; text-decoration: none; }
.command.primary { border-color: rgba(53, 245, 255, 0.6); color: var(--cyan); box-shadow: 0 0 18px rgba(53, 245, 255, 0.13); }
.notice { border: 1px solid rgba(255, 56, 100, 0.32); color: var(--red); background: rgba(255, 56, 100, 0.06); padding: 10px 12px; font-family: var(--mono); font-size: 11px; }
.ticker { min-width: 0; display: grid; grid-template-columns: repeat(8, minmax(0, 1fr)); border: 1px solid var(--line); background: #06020b; }
.ticker .cell { min-width: 0; padding: 9px 10px; border-right: 1px solid rgba(155, 86, 255, 0.14); }
.ticker .cell:last-child { border-right: 0; }
.ticker span, .label { display: block; min-width: 0; color: var(--muted); font-size: 10px; font-family: var(--mono); text-transform: uppercase; }
.ticker strong { display: block; margin-top: 3px; color: var(--purple); font-family: var(--display); font-size: clamp(15px, 1.3vw, 22px); text-shadow: 0 0 18px rgba(168, 85, 255, 0.62); }
.screen { min-width: 0; display: block; }
.grid { min-width: 0; display: grid; gap: 10px; }
.cols-2 { grid-template-columns: repeat(2, minmax(0, 1fr)); }
.cols-3 { grid-template-columns: repeat(3, minmax(0, 1fr)); }
.cols-4 { grid-template-columns: repeat(4, minmax(0, 1fr)); }
.overview-layout { grid-template-columns: minmax(260px, 0.78fr) minmax(0, 1.55fr) minmax(280px, 0.78fr); margin-top: 10px; }
.aether-layout { grid-template-columns: minmax(0, 1.52fr) minmax(340px, 0.72fr); margin-top: 10px; }
.kalshi-layout, .news-layout, .journal-layout { grid-template-columns: minmax(0, 1.35fr) minmax(320px, 0.65fr); margin-top: 10px; }
.panel, .metric, .event, .position, .terminal-row, .tile { min-width: 0; background: rgba(10, 5, 18, 0.94); border: 1px solid var(--line); box-shadow: var(--shadow); }
.panel { overflow: hidden; }
.panel-head { min-width: 0; min-height: 48px; display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 10px; align-items: center; padding: 10px 12px; border-bottom: 1px solid rgba(155, 86, 255, 0.18); }
.panel-head h3 { min-width: 0; font-family: var(--display); font-size: 12px; letter-spacing: 1px; text-transform: uppercase; }
.panel-head p { min-width: 0; margin-top: 3px; color: var(--muted); font-family: var(--mono); font-size: 10px; }
.panel-body { padding: 10px; }
.metric { height: 86px; padding: 11px; display: grid; align-content: space-between; border-color: rgba(155, 86, 255, 0.2); }
.metric strong { min-width: 0; font-family: var(--display); font-size: clamp(18px, 1.8vw, 30px); color: var(--metric, var(--purple)); text-shadow: 0 0 18px color-mix(in srgb, var(--metric, var(--purple)), transparent 48%); }
.metric small { min-width: 0; color: var(--muted); font-family: var(--mono); font-size: 10px; }
.stack { min-width: 0; display: grid; gap: 10px; align-content: start; }
.terminal { padding: 8px; display: grid; gap: 6px; max-height: 520px; overflow: auto; }
.terminal-row { min-height: 38px; padding: 7px 8px; display: grid; grid-template-columns: 64px minmax(0, 1fr) auto; align-items: center; gap: 8px; font-family: var(--mono); font-size: 10px; }
.terminal-row time { color: var(--dim); }
.terminal-row strong { min-width: 0; color: var(--text); font-weight: 700; }
.terminal-row p { color: var(--muted); }
.badge { height: 22px; padding: 0 7px; border-color: rgba(168, 85, 255, 0.32); color: var(--purple); }
.badge.live { color: var(--green); border-color: rgba(69, 255, 200, 0.45); }
.badge.watch { color: var(--amber); border-color: rgba(255, 179, 71, 0.45); }
.badge.block { color: var(--red); border-color: rgba(255, 56, 100, 0.52); }
.badge.info { color: var(--cyan); border-color: rgba(53, 245, 255, 0.45); }
.chart { height: 370px; background: #050109; }
.scene-wrap { position: relative; min-height: 650px; background: #030006; }
.aetherflow-scene { width: 100%; height: 650px; cursor: grab; touch-action: none; user-select: none; }
.aetherflow-scene:active { cursor: grabbing; }
.scene-hud { position: absolute; left: 12px; right: 12px; bottom: 12px; display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px; pointer-events: none; }
.hud-cell { min-width: 0; border: 1px solid rgba(168, 85, 255, 0.2); background: rgba(4, 1, 8, 0.74); padding: 8px; }
.hud-cell strong { display: block; margin-top: 2px; font-family: var(--mono); font-size: 11px; }
.mini-grid { min-width: 0; display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); border-top: 1px solid rgba(155, 86, 255, 0.18); }
.mini { min-width: 0; padding: 9px 10px; border-right: 1px solid rgba(155, 86, 255, 0.15); }
.mini:last-child { border-right: 0; }
.mini strong { display: block; min-width: 0; margin-top: 3px; font-family: var(--mono); font-size: 12px; }
.feature { min-width: 0; padding: 8px 0; border-bottom: 1px solid rgba(155, 86, 255, 0.13); }
.feature:last-child { border-bottom: 0; }
.feature-head { min-width: 0; display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 8px; margin-bottom: 7px; font-family: var(--mono); font-size: 10px; color: var(--muted); text-transform: uppercase; }
.meter { height: 8px; background: rgba(255, 255, 255, 0.045); border: 1px solid rgba(155, 86, 255, 0.18); overflow: hidden; }
.meter span { display: block; height: 100%; width: 50%; background: linear-gradient(90deg, var(--meter, var(--purple)), rgba(255, 255, 255, 0.2)); box-shadow: 0 0 14px var(--meter, var(--purple)); }
.state-matrix, .truth-grid { min-width: 0; display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }
.tile { min-height: 88px; padding: 10px; display: grid; align-content: space-between; }
.tile.active { border-color: var(--tile, var(--purple)); box-shadow: 0 0 0 1px color-mix(in srgb, var(--tile, var(--purple)), transparent 72%), 0 0 26px color-mix(in srgb, var(--tile, var(--purple)), transparent 76%); }
.tile-row { min-width: 0; display: grid; grid-template-columns: minmax(0, 1fr) auto; align-items: center; gap: 8px; }
.tile strong { min-width: 0; font-family: var(--mono); font-size: 11px; }
.tile p { min-width: 0; margin-top: 8px; color: var(--muted); font-size: 12px; line-height: 1.25; }
.position { min-height: 52px; padding: 9px; display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 10px; align-items: center; }
.position strong, .position span { min-width: 0; }
.position strong { font-family: var(--mono); font-size: 11px; }
.position p { color: var(--muted); font-family: var(--mono); font-size: 10px; }
.flow-bars { display: grid; gap: 4px; padding: 10px; }
.flow-bar { min-width: 0; height: 21px; display: grid; grid-template-columns: 54px minmax(0, 1fr) 52px; align-items: center; gap: 8px; font-family: var(--mono); font-size: 10px; color: var(--muted); }
.bar-track { height: 16px; background: rgba(255, 255, 255, 0.045); border: 1px solid rgba(155, 86, 255, 0.13); overflow: hidden; }
.bar-fill { height: 100%; width: 50%; background: linear-gradient(90deg, rgba(168, 85, 255, 0.22), var(--purple)); }
.table { width: 100%; border-collapse: collapse; table-layout: fixed; font-family: var(--mono); font-size: 10px; }
.table th, .table td { min-width: 0; padding: 9px 8px; border-bottom: 1px solid rgba(155, 86, 255, 0.13); text-align: left; color: var(--muted); }
.table th { color: var(--dim); text-transform: uppercase; font-weight: 700; }
.table td strong { color: var(--text); }
.event { min-height: 58px; padding: 9px; display: grid; grid-template-columns: 72px minmax(0, 1fr) auto; gap: 10px; align-items: center; font-family: var(--mono); font-size: 10px; }
.event time { color: var(--dim); }
.event strong { min-width: 0; display: block; color: var(--text); }
.event p { min-width: 0; margin-top: 4px; color: var(--muted); }
.command-grid { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 8px; }
.command-tile { height: 118px; border: 1px solid var(--line); background: rgba(10, 5, 18, 0.94); padding: 10px; display: grid; align-content: space-between; }
.command-tile strong { min-width: 0; font-family: var(--display); font-size: 11px; }
.command-tile small { min-width: 0; color: var(--muted); font-family: var(--mono); font-size: 10px; }
.up { color: var(--green) !important; }
.down { color: var(--red) !important; }
.info { color: var(--cyan) !important; }
.warn { color: var(--amber) !important; }
.violet { color: var(--purple) !important; }
.truncate { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.clamp { display: -webkit-box; -webkit-box-orient: vertical; -webkit-line-clamp: 2; overflow: hidden; }
@media (max-width: 1180px) {
  .app { grid-template-columns: 1fr; }
  .rail { position: sticky; top: 0; z-index: 5; grid-template-rows: auto auto; border-right: 0; border-bottom: 1px solid var(--line); }
  .brand { display: none; }
  .nav { grid-template-columns: repeat(7, minmax(0, 1fr)); }
  .rail-bottom { display: none; }
  .overview-layout, .aether-layout, .kalshi-layout, .news-layout, .journal-layout { grid-template-columns: 1fr; }
}
@media (max-width: 820px) {
  .deck { padding: 10px; }
  .top, .ticker, .cols-2, .cols-3, .cols-4, .mini-grid, .state-matrix, .truth-grid, .command-grid, .scene-hud { grid-template-columns: 1fr; }
  .nav { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .nav button { height: 34px; }
  .actions { justify-content: flex-start; }
  .scene-wrap { min-height: 500px; }
  .aetherflow-scene { height: 500px; }
}
`;

function clip(value: number, low: number, high: number): number {
  return Math.max(low, Math.min(high, value));
}

function clip01(value: number): number {
  return clip(value, 0, 1);
}

function fmt(value?: number | null, digits = 2): string {
  if (value == null || Number.isNaN(value)) return '--';
  return Number(value).toFixed(digits);
}

function pct(value?: number | null, digits = 0): string {
  if (value == null || Number.isNaN(value)) return '--';
  return `${(clip01(value) * 100).toFixed(digits)}%`;
}

function formatMoney(value?: number | null): string {
  if (value == null || Number.isNaN(value)) return '--';
  return `${value >= 0 ? '+$' : '-$'}${Math.abs(value).toFixed(0)}`;
}

function formatSigned(value?: number | null, digits = 2): string {
  if (value == null || Number.isNaN(value)) return '--';
  return `${value >= 0 ? '+' : ''}${value.toFixed(digits)}`;
}

function formatPrice(value?: number | null): string {
  if (value == null || Number.isNaN(value)) return '--';
  return value.toFixed(2);
}

function formatToken(value?: string | null): string {
  if (!value) return '--';
  return value.replace(/_/g, ' ');
}

function formatShortTime(value?: string | null): string {
  if (!value) return '--';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return '--';
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function formatRelativeTime(value?: string | null): string {
  if (!value) return 'no feed';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return 'no feed';
  const diffSeconds = Math.max(0, Math.round((Date.now() - date.getTime()) / 1000));
  if (diffSeconds < 60) return `${diffSeconds}s ago`;
  if (diffSeconds < 3600) return `${Math.round(diffSeconds / 60)}m ago`;
  return `${Math.round(diffSeconds / 3600)}h ago`;
}

function kalshiEventHour(metrics?: FilterlessKalshiMetrics | null): number | null {
  const match = String(metrics?.event_ticker || '').match(/H(\d{3,4})$/i);
  if (!match) return null;
  const raw = Number(match[1]);
  if (!Number.isFinite(raw)) return null;
  const hour = Math.floor(raw / 100);
  return hour >= 0 && hour <= 23 ? hour : null;
}

function activeKalshiGatingHour(generatedAt?: string | null): number | null {
  if (!generatedAt) return null;
  const date = new Date(generatedAt);
  if (Number.isNaN(date.getTime())) return null;
  const parts = new Intl.DateTimeFormat('en-US', {
    timeZone: 'America/New_York',
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
  }).formatToParts(date);
  const hour = Number(parts.find((part) => part.type === 'hour')?.value);
  const minute = Number(parts.find((part) => part.type === 'minute')?.value);
  if (!Number.isFinite(hour) || !Number.isFinite(minute)) return null;
  for (const settlementHour of [12, 13, 14, 15, 16]) {
    if (settlementHour > hour || (settlementHour === hour && minute < 5)) return settlementHour;
  }
  return null;
}

function kalshiRouteState(metrics?: FilterlessKalshiMetrics | null, generatedAt?: string | null): {
  value: string;
  row: string;
  badge: string;
  tone: BadgeTone;
  color: string;
  hint: string;
  detail: string;
  impact: string;
} {
  const statusText = String(metrics?.status_label || metrics?.status_reason || '').toLowerCase();
  const morningStandby = statusText.includes('morning') || statusText.includes('10') || statusText.includes('11');
  const inferredGatingHour = activeKalshiGatingHour(generatedAt);
  const eventHour = kalshiEventHour(metrics);
  const activeBySnapshotTime = inferredGatingHour != null && eventHour === inferredGatingHour;
  if (!metrics?.enabled) {
    return {
      value: 'OFF',
      row: 'offline',
      badge: 'off',
      tone: 'block',
      color: COLORS.red,
      hint: 'provider disabled',
      detail: 'Kalshi provider is disabled or not requested.',
      impact: 'No strategy impact while the provider is off.',
    };
  }
  if (!metrics.healthy) {
    return {
      value: metrics.configured ? 'WATCH' : 'CONFIG',
      row: 'watch',
      badge: 'watch',
      tone: 'watch',
      color: COLORS.amber,
      hint: metrics.configured ? 'waiting for healthy ladder' : 'credentials or config missing',
      detail: metrics.status_reason || 'Kalshi is configured but the ladder is not healthy yet.',
      impact: 'Execution falls back to ML-only behavior until Kalshi is healthy.',
    };
  }
  if (metrics.trade_gating_active || activeBySnapshotTime) {
    const activeHour = metrics.trade_gating_hour ?? inferredGatingHour;
    const hour = activeHour == null ? 'active' : `${activeHour}:00 ET`;
    return {
      value: 'ARMED',
      row: 'gate',
      badge: 'armed',
      tone: 'live',
      color: COLORS.lime,
      hint: `active ${hour}`,
      detail: `Kalshi is actively gating the ${hour} settlement window.`,
      impact: 'Can block entries, trim or boost size, adjust TP, trail stops, and trigger hour-turn exits.',
    };
  }
  if (metrics.observer_only) {
    return {
      value: morningStandby ? 'MORNING' : 'STANDBY',
      row: 'standby',
      badge: 'standby',
      tone: 'info',
      color: COLORS.cyan,
      hint: morningStandby ? 'morning data-only window' : 'arms on 12-16 ET settlements',
      detail: morningStandby
        ? 'Morning contracts are monitored but not used for execution because the crowd was contrarian in testing.'
        : 'Kalshi data is live; execution overlay arms only during the 12-16 ET settlement windows.',
      impact: 'DE3, RegimeAdaptive, and AetherFlow receive Kalshi routing when the window is armed.',
    };
  }
  return {
    value: 'READY',
    row: 'ready',
    badge: 'ready',
    tone: 'watch',
    color: COLORS.purple,
    hint: 'live ladder available',
    detail: metrics.status_reason || 'Kalshi ladder is live and available for the execution overlay.',
    impact: 'Overlay impact depends on the active settlement window and strategy signal.',
  };
}

function describeKalshiPositionImpact(position?: FilterlessPosition | null): string {
  if (!position) return 'No active position is carrying a Kalshi overlay stamp.';
  if (position.kalshi_trade_overlay_applied) {
    const role = formatToken(position.kalshi_trade_overlay_role);
    const probability = pct(position.kalshi_entry_probability, 1);
    const support = fmt(position.kalshi_entry_support_score, 2);
    const threshold = fmt(position.kalshi_entry_threshold, 2);
    const trail = position.kalshi_tp_trail_enabled ? 'trail armed' : 'trail idle';
    return `${role} overlay, entry ${probability}, score ${support}/${threshold}, ${trail}`;
  }
  if (position.kalshi_gate_applied) {
    return `size gate ${fmt(position.kalshi_gate_multiplier, 2)}x, ${position.kalshi_gate_reason || 'no reason'}`;
  }
  const reason = position.kalshi_trade_overlay_reason || position.kalshi_gate_reason;
  return reason ? `No active Kalshi change: ${reason}` : 'No active Kalshi change on this position.';
}

function computeFeedStatus(status: string, generatedAt?: string | null): string {
  const date = generatedAt ? new Date(generatedAt) : null;
  if (!date || Number.isNaN(date.getTime())) return status;
  const ageSeconds = Math.max(0, Math.round((Date.now() - date.getTime()) / 1000));
  if (ageSeconds >= FEED_OFFLINE_SECONDS) return 'offline';
  if (ageSeconds >= FEED_STALE_SECONDS) return 'stale';
  return status;
}

function hexToRgb(hex: string): { r: number; g: number; b: number } {
  const clean = hex.replace('#', '');
  return {
    r: parseInt(clean.slice(0, 2), 16),
    g: parseInt(clean.slice(2, 4), 16),
    b: parseInt(clean.slice(4, 6), 16),
  };
}

function shadedColor(hex: string, shade: number, alpha = 1): string {
  const rgb = hexToRgb(hex);
  const s = clip(shade, 0, 1.5);
  return `rgba(${Math.round(rgb.r * s)}, ${Math.round(rgb.g * s)}, ${Math.round(rgb.b * s)}, ${alpha})`;
}

function alphaColor(hex: string, alpha: number): string {
  const rgb = hexToRgb(hex);
  return `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, ${alpha})`;
}

function getPositionEntryPrice(position?: FilterlessPosition | null): number | null {
  if (!position) return null;
  return position.entry_price ?? position.avg_price ?? position.signal_entry_price ?? null;
}

function getPositionCurrentPrice(position?: FilterlessPosition | null, fallback?: number | null): number | null {
  if (!position) return fallback ?? null;
  if (position.current_price != null && !Number.isNaN(position.current_price)) return position.current_price;
  const entry = getPositionEntryPrice(position);
  const openPoints = position.open_pnl_points;
  if (entry != null && openPoints != null && !Number.isNaN(openPoints)) {
    if (String(position.side).toUpperCase() === 'LONG') return entry + openPoints;
    if (String(position.side).toUpperCase() === 'SHORT') return entry - openPoints;
  }
  return fallback ?? null;
}

function positionStop(position?: FilterlessPosition | null): number | null {
  return position?.stop_price ?? position?.sl_price ?? null;
}

function positionTarget(position?: FilterlessPosition | null): number | null {
  return position?.target_price ?? position?.tp_price ?? null;
}

function sideTone(side?: string | null): string {
  const normalized = String(side || '').toUpperCase();
  if (normalized === 'LONG') return 'up';
  if (normalized === 'SHORT') return 'down';
  return 'info';
}

function statusBadge(status?: string | null): 'live' | 'watch' | 'block' | 'info' {
  const normalized = String(status || '').toLowerCase();
  if (['online', 'ready', 'idle', 'healthy'].includes(normalized)) return 'live';
  if (['stale', 'candidate', 'queued', 'in_trade'].includes(normalized)) return 'watch';
  if (['offline', 'blocked', 'error', 'disabled'].includes(normalized)) return 'block';
  return 'info';
}

function regimeColor(regime: ManifoldRegime): string {
  if (regime === 'TREND_GEODESIC') return COLORS.green;
  if (regime === 'DISPERSED') return COLORS.violet;
  if (regime === 'ROTATIONAL_TURBULENCE') return COLORS.red;
  return COLORS.cyan;
}

function dominantColor(name: string): string {
  if (name === 'trend') return COLORS.green;
  if (name === 'burst') return COLORS.amber;
  if (name === 'dispersed') return COLORS.violet;
  if (name === 'rot') return COLORS.red;
  return COLORS.cyan;
}

function deriveFeatures(
  liveState: FilterlessLiveState,
  effectiveStatus: string,
  openPositions: FilterlessPosition[],
  sentiment: FilterlessSentimentMetrics,
): AetherFeatures {
  const prices = liveState.bot.price_history
    .map((point) => point.price)
    .filter((price): price is number => price != null && !Number.isNaN(price));
  const fallbackPrice = prices.length ? prices[prices.length - 1] : null;
  const price = liveState.bot.price ?? fallbackPrice;
  const last = price ?? fallbackPrice ?? 0;
  const recent = prices.slice(-80);
  const min = recent.length ? Math.min(...recent) : last - 1;
  const max = recent.length ? Math.max(...recent) : last + 1;
  const range = Math.max(1, max - min);
  const previous = (bars: number) => prices[Math.max(0, prices.length - 1 - bars)] ?? last;
  const pressure10 = clip((last - previous(10)) / Math.max(1, range * 0.38), -1, 1);
  const pressure30 = clip((last - previous(30)) / Math.max(1, range * 0.52), -1, 1);
  const returns = recent.slice(1).map((value, index) => value - recent[index]);
  const avgAbsReturn = returns.length ? returns.reduce((sum, value) => sum + Math.abs(value), 0) / returns.length : 0;
  const volatility = clip01(avgAbsReturn / Math.max(0.25, range * 0.09));
  const sameDirection = Math.sign(pressure10 || 0.001) === Math.sign(pressure30 || 0.001);
  const aetherStrategy = liveState.strategies.find((strategy) => strategy.id === 'aetherflow');
  const blockedText = `${aetherStrategy?.last_block_reason || ''} ${aetherStrategy?.last_reason || ''}`.toLowerCase();
  const circuit = liveState.bot.risk.circuit_tripped === true || liveState.bot.risk.hostile_day_active === true;
  const losingPressure = clip01(Math.abs(liveState.bot.risk.daily_pnl ?? 0) / 1500);
  const truthScore = clip(sentiment.sentiment_score ?? 0, -1, 1);
  const truthConfidence = clip01(sentiment.finbert_confidence ?? 0);

  const alignment = clip01(0.34 + Math.abs(pressure30) * 0.34 + (sameDirection ? 0.16 : -0.08) + (aetherStrategy?.status === 'ready' ? 0.08 : 0));
  const smoothness = clip01(0.72 - volatility * 0.38 - Math.abs(pressure10 - pressure30) * 0.16);
  const dispersion = clip01(0.22 + volatility * 0.42 + (blockedText.includes('dispers') ? 0.24 : 0) + (effectiveStatus !== 'online' ? 0.12 : 0));
  const meanAbsDphi = clip01(0.18 + Math.abs(pressure10 - pressure30) * 0.45 + volatility * 0.24);
  const stress = clip01(0.2 + volatility * 0.28 + Math.abs(pressure10 - pressure30) * 0.32 + losingPressure * 0.18 + (circuit ? 0.36 : 0));
  const novelty = clip01(0.24 + Math.abs(truthScore) * 0.18 + dispersion * 0.24 + Math.abs(pressure30) * 0.16);
  const transitionEnergy = clip01(0.46 * stress + 0.24 * dispersion + 0.2 * novelty + 0.1 * Math.abs(pressure10 - pressure30));
  const directionalBias = clip((pressure10 * 0.58) + (pressure30 * 0.62), -1, 1);
  const flowFast = clip01(0.5 + pressure10 * 0.32);
  const flowSlow = clip01(0.5 + pressure30 * 0.24);
  const flowMagFast = clip01(flowFast * 0.7 + flowSlow * 0.3);
  const burstPressure = clip01(transitionEnergy * Math.abs(pressure30) * (0.5 + 0.5 * flowMagFast));
  const R = clip01((0.42 * alignment) + (0.34 * smoothness) + (0.24 * (1 - dispersion)) - (0.22 * stress));

  let regime: ManifoldRegime = 'CHOP_SPIRAL';
  if (circuit || (meanAbsDphi >= 0.62 && stress >= 0.62)) {
    regime = 'ROTATIONAL_TURBULENCE';
  } else if (alignment >= 0.62 && smoothness >= 0.56 && dispersion <= 0.5) {
    regime = 'TREND_GEODESIC';
  } else if (dispersion >= 0.66 && alignment <= 0.48) {
    regime = 'DISPERSED';
  }

  const noTrade = circuit || regime === 'DISPERSED' || regime === 'ROTATIONAL_TURBULENCE' || effectiveStatus !== 'online';
  const trendPersistence = clip01(0.45 + 0.28 * alignment + 0.18 * smoothness - 0.2 * stress);
  const alignedFlow = clip01(((directionalBias + 1) / 2) * alignment * smoothness * (1 - Math.max(0, stress - 0.35)));
  const transitionBurst = clip01(transitionEnergy * (0.6 * burstPressure + 0.4 * novelty));
  const nypmTrendEdge = clip01(alignedFlow * trendPersistence * (regime === 'TREND_GEODESIC' ? 1 : 0.48));
  const chopFadeEdge = clip01((regime === 'CHOP_SPIRAL' ? 1 : 0.45) * stress * (1 - Math.abs(directionalBias)) * (1 - Number(noTrade) * 0.55));
  const spread = clip01(0.026 + 0.04 * burstPressure + 0.025 * stress);
  const edge = clip01((liveState.kalshi_metrics?.probability_60m ?? 0.42) * 0.12 + 0.035 + 0.075 * alignedFlow + 0.05 * transitionBurst - 0.025 * Number(noTrade));
  const headlineRisk = clip01(0.28 + Math.abs(truthScore) * 0.24 + stress * 0.28);
  const advisoryPressure = clip01(Math.max(0, headlineRisk - 0.58) * 1.8 + (sentiment.last_error ? 0.22 : 0));
  const truthRiskWatch = advisoryPressure > 0.62;

  const features = {
    pressure10,
    pressure30,
    flowFast,
    flowSlow,
    emaSpread: pressure10 - pressure30,
    directionalBias,
    alignment,
    smoothness,
    dispersion,
    meanAbsDphi,
    stress,
    novelty,
    transitionEnergy,
    flowMagFast,
    burstPressure,
    R,
    regime,
    riskMult: clip(0.5 + (1.2 * R) - (0.8 * stress), 0.25, 1.5),
    noTrade,
    trendPersistence,
    alignedFlow,
    transitionBurst,
    nypmTrendEdge,
    chopFadeEdge,
    price,
    spread,
    edge,
    truthScore,
    truthConfidence,
    headlineRisk,
    advisoryPressure,
    truthRiskWatch,
    foldDepth: 0,
  };
  return { ...features, foldDepth: estimateFoldDepth(features) };
}

function surfaceHeight(x: number, y: number, f: AetherFeatures): number {
  const nx = x / 2.4;
  const ny = (y + 1.6) / 3.2;
  const trendRidge = f.alignment * f.smoothness * Math.exp(-((nx - f.pressure30) ** 2) / 0.11 - ((ny - 0.78) ** 2) / 0.05);
  const burstPeak = 1.35 * f.burstPressure * Math.exp(-((nx - Math.sign(f.pressure30 || 0.001) * 0.55) ** 2) / 0.08 - ((ny - f.transitionEnergy) ** 2) / 0.04);
  const chopShelf = f.stress * (1 - f.smoothness) * Math.exp(-((nx + 0.2 * f.directionalBias) ** 2) / 0.38 - ((ny - 0.38) ** 2) / 0.16);
  const dispersedBasin = -0.42 * f.dispersion * Math.exp(-(nx ** 2) / 0.34 - ((ny - 0.22) ** 2) / 0.06);
  const rotWall = 0.54 * f.stress * f.meanAbsDphi * (0.35 + 0.65 * Math.abs(Math.sin(5.5 * nx + 7.2 * ny))) * Math.exp(-((ny - 0.55) ** 2) / 0.25);
  const foldRipple = 0.08 * Math.sin(8 * nx + 2.5 * Math.sin(4 * ny)) * (0.35 + f.stress);
  const pressureSlope = 0.2 * f.directionalBias * nx + 0.13 * (f.transitionEnergy - 0.5) * ny;
  return clip(0.02 + trendRidge + burstPeak + chopShelf + rotWall + dispersedBasin + foldRipple + pressureSlope, -0.28, 1.64);
}

function visualSurfaceHeight(height: number): number {
  if (height <= 0) return height * 0.92;
  const peakBoost = 1.16 + 0.1 * clip01(height / 1.64);
  return clip(height * peakBoost, -0.26, 2.05);
}

function pressurePlaneHeight(x: number, y: number, f: AetherFeatures): number {
  const raw = surfaceHeight(x * 2.4, (y * 3.2) - 1.6, f);
  const directionalFold = 0.055 * Math.sin((7.4 * x) + (4.2 * y) + (1.7 * f.pressure30)) * (0.35 + f.stress + f.transitionEnergy);
  const streamFold = 0.04 * Math.cos((5.2 * y) - (3.2 * x) + (2.1 * f.directionalBias)) * (0.25 + f.alignment);
  return clip((visualSurfaceHeight(raw) * 0.54) + directionalFold + streamFold, -0.18, 1.12);
}

function dominantSurface(x: number, y: number, f: AetherFeatures): string {
  const nx = x / 2.4;
  const ny = (y + 1.6) / 3.2;
  const trend = f.alignment * f.smoothness * Math.exp(-((nx - f.pressure30) ** 2) / 0.11 - ((ny - 0.78) ** 2) / 0.05);
  const burst = 1.35 * f.burstPressure * Math.exp(-((nx - Math.sign(f.pressure30 || 0.001) * 0.55) ** 2) / 0.08 - ((ny - f.transitionEnergy) ** 2) / 0.04);
  const chop = f.stress * (1 - f.smoothness) * Math.exp(-((nx + 0.2 * f.directionalBias) ** 2) / 0.38 - ((ny - 0.38) ** 2) / 0.16);
  const dispersed = f.dispersion * Math.exp(-(nx ** 2) / 0.34 - ((ny - 0.22) ** 2) / 0.06);
  const rot = f.stress * f.meanAbsDphi * Math.exp(-((ny - 0.55) ** 2) / 0.25);
  return [
    ['trend', trend],
    ['burst', burst],
    ['chop', chop],
    ['dispersed', dispersed],
    ['rot', rot],
  ].sort((a, b) => Number(b[1]) - Number(a[1]))[0][0] as string;
}

function pressurePlaneDominant(x: number, y: number, f: AetherFeatures): string {
  return dominantSurface(x * 2.4, (y * 3.2) - 1.6, f);
}

function estimateFoldDepth(f: AetherFeatures): number {
  const samples = [
    pressurePlaneHeight(-0.5, 0.22, f),
    pressurePlaneHeight(0.1, 0.4, f),
    pressurePlaneHeight(0.55, 0.62, f),
    pressurePlaneHeight(-0.25, 0.78, f),
  ];
  return clip01((Math.max(...samples) - Math.min(...samples)) / 1.15);
}

function regimeDominantName(regime: ManifoldRegime): string {
  if (regime === 'TREND_GEODESIC') return 'trend';
  if (regime === 'DISPERSED') return 'dispersed';
  if (regime === 'ROTATIONAL_TURBULENCE') return 'rot';
  return 'chop';
}

function markerTargetForRegime(f: AetherFeatures): { x: number; y: number } {
  const target = regimeDominantName(f.regime);
  let desiredX = clip(f.pressure30, -0.98, 0.98);
  let desiredY = clip01(0.18 + 0.64 * f.transitionEnergy + 0.18 * f.novelty);
  if (target === 'trend') {
    desiredY = 0.78;
  } else if (target === 'chop') {
    desiredX = clip(-0.2 * f.directionalBias, -0.98, 0.98);
    desiredY = 0.38;
  } else if (target === 'dispersed') {
    desiredX = 0;
    desiredY = 0.22;
  } else if (target === 'rot') {
    desiredX = clip(f.pressure30 < 0 ? f.pressure30 : -0.72, -0.98, 0.98);
    desiredY = 0.55;
  }

  let best = { x: desiredX, y: desiredY };
  let bestScore = Infinity;
  for (let row = 0; row <= 34; row += 1) {
    const y = row / 34;
    for (let col = 0; col <= 52; col += 1) {
      const x = -0.98 + (1.96 * col) / 52;
      if (pressurePlaneDominant(x, y, f) !== target) continue;
      const height = pressurePlaneHeight(x, y, f);
      const distance = Math.hypot((x - desiredX) * 1.15, (y - desiredY) * 1.55);
      const score = distance - (height * 0.18);
      if (score < bestScore) {
        bestScore = score;
        best = { x, y };
      }
    }
  }
  return best;
}

function setupCanvas(canvas: HTMLCanvasElement): { ctx: CanvasRenderingContext2D; width: number; height: number; dpr: number } | null {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.floor(rect.width * dpr));
  const height = Math.max(1, Math.floor(rect.height * dpr));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  const ctx = canvas.getContext('2d');
  if (!ctx) return null;
  return { ctx, width, height, dpr };
}

function projectSurfacePoint(x: number, y: number, z: number, width: number, height: number, scene: SceneState) {
  const centeredY = (y - 0.5) * 1.56;
  const cosY = Math.cos(scene.yaw);
  const sinY = Math.sin(scene.yaw);
  const rx = (x * cosY) - (centeredY * sinY);
  const ry = (x * sinY) + (centeredY * cosY);
  const sx = width * 0.25 * scene.zoom;
  const sy = height * (0.16 + scene.pitch * 0.1) * scene.zoom;
  const lift = height * (0.31 + scene.pitch * 0.1) * scene.zoom;
  return {
    x: (width * 0.5) + ((rx - ry) * sx),
    y: (height * (0.62 + scene.pitch * 0.04)) + ((rx + ry) * sy) - (z * lift),
    depth: ry + z * 0.7,
    z,
  };
}

const Panel: React.FC<{ title: string; subtitle?: string; badge?: React.ReactNode; children: React.ReactNode; className?: string }> = ({
  title,
  subtitle,
  badge,
  children,
  className,
}) => (
  <div className={`panel ${className || ''}`}>
    <div className="panel-head">
      <div>
        <h3>{title}</h3>
        {subtitle ? <p className="truncate">{subtitle}</p> : null}
      </div>
      {badge}
    </div>
    {children}
  </div>
);

const Badge: React.FC<{ tone?: BadgeTone; children: React.ReactNode }> = ({ tone = 'info', children }) => (
  <span className={`badge ${tone}`}>{children}</span>
);

const Metric: React.FC<{ label: string; value: React.ReactNode; hint?: React.ReactNode; color?: string; className?: string }> = ({
  label,
  value,
  hint,
  color = COLORS.purple,
  className,
}) => (
  <div className={`metric ${className || ''}`} style={{ '--metric': color } as React.CSSProperties}>
    <span className="label truncate">{label}</span>
    <strong className="truncate">{value}</strong>
    {hint ? <small className="truncate">{hint}</small> : <small />}
  </div>
);

const Meter: React.FC<{ label: string; value: number; color?: string; text?: string }> = ({ label, value, color = COLORS.purple, text }) => (
  <div className="feature">
    <div className="feature-head">
      <span className="truncate">{label}</span>
      <strong>{text ?? pct(value)}</strong>
    </div>
    <div className="meter" style={{ '--meter': color } as React.CSSProperties}>
      <span style={{ width: pct(value) }} />
    </div>
  </div>
);

const TerminalRow: React.FC<{ time?: string | null; title: React.ReactNode; text: React.ReactNode; badge?: React.ReactNode }> = ({
  time,
  title,
  text,
  badge,
}) => (
  <div className="terminal-row">
    <time>{formatShortTime(time)}</time>
    <div className="truncate">
      <strong className="truncate">{title}</strong>
      <p className="truncate">{text}</p>
    </div>
    {badge}
  </div>
);

const Tile: React.FC<{ title: string; text: string; color?: string; badge?: React.ReactNode; active?: boolean }> = ({
  title,
  text,
  color = COLORS.purple,
  badge,
  active,
}) => (
  <div className={`tile ${active ? 'active' : ''}`} style={{ '--tile': color } as React.CSSProperties}>
    <div className="tile-row">
      <strong className="truncate">{title}</strong>
      {badge}
    </div>
    <p className="clamp">{text}</p>
  </div>
);

const FlowBars: React.FC<{ values: Array<{ label: string; value: number; tail?: string; color?: string }> }> = ({ values }) => (
  <div className="flow-bars">
    {values.map((item) => (
      <div className="flow-bar" key={item.label}>
        <span className="truncate">{item.label}</span>
        <div className="bar-track">
          <div className="bar-fill" style={{ width: pct(item.value), background: item.color ? `linear-gradient(90deg, rgba(168,85,255,0.22), ${item.color})` : undefined }} />
        </div>
        <strong className="truncate">{item.tail ?? pct(item.value)}</strong>
      </div>
    ))}
  </div>
);

const PriceCanvas: React.FC<{ state: FilterlessLiveState; position: FilterlessPosition | null }> = ({ state, position }) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const setup = setupCanvas(canvas);
    if (!setup) return;
    const { ctx, width, height, dpr } = setup;
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = '#040107';
    ctx.fillRect(0, 0, width, height);

    const history = state.bot.price_history
      .filter((point) => point.price != null && !Number.isNaN(point.price))
      .slice(-150);
    const current = getPositionCurrentPrice(position, state.bot.price);
    const entry = getPositionEntryPrice(position);
    const stop = positionStop(position);
    const target = positionTarget(position);
    const values = history.map((point) => point.price as number).concat([current, entry, stop, target].filter((value): value is number => value != null && !Number.isNaN(value)));
    if (values.length < 2) {
      ctx.fillStyle = COLORS.muted;
      ctx.font = `${12 * dpr}px "JetBrains Mono", monospace`;
      ctx.fillText('waiting for live price history', 24 * dpr, 42 * dpr);
      return;
    }

    const min = Math.min(...values) - 1;
    const max = Math.max(...values) + 1;
    const left = 34 * dpr;
    const right = width - 86 * dpr;
    const top = 32 * dpr;
    const bottom = height - 30 * dpr;
    const yFor = (price: number) => bottom - ((price - min) / (max - min || 1)) * (bottom - top);

    ctx.strokeStyle = 'rgba(168, 85, 255, 0.11)';
    ctx.lineWidth = 1 * dpr;
    for (let i = 0; i < 5; i += 1) {
      const y = top + ((bottom - top) * i) / 4;
      ctx.beginPath();
      ctx.moveTo(left, y);
      ctx.lineTo(right, y);
      ctx.stroke();
    }

    [
      ['STOP', stop, COLORS.red],
      ['ENTRY', entry, COLORS.cyan],
      ['TP', target, COLORS.green],
    ].forEach(([label, price, color]) => {
      if (typeof price !== 'number' || Number.isNaN(price)) return;
      const y = yFor(price);
      ctx.strokeStyle = color as string;
      ctx.globalAlpha = label === 'ENTRY' ? 0.82 : 0.64;
      ctx.setLineDash([8 * dpr, 7 * dpr]);
      ctx.beginPath();
      ctx.moveTo(left, y);
      ctx.lineTo(right, y);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;
      ctx.fillStyle = color as string;
      ctx.font = `${10 * dpr}px "JetBrains Mono", monospace`;
      ctx.fillText(String(label), right + 10 * dpr, y + 3 * dpr);
    });

    const grad = ctx.createLinearGradient(0, top, 0, bottom);
    grad.addColorStop(0, 'rgba(168,85,255,0.28)');
    grad.addColorStop(1, 'rgba(168,85,255,0)');
    ctx.beginPath();
    history.forEach((point, index) => {
      const x = left + ((right - left) * index) / Math.max(1, history.length - 1);
      const y = yFor(point.price as number);
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.lineTo(right, bottom);
    ctx.lineTo(left, bottom);
    ctx.closePath();
    ctx.fillStyle = grad;
    ctx.fill();

    ctx.beginPath();
    history.forEach((point, index) => {
      const x = left + ((right - left) * index) / Math.max(1, history.length - 1);
      const y = yFor(point.price as number);
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = COLORS.purple;
    ctx.shadowBlur = 18 * dpr;
    ctx.shadowColor = COLORS.purple;
    ctx.lineWidth = 2 * dpr;
    ctx.stroke();
    ctx.shadowBlur = 0;

    if (typeof current === 'number' && !Number.isNaN(current)) {
      ctx.fillStyle = regimeColor('CHOP_SPIRAL');
      ctx.beginPath();
      ctx.arc(right, yFor(current), 4 * dpr, 0, TAU);
      ctx.fill();
    }
  }, [state, position]);

  return <canvas ref={canvasRef} className="chart" />;
};

const SentimentCanvas: React.FC<{ sentiment: FilterlessSentimentMetrics }> = ({ sentiment }) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const setup = setupCanvas(canvas);
    if (!setup) return;
    const { ctx, width, height, dpr } = setup;
    const score = clip(sentiment.sentiment_score ?? 0, -1, 1);
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = '#040107';
    ctx.fillRect(0, 0, width, height);
    const mid = height * 0.52;
    ctx.strokeStyle = 'rgba(168,85,255,0.18)';
    ctx.lineWidth = 1 * dpr;
    ctx.beginPath();
    ctx.moveTo(20 * dpr, mid);
    ctx.lineTo(width - 20 * dpr, mid);
    ctx.stroke();

    ctx.beginPath();
    for (let step = 0; step <= 120; step += 1) {
      const t = step / 120;
      const x = 20 * dpr + (width - 40 * dpr) * t;
      const wave = Math.sin(t * TAU * 2.8) * 0.12 + score * (0.46 + 0.12 * Math.sin(t * TAU));
      const y = mid - wave * height * 0.36;
      if (step === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    const color = score > 0.2 ? COLORS.green : score < -0.2 ? COLORS.red : COLORS.purple;
    ctx.strokeStyle = color;
    ctx.shadowBlur = 16 * dpr;
    ctx.shadowColor = color;
    ctx.lineWidth = 2 * dpr;
    ctx.stroke();
    ctx.shadowBlur = 0;
    ctx.fillStyle = COLORS.muted;
    ctx.font = `${10 * dpr}px "JetBrains Mono", monospace`;
    ctx.fillText(`score ${formatSigned(score)}`, 20 * dpr, 24 * dpr);
    ctx.fillText(`confidence ${pct(sentiment.finbert_confidence)}`, 20 * dpr, 42 * dpr);
  }, [sentiment]);

  return <canvas ref={canvasRef} className="chart" />;
};

const AetherflowCanvas: React.FC<{ features: AetherFeatures }> = ({ features }) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [scene, setScene] = useState<SceneState>({ yaw: 0, pitch: 0, zoom: 1.04 });
  const dragRef = useRef<{ dragging: boolean; x: number; y: number; yaw: number; pitch: number }>({
    dragging: false,
    x: 0,
    y: 0,
    yaw: 0,
    pitch: 0,
  });

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const setup = setupCanvas(canvas);
    if (!setup) return;
    const { ctx, width, height, dpr } = setup;
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = '#030006';
    ctx.fillRect(0, 0, width, height);

    const drawLabel = (text: string, point: { x: number; y: number }, color: string, align: CanvasTextAlign = 'center') => {
      ctx.save();
      ctx.font = `${10 * dpr}px "JetBrains Mono", monospace`;
      ctx.textAlign = align;
      ctx.textBaseline = 'middle';
      ctx.shadowBlur = 13 * dpr;
      ctx.shadowColor = color;
      ctx.fillStyle = color;
      ctx.fillText(text, point.x, point.y);
      ctx.restore();
    };

    const strokePath = (points: Array<{ x: number; y: number }>, color: string, lineWidth: number) => {
      ctx.save();
      ctx.beginPath();
      points.forEach((point, index) => {
        if (index === 0) ctx.moveTo(point.x, point.y);
        else ctx.lineTo(point.x, point.y);
      });
      ctx.strokeStyle = color;
      ctx.lineWidth = lineWidth * dpr;
      ctx.stroke();
      ctx.restore();
    };

    const cols = 48;
    const rows = 28;
    const points: Array<Array<{ x: number; y: number; z: number; p: ReturnType<typeof projectSurfacePoint>; dominant: string }>> = [];
    for (let row = 0; row <= rows; row += 1) {
      const line = [];
      const y = row / rows;
      for (let col = 0; col <= cols; col += 1) {
        const x = -1 + (2 * col) / cols;
        const z = pressurePlaneHeight(x, y, features);
        line.push({
          x,
          y,
          z,
          p: projectSurfacePoint(x, y, z, width, height, scene),
          dominant: pressurePlaneDominant(x, y, features),
        });
      }
      points.push(line);
    }

    const faces = [];
    for (let row = 0; row < rows; row += 1) {
      for (let col = 0; col < cols; col += 1) {
        const a = points[row][col];
        const b = points[row][col + 1];
        const c = points[row + 1][col + 1];
        const d = points[row + 1][col];
        const heightAvg = (a.z + b.z + c.z + d.z) / 4;
        faces.push({
          pts: [a, b, c, d],
          depth: (a.p.depth + b.p.depth + c.p.depth + d.p.depth) / 4,
          height: heightAvg,
          dominant: [a, b, c, d].sort((p1, p2) => p2.z - p1.z)[0].dominant,
        });
      }
    }

    faces.sort((a, b) => a.depth - b.depth);
    faces.forEach((face) => {
      const v10Height = face.height / 0.54;
      const shade = clip(0.42 + ((v10Height + 0.28) * 0.54) + (((face.depth + 2) / 8) * 0.22), 0.22, 1.22);
      ctx.beginPath();
      face.pts.forEach((point, index) => {
        if (index === 0) ctx.moveTo(point.p.x, point.p.y);
        else ctx.lineTo(point.p.x, point.p.y);
      });
      ctx.closePath();
      ctx.fillStyle = shadedColor(dominantColor(face.dominant), shade, 0.72);
      ctx.fill();
      ctx.strokeStyle = v10Height > 0.78 ? 'rgba(255,255,255,0.16)' : 'rgba(168,85,255,0.13)';
      ctx.lineWidth = 0.68 * dpr;
      ctx.stroke();
    });

    ctx.save();
    ctx.lineWidth = 1.1 * dpr;
    [0.16, 0.28, 0.4, 0.52, 0.64, 0.76, 0.88].forEach((y, index) => {
      ctx.beginPath();
      for (let step = 0; step <= 144; step += 1) {
        const x = -0.98 + (1.96 * step) / 144;
        const z = pressurePlaneHeight(x, y, features) + 0.014;
        const p = projectSurfacePoint(x, y, z, width, height, scene);
        if (step === 0) ctx.moveTo(p.x, p.y);
        else ctx.lineTo(p.x, p.y);
      }
      ctx.strokeStyle = index % 2 ? 'rgba(168,85,255,0.30)' : 'rgba(53,245,255,0.25)';
      ctx.stroke();
    });
    ctx.restore();

    const origin = projectSurfacePoint(-1.08, 0, 0, width, height, scene);
    const farRight = projectSurfacePoint(1.1, 0, 0, width, height, scene);
    const nearRight = projectSurfacePoint(1.1, 1.08, 0, width, height, scene);
    const nearLeft = projectSurfacePoint(-1.08, 1.08, 0, width, height, scene);
    const axisZ = { x: origin.x, y: origin.y - height * 0.43 };
    strokePath([origin, farRight, nearRight, nearLeft, origin], 'rgba(168,85,255,0.42)', 1.5);
    strokePath([origin, axisZ], 'rgba(168,85,255,0.72)', 1.5);
    strokePath([origin, nearLeft], 'rgba(168,85,255,0.42)', 1.2);

    ctx.save();
    ctx.font = `${10 * dpr}px "JetBrains Mono", monospace`;
    ctx.fillStyle = alphaColor(COLORS.muted, 0.92);
    const pressureLabel = projectSurfacePoint(1.14, 0.66, 0.1, width, height, scene);
    const transitionLabelX = clip(nearLeft.x + 8 * dpr, 8 * dpr, width - 96 * dpr);
    ctx.fillText('INTENSITY', axisZ.x + 10 * dpr, axisZ.y + 8 * dpr);
    ctx.fillText('PRESSURE -1', pressureLabel.x + 8 * dpr, pressureLabel.y - 4 * dpr);
    ctx.fillText('TRANSITION', transitionLabelX, nearLeft.y + 18 * dpr);
    ctx.restore();

    const markerTarget = markerTargetForRegime(features);
    const markerZ = pressurePlaneHeight(markerTarget.x, markerTarget.y, features);
    const marker = projectSurfacePoint(markerTarget.x, markerTarget.y, markerZ + 0.05, width, height, scene);
    const base = projectSurfacePoint(markerTarget.x, markerTarget.y, 0, width, height, scene);
    const activeColor = regimeColor(features.regime);
    ctx.save();
    ctx.strokeStyle = activeColor;
    ctx.lineWidth = 2 * dpr;
    ctx.shadowBlur = 14 * dpr;
    ctx.shadowColor = activeColor;
    ctx.beginPath();
    ctx.moveTo(base.x, base.y);
    ctx.lineTo(marker.x, marker.y);
    ctx.stroke();
    ctx.fillStyle = activeColor;
    ctx.beginPath();
    ctx.arc(marker.x, marker.y, 6 * dpr, 0, TAU);
    ctx.fill();
    ctx.shadowBlur = 0;
    ctx.font = `${11 * dpr}px "JetBrains Mono", monospace`;
    ctx.textBaseline = 'middle';
    ctx.fillStyle = COLORS.text;
    ctx.fillText(features.regime, marker.x + 13 * dpr, marker.y - 8 * dpr);
    ctx.fillStyle = COLORS.muted;
    ctx.fillText(`pressure ${fmt(features.pressure30)} / burst ${fmt(features.burstPressure)}`, marker.x + 13 * dpr, marker.y + 9 * dpr);
    ctx.restore();

    drawLabel('trend ridge', projectSurfacePoint(features.pressure30, 0.78, 0.4, width, height, scene), COLORS.green);
    drawLabel('burst peak', projectSurfacePoint(Math.sign(features.pressure30 || 0.001) * 0.55, features.transitionEnergy, 0.58, width, height, scene), COLORS.amber);
    drawLabel('chop shelf', projectSurfacePoint(0, 0.38, 0.44, width, height, scene), COLORS.cyan);
    drawLabel('rot wall', projectSurfacePoint(-0.72, 0.58, 0.5, width, height, scene), COLORS.red);

    const cx = width - 58 * dpr;
    const cy = 48 * dpr;
    const rx = 28 * dpr;
    const ry = 15 * dpr;
    const dotX = cx + Math.sin(scene.yaw) * rx;
    const dotY = cy - scene.pitch * 38 * dpr;
    ctx.save();
    ctx.strokeStyle = 'rgba(168,85,255,0.36)';
    ctx.lineWidth = 1 * dpr;
    ctx.beginPath();
    ctx.ellipse(cx, cy, rx, ry, 0, 0, TAU);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(cx - rx, cy);
    ctx.lineTo(cx + rx, cy);
    ctx.moveTo(cx, cy - 24 * dpr);
    ctx.lineTo(cx, cy + 24 * dpr);
    ctx.stroke();
    ctx.fillStyle = COLORS.cyan;
    ctx.shadowBlur = 12 * dpr;
    ctx.shadowColor = COLORS.cyan;
    ctx.beginPath();
    ctx.arc(dotX, dotY, 4 * dpr, 0, TAU);
    ctx.fill();
    ctx.restore();
  }, [features, scene]);

  const onPointerDown = (event: React.PointerEvent<HTMLCanvasElement>) => {
    dragRef.current = {
      dragging: true,
      x: event.clientX,
      y: event.clientY,
      yaw: scene.yaw,
      pitch: scene.pitch,
    };
    event.currentTarget.setPointerCapture(event.pointerId);
  };

  const onPointerMove = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (!dragRef.current.dragging) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const dx = event.clientX - dragRef.current.x;
    const dy = event.clientY - dragRef.current.y;
    setScene({
      yaw: clip(dragRef.current.yaw - (dx / Math.max(1, rect.width)) * 1.45, -0.62, 0.62),
      pitch: clip(dragRef.current.pitch + (dy / Math.max(1, rect.height)) * 0.84, -0.28, 0.42),
      zoom: scene.zoom,
    });
  };

  const onPointerUp = (event: React.PointerEvent<HTMLCanvasElement>) => {
    dragRef.current.dragging = false;
    try {
      event.currentTarget.releasePointerCapture(event.pointerId);
    } catch {
      // Pointer capture may already have been released by the browser.
    }
  };

  const onWheel = (event: React.WheelEvent<HTMLCanvasElement>) => {
    event.preventDefault();
    setScene((current) => ({ ...current, zoom: clip(current.zoom * (event.deltaY > 0 ? 0.93 : 1.07), 0.72, 1.42) }));
  };

  return (
    <div className="scene-wrap">
      <canvas
        ref={canvasRef}
        className="aetherflow-scene"
        aria-label="AetherFlow manifold view"
        title="Drag to rotate. Wheel to zoom. Double-click to reset."
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
        onWheel={onWheel}
        onDoubleClick={() => setScene({ yaw: 0, pitch: 0, zoom: 1.04 })}
      />
      <div className="scene-hud">
        <div className="hud-cell"><span className="label">pressure x</span><strong className="truncate">{fmt(features.pressure30)}</strong></div>
        <div className="hud-cell"><span className="label">transition y</span><strong className="truncate">{pct(features.transitionEnergy)}</strong></div>
        <div className="hud-cell"><span className="label">height z</span><strong className="truncate">{fmt(pressurePlaneHeight(markerTargetForRegime(features).x, markerTargetForRegime(features).y, features))}</strong></div>
        <div className="hud-cell"><span className="label">fold shade</span><strong className="truncate">{fmt(features.foldDepth)}</strong></div>
      </div>
    </div>
  );
};

function FilterlessLiveCockpit() {
  const [state, setState] = useState<FilterlessLiveState>(EMPTY_STATE);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeScreen, setActiveScreen] = useState<ScreenId>('overview');
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
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
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
        if (!cancelled && !isAbortError) {
          startTransition(() => {
            setError(err instanceof Error ? err.message : String(err));
            setLoading(false);
          });
        }
      } finally {
        inFlightRef.current = false;
      }
    };

    void loadState();
    const timer = window.setInterval(loadState, REFRESH_MS);
    const handleVisibilityChange = () => {
      if (document.visibilityState === 'visible') void loadState();
    };
    document.addEventListener('visibilitychange', handleVisibilityChange);
    return () => {
      cancelled = true;
      abortRef.current?.abort();
      window.clearInterval(timer);
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, []);

  const effectiveStatus = computeFeedStatus(state.bot.status, state.generated_at);
  const openPositions = useMemo(() => {
    if (effectiveStatus !== 'online') return [];
    const positions = Array.isArray(state.bot.current_positions)
      ? state.bot.current_positions.filter((position): position is FilterlessPosition => position != null)
      : [];
    return positions.length > 0 ? positions : state.bot.current_position ? [state.bot.current_position] : [];
  }, [effectiveStatus, state.bot.current_position, state.bot.current_positions]);
  const primaryPosition = openPositions[0] ?? null;
  const sentiment = useMemo(() => {
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
  const features = useMemo(() => deriveFeatures(state, effectiveStatus, openPositions, sentiment), [state, effectiveStatus, openPositions, sentiment]);
  const nav = NAV.find((item) => item.id === activeScreen) ?? NAV[0];
  const price = state.bot.price ?? features.price;
  const dailyPnl = state.bot.risk.daily_pnl ?? null;
  const totalOpenPnl = openPositions.reduce((sum, position) => sum + (position.open_pnl_dollars ?? 0), 0);
  const totalLots = openPositions.reduce((sum, position) => sum + (position.size ?? 0), 0);
  const entry = getPositionEntryPrice(primaryPosition);
  const stop = positionStop(primaryPosition);
  const target = positionTarget(primaryPosition);
  const targetRoom = price != null && target != null ? Math.abs(price - target) : null;
  const kalshi = state.kalshi_metrics ?? null;
  const feedWarnings = useMemo(() => {
    const warnings = [...state.bot.warnings];
    if (effectiveStatus !== state.bot.status) {
      warnings.unshift(`Dashboard feed stopped refreshing. Last update was ${formatRelativeTime(state.generated_at)}.`);
    }
    return warnings;
  }, [effectiveStatus, state.bot.status, state.bot.warnings, state.generated_at]);

  const logicRows = [
    ['FEED', `status ${effectiveStatus}, heartbeat ${formatRelativeTime(state.bot.last_heartbeat_time)}`, effectiveStatus === 'online' ? 'pass' : 'block'],
    ['ENTRY', primaryPosition ? `${primaryPosition.side} ${primaryPosition.size ?? '--'} @ ${formatPrice(entry)}` : 'flat, no active broker position', primaryPosition ? 'live' : 'idle'],
    ['GUARD', features.noTrade ? `${features.regime} lockout / stress ${pct(features.stress)}` : `risk mult ${fmt(features.riskMult)}x inside bounds`, features.noTrade ? 'block' : 'pass'],
    ['TRUTH', features.truthRiskWatch ? `advisory pressure ${pct(features.advisoryPressure)}` : `score ${formatSigned(features.truthScore)} / confidence ${pct(features.truthConfidence)}`, features.truthRiskWatch ? 'watch' : 'info'],
  ];

  const renderOverview = () => (
    <section className="screen">
      <div className="grid overview-layout">
        <div className="stack">
          <Panel title="Execution Logic" subtitle="Execution score, guard rails, and live context." badge={<Badge tone={features.noTrade ? 'block' : 'live'}>{features.noTrade ? 'guard' : 'armed'}</Badge>}>
            <div className="terminal">
              {logicRows.map(([title, text, tone]) => (
                <TerminalRow key={title} time={state.generated_at} title={title} text={text} badge={<Badge tone={tone === 'block' ? 'block' : tone === 'watch' ? 'watch' : 'info'}>{tone}</Badge>} />
              ))}
            </div>
          </Panel>
          <Panel title="Risk Monitor" subtitle="Kelly, drawdown, and correlation pressure." badge={<Badge tone={state.bot.risk.circuit_tripped ? 'block' : 'watch'}>{state.bot.risk.circuit_tripped ? 'tripped' : 'watch'}</Badge>}>
            <div className="panel-body">
              <Meter label="risk mult" value={features.riskMult / 1.5} text={`${fmt(features.riskMult)}x`} color={COLORS.purple} />
              <Meter label="drawdown" value={clip01(Math.abs(dailyPnl ?? 0) / 2000)} text={formatMoney(dailyPnl)} color={COLORS.red} />
              <Meter label="correlation" value={features.R} text={fmt(features.R)} color={COLORS.cyan} />
            </div>
          </Panel>
        </div>

        <div className="stack">
          <Panel title="Execution Chart" subtitle="Price path with entry, stop, and take-profit rails." badge={<Badge tone={primaryPosition ? 'info' : 'watch'}>{primaryPosition ? 'live trade' : 'flat'}</Badge>}>
            <PriceCanvas state={state} position={primaryPosition} />
            <div className="mini-grid">
              <div className="mini"><span className="label">position</span><strong className={`truncate ${sideTone(primaryPosition?.side)}`}>{primaryPosition ? `${primaryPosition.side} ${primaryPosition.size ?? '--'} MES` : 'FLAT'}</strong></div>
              <div className="mini"><span className="label">entry</span><strong className="truncate info">{formatPrice(entry)}</strong></div>
              <div className="mini"><span className="label">stop</span><strong className="truncate down">{formatPrice(stop)}</strong></div>
              <div className="mini"><span className="label">tp route</span><strong className="truncate up">{formatPrice(target)}</strong></div>
            </div>
          </Panel>
          <div className="grid cols-3">
            <Metric label="last price" value={formatPrice(price)} hint="streaming tape" color={COLORS.cyan} />
            <Metric label="open pnl" value={primaryPosition ? formatMoney(totalOpenPnl) : '--'} hint="broker shadow" color={totalOpenPnl >= 0 ? COLORS.lime : COLORS.amber} />
            <Metric label="target room" value={targetRoom == null ? '--' : `${fmt(targetRoom, 1)}pt`} hint="to active TP" color={COLORS.lime} />
          </div>
        </div>

        <div className="stack">
          <Panel title="Active Positions" subtitle="Live P/L and route notes." badge={<Badge tone={openPositions.length ? 'live' : 'watch'}>{openPositions.length ? `${openPositions.length} live` : 'flat'}</Badge>}>
            <div className="terminal">
              {openPositions.length ? openPositions.map((position, index) => (
                <div className="position" key={`${position.strategy_id}-${position.order_id || position.opened_at || index}`}>
                  <div>
                    <strong className="truncate">{position.strategy_label || position.strategy_id}</strong>
                    <p className="truncate">entry {formatPrice(getPositionEntryPrice(position))} / {formatToken(position.entry_mode || position.rule_id)}</p>
                  </div>
                  <span className={(position.open_pnl_dollars ?? 0) >= 0 ? 'up' : 'warn'}>{formatMoney(position.open_pnl_dollars)}</span>
                </div>
              )) : <TerminalRow title="FLAT" text="No filterless positions are currently open." badge={<Badge tone="watch">idle</Badge>} />}
            </div>
          </Panel>
          <Panel title="Order Flow" subtitle="Live pressure derived from tape and route state." badge={<Badge tone="info">depth</Badge>}>
            <FlowBars values={[
              { label: 'align', value: features.alignment, color: COLORS.green },
              { label: 'smooth', value: features.smoothness, color: COLORS.cyan },
              { label: 'stress', value: features.stress, color: COLORS.amber },
              { label: 'burst', value: features.burstPressure, color: COLORS.pink },
              { label: 'risk', value: features.riskMult / 1.5, color: COLORS.purple },
            ]} />
          </Panel>
        </div>
      </div>
    </section>
  );

  const renderAetherflow = () => (
    <section className="screen">
      <div className="grid cols-4">
        <Metric label="manifold regime" value={features.regime} hint="threshold order mirror" color={COLORS.purple} />
        <Metric label="pressure imb 30" value={fmt(features.pressure30)} hint="up minus down flow" color={COLORS.cyan} />
        <Metric label="fold depth" value={fmt(features.foldDepth)} hint="surface curvature" color={COLORS.amber} />
        <Metric label="lockout" value={String(features.noTrade).toUpperCase()} hint="state guard" color={features.noTrade ? COLORS.red : COLORS.green} />
      </div>
      <div className="grid aether-layout">
        <Panel title="AetherFlow Manifold" subtitle="Folded 3D state surface from pressure, transition, and manifold feature fields." badge={<Badge tone={features.noTrade ? 'block' : 'info'}>{features.noTrade ? 'lockout' : 'orbit live'}</Badge>}>
          <AetherflowCanvas features={features} />
          <div className="mini-grid">
            <div className="mini"><span className="label">trend ridge</span><strong className="truncate up">alignment + smoothness</strong></div>
            <div className="mini"><span className="label">burst peak</span><strong className="truncate warn">transition pressure</strong></div>
            <div className="mini"><span className="label">chop shelf</span><strong className="truncate info">stress roughness</strong></div>
            <div className="mini"><span className="label">rot wall</span><strong className="truncate down">dPhi lockout</strong></div>
          </div>
        </Panel>
        <div className="stack">
          <Panel title="Manifold Features" subtitle="Fields exported into AetherFlow feature space.">
            <div className="panel-body">
              <Meter label="alignment pct" value={features.alignment} color={COLORS.green} />
              <Meter label="smoothness pct" value={features.smoothness} color={COLORS.cyan} />
              <Meter label="stress pct" value={features.stress} color={COLORS.amber} />
              <Meter label="dispersion pct" value={features.dispersion} color={COLORS.violet} />
              <Meter label="novelty score" value={features.novelty} color={COLORS.pink} />
            </div>
          </Panel>
          <Panel title="State Regions" subtitle="Active state controls marker and routing gates.">
            <div className="panel-body">
              <div className="state-matrix">
                <Tile title="TREND_GEODESIC" text="High alignment and smoothness, low dispersion." color={COLORS.green} active={features.regime === 'TREND_GEODESIC'} badge={<Badge tone="live">ridge</Badge>} />
                <Tile title="CHOP_SPIRAL" text="Stress or roughness dominates mean-reversion routes." color={COLORS.cyan} active={features.regime === 'CHOP_SPIRAL'} badge={<Badge tone="info">shelf</Badge>} />
                <Tile title="DISPERSED" text="Dispersion is high while alignment is weak." color={COLORS.violet} active={features.regime === 'DISPERSED'} badge={<Badge tone="watch">basin</Badge>} />
                <Tile title="ROTATIONAL" text="Stress and dPhi create no-trade turbulence." color={COLORS.red} active={features.regime === 'ROTATIONAL_TURBULENCE'} badge={<Badge tone="block">wall</Badge>} />
              </div>
            </div>
          </Panel>
          <Panel title="AetherFlow Setups" subtitle="Feature interactions projected onto the manifold.">
            <div className="panel-body">
              <Meter label="aligned_flow" value={features.alignedFlow} color={COLORS.green} />
              <Meter label="transition_burst" value={features.transitionBurst} color={COLORS.amber} />
              <Meter label="nypm_trend_edge" value={features.nypmTrendEdge} color={COLORS.cyan} />
              <Meter label="chop_fade_edge" value={features.chopFadeEdge} color={COLORS.pink} />
            </div>
          </Panel>
        </div>
      </div>
    </section>
  );

  const renderKalshi = () => {
    const strikes = (kalshi?.strikes || []).slice().sort((a, b) => a.strike - b.strike);
    const rows = strikes.slice(Math.max(0, Math.floor(strikes.length / 2) - 8), Math.max(0, Math.floor(strikes.length / 2) + 9));
    const route = kalshiRouteState(kalshi, state.generated_at);
    const primaryKalshiPosition = openPositions[0] ?? null;
    return (
      <section className="screen">
        <div className="grid cols-4">
          <Metric label="Kalshi route" value={route.value} hint={route.hint} color={route.color} />
          <Metric label="book edge" value={fmt(features.edge * 10, 2)} hint={kalshi?.event_ticker || 'best contract'} color={COLORS.cyan} />
          <Metric label="strategy impact" value="DE3 / RA / AF" hint="entry, size, TP, exits" color={COLORS.amber} />
          <Metric label="probability" value={pct(kalshi?.probability_60m, 1)} hint="60m contract" color={COLORS.lime} />
        </div>
        <div className="grid kalshi-layout">
          <Panel title="Market Scanner" subtitle="Hourly ES contracts ranked by edge, pressure, and liquidity." badge={<Badge tone={route.tone}>{route.badge}</Badge>}>
            <table className="table">
              <thead><tr><th>contract</th><th>probability</th><th>volume</th><th>status</th><th>route</th></tr></thead>
              <tbody>
                {rows.length ? rows.map((strike, index) => (
                  <tr key={`${strike.strike}-${index}`}>
                    <td><strong>{fmt(strike.strike, 0)}</strong></td>
                    <td>{pct(strike.probability, 2)}</td>
                    <td>{strike.volume ?? '--'}</td>
                    <td>{strike.status || strike.result || '--'}</td>
                    <td>{route.row}</td>
                  </tr>
                )) : (
                  <tr><td colSpan={5}>No Kalshi ladder is available in the current snapshot.</td></tr>
                )}
              </tbody>
            </table>
          </Panel>
          <div className="stack">
            <Panel title="Consensus Feed" subtitle="Macro consensus and headline risk.">
              <div className="terminal">
                <TerminalRow title="EVENT" text={kalshi?.event_ticker || 'waiting for active contract'} badge={<Badge tone="info">ticker</Badge>} />
                <TerminalRow title="REFERENCE" text={`ES ${formatPrice(kalshi?.es_reference_price ?? price)} / SPX ${formatPrice(kalshi?.spx_reference_price)}`} badge={<Badge tone="info">basis</Badge>} />
                <TerminalRow title="ROUTE STATE" text={route.detail} badge={<Badge tone={route.tone}>{route.badge}</Badge>} />
                <TerminalRow title="STRATEGY IMPACT" text={route.impact} badge={<Badge tone={route.tone}>{route.value.toLowerCase()}</Badge>} />
                <TerminalRow title="LIVE POSITION" text={describeKalshiPositionImpact(primaryKalshiPosition)} badge={<Badge tone={primaryKalshiPosition?.kalshi_trade_overlay_applied || primaryKalshiPosition?.kalshi_gate_applied ? 'live' : 'info'}>{primaryKalshiPosition ? 'position' : 'flat'}</Badge>} />
              </div>
            </Panel>
            <Panel title="Spread Ladder" subtitle="Execution spread by bucket.">
              <FlowBars values={[
                { label: 'book', value: features.edge, color: COLORS.cyan },
                { label: 'spread', value: features.spread * 8, color: COLORS.amber },
                { label: 'prob', value: kalshi?.probability_60m ?? 0, color: COLORS.lime },
                { label: 'stress', value: features.stress, color: COLORS.red },
              ]} />
            </Panel>
          </div>
        </div>
      </section>
    );
  };

  const renderNews = () => {
    const excerpt = String(sentiment.latest_post_text || '').trim();
    return (
      <section className="screen">
        <div className="grid cols-4">
          <Metric label="truth monitor" value={sentiment.last_error ? 'ISSUE' : sentiment.healthy ? 'HEALTHY' : 'WATCH'} hint="rss + finbert state" color={sentiment.last_error ? COLORS.red : COLORS.green} />
          <Metric label="sentiment" value={formatSigned(features.truthScore)} hint={sentiment.sentiment_label || 'neutral'} color={COLORS.purple} />
          <Metric label="last poll" value={formatRelativeTime(sentiment.last_poll_at)} hint={`@${sentiment.target_handle || 'realDonaldTrump'}`} color={COLORS.cyan} />
          <Metric label="truth advisory" value={features.truthRiskWatch ? 'WATCH' : 'OBSERVE'} hint="observe-only" color={features.truthRiskWatch ? COLORS.red : COLORS.amber} />
        </div>
        <div className="grid news-layout">
          <div className="stack">
            <Panel title="Truth Social Monitor" subtitle="Persisted sentiment snapshot for operator context." badge={<Badge tone={sentiment.last_error ? 'block' : 'live'}>{sentiment.last_error ? 'issue' : 'healthy'}</Badge>}>
              <div className="panel-body">
                <div className="truth-grid">
                  <Tile title="latest post" text={excerpt || 'No post has been analyzed yet. Waiting for new Truth Social activity.'} color={COLORS.cyan} badge={<Badge tone="info">rss</Badge>} />
                  <Tile title="model mode" text={sentiment.quantized_8bit ? 'Quantized FinBERT path is active.' : 'FinBERT runtime path is active.'} color={COLORS.purple} badge={<Badge tone="live">finbert</Badge>} />
                  <Tile title="long-side watch" text="Negative sentiment is displayed only; position management remains unchanged." color={COLORS.amber} badge={<Badge tone="info">observe</Badge>} />
                  <Tile title="short-side watch" text="Positive sentiment is displayed only; position management remains unchanged." color={COLORS.red} badge={<Badge tone="info">observe</Badge>} />
                </div>
              </div>
            </Panel>
            <Panel title="News Tape" subtitle="Calendar, macro, and truth-feed context." badge={<Badge tone="info">live</Badge>}>
              <div className="terminal">
                <TerminalRow time={sentiment.latest_post_created_at} title="TRUTH" text={sentiment.trigger_reason || sentiment.sentiment_label || 'neutral watch'} badge={<Badge tone="info">feed</Badge>} />
                <TerminalRow time={sentiment.last_analysis_at} title="FINBERT" text={`score ${formatSigned(features.truthScore)} / confidence ${pct(sentiment.finbert_confidence)}`} badge={<Badge tone="live">model</Badge>} />
                <TerminalRow time={state.generated_at} title="ADVISORY" text={`pressure ${pct(features.advisoryPressure)} / headline risk ${pct(features.headlineRisk)}`} badge={<Badge tone={features.truthRiskWatch ? 'watch' : 'info'}>{features.truthRiskWatch ? 'watch' : 'clear'}</Badge>} />
              </div>
            </Panel>
          </div>
          <div className="stack">
            <Panel title="Sentiment Pulse" subtitle="Truth Social score and confidence.">
              <SentimentCanvas sentiment={sentiment} />
            </Panel>
            <Panel title="Monitor Status" subtitle="Observe-only sentiment health and headline-risk context.">
              <div className="panel-body">
                <Meter label="finbert confidence" value={sentiment.finbert_confidence ?? 0} color={COLORS.purple} />
                <Meter label="signal freshness" value={sentiment.last_analysis_at ? clip01(1 - ((Date.now() - new Date(sentiment.last_analysis_at).getTime()) / 3_600_000)) : 0} color={COLORS.cyan} />
                <Meter label="headline risk" value={features.headlineRisk} color={COLORS.amber} />
                <Meter label="advisory pressure" value={features.advisoryPressure} color={COLORS.red} />
              </div>
            </Panel>
          </div>
        </div>
      </section>
    );
  };

  const renderStrategies = () => (
    <section className="screen">
      <div className="grid cols-3">
        <Metric label="AetherFlow" value={(state.strategies.find((strategy) => strategy.id === 'aetherflow')?.status || 'READY').toUpperCase()} hint="routed ensemble" color={COLORS.green} />
        <Metric label="DE3" value={(state.strategies.find((strategy) => strategy.id === 'dynamic_engine3')?.status || 'WATCH').toUpperCase()} hint="ML LFO scoped" color={COLORS.amber} />
        <Metric label="truth overlay" value={sentiment.last_error ? 'ISSUE' : 'OBSERVE'} hint="sentiment context" color={COLORS.cyan} />
      </div>
      <Panel title="Strategy Stack" subtitle="Modules mapped to manifold, news, and execution state." badge={<Badge tone="live">loaded</Badge>} className="mt-panel">
        <table className="table">
          <thead><tr><th>module</th><th>state</th><th>trigger</th><th>manifold</th><th>truth</th><th>action</th></tr></thead>
          <tbody>
            {state.strategies.length ? state.strategies.map((strategy: FilterlessStrategyState) => (
              <tr key={strategy.id}>
                <td><strong>{strategy.label}</strong></td>
                <td>{strategy.status}</td>
                <td>{formatToken(strategy.latest_activity_type || strategy.priority || strategy.entry_mode)}</td>
                <td>{features.regime}</td>
                <td>{formatSigned(features.truthScore)}</td>
                <td>{strategy.last_block_reason ? 'gate' : strategy.status === 'in_trade' ? 'manage' : 'observe'}</td>
              </tr>
            )) : (
              <tr><td colSpan={6}>No strategies are present in the live snapshot.</td></tr>
            )}
          </tbody>
        </table>
      </Panel>
    </section>
  );

  const renderJournal = () => (
    <section className="screen">
      <div className="grid journal-layout">
        <Panel title="Trace Log" subtitle="Decision journal with trade levels, news, and manifold snapshots." badge={<Badge tone="info">live</Badge>}>
          <div className="terminal">
            {state.events.length ? state.events.slice(0, 28).map((event: FilterlessEvent, index) => (
              <TerminalRow key={`${event.event_type}-${event.time}-${index}`} time={event.time} title={event.event_type} text={event.message} badge={<Badge tone={event.severity === 'error' || event.severity === 'danger' ? 'block' : event.severity === 'warning' ? 'watch' : 'info'}>{event.severity}</Badge>} />
            )) : <TerminalRow title="WAIT" text="No filterless events have been bridged yet." badge={<Badge tone="watch">idle</Badge>} />}
          </div>
        </Panel>
        <div className="stack">
          <Panel title="State Summary" subtitle="Latest engine snapshot.">
            <div className="panel-body">
              <Meter label="regime" value={1} text={features.regime} color={COLORS.purple} />
              <Meter label="risk mult" value={features.riskMult / 1.5} text={`${fmt(features.riskMult)}x`} color={COLORS.lime} />
              <Meter label="truth score" value={(features.truthScore + 1) / 2} text={formatSigned(features.truthScore)} color={COLORS.pink} />
            </div>
          </Panel>
          <Panel title="Trade Blotter" subtitle="Recent closed filterless trades.">
            <div className="terminal">
              {state.trades.length ? state.trades.slice(0, 14).map((trade: FilterlessTrade, index) => (
                <TerminalRow
                  key={`${trade.time}-${index}`}
                  time={trade.time}
                  title={`${trade.strategy_label} ${trade.side}`}
                  text={`${formatPrice(trade.entry_price)} -> ${formatPrice(trade.exit_price)} / ${formatMoney(trade.pnl_dollars_net ?? trade.pnl_dollars)}`}
                  badge={<Badge tone={(trade.pnl_dollars_net ?? trade.pnl_dollars ?? 0) >= 0 ? 'live' : 'block'}>{trade.result || 'done'}</Badge>}
                />
              )) : <TerminalRow title="EMPTY" text="No recent closed filterless trades have been detected." badge={<Badge tone="watch">flat</Badge>} />}
            </div>
          </Panel>
        </div>
      </div>
    </section>
  );

  const renderCommand = () => (
    <section className="screen">
      <Panel title="Command Matrix" subtitle="Runtime controls, guard rails, and operator actions." badge={<Badge tone={effectiveStatus === 'online' ? 'live' : 'block'}>{effectiveStatus}</Badge>}>
        <div className="panel-body">
          <div className="command-grid">
            <div className="command-tile"><strong className="truncate">Feed Bridge</strong><small className="truncate">{formatRelativeTime(state.generated_at)}</small><Badge tone={statusBadge(effectiveStatus)}>{effectiveStatus}</Badge></div>
            <div className="command-tile"><strong className="truncate">Circuit</strong><small className="truncate">loss breaker</small><Badge tone={state.bot.risk.circuit_tripped ? 'block' : 'live'}>{state.bot.risk.circuit_tripped ? 'tripped' : 'clear'}</Badge></div>
            <div className="command-tile"><strong className="truncate">Position Sync</strong><small className="truncate">{state.bot.position_sync_status || 'none'}</small><Badge tone={openPositions.length ? 'watch' : 'info'}>{openPositions.length ? 'active' : 'flat'}</Badge></div>
            <div className="command-tile"><strong className="truncate">Kalshi</strong><small className="truncate">{kalshi?.status_reason || kalshi?.event_ticker || 'no ladder'}</small><Badge tone={kalshi?.healthy ? 'live' : 'watch'}>{kalshi?.healthy ? 'online' : 'watch'}</Badge></div>
            <div className="command-tile"><strong className="truncate">Truth</strong><small className="truncate">{sentiment.trigger_reason || sentiment.sentiment_label || 'neutral'}</small><Badge tone={sentiment.last_error ? 'block' : 'info'}>{sentiment.last_error ? 'issue' : 'watch'}</Badge></div>
          </div>
        </div>
      </Panel>
      {feedWarnings.length > 0 ? (
        <Panel title="Feed Notes" subtitle="Observability gaps and runtime warnings." badge={<Badge tone="watch">{feedWarnings.length} notes</Badge>} className="mt-panel">
          <div className="terminal">
            {feedWarnings.map((warning, index) => (
              <TerminalRow key={`${warning}-${index}`} title="WARNING" text={warning} badge={<Badge tone="watch">note</Badge>} />
            ))}
          </div>
        </Panel>
      ) : null}
    </section>
  );

  const renderScreen = () => {
    if (activeScreen === 'aetherflow') return renderAetherflow();
    if (activeScreen === 'kalshi') return renderKalshi();
    if (activeScreen === 'news') return renderNews();
    if (activeScreen === 'strategies') return renderStrategies();
    if (activeScreen === 'journal') return renderJournal();
    if (activeScreen === 'command') return renderCommand();
    return renderOverview();
  };

  return (
    <div className="fl-cockpit">
      <style>{`${COCKPIT_CSS}.mt-panel{margin-top:10px;}`}</style>
      <div className="app">
        <aside className="rail">
          <div className="brand">
            <h1 className="truncate">JULIE LIVE</h1>
            <p className="truncate">projectx / kalshi / truth monitor</p>
            <div className="wire" aria-hidden="true" />
          </div>
          <nav className="nav" aria-label="Live screens">
            {NAV.map((item) => (
              <button key={item.id} className={activeScreen === item.id ? 'active' : ''} onClick={() => setActiveScreen(item.id)}>
                <span className="truncate">{item.label}</span>
                <small>{item.code}</small>
              </button>
            ))}
          </nav>
          <div className="rail-bottom">
            <div className="status-line"><span className="truncate">bridge</span><strong><span className={`dot ${effectiveStatus === 'online' ? '' : effectiveStatus === 'stale' ? 'warn-dot' : 'down-dot'}`} />{effectiveStatus}</strong></div>
            <div className="status-line"><span className="truncate">regime</span><strong className="truncate">{features.regime.replace('_', ' ')}</strong></div>
            <div className="status-line"><span className="truncate">truth</span><strong className="truncate">{sentiment.sentiment_label || 'neutral'}</strong></div>
            <div className="status-line"><span className="truncate">risk</span><strong className="truncate">{fmt(features.riskMult)}x</strong></div>
          </div>
        </aside>

        <main className="deck">
          <header className="top">
            <div>
              <div className="kicker">markets scanned {state.kalshi_metrics?.strikes?.length ?? 0} / trades {state.trades.length} / feed {formatRelativeTime(state.generated_at)}</div>
              <h2 className="truncate">{nav.title}</h2>
              <p className="truncate">{nav.subtitle}</p>
            </div>
            <div className="actions">
              <span className="chip"><span className={`dot ${effectiveStatus === 'online' ? '' : 'down-dot'}`} />live</span>
              <span className="chip">{new Date().toLocaleTimeString('en-US', { timeZone: 'America/New_York', hour12: false })} ET</span>
              <button className="command primary" type="button" onClick={() => setActiveScreen('command')}>Arm Guard</button>
            </div>
          </header>

          {error ? <div className="notice">Dashboard feed error: {error}</div> : null}
          {loading ? <div className="notice">Loading filterless dashboard state...</div> : null}

          <section className="ticker" aria-label="Session metrics">
            <div className="cell"><span>net</span><strong className={dailyPnl != null && dailyPnl < 0 ? 'down' : ''}>{formatMoney(dailyPnl)}</strong></div>
            <div className="cell"><span>trades</span><strong>{state.trades.length}</strong></div>
            <div className="cell"><span>risk</span><strong>{fmt(features.riskMult)}x</strong></div>
            <div className="cell"><span>live price</span><strong>{formatPrice(price)}</strong></div>
            <div className="cell"><span>entry</span><strong>{formatPrice(entry)}</strong></div>
            <div className="cell"><span>stop</span><strong>{formatPrice(stop)}</strong></div>
            <div className="cell"><span>tp</span><strong>{formatPrice(target)}</strong></div>
            <div className="cell"><span>truth</span><strong>{formatSigned(features.truthScore)}</strong></div>
          </section>

          {renderScreen()}
        </main>
      </div>
    </div>
  );
}

export default FilterlessLiveCockpit;
