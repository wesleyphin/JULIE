import React, { startTransition, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  AreaSeries,
  CandlestickSeries,
  ColorType,
  LineSeries,
  LineStyle,
  createChart,
  type IChartApi,
  type IPriceLine,
  type ISeriesApi,
  type UTCTimestamp,
} from 'lightweight-charts';
import Plotly from 'plotly.js-dist';
import type {
  FilterlessEvent,
  FilterlessKalshiMetrics,
  FilterlessLiveState,
  FilterlessOhlcBar,
  FilterlessPipelineState,
  FilterlessPosition,
  FilterlessSentimentMetrics,
  FilterlessStrategyState,
  FilterlessTrade,
} from './filterlessLiveTypes';

const REFRESH_MS = 3000;
const FEED_STALE_SECONDS = 90;
const FEED_OFFLINE_SECONDS = 300;
const MANIFOLD_IDLE_FPS = 24;
const OPERATOR_CONTROL_URL = 'http://127.0.0.1:3011';
const TAU = Math.PI * 2;

const COLORS = {
  // Restricted palette per user spec: only RED, GREEN, WHITE, and shades
  // of PURPLE for any text/UI accent. Lightened muted/dim/amber so they
  // stay legible on the frosted-grey glass + video bg.
  // Unified pink palette — all pink/purple/rose variants point to the same shade.
  purple: '#ffffff',
  violet: '#ffffff',
  cyan: '#ffffff',
  pink: '#ffffff',     // canonical pink
  amber: '#ffffff',
  lime: '#45ffc8',     // green (allowed)
  red: '#ff3864',
  green: '#45ffc8',
  muted: '#ffffff',
  text: '#ffffff',
  dim: '#ffffff',
};

type ScreenId = 'overview' | 'aetherflow' | 'kalshi' | 'news' | 'strategies' | 'pipeline' | 'journal' | 'command' | 'docs';

type ManifoldRegime = 'TREND_GEODESIC' | 'CHOP_SPIRAL' | 'DISPERSED' | 'ROTATIONAL_TURBULENCE';
type BadgeTone = 'live' | 'watch' | 'block' | 'info';
type OperatorCommandAction = 'restart_bot' | 'restart_bridge' | 'restart_frontend';

interface OperatorProcessStatus {
  name: string;
  pid?: number | null;
  running: boolean;
  exit_code?: number | null;
  restart_count?: number | null;
  watch_age_seconds?: number | null;
}

interface OperatorControlStatus {
  ok: boolean;
  generated_at: string;
  dashboard_state_age_seconds?: number | null;
  dashboard_state_path?: string | null;
  processes: OperatorProcessStatus[];
}

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
  pipeline: null,
};

const NAV: Array<{ id: ScreenId; label: string; code: string; title: string; subtitle: string }> = [
  { id: 'overview', label: 'Market Core', code: '01', title: 'MARKET CORE', subtitle: 'Live execution cockpit with entry, stop, target, news, and position context.' },
  { id: 'aetherflow', label: 'AetherFlow', code: '02', title: 'AETHERFLOW MANIFOLD', subtitle: 'Rotatable folded pressure surface with state shading, depth, and route features.' },
  { id: 'kalshi', label: 'Kalshi', code: '03', title: 'KALSHI MARKET ARRAY', subtitle: 'Hourly contract routing, book edge, spread, and consensus flow.' },
  { id: 'news', label: 'News', code: '04', title: 'NEWS AND TRUTH MONITOR', subtitle: 'Truth Social sentiment, macro tape, advisory risk, and calendar context.' },
  { id: 'strategies', label: 'Strategies', code: '05', title: 'STRATEGY STACK', subtitle: 'Filterless strategy modules mapped to current live state.' },
  { id: 'pipeline', label: 'V18 Pipeline', code: '06', title: 'V18 PIPELINE', subtitle: 'Stacked-meta + Kronos + Recipe B + regime ML — current live wiring.' },
  { id: 'journal', label: 'Journal', code: '07', title: 'TRACE JOURNAL', subtitle: 'Decision log with trade levels, news fields, and manifold snapshots.' },
  { id: 'command', label: 'Command', code: '08', title: 'COMMAND MATRIX', subtitle: 'Runtime controls, guard rails, and operator actions.' },
  { id: 'docs', label: 'Readme', code: '09', title: 'README', subtitle: 'Filterless live README — architecture, strategies, and recent changes.' },
];

const COCKPIT_CSS = `
@import url('https://fonts.googleapis.com/css2?family=Big+Shoulders+Display:wght@800;900&family=Black+Han+Sans&family=Archivo+Black&family=Russo+One&family=JetBrains+Mono:wght@400;600;700&family=Rajdhani:wght@400;500;600;700&display=swap');
@font-face {
  font-family: 'Ghastly Panic';
  src: url('/fonts/GhastlyPanic.ttf') format('truetype');
  font-display: swap;
}
@font-face {
  font-family: 'Zombified';
  src: url('/fonts/Zombified.ttf') format('truetype');
  font-display: swap;
}
:root {
  --bg: #000;
  /* Grey-blurred outline color — replaces the old purple. Anything that
     references var(--line) or var(--line-strong) now reads as a soft
     translucent grey. Hardcoded purple rgba()'s are swept separately. */
  --line: rgba(255, 255, 255, 0.10);
  --line-strong: rgba(255, 255, 255, 0.22);
  --text: #ffffff;
  /* Unified pink palette — every pink/purple/rose variant points to the same
     canonical pink so the dashboard reads with one consistent accent. */
  --muted: #ffffff;
  --dim: #ffffff;
  --purple: #ffffff;
  --violet: #ffffff;
  --cyan: #ffffff;
  --pink: #ffffff;
  --amber: #ffffff;
  --lime: #45ffc8;
  --red: #ff3864;
  --green: #45ffc8;
  --shadow: 0 0 0 1px rgba(255,255,255,0.04), 0 18px 60px rgba(0, 0, 0, 0.62);
  /* Streetwear-style title stack: Anton (bold condensed, the Supreme/Off-
     White poster look) → Bebas Neue (cleaner condensed) → Archivo Black
     (heavier slab-like) as fallbacks. JetBrains Mono and Rajdhani retained
     for data values and body text. */
  /* Stretched-out bold display stack — serious industrial/control-panel
     feel. Big Shoulders Display 900 is designed for wide signage; Black
     Han Sans is extra-wide and heavy; Archivo Black and Russo One are
     wide bold sans fallbacks. */
  --display: 'Arial', sans-serif;
  --body: 'Arial', sans-serif;
  --mono: 'Arial', sans-serif;
  /* Body / non-title font: Zombified by Chad Savage (same hand-drawn horror
     designer as Ghastly Panic). Self-hosted at /fonts/Zombified.ttf via the
     @font-face above. Pairs with Ghastly Panic by design lineage. */
  --archaic: 'Arial', sans-serif;
}
body { background: #000; color: var(--text); font-family: var(--archaic); letter-spacing: 0; overflow-x: hidden; }
button { font: inherit; color: inherit; }
canvas { display: block; width: 100%; }
h1, h2, h3, p { margin: 0; }
.fl-cockpit * { box-sizing: border-box; }
/* Curved corners on every box-like surface (panels already have 14px,
   chart 14px). Buttons, chips, badges, ticker cells, command tiles etc.
   pick this up so the entire UI reads as rounded glassmorphism. */
.chip, .command, .badge, .nav button, .ticker, .ticker .cell, .hud-cell,
.command-tile, .control-tile, .operator-note, .position, .terminal-row,
.event, .tile, .metric, .meter, .bar-track, .panel-head {
  border-radius: 10px;
}
.ticker { border-radius: 14px; overflow: hidden; }
.ticker .cell { border-radius: 0; }
.panel { border-radius: 14px; }
.panel-head { border-radius: 14px 14px 0 0; }

/* Looping background video — covers full viewport, sits behind every UI
   layer via negative z-index. fl-cockpit is its positioning context. */
.fl-cockpit { position: relative; min-height: 100vh; }
.bg-video {
  position: fixed;
  inset: 0;
  width: 100vw;
  height: 100vh;
  object-fit: cover;
  z-index: -2;
  pointer-events: none;
  /* 15% darker than before (was brightness 0.55, now 0.55 * 0.85 ≈ 0.47). */
  filter: brightness(0.47) contrast(1.05);
}
/* Subtle dark veil on top of the video so text stays legible. */
.fl-cockpit::before {
  content: '';
  position: fixed;
  inset: 0;
  z-index: -1;
  background: linear-gradient(180deg, rgba(0,0,0,0.55) 0%, rgba(0,0,0,0.42) 50%, rgba(0,0,0,0.6) 100%);
  pointer-events: none;
}
/* Bouncing head easter-egg — DVD-style floating PNG. mix-blend-mode: screen
   knocks out the black surrounding pixels so it looks like a floating head
   rather than a black square. translate3d is updated each frame via direct
   DOM write, no React re-render. */
.bouncing-head {
  position: fixed;
  top: 0;
  left: 0;
  width: 90px;
  height: 90px;
  z-index: 9999;
  cursor: pointer;
  pointer-events: auto;
  will-change: transform;
  filter: drop-shadow(0 0 14px rgba(255, 255, 255, 0.55));
  transition: filter 0.18s ease;
}
.bouncing-head:hover { filter: drop-shadow(0 0 22px rgba(255, 56, 100, 0.75)); }
.bouncing-head img {
  width: 100%;
  height: 100%;
  display: block;
  user-select: none;
  -webkit-user-drag: none;
  pointer-events: none;
}
.head-explosion {
  position: fixed;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
  z-index: 9998;
  pointer-events: none;
}
/* App + rail are now transparent so video shows through to the panels.
   Panels themselves get the frosted-grey glass treatment below. */
.app { min-height: 100vh; display: grid; grid-template-columns: 248px minmax(0, 1fr); background: transparent; }
.rail { min-width: 0; border-right: 1px solid rgba(255,255,255,0.08); background: rgba(20, 22, 30, 0.42); display: grid; grid-template-rows: auto 1fr auto; }
.brand { min-width: 0; padding: 18px 16px 16px; border-bottom: 1px solid var(--line); }
.brand h1 { font-family: var(--display); font-weight: 900; font-size: 13px; letter-spacing: 1.8px; color: #ffffff; text-shadow: 0 0 18px rgba(255, 255, 255, 0.9); }
.brand p { margin-top: 8px; color: var(--muted); font-size: 12px; line-height: 1.35; font-family: var(--mono); }
.brand .wire { height: 2px; margin-top: 15px; background: linear-gradient(90deg, var(--purple), transparent 70%); box-shadow: 0 0 18px rgba(255, 255, 255, 0.8); }
.nav { min-width: 0; padding: 12px; display: grid; align-content: start; gap: 6px; }
.nav button { min-width: 0; height: 40px; border: 1px solid transparent; background: transparent; border-radius: 0; display: grid; grid-template-columns: minmax(0, 1fr) auto; align-items: center; gap: 10px; padding: 0 9px; cursor: pointer; color: var(--muted); text-align: left; text-transform: uppercase; }
.nav button:hover, .nav button.active { color: var(--text); border-color: var(--line-strong); background: rgba(255,255,255,0.06); box-shadow: inset 2px 0 0 var(--purple), 0 0 20px rgba(255,255,255,0.06); }
.nav span { min-width: 0; font-weight: 700; letter-spacing: 0.8px; font-size: 12px; }
.nav small, .micro { font-family: var(--mono); font-size: 10px; color: var(--dim); }
.rail-bottom { min-width: 0; padding: 14px; border-top: 1px solid var(--line); display: grid; gap: 8px; }
.status-line { min-width: 0; display: grid; grid-template-columns: minmax(0, 1fr) auto; align-items: center; gap: 10px; font-family: var(--mono); font-size: 10px; color: var(--muted); }
.status-line strong { min-width: 0; color: var(--text); font-weight: 700; }
.dot { display: inline-block; width: 6px; height: 6px; margin-right: 6px; border-radius: 50%; background: var(--green); box-shadow: 0 0 14px var(--green); }
.dot.warn-dot { background: var(--amber); box-shadow: 0 0 14px var(--amber); }
.dot.down-dot { background: var(--red); box-shadow: 0 0 14px var(--red); }
.deck { min-width: 0; padding: 14px; display: grid; grid-template-rows: auto auto auto 1fr; gap: 10px; }
.top { min-width: 0; display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 14px; align-items: start; padding-bottom: 10px; }
.kicker { font-family: var(--mono); color: var(--muted); font-size: 10px; text-transform: uppercase; }
.top h2 { margin-top: 4px; font-family: var(--display); font-weight: 900; color: #ffffff; font-size: clamp(19px, 2.0vw, 30px); letter-spacing: 1px; text-shadow: 0 0 24px rgba(255, 255, 255, 0.85); }
.top p { margin-top: 6px; color: var(--muted); font-family: var(--mono); font-size: 11px; }
.actions { min-width: 0; display: flex; justify-content: flex-end; flex-wrap: wrap; gap: 8px; }
.chip, .command, .badge { min-width: 0; border: 1px solid var(--line); background: rgba(255,255,255,0.04); color: var(--text); height: 30px; padding: 0 10px; display: inline-flex; align-items: center; justify-content: center; gap: 7px; font-family: var(--mono); font-size: 10px; text-transform: uppercase; white-space: nowrap; }
.command { cursor: pointer; text-decoration: none; }
.command.primary { border-color: rgba(255, 255, 255, 0.6); color: var(--cyan); box-shadow: 0 0 18px rgba(255, 255, 255, 0.13); }
.notice { border: 1px solid rgba(255, 56, 100, 0.32); color: var(--red); background: rgba(255, 56, 100, 0.06); padding: 10px 12px; font-family: var(--mono); font-size: 11px; }
.ticker { min-width: 0; display: grid; grid-template-columns: repeat(8, minmax(0, 1fr)); border: 1px solid var(--line); background: rgba(20, 22, 30, 0.42); }
.ticker .cell { min-width: 0; padding: 9px 10px; border-right: 1px solid rgba(255,255,255,0.08); }
.ticker .cell:last-child { border-right: 0; }
.ticker span, .label { display: block; min-width: 0; color: var(--muted); font-size: 10px; font-family: var(--mono); text-transform: uppercase; }
.ticker strong { display: block; margin-top: 3px; color: var(--purple); font-family: var(--display); font-size: clamp(15px, 1.3vw, 22px); text-shadow: 0 0 18px rgba(255, 255, 255, 0.62); }
.screen { min-width: 0; display: block; }
.grid { min-width: 0; display: grid; gap: 10px; }
.cols-2 { grid-template-columns: repeat(2, minmax(0, 1fr)); }
.cols-3 { grid-template-columns: repeat(3, minmax(0, 1fr)); }
.cols-4 { grid-template-columns: repeat(4, minmax(0, 1fr)); }
.overview-layout { grid-template-columns: minmax(220px, 0.55fr) minmax(0, 2.6fr) minmax(240px, 0.55fr); margin-top: 10px; }
.aether-layout { grid-template-columns: minmax(0, 1.52fr) minmax(340px, 0.72fr); margin-top: 10px; }
.kalshi-layout { grid-template-columns: 1fr; margin-top: 10px; }
.kalshi-layout-secondary { grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); margin-top: 10px; gap: 10px; }
.news-layout, .journal-layout { grid-template-columns: minmax(0, 1.35fr) minmax(320px, 0.65fr); margin-top: 10px; }
/* Glassmorphism: frosted grey panels with curved corners. The translucent
   grey + backdrop-filter blur lets the video show through subtly while
   keeping content legible. Border-radius 14px gives the curved outline
   the user asked for. */
.panel, .metric, .event, .position, .terminal-row, .tile {
  min-width: 0;
  background: rgba(20, 22, 30, 0.42);
  border: 1px solid rgba(255, 255, 255, 0.10);
  border-radius: 14px;
  box-shadow: 0 0 0 1px rgba(255, 255, 255, 0.03), 0 18px 60px rgba(0, 0, 0, 0.45);
}
.panel { overflow: hidden; }
.panel-head { min-width: 0; min-height: 48px; display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 10px; align-items: center; padding: 10px 12px; }
.panel-head h3 { min-width: 0; font-family: var(--display); font-weight: 900; font-size: 12px; letter-spacing: 0.8px; text-transform: uppercase; color: #ffffff; text-shadow: 0 0 18px rgba(255, 255, 255, 0.7); }
.brand h1, .top h2, .ticker strong, .metric strong, .command-tile strong { font-weight: 900; }
.panel-head p { min-width: 0; margin-top: 3px; color: var(--muted); font-family: var(--mono); font-size: 10px; }
.panel-body { padding: 10px; }
.metric { height: 86px; padding: 11px; display: grid; align-content: space-between; border-color: rgba(255,255,255,0.12); }
.metric strong { min-width: 0; font-family: var(--display); font-size: clamp(18px, 1.8vw, 30px); color: var(--metric, var(--purple)); text-shadow: 0 0 18px color-mix(in srgb, var(--metric, var(--purple)), transparent 48%); }
.metric small { min-width: 0; color: var(--muted); font-family: var(--mono); font-size: 10px; }
.stack { min-width: 0; display: grid; gap: 10px; align-content: start; }
.terminal { padding: 8px; display: grid; gap: 6px; max-height: 520px; overflow: auto; }
.terminal-row { min-height: 38px; padding: 7px 8px; display: grid; grid-template-columns: 64px minmax(0, 1fr) auto; align-items: center; gap: 8px; font-family: var(--mono); font-size: 10px; }
.terminal-row time { color: var(--dim); }
.terminal-row strong { min-width: 0; color: var(--text); font-weight: 700; }
.terminal-row p { color: var(--muted); }
.badge { height: 22px; padding: 0 7px; border-color: rgba(255,255,255,0.18); color: var(--purple); }
.badge.live { color: var(--green); border-color: rgba(69, 255, 200, 0.45); }
.badge.watch { color: var(--amber); border-color: rgba(255, 255, 255, 0.45); }
.badge.block { color: var(--red); border-color: rgba(255, 56, 100, 0.52); }
.badge.info { color: var(--cyan); border-color: rgba(255, 255, 255, 0.45); }
.chart {
  height: 620px;
  background: rgba(20, 22, 30, 0.42);
  border-radius: 14px;
  overflow: hidden;
}
.scene-wrap {
  position: relative;
  min-height: 650px;
  background: rgba(20, 22, 30, 0.42);
  border-radius: 14px;
}
.aetherflow-scene { width: 100%; height: 650px; cursor: grab; touch-action: none; user-select: none; }
.aetherflow-scene:active { cursor: grabbing; }
.scene-hud { position: absolute; left: 12px; right: 12px; bottom: 12px; display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px; pointer-events: none; }
.hud-cell { min-width: 0; border: 1px solid rgba(255,255,255,0.10); background: rgba(180,184,200,0.08); padding: 8px; }
.hud-cell strong { display: block; margin-top: 2px; font-family: var(--mono); font-size: 11px; }
.mini-grid { min-width: 0; display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); border-top: 1px solid rgba(255,255,255,0.10); }
.mini { min-width: 0; padding: 9px 10px; border-right: 1px solid rgba(255,255,255,0.08); }
.mini:last-child { border-right: 0; }
.mini strong { display: block; min-width: 0; margin-top: 3px; font-family: var(--mono); font-size: 12px; }
.feature { min-width: 0; padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.06); }
.feature:last-child { border-bottom: 0; }
.feature-head { min-width: 0; display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 8px; margin-bottom: 7px; font-family: var(--mono); font-size: 10px; color: var(--muted); text-transform: uppercase; }
.meter { height: 8px; background: rgba(255, 255, 255, 0.045); border: 1px solid rgba(255,255,255,0.10); overflow: hidden; }
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
.bar-track { height: 16px; background: rgba(255, 255, 255, 0.045); border: 1px solid rgba(255,255,255,0.06); overflow: hidden; }
.bar-fill { height: 100%; width: 50%; background: linear-gradient(90deg, rgba(255,255,255,0.10), var(--purple)); }
.table { width: 100%; border-collapse: collapse; table-layout: fixed; font-family: var(--mono); font-size: 10px; }
.table th, .table td { min-width: 0; padding: 9px 8px; border-bottom: 1px solid rgba(255,255,255,0.06); text-align: left; color: var(--muted); }
.table th { color: var(--dim); text-transform: uppercase; font-weight: 700; }
.table td strong { color: var(--text); }
.event { min-height: 58px; padding: 9px; display: grid; grid-template-columns: 72px minmax(0, 1fr) auto; gap: 10px; align-items: center; font-family: var(--mono); font-size: 10px; }
.event time { color: var(--dim); }
.event strong { min-width: 0; display: block; color: var(--text); }
.event p { min-width: 0; margin-top: 4px; color: var(--muted); }
.command-grid { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 8px; }
.command-tile { height: 118px; border: 1px solid var(--line); background: rgba(180,184,200,0.08); padding: 10px; display: grid; align-content: space-between; }
.command-tile strong { min-width: 0; font-family: var(--display); font-size: 11px; }
.command-tile small { min-width: 0; color: var(--muted); font-family: var(--mono); font-size: 10px; }
.control-tile { cursor: pointer; color: inherit; text-align: left; }
.control-tile:hover { border-color: var(--line-strong); background: rgba(255,255,255,0.08); }
.control-tile:disabled { cursor: not-allowed; opacity: 0.48; }
.operator-footer { min-width: 0; margin-top: 9px; display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px; }
.operator-note { min-height: 34px; padding: 8px 10px; border: 1px solid rgba(255,255,255,0.08); background: rgba(180,184,200,0.06); color: var(--muted); font-family: var(--mono); font-size: 10px; }
.operator-note strong { display: block; color: var(--text); font-size: 10px; }
.up { color: var(--green) !important; }
.down { color: var(--red) !important; }
.info { color: var(--cyan) !important; }
.warn { color: var(--amber) !important; }
.violet { color: var(--purple) !important; }
.truncate { overflow: visible; white-space: normal; text-overflow: clip; word-break: break-word; overflow-wrap: anywhere; min-width: 0; }
.clamp { display: -webkit-box; -webkit-box-orient: vertical; -webkit-line-clamp: 2; overflow: hidden; }

@media (max-width: 1180px) {
  .app { grid-template-columns: 1fr; }
  .rail { position: sticky; top: 0; z-index: 5; grid-template-rows: auto auto; border-right: 0; border-bottom: 1px solid var(--line); }
  .brand { display: none; }
  .nav { grid-template-columns: repeat(7, minmax(0, 1fr)); }
  .rail-bottom { display: none; }
  .overview-layout, .aether-layout, .kalshi-layout, .news-layout, .journal-layout { grid-template-columns: 1fr; }
}
.pipeline-grid { grid-template-columns: repeat(4, minmax(0, 1fr)); }
/* Phone-only dropdown nav: hidden on desktop/tablet, shown only at <=480px */
.nav-select { display: none; }
@media (max-width: 820px) {
  .deck { padding: 10px; }
  /* Tablet: keep cols-2/3/4 multi-column (preserves desktop layout shape).
     Only collapse the things that genuinely need single-column space. */
  .top, .scene-hud { grid-template-columns: 1fr; }
  .nav { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .nav button { height: 34px; }
  .actions { justify-content: flex-start; }
  .scene-wrap { min-height: 500px; }
  .aetherflow-scene { height: 500px; }
}
@media (max-width: 480px) {
  /* iPhone 16 portrait (393 CSS px) and similar phones.
     PRESERVE multi-column layouts (cols-2/3/4) like desktop — just shrink
     content tight enough to fit. Chart stays as-is since it's its own
     full-width panel. */
  body { -webkit-text-size-adjust: 100%; }
  /* Phone: kill the rail's frosted-grey background entirely. The rail
     wraps the dropdown — when its bg renders, the dropdown looks like
     it's inside a separate rounded rectangle. Going transparent + 0
     min-height collapses that container to just the dropdown itself.
     The 1fr grid-row that was stretching empty space (especially on
     Strategies tab) is also collapsed via grid-template-rows: auto. */
  .rail {
    position: static !important;
    top: auto !important;
    background: transparent !important;
    backdrop-filter: none !important;
    -webkit-backdrop-filter: none !important;
    border: 0 !important;
    grid-template-rows: auto !important;
    min-height: 0 !important;
    height: auto !important;
    padding: 0 !important;
  }
  /* The .app grid was distributing the 100vh min-height across both
     children (rail + deck) via the default align-content stretch,
     making the rail balloon to half the screen with empty space below
     the dropdown. align-content: start packs both children at the top
     so the rail stays at content height (just the dropdown). */
  .app {
    align-content: start !important;
    grid-auto-rows: min-content !important;
  }
  .deck { padding: 6px; gap: 6px; }
  .top { gap: 6px; padding-bottom: 6px; grid-template-columns: 1fr; }
  /* Title + subtitle wrap onto multiple rows on phone instead of being
     truncated with an ellipsis. The .truncate class normally forces
     nowrap+ellipsis; here we override it for the screen-level header
     specifically so long titles like "STRATEGY STACK" and long
     subtitles like "Live execution cockpit with entry, stop, target,
     news, and position context." fit cleanly within the viewport. */
  .top h2.truncate, .top p.truncate, .top .kicker {
    white-space: normal;
    overflow: visible;
    text-overflow: clip;
    word-break: break-word;
  }
  .top h2 { font-size: 16px; letter-spacing: 0.5px; line-height: 1.15; }
  .top p { font-size: 9px; line-height: 1.3; }
  .top .kicker { font-size: 8px; line-height: 1.3; }
  .actions { gap: 4px; justify-content: flex-start; }
  .actions .chip, .actions .command { font-size: 8px; height: 24px; padding: 0 6px; }
  .ticker { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .ticker .cell { padding: 8px 10px; border-right: 0; border-bottom: 1px solid rgba(255,255,255,0.08); }
  .ticker .cell:nth-child(odd) { border-right: 1px solid rgba(255,255,255,0.08); }
  .ticker .cell:nth-last-child(-n+2) { border-bottom: 0; }
  .ticker strong { font-size: 10px; }
  .ticker span { font-size: 7px; }
  /* Phone: dropdown wrapped in a curved dark-grey blurred glass box
     matching the panel/chart aesthetic. The wrapper provides the
     visible box; the inner <select> is fully transparent so iOS
     can't draw its own native rectangle inside the curve. */
  .nav { display: none; }
  .nav-select {
    display: block;
    position: relative;
    margin: 8px 12px;
    padding: 0;
    background: rgba(20, 22, 30, 0.42);
    border: 1px solid rgba(255, 255, 255, 0.10);
    border-radius: 14px;
    overflow: hidden;
    box-shadow: 0 0 0 1px rgba(255, 255, 255, 0.03);
  }
  .nav-select select {
    width: 100%;
    height: 38px;
    padding: 0 30px 0 14px;
    background: transparent !important;
    color: var(--text);
    border: 0 !important;
    border-radius: 0 !important;
    font-family: var(--display);
    font-size: 12px;
    letter-spacing: 0.6px;
    text-transform: uppercase;
    appearance: none;
    -webkit-appearance: none;
    cursor: pointer;
    outline: none !important;
    box-shadow: none !important;
    -webkit-tap-highlight-color: transparent;
  }
  .nav-select select:focus,
  .nav-select select:focus-visible,
  .nav-select select:active,
  .nav-select select:hover {
    outline: none !important;
    box-shadow: none !important;
    border: 0 !important;
    background: transparent !important;
  }
  .nav-select-caret {
    position: absolute;
    right: 14px;
    top: 50%;
    transform: translateY(-50%);
    color: var(--muted);
    font-size: 11px;
    pointer-events: none;
    opacity: 0.7;
  }
  /* Allow paragraph text marked .truncate to wrap onto multiple lines
     on phone instead of being clipped with "...". Headers/labels marked
     .truncate (h1/strong/small/span) stay nowrap so tight chips don't
     blow up vertically. */
  p.truncate {
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: clip !important;
    word-break: break-word;
    line-height: 1.3;
  }
  .terminal-row p, .terminal-row p.truncate,
  .event p, .event p.truncate,
  .position p, .position p.truncate,
  .tile p, .tile p.truncate,
  .operator-note span, .operator-note span.truncate,
  .command-tile small, .command-tile small.truncate {
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: clip !important;
    word-break: break-word;
    line-height: 1.25;
  }
  /* Stack badge UNDER the title on phone — at ~95px column width the
     side-by-side grid was overlapping "EXECUTION LOGIC" with "armed".
     Vertical stack uses ~6 more px but eliminates the collision entirely.
     Inner div MUST have min-width:0 + max-width:100% or its h3/p escape
     the grid cell and bleed past the panel border (children of grid
     cells default to min-content size, which forces the cell wider). */
  .panel-head {
    padding: 7px 10px;
    min-height: 28px;
    gap: 3px;
    grid-template-columns: 1fr;
    grid-auto-rows: auto;
    overflow: hidden;
  }
  .panel-head > * {
    justify-self: start;
    min-width: 0;
    max-width: 100%;
  }
  .panel-head h3 {
    /* Bungee is very wide — 9px keeps "EXECUTION LOGIC" / "ACTIVE
       POSITIONS" inside a 190px phone column. */
    font-size: 9px;
    letter-spacing: 0.4px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    max-width: 100%;
  }
  /* Subtitle is decorative ("Execution score, guard rails, and live
     context." etc.). Wrap onto multiple rows + tiny font so it fits
     under the title at ~80px column width. */
  .panel-head p {
    display: block;
    font-size: 5px;
    line-height: 1.25;
    margin-top: 2px;
    color: var(--muted);
    white-space: normal;
    overflow: visible;
    text-overflow: clip;
    word-break: break-word;
  }
  .panel-body { padding: 8px 10px; }
  /* Multi-column rows on phone — kept multi-col so layout reads like
     desktop, but cols-4 collapses to 2x2 so panels under the chart don't
     end up tall+narrow (elongated). 4-col was making each panel ~95px
     wide × 200px+ tall which looked stretched. 2-col means ~190px wide,
     much shorter, proportionate. */
  .cols-3 { gap: 4px; grid-template-columns: repeat(3, minmax(0, 1fr)); }
  .cols-4 { gap: 4px; grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .cols-2 { gap: 4px; grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .pipeline-grid { gap: 4px; grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .state-matrix, .truth-grid, .mini-grid { gap: 4px; }
  .grid { gap: 4px; }
  /* Compact metric/panel content so they fit at ~95px column width.
     Padding bumped from tight (5-6px) to comfy (8-10px) so text isn't
     crammed against box edges. */
  .metric { height: 70px; padding: 8px 10px; }
  /* Big glowing values (NET / RISK / LIVE PRICE / etc) bumped 11 -> 14
     so the headline numbers actually read on phone. */
  .metric strong { font-size: 14px; }
  .metric small, .label { font-size: 7px; letter-spacing: 0.4px; }
  .mini { padding: 8px 9px; }
  .mini strong { font-size: 9px; }
  .terminal { max-height: 220px; padding: 6px; gap: 3px; }
  .terminal-row { grid-template-columns: 28px minmax(0, 1fr) auto; padding: 5px 7px; min-height: 24px; gap: 4px; font-size: 6px; }
  .terminal-row strong { font-size: 6px; }
  .terminal-row p { font-size: 6px; }
  .terminal-row time { font-size: 5px; }
  .command-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 4px; }
  .command-tile { height: 76px; padding: 8px 10px; }
  .command-tile strong { font-size: 8px; }
  .command-tile small { font-size: 7px; }
  .operator-footer { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .scene-wrap { min-height: 380px; }
  .aetherflow-scene { height: 380px; }
  .scene-hud { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  /* Chart on phone — additional 20% trim. 465 * 0.80 = 372px. */
  .chart { height: 372px; }
  /* Inline overlay labels on the chart — even tinier on phone so they
     don't dominate the candle area. The .ov-label class is set via React
     on the absolute-positioned label divs. !important is needed because
     the React inline style sets fontSize: 8 with higher specificity. */
  .ov-label {
    font-size: 6px !important;
    line-height: 8px !important;
    padding: 0px 3px !important;
    border-radius: 1px !important;
  }
  /* Keep table as native table-layout (NOT display:block) so width:100%
     works and the table fills its parent panel — display:block was
     making the table shrink-wrap to content width and leaving a huge
     gap to the right of e.g. the Kalshi Market Scanner. */
  .table { font-size: 7px; width: 100%; table-layout: fixed; }
  .table th, .table td { padding: 6px 5px; word-break: break-word; }
  .badge { height: 16px; padding: 0 4px; font-size: 7px; }
  .chip { font-size: 7px; height: 20px; padding: 0 5px; }
  .tile { min-height: 70px; padding: 8px 10px; }
  .tile strong { font-size: 10px; }
  .tile p { font-size: 8px; line-height: 1.2; margin-top: 4px; }
  .position { padding: 7px 9px; min-height: 32px; gap: 5px; }
  .position strong { font-size: 7px; }
  .position p { font-size: 6px; }
  .feature { padding: 5px 2px; }
  .feature-head { font-size: 6px; gap: 4px; margin-bottom: 2px; }
  .meter { height: 4px; }
  .flow-bars { padding: 7px 9px; gap: 3px; }
  .flow-bar { grid-template-columns: 24px minmax(0, 1fr) 28px; gap: 4px; font-size: 6px; height: 12px; }
  .bar-track { height: 8px; }
  .notice { padding: 5px 7px; font-size: 8px; }
  .event { grid-template-columns: 36px minmax(0, 1fr) auto; padding: 7px 9px; min-height: 38px; font-size: 7px; gap: 6px; }
}

/* ----------------------------------------------------------------------
   Archaic body sweep — switches all label/text/subtitle/description
   selectors that previously used var(--mono) to var(--archaic), preserving
   monospace on numeric value selectors for column alignment. Italic on
   descriptive subtitle paragraphs (brand / page H2 / panel-head) for the
   manuscript-narration feel. Title selectors (Ghastly Panic) and big
   value selectors (var(--display) — .metric strong, .ticker strong) are
   intentionally untouched.
---------------------------------------------------------------------- */
.brand p,
.top p,
.panel-head p,
.tile p,
.nav small, .micro,
.status-line,
.kicker,
.chip, .command, .badge,
.notice,
.ticker span, .label,
.metric small,
.terminal-row,
.terminal-row p,
.terminal-row strong:not(.display-title),
.position, .position p, .position strong,
.feature-head, .feature-head span,
.flow-bar, .flow-bar span,
.table th,
.event, .event strong, .event p,
.command-tile small,
.operator-note {
  font-family: var(--archaic);
}

/* Italic on the descriptive subtitle paragraphs only — gives a manuscript
   feel without tiring the eye on short labels. */
.brand p,
.top p,
.panel-head p {
  font-style: italic;
}

/* Numeric value displays — keep monospace for column alignment. These
   would otherwise inherit archaic from the row/group rules above. */
.terminal-row time,
.feature-head strong,
.flow-bar strong,
.table td,
.event time,
.hud-cell strong,
.mini strong {
  font-family: var(--mono);
}

/* Description texts under titles — hidden per user. Reversible: delete
   this rule and the descriptions return to view (JSX is untouched, the
   subtitle props still get rendered into the DOM, just visually hidden). */
.brand p,
.top p,
.panel-head p {
  display: none !important;
}

/* Page H2 titles get Helvetica Bold for emphasis above the otherwise-uniform
   Arial UI. Placed at end of CSS so the same-specificity rule above
   (.top h2 { font-family: var(--display); ... }) is overridden by source
   order. Other titles (panel headers, tiles) stay Arial regular. */
.top h2 {
  font-family: 'Helvetica', 'Arial', sans-serif;
  font-weight: bold;
}

/* === Strategy Stack cards === */
.strategy-card-grid {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 12px;
  margin-top: 4px;
}
.strategy-card {
  flex: 0 1 380px;
  min-width: 320px;
  max-width: 460px;
  background: rgba(0, 0, 0, 0.35);
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 6px;
  padding: 12px 14px;
  display: flex;
  flex-direction: column;
  gap: 10px;
  transition: transform 120ms ease, box-shadow 120ms ease, border-color 120ms ease;
}
.strategy-card:hover {
  transform: translateY(-1px);
  box-shadow: 0 4px 18px rgba(0, 0, 0, 0.45);
  border-color: rgba(255, 255, 255, 0.18);
}
.strategy-card-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 8px;
}
.strategy-card-title {
  font-family: var(--display, 'Helvetica', sans-serif);
  font-weight: 700;
  font-size: 16px;
  letter-spacing: 0.02em;
  text-transform: uppercase;
  margin: 0;
}
.strategy-card-id {
  opacity: 0.55;
  font-size: 10px;
  letter-spacing: 0.06em;
  margin-top: 2px;
}
.strategy-substrategies {
  display: flex;
  flex-direction: column;
  gap: 5px;
}
.strategy-substrategies-label {
  opacity: 0.55;
  font-size: 9.5px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}
.strategy-chip-row {
  display: flex;
  flex-wrap: wrap;
  gap: 5px;
}
.strategy-chip {
  font-size: 11px;
  font-family: 'SF Mono', 'Menlo', monospace;
  padding: 3px 8px;
  border-radius: 3px;
  border: 1px solid rgba(255, 255, 255, 0.12);
  color: rgba(255, 255, 255, 0.45);
  background: rgba(255, 255, 255, 0.03);
  letter-spacing: 0.02em;
  white-space: nowrap;
  transition: all 140ms ease;
}
.strategy-chip-idle { /* default — see base .strategy-chip */ }
.strategy-chip-recent {
  border-width: 1px;
  font-weight: 600;
}
.strategy-chip-active {
  border-width: 1.5px;
  font-weight: 700;
  box-shadow: 0 0 8px currentColor;
  animation: chipPulse 1.6s ease-in-out infinite;
}
@keyframes chipPulse {
  0%, 100% { box-shadow: 0 0 6px currentColor; }
  50% { box-shadow: 0 0 14px currentColor; }
}
.strategy-card-stats {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 8px;
  padding: 8px 0;
  border-top: 1px solid rgba(255, 255, 255, 0.06);
  border-bottom: 1px solid rgba(255, 255, 255, 0.06);
}
.strategy-stat {
  display: flex;
  flex-direction: column;
  gap: 2px;
  min-width: 0;
}
.strategy-stat-label {
  font-size: 9.5px;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  opacity: 0.55;
}
.strategy-stat-value {
  font-family: var(--display, 'Helvetica', sans-serif);
  font-size: 13px;
  font-weight: 600;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.strategy-stat-sub {
  font-weight: 400;
  opacity: 0.75;
  font-size: 12px;
}
.strategy-stat-meta {
  opacity: 0.5;
  font-size: 10px;
}
.strategy-card-footer {
  display: flex;
  flex-wrap: wrap;
  gap: 5px;
  align-items: center;
  margin-top: 2px;
}
.strategy-tag {
  font-family: 'SF Mono', 'Menlo', monospace;
  font-size: 10px;
  padding: 2px 6px;
  border-radius: 2px;
  border: 1px solid currentColor;
  background: transparent;
  letter-spacing: 0.02em;
}
.strategy-tag-block {
  border-color: var(--red, #ef4444);
  color: var(--red, #ef4444);
}
.strategy-tag-muted {
  opacity: 0.55;
}
@media (max-width: 720px) {
  .strategy-card {
    flex: 1 1 100%;
    max-width: 100%;
  }
  .strategy-card-stats {
    grid-template-columns: 1fr;
    gap: 6px;
  }
}

/* === Terminal-row stacked timestamp (Trade Blotter, Trace Log) === */
.terminal-row time.terminal-stamp-stacked {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 1px;
  line-height: 1.05;
}
.terminal-row .terminal-stamp-date {
  font-size: 9px;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  opacity: 0.55;
  font-family: 'SF Mono', 'Menlo', monospace;
}
.terminal-row .terminal-stamp-time {
  font-variant-numeric: tabular-nums;
}

/* === Daily Journal cards === */
.daily-journal-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 12px;
  margin-top: 4px;
  align-items: stretch;
  justify-items: stretch;
}
.daily-journal-card {
  width: 100%;
  background: rgba(0, 0, 0, 0.35);
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 6px;
  padding: 12px 14px;
  display: flex;
  flex-direction: column;
  gap: 8px;
  transition: transform 120ms ease, box-shadow 120ms ease, border-color 120ms ease;
}
.daily-journal-card:hover {
  transform: translateY(-1px);
  box-shadow: 0 4px 18px rgba(0, 0, 0, 0.45);
  border-color: rgba(255, 255, 255, 0.18);
}
.daily-journal-head {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 8px;
}
.daily-journal-date {
  font-family: var(--display, 'Helvetica', sans-serif);
  font-weight: 700;
  font-size: 13px;
  letter-spacing: 0.02em;
  text-transform: uppercase;
}
.daily-journal-pnl {
  font-family: var(--display, 'Helvetica', sans-serif);
  font-size: 22px;
  font-weight: 700;
  font-variant-numeric: tabular-nums;
  letter-spacing: 0.01em;
}
.daily-journal-stats {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 4px 10px;
  padding-top: 6px;
  border-top: 1px solid rgba(255, 255, 255, 0.06);
}
.daily-journal-stat {
  display: flex;
  justify-content: space-between;
  font-size: 11px;
}
.daily-journal-stat span {
  opacity: 0.55;
  font-size: 9.5px;
  letter-spacing: 0.05em;
  text-transform: uppercase;
}
.daily-journal-stat strong {
  font-family: 'SF Mono', 'Menlo', monospace;
  font-weight: 600;
  font-variant-numeric: tabular-nums;
}
.daily-journal-context {
  padding: 4px 0;
  border-top: 1px solid rgba(255, 255, 255, 0.06);
  border-bottom: 1px solid rgba(255, 255, 255, 0.06);
  opacity: 0.7;
  font-family: 'SF Mono', 'Menlo', monospace;
  font-size: 10px;
  letter-spacing: 0.01em;
}
.daily-journal-flags {
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.daily-journal-flag {
  background: rgba(255, 56, 100, 0.06);
  border-left: 2px solid rgba(255, 56, 100, 0.4);
  padding: 4px 8px;
  font-size: 10px;
  line-height: 1.35;
  border-radius: 0 3px 3px 0;
  opacity: 0.85;
}

/* === Trade-lifecycle charts (MFE/MAE scatter + Sankey) === */
.lifecycle-chart-wrap {
  position: relative;
  width: 100%;
  height: 460px;
  min-height: 380px;
  background: rgba(0, 0, 0, 0.25);
  border: 1px solid rgba(255, 255, 255, 0.06);
  border-radius: 6px;
  padding: 8px;
}
.lifecycle-empty {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 100%;
  height: 100%;
  font-family: 'SF Mono', 'Menlo', monospace;
  font-size: 11px;
  opacity: 0.55;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}

/* === Docs (README.pdf) viewer === */
.docs-panel-body {
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.docs-toolbar {
  display: flex;
  gap: 10px;
  align-items: center;
  flex-wrap: wrap;
  padding-bottom: 8px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.08);
}
.docs-link {
  font-family: var(--display, 'Helvetica', sans-serif);
  font-size: 11px;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  padding: 6px 12px;
  border: 1px solid rgba(255, 255, 255, 0.18);
  border-radius: 4px;
  color: inherit;
  text-decoration: none;
  transition: border-color 120ms ease, background 120ms ease;
}
.docs-link:hover {
  border-color: rgba(255, 255, 255, 0.45);
  background: rgba(255, 255, 255, 0.05);
}
.docs-hint {
  margin-left: auto;
  font-family: 'SF Mono', 'Menlo', monospace;
  font-size: 10px;
  opacity: 0.55;
}
.docs-viewer {
  position: relative;
  width: 100%;
  height: calc(100vh - 140px);
  min-height: 1100px;
  background: #1c1c1c;
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 6px;
  overflow: hidden;
}
.docs-viewer object,
.docs-viewer iframe,
.docs-frame-fallback {
  width: 100%;
  height: 100%;
  border: 0;
  display: block;
}
.docs-fallback-note {
  padding: 24px;
  font-size: 12px;
  opacity: 0.7;
  text-align: center;
}
.docs-fallback-note a {
  color: inherit;
  text-decoration: underline;
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
  const now = new Date();
  const sameDay =
    date.getFullYear() === now.getFullYear() &&
    date.getMonth() === now.getMonth() &&
    date.getDate() === now.getDate();
  const time = date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  if (sameDay) return time;
  const day = date.toLocaleDateString([], { month: 'short', day: '2-digit' });
  return `${day} ${time}`;
}

// Date/time split helper for stacked rendering (Trade Blotter, Trace Log, etc.)
// Returns `{ date: 'Apr 28' | null, time: '14:53' }` so the UI can render the
// date as its own row above the time when applicable.
function splitTimeForStackedDisplay(value?: string | null): { date: string | null; time: string } {
  if (!value) return { date: null, time: '--' };
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return { date: null, time: '--' };
  const now = new Date();
  const sameDay =
    d.getFullYear() === now.getFullYear() &&
    d.getMonth() === now.getMonth() &&
    d.getDate() === now.getDate();
  const time = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  if (sameDay) return { date: null, time };
  const datePart = d.toLocaleDateString([], { month: 'short', day: '2-digit' });
  return { date: datePart, time };
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

function formatAgeSeconds(value?: number | null): string {
  if (value == null || Number.isNaN(value)) return '--';
  if (value < 60) return `${Math.round(value)}s`;
  if (value < 3600) return `${Math.round(value / 60)}m`;
  return `${Math.round(value / 3600)}h`;
}

function findOperatorProcess(status: OperatorControlStatus | null, token: string): OperatorProcessStatus | null {
  const needle = token.toLowerCase();
  return status?.processes.find((process) => process.name.toLowerCase().includes(needle)) ?? null;
}

function processTone(process?: OperatorProcessStatus | null): BadgeTone {
  if (!process) return 'watch';
  if (!process.running) return 'block';
  return 'live';
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
  return `rgba(${clip(Math.round(rgb.r * s), 0, 255)}, ${clip(Math.round(rgb.g * s), 0, 255)}, ${clip(Math.round(rgb.b * s), 0, 255)}, ${alpha})`;
}

function alphaColor(hex: string, alpha: number): string {
  const rgb = hexToRgb(hex);
  return `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, ${alpha})`;
}

function rgbToHex(value: number): string {
  return clip(Math.round(value), 0, 255).toString(16).padStart(2, '0');
}

function blendHexColors(fromHex: string, toHex: string, amount: number): string {
  const from = hexToRgb(fromHex);
  const to = hexToRgb(toHex);
  const mix = clip01(amount);
  return `#${rgbToHex(from.r + ((to.r - from.r) * mix))}${rgbToHex(from.g + ((to.g - from.g) * mix))}${rgbToHex(from.b + ((to.b - from.b) * mix))}`;
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

function companionDominantColor(name: string): string {
  if (name === 'trend') return COLORS.cyan;
  if (name === 'burst') return COLORS.lime;
  if (name === 'dispersed') return COLORS.pink;
  if (name === 'rot') return COLORS.amber;
  return COLORS.violet;
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

function manifoldIdleWave(x: number, y: number, f: AetherFeatures, timeSeconds: number): number {
  const swellAmp = 0.01 + (0.015 * f.foldDepth) + (0.009 * f.transitionEnergy);
  const rippleAmp = 0.005 + (0.006 * f.stress) + (0.004 * f.burstPressure);
  const microAmp = 0.0025 + (0.004 * f.transitionEnergy) + (0.002 * f.novelty);
  const stream = Math.sin((timeSeconds * 1.02) + (x * 4.8) - (y * 6.2) + (f.pressure30 * 1.7));
  const cross = Math.sin((timeSeconds * 0.78) + ((x + y) * 5.8) + (f.directionalBias * 1.9));
  const chopA = Math.sin((timeSeconds * 2.3) + (x * 13.6) + (y * 8.4) + (f.R * 2.1));
  const chopB = Math.cos((timeSeconds * 2.9) - (x * 17.2) + (y * 11.8) + (f.stress * 2.7));
  const chopC = Math.sin((timeSeconds * 3.7) + (x * 23.4) - (y * 19.6) + Math.sin((x - y) * 4.2));
  const shimmer = Math.sin((timeSeconds * 4.4) + (x * 31.0) + (y * 14.5)) * Math.cos((timeSeconds * 2.6) - (x * 9.8) + (y * 21.0));
  return (swellAmp * ((0.58 * stream) + (0.42 * cross))) + (rippleAmp * ((0.4 * chopA) + (0.35 * chopB) + (0.25 * chopC))) + (microAmp * shimmer);
}

function idleColorPulse(x: number, y: number, wave: number, f: AetherFeatures, timeSeconds: number): number {
  const tide = 0.5 + (0.5 * Math.sin((timeSeconds * 0.58) + (x * 2.4) + (y * 3.1)));
  const sparkle = 0.5 + (0.5 * Math.sin((timeSeconds * 2.6) + (x * 16.0) - (y * 12.0)));
  return (tide * 0.055) + (sparkle * 0.025) + (wave * 1.1) + (0.025 * f.transitionEnergy);
}

function shiftedManifoldColor(dominantName: string, x: number, y: number, wave: number, f: AetherFeatures, timeSeconds: number, strength: number): string {
  const baseHex = dominantColor(dominantName);
  const drift = 0.5 + (0.5 * Math.sin((timeSeconds * 0.34) + (x * 2.2) - (y * 1.7) + f.pressure30));
  const glint = 0.5 + (0.5 * Math.sin((timeSeconds * 1.6) + (x * 9.0) + (y * 7.2)));
  const eddy = Math.sin((timeSeconds * 0.82) + (x * 8.8) - (y * 7.4) + (f.transitionEnergy * 2.1));
  const colorX = clip(
    x + (strength * ((0.045 * Math.sin((timeSeconds * 0.42) + (y * 5.6) + f.pressure30)) + (0.024 * eddy))),
    -1,
    1,
  );
  const colorY = clip01(
    y + (strength * ((0.038 * Math.cos((timeSeconds * 0.5) + (x * 4.8) - f.directionalBias)) - (0.02 * eddy))),
  );
  const shiftedDominant = pressurePlaneDominant(colorX, colorY, f);
  const shiftedHex = dominantColor(shiftedDominant);
  const companionHex = companionDominantColor(dominantName);
  const coolTone = blendHexColors(COLORS.cyan, COLORS.violet, 0.22 + (0.38 * drift));
  const warmTone = blendHexColors(COLORS.green, COLORS.amber, clip01(f.burstPressure + (0.35 * glint)));
  const localTone = blendHexColors(companionHex, shiftedHex, shiftedDominant === dominantName ? 0.28 : 0.72);
  const localMix = (0.055 + (0.075 * drift) + (0.035 * clip01(Math.abs(wave) * 18))) * strength;
  const boundaryMix = (shiftedDominant === dominantName ? 0.045 + (0.03 * glint) : 0.18 + (0.1 * glint)) * strength;
  const coolMix = (0.06 + (0.07 * drift) + (0.04 * clip01(f.foldDepth + Math.abs(wave) * 8))) * strength;
  const warmMix = (0.025 + (0.035 * glint * clip01(f.transitionEnergy + f.burstPressure))) * strength;
  return blendHexColors(
    blendHexColors(
      blendHexColors(
        blendHexColors(baseHex, localTone, localMix),
        shiftedHex,
        boundaryMix,
      ),
      coolTone,
      coolMix,
    ),
    warmTone,
    warmMix,
  );
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
    // Center vertically lifted from 0.62 -> 0.42 so the plot sits well
    // above the bottom HUD (pressure x / transition y / height z / phase).
    y: (height * (0.42 + scene.pitch * 0.04)) + ((rx + ry) * sy) - (z * lift),
    depth: ry + z * 0.7,
    z,
  };
}

const Panel: React.FC<{ title: string; subtitle?: string; badge?: React.ReactNode; children: React.ReactNode; className?: string; titleClassName?: string }> = ({
  title,
  subtitle,
  badge,
  children,
  className,
  titleClassName,
}) => (
  <div className={`panel ${className || ''}`}>
    <div className="panel-head">
      <div>
        <h3 className={titleClassName}>{title}</h3>
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

const TerminalRow: React.FC<{ time?: string | null; title: React.ReactNode; text: React.ReactNode; badge?: React.ReactNode; titleClassName?: string }> = ({
  time,
  title,
  text,
  badge,
  titleClassName,
}) => {
  const stamp = splitTimeForStackedDisplay(time);
  return (
    <div className="terminal-row">
      <time className={stamp.date ? 'terminal-stamp-stacked' : 'terminal-stamp'}>
        {stamp.date ? <span className="terminal-stamp-date">{stamp.date}</span> : null}
        <span className="terminal-stamp-time">{stamp.time}</span>
      </time>
      <div className="truncate">
        <strong className={`truncate${titleClassName ? ' ' + titleClassName : ''}`}>{title}</strong>
        <p className="truncate">{text}</p>
      </div>
      {badge}
    </div>
  );
};

const Tile: React.FC<{ title: string; text: string; color?: string; badge?: React.ReactNode; active?: boolean; titleClassName?: string }> = ({
  title,
  text,
  color = COLORS.purple,
  badge,
  active,
  titleClassName,
}) => (
  <div className={`tile ${active ? 'active' : ''}`} style={{ '--tile': color } as React.CSSProperties}>
    <div className="tile-row">
      <strong className={`truncate${titleClassName ? ' ' + titleClassName : ''}`}>{title}</strong>
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
          <div className="bar-fill" style={{ width: pct(item.value), background: item.color ? `linear-gradient(90deg, rgba(255, 255, 255,0.22), ${item.color})` : undefined }} />
        </div>
        <strong className="truncate">{item.tail ?? pct(item.value)}</strong>
      </div>
    ))}
  </div>
);

// ─── Pine-script-port: session levels + ORB + bank levels ───────────────
// Original Pine: True Open / Sessions / ORB / SPY indicator (PT-defined).
// All session boundaries below are EXPRESSED IN ET — same absolute moments,
// just different clock labels. PT 18:00 == ET 21:00, etc.
//
// Sessions (ET):
//   Asia:    21:00 - 02:59 ET (wraps midnight)
//   London:  03:00 - 08:59 ET
//   NY:      09:00 - 14:59 ET
//   PM:      15:00 - 20:59 ET
//
// True Open: bar AT (session_start + 1.5h). e.g. NY TO bar = 10:30 ET.
// Daily Open: 18:00 ET (futures session start).
// Midnight ORB: 03:00 - 03:30 ET (was 00:00 - 00:30 PT).
// Morning ORB:  09:30 - 10:00 ET (was 06:30 - 07:00 PT).

type SessionId = 'Asia' | 'London' | 'NY' | 'PM';

interface ETBar {
  ts: Date;            // wall-clock as ET-interpreted Date (for hour/min ops)
  unix: number;        // unix seconds (chart time axis)
  open: number;
  high: number;
  low: number;
  close: number;
}

function etPartsFor(unixSec: number): { hour: number; minute: number; date: string } {
  const d = new Date(unixSec * 1000);
  const parts = new Intl.DateTimeFormat('en-US', {
    timeZone: 'America/New_York',
    year: 'numeric', month: '2-digit', day: '2-digit',
    hour: '2-digit', minute: '2-digit', hour12: false,
  }).formatToParts(d);
  const get = (k: string) => Number(parts.find((p) => p.type === k)?.value ?? '0');
  const h = get('hour') % 24;
  return {
    hour: h,
    minute: get('minute'),
    date: `${parts.find((p) => p.type === 'year')?.value}-${parts.find((p) => p.type === 'month')?.value}-${parts.find((p) => p.type === 'day')?.value}`,
  };
}

function sessionForET(hourET: number): SessionId {
  if (hourET >= 21 || hourET < 3) return 'Asia';
  if (hourET >= 3 && hourET < 9) return 'London';
  if (hourET >= 9 && hourET < 15) return 'NY';
  return 'PM';
}

interface Level {
  price: number;
  setAt: number;  // unix sec of the bar where the level was established
}

interface SessionLevels {
  currentSession: SessionId | null;
  prevSession: SessionId | null;
  prevHigh: Level | null;
  prevLow: Level | null;
  q1High: Level | null;
  q1Low: Level | null;
  trueOpen: Level | null;
  dailyOpen: Level | null;
  midOrbHigh: Level | null;
  midOrbLow: Level | null;
  mornOrbHigh: Level | null;
  mornOrbLow: Level | null;
  sessionAnchor: number | null;
}

function computeSessionLevels(bars: FilterlessOhlcBar[]): SessionLevels {
  const result: SessionLevels = {
    currentSession: null, prevSession: null,
    prevHigh: null, prevLow: null,
    q1High: null, q1Low: null,
    trueOpen: null, dailyOpen: null,
    midOrbHigh: null, midOrbLow: null,
    mornOrbHigh: null, mornOrbLow: null,
    sessionAnchor: null,
  };
  if (bars.length === 0) return result;

  let curSession: SessionId | null = null;
  // Running session H/L tracking
  let sessHigh = -Infinity, sessHighAt = 0;
  let sessLow = Infinity, sessLowAt = 0;
  let q1Active = true;
  let q1High = -Infinity, q1HighAt = 0;
  let q1Low = Infinity, q1LowAt = 0;
  let toCaptured = false;
  let trueOpen: Level | null = null;
  // Levels carried forward
  let prevSession: SessionId | null = null;
  let prevHigh: Level | null = null;
  let prevLow: Level | null = null;
  let dailyOpen: Level | null = null;
  let midOrbHigh: Level | null = null;
  let midOrbLow: Level | null = null;
  let mornOrbHigh: Level | null = null;
  let mornOrbLow: Level | null = null;
  let anchor: number | null = null;

  for (const bar of bars) {
    const unix = Math.floor(new Date(bar.t).getTime() / 1000);
    if (!Number.isFinite(unix) || unix <= 0) continue;
    const { hour, minute } = etPartsFor(unix);
    const sess = sessionForET(hour);

    // Session transition
    if (sess !== curSession) {
      if (curSession !== null) {
        prevSession = curSession;
        prevHigh = sessHigh === -Infinity ? null : { price: sessHigh, setAt: sessHighAt };
        prevLow = sessLow === Infinity ? null : { price: sessLow, setAt: sessLowAt };
      }
      curSession = sess;
      sessHigh = bar.h; sessHighAt = unix;
      sessLow = bar.l;  sessLowAt = unix;
      q1Active = true;
      q1High = bar.h; q1HighAt = unix;
      q1Low = bar.l;  q1LowAt = unix;
      trueOpen = null;
      toCaptured = false;
      anchor = bar.o;  // session open price
    } else {
      if (bar.h > sessHigh) { sessHigh = bar.h; sessHighAt = unix; }
      if (bar.l < sessLow)  { sessLow = bar.l;  sessLowAt = unix; }
    }

    // Q1 tracking (until True Open bar fires)
    if (q1Active) {
      if (bar.h > q1High) { q1High = bar.h; q1HighAt = unix; }
      if (bar.l < q1Low)  { q1Low = bar.l;  q1LowAt = unix; }
    }

    // True Open bar (90 min into session)
    const isToBar =
      (sess === 'Asia' && hour === 22 && minute === 30) ||
      (sess === 'London' && hour === 4 && minute === 30) ||
      (sess === 'NY' && hour === 10 && minute === 30) ||
      (sess === 'PM' && hour === 16 && minute === 30);
    if (isToBar && !toCaptured) {
      q1Active = false;
      toCaptured = true;
      trueOpen = { price: bar.o, setAt: unix };
    }

    // Daily Open: 18:00 ET (futures session reopen)
    if (hour === 18 && minute === 0) {
      dailyOpen = { price: bar.o, setAt: unix };
    }

    // Midnight ORB: 03:00 - 03:30 ET (window start at exactly 03:00)
    if (hour === 3 && minute === 0) {
      midOrbHigh = { price: bar.h, setAt: unix };
      midOrbLow = { price: bar.l, setAt: unix };
    } else if (hour === 3 && minute < 30 && midOrbHigh !== null && midOrbLow !== null) {
      if (bar.h > midOrbHigh.price) midOrbHigh = { price: bar.h, setAt: unix };
      if (bar.l < midOrbLow.price)  midOrbLow = { price: bar.l, setAt: unix };
    }

    // Morning ORB: 09:30 - 10:00 ET
    if (hour === 9 && minute === 30) {
      mornOrbHigh = { price: bar.h, setAt: unix };
      mornOrbLow = { price: bar.l, setAt: unix };
    } else if (hour === 9 && minute > 30 && mornOrbHigh !== null && mornOrbLow !== null) {
      if (bar.h > mornOrbHigh.price) mornOrbHigh = { price: bar.h, setAt: unix };
      if (bar.l < mornOrbLow.price)  mornOrbLow = { price: bar.l, setAt: unix };
    }
  }

  result.currentSession = curSession;
  result.prevSession = prevSession;
  result.prevHigh = prevHigh;
  result.prevLow = prevLow;
  result.q1High = q1High === -Infinity ? null : { price: q1High, setAt: q1HighAt };
  result.q1Low = q1Low === Infinity ? null : { price: q1Low, setAt: q1LowAt };
  result.trueOpen = trueOpen;
  result.dailyOpen = dailyOpen;
  result.midOrbHigh = midOrbHigh;
  result.midOrbLow = midOrbLow;
  result.mornOrbHigh = mornOrbHigh;
  result.mornOrbLow = mornOrbLow;
  result.sessionAnchor = anchor;
  return result;
}

// ET-formatted tick + crosshair time labels for Lightweight Charts.
// Lightweight Charts passes UTCTimestamp (seconds since epoch).
function formatETTime(unixSec: number, includeSeconds: boolean): string {
  const d = new Date(unixSec * 1000);
  return new Intl.DateTimeFormat('en-US', {
    timeZone: 'America/New_York',
    hour: '2-digit', minute: '2-digit',
    second: includeSeconds ? '2-digit' : undefined,
    hour12: false,
  }).format(d);
}

function formatETTickMark(unixSec: number): string {
  const d = new Date(unixSec * 1000);
  const parts = new Intl.DateTimeFormat('en-US', {
    timeZone: 'America/New_York',
    month: 'short', day: '2-digit',
    hour: '2-digit', minute: '2-digit', hour12: false,
  }).formatToParts(d);
  const month = parts.find((p) => p.type === 'month')?.value ?? '';
  const day = parts.find((p) => p.type === 'day')?.value ?? '';
  const hh = parts.find((p) => p.type === 'hour')?.value ?? '00';
  const mm = parts.find((p) => p.type === 'minute')?.value ?? '00';
  // If midnight ET, show date label; otherwise show HH:MM ET
  if (hh === '00' && mm === '00') {
    return `${month} ${day}`;
  }
  return `${hh}:${mm}`;
}

const PriceCanvas: React.FC<{ state: FilterlessLiveState; position: FilterlessPosition | null }> = ({ state, position }) => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<IChartApi | null>(null);
  // Two series — only one is active at a time depending on whether OHLC is available.
  const candleSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const areaSeriesRef = useRef<ISeriesApi<'Area'> | null>(null);
  const priceLinesRef = useRef<{ series: ISeriesApi<'Candlestick'> | ISeriesApi<'Area'>; line: IPriceLine }[]>([]);
  // fitContent() snaps to the full data range — we only want that ONCE, on
  // first non-empty load of each series mode. After that, leave the user's
  // scroll/zoom alone. New bars still stream in via setData(); the time
  // window just doesn't reset under their finger.
  const candleFitDoneRef = useRef(false);
  const areaFitDoneRef = useRef(false);
  // Pine-port overlay split into two refs:
  //   overlayLineSeriesRef: bar-anchored level lines (session H/L, Q1, TO,
  //     Daily Open, ORBs, midpoints). Each is its own LineSeries so the
  //     line can START at the candle where the level was set and extend
  //     rightward — matching the original Pine behavior.
  //   overlayBankLinesRef: bank-level horizontals ($12.50 increments) drawn
  //     as full-width createPriceLine calls — Pine uses extend.both for
  //     these. Red above price, green below price, no labels.
  const overlayLineSeriesRef = useRef<ISeriesApi<'Line'>[]>([]);
  const overlayBankLinesRef = useRef<{ series: ISeriesApi<'Candlestick'> | ISeriesApi<'Area'>; line: IPriceLine }[]>([]);
  // Inline DOM-overlay labels for each session level — positioned at the
  // RIGHT end of each line via timeToCoordinate/priceToCoordinate so they
  // float inline with the candles instead of pinning to the right axis.
  const [overlayLabels, setOverlayLabels] = useState<Array<{
    key: string;
    time: UTCTimestamp;
    price: number;
    color: string;
    bg: string;
    text: string;
  }>>([]);
  // DOM refs to each label element so the RAF loop can update positions
  // directly without going through React's render cycle (eliminates lag
  // during pan/zoom — React state updates were too slow).
  const labelDomRefs = useRef<Array<HTMLDivElement | null>>([]);
  // Live in-progress minute bar — Topstep's REST history endpoint only
  // returns closed bars, so without this the chart would always be one
  // minute behind. We synthesize the current minute's OHLC from
  // state.bot.price ticks and append it to the chart series.
  const liveBarRef = useRef<{ minute: number; open: number; high: number; low: number; close: number } | null>(null);

  // Stable references to incoming data so dependency arrays compare cleanly.
  const ohlcBars: FilterlessOhlcBar[] = (state.bot.price_history_ohlc as FilterlessOhlcBar[] | undefined | null) || [];
  const hasOhlc = ohlcBars.length > 0;

  // Mount/unmount: create chart + both possible series. We swap visibility via setData([]) on the inactive one.
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const chart = createChart(container, {
      layout: {
        // Fully transparent so the .chart div's frosted-grey glass +
        // bg video show through behind the candles.
        background: { type: ColorType.Solid, color: 'rgba(0, 0, 0, 0)' },
        textColor: COLORS.muted,
        fontFamily: '"Arial", sans-serif',
        fontSize: 11,
      },
      grid: {
        vertLines: { color: 'rgba(255,255,255,0.04)' },
        horzLines: { color: 'rgba(255, 255, 255, 0.11)' },
      },
      rightPriceScale: {
        borderColor: 'rgba(255,255,255,0.10)',
        scaleMargins: { top: 0.1, bottom: 0.08 },
      },
      // All time labels rendered in America/New_York (ET) regardless of viewer's
      // local timezone. Crosshair = HH:MM ET; tick marks = HH:MM ET (or "Mon DD"
      // at midnight ET).
      timeScale: {
        borderColor: 'rgba(255,255,255,0.10)',
        timeVisible: true,
        secondsVisible: false,  // 1-min bars; minute precision is enough
        tickMarkFormatter: (time: UTCTimestamp) => formatETTickMark(time as number),
      },
      localization: {
        timeFormatter: (time: UTCTimestamp) => formatETTime(time as number, false) + ' ET',
      },
      crosshair: {
        mode: 1,
        vertLine: { color: 'rgba(255,255,255,0.22)', width: 1, style: 0 },
        horzLine: { color: 'rgba(255,255,255,0.22)', width: 1, style: 0 },
      },
      handleScroll: true,
      handleScale: true,
      autoSize: true,
    });
    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor: COLORS.green,
      downColor: COLORS.red,
      borderUpColor: COLORS.green,
      borderDownColor: COLORS.red,
      wickUpColor: COLORS.green,
      wickDownColor: COLORS.red,
      priceLineVisible: false,
      lastValueVisible: true,
    });
    const areaSeries = chart.addSeries(AreaSeries, {
      topColor: 'rgba(255,255,255,0.18)',
      bottomColor: 'rgba(255, 255, 255, 0.02)',
      lineColor: COLORS.purple,
      lineWidth: 2,
      priceLineVisible: false,
      lastValueVisible: true,
      crosshairMarkerVisible: true,
      crosshairMarkerRadius: 4,
      crosshairMarkerBackgroundColor: COLORS.cyan,
      crosshairMarkerBorderColor: COLORS.purple,
    });
    chartRef.current = chart;
    candleSeriesRef.current = candleSeries;
    areaSeriesRef.current = areaSeries;

    return () => {
      chart.remove();
      chartRef.current = null;
      candleSeriesRef.current = null;
      areaSeriesRef.current = null;
      priceLinesRef.current = [];
      overlayLineSeriesRef.current = [];
      overlayBankLinesRef.current = [];
    };
  }, []);

  // Continuous RAF loop: updates inline label positions directly via DOM
  // transform writes. Bypasses React's render cycle entirely so labels
  // track the chart with zero perceptible lag during pan/zoom.
  // Re-armed whenever the label set changes (data refresh ~ every 3s).
  // Labels are CLIPPED to the chart's plot area — when their pixel coords
  // fall outside the bounds (price out of range, time scrolled off-screen)
  // they're hidden so they don't escape into the rest of the page.
  useEffect(() => {
    let rafId = 0;
    const tick = () => {
      const chart = chartRef.current;
      const container = containerRef.current;
      const series: ISeriesApi<'Candlestick'> | ISeriesApi<'Area'> | null =
        chart && container
          ? (hasOhlc ? candleSeriesRef.current : areaSeriesRef.current)
          : null;
      if (chart && series && container) {
        const ts = chart.timeScale();
        // Plot area = container size minus axes. Subtract a small margin
        // so labels don't bleed visibly past the price/time scale borders.
        const W = container.clientWidth;
        const H = container.clientHeight;
        let priceAxisW = 0;
        let timeAxisH = 0;
        try { priceAxisW = chart.priceScale('right').width(); } catch { /* api drift */ }
        try { timeAxisH = ts.height(); } catch { /* api drift */ }
        const plotRight = W - priceAxisW - 4;
        const plotBottom = H - timeAxisH - 4;
        for (let i = 0; i < overlayLabels.length; i += 1) {
          const lbl = overlayLabels[i];
          const el = labelDomRefs.current[i];
          if (!el) continue;
          const x = ts.timeToCoordinate(lbl.time);
          const y = (series as ISeriesApi<'Line'>).priceToCoordinate(lbl.price);
          // Hide if coord null OR outside plot area. Use a small padding
          // so a label that's barely peeking off-screen is hidden cleanly.
          if (
            x == null || y == null ||
            x < 4 || x > plotRight ||
            y < 4 || y > plotBottom
          ) {
            el.style.visibility = 'hidden';
            continue;
          }
          el.style.visibility = 'visible';
          // translate3d engages GPU compositing; avoids layout reflow.
          el.style.transform = `translate3d(${x + 2}px, ${y - 7}px, 0)`;
        }
      }
      rafId = requestAnimationFrame(tick);
    };
    rafId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafId);
  }, [overlayLabels, hasOhlc]);

  // Feed OHLC candles when available; otherwise feed area-line close prices.
  useEffect(() => {
    const candle = candleSeriesRef.current;
    const area = areaSeriesRef.current;
    if (!candle || !area) return;

    if (hasOhlc) {
      const points = ohlcBars
        .map((bar) => {
          if (!bar || bar.o == null || bar.h == null || bar.l == null || bar.c == null) return null;
          const ts = Math.floor(new Date(bar.t).getTime() / 1000);
          if (!Number.isFinite(ts) || ts <= 0) return null;
          return { time: ts as UTCTimestamp, open: bar.o, high: bar.h, low: bar.l, close: bar.c };
        })
        .filter((p): p is { time: UTCTimestamp; open: number; high: number; low: number; close: number } => p !== null)
        .sort((a, b) => a.time - b.time);
      const dedup: typeof points = [];
      for (const p of points) {
        if (dedup.length > 0 && dedup[dedup.length - 1].time === p.time) {
          dedup[dedup.length - 1] = p;
        } else {
          dedup.push(p);
        }
      }
      // Append a synthetic in-progress bar for the current minute so the
      // chart isn't 1-min behind the live tape. Topstep's REST history
      // endpoint only returns CLOSED bars; without this synthesis the most
      // recent candle would be the previous minute. The bar advances H/L
      // as state.bot.price ticks arrive between dashboard polls.
      const livePrice = state.bot.price;
      if (livePrice != null && Number.isFinite(livePrice)) {
        const nowMin = Math.floor(Date.now() / 1000 / 60) * 60;
        const lastHistTime = dedup.length > 0 ? (dedup[dedup.length - 1].time as number) : 0;
        if (nowMin > lastHistTime) {
          // New minute or no history yet — start (or update) the live bar
          if (liveBarRef.current == null || liveBarRef.current.minute !== nowMin) {
            liveBarRef.current = { minute: nowMin, open: livePrice, high: livePrice, low: livePrice, close: livePrice };
          } else {
            liveBarRef.current.close = livePrice;
            if (livePrice > liveBarRef.current.high) liveBarRef.current.high = livePrice;
            if (livePrice < liveBarRef.current.low) liveBarRef.current.low = livePrice;
          }
          dedup.push({
            time: liveBarRef.current.minute as UTCTimestamp,
            open: liveBarRef.current.open,
            high: liveBarRef.current.high,
            low: liveBarRef.current.low,
            close: liveBarRef.current.close,
          });
        } else if (nowMin === lastHistTime) {
          // Last historical bar IS the current minute (rare, only if API
          // ever returns the in-progress bar). Update its close + extend
          // H/L from live ticks; clear the synthetic ref since the real
          // bar has taken over.
          liveBarRef.current = null;
          const last = dedup[dedup.length - 1];
          if (livePrice > last.high) last.high = livePrice;
          if (livePrice < last.low) last.low = livePrice;
          last.close = livePrice;
        }
      }
      candle.setData(dedup);
      area.setData([]);  // hide area when candles are live
      if (dedup.length > 0 && !candleFitDoneRef.current) {
        chartRef.current?.timeScale().fitContent();
        candleFitDoneRef.current = true;
      }
    } else {
      // Fallback to close-only line via AreaSeries
      const points = state.bot.price_history
        .map((point) => {
          if (point.price == null || Number.isNaN(point.price)) return null;
          const ts = Math.floor(new Date(point.time).getTime() / 1000);
          if (!Number.isFinite(ts) || ts <= 0) return null;
          return { time: ts as UTCTimestamp, value: point.price as number };
        })
        .filter((point): point is { time: UTCTimestamp; value: number } => point !== null)
        .sort((a, b) => a.time - b.time);
      const dedup: typeof points = [];
      for (const p of points) {
        if (dedup.length > 0 && dedup[dedup.length - 1].time === p.time) {
          dedup[dedup.length - 1] = p;
        } else {
          dedup.push(p);
        }
      }
      area.setData(dedup);
      candle.setData([]);
      if (dedup.length > 0 && !areaFitDoneRef.current) {
        chartRef.current?.timeScale().fitContent();
        areaFitDoneRef.current = true;
      }
    }
  }, [hasOhlc, ohlcBars, state.bot.price_history, state.bot.price]);

  // Refresh the STOP / ENTRY / TP horizontal lines whenever the position OR active series changes.
  useEffect(() => {
    const series: ISeriesApi<'Candlestick'> | ISeriesApi<'Area'> | null =
      hasOhlc ? candleSeriesRef.current : areaSeriesRef.current;
    if (!series) return;
    for (const entry of priceLinesRef.current) {
      try { entry.series.removePriceLine(entry.line); } catch { /* chart may be torn down */ }
    }
    priceLinesRef.current = [];
    const stop = positionStop(position);
    const entry = getPositionEntryPrice(position);
    const target = positionTarget(position);
    const overlays: Array<{ price: number | null; color: string; title: string }> = [
      { price: stop, color: COLORS.red, title: 'STOP' },
      { price: entry, color: COLORS.cyan, title: 'ENTRY' },
      { price: target, color: COLORS.green, title: 'TP' },
    ];
    for (const overlay of overlays) {
      if (overlay.price == null || Number.isNaN(overlay.price)) continue;
      const line = series.createPriceLine({
        price: overlay.price,
        color: overlay.color,
        lineWidth: 1,
        lineStyle: LineStyle.Dashed,
        axisLabelVisible: true,
        title: overlay.title,
      });
      priceLinesRef.current.push({ series, line });
    }
  }, [position, hasOhlc]);

  // Pine-script-port overlay.
  // (1) Session-anchored level lines (PrevH/L, Q1, TO, DO, ORBs, midpoints):
  //     each is its own LineSeries with two data points — start at the
  //     candle where the level was set, end ~12 bars past current. Only
  //     ONE of each line type exists at any time; old ones get removed
  //     before drawing new ones (matches Pine's line.delete/line.new
  //     refresh on every session boundary).
  // (2) Bank levels: createPriceLine for full-width horizontals. Red if
  //     above current price, green if below. NO labels (axisLabelVisible
  //     false, title empty) per user spec.
  useEffect(() => {
    const chart = chartRef.current;
    const overlaySeries: ISeriesApi<'Candlestick'> | ISeriesApi<'Area'> | null =
      hasOhlc ? candleSeriesRef.current : areaSeriesRef.current;
    if (!chart || !overlaySeries) return;

    // Clear all old overlay artifacts.
    for (const s of overlayLineSeriesRef.current) {
      try { chart.removeSeries(s); } catch { /* chart torn down */ }
    }
    overlayLineSeriesRef.current = [];
    for (const entry of overlayBankLinesRef.current) {
      try { entry.series.removePriceLine(entry.line); } catch { /* chart torn down */ }
    }
    overlayBankLinesRef.current = [];

    if (!hasOhlc || ohlcBars.length === 0) return;

    const levels = computeSessionLevels(ohlcBars);
    const lastBarUnix = Math.floor(new Date(ohlcBars[ohlcBars.length - 1].t).getTime() / 1000);
    if (!Number.isFinite(lastBarUnix) || lastBarUnix <= 0) return;
    // Extend lines ~12 bars past current to mimic Pine's `bar_index + offset`.
    // 1-min bars => 12 minutes ahead.
    const extendTo = (lastBarUnix + 12 * 60) as UTCTimestamp;

    // Collect label data alongside drawing — rendered as DOM overlays
    // anchored to the line's right endpoint (at extendTo time).
    const labels: Array<{ key: string; time: UTCTimestamp; price: number; color: string; bg: string; text: string }> = [];

    const drawAnchored = (
      level: Level | null,
      color: string,
      title: string,
      style: LineStyle = LineStyle.Solid,
      width: 1 | 2 = 1,
      labelBg: string = 'rgba(67, 70, 81, 0.85)',
    ): void => {
      if (level == null || !Number.isFinite(level.price)) return;
      const startTime = level.setAt as UTCTimestamp;
      const safeStart = (level.setAt > lastBarUnix ? lastBarUnix : level.setAt) as UTCTimestamp;
      try {
        const lineSeries = chart.addSeries(LineSeries, {
          color,
          lineWidth: width,
          lineStyle: style,
          // No right-axis chip — we render an inline DOM label at the line's
          // right tip instead, matching the TradingView Pine-script visual.
          lastValueVisible: false,
          priceLineVisible: false,
          crosshairMarkerVisible: false,
        });
        lineSeries.setData([
          { time: startTime <= extendTo ? startTime : safeStart, value: level.price },
          { time: extendTo, value: level.price },
        ]);
        overlayLineSeriesRef.current.push(lineSeries);
      } catch { /* chart may be torn down */ }
      labels.push({
        key: `${title}-${level.setAt}-${level.price}`,
        time: extendTo,
        price: level.price,
        color,
        bg: labelBg,
        // Title only — price is already shown on the right y-axis,
        // no need to duplicate it here.
        text: title,
      });
    };

    const drawAnchoredMidpoint = (
      hi: Level | null,
      lo: Level | null,
      color: string,
      title: string,
    ): void => {
      if (hi == null || lo == null) return;
      const mid = (hi.price + lo.price) / 2;
      // "Lock onto the candlestick on that midpoint": walk bars backwards
      // from current and find the most recent bar whose [low, high] range
      // contains the midpoint price. We walk the FULL bar history (no
      // setAt cutoff) — for the prev-session midpoint specifically, the
      // bars between when high was set and when low was set ARE in the
      // previous session and may be the only ones whose range contains
      // the midpoint price. By intermediate-value-theorem logic, a session
      // whose extremes are H and L must have at least one bar straddling
      // (H+L)/2. Falls back to the later of (hi.setAt, lo.setAt) only if
      // no bar in the entire history touches the midpoint (shouldn't
      // happen in practice).
      let setAt = Math.max(hi.setAt, lo.setAt);
      for (let i = ohlcBars.length - 1; i >= 0; i -= 1) {
        const bar = ohlcBars[i];
        const barUnix = Math.floor(new Date(bar.t).getTime() / 1000);
        if (!Number.isFinite(barUnix)) continue;
        if (bar.l <= mid && mid <= bar.h) {
          setAt = barUnix;
          break;
        }
      }
      drawAnchored({ price: mid, setAt }, color, title, LineStyle.Dotted, 1, 'rgba(20, 12, 30, 0.65)');
    };

    const sessName = levels.currentSession ?? 'SESS';
    const prevName = levels.prevSession ?? 'PREV';

    // Previous session H/L + midpoint
    drawAnchored(levels.prevHigh, '#ffffff', `${prevName} High`, LineStyle.Solid, 1);
    drawAnchored(levels.prevLow, '#ffffff', `${prevName} Low`, LineStyle.Solid, 1);
    drawAnchoredMidpoint(levels.prevHigh, levels.prevLow, COLORS.muted, 'Prev Session Midpoint');

    // Q1 H/L + midpoint
    drawAnchored(levels.q1High, '#ffffff', `Q1H-${sessName}`, LineStyle.Dashed, 1);
    drawAnchored(levels.q1Low, '#ffffff', `Q1L-${sessName}`, LineStyle.Dashed, 1);
    drawAnchoredMidpoint(levels.q1High, levels.q1Low, COLORS.muted, 'Q1 Midpoint');

    // True Open
    drawAnchored(levels.trueOpen, '#ffffff', `TO-${sessName}`, LineStyle.Solid, 1);

    // Daily Open
    drawAnchored(levels.dailyOpen, '#ffffff', 'Daily Open', LineStyle.Solid, 2);

    // Midnight ORB
    drawAnchored(levels.midOrbHigh, COLORS.red, 'Mid ORB High', LineStyle.Solid, 1);
    drawAnchored(levels.midOrbLow, COLORS.green, 'Mid ORB Low', LineStyle.Solid, 1);
    drawAnchoredMidpoint(levels.midOrbHigh, levels.midOrbLow, COLORS.muted, 'Midnight ORB Midpoint');

    // Morning ORB
    drawAnchored(levels.mornOrbHigh, COLORS.red, 'Morn ORB High', LineStyle.Solid, 1);
    drawAnchored(levels.mornOrbLow, COLORS.green, 'Morn ORB Low', LineStyle.Solid, 1);
    drawAnchoredMidpoint(levels.mornOrbHigh, levels.mornOrbLow, COLORS.muted, 'Morn ORB Midpoint');

    // Bank levels: $12.50 increments, full-width horizontals.
    // Red if level is above price (resistance), green if below (support).
    // No labels, no axis labels — pure visual reference grid.
    const lastClose = ohlcBars[ohlcBars.length - 1].c;
    const STEP = 12.5;
    const ABS_RANGE = 5;   // ±5 absolute bank levels around current price
    const REL_RANGE = 5;   // ±5 relative bank levels anchored to session open
    const drawBank = (price: number): void => {
      if (!Number.isFinite(price)) return;
      const isAbove = price > lastClose;
      const color = isAbove
        ? 'rgba(255, 56, 100, 0.40)'   // red — resistance above
        : 'rgba(69, 255, 200, 0.40)';  // green — support below
      try {
        const line = overlaySeries.createPriceLine({
          price,
          color,
          lineWidth: 1,
          lineStyle: LineStyle.Dashed,
          axisLabelVisible: false,
          title: '',
        });
        overlayBankLinesRef.current.push({ series: overlaySeries, line });
      } catch { /* chart may be torn down */ }
    };
    if (Number.isFinite(lastClose)) {
      const baseAbs = Math.floor(lastClose / STEP) * STEP;
      for (let i = -ABS_RANGE; i <= ABS_RANGE; i += 1) {
        const lvl = baseAbs + i * STEP;
        if (Math.abs(lvl - lastClose) < 0.01) continue;
        drawBank(lvl);
      }
    }
    if (levels.sessionAnchor != null && Number.isFinite(levels.sessionAnchor)) {
      for (let i = -REL_RANGE; i <= REL_RANGE; i += 1) {
        if (i === 0) continue;
        drawBank(levels.sessionAnchor + i * STEP);
      }
    }

    setOverlayLabels(labels);
  }, [hasOhlc, ohlcBars]);

  const hasAnyData = hasOhlc || state.bot.price_history.some((point) => point.price != null && !Number.isNaN(point.price));

  // Trim the labelDomRefs array to match current overlayLabels count so
  // stale refs from previous renders don't accumulate or cause null reads.
  if (labelDomRefs.current.length > overlayLabels.length) {
    labelDomRefs.current.length = overlayLabels.length;
  }

  return (
    <div className="chart" style={{ position: 'relative' }}>
      <div ref={containerRef} style={{ position: 'absolute', inset: 0 }} />
      {!hasAnyData ? (
        <div style={{
          position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center',
          color: COLORS.muted, fontFamily: '"Arial", sans-serif', fontSize: 13, fontStyle: 'italic',
          pointerEvents: 'none',
        }}>
          waiting for live price history
        </div>
      ) : null}
      {hasOhlc ? (
        <div style={{
          position: 'absolute', top: 8, left: 12,
          color: COLORS.muted, fontFamily: '"Arial", sans-serif', fontSize: 11,
          letterSpacing: 1, opacity: 0.6, pointerEvents: 'none', fontStyle: 'italic',
        }}>
          1m · ohlc
        </div>
      ) : null}
      {/* Inline TradingView-style level labels. Position is updated every
          frame by the RAF loop above (see useEffect on overlayLabels) — NOT
          by React state, so there's zero pan/zoom lag. Initial transform
          parks them off-screen until the first RAF tick lands them. */}
      {overlayLabels.map((lbl, i) => (
        <div
          key={lbl.key}
          className="ov-label"
          ref={(el) => { labelDomRefs.current[i] = el; }}
          style={{
            position: 'absolute',
            left: 0,
            top: 0,
            transform: 'translate3d(-9999px, -9999px, 0)',
            color: lbl.color,
            background: lbl.bg,
            padding: '1px 4px',
            borderRadius: 2,
            fontFamily: '"Arial", sans-serif',
            fontSize: 10,
            lineHeight: '14px',
            whiteSpace: 'nowrap',
            pointerEvents: 'none',
            zIndex: 2,
            opacity: 0.92,
            willChange: 'transform',
          }}
        >
          {lbl.text}
        </div>
      ))}
    </div>
  );
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
    // Transparent canvas so parent panel's frosted-grey glass + video bg
    // show through. clearRect alone is enough.
    ctx.clearRect(0, 0, width, height);
    const mid = height * 0.52;
    ctx.strokeStyle = 'rgba(255,255,255,0.10)';
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
    ctx.font = `${10.0 * dpr}px "Arial", sans-serif`;
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
    let frameId = 0;
    let lastFrameTime = 0;
    const frameIntervalMs = 1000 / MANIFOLD_IDLE_FPS;
    const reduceMotion = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false;

    const drawFrame = (frameTime: number) => {
      const setup = setupCanvas(canvas);
      if (!setup) return;
      const { ctx, width, height, dpr } = setup;
      const timeSeconds = reduceMotion ? 0 : frameTime / 1000;
      const waveStrength = reduceMotion ? 0 : (dragRef.current.dragging ? 0.36 : 1);
      // clearRect alone leaves the canvas transparent so the parent
      // .scene-wrap's frosted-grey glass + the page video bg show through.
      // Previously this filled solid #030006 black, blocking the bg.
      ctx.clearRect(0, 0, width, height);

      const drawLabel = (text: string, point: { x: number; y: number }, color: string, align: CanvasTextAlign = 'center') => {
        ctx.save();
        ctx.font = `italic ${12.0 * dpr}px "Arial", sans-serif`;
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
      const points: Array<Array<{ x: number; y: number; z: number; baseZ: number; wave: number; p: ReturnType<typeof projectSurfacePoint>; dominant: string }>> = [];
      for (let row = 0; row <= rows; row += 1) {
        const line = [];
        const y = row / rows;
        for (let col = 0; col <= cols; col += 1) {
          const x = -1 + (2 * col) / cols;
          const baseZ = pressurePlaneHeight(x, y, features);
          const wave = manifoldIdleWave(x, y, features, timeSeconds) * waveStrength;
          const z = clip(baseZ + wave, -0.2, 1.16);
          line.push({
            x,
            y,
            z,
            baseZ,
            wave,
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
          const heightAvg = (a.baseZ + b.baseZ + c.baseZ + d.baseZ) / 4;
          const waveAvg = (a.wave + b.wave + c.wave + d.wave) / 4;
          faces.push({
            pts: [a, b, c, d],
            depth: (a.p.depth + b.p.depth + c.p.depth + d.p.depth) / 4,
            height: heightAvg,
            wave: waveAvg,
            dominant: [a, b, c, d].sort((p1, p2) => p2.baseZ - p1.baseZ)[0].dominant,
          });
        }
      }

      faces.sort((a, b) => a.depth - b.depth);
      faces.forEach((face) => {
        const v10Height = face.height / 0.54;
        const center = face.pts.reduce((acc, point) => ({ x: acc.x + point.x, y: acc.y + point.y }), { x: 0, y: 0 });
        const colorPulse = idleColorPulse(center.x / 4, center.y / 4, face.wave, features, timeSeconds) * waveStrength;
        const shade = clip(0.42 + ((v10Height + 0.28) * 0.54) + (((face.depth + 2) / 8) * 0.22) + colorPulse, 0.22, 1.28);
        const fillColor = shiftedManifoldColor(face.dominant, center.x / 4, center.y / 4, face.wave, features, timeSeconds, waveStrength);
        ctx.beginPath();
        face.pts.forEach((point, index) => {
          if (index === 0) ctx.moveTo(point.p.x, point.p.y);
          else ctx.lineTo(point.p.x, point.p.y);
        });
        ctx.closePath();
        ctx.fillStyle = shadedColor(fillColor, shade, 0.7 + (0.06 * waveStrength));
        ctx.fill();
        ctx.strokeStyle = v10Height > 0.78 ? 'rgba(255,255,255,0.16)' : `rgba(255, 255, 255,${0.12 + (0.05 * waveStrength)})`;
        ctx.lineWidth = 0.68 * dpr;
        ctx.stroke();
      });

      ctx.save();
      ctx.lineWidth = 1.1 * dpr;
      [0.16, 0.28, 0.4, 0.52, 0.64, 0.76, 0.88].forEach((y, index) => {
        ctx.beginPath();
        for (let step = 0; step <= 144; step += 1) {
          const x = -0.98 + (1.96 * step) / 144;
          const z = clip(pressurePlaneHeight(x, y, features) + (manifoldIdleWave(x, y, features, timeSeconds) * waveStrength) + 0.014, -0.2, 1.18);
          const p = projectSurfacePoint(x, y, z, width, height, scene);
          if (step === 0) ctx.moveTo(p.x, p.y);
          else ctx.lineTo(p.x, p.y);
        }
        ctx.strokeStyle = index % 2 ? `rgba(255, 255, 255,${0.24 + (0.1 * waveStrength)})` : `rgba(53,245,255,${0.21 + (0.08 * waveStrength)})`;
        ctx.stroke();
      });
      ctx.restore();

      const origin = projectSurfacePoint(-1.08, 0, 0, width, height, scene);
      const farRight = projectSurfacePoint(1.1, 0, 0, width, height, scene);
      const nearRight = projectSurfacePoint(1.1, 1.08, 0, width, height, scene);
      const nearLeft = projectSurfacePoint(-1.08, 1.08, 0, width, height, scene);
      const axisZ = { x: origin.x, y: origin.y - height * 0.43 };
      strokePath([origin, farRight, nearRight, nearLeft, origin], 'rgba(255, 255, 255,0.42)', 1.5);
      strokePath([origin, axisZ], 'rgba(255, 255, 255,0.72)', 1.5);
      strokePath([origin, nearLeft], 'rgba(255, 255, 255,0.42)', 1.2);

      ctx.save();
      ctx.font = `italic ${12.0 * dpr}px "Arial", sans-serif`;
      ctx.fillStyle = alphaColor(COLORS.muted, 0.92);
      const pressureLabel = projectSurfacePoint(1.14, 0.66, 0.1, width, height, scene);
      const transitionLabelX = clip(nearLeft.x + 8 * dpr, 8 * dpr, width - 96 * dpr);
      ctx.fillText('INTENSITY', axisZ.x + 10 * dpr, axisZ.y + 8 * dpr);
      ctx.fillText('PRESSURE -1', pressureLabel.x + 8 * dpr, pressureLabel.y - 4 * dpr);
      ctx.fillText('TRANSITION', transitionLabelX, nearLeft.y + 18 * dpr);
      ctx.restore();

      const markerTarget = markerTargetForRegime(features);
      const markerZ = pressurePlaneHeight(markerTarget.x, markerTarget.y, features);
      const markerWave = manifoldIdleWave(markerTarget.x, markerTarget.y, features, timeSeconds) * waveStrength;
      const marker = projectSurfacePoint(markerTarget.x, markerTarget.y, markerZ + markerWave + 0.05, width, height, scene);
      const base = projectSurfacePoint(markerTarget.x, markerTarget.y, 0, width, height, scene);
      const activeColor = regimeColor(features.regime);
      ctx.save();
      ctx.strokeStyle = activeColor;
      ctx.lineWidth = 2 * dpr;
      ctx.shadowBlur = (14 + (4 * waveStrength * Math.sin(timeSeconds * 1.4))) * dpr;
      ctx.shadowColor = activeColor;
      ctx.beginPath();
      ctx.moveTo(base.x, base.y);
      ctx.lineTo(marker.x, marker.y);
      ctx.stroke();
      ctx.fillStyle = activeColor;
      ctx.beginPath();
      ctx.arc(marker.x, marker.y, (6 + (1.2 * waveStrength * (0.5 + (0.5 * Math.sin(timeSeconds * 1.8))))) * dpr, 0, TAU);
      ctx.fill();
      ctx.shadowBlur = 0;
      ctx.font = `italic ${13.0 * dpr}px "Arial", sans-serif`;
      ctx.textBaseline = 'middle';
      ctx.fillStyle = COLORS.text;
      ctx.fillText(features.regime, marker.x + 13 * dpr, marker.y - 8 * dpr);
      ctx.fillStyle = COLORS.muted;
      ctx.fillText(`pressure ${fmt(features.pressure30)} / burst ${fmt(features.burstPressure)}`, marker.x + 13 * dpr, marker.y + 9 * dpr);
      ctx.restore();

      drawLabel('trend ridge', projectSurfacePoint(features.pressure30, 0.78, 0.4 + (manifoldIdleWave(features.pressure30, 0.78, features, timeSeconds) * waveStrength), width, height, scene), COLORS.green);
      drawLabel('burst peak', projectSurfacePoint(Math.sign(features.pressure30 || 0.001) * 0.55, features.transitionEnergy, 0.58 + (manifoldIdleWave(Math.sign(features.pressure30 || 0.001) * 0.55, features.transitionEnergy, features, timeSeconds) * waveStrength), width, height, scene), COLORS.amber);
      drawLabel('chop shelf', projectSurfacePoint(0, 0.38, 0.44 + (manifoldIdleWave(0, 0.38, features, timeSeconds) * waveStrength), width, height, scene), COLORS.cyan);
      drawLabel('rot wall', projectSurfacePoint(-0.72, 0.58, 0.5 + (manifoldIdleWave(-0.72, 0.58, features, timeSeconds) * waveStrength), width, height, scene), COLORS.red);

      const cx = width - 58 * dpr;
      const cy = 48 * dpr;
      const rx = 28 * dpr;
      const ry = 15 * dpr;
      const dotX = cx + Math.sin(scene.yaw) * rx;
      const dotY = cy - scene.pitch * 38 * dpr;
      ctx.save();
      ctx.strokeStyle = 'rgba(255, 255, 255,0.36)';
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
    };

    const animate = (frameTime: number) => {
      if (frameTime - lastFrameTime >= frameIntervalMs) {
        lastFrameTime = frameTime;
        drawFrame(frameTime);
      }
      frameId = window.requestAnimationFrame(animate);
    };

    drawFrame(performance.now());
    if (!reduceMotion) {
      frameId = window.requestAnimationFrame(animate);
    }
    return () => {
      if (frameId) window.cancelAnimationFrame(frameId);
    };
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
      // Wider rotation envelope so you can spin the manifold around more
      // freely. Sensitivity (1.45 / 0.84 multipliers) kept the same so
      // dragging the same distance still moves about the same — only the
      // clip-stops are pushed further out.
      yaw: clip(dragRef.current.yaw - (dx / Math.max(1, rect.width)) * 1.45, -1.4, 1.4),
      pitch: clip(dragRef.current.pitch + (dy / Math.max(1, rect.height)) * 0.84, -0.7, 0.9),
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
    // Wider zoom range (was 0.72-1.42, now 0.45-2.6) for more pull-out
    // and push-in headroom. Per-tick scale (0.93/1.07 = ~7% per scroll
    // line) kept the same so the wheel still feels controlled, just
    // covers more ground over a longer scroll.
    setScene((current) => ({ ...current, zoom: clip(current.zoom * (event.deltaY > 0 ? 0.93 : 1.07), 0.45, 2.6) }));
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

/* ----------------------------------------------------------------------
   BouncingHeads — DVD-screensaver-style heads that bounce around the
   viewport, off the walls, and off each other. Click any head to detonate
   it into a fire explosion. Each head respawns 5 minutes after its own
   explosion and resumes bouncing. The loop is infinite, per-head.
---------------------------------------------------------------------- */
const HEAD_SRC = '/head.png';
const HEAD_SIZE = 90;
const HEAD_RESPAWN_MS = 5 * 60 * 1000;
const HEAD_COUNT = 4;

type HeadPhase = 'bouncing' | 'hidden';

const HEAD_PARTICLE_STEP = 3;

interface PixelParticle {
  x: number; y: number;
  vx: number; vy: number;
  size: number;
  color: string;
  life: number;
}

// Module-scope pixel cache: filled once when the head image first loads.
// Click → explode reads this synchronously so there's no async wait.
let HEAD_PIXEL_DATA: Uint8ClampedArray | null = null;
let HEAD_PRELOAD_PROMISE: Promise<void> | null = null;

function preloadHeadPixels(): Promise<void> {
  if (HEAD_PIXEL_DATA) return Promise.resolve();
  if (HEAD_PRELOAD_PROMISE) return HEAD_PRELOAD_PROMISE;
  HEAD_PRELOAD_PROMISE = new Promise<void>((resolve) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      try {
        const off = document.createElement('canvas');
        off.width = HEAD_SIZE;
        off.height = HEAD_SIZE;
        const octx = off.getContext('2d');
        if (!octx) { resolve(); return; }
        octx.drawImage(img, 0, 0, HEAD_SIZE, HEAD_SIZE);
        HEAD_PIXEL_DATA = octx.getImageData(0, 0, HEAD_SIZE, HEAD_SIZE).data;
      } catch { /* CORS — leave cache null */ }
      resolve();
    };
    img.onerror = () => resolve();
    img.src = HEAD_SRC;
  });
  return HEAD_PRELOAD_PROMISE;
}

/* Pixel-explosion: sample the head image into colored particles bursting
   outward from the head center. Skips near-black background pixels so the
   burst is only the face/hair. */
function spawnExplosionParticles(list: PixelParticle[], ox: number, oy: number): void {
  const data = HEAD_PIXEL_DATA;
  if (!data) return;
  const cx = HEAD_SIZE / 2;
  const cy = HEAD_SIZE / 2;
  const baseX = ox - HEAD_SIZE / 2;
  const baseY = oy - HEAD_SIZE / 2;
  for (let py = 0; py < HEAD_SIZE; py += HEAD_PARTICLE_STEP) {
    for (let px = 0; px < HEAD_SIZE; px += HEAD_PARTICLE_STEP) {
      const i = (py * HEAD_SIZE + px) * 4;
      const r = data[i];
      const g = data[i + 1];
      const b = data[i + 2];
      const a = data[i + 3];
      if (a < 30) continue;
      if (r < 22 && g < 22 && b < 22) continue; // skip black surround
      const dx = px - cx;
      const dy = py - cy;
      const dist = Math.sqrt(dx * dx + dy * dy) || 1;
      const speed = 3 + Math.random() * 5;
      const jitter = (Math.random() - 0.5) * 1.4;
      list.push({
        x: baseX + px,
        y: baseY + py,
        vx: (dx / dist) * speed + jitter,
        vy: (dy / dist) * speed + jitter - 1.5,
        size: HEAD_PARTICLE_STEP,
        color: `rgb(${r},${g},${b})`,
        life: 1,
      });
    }
  }
}

/* Each head moves independently with its own velocity. Heads bounce off
   viewport walls AND off each other (treated as equal-mass circles, with
   elastic collisions). One shared rAF drives all motion, integration,
   and collision resolution. Initial positions are pre-staggered so no two
   heads overlap on spawn. */
interface HeadMotion {
  x: number;
  y: number;
  vx: number;
  vy: number;
  angle: number;
  angularV: number;
}

function initialHeadMotion(idx: number): HeadMotion {
  const w = typeof window !== 'undefined' ? window.innerWidth : 1200;
  const h = typeof window !== 'undefined' ? window.innerHeight : 800;
  // Pre-stagger across the diagonal so no two heads start overlapping
  // even on a narrow phone viewport.
  const fracX = (idx + 1) / (HEAD_COUNT + 1);
  const fracY = idx % 2 === 0 ? 0.25 : 0.65;
  const dir = idx % 2 === 0 ? 1 : -1;
  const speed = 2.6 + idx * 0.4;
  const heading = (Math.PI / 4) + idx * 0.7;
  return {
    x: Math.max(20, fracX * w - HEAD_SIZE / 2),
    y: Math.max(20, fracY * h - HEAD_SIZE / 2),
    vx: Math.cos(heading) * speed * dir,
    vy: Math.sin(heading) * speed,
    angle: idx * 47,
    angularV: (idx % 2 === 0 ? 1 : -1) * (0.25 + (idx * 0.07)),
  };
}

const BouncingHead: React.FC = React.memo(() => {
  const [phases, setPhases] = useState<HeadPhase[]>(() => Array(HEAD_COUNT).fill('bouncing'));
  const [imgLoadFailed, setImgLoadFailed] = useState(false);

  const headRefs = useRef<Array<HTMLDivElement | null>>(Array(HEAD_COUNT).fill(null));
  const motionsRef = useRef<HeadMotion[]>(
    Array.from({ length: HEAD_COUNT }, (_, i) => initialHeadMotion(i)),
  );
  const phasesRef = useRef<HeadPhase[]>(phases);
  useEffect(() => { phasesRef.current = phases; }, [phases]);

  // Shared explosion canvas — all clicks dump particles into one list and
  // get rendered on one canvas. Prevents stacked canvases that previously
  // made multiple explosions appear on top of each other when heads were
  // clicked in quick succession.
  const explosionCanvasRef = useRef<HTMLCanvasElement>(null);
  const particlesRef = useRef<PixelParticle[]>([]);
  const fireRafRef = useRef<number | null>(null);

  useEffect(() => { void preloadHeadPixels(); }, []);

  // Size the explosion canvas once mounted and on resize.
  useEffect(() => {
    const setup = () => {
      const canvas = explosionCanvasRef.current;
      if (!canvas) return;
      const dpr = window.devicePixelRatio || 1;
      const w = window.innerWidth;
      const h = window.innerHeight;
      canvas.width = w * dpr;
      canvas.height = h * dpr;
      canvas.style.width = `${w}px`;
      canvas.style.height = `${h}px`;
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.scale(dpr, dpr);
      }
    };
    setup();
    window.addEventListener('resize', setup);
    return () => window.removeEventListener('resize', setup);
  }, []);

  // Drive the fire-explosion render loop only while particles are alive.
  // When all particles die, the rAF self-cancels — no idle work, no
  // residual frames lingering on the canvas. Re-armed on each click.
  const runFireLoop = useCallback(() => {
    if (fireRafRef.current !== null) return; // already running
    const canvas = explosionCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const w = window.innerWidth;
    const h = window.innerHeight;

    const tick = () => {
      const particles = particlesRef.current;
      if (particles.length === 0) {
        ctx.clearRect(0, 0, w, h);
        fireRafRef.current = null;
        return;
      }

      // Hard clear each frame — no motion blur. Pixel particles look right
      // as crisp dots; trails would just look smeared.
      ctx.clearRect(0, 0, w, h);

      for (const p of particles) {
        if (p.life <= 0) continue;
        p.x += p.vx;
        p.y += p.vy;
        p.vy += 0.18;
        p.vx *= 0.992;
        p.life -= 0.011;
        ctx.globalAlpha = Math.max(0, p.life);
        ctx.fillStyle = p.color;
        ctx.fillRect(p.x, p.y, p.size, p.size);
      }
      ctx.globalAlpha = 1;

      particlesRef.current = particles.filter((p) => p.life > 0);
      fireRafRef.current = requestAnimationFrame(tick);
    };
    fireRafRef.current = requestAnimationFrame(tick);
  }, []);

  // Single shared rAF: integrate motion, resolve wall + head-to-head
  // collisions, write each head's transform directly to its DOM node.
  useEffect(() => {
    let w = window.innerWidth;
    let h = window.innerHeight;
    const onResize = () => {
      w = window.innerWidth;
      h = window.innerHeight;
    };
    window.addEventListener('resize', onResize, { passive: true });

    const RADIUS = HEAD_SIZE / 2;
    const COLLISION_DIST = HEAD_SIZE; // sum of radii for two equal heads

    let raf = 0;
    const step = () => {
      const motions = motionsRef.current;
      const activePhases = phasesRef.current;

      // 1. Integrate position + wall bounces (only for active heads)
      for (let i = 0; i < HEAD_COUNT; i++) {
        if (activePhases[i] !== 'bouncing') continue;
        const m = motions[i];
        m.x += m.vx;
        m.y += m.vy;
        m.angle += m.angularV;
        if (m.x <= 0) { m.x = 0; m.vx = Math.abs(m.vx); m.angularV = -m.angularV; }
        if (m.x + HEAD_SIZE >= w) { m.x = w - HEAD_SIZE; m.vx = -Math.abs(m.vx); m.angularV = -m.angularV; }
        if (m.y <= 0) { m.y = 0; m.vy = Math.abs(m.vy); }
        if (m.y + HEAD_SIZE >= h) { m.y = h - HEAD_SIZE; m.vy = -Math.abs(m.vy); }
      }

      // 2. Pairwise elastic collisions between active heads
      for (let i = 0; i < HEAD_COUNT; i++) {
        if (activePhases[i] !== 'bouncing') continue;
        for (let j = i + 1; j < HEAD_COUNT; j++) {
          if (activePhases[j] !== 'bouncing') continue;
          const a = motions[i];
          const b = motions[j];
          const ax = a.x + RADIUS;
          const ay = a.y + RADIUS;
          const bx = b.x + RADIUS;
          const by = b.y + RADIUS;
          const dx = bx - ax;
          const dy = by - ay;
          const distSq = dx * dx + dy * dy;
          if (distSq >= COLLISION_DIST * COLLISION_DIST || distSq === 0) continue;
          const dist = Math.sqrt(distSq);
          const nx = dx / dist;
          const ny = dy / dist;
          // Push apart so they don't stick when overlapping
          const overlap = (COLLISION_DIST - dist) / 2;
          a.x -= nx * overlap;
          a.y -= ny * overlap;
          b.x += nx * overlap;
          b.y += ny * overlap;
          // Relative velocity along the normal
          const rvx = b.vx - a.vx;
          const rvy = b.vy - a.vy;
          const velAlongNormal = rvx * nx + rvy * ny;
          if (velAlongNormal > 0) continue; // already separating
          // Equal-mass elastic collision: swap velocity components along the normal
          a.vx += velAlongNormal * nx;
          a.vy += velAlongNormal * ny;
          b.vx -= velAlongNormal * nx;
          b.vy -= velAlongNormal * ny;
          // Reverse rotation directions to add visual punch on contact
          a.angularV = -a.angularV;
          b.angularV = -b.angularV;
        }
      }

      // 3. Write transforms
      for (let i = 0; i < HEAD_COUNT; i++) {
        if (activePhases[i] !== 'bouncing') continue;
        const m = motions[i];
        const el = headRefs.current[i];
        if (el) {
          el.style.transform = `translate3d(${m.x}px, ${m.y}px, 0) rotate(${m.angle}deg)`;
        }
      }
      raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener('resize', onResize);
    };
  }, []);

  // Respawn each hidden head 5 minutes after its own explosion.
  useEffect(() => {
    const timers: number[] = [];
    phases.forEach((phase, i) => {
      if (phase !== 'hidden') return;
      const t = window.setTimeout(() => {
        setPhases((prev) => {
          if (prev[i] !== 'hidden') return prev;
          const next = [...prev];
          next[i] = 'bouncing';
          return next;
        });
      }, HEAD_RESPAWN_MS);
      timers.push(t);
    });
    return () => { for (const t of timers) window.clearTimeout(t); };
  }, [phases]);

  const handleClick = useCallback((idx: number) => {
    const m = motionsRef.current[idx];
    const ox = m.x + HEAD_SIZE / 2;
    const oy = m.y + HEAD_SIZE / 2;
    spawnExplosionParticles(particlesRef.current, ox, oy);
    runFireLoop();
    setPhases((prev) => {
      const next = [...prev];
      next[idx] = 'hidden';
      return next;
    });
  }, [runFireLoop]);

  if (imgLoadFailed) return null;

  return (
    <>
      <canvas ref={explosionCanvasRef} className="head-explosion" />
      {phases.map((phase, i) => (
        <React.Fragment key={i}>
          {phase === 'bouncing' ? (
            <div
              ref={(el) => { headRefs.current[i] = el; }}
              className="bouncing-head"
              onClick={() => handleClick(i)}
              role="button"
              aria-label="floating head"
            >
              <img
                src={HEAD_SRC}
                alt=""
                draggable={false}
                onError={() => setImgLoadFailed(true)}
              />
            </div>
          ) : null}
        </React.Fragment>
      ))}
    </>
  );
});

function FilterlessLiveCockpit() {
  const [state, setState] = useState<FilterlessLiveState>(EMPTY_STATE);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeScreen, setActiveScreen] = useState<ScreenId>('overview');
  const [pollingPaused, setPollingPaused] = useState(false);
  const [refreshNonce, setRefreshNonce] = useState(0);
  const [controlStatus, setControlStatus] = useState<OperatorControlStatus | null>(null);
  const [controlError, setControlError] = useState<string | null>(null);
  const [controlMessage, setControlMessage] = useState<string | null>(null);
  const inFlightRef = useRef(false);
  const abortRef = useRef<AbortController | null>(null);
  const lastGeneratedAtRef = useRef<string | null>(null);
  const lastGoodSentimentRef = useRef<FilterlessSentimentMetrics>(DEFAULT_SENTIMENT_METRICS);

  const fetchControlStatus = useCallback(async () => {
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), 1800);
    try {
      const response = await fetch(`${OPERATOR_CONTROL_URL}/status?ts=${Date.now()}`, {
        cache: 'no-store',
        signal: controller.signal,
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const payload = (await response.json()) as OperatorControlStatus;
      startTransition(() => {
        setControlStatus(payload);
        setControlError(null);
      });
    } catch (err) {
      const isAbortError =
        (err instanceof DOMException && err.name === 'AbortError') ||
        (err instanceof Error && err.name === 'AbortError');
      startTransition(() => {
        setControlStatus(null);
        setControlError(isAbortError ? 'launcher controls timeout' : err instanceof Error ? err.message : String(err));
      });
    } finally {
      window.clearTimeout(timeout);
    }
  }, []);

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
    if (pollingPaused) {
      return () => {
        cancelled = true;
        abortRef.current?.abort();
      };
    }
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
  }, [pollingPaused, refreshNonce]);

  useEffect(() => {
    void fetchControlStatus();
    const timer = window.setInterval(fetchControlStatus, 5000);
    return () => window.clearInterval(timer);
  }, [fetchControlStatus]);

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
  const botProcess = findOperatorProcess(controlStatus, 'filterless bot');
  const bridgeProcess = findOperatorProcess(controlStatus, 'dashboard bridge');
  const frontendProcess = findOperatorProcess(controlStatus, 'live ui');
  const controlOnline = Boolean(controlStatus?.ok);

  const forceFeedRefresh = () => {
    setControlMessage('Feed refresh queued');
    setRefreshNonce((value) => value + 1);
    void fetchControlStatus();
  };

  const copyOperatorSnapshot = async () => {
    const snapshot = {
      captured_at: new Date().toISOString(),
      feed: {
        status: effectiveStatus,
        generated_at: state.generated_at,
        error,
        warnings: feedWarnings,
      },
      bot: {
        status: state.bot.status,
        session: state.bot.session,
        price: state.bot.price,
        last_heartbeat_time: state.bot.last_heartbeat_time,
        last_bar_time: state.bot.last_bar_time,
        position_sync_status: state.bot.position_sync_status,
        daily_pnl: state.bot.risk.daily_pnl,
        circuit_tripped: state.bot.risk.circuit_tripped,
      },
      exposure: {
        open_positions: openPositions,
        total_lots: totalLots,
        total_open_pnl: totalOpenPnl,
      },
      manifold: {
        regime: features.regime,
        pressure30: features.pressure30,
        fold_depth: features.foldDepth,
        risk_mult: features.riskMult,
        no_trade: features.noTrade,
      },
      kalshi: {
        healthy: kalshi?.healthy,
        event_ticker: kalshi?.event_ticker,
        status_reason: kalshi?.status_reason,
        probability_60m: kalshi?.probability_60m,
      },
      truth: {
        healthy: sentiment.healthy,
        label: sentiment.sentiment_label,
        score: sentiment.sentiment_score,
        trigger_reason: sentiment.trigger_reason,
        last_error: sentiment.last_error,
      },
      recent_events: state.events.slice(0, 12),
    };
    const text = JSON.stringify(snapshot, null, 2);
    try {
      await navigator.clipboard.writeText(text);
      setControlMessage('Snapshot copied');
    } catch {
      const textarea = document.createElement('textarea');
      textarea.value = text;
      textarea.style.position = 'fixed';
      textarea.style.left = '-9999px';
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand('copy');
      document.body.removeChild(textarea);
      setControlMessage('Snapshot copied');
    }
  };

  const sendOperatorCommand = async (action: OperatorCommandAction, label: string) => {
    if (!controlOnline) {
      setControlMessage('Launcher controls offline');
      return;
    }
    if (action === 'restart_bot' && openPositions.length > 0) {
      const ok = window.confirm('Restart the bot while a broker position is active? Exits are broker-side, but the runtime loop will reconnect.');
      if (!ok) return;
    }
    try {
      setControlMessage(`${label} requested`);
      const response = await fetch(`${OPERATOR_CONTROL_URL}/command`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action }),
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      setControlMessage(`${label} queued`);
      window.setTimeout(() => {
        void fetchControlStatus();
        setRefreshNonce((value) => value + 1);
      }, 1000);
    } catch (err) {
      setControlMessage(`${label} failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  };

  const logicRows = [
    ['FEED', `status ${effectiveStatus}, heartbeat ${formatRelativeTime(state.bot.last_heartbeat_time)}`, effectiveStatus === 'online' ? 'pass' : 'block'],
    ['ENTRY', primaryPosition ? `${primaryPosition.side} ${primaryPosition.size ?? '--'} @ ${formatPrice(entry)}` : 'flat, no active broker position', primaryPosition ? 'live' : 'idle'],
    ['GUARD', features.noTrade ? `${features.regime} lockout / stress ${pct(features.stress)}` : `risk mult ${fmt(features.riskMult)}x inside bounds`, features.noTrade ? 'block' : 'pass'],
    ['TRUTH', features.truthRiskWatch ? `advisory pressure ${pct(features.advisoryPressure)}` : `score ${formatSigned(features.truthScore)} / confidence ${pct(features.truthConfidence)}`, features.truthRiskWatch ? 'watch' : 'info'],
  ];

  const renderOverview = () => (
    <section className="screen">
      {/* TOP: 3-card metric row (last price / open pnl / target room) — full width */}
      <div className="grid cols-3">
        <Metric label="last price" value={formatPrice(price)} hint="streaming tape" color={COLORS.cyan} />
        <Metric label="open pnl" value={primaryPosition ? formatMoney(totalOpenPnl) : '--'} hint="broker shadow" color={totalOpenPnl >= 0 ? COLORS.lime : COLORS.amber} />
        <Metric label="target room" value={targetRoom == null ? '--' : `${fmt(targetRoom, 1)}pt`} hint="to active TP" color={COLORS.lime} />
      </div>

      {/* MIDDLE: Full-width execution chart with position metadata strip */}
      <Panel title="Execution Chart" titleClassName="display-title" subtitle="Price path with entry, stop, and take-profit rails." badge={<Badge tone={primaryPosition ? 'info' : 'watch'}>{primaryPosition ? 'live trade' : 'flat'}</Badge>} className="mt-panel">
        <PriceCanvas state={state} position={primaryPosition} />
        <div className="mini-grid">
          <div className="mini"><span className="label">position</span><strong className={`truncate ${sideTone(primaryPosition?.side)}`}>{primaryPosition ? `${primaryPosition.side} ${primaryPosition.size ?? '--'} MES` : 'FLAT'}</strong></div>
          <div className="mini"><span className="label">entry</span><strong className="truncate info">{formatPrice(entry)}</strong></div>
          <div className="mini"><span className="label">stop</span><strong className="truncate down">{formatPrice(stop)}</strong></div>
          <div className="mini"><span className="label">tp route</span><strong className="truncate up">{formatPrice(target)}</strong></div>
        </div>
      </Panel>

      {/* BOTTOM: All four side panels (Execution Logic / Risk Monitor /
          Active Positions / Order Flow) in a single row beneath the chart. */}
      <div className="grid cols-4 mt-panel">
        <Panel title="Execution Logic" titleClassName="display-title" subtitle="Execution score, guard rails, and live context." badge={<Badge tone={features.noTrade ? 'block' : 'live'}>{features.noTrade ? 'guard' : 'armed'}</Badge>}>
          <div className="terminal">
            {logicRows.map(([title, text, tone]) => (
              <TerminalRow key={title} time={state.generated_at} title={title} text={text} badge={<Badge tone={tone === 'block' ? 'block' : tone === 'watch' ? 'watch' : 'info'}>{tone}</Badge>} />
            ))}
          </div>
        </Panel>
        <Panel title="Risk Monitor" titleClassName="display-title" subtitle="Kelly, drawdown, and correlation pressure." badge={<Badge tone={state.bot.risk.circuit_tripped ? 'block' : 'watch'}>{state.bot.risk.circuit_tripped ? 'tripped' : 'watch'}</Badge>}>
          <div className="panel-body">
            <Meter label="risk mult" value={features.riskMult / 1.5} text={`${fmt(features.riskMult)}x`} color={COLORS.purple} />
            <Meter label="drawdown" value={clip01(Math.abs(dailyPnl ?? 0) / 2000)} text={formatMoney(dailyPnl)} color={COLORS.red} />
            <Meter label="correlation" value={features.R} text={fmt(features.R)} color={COLORS.cyan} />
          </div>
        </Panel>
        <Panel title="Active Positions" titleClassName="display-title" subtitle="Live P/L and route notes." badge={<Badge tone={openPositions.length ? 'live' : 'watch'}>{openPositions.length ? `${openPositions.length} live` : 'flat'}</Badge>}>
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
        <Panel title="Order Flow" titleClassName="display-title" subtitle="Live pressure derived from tape and route state." badge={<Badge tone="info">depth</Badge>}>
          <FlowBars values={[
            { label: 'align', value: features.alignment, color: COLORS.green },
            { label: 'smooth', value: features.smoothness, color: COLORS.cyan },
            { label: 'stress', value: features.stress, color: COLORS.amber },
            { label: 'burst', value: features.burstPressure, color: COLORS.pink },
            { label: 'risk', value: features.riskMult / 1.5, color: COLORS.purple },
          ]} />
        </Panel>
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
        <Panel title="AetherFlow Manifold" titleClassName="display-title" subtitle="Folded 3D state surface from pressure, transition, and manifold feature fields." badge={<Badge tone={features.noTrade ? 'block' : 'info'}>{features.noTrade ? 'lockout' : 'orbit live'}</Badge>}>
          <AetherflowCanvas features={features} />
          <div className="mini-grid">
            <div className="mini"><span className="label">trend ridge</span><strong className="truncate up">alignment + smoothness</strong></div>
            <div className="mini"><span className="label">burst peak</span><strong className="truncate warn">transition pressure</strong></div>
            <div className="mini"><span className="label">chop shelf</span><strong className="truncate info">stress roughness</strong></div>
            <div className="mini"><span className="label">rot wall</span><strong className="truncate down">dPhi lockout</strong></div>
          </div>
        </Panel>
        <div className="stack">
          <Panel title="Manifold Features" titleClassName="display-title" subtitle="Fields exported into AetherFlow feature space.">
            <div className="panel-body">
              <Meter label="alignment pct" value={features.alignment} color={COLORS.green} />
              <Meter label="smoothness pct" value={features.smoothness} color={COLORS.cyan} />
              <Meter label="stress pct" value={features.stress} color={COLORS.amber} />
              <Meter label="dispersion pct" value={features.dispersion} color={COLORS.violet} />
              <Meter label="novelty score" value={features.novelty} color={COLORS.pink} />
            </div>
          </Panel>
          <Panel title="State Regions" titleClassName="display-title" subtitle="Active state controls marker and routing gates.">
            <div className="panel-body">
              <div className="state-matrix">
                <Tile title="TREND_GEODESIC" titleClassName="display-title" text="High alignment and smoothness, low dispersion." color={COLORS.green} active={features.regime === 'TREND_GEODESIC'} badge={<Badge tone="live">ridge</Badge>} />
                <Tile title="CHOP_SPIRAL" text="Stress or roughness dominates mean-reversion routes." color={COLORS.cyan} active={features.regime === 'CHOP_SPIRAL'} badge={<Badge tone="info">shelf</Badge>} />
                <Tile title="DISPERSED" text="Dispersion is high while alignment is weak." color={COLORS.violet} active={features.regime === 'DISPERSED'} badge={<Badge tone="watch">basin</Badge>} />
                <Tile title="ROTATIONAL" text="Stress and dPhi create no-trade turbulence." color={COLORS.red} active={features.regime === 'ROTATIONAL_TURBULENCE'} badge={<Badge tone="block">wall</Badge>} />
              </div>
            </div>
          </Panel>
          <Panel title="AetherFlow Setups" titleClassName="display-title" subtitle="Feature interactions projected onto the manifold.">
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
        {/* Market Scanner — full width so the strike table fills the
            section box instead of being cramped in a 1.35fr column with
            empty space to the right. */}
        <Panel title="Market Scanner" titleClassName="display-title" subtitle="Hourly ES contracts ranked by edge, pressure, and liquidity." badge={<Badge tone={route.tone}>{route.badge}</Badge>} className="mt-panel">
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
        {/* Consensus + Spread side by side under the scanner. */}
        <div className="grid kalshi-layout-secondary">
          <Panel title="Consensus Feed" titleClassName="display-title" subtitle="Macro consensus and headline risk.">
            <div className="terminal">
              <TerminalRow title="EVENT" titleClassName="display-title" text={kalshi?.event_ticker || 'waiting for active contract'} badge={<Badge tone="info">ticker</Badge>} />
              <TerminalRow title="REFERENCE" titleClassName="display-title" text={`ES ${formatPrice(kalshi?.es_reference_price ?? price)} / SPX ${formatPrice(kalshi?.spx_reference_price)}`} badge={<Badge tone="info">basis</Badge>} />
              <TerminalRow title="ROUTE STATE" titleClassName="display-title" text={route.detail} badge={<Badge tone={route.tone}>{route.badge}</Badge>} />
              <TerminalRow title="STRATEGY IMPACT" titleClassName="display-title" text={route.impact} badge={<Badge tone={route.tone}>{route.value.toLowerCase()}</Badge>} />
              <TerminalRow title="LIVE POSITION" titleClassName="display-title" text={describeKalshiPositionImpact(primaryKalshiPosition)} badge={<Badge tone={primaryKalshiPosition?.kalshi_trade_overlay_applied || primaryKalshiPosition?.kalshi_gate_applied ? 'live' : 'info'}>{primaryKalshiPosition ? 'position' : 'flat'}</Badge>} />
            </div>
          </Panel>
          <Panel title="Spread Ladder" titleClassName="display-title" subtitle="Execution spread by bucket.">
            <FlowBars values={[
              { label: 'book', value: features.edge, color: COLORS.cyan },
              { label: 'spread', value: features.spread * 8, color: COLORS.amber },
              { label: 'prob', value: kalshi?.probability_60m ?? 0, color: COLORS.lime },
              { label: 'stress', value: features.stress, color: COLORS.red },
            ]} />
          </Panel>
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
            <Panel title="Truth Social Monitor" titleClassName="display-title" subtitle="Persisted sentiment snapshot for operator context." badge={<Badge tone={sentiment.last_error ? 'block' : 'live'}>{sentiment.last_error ? 'issue' : 'healthy'}</Badge>}>
              <div className="panel-body">
                <div className="truth-grid">
                  <Tile title="latest post" text={excerpt || 'No post has been analyzed yet. Waiting for new Truth Social activity.'} color={COLORS.cyan} badge={<Badge tone="info">rss</Badge>} />
                  <Tile title="model mode" titleClassName="display-title" text={sentiment.quantized_8bit ? 'Quantized FinBERT path is active.' : 'FinBERT runtime path is active.'} color={COLORS.purple} badge={<Badge tone="live">finbert</Badge>} />
                  <Tile title="long-side watch" titleClassName="display-title" text="Negative sentiment is displayed only; position management remains unchanged." color={COLORS.amber} badge={<Badge tone="info">observe</Badge>} />
                  <Tile title="short-side watch" titleClassName="display-title" text="Positive sentiment is displayed only; position management remains unchanged." color={COLORS.red} badge={<Badge tone="info">observe</Badge>} />
                </div>
              </div>
            </Panel>
            <Panel title="News Tape" titleClassName="display-title" subtitle="Calendar, macro, and truth-feed context." badge={<Badge tone="info">live</Badge>}>
              <div className="terminal">
                <TerminalRow time={sentiment.latest_post_created_at} title="TRUTH" titleClassName="display-title" text={sentiment.trigger_reason || sentiment.sentiment_label || 'neutral watch'} badge={<Badge tone="info">feed</Badge>} />
                <TerminalRow time={sentiment.last_analysis_at} title="FINBERT" titleClassName="display-title" text={`score ${formatSigned(features.truthScore)} / confidence ${pct(sentiment.finbert_confidence)}`} badge={<Badge tone="live">model</Badge>} />
                <TerminalRow time={state.generated_at} title="ADVISORY" titleClassName="display-title" text={`pressure ${pct(features.advisoryPressure)} / headline risk ${pct(features.headlineRisk)}`} badge={<Badge tone={features.truthRiskWatch ? 'watch' : 'info'}>{features.truthRiskWatch ? 'watch' : 'clear'}</Badge>} />
              </div>
            </Panel>
          </div>
          <div className="stack">
            <Panel title="Sentiment Pulse" titleClassName="display-title" subtitle="Truth Social score and confidence.">
              <SentimentCanvas sentiment={sentiment} />
            </Panel>
            <Panel title="Monitor Status" titleClassName="display-title" subtitle="Observe-only sentiment health and headline-risk context.">
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

  const renderStrategies = () => {
    // Frontend-side roster (mirrors bridge's STRATEGY_ORDER). Cards render
    // for every entry here even if the backend snapshot hasn't yet populated
    // a strategies[].id matching it — falls back to a placeholder card.
    const STRATEGY_ROSTER: Array<{ id: string; label: string }> = [
      { id: 'dynamic_engine3', label: 'Dynamic Engine 3' },
      { id: 'regime_adaptive', label: 'RegimeAdaptive' },
      { id: 'ml_physics', label: 'ML Physics' },
      { id: 'aetherflow', label: 'AetherFlow' },
      { id: 'fib_h1214', label: 'Fibonacci' },
    ];

    // Substrategy universe per module — chips light up when matched in live state
    const SUBSTRATEGIES: Record<string, Array<{ key: string; label: string; matchers: string[] }>> = {
      fib_h1214: [
        { key: 'fib_236', label: '0.236', matchers: ['236', 'fib_236'] },
        { key: 'fib_382', label: '0.382', matchers: ['382', 'fib_382'] },
        { key: 'fib_500', label: '0.500', matchers: ['500', 'fib_500'] },
        { key: 'fib_618', label: '0.618', matchers: ['618', 'fib_618'] },
        { key: 'fib_705', label: '0.705', matchers: ['705', 'fib_705'] },
        { key: 'fib_786', label: '0.786', matchers: ['786', 'fib_786'] },
        { key: 'fib_1000', label: '1.000', matchers: ['1000', 'fib_1000'] },
      ],
      dynamic_engine3: [
        { key: 'long_rev_5min', label: '5m Long Rev', matchers: ['5min', 'long_rev', '5m_long_rev'] },
        { key: 'long_rev_15min', label: '15m Long Rev', matchers: ['15min', '15m_long_rev'] },
        { key: 'short_rev_5min', label: '5m Short Rev', matchers: ['5min_short', '5m_short_rev'] },
        { key: 'short_rev_15min', label: '15m Short Rev', matchers: ['15min_short', '15m_short_rev'] },
        { key: 'ny_am_bypass', label: 'NY-AM Bypass', matchers: ['06-09', 'ny_am', 'bypass'] },
        { key: 'london_ext', label: 'London Ext', matchers: ['ldn', 'london'] },
      ],
      aetherflow: [
        { key: 'aligned_flow', label: 'Aligned Flow', matchers: ['aligned_flow'] },
        { key: 'exhaustion_reversal', label: 'Exhaustion Rev', matchers: ['exhaustion_reversal', 'exhaustion'] },
        { key: 'transition_burst', label: 'Transition Burst', matchers: ['transition_burst', 'transition'] },
        { key: 'compression_break', label: 'Compression Break', matchers: ['compression', 'compress'] },
        { key: 'flow_continuation', label: 'Flow Cont.', matchers: ['continuation', 'flow_cont'] },
      ],
      ml_physics: [
        { key: 'ny_am_h1011', label: 'NY-AM h10-11', matchers: ['ny_am', 'h1011', 'h10_11'] },
        { key: 'ny_pm', label: 'NY-PM', matchers: ['ny_pm'] },
        { key: 'asia', label: 'Asia', matchers: ['asia'] },
        { key: 'london', label: 'London', matchers: ['london', 'ldn'] },
      ],
      regime_adaptive: [
        { key: 'whipsaw', label: 'Whipsaw', matchers: ['whipsaw'] },
        { key: 'calm_trend', label: 'Calm Trend', matchers: ['calm_trend', 'calm'] },
        { key: 'neutral', label: 'Neutral', matchers: ['neutral'] },
      ],
    };

    const matchSubstrategy = (strategy: FilterlessStrategyState, sub: { key: string; label: string; matchers: string[] }): 'active' | 'recent' | 'idle' => {
      const haystack = [
        strategy.sub_strategy,
        strategy.combo_key,
        strategy.rule_id,
        strategy.latest_activity,
        strategy.latest_activity_type,
      ].filter(Boolean).map((v) => String(v).toLowerCase()).join(' | ');
      const m = sub.matchers.some((token) => haystack.includes(token.toLowerCase()));
      if (m && strategy.status === 'in_trade') return 'active';
      if (m) return 'recent';
      return 'idle';
    };

    const statusTone = (status: string): BadgeTone => {
      const s = (status || '').toLowerCase();
      if (s === 'in_trade' || s === 'active') return 'live';
      if (s === 'idle' || s === 'ready') return 'watch';
      if (s.includes('disabled') || s.includes('block')) return 'block';
      if (s === 'awaiting_snapshot' || s === 'standby') return 'info';
      return 'watch';
    };

    const statusLabel = (status: string): string => {
      const s = (status || 'idle').toLowerCase();
      // Bridge hasn't pushed live state yet — strategy is loaded + registered
      // but no signals/trades have flowed through the per-strategy slot.
      // 'STANDBY' matches the Kronos daemon convention used elsewhere
      // (loaded, available, not actively firing right now).
      if (s === 'awaiting_snapshot') return 'STANDBY';
      return (status || 'idle').toUpperCase();
    };

    const colorForStrategy = (id: string): string => {
      const m: Record<string, string> = {
        aetherflow: COLORS.green,
        dynamic_engine3: COLORS.amber,
        ml_physics: COLORS.cyan,
        regime_adaptive: COLORS.purple,
        fib_h1214: COLORS.pink,
      };
      return m[id] || COLORS.cyan;
    };

    const fmtTime = (iso?: string | null): string => {
      if (!iso) return '—';
      try {
        const d = new Date(iso);
        if (Number.isNaN(d.getTime())) return iso;
        // Compute "today in NY" via Intl tz-conversion so cross-tz hosts
        // (PT, etc.) don't show yesterday's date for events recorded today.
        const nyToday = new Date().toLocaleDateString('en-US', { timeZone: 'America/New_York' });
        const tradeDay = d.toLocaleDateString('en-US', { timeZone: 'America/New_York' });
        const time = d.toLocaleTimeString('en-US', {
          hour12: false,
          timeZone: 'America/New_York',
          hour: '2-digit',
          minute: '2-digit',
        });
        if (tradeDay === nyToday) return `${time} ET`;
        const day = d.toLocaleDateString('en-US', {
          month: 'short',
          day: '2-digit',
          timeZone: 'America/New_York',
        });
        return `${day} ${time} ET`;
      } catch { return iso; }
    };

    const fmtMoney = (v?: number | null): string => {
      if (v == null) return '—';
      return (v >= 0 ? '+$' : '-$') + Math.abs(v).toFixed(2);
    };

    // Pull last-trade info from state.trades as a fallback when the
    // strategy-state entry doesn't have it yet (e.g., bridge hasn't been
    // restarted to pick up a new strategy_id, so per-strategy state is stale
    // but the trades list still has the entry once canonical_strategy_id
    // matches the strategy name).
    const lastTradeForStrategy = (id: string) => {
      const trades = state.trades || [];
      // Search by canonical id first
      let match = [...trades].reverse().find((t) => t.strategy_id === id);
      // Fallback for fib_h1214: also match anything containing "fib"
      // (e.g., strategy_label="FibH1214_fib_236") in case bridge canonical id
      // matcher is stale.
      if (!match && id === 'fib_h1214') {
        match = [...trades].reverse().find((t) =>
          /fib/i.test(String(t.strategy_label || '')) ||
          /fib/i.test(String(t.strategy_id || ''))
        );
      }
      return match;
    };

    // Build the rendered list: pull each strategy from state if present,
    // else render a placeholder card so the roster is always visible even
    // before the bridge backend re-sends the snapshot with new strategies.
    const renderedStrategies: FilterlessStrategyState[] = STRATEGY_ROSTER.map((entry) => {
      const live = state.strategies.find((s) => s.id === entry.id);
      // If we have a trade in state.trades that matches but no strategy state
      // entry, synthesize one so the card shows real PnL data.
      const fallbackTrade = lastTradeForStrategy(entry.id);
      if (live) {
        // Override label so frontend stays the source of truth for display name.
        // If live state lacks last_trade fields but state.trades has one, hydrate.
        if (!live.last_trade_pnl && fallbackTrade) {
          return {
            ...live,
            label: entry.label,
            last_trade_pnl: fallbackTrade.pnl_dollars ?? null,
            last_trade_points: fallbackTrade.pnl_points ?? null,
            last_trade_time: fallbackTrade.time ?? null,
            last_trade_side: fallbackTrade.side ?? null,
            last_trade_entry: fallbackTrade.entry_price ?? null,
            last_trade_exit: fallbackTrade.exit_price ?? null,
          };
        }
        return { ...live, label: entry.label };
      }
      // No live strategy state — synthesize from any matching trade
      if (fallbackTrade) {
        return {
          id: entry.id,
          label: entry.label,
          status: 'ready',
          last_trade_pnl: fallbackTrade.pnl_dollars ?? null,
          last_trade_points: fallbackTrade.pnl_points ?? null,
          last_trade_time: fallbackTrade.time ?? null,
          last_trade_side: fallbackTrade.side ?? null,
          last_trade_entry: fallbackTrade.entry_price ?? null,
          last_trade_exit: fallbackTrade.exit_price ?? null,
          sub_strategy: fallbackTrade.strategy_label?.replace(/^FibH1214_/i, '') ?? null,
        } as FilterlessStrategyState;
      }
      return {
        id: entry.id,
        label: entry.label,
        status: 'awaiting_snapshot',
      } as FilterlessStrategyState;
    });

    const activeCount = renderedStrategies.filter((s) => (s.status || '').toLowerCase() === 'in_trade').length;
    const readyCount = renderedStrategies.filter((s) => ['idle', 'ready'].includes((s.status || '').toLowerCase())).length;
    const loadedCount = renderedStrategies.length;

    return (
      <section className="screen">
        <div className="grid cols-3">
          <Metric label="active modules" value={String(activeCount)} hint="currently in trade" color={COLORS.green} />
          <Metric label="ready modules" value={String(readyCount)} hint="awaiting setup" color={COLORS.amber} />
          <Metric label="loaded modules" value={String(loadedCount)} hint="in roster" color={COLORS.cyan} />
        </div>

        <Panel
          title="Strategy Stack"
          titleClassName="display-title"
          subtitle="Modules with active sub-strategies — recent fires highlighted."
          badge={<Badge tone="live">loaded</Badge>}
          className="mt-panel"
        >
          {renderedStrategies.length === 0 ? (
            <p className="micro">No strategies are present in the live snapshot.</p>
          ) : (
            <div className="strategy-card-grid">
              {renderedStrategies.map((strategy: FilterlessStrategyState) => {
                const subs = SUBSTRATEGIES[strategy.id] || [];
                const accent = colorForStrategy(strategy.id);
                const tone = statusTone(strategy.status);
                const lastSubLabel = strategy.sub_strategy || strategy.combo_key || strategy.rule_id;
                return (
                  <div
                    key={strategy.id}
                    className="strategy-card"
                    style={{
                      borderTop: `2px solid ${accent}`,
                    }}
                  >
                    <div className="strategy-card-header">
                      <div>
                        <div className="strategy-card-title" style={{ color: accent }}>
                          {strategy.label}
                        </div>
                        <div className="strategy-card-id micro">{strategy.id}</div>
                      </div>
                      <Badge tone={tone}>{statusLabel(strategy.status)}</Badge>
                    </div>

                    {subs.length > 0 && (
                      <div className="strategy-substrategies">
                        <div className="strategy-substrategies-label micro">substrategies</div>
                        <div className="strategy-chip-row">
                          {subs.map((sub) => {
                            const state = matchSubstrategy(strategy, sub);
                            return (
                              <span
                                key={sub.key}
                                className={`strategy-chip strategy-chip-${state}`}
                                style={state !== 'idle' ? {
                                  borderColor: accent,
                                  color: accent,
                                  backgroundColor: state === 'active' ? `${accent}33` : `${accent}1A`,
                                } : undefined}
                                title={`${sub.label} — ${state}`}
                              >
                                {state === 'active' ? '◉ ' : state === 'recent' ? '○ ' : ''}
                                {sub.label}
                              </span>
                            );
                          })}
                        </div>
                      </div>
                    )}

                    <div className="strategy-card-stats">
                      <div className="strategy-stat">
                        <div className="strategy-stat-label">last signal</div>
                        <div className="strategy-stat-value">
                          {strategy.last_signal_side ? (
                            <span style={{ color: strategy.last_signal_side === 'LONG' ? COLORS.green : COLORS.red }}>
                              {strategy.last_signal_side}
                            </span>
                          ) : '—'}
                          {strategy.last_signal_price != null && (
                            <span className="strategy-stat-sub"> @ {strategy.last_signal_price.toFixed(2)}</span>
                          )}
                        </div>
                        <div className="strategy-stat-meta micro">{fmtTime(strategy.last_signal_time)}</div>
                      </div>
                      <div className="strategy-stat">
                        <div className="strategy-stat-label">last trade</div>
                        <div
                          className="strategy-stat-value"
                          style={{
                            color: strategy.last_trade_pnl == null ? undefined : (strategy.last_trade_pnl >= 0 ? COLORS.green : COLORS.red),
                          }}
                        >
                          {fmtMoney(strategy.last_trade_pnl)}
                        </div>
                        <div className="strategy-stat-meta micro">
                          {strategy.last_trade_points != null ? `${strategy.last_trade_points >= 0 ? '+' : ''}${strategy.last_trade_points.toFixed(2)} pts` : ''}
                          {strategy.last_trade_time ? ' • ' + fmtTime(strategy.last_trade_time) : ''}
                        </div>
                      </div>
                      <div className="strategy-stat">
                        <div className="strategy-stat-label">brackets</div>
                        <div className="strategy-stat-value">
                          {strategy.tp_dist != null ? `TP ${strategy.tp_dist.toFixed(2)}` : '—'}
                          {strategy.sl_dist != null ? ` / SL ${strategy.sl_dist.toFixed(2)}` : ''}
                        </div>
                        <div className="strategy-stat-meta micro">
                          {strategy.entry_mode || strategy.priority || ''}
                        </div>
                      </div>
                    </div>

                    {(strategy.last_block_reason || strategy.last_reason || lastSubLabel) && (
                      <div className="strategy-card-footer">
                        {lastSubLabel && (
                          <span className="strategy-tag" style={{ color: accent, borderColor: accent }}>
                            {lastSubLabel}
                          </span>
                        )}
                        {strategy.last_block_reason && (
                          <span className="strategy-tag strategy-tag-block">
                            blocked: {strategy.last_block_reason}
                          </span>
                        )}
                        {strategy.last_reason && !strategy.last_block_reason && (
                          <span className="strategy-tag-muted micro">{strategy.last_reason}</span>
                        )}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </Panel>
      </section>
    );
  };

  const renderPipeline = () => {
    const pipeline: FilterlessPipelineState | null = state.pipeline ?? null;
    const v18 = pipeline?.v18_stacker;
    const kr = pipeline?.kronos;
    const rb = pipeline?.recipe_b;
    const rml = pipeline?.regime_ml;
    const ny = pipeline?.ny_am_bypass;
    const ss = pipeline?.sameside_ml;
    const af = pipeline?.af_regime_allowlist;
    const tri = pipeline?.triathlon;
    const fg = pipeline?.filter_g;

    const onTone = (flag?: boolean): BadgeTone => (flag ? 'live' : 'block');
    const onText = (flag?: boolean): string => (flag ? 'on' : 'off');

    const kronosRowTone: BadgeTone = !kr
      ? 'watch'
      : kr.daemon_running
        ? 'live'
        : kr.available
          ? 'watch'
          : 'block';
    // Tri-state language for the Kronos status. The daemon is lazy-spawned
    // on first need (see _ensure_kronos_daemon in julie001.py) — so a
    // dormant-but-available state is normal, not an outage. STANDBY makes
    // that distinction explicit instead of reading as "OFF / broken".
    const kronosRowText = !kr
      ? 'unknown'
      : kr.daemon_running
        ? 'ready'
        : kr.available
          ? 'standby'
          : 'missing';
    const kronosRunningText = kr?.daemon_running
      ? 'ON'
      : kr?.available
        ? 'STANDBY'
        : 'OFF';
    const kronosRunningColor = kr?.daemon_running
      ? COLORS.green
      : kr?.available
        ? COLORS.amber
        : COLORS.red;
    const kronosRunningHint = kr?.daemon_running
      ? 'daemon process up'
      : kr?.available
        ? 'spawns lazily on first non-bypass DE3 candidate'
        : 'unavailable — venv or script missing';

    const tierRows = rb?.tiers ?? [];
    const recipeBNote = rb?.skip_whipsaw_tier4
      ? 'tier-4 (0.65-0.85): calm_trend=4 / whipsaw=skip / neutral=1'
      : rb?.regime_aware_tier4
        ? 'tier-4 (0.65-0.85): calm_trend=4 / whipsaw=1 / neutral=1'
        : 'tier-4 (0.65-0.85): flat 4 contracts (regime-blind)';

    if (!pipeline) {
      return (
        <section className="screen">
          <Panel
            title="V18 Pipeline" titleClassName="display-title"
            subtitle="Awaiting bot state snapshot for live wiring details."
            badge={<Badge tone="watch">no snapshot</Badge>}
          >
            <div className="panel-body">
              <p className="micro">
                The bot has not yet written a pipeline block to bot_state.json. This populates
                on the next bot-state save (typically every few seconds while the bot is online).
              </p>
            </div>
          </Panel>
        </section>
      );
    }

    return (
      <section className="screen">
        <div className="grid pipeline-grid">
          <Metric
            label="V18 stacker"
            value={onText(v18?.enabled).toUpperCase()}
            hint={`thr ${fmt(v18?.threshold ?? 0, 2)}`}
            color={v18?.enabled ? COLORS.green : COLORS.red}
          />
          <Metric
            label="Kronos daemon"
            value={kronosRowText.toUpperCase()}
            hint={`restarts ${kr?.daemon_restarts ?? 0}`}
            color={
              kr?.daemon_running
                ? COLORS.green
                : kr?.available
                  ? COLORS.amber
                  : COLORS.red
            }
          />
          <Metric
            label="Recipe B sizing"
            value={onText(rb?.enabled).toUpperCase()}
            hint={
              rb?.skip_whipsaw_tier4
                ? 'opt 4b (skip whipsaw)'
                : rb?.regime_aware_tier4
                  ? 'opt 4 (demote)'
                  : 'flat tiers'
            }
            color={rb?.enabled ? COLORS.cyan : COLORS.dim}
          />
          <Metric
            label="Regime classifier"
            value={onText(rml?.classifier_enabled).toUpperCase()}
            hint="regime tag drives tier-4 + size-cap"
            color={rml?.classifier_enabled ? COLORS.green : COLORS.dim}
          />
        </div>

        <Panel
          title="Stacked Meta Gate (V18)" titleClassName="display-title"
          subtitle="LogReg meta-classifier on 6 base probas + 5 Kronos features."
          badge={<Badge tone={onTone(v18?.enabled)}>{onText(v18?.enabled)}</Badge>}
          className="mt-panel"
        >
          <div className="panel-body">
            <div className="grid cols-2">
              <Tile
                title="Threshold"
                text={`v18_proba >= ${fmt(v18?.threshold ?? 0, 2)} keeps the candidate; below blocks.`}
                color={COLORS.green}
                badge={<Badge tone="info">{fmt(v18?.threshold ?? 0, 2)}</Badge>}
                active={v18?.enabled}
              />
              <Tile
                title="Bundle"
                text={v18?.bundle_path ? v18.bundle_path.split('/').slice(-3).join('/') : 'not loaded'}
                color={v18?.enabled ? COLORS.cyan : COLORS.red}
                badge={<Badge tone={onTone(v18?.enabled)}>joblib</Badge>}
                active={v18?.enabled}
              />
            </div>
          </div>
        </Panel>

        <Panel
          title="Kronos Daemon" titleClassName="display-title"
          subtitle="Subprocess in .kronos_venv producing the 5 Kronos features per V18 candidate."
          badge={<Badge tone={kronosRowTone}>{kronosRowText}</Badge>}
          className="mt-panel"
        >
          <div className="panel-body">
            <div className="grid cols-3">
              <Metric
                label="Available"
                value={onText(kr?.available).toUpperCase()}
                hint=".kronos_venv + script present"
                color={kr?.available ? COLORS.green : COLORS.red}
              />
              <Metric
                label="Running"
                value={kronosRunningText}
                hint={kronosRunningHint}
                color={kronosRunningColor}
              />
              <Metric
                label="Restarts"
                value={String(kr?.daemon_restarts ?? 0)}
                hint={`per-call timeout ${fmt(kr?.timeout_s ?? 0, 0)}s`}
                color={COLORS.purple}
              />
            </div>
          </div>
        </Panel>

        <Panel
          title="Recipe B Tiered Sizing (Option 4b)" titleClassName="display-title"
          subtitle={recipeBNote}
          badge={<Badge tone={onTone(rb?.enabled)}>{onText(rb?.enabled)}</Badge>}
          className="mt-panel"
        >
          <div className="panel-body">
            {tierRows.length > 0 ? (
              <table className="table">
                <thead>
                  <tr><th>tier</th><th>v18_proba &gt;=</th><th>contracts</th></tr>
                </thead>
                <tbody>
                  {tierRows.map((row, idx) => (
                    <tr key={`tier-${idx}`}>
                      <td><strong>T{row[1]}</strong></td>
                      <td>{fmt(row[0], 2)}</td>
                      <td>{row[1]}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <p className="micro">Tier configuration not present in snapshot.</p>
            )}
          </div>
        </Panel>

        <Panel
          title="Regime ML Stack" titleClassName="display-title"
          subtitle="HGB classifiers gating BE / brackets / size from regime features."
          badge={<Badge tone={onTone(rml?.classifier_enabled)}>{onText(rml?.classifier_enabled)}</Badge>}
          className="mt-panel"
        >
          <div className="panel-body">
            <div className="grid cols-2">
              <Tile
                title="Classifier (master)"
                text="Tags each bar: calm_trend / neutral / whipsaw / dead_tape."
                color={rml?.classifier_enabled ? COLORS.green : COLORS.red}
                badge={<Badge tone={onTone(rml?.classifier_enabled)}>{onText(rml?.classifier_enabled)}</Badge>}
                active={rml?.classifier_enabled}
              />
              <Tile
                title="v6_be (BE-disable ML)" titleClassName="display-title"
                text="Disables breakeven on candidates the model thinks shouldn't trail."
                color={rml?.be_disable_ml ? COLORS.green : COLORS.dim}
                badge={<Badge tone={onTone(rml?.be_disable_ml)}>{onText(rml?.be_disable_ml)}</Badge>}
                active={rml?.be_disable_ml}
              />
              <Tile
                title="v5_brackets (scalp ML)"
                text="OFF — destroyed -$13.4k tier-10 PnL on 2026 OOS (out-of-distribution on V18 fires)."
                color={rml?.scalp_brackets_ml ? COLORS.red : COLORS.dim}
                badge={<Badge tone={rml?.scalp_brackets_ml ? 'block' : 'info'}>{onText(rml?.scalp_brackets_ml)}</Badge>}
                active={rml?.scalp_brackets_ml}
              />
              <Tile
                title="v6_size (size-reduction ML)" titleClassName="display-title"
                text="OFF — net negative on V18 selection; needs retraining on V18-filtered data first."
                color={rml?.size_reduction_ml ? COLORS.red : COLORS.dim}
                badge={<Badge tone={rml?.size_reduction_ml ? 'block' : 'info'}>{onText(rml?.size_reduction_ml)}</Badge>}
                active={rml?.size_reduction_ml}
              />
            </div>
          </div>
        </Panel>

        <Panel
          title="NY-AM Long_Rev Bypass" titleClassName="display-title"
          subtitle={`Routes around V18 + Recipe B + v6_be for hour ${ny?.hour_et ?? '--'} ET fires.`}
          badge={<Badge tone={onTone(ny?.enabled)}>{onText(ny?.enabled)}</Badge>}
          className="mt-panel"
        >
          <div className="panel-body">
            {ny?.subs?.length ? (
              <table className="table">
                <thead><tr><th>sub-strategy</th><th>hour ET</th></tr></thead>
                <tbody>
                  {ny.subs.map((sub) => (
                    <tr key={sub}>
                      <td><strong>{sub}</strong></td>
                      <td>{ny.hour_et}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <p className="micro">No bypass sub-strategies configured.</p>
            )}
          </div>
        </Panel>

        <Panel
          title="Adjacent Layers" titleClassName="display-title"
          subtitle="SameSide ML, AF regime allowlist, Filter G, Triathlon."
          badge={<Badge tone="info">live</Badge>}
          className="mt-panel"
        >
          <div className="panel-body">
            <div className="grid cols-2">
              <Tile
                title={`SameSide ML  (cap ${ss?.max_contracts ?? '--'})`}
                text="HGB gate that endorses adding a 2nd contract to an existing same-side family position."
                color={ss?.enabled ? COLORS.green : COLORS.dim}
                badge={<Badge tone={onTone(ss?.enabled)}>{onText(ss?.enabled)}</Badge>}
                active={ss?.enabled}
              />
              <Tile
                title="AF regime allowlist" titleClassName="display-title"
                text={af?.allowed_regimes?.length ? af.allowed_regimes.join(' + ') : 'no restriction'}
                color={af?.enabled ? COLORS.green : COLORS.dim}
                badge={<Badge tone={onTone(af?.enabled)}>{onText(af?.enabled)}</Badge>}
                active={af?.enabled}
              />
              <Tile
                title="Filter G (case-fix)" titleClassName="display-title"
                text="Per-cell normalized threshold — fires on active veto path (shadow-only on bypass)."
                color={fg?.enabled ? COLORS.green : COLORS.dim}
                badge={<Badge tone={onTone(fg?.enabled)}>{onText(fg?.enabled)}</Badge>}
                active={fg?.enabled}
              />
              <Tile
                title="Triathlon Engine" titleClassName="display-title"
                text="Per-cell medal-driven size + priority deltas. Off in default deployment."
                color={tri?.enabled ? COLORS.green : COLORS.dim}
                badge={<Badge tone={onTone(tri?.enabled)}>{onText(tri?.enabled)}</Badge>}
                active={tri?.enabled}
              />
            </div>
          </div>
        </Panel>
      </section>
    );
  };

  // ===========================================================================
  // Trade-lifecycle helpers (used by MFE/MAE scatter + Sankey)
  // ===========================================================================

  // Compute Maximum Favorable Excursion (MFE) and Maximum Adverse Excursion
  // (MAE) for a trade by scanning OHLC bars between opened_at and time.
  // Returns nulls if entry_price or opened_at is missing, or if no bars
  // overlap the trade window. Both are returned as POINTS (positive numbers).
  const computeMfeMae = (
    trade: FilterlessTrade,
    bars: FilterlessOhlcBar[],
  ): { mfe: number | null; mae: number | null; durationSec: number | null } => {
    const entry = trade.entry_price ?? null;
    const openedAt = (trade as any).opened_at as string | null | undefined;
    const closedAt = trade.time ?? null;
    if (entry == null || !openedAt || !closedAt || !bars || bars.length === 0) {
      return { mfe: null, mae: null, durationSec: null };
    }
    const t0 = new Date(openedAt).getTime();
    const t1 = new Date(closedAt).getTime();
    if (!Number.isFinite(t0) || !Number.isFinite(t1) || t1 < t0) {
      return { mfe: null, mae: null, durationSec: null };
    }
    let highest = -Infinity;
    let lowest = Infinity;
    for (const bar of bars) {
      const ts = new Date(bar.t).getTime();
      if (!Number.isFinite(ts)) continue;
      if (ts < t0 || ts > t1) continue;
      if (bar.h > highest) highest = bar.h;
      if (bar.l < lowest) lowest = bar.l;
    }
    if (!Number.isFinite(highest) || !Number.isFinite(lowest)) {
      return { mfe: null, mae: null, durationSec: Math.max(0, (t1 - t0) / 1000) };
    }
    const isLong = String(trade.side || '').toUpperCase() === 'LONG';
    const mfe = isLong ? Math.max(0, highest - entry) : Math.max(0, entry - lowest);
    const mae = isLong ? Math.max(0, entry - lowest) : Math.max(0, highest - entry);
    return { mfe, mae, durationSec: Math.max(0, (t1 - t0) / 1000) };
  };

  // Bucket the exit reason from available fields. Without an explicit
  // exit_reason column we use the result + price proximity as a proxy.
  const inferExitReason = (trade: FilterlessTrade): string => {
    const result = String(trade.result || '').toLowerCase();
    if (result === 'win' || result === 'tp' || result === 'take_profit') return 'Take Profit';
    if (result === 'loss' || result === 'sl' || result === 'stop_loss') return 'Stop Loss';
    if (result === 'timeout' || result === 'time_out') return 'Time-Out';
    if (result === 'flat' || result === 'breakeven' || result === 'be') return 'Breakeven';
    // Fallback: use realized PnL sign
    const pnl = trade.pnl_dollars_net ?? trade.pnl_dollars ?? trade.pnl_points ?? 0;
    if (pnl > 0) return 'Take Profit';
    if (pnl < 0) return 'Stop Loss';
    return 'Other';
  };

  const inferEntryType = (trade: FilterlessTrade): string => {
    const mode = ((trade as any).entry_mode as string | null | undefined) || '';
    if (mode) {
      const m = mode.toLowerCase();
      if (m.includes('limit')) return 'Limit';
      if (m.includes('market')) return 'Market';
      if (m.includes('stop')) return 'Stop';
      return mode;
    }
    return 'Market';
  };

  const inferSignalSource = (trade: FilterlessTrade): string => {
    const label = (trade.strategy_label || '').trim();
    if (label) return label;
    const id = (trade.strategy_id || '').trim();
    return id || 'Unknown';
  };

  // ===========================================================================
  // MFE / MAE Efficiency Scatter
  // ===========================================================================

  const MfeMaeScatterChart: React.FC<{ trades: FilterlessTrade[]; bars: FilterlessOhlcBar[] }> = ({ trades, bars }) => {
    const ref = useRef<HTMLDivElement>(null);

    const points = useMemo(() => {
      const enriched = trades.map((t) => {
        const ex = computeMfeMae(t, bars);
        const pnlPts = t.pnl_points ?? 0;
        const pnlDollars = t.pnl_dollars_net ?? t.pnl_dollars ?? 0;
        // If we have no OHLC overlap, fall back to using realized pnl as the
        // observed favorable/adverse magnitude. This still produces a useful
        // scatter — just less precise than true MFE/MAE.
        const mfe = ex.mfe ?? Math.max(0, pnlPts);
        const mae = ex.mae ?? Math.max(0, -pnlPts);
        const exitReason = inferExitReason(t);
        const isWin = (pnlDollars ?? 0) > 0 || /win|tp/i.test(t.result || '');
        return {
          mfe, mae,
          pnlPts,
          pnlDollars,
          isWin,
          win: isWin ? 'Win' : 'Loss',
          source: inferSignalSource(t),
          exitReason,
          side: String(t.side || '').toUpperCase(),
          entry: t.entry_price ?? null,
          exit: t.exit_price ?? null,
          time: t.time ?? null,
          duration: ex.durationSec,
          inferred: ex.mfe == null,
        };
      }).filter((p) => p.mfe != null && p.mae != null);
      return enriched;
    }, [trades, bars]);

    useEffect(() => {
      if (!ref.current) return;
      if (!points.length) {
        try { Plotly.purge(ref.current); } catch { /* noop */ }
        return;
      }
      const wins = points.filter((p) => p.isWin);
      const losses = points.filter((p) => !p.isWin);

      const traceFor = (pts: typeof points, name: string, color: string) => ({
        x: pts.map((p) => p.mae),
        y: pts.map((p) => p.mfe),
        text: pts.map((p) =>
          `<b>${p.source}</b> · ${p.side}<br>` +
          `MFE: ${p.mfe.toFixed(2)} pts<br>` +
          `MAE: ${p.mae.toFixed(2)} pts<br>` +
          `Realized: ${p.pnlPts.toFixed(2)} pts ($${(p.pnlDollars ?? 0).toFixed(2)})<br>` +
          `Exit: ${p.exitReason}<br>` +
          `Duration: ${p.duration != null ? `${(p.duration / 60).toFixed(1)} min` : '—'}<br>` +
          `Time: ${p.time || '—'}` +
          (p.inferred ? '<br><i>(approx — no OHLC overlap)</i>' : '')
        ),
        hoverinfo: 'text' as const,
        mode: 'markers' as const,
        type: 'scatter' as const,
        name,
        marker: {
          size: 10,
          color,
          line: { color: 'rgba(255,255,255,0.4)', width: 0.5 },
          opacity: 0.85,
        },
      });

      // 45° diagonal MFE = MAE: anything below this line means the trade went
      // further against you than for you (a likely "wasted edge").
      const allMags = points.flatMap((p) => [p.mfe, p.mae]);
      const maxMag = Math.max(1, ...allMags) * 1.1;

      const data: any[] = [
        traceFor(losses, 'Losses', COLORS.red),
        traceFor(wins, 'Wins', COLORS.green),
        {
          x: [0, maxMag], y: [0, maxMag],
          mode: 'lines', type: 'scatter',
          name: 'MFE = MAE',
          line: { color: 'rgba(255,255,255,0.25)', width: 1, dash: 'dash' },
          hoverinfo: 'skip',
          showlegend: true,
        },
      ];

      const layout: any = {
        autosize: true,
        margin: { l: 56, r: 24, t: 16, b: 48 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#d6d6d6', family: 'JetBrains Mono, monospace', size: 11 },
        xaxis: {
          title: { text: 'MAE — points against you', font: { size: 11 } },
          gridcolor: 'rgba(255,255,255,0.06)',
          zerolinecolor: 'rgba(255,255,255,0.18)',
          range: [0, maxMag],
        },
        yaxis: {
          title: { text: 'MFE — points in your favor', font: { size: 11 } },
          gridcolor: 'rgba(255,255,255,0.06)',
          zerolinecolor: 'rgba(255,255,255,0.18)',
          range: [0, maxMag],
        },
        legend: {
          orientation: 'h' as const,
          y: 1.08, x: 0,
          bgcolor: 'rgba(0,0,0,0)',
          font: { size: 10 },
        },
        hoverlabel: { bgcolor: '#1a1a1a', bordercolor: 'rgba(255,255,255,0.2)' },
      };

      const config: any = { displayModeBar: false, responsive: true };
      Plotly.newPlot(ref.current, data, layout, config);
      const onResize = () => { if (ref.current) Plotly.Plots.resize(ref.current); };
      window.addEventListener('resize', onResize);
      return () => {
        window.removeEventListener('resize', onResize);
        if (ref.current) try { Plotly.purge(ref.current); } catch { /* noop */ }
      };
    }, [points]);

    return (
      <div className="lifecycle-chart-wrap">
        {points.length === 0 ? (
          <div className="lifecycle-empty">No closed trades with OHLC overlap yet.</div>
        ) : (
          <div ref={ref} style={{ width: '100%', height: '100%' }} />
        )}
      </div>
    );
  };

  // ===========================================================================
  // Trade Lifecycle Sankey: Signal Source → Entry Type → Exit Reason
  // ===========================================================================

  const TradeLifecycleSankey: React.FC<{ trades: FilterlessTrade[] }> = ({ trades }) => {
    const ref = useRef<HTMLDivElement>(null);

    const sankeyData = useMemo(() => {
      if (!trades.length) return null;
      // Build node + link tables
      const nodeSet: string[] = [];
      const nodeIndex = new Map<string, number>();
      const addNode = (label: string): number => {
        if (nodeIndex.has(label)) return nodeIndex.get(label)!;
        const i = nodeSet.length;
        nodeSet.push(label);
        nodeIndex.set(label, i);
        return i;
      };

      // Group A: signal sources (strategies)
      // Group B: entry types
      // Group C: exit reasons
      const linkCounts = new Map<string, { source: number; target: number; count: number; pnl: number }>();
      const bumpLink = (a: number, b: number, pnlDelta: number) => {
        const key = `${a}->${b}`;
        const prior = linkCounts.get(key);
        if (prior) {
          prior.count += 1;
          prior.pnl += pnlDelta;
        } else {
          linkCounts.set(key, { source: a, target: b, count: 1, pnl: pnlDelta });
        }
      };

      for (const t of trades) {
        const src = `Signal · ${inferSignalSource(t)}`;
        const ent = `Entry · ${inferEntryType(t)}`;
        const exi = `Exit · ${inferExitReason(t)}`;
        const a = addNode(src);
        const b = addNode(ent);
        const c = addNode(exi);
        const pnl = t.pnl_dollars_net ?? t.pnl_dollars ?? 0;
        bumpLink(a, b, pnl);
        bumpLink(b, c, pnl);
      }

      const links = Array.from(linkCounts.values());
      // Color exit-reason links by tone
      const linkColors = links.map((l) => {
        const targetLabel = nodeSet[l.target];
        if (targetLabel.startsWith('Exit · Take Profit')) return 'rgba(0, 230, 118, 0.35)';
        if (targetLabel.startsWith('Exit · Stop Loss')) return 'rgba(255, 64, 90, 0.35)';
        if (targetLabel.startsWith('Exit · Time-Out')) return 'rgba(255, 200, 0, 0.35)';
        if (targetLabel.startsWith('Exit · Breakeven')) return 'rgba(180, 180, 180, 0.35)';
        return 'rgba(120, 130, 200, 0.30)';
      });

      const nodeColors = nodeSet.map((label) => {
        if (label.startsWith('Signal · ')) return 'rgba(120, 130, 230, 0.85)';
        if (label.startsWith('Entry · ')) return 'rgba(180, 180, 180, 0.85)';
        if (label.startsWith('Exit · Take Profit')) return 'rgba(0, 230, 118, 0.85)';
        if (label.startsWith('Exit · Stop Loss')) return 'rgba(255, 64, 90, 0.85)';
        if (label.startsWith('Exit · Time-Out')) return 'rgba(255, 200, 0, 0.85)';
        if (label.startsWith('Exit · Breakeven')) return 'rgba(180, 180, 180, 0.85)';
        return 'rgba(140, 140, 140, 0.85)';
      });

      return {
        nodes: nodeSet.map((label) => label.replace(/^[^·]+·\s*/, '')),
        nodeColors,
        sources: links.map((l) => l.source),
        targets: links.map((l) => l.target),
        values: links.map((l) => l.count),
        labels: links.map((l) => `count: ${l.count} · net PnL: $${l.pnl.toFixed(2)}`),
        linkColors,
      };
    }, [trades]);

    useEffect(() => {
      if (!ref.current) return;
      if (!sankeyData) { try { Plotly.purge(ref.current); } catch { /* noop */ } return; }

      const data: any[] = [{
        type: 'sankey',
        orientation: 'h',
        valueformat: 'd',
        node: {
          pad: 14,
          thickness: 18,
          line: { color: 'rgba(255,255,255,0.15)', width: 0.5 },
          label: sankeyData.nodes,
          color: sankeyData.nodeColors,
        },
        link: {
          source: sankeyData.sources,
          target: sankeyData.targets,
          value: sankeyData.values,
          color: sankeyData.linkColors,
          customdata: sankeyData.labels,
          hovertemplate: '%{source.label} → %{target.label}<br>%{customdata}<extra></extra>',
        },
      }];

      const layout: any = {
        autosize: true,
        margin: { l: 8, r: 8, t: 16, b: 8 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#d6d6d6', family: 'JetBrains Mono, monospace', size: 11 },
        hoverlabel: { bgcolor: '#1a1a1a', bordercolor: 'rgba(255,255,255,0.2)' },
      };

      const config: any = { displayModeBar: false, responsive: true };
      Plotly.newPlot(ref.current, data, layout, config);
      const onResize = () => { if (ref.current) Plotly.Plots.resize(ref.current); };
      window.addEventListener('resize', onResize);
      return () => {
        window.removeEventListener('resize', onResize);
        if (ref.current) try { Plotly.purge(ref.current); } catch { /* noop */ }
      };
    }, [sankeyData]);

    return (
      <div className="lifecycle-chart-wrap">
        {!sankeyData ? (
          <div className="lifecycle-empty">No trades to flow yet.</div>
        ) : (
          <div ref={ref} style={{ width: '100%', height: '100%' }} />
        )}
      </div>
    );
  };

  const renderJournal = () => {
    const ohlcBars: FilterlessOhlcBar[] = (state.bot as any).price_history_ohlc || [];
    const journals = (state as any).daily_journals as Array<any> | undefined;
    const fmtDailyDate = (iso?: string | null): string => {
      if (!iso) return '—';
      try {
        return new Date(iso + 'T12:00:00Z').toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: '2-digit', timeZone: 'America/New_York' });
      } catch { return iso; }
    };
    const fmtMoney = (v?: number | null): string => {
      if (v == null || Number.isNaN(v)) return '—';
      const sign = v >= 0 ? '+' : '−';
      return `${sign}$${Math.abs(v).toFixed(2)}`;
    };

    return (
    <section className="screen">
      {/* Daily Summary panel — hydrated from ai_loop_data/journals/*.json by the bridge */}
      <Panel
        title="Daily Summary"
        titleClassName="display-title"
        subtitle="End-of-day journal — most recent days first."
        badge={<Badge tone={journals && journals.length ? 'live' : 'watch'}>{journals && journals.length ? `${journals.length} days` : 'no data'}</Badge>}
      >
        <div className="daily-journal-grid">
          {journals && journals.length > 0 ? journals.slice(0, 7).map((j: any, i: number) => {
            const s = j.summary || {};
            const pnl = s.total_pnl ?? 0;
            const wr = s.win_rate ?? 0;
            const pc = j.price_context || {};
            const flags: string[] = j.pattern_flags || [];
            const tone: BadgeTone = pnl > 0 ? 'live' : pnl < 0 ? 'block' : 'watch';
            return (
              <div key={`${j.date}-${i}`} className="daily-journal-card" style={{ borderTop: `2px solid ${pnl > 0 ? COLORS.green : pnl < 0 ? COLORS.red : COLORS.purple}` }}>
                <div className="daily-journal-head">
                  <div>
                    <div className="daily-journal-date">{fmtDailyDate(j.date)}</div>
                    <div className="micro" style={{ opacity: 0.55 }}>{j.date}</div>
                  </div>
                  <Badge tone={tone}>{pnl > 0 ? 'WIN DAY' : pnl < 0 ? 'LOSS DAY' : 'FLAT'}</Badge>
                </div>
                <div className="daily-journal-pnl" style={{ color: pnl >= 0 ? COLORS.green : COLORS.red }}>
                  {fmtMoney(pnl)}
                </div>
                <div className="daily-journal-stats">
                  <div className="daily-journal-stat"><span className="micro">trades</span><strong>{s.n_trades ?? '—'}</strong></div>
                  <div className="daily-journal-stat"><span className="micro">WR</span><strong>{wr != null ? `${wr.toFixed(1)}%` : '—'}</strong></div>
                  <div className="daily-journal-stat"><span className="micro">W/L</span><strong>{(s.n_wins ?? 0)}/{(s.n_losses ?? 0)}</strong></div>
                  <div className="daily-journal-stat"><span className="micro">max DD</span><strong>{s.max_drawdown != null ? `$${s.max_drawdown.toFixed(0)}` : '—'}</strong></div>
                  <div className="daily-journal-stat"><span className="micro">signals</span><strong>{s.n_signals_fired ?? '—'}</strong></div>
                  <div className="daily-journal-stat"><span className="micro">kalshi blocks</span><strong>{s.n_kalshi_blocks ?? '—'}</strong></div>
                </div>
                {pc && (pc.range_pts != null || pc.trend_pts != null) && (
                  <div className="daily-journal-context micro">
                    range {pc.range_pts != null ? `${pc.range_pts.toFixed(1)} pts` : '—'}
                    {pc.trend_pts != null ? ` · trend ${pc.trend_pts >= 0 ? '+' : ''}${pc.trend_pts.toFixed(1)} pts ${pc.trend_dir || ''}` : ''}
                    {pc.open != null && pc.close != null ? ` · ${pc.open.toFixed(2)} → ${pc.close.toFixed(2)}` : ''}
                  </div>
                )}
                {flags.length > 0 && (
                  <div className="daily-journal-flags">
                    {flags.slice(0, 3).map((f, fi) => (
                      <div key={fi} className="daily-journal-flag micro">{f.replace(/^[⚠✅⚑]\s*/, '⚠ ')}</div>
                    ))}
                  </div>
                )}
              </div>
            );
          }) : (
            <p className="micro" style={{ padding: '8px 0' }}>
              No daily journals available yet. Bridge reads from <code>ai_loop_data/journals/YYYY-MM-DD.json</code> — run <code>python3 -m tools.ai_loop.run_daily</code> nightly (or set up cron) to generate.
            </p>
          )}
        </div>
      </Panel>

      {/* === MFE/MAE Efficiency Scatter — every closed trade as a coordinate === */}
      <Panel
        title="MFE / MAE Efficiency"
        titleClassName="display-title"
        subtitle="Each trade plotted by max favorable excursion (Y) vs max adverse excursion (X). Below the diagonal = wasted edge. Wins green, losses red."
        badge={<Badge tone={state.trades.length ? 'live' : 'watch'}>{state.trades.length ? `${state.trades.length} trades` : 'no data'}</Badge>}
        className="mt-panel"
      >
        <div className="panel-body">
          <MfeMaeScatterChart trades={state.trades || []} bars={ohlcBars} />
        </div>
      </Panel>

      {/* === Trade Lifecycle Sankey — Signal → Entry → Exit flow attribution === */}
      <Panel
        title="Trade Lifecycle Flow"
        titleClassName="display-title"
        subtitle="Signal source → entry type → exit reason. Width = trade count, color = exit tone (green TP, red SL, yellow time-out)."
        badge={<Badge tone={state.trades.length ? 'live' : 'watch'}>{state.trades.length ? `${state.trades.length} trades` : 'no data'}</Badge>}
        className="mt-panel"
      >
        <div className="panel-body">
          <TradeLifecycleSankey trades={state.trades || []} />
        </div>
      </Panel>

      <div className="grid journal-layout mt-panel">
        <Panel title="Trace Log" titleClassName="display-title" subtitle="Decision journal with trade levels, news, and manifold snapshots." badge={<Badge tone="info">live</Badge>}>
          <div className="terminal">
            {state.events.length ? state.events.slice(0, 28).map((event: FilterlessEvent, index) => (
              <TerminalRow key={`${event.event_type}-${event.time}-${index}`} time={event.time} title={event.event_type} text={event.message} badge={<Badge tone={event.severity === 'error' || event.severity === 'danger' ? 'block' : event.severity === 'warning' ? 'watch' : 'info'}>{event.severity}</Badge>} />
            )) : <TerminalRow title="WAIT" text="No filterless events have been bridged yet." badge={<Badge tone="watch">idle</Badge>} />}
          </div>
        </Panel>
        <div className="stack">
          <Panel title="State Summary" titleClassName="display-title" subtitle="Latest engine snapshot.">
            <div className="panel-body">
              <Meter label="regime" value={1} text={features.regime} color={COLORS.purple} />
              <Meter label="risk mult" value={features.riskMult / 1.5} text={`${fmt(features.riskMult)}x`} color={COLORS.lime} />
              <Meter label="truth score" value={(features.truthScore + 1) / 2} text={formatSigned(features.truthScore)} color={COLORS.pink} />
            </div>
          </Panel>
          <Panel title="Trade Blotter" titleClassName="display-title" subtitle="Recent closed filterless trades.">
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
  };

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
      <Panel title="Operator Controls" subtitle="Live runtime actions for the bot loop, bridge, UI feed, and operator journal." badge={<Badge tone={controlOnline ? 'live' : 'watch'}>{controlOnline ? 'control online' : 'local only'}</Badge>} className="mt-panel">
        <div className="panel-body">
          <div className="command-grid">
            <button className="command-tile control-tile" type="button" onClick={forceFeedRefresh}>
              <strong className="truncate">Refresh Feed</strong>
              <small className="truncate">{error ? 'endpoint check' : formatRelativeTime(state.generated_at)}</small>
              <Badge tone={error ? 'block' : statusBadge(effectiveStatus)}>fetch</Badge>
            </button>
            <button className="command-tile control-tile" type="button" onClick={() => setPollingPaused((value) => !value)}>
              <strong className="truncate">{pollingPaused ? 'Resume Polling' : 'Pause Polling'}</strong>
              <small className="truncate">{pollingPaused ? 'manual refresh only' : `${Math.round(REFRESH_MS / 1000)}s dashboard cadence`}</small>
              <Badge tone={pollingPaused ? 'watch' : 'live'}>{pollingPaused ? 'paused' : 'live'}</Badge>
            </button>
            <button className="command-tile control-tile" type="button" disabled={!controlOnline} onClick={() => void sendOperatorCommand('restart_bridge', 'Bridge restart')}>
              <strong className="truncate">Restart Bridge</strong>
              <small className="truncate">dashboard JSON writer</small>
              <Badge tone={processTone(bridgeProcess)}>{bridgeProcess?.running ? 'running' : controlOnline ? 'down' : 'offline'}</Badge>
            </button>
            <button className="command-tile control-tile" type="button" disabled={!controlOnline} onClick={() => void sendOperatorCommand('restart_bot', 'Bot restart')}>
              <strong className="truncate">Restart Bot</strong>
              <small className="truncate">{openPositions.length ? `${openPositions.length} position${openPositions.length === 1 ? '' : 's'} active` : 'flat runtime'}</small>
              <Badge tone={openPositions.length ? 'watch' : processTone(botProcess)}>{botProcess?.running ? 'loop' : controlOnline ? 'down' : 'offline'}</Badge>
            </button>
            <button className="command-tile control-tile" type="button" disabled={!controlOnline} onClick={() => void sendOperatorCommand('restart_frontend', 'UI restart')}>
              <strong className="truncate">Restart UI</strong>
              <small className="truncate">live cockpit server</small>
              <Badge tone={processTone(frontendProcess)}>{frontendProcess?.running ? 'serving' : controlOnline ? 'down' : 'offline'}</Badge>
            </button>
            <button className="command-tile control-tile" type="button" onClick={() => setActiveScreen('kalshi')}>
              <strong className="truncate">Kalshi Gate</strong>
              <small className="truncate">{kalshi?.status_reason || kalshi?.event_ticker || 'no ladder'}</small>
              <Badge tone={kalshi?.healthy ? 'live' : 'watch'}>{kalshi?.healthy ? 'online' : 'review'}</Badge>
            </button>
            <button className="command-tile control-tile" type="button" onClick={() => setActiveScreen('news')}>
              <strong className="truncate">Truth Monitor</strong>
              <small className="truncate">{sentiment.trigger_reason || sentiment.sentiment_label || 'neutral'}</small>
              <Badge tone={sentiment.last_error ? 'block' : features.truthRiskWatch ? 'watch' : 'info'}>{sentiment.last_error ? 'issue' : features.truthRiskWatch ? 'watch' : 'clear'}</Badge>
            </button>
            <button className="command-tile control-tile" type="button" onClick={() => setActiveScreen(primaryPosition ? 'overview' : 'journal')}>
              <strong className="truncate">Position Review</strong>
              <small className="truncate">{primaryPosition ? `${primaryPosition.side} ${primaryPosition.size ?? '--'} @ ${formatPrice(entry)}` : 'flat journal'}</small>
              <Badge tone={primaryPosition ? 'watch' : 'info'}>{primaryPosition ? 'active' : 'flat'}</Badge>
            </button>
            <button className="command-tile control-tile" type="button" onClick={() => setActiveScreen('aetherflow')}>
              <strong className="truncate">Manifold Check</strong>
              <small className="truncate">{features.regime}</small>
              <Badge tone={features.noTrade ? 'block' : 'info'}>{features.noTrade ? 'lockout' : 'clear'}</Badge>
            </button>
            <button className="command-tile control-tile" type="button" onClick={() => void copyOperatorSnapshot()}>
              <strong className="truncate">Copy Snapshot</strong>
              <small className="truncate">state, risk, gates, events</small>
              <Badge tone="info">journal</Badge>
            </button>
            <button className="command-tile control-tile" type="button" onClick={() => setActiveScreen('docs')}>
              <strong className="truncate">Documentation</strong>
              <small className="truncate">README.pdf · architecture + recent changes</small>
              <Badge tone="info">read</Badge>
            </button>
          </div>
          <div className="operator-footer">
            <div className="operator-note"><strong className="truncate">control</strong><span className="truncate">{controlOnline ? 'launcher API online' : controlError || 'launcher controls unavailable'}</span></div>
            <div className="operator-note"><strong className="truncate">state file</strong><span className="truncate">{formatAgeSeconds(controlStatus?.dashboard_state_age_seconds)} old</span></div>
            <div className="operator-note"><strong className="truncate">bot pid</strong><span className="truncate">{botProcess?.pid ?? '--'} / {botProcess?.running ? 'running' : 'stopped'}</span></div>
            <div className="operator-note"><strong className="truncate">last action</strong><span className="truncate">{controlMessage || 'none'}</span></div>
          </div>
        </div>
      </Panel>
    </section>
  );

  const renderDocs = () => (
    <section className="screen">
      <Panel
        title="Documentation"
        subtitle="Filterless live README — architecture, strategies, ML stack, and recent changes (Apr 26 → Apr 29)."
        badge={<Badge tone="info">README.pdf</Badge>}
      >
        <div className="panel-body docs-panel-body">
          <div className="docs-toolbar">
            <a className="docs-link" href="/README.pdf" target="_blank" rel="noopener noreferrer">
              Open in new tab
            </a>
            <a className="docs-link" href="/README.pdf" download="JULIE001-README.pdf">
              Download
            </a>
            <span className="docs-hint truncate">
              Source: README.md · regenerated by tools/render_readme_pdf.py
            </span>
          </div>
          <div className="docs-viewer">
            <object
              data="/README.pdf#zoom=page-width&navpanes=0"
              type="application/pdf"
              aria-label="Filterless live README"
            >
              <iframe
                src="/README.pdf"
                title="Filterless live README"
                className="docs-frame-fallback"
              />
              <div className="docs-fallback-note">
                Your browser cannot display the PDF inline.{' '}
                <a href="/README.pdf" target="_blank" rel="noopener noreferrer">
                  Open README.pdf in a new tab
                </a>
                .
              </div>
            </object>
          </div>
        </div>
      </Panel>
    </section>
  );

  const renderScreen = () => {
    if (activeScreen === 'aetherflow') return renderAetherflow();
    if (activeScreen === 'kalshi') return renderKalshi();
    if (activeScreen === 'news') return renderNews();
    if (activeScreen === 'strategies') return renderStrategies();
    if (activeScreen === 'pipeline') return renderPipeline();
    if (activeScreen === 'journal') return renderJournal();
    if (activeScreen === 'command') return renderCommand();
    if (activeScreen === 'docs') return renderDocs();
    return renderOverview();
  };

  return (
    <div className="fl-cockpit">
      <style>{`${COCKPIT_CSS}.mt-panel{margin-top:10px;}`}</style>
      {/* Looping video background — sits behind everything via z-index.
          The .mov is the active backdrop; bg.mp4 stays as a fallback for
          any browser that can't decode the QuickTime H.264. */}
      <video
        className="bg-video"
        autoPlay
        muted
        loop
        playsInline
        preload="auto"
        aria-hidden="true"
      >
        <source src="/bg.mov" type="video/quicktime" />
        <source src="/bg.mov" type="video/mp4" />
        <source src="/bg.mp4" type="video/mp4" />
      </video>
      <div className="app">
        <aside className="rail">
          <div className="brand">
            <h1 className="truncate">JULIE</h1>
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
          {/* Phone-only sleek dropdown — replaces the button grid at <=480px */}
          <div className="nav-select" aria-label="Live screens (phone)">
            <select
              value={activeScreen}
              onChange={(e) => setActiveScreen(e.target.value as ScreenId)}
            >
              {NAV.map((item) => (
                <option key={item.id} value={item.id}>
                  {item.code} · {item.label}
                </option>
              ))}
            </select>
            <span className="nav-select-caret" aria-hidden="true">&#x25BE;</span>
          </div>
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
              <button className="command primary" type="button" onClick={() => setActiveScreen('command')}>Command Center</button>
            </div>
          </header>

          {error ? <div className="notice">Dashboard feed endpoint error: {error}</div> : null}
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
      <BouncingHead />
    </div>
  );
}

export default FilterlessLiveCockpit;
