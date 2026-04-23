/**
 * TriathlonTab — live view of the Triathlon Engine's current cell medals.
 *
 * Reads `/triathlon_state.json` on the same cadence as FilterlessLiveApp
 * reads the filterless state. Renders:
 *   - a headline strip with medal tallies + totals
 *   - a table of cells sorted by medal (gold → probation → unrated)
 *   - a recent-signals section (last 50 live signals with block reasons)
 *   - the retrain queue (if non-empty)
 *
 * Zero third-party deps beyond what the dashboard already bundles.
 */
import * as React from 'react';
import { useEffect, useMemo, useState } from 'react';

type CellRow = {
  cell_key: string;
  strategy: string;
  regime: string;
  time_bucket: string;
  medal: 'gold' | 'silver' | 'bronze' | 'probation' | 'unrated';
  n_signals: number;
  n_fired: number | null;
  n_blocked: number | null;
  purity: number | null;
  cash: number | null;
  velocity: number | null;
  purity_rank: number | null;
  cash_rank: number | null;
  velocity_rank: number | null;
  scored_at: string;
};

type RecentSignal = {
  signal_id: string;
  ts: string;
  strategy: string;
  side: string;
  regime: string;
  time_bucket: string;
  status: string;
  block_filter: string | null;
  block_reason: string | null;
  entry_price: number | null;
  size: number | null;
  pnl_dollars: number | null;
  exit_source: string | null;
  counterfactual: number | null;
};

type RetrainEntry = {
  cell_key: string;
  queued_at: string;
  reason: string;
  status: string;
};

type TriathlonState = {
  generated_at: string;
  n_cells_total: number;
  n_cells_rated: number;
  medal_counts: Record<string, number>;
  totals: { n_fired_all: number; n_blocked_all: number; retrain_queue_open: number };
  cells: CellRow[];
  recent_signals: RecentSignal[];
  retrain_queue: RetrainEntry[];
};

const MEDAL_COLORS: Record<string, string> = {
  gold:      '#f4b400',
  silver:    '#c0c0c0',
  bronze:    '#cd7f32',
  probation: '#d93025',
  unrated:   '#6c6c6c',
};

const MEDAL_ORDER = ['gold', 'silver', 'bronze', 'probation', 'unrated'];

const POLL_INTERVAL_MS = 10_000;

export function TriathlonTab(): React.ReactElement {
  const [state, setState] = useState<TriathlonState | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [filterMedal, setFilterMedal] = useState<string>('all');

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const resp = await fetch(`/triathlon_state.json?ts=${Date.now()}`, {
          cache: 'no-store',
          headers: { 'ngrok-skip-browser-warning': 'true' },
        });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const payload = (await resp.json()) as TriathlonState;
        if (!cancelled) {
          setState(payload);
          setError(null);
          setLoading(false);
        }
      } catch (err: any) {
        if (!cancelled) {
          setError(err?.message ?? 'fetch failed');
          setLoading(false);
        }
      }
    };
    load();
    const id = window.setInterval(load, POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, []);

  const visibleCells = useMemo(() => {
    if (!state) return [];
    if (filterMedal === 'all') return state.cells;
    return state.cells.filter((c) => c.medal === filterMedal);
  }, [state, filterMedal]);

  if (loading) {
    return <div style={{ padding: 16, color: '#bbb' }}>Loading Triathlon state…</div>;
  }
  if (error || !state) {
    return (
      <div style={{ padding: 16, color: '#d93025' }}>
        Unable to load <code>/triathlon_state.json</code>: {error ?? 'no data'}.
        Run <code>python3 -m triathlon export</code> or <code>python3 -m triathlon rescore</code>{' '}
        to generate it.
      </div>
    );
  }

  return (
    <div style={{ padding: 16, color: '#ddd', fontFamily: 'system-ui, sans-serif' }}>
      <h2 style={{ marginTop: 0, color: '#fff' }}>Triathlon Engine</h2>
      <div style={{ marginBottom: 8, fontSize: 12, color: '#888' }}>
        generated {new Date(state.generated_at).toLocaleString()} ·{' '}
        {state.n_cells_rated} / {state.n_cells_total} cells rated ·{' '}
        {state.totals.n_fired_all.toLocaleString()} fired ·{' '}
        {state.totals.n_blocked_all.toLocaleString()} blocked
      </div>

      {/* Medal tally strip */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 16 }}>
        <button
          onClick={() => setFilterMedal('all')}
          style={{
            padding: '6px 12px',
            background: filterMedal === 'all' ? '#444' : '#222',
            color: '#fff',
            border: '1px solid #555',
            borderRadius: 4,
            cursor: 'pointer',
          }}
        >
          ALL ({state.cells.length})
        </button>
        {MEDAL_ORDER.map((medal) => {
          const n = state.medal_counts[medal] ?? 0;
          if (n === 0) return null;
          const bg = filterMedal === medal ? MEDAL_COLORS[medal] : '#222';
          return (
            <button
              key={medal}
              onClick={() => setFilterMedal(medal)}
              style={{
                padding: '6px 12px',
                background: bg,
                color: filterMedal === medal ? '#000' : '#fff',
                border: `2px solid ${MEDAL_COLORS[medal]}`,
                borderRadius: 4,
                cursor: 'pointer',
                textTransform: 'uppercase',
                fontWeight: 700,
                fontSize: 12,
              }}
            >
              {medal} ({n})
            </button>
          );
        })}
      </div>

      {/* Cell table */}
      <div style={{ overflowX: 'auto', marginBottom: 24 }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ borderBottom: '1px solid #444', textAlign: 'left' }}>
              <th style={thStyle}>medal</th>
              <th style={thStyle}>strategy</th>
              <th style={thStyle}>regime</th>
              <th style={thStyle}>bucket</th>
              <th style={{ ...thStyle, textAlign: 'right' }}>n</th>
              <th style={{ ...thStyle, textAlign: 'right' }}>purity</th>
              <th style={{ ...thStyle, textAlign: 'right' }}>cash $</th>
              <th style={{ ...thStyle, textAlign: 'right' }}>velocity</th>
              <th style={{ ...thStyle, textAlign: 'right' }}>ranks (P/C/V)</th>
            </tr>
          </thead>
          <tbody>
            {visibleCells.map((c) => (
              <tr key={c.cell_key} style={{ borderBottom: '1px solid #2a2a2a' }}>
                <td style={tdStyle}>
                  <span
                    style={{
                      display: 'inline-block',
                      padding: '2px 8px',
                      background: MEDAL_COLORS[c.medal],
                      color:
                        c.medal === 'gold' || c.medal === 'silver' ? '#000' : '#fff',
                      fontSize: 11,
                      fontWeight: 700,
                      textTransform: 'uppercase',
                      borderRadius: 3,
                    }}
                  >
                    {c.medal}
                  </span>
                </td>
                <td style={tdStyle}>{c.strategy}</td>
                <td style={tdStyle}>{c.regime}</td>
                <td style={tdStyle}>{c.time_bucket}</td>
                <td style={{ ...tdStyle, textAlign: 'right' }}>{c.n_signals}</td>
                <td style={{ ...tdStyle, textAlign: 'right' }}>
                  {c.purity !== null ? (c.purity * 100).toFixed(1) + '%' : '—'}
                </td>
                <td style={{ ...tdStyle, textAlign: 'right',
                  color: c.cash !== null ? (c.cash >= 0 ? '#7ecb70' : '#ff6e6e') : undefined }}>
                  {c.cash !== null ? (c.cash >= 0 ? '+' : '') + c.cash.toFixed(2) : '—'}
                </td>
                <td style={{ ...tdStyle, textAlign: 'right' }}>
                  {c.velocity !== null ? c.velocity.toFixed(4) : '—'}
                </td>
                <td style={{ ...tdStyle, textAlign: 'right', color: '#888', fontSize: 11 }}>
                  {c.purity_rank ?? '—'} / {c.cash_rank ?? '—'} / {c.velocity_rank ?? '—'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Recent signals */}
      <h3 style={{ color: '#fff', marginTop: 8 }}>
        Recent live signals ({state.recent_signals.length})
      </h3>
      {state.recent_signals.length === 0 ? (
        <div style={{ fontSize: 13, color: '#888' }}>
          No live signals recorded yet. The engine begins recording when the bot
          (with <code>JULIE_TRIATHLON_ACTIVE=1</code>) emits its next signal.
        </div>
      ) : (
        <div style={{ overflowX: 'auto', marginBottom: 24 }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ borderBottom: '1px solid #444' }}>
                <th style={thStyle}>ts</th>
                <th style={thStyle}>status</th>
                <th style={thStyle}>strat</th>
                <th style={thStyle}>side</th>
                <th style={thStyle}>regime</th>
                <th style={thStyle}>bucket</th>
                <th style={{ ...thStyle, textAlign: 'right' }}>entry</th>
                <th style={{ ...thStyle, textAlign: 'right' }}>size</th>
                <th style={{ ...thStyle, textAlign: 'right' }}>pnl $</th>
                <th style={thStyle}>block reason</th>
              </tr>
            </thead>
            <tbody>
              {state.recent_signals.map((r) => (
                <tr key={r.signal_id} style={{ borderBottom: '1px solid #2a2a2a' }}>
                  <td style={tdStyle}>{new Date(r.ts).toLocaleTimeString()}</td>
                  <td style={{ ...tdStyle,
                    color: r.status === 'fired' ? '#7ecb70'
                         : r.status === 'blocked' ? '#ff6e6e' : '#888' }}>
                    {r.status}
                  </td>
                  <td style={tdStyle}>{r.strategy}</td>
                  <td style={tdStyle}>{r.side}</td>
                  <td style={tdStyle}>{r.regime}</td>
                  <td style={tdStyle}>{r.time_bucket}</td>
                  <td style={{ ...tdStyle, textAlign: 'right' }}>
                    {r.entry_price?.toFixed(2) ?? '—'}
                  </td>
                  <td style={{ ...tdStyle, textAlign: 'right' }}>{r.size ?? '—'}</td>
                  <td style={{ ...tdStyle, textAlign: 'right',
                    color: r.pnl_dollars !== null ?
                      (r.pnl_dollars >= 0 ? '#7ecb70' : '#ff6e6e') : undefined }}>
                    {r.pnl_dollars !== null
                      ? (r.pnl_dollars >= 0 ? '+' : '') + r.pnl_dollars.toFixed(2)
                      : '—'}
                  </td>
                  <td style={{ ...tdStyle, color: '#888', fontSize: 11 }}>
                    {r.block_filter ? `${r.block_filter}: ${r.block_reason ?? ''}` : ''}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Retrain queue */}
      {state.retrain_queue.length > 0 && (
        <div>
          <h3 style={{ color: '#fff' }}>Retrain queue ({state.retrain_queue.length})</h3>
          <ul style={{ fontSize: 13 }}>
            {state.retrain_queue.map((r) => (
              <li key={r.cell_key + r.queued_at}>
                <strong>{r.cell_key}</strong> — {r.reason} ({r.status},{' '}
                {new Date(r.queued_at).toLocaleString()})
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

const thStyle: React.CSSProperties = {
  padding: '6px 10px',
  color: '#999',
  fontWeight: 600,
  fontSize: 11,
  textTransform: 'uppercase',
};

const tdStyle: React.CSSProperties = {
  padding: '6px 10px',
  color: '#ddd',
};

export default TriathlonTab;
