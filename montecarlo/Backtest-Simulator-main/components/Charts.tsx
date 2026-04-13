import React, { useMemo } from 'react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  BarChart, Bar, TooltipProps, ScatterChart, Scatter, Cell, ReferenceLine, AreaChart, Area, Legend, ComposedChart
} from 'recharts';
import { SimulationResult, Trade } from '../types';
import { calculateRollingStats, calculateDistributionStats } from '../utils/analytics';
import { AlertCircle } from 'lucide-react';

interface BaseChartProps {
    className?: string;
}

interface EquityChartProps extends BaseChartProps {
  results: SimulationResult[];
  limitLines?: number;
}

const EquityCustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    const values = payload.map((p: any) => p.value as number);
    const min = Math.min(...values);
    const max = Math.max(...values);
    const avg = values.reduce((a: number, b: number) => a + b, 0) / values.length;

    return (
      <div className="bg-surface border border-neutral-800 p-3 rounded shadow-lg text-sm">
        <p className="font-semibold text-neutral-200 mb-2">Trade #{label}</p>
        <div className="space-y-1">
          <p className="text-blue-400">Avg Equity: ${Math.round(avg).toLocaleString()}</p>
          <p className="text-emerald-400">Best: ${Math.round(max).toLocaleString()}</p>
          <p className="text-rose-400">Worst: ${Math.round(min).toLocaleString()}</p>
        </div>
      </div>
    );
  }
  return null;
};

export const EquityChart: React.FC<EquityChartProps> = ({ results, limitLines = 50, className }) => {
  const displayResults = results.slice(0, limitLines);
  
  const data = useMemo(() => {
    if (displayResults.length === 0) return [];
    
    const maxLength = Math.max(...displayResults.map(r => r.equityCurve.length));
    const chartData = [];

    for (let i = 0; i < maxLength; i++) {
      const point: any = { name: i };
      displayResults.forEach((r, idx) => {
        if (i < r.equityCurve.length) {
          point[`sim${idx}`] = r.equityCurve[i];
        }
      });
      chartData.push(point);
    }
    return chartData;
  }, [displayResults]);

  if (results.length === 0) return <div className={`flex items-center justify-center text-neutral-500 ${className || 'h-64'}`}>No Data</div>;

  return (
    <div className={`w-full ${className || 'h-80'}`}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#333333" vertical={false} />
          <XAxis 
            dataKey="name" 
            stroke="#666666" 
            tickLine={false} 
            tick={{fontSize: 12}}
            label={{ value: 'Trade Sequence', position: 'insideBottom', offset: -5, fill: '#666666' }}
          />
          <YAxis 
            stroke="#666666" 
            tickLine={false}
            tickFormatter={(val) => `$${val / 1000}k`}
            tick={{fontSize: 12}}
            width={60}
          />
          <Tooltip content={<EquityCustomTooltip />} />
          {displayResults.map((_, idx) => (
            <Line 
              key={idx}
              type="monotone" 
              dataKey={`sim${idx}`} 
              stroke="#ffffff" 
              strokeWidth={1}
              dot={false}
              strokeOpacity={0.4}
              activeDot={{ r: 4, stroke: '#fff', strokeWidth: 2 }}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

interface DistributionChartProps extends BaseChartProps {
  data: number[];
  title: string;
  unit?: string;
  color?: string;
  bins?: number;
}

const DistributionCustomTooltip = ({ active, payload }: any) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    return (
      <div className="bg-surface border border-neutral-800 p-3 rounded shadow-lg text-sm z-50">
        <p className="font-semibold text-neutral-200 mb-1">Range</p>
        <p className="text-neutral-300 mb-2">{data.rangeLabel}</p>
        <p className="text-primary font-bold">Count: {data.count}</p>
        <p className="text-neutral-500 text-xs">{(data.percentage || 0).toFixed(1)}% of simulations</p>
      </div>
    );
  }
  return null;
};

export const DistributionChart: React.FC<DistributionChartProps> = ({ 
  data, 
  title, 
  unit = '', 
  color = '#10b981',
  bins: binCount = 20,
  className
}) => {
  const chartData = useMemo(() => {
    if (data.length === 0) return [];
    
    const min = Math.min(...data);
    const max = Math.max(...data);
    const step = (max - min) / binCount || 1; 
    
    const bins = Array.from({ length: binCount }, (_, i) => {
      const start = min + i * step;
      const end = min + (i + 1) * step;
      return {
        rangeStart: start,
        rangeEnd: end,
        rangeLabel: `${unit === '$' ? '$' : ''}${start.toLocaleString(undefined, {maximumFractionDigits:1})}${unit !== '$' ? unit : ''} - ${unit === '$' ? '$' : ''}${end.toLocaleString(undefined, {maximumFractionDigits:1})}${unit !== '$' ? unit : ''}`,
        count: 0,
        label: `${(start + step/2).toFixed(unit === '$' ? 0 : 1)}`, 
        percentage: 0
      };
    });

    data.forEach(val => {
      const binIndex = Math.min(
        Math.floor((val - min) / step), 
        binCount - 1
      );
      if (binIndex >= 0) bins[binIndex].count++;
    });

    bins.forEach(b => b.percentage = (b.count / data.length) * 100);

    return bins;
  }, [data, unit, binCount]);

  if (data.length === 0) return <div className={`flex items-center justify-center text-neutral-500 ${className || 'h-64'}`}>No Data</div>;

  return (
    <div className={`w-full bg-surface border border-neutral-800 rounded-xl p-4 flex flex-col ${className || 'h-80'}`}>
      <h4 className="text-neutral-300 text-sm font-semibold mb-4 text-center">{title}</h4>
      <div className="flex-1 w-full min-h-0">
        <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#333333" />
            <XAxis 
                dataKey="label" 
                stroke="#666666" 
                tick={{fontSize: 10}}
                interval="preserveStartEnd"
            />
            <YAxis stroke="#666666" tick={{fontSize: 10}} />
            <Tooltip content={<DistributionCustomTooltip />} cursor={{fill: '#333333', opacity: 0.4}} />
            <Bar dataKey="count" fill={color} radius={[4, 4, 0, 0]} />
            </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export const VaRCurveChart: React.FC<{ drawdowns: number[], className?: string }> = ({ drawdowns, className }) => {
    const data = useMemo(() => {
        if (!drawdowns.length) return [];
        const sorted = [...drawdowns].sort((a, b) => a - b);
        const points = [];
        // Plot from 50% to 100% confidence
        for (let i = 50; i <= 99; i++) {
            const index = Math.floor((i / 100) * sorted.length);
            points.push({
                confidence: i,
                drawdown: sorted[Math.min(index, sorted.length - 1)]
            });
        }
        return points;
    }, [drawdowns]);

    return (
        <div className={`w-full bg-surface border border-neutral-800 rounded-xl p-4 flex flex-col ${className || 'h-80'}`}>
             <h4 className="text-neutral-300 text-sm font-semibold mb-4 text-center">Value at Risk (VaR) Curve</h4>
             <div className="flex-1 w-full min-h-0">
                <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={data}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#333333" />
                        <XAxis 
                            dataKey="confidence" 
                            label={{ value: 'Confidence Level (%)', position: 'insideBottom', offset: -5, fill: '#666666' }}
                            stroke="#666666"
                            type="number"
                            domain={[50, 100]}
                        />
                        <YAxis 
                            label={{ value: 'Max Drawdown (%)', angle: -90, position: 'insideLeft', fill: '#666666' }}
                            stroke="#666666"
                        />
                        <Tooltip 
                            contentStyle={{ backgroundColor: '#0a0a0a', borderColor: '#333' }}
                            itemStyle={{ color: '#fbbf24' }}
                            formatter={(val: number) => [`${val.toFixed(2)}%`, 'Max Drawdown']}
                            labelFormatter={(label) => `Confidence: ${label}%`}
                        />
                        <Line type="monotone" dataKey="drawdown" stroke="#fbbf24" strokeWidth={2} dot={false} />
                    </LineChart>
                </ResponsiveContainer>
             </div>
        </div>
    );
};

export const ScatterPnLDuration: React.FC<{ trades: Trade[], className?: string }> = ({ trades, className }) => {
    const data = useMemo(() => {
        return trades.map((t, i) => {
             const start = new Date(t.entryTime).getTime();
             const end = new Date(t.exitTime).getTime();
             let durationHours = 0;
             if (!isNaN(start) && !isNaN(end)) {
                 durationHours = (end - start) / (1000 * 60 * 60);
             }
             if (durationHours < 0) return null; 
             
             return { x: durationHours, y: t.pnl, id: t.id, type: t.type };
        }).filter(d => d !== null) as any[];
    }, [trades]);

    return (
        <div className={`w-full bg-surface border border-neutral-800 rounded-xl p-4 flex flex-col justify-center ${className || 'h-80'}`}>
             <h4 className="text-neutral-300 text-sm font-semibold mb-4 text-center">PnL vs Duration</h4>
             {data.length > 0 ? (
                 <div className="flex-1 w-full min-h-0">
                    <ResponsiveContainer width="100%" height="100%">
                        <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#333333" />
                            <XAxis 
                                type="number" 
                                dataKey="x" 
                                name="Duration" 
                                unit="h" 
                                stroke="#666666"
                                label={{ value: 'Duration (Hours)', position: 'insideBottom', offset: -10, fill: '#666666' }}
                            />
                            <YAxis 
                                type="number" 
                                dataKey="y" 
                                name="PnL" 
                                unit="$" 
                                stroke="#666666"
                            />
                            <Tooltip 
                                cursor={{ strokeDasharray: '3 3' }}
                                contentStyle={{ backgroundColor: '#0a0a0a', borderColor: '#333' }}
                                formatter={(value: any, name: any) => [name === 'PnL' ? `$${value}` : `${parseFloat(value).toFixed(2)}h`, name]}
                            />
                            <Scatter name="Trades" data={data} fill="#8884d8">
                                {data.map((entry, index) => (
                                    <Cell key={`cell-${index}`} fill={entry.y >= 0 ? '#10b981' : '#ef4444'} />
                                ))}
                            </Scatter>
                        </ScatterChart>
                    </ResponsiveContainer>
                 </div>
             ) : (
                <div className="flex flex-col items-center justify-center text-neutral-500 h-full">
                    <AlertCircle className="w-8 h-8 mb-2 opacity-50" />
                    <p className="text-sm">Duration data unavailable</p>
                    <p className="text-xs mt-1">Check Entry/Exit Time cols</p>
                </div>
             )}
        </div>
    );
};

export const HistoricalEquityChart: React.FC<{ trades: Trade[], className?: string }> = ({ trades, className }) => {
    const data = useMemo(() => {
        let cumulative = 0;
        return trades.map((t, i) => {
            cumulative += t.pnl;
            return {
                index: i + 1,
                equity: cumulative,
                pnl: t.pnl
            };
        });
    }, [trades]);

    return (
        <div className={`w-full bg-surface border border-neutral-800 rounded-xl p-4 flex flex-col ${className || 'h-80'}`}>
            <h4 className="text-neutral-300 text-sm font-semibold mb-4 text-center">Realized Historical Equity</h4>
            <div className="flex-1 w-full min-h-0">
                <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={data}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#333333" />
                        <XAxis 
                            dataKey="index" 
                            stroke="#666666"
                            tick={{fontSize: 10}}
                        />
                        <YAxis 
                            stroke="#666666"
                            tick={{fontSize: 10}}
                            tickFormatter={(val) => `$${val}`}
                        />
                        <Tooltip 
                            contentStyle={{ backgroundColor: '#0a0a0a', borderColor: '#333' }}
                            formatter={(val: number) => [`$${val.toLocaleString()}`, 'Equity']}
                        />
                        <ReferenceLine y={0} stroke="#666" />
                        <Line type="monotone" dataKey="equity" stroke="#3b82f6" strokeWidth={2} dot={false} />
                    </LineChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
};

export const SkewKurtosisChart: React.FC<{ trades: Trade[], className?: string }> = ({ trades, className }) => {
  const { data, stats } = useMemo(() => {
    const pnls = trades.map(t => t.pnl);
    if(pnls.length === 0) return { data: [], stats: null };

    const stats = calculateDistributionStats(pnls);
    const min = Math.min(...pnls);
    const max = Math.max(...pnls);
    const binCount = 20;
    const step = (max - min) / binCount || 1;

    const bins = Array.from({ length: binCount }, (_, i) => {
      const start = min + i * step;
      const end = min + (i + 1) * step;
      return {
        rangeLabel: `$${Math.round(start)} - $${Math.round(end)}`,
        count: 0,
        mid: (start + end) / 2
      };
    });

    pnls.forEach(val => {
      const idx = Math.min(Math.floor((val - min) / step), binCount - 1);
      if(idx >= 0) bins[idx].count++;
    });

    return { data: bins, stats };
  }, [trades]);

  if (!stats) return <div className={`flex items-center justify-center ${className || 'h-64'}`}>No Data</div>;

  return (
    <div className={`w-full bg-surface border border-neutral-800 rounded-xl p-4 flex flex-col ${className || 'h-80'}`}>
        <div className="flex justify-between items-start mb-4">
             <h4 className="text-neutral-300 text-sm font-semibold">PnL Distribution</h4>
             <div className="flex gap-4 text-xs">
                <div className="text-right">
                    <p className="text-neutral-500">Skew</p>
                    <p className={`font-mono ${stats.skew > 0 ? 'text-emerald-400' : 'text-rose-400'}`}>{stats.skew.toFixed(2)}</p>
                </div>
                <div className="text-right">
                    <p className="text-neutral-500">Kurtosis</p>
                    <p className="font-mono text-neutral-200">{stats.kurtosis.toFixed(2)}</p>
                </div>
             </div>
        </div>
        <div className="flex-1 w-full min-h-0">
            <ResponsiveContainer width="100%" height="100%">
                <BarChart data={data}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#333333" />
                    <XAxis dataKey="mid" stroke="#666666" tickFormatter={(v) => `$${Math.round(v)}`} tick={{fontSize: 10}} />
                    <YAxis stroke="#666666" tick={{fontSize: 10}} />
                    <Tooltip 
                        contentStyle={{ backgroundColor: '#0a0a0a', borderColor: '#333' }}
                        formatter={(val, name, props) => [val, props.payload.rangeLabel]}
                        labelFormatter={() => ''}
                    />
                    <ReferenceLine x={0} stroke="#666" strokeDasharray="3 3" />
                    <Bar dataKey="count">
                        {data.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.mid >= 0 ? '#10b981' : '#ef4444'} />
                        ))}
                    </Bar>
                </BarChart>
            </ResponsiveContainer>
        </div>
        <p className="text-[10px] text-neutral-500 text-center mt-2">
            {stats.skew > 0 ? "Positive Skew: Frequent small losses, occasional large wins." : "Negative Skew: Frequent small wins, occasional large losses."}
        </p>
    </div>
  )
};

export const ScatterMAEMFE: React.FC<{ trades: Trade[], className?: string }> = ({ trades, className }) => {
    const data = useMemo(() => {
        return trades
            .filter(t => t.mae !== undefined) // Must have MAE
            .map(t => ({
                mae: t.mae,
                mfe: t.mfe || 0,
                pnl: t.pnl,
                id: t.id
            }));
    }, [trades]);

    if (data.length === 0) {
        return (
            <div className={`w-full bg-surface border border-neutral-800 rounded-xl p-4 flex flex-col items-center justify-center text-neutral-500 ${className || 'h-80'}`}>
                 <AlertCircle className="w-8 h-8 mb-2 opacity-50" />
                 <p className="text-sm">MAE/MFE data not found</p>
                 <p className="text-xs mt-1 text-center">Ensure CSV has "Run-up USD" and "Drawdown USD" columns.</p>
            </div>
        );
    }

    return (
        <div className={`w-full bg-surface border border-neutral-800 rounded-xl p-4 flex flex-col justify-center ${className || 'h-80'}`}>
             <h4 className="text-neutral-300 text-sm font-semibold mb-4 text-center">PnL vs MAE (Adverse Excursion)</h4>
             <div className="flex-1 w-full min-h-0">
                <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#333333" />
                        <XAxis 
                            type="number" 
                            dataKey="mae" 
                            name="MAE" 
                            unit="$" 
                            stroke="#666666"
                            label={{ value: 'MAE (Max Adverse Drawdown)', position: 'insideBottom', offset: -10, fill: '#666666' }}
                        />
                        <YAxis 
                            type="number" 
                            dataKey="pnl" 
                            name="PnL" 
                            unit="$" 
                            stroke="#666666"
                            label={{ value: 'Realized PnL', angle: -90, position: 'insideLeft', fill: '#666666' }}
                        />
                        <Tooltip 
                            cursor={{ strokeDasharray: '3 3' }}
                            contentStyle={{ backgroundColor: '#0a0a0a', borderColor: '#333' }}
                        />
                        <Scatter name="Trades" data={data} fill="#8884d8">
                            {data.map((entry, index) => (
                                <Cell key={`cell-${index}`} fill={entry.pnl >= 0 ? '#10b981' : '#ef4444'} />
                            ))}
                        </Scatter>
                    </ScatterChart>
                </ResponsiveContainer>
             </div>
        </div>
    );
};

export const UnderwaterChart: React.FC<{ trades: Trade[], className?: string }> = ({ trades, className }) => {
    const data = useMemo(() => {
        let equity = 0;
        let peak = 0;
        return trades.map((t, i) => {
            equity += t.pnl;
            if (equity > peak) peak = equity;
            const syntheticBase = 100000;
            const currentTotal = syntheticBase + equity;
            const peakTotal = syntheticBase + peak; 
            
            const dd = ((currentTotal - peakTotal) / peakTotal) * 100;
            
            return {
                index: i,
                drawdown: dd
            };
        });
    }, [trades]);

    return (
        <div className={`w-full bg-surface border border-neutral-800 rounded-xl p-4 flex flex-col ${className || 'h-80'}`}>
             <h4 className="text-neutral-300 text-sm font-semibold mb-4 text-center">Underwater Plot (Drawdown %)</h4>
             <div className="flex-1 w-full min-h-0">
                <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={data}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#333333" />
                        <XAxis dataKey="index" stroke="#666666" hide />
                        <YAxis stroke="#666666" tickFormatter={(val) => `${val.toFixed(1)}%`} />
                        <Tooltip 
                            contentStyle={{ backgroundColor: '#0a0a0a', borderColor: '#333' }}
                            formatter={(val: number) => [`${val.toFixed(2)}%`, 'Drawdown']}
                        />
                        <Area type="monotone" dataKey="drawdown" stroke="#ef4444" fill="#ef4444" fillOpacity={0.3} />
                    </AreaChart>
                </ResponsiveContainer>
             </div>
        </div>
    );
};

export const RollingStatsChart: React.FC<{ trades: Trade[], className?: string }> = ({ trades, className }) => {
    const data = useMemo(() => calculateRollingStats(trades, 50), [trades]);

    if (data.length === 0) return null;

    return (
        <div className={`w-full bg-surface border border-neutral-800 rounded-xl p-4 flex flex-col ${className || 'h-80'}`}>
            <h4 className="text-neutral-300 text-sm font-semibold mb-4 text-center">Rolling 50-Trade Stats</h4>
            <div className="flex-1 w-full min-h-0">
                <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={data}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#333333" />
                        <XAxis dataKey="index" stroke="#666666" />
                        <YAxis yAxisId="left" stroke="#3b82f6" tickFormatter={(val) => `${val}%`} domain={[0, 100]} />
                        <YAxis yAxisId="right" orientation="right" stroke="#10b981" />
                        <Tooltip contentStyle={{ backgroundColor: '#0a0a0a', borderColor: '#333' }} />
                        <Legend />
                        <Line yAxisId="left" type="monotone" dataKey="winRate" name="Win Rate %" stroke="#3b82f6" dot={false} strokeWidth={2} />
                        <Line yAxisId="right" type="monotone" dataKey="sharpe" name="Sharpe Ratio" stroke="#10b981" dot={false} strokeWidth={2} />
                    </LineChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
};

export const StreakProbabilityChart: React.FC<{ results: SimulationResult[], winRate: number, className?: string }> = ({ results, winRate, className }) => {
    const data = useMemo(() => {
        if (!results.length) return [];
        const counts = new Map<number, number>();
        let maxObservedStreak = 0;

        results.forEach(r => {
            const streak = r.maxConsecutiveLosses;
            counts.set(streak, (counts.get(streak) || 0) + 1);
            if (streak > maxObservedStreak) maxObservedStreak = streak;
        });

        const chartData = [];
        const lossRate = 1 - (winRate / 100);
        const displayLimit = Math.max(10, Math.min(maxObservedStreak + 2, 20));

        for (let i = 1; i <= displayLimit; i++) {
            const count = counts.get(i) || 0;
            const frequency = (count / results.length) * 100;
            const theoretical = Math.pow(lossRate, i) * 100;

            chartData.push({
                streak: i,
                empirical: frequency,
                theoretical: theoretical
            });
        }
        return chartData;
    }, [results, winRate]);

    if (!results.length) return null;

    return (
        <div className={`w-full bg-surface border border-neutral-800 rounded-xl p-4 flex flex-col ${className || 'h-80'}`}>
             <h4 className="text-neutral-300 text-sm font-semibold mb-4 text-center">Streak Probability: Actual vs Theoretical</h4>
             <div className="flex-1 w-full min-h-0">
                <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart data={data}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#333333" vertical={false} />
                        <XAxis 
                            dataKey="streak" 
                            stroke="#666666"
                            label={{ value: 'Consecutive Losses (N)', position: 'insideBottom', offset: -5, fill: '#666666' }}
                        />
                        <YAxis 
                            stroke="#666666" 
                            tickFormatter={(v) => `${v.toFixed(1)}%`}
                            label={{ value: 'Frequency / Probability', angle: -90, position: 'insideLeft', fill: '#666666' }}
                        />
                        <Tooltip 
                            contentStyle={{ backgroundColor: '#0a0a0a', borderColor: '#333' }}
                            formatter={(val: number, name: string) => [`${val.toFixed(2)}%`, name === 'empirical' ? 'MC Actuals' : 'Theoretical (1-WR)^N']}
                        />
                        <Legend />
                        <Bar dataKey="empirical" name="MC Actuals" fill="#ef4444" barSize={20} radius={[4, 4, 0, 0]} />
                        <Line type="monotone" dataKey="theoretical" name="Theoretical Math" stroke="#3b82f6" strokeWidth={2} dot={{r: 4}} />
                    </ComposedChart>
                </ResponsiveContainer>
             </div>
             <p className="text-[10px] text-neutral-500 text-center mt-2">
                 If Red Bars (Actual) {'>'} Blue Line (Theory), losses are clustering more than random chance (tilt risk).
             </p>
        </div>
    );
};
