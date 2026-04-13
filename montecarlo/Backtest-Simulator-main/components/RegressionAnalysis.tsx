import React, { useMemo, useState } from 'react';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, ReferenceLine 
} from 'recharts';
import { Trade } from '../types';
import { performRegressionAnalysis, RegressionResult } from '../utils/analytics';
import { Microscope, Info, Filter, BarChart2, ArrowLeftRight } from 'lucide-react';

interface Props {
  trades: Trade[];
  className?: string;
}

const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    return (
      <div className="bg-surface border border-neutral-800 p-3 rounded shadow-lg text-sm z-50">
        <p className="font-semibold text-neutral-200 mb-1">{data.feature}</p>
        <div className="space-y-1">
          <p className="text-neutral-400">Correlation: <span className="text-neutral-200 font-mono">{data.correlation.toFixed(3)}</span></p>
          <p className="text-neutral-400">Importance: <span className="text-neutral-200 font-mono">{data.importance.toFixed(3)}</span></p>
          <p className="text-neutral-400">R-Squared: <span className="text-neutral-200 font-mono">{data.rSquared.toFixed(3)}</span></p>
        </div>
      </div>
    );
  }
  return null;
};

const RegressionAnalysis: React.FC<Props> = ({ trades, className }) => {
  const [tab, setTab] = useState<'all' | 'long' | 'short'>('all');
  const [view, setView] = useState<'correlation' | 'importance'>('correlation');

  const analysis = useMemo(() => performRegressionAnalysis(trades), [trades]);

  const currentData = analysis[tab];

  if (trades.length < 5) {
    return (
      <div className={`bg-surface rounded-xl p-6 border border-neutral-800 flex flex-col items-center justify-center text-neutral-500 ${className || 'h-80'}`}>
        <Microscope className="w-12 h-12 mb-4 opacity-50" />
        <p>Insufficient data for regression analysis.</p>
        <p className="text-sm">Need at least 5 trades.</p>
      </div>
    );
  }

  return (
    <div className={`bg-surface rounded-xl p-6 border border-neutral-800 shadow-xl flex flex-col ${className || 'h-80'}`}>
      <div className="flex flex-col gap-4 mb-6">
        <div className="flex items-start justify-between">
            <div>
            <h3 className="text-lg font-semibold flex items-center gap-2">
                <Microscope className="w-5 h-5 text-primary" />
                Performance Drivers
            </h3>
            <p className="text-neutral-400 text-sm mt-1">
                Factors impacting PnL.
            </p>
            </div>
            
            <div className="flex bg-neutral-800 rounded-lg p-1 border border-neutral-700">
                {(['all', 'long', 'short'] as const).map((mode) => (
                    <button
                        key={mode}
                        onClick={() => setTab(mode)}
                        className={`px-3 py-1 text-xs font-medium rounded capitalize transition-colors ${
                            tab === mode ? 'bg-neutral-600 text-white' : 'text-neutral-400 hover:text-white'
                        }`}
                    >
                        {mode}
                    </button>
                ))}
            </div>
        </div>

        {/* View Toggle */}
        <div className="flex justify-end border-b border-neutral-800 pb-2">
             <div className="flex gap-4">
                 <button 
                    onClick={() => setView('correlation')}
                    className={`text-xs flex items-center gap-1 ${view === 'correlation' ? 'text-primary font-bold' : 'text-neutral-500 hover:text-neutral-300'}`}
                 >
                    <ArrowLeftRight className="w-3 h-3" /> Correlation
                 </button>
                 <button 
                    onClick={() => setView('importance')}
                    className={`text-xs flex items-center gap-1 ${view === 'importance' ? 'text-primary font-bold' : 'text-neutral-500 hover:text-neutral-300'}`}
                 >
                    <BarChart2 className="w-3 h-3" /> Importance
                 </button>
             </div>
        </div>
      </div>

      <div className="flex-1 min-h-0 flex flex-col">
         {currentData.length > 0 ? (
            <ResponsiveContainer width="100%" height="100%">
                <BarChart 
                    data={currentData} 
                    layout="vertical" 
                    margin={{ top: 5, right: 30, left: 40, bottom: 5 }}
                >
                <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={true} stroke="#333333" />
                <XAxis type="number" domain={view === 'correlation' ? [-1, 1] : [0, 1]} hide />
                <YAxis 
                    type="category" 
                    dataKey="feature" 
                    width={100} 
                    tick={{fontSize: 11, fill: '#888888'}} 
                />
                <Tooltip content={<CustomTooltip />} cursor={{fill: '#333333', opacity: 0.2}} />
                {view === 'correlation' && <ReferenceLine x={0} stroke="#555555" />}
                <Bar dataKey={view} radius={[0, 4, 4, 0]} barSize={15}>
                    {currentData.map((entry, index) => (
                        <Cell 
                            key={`cell-${index}`} 
                            fill={
                                view === 'importance' 
                                ? '#8b5cf6' 
                                : entry.correlation >= 0 ? '#10b981' : '#ef4444'
                            } 
                        />
                    ))}
                </Bar>
                </BarChart>
            </ResponsiveContainer>
         ) : (
             <div className="flex-1 flex items-center justify-center text-xs text-neutral-500">
                 Not enough data for this segment.
             </div>
         )}
      </div>
      
      <div className="mt-4 pt-4 border-t border-neutral-800">
           <p className="text-xs text-neutral-500 leading-relaxed">
             {interpretTopFactor(currentData)}
           </p>
      </div>
    </div>
  );
};

const interpretTopFactor = (data: RegressionResult[]) => {
    if (!data || data.length === 0) return "";
    const top = data[0];
    if (top.importance < 0.1) return "No strong correlations found in this dataset.";
    
    const direction = top.correlation > 0 ? "positive" : "negative";
    return `Dominant Factor: ${top.feature} shows a ${direction} relationship (r=${top.correlation.toFixed(2)}).`;
};

export default RegressionAnalysis;