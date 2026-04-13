import React, { useEffect, useRef } from 'react';
import Plotly from 'plotly.js-dist';
import { SimulationResult } from '../types';
import { Activity } from 'lucide-react';

interface Props {
  results: SimulationResult[];
}

const ProbabilityHeatmap: React.FC<Props> = React.memo(({ results }) => {
  const chartRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!chartRef.current || results.length === 0) return;

    const numSims = results.length;
    // Assume all sims have same length, use the first one
    const numTrades = results[0].equityCurve.length; 
    
    // Flatten to find global min/max for Y-axis scaling
    let minEq = Number.MAX_VALUE;
    let maxEq = Number.MIN_VALUE;
    
    // Sample a subset for performance if huge, but usually fine
    // Check first, middle, last sim to estimate range + a quick pass on best/worst final equity sims
    // Or just iterate all. 1000 * 100 = 100k ops, totally fine.
    for (let i = 0; i < numSims; i++) {
        for (let j = 0; j < numTrades; j++) {
            const val = results[i].equityCurve[j];
            if (val < minEq) minEq = val;
            if (val > maxEq) maxEq = val;
        }
    }
    
    // Add 5% padding
    const range = maxEq - minEq;
    minEq -= range * 0.05;
    maxEq += range * 0.05;

    // Configuration
    const yBins = 50; 
    const xStep = Math.max(1, Math.floor(numTrades / 100)); // Target ~100 x-axis points
    
    // Initialize Z matrix (Rows = Y bins, Cols = X time steps)
    // Note: Plotly Heatmap z is an array of arrays where z[i] corresponds to the i-th Y coordinate (row).
    const zData: number[][] = Array(yBins).fill(0).map(() => Array(Math.ceil(numTrades / xStep)).fill(0));
    
    const binSize = (maxEq - minEq) / yBins;
    
    // Calculate Y Labels (centers)
    const yLabels = Array(yBins).fill(0).map((_, i) => minEq + (i + 0.5) * binSize);
    
    // Calculate X Labels
    const xLabels: number[] = [];
    
    // Populate Z
    for (let t = 0; t < numTrades; t += xStep) {
        const colIdx = Math.floor(t / xStep);
        xLabels.push(t);
        
        for (let s = 0; s < numSims; s++) {
            const equity = results[s].equityCurve[t];
            // Determine which bin this equity falls into
            const binIdx = Math.floor((equity - minEq) / binSize);
            const safeBinIdx = Math.max(0, Math.min(yBins - 1, binIdx));
            
            zData[safeBinIdx][colIdx] += 1;
        }
    }

    // Render
    const data = [{
        z: zData,
        x: xLabels,
        y: yLabels,
        type: 'heatmap',
        colorscale: 'Jet', 
        zsmooth: 'best',
        hoverongaps: false,
        hovertemplate: 'Trade: %{x}<br>Equity: $%{y:.2f}<br>Count: %{z}<extra></extra>'
    }];

    const layout = {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        margin: { l: 60, r: 20, b: 40, t: 20 },
        xaxis: { 
            title: 'Trade Sequence', 
            color: '#888', 
            gridcolor: '#333',
            zerolinecolor: '#333' 
        },
        yaxis: { 
            title: 'Equity ($)', 
            color: '#888', 
            gridcolor: '#333',
            zerolinecolor: '#333'
        },
        autosize: true,
        showlegend: false
    };

    const config = { 
        displayModeBar: false, 
        responsive: true 
    };

    Plotly.newPlot(chartRef.current, data as any, layout as any, config);

    return () => {
        if (chartRef.current) Plotly.purge(chartRef.current);
    };

  }, [results]);

  return (
    <div className="bg-surface rounded-xl border border-neutral-800 shadow-xl flex flex-col h-[300px]">
        <div className="flex items-center gap-2 p-4 border-b border-neutral-800 bg-neutral-900/30">
            <Activity className="w-5 h-5 text-purple-400" />
            <h4 className="text-neutral-200 font-semibold text-sm">Equity Probability Heatmap</h4>
        </div>
        <div className="flex-1 w-full p-2">
             <div ref={chartRef} className="w-full h-full" />
        </div>
    </div>
  );
});

export default ProbabilityHeatmap;