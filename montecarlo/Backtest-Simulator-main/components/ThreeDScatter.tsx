import React, { useEffect, useRef, useState, useMemo } from 'react';
import Plotly from 'plotly.js-dist';
import { Trade } from '../types';
import { Box, Activity, Calendar, Grid, Wind, Clock, Zap, TrendingUp, CandlestickChart, AlertCircle, BarChart3, Layers, Hourglass } from 'lucide-react';

interface Props {
  trades: Trade[];
  className?: string;
}

type MetricType = 
  | 'price_action' 
  | 'regime'              // Restored: Hour vs Day vs PnL
  | 'hourly_density'      // Restored: Hour vs PnL vs Count
  | 'efficiency_hr_dur'   // Restored: Hour vs Duration vs PnL
  | 'trend_seq' 
  | 'efficiency_wr' 
  | 'volatility_pnl' 
  | 'efficiency_dur_vol' 
  | 'volume_dur';

type SideType = 'all' | 'long' | 'short';

// Wrap in React.memo to prevent re-renders when parent state changes
const ThreeDScatter: React.FC<Props> = React.memo(({ trades, className }) => {
  const chartRef = useRef<HTMLDivElement>(null);
  const [metric, setMetric] = useState<MetricType>('regime'); 
  const [side, setSide] = useState<SideType>('all');
  
  // Visualization Options
  const [gridSize, setGridSize] = useState<number>(30); 
  const [smoothing, setSmoothing] = useState<number>(3); 

  // Filter trades based on side
  const filteredTrades = useMemo(() => {
    if (side === 'all') return trades;
    return trades.filter(t => t.type.toLowerCase() === side);
  }, [trades, side]);

  useEffect(() => {
    if (!chartRef.current) return;
    
    if (filteredTrades.length === 0) {
        Plotly.purge(chartRef.current);
        return;
    }

    const config = { displayModeBar: false, responsive: true };
    
    // --- MODE 1: SCATTER (Raw Price Action) ---
    if (metric === 'price_action') {
        const xValues = filteredTrades.map((_, i) => i);
        const yValues = filteredTrades.map(t => t.entryPrice || 0);
        const zValues = filteredTrades.map(t => t.pnl);
        const texts = filteredTrades.map(t => `Trade #${t.id}<br>Price: ${t.entryPrice}<br>PnL: $${t.pnl}`);
        const hasPrice = yValues.some(p => p > 0);
        const maxAbsPnl = Math.max(...zValues.map(Math.abs));
        
        const data = [{
            x: xValues,
            y: yValues,
            z: zValues,
            mode: 'markers',
            type: 'scatter3d',
            marker: {
                size: 3,
                color: zValues,
                colorscale: 'RdBu',
                cmin: -maxAbsPnl,
                cmax: maxAbsPnl,
                showscale: true,
                colorbar: { title: 'PnL', len: 0.5, thickness: 10 },
                opacity: 0.8
            },
            text: texts,
            hoverinfo: 'text'
        }];

        const layout = {
            autosize: true,
            margin: { l: 0, r: 0, b: 0, t: 0 },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            scene: {
                xaxis: { title: 'Trade Seq', color: '#888', gridcolor: '#222' },
                yaxis: { title: hasPrice ? 'Price' : 'N/A', color: '#888', gridcolor: '#222' },
                zaxis: { title: 'PnL', color: '#888', gridcolor: '#222' },
                camera: { eye: { x: 1.6, y: 1.6, z: 1.2 } },
                aspectmode: 'cube'
            }
        };

        Plotly.newPlot(chartRef.current, data as any, layout as any, config);
        return;
    }

    // --- MODE 2: SURFACE (Aggregated Clusters) ---
    let xLabel = '';
    let yLabel = '';
    let zLabel = '';
    let aggregationType: 'avg' | 'count' | 'winrate' | 'sum' = 'avg';
    
    // Arrays for raw data extraction
    let xValues: number[] = [];
    let yValues: number[] = [];
    let zValues: number[] = []; 
    
    const getHour = (d: string) => { const h = new Date(d).getHours(); return isNaN(h) ? 12 : h; };
    const getDay = (d: string) => { const day = new Date(d).getDay(); return day === 0 ? 7 : day; };
    const getDuration = (t: Trade) => {
        const start = new Date(t.entryTime).getTime();
        const end = new Date(t.exitTime).getTime();
        if (isNaN(start) || isNaN(end)) return 0;
        const diff = (end - start) / (1000 * 60); // minutes
        return diff > 0 ? diff : 0;
    };

    // --- MAPPING LOGIC ---
    
    if (metric === 'regime') {
        // X: Hour, Y: Day, Z: PnL (Avg)
        xLabel = 'Hour of Day';
        yLabel = 'Day of Week';
        zLabel = 'Avg PnL';
        aggregationType = 'avg';
        filteredTrades.forEach(t => {
            xValues.push(getHour(t.entryTime));
            yValues.push(getDay(t.entryTime));
            zValues.push(t.pnl);
        });

    } else if (metric === 'hourly_density') {
        // X: Hour, Y: PnL, Z: Count
        xLabel = 'Hour of Day';
        yLabel = 'PnL Value';
        zLabel = 'Trade Count';
        aggregationType = 'count';
        filteredTrades.forEach(t => {
            xValues.push(getHour(t.entryTime));
            yValues.push(t.pnl);
            zValues.push(1);
        });

    } else if (metric === 'efficiency_hr_dur') {
        // X: Hour, Y: Duration, Z: PnL (Avg)
        xLabel = 'Hour of Day';
        yLabel = 'Duration (m)';
        zLabel = 'Avg PnL';
        aggregationType = 'avg';
        filteredTrades.forEach(t => {
            xValues.push(getHour(t.entryTime));
            yValues.push(getDuration(t));
            zValues.push(t.pnl);
        });

    } else if (metric === 'trend_seq') {
        // X: Sequence, Y: Duration, Z: PnL (Avg)
        xLabel = 'Trade Sequence';
        yLabel = 'Duration (m)';
        zLabel = 'Avg PnL ($)';
        aggregationType = 'avg';
        filteredTrades.forEach((t, i) => {
            xValues.push(i);
            yValues.push(getDuration(t));
            zValues.push(t.pnl);
        });

    } else if (metric === 'efficiency_wr') {
        // X: Hour, Y: Duration, Z: Win Rate %
        xLabel = 'Hour of Day';
        yLabel = 'Duration (m)';
        zLabel = 'Win Rate %';
        aggregationType = 'winrate';
        filteredTrades.forEach(t => {
            xValues.push(getHour(t.entryTime));
            yValues.push(getDuration(t));
            zValues.push(t.pnl); // We use PnL > 0 check later
        });

    } else if (metric === 'volatility_pnl') {
        // X: Duration, Y: PnL % (signed), Z: Density
        xLabel = 'Duration (m)';
        yLabel = 'PnL %';
        zLabel = 'Density';
        aggregationType = 'count';
        filteredTrades.forEach(t => {
            xValues.push(getDuration(t));
            yValues.push(t.pnlPercent);
            zValues.push(1);
        });

    } else if (metric === 'efficiency_dur_vol') {
        // X: Duration, Y: MAE (Vol proxy), Z: PnL
        xLabel = 'Duration (m)';
        yLabel = 'MAE / Volatility';
        zLabel = 'Avg PnL ($)';
        aggregationType = 'avg';
        filteredTrades.forEach(t => {
            xValues.push(getDuration(t));
            // Use MAE (absolute) as proxy for volatility/risk taken
            yValues.push(t.mae ? Math.abs(t.mae) : 0);
            zValues.push(t.pnl);
        });

    } else if (metric === 'volume_dur') {
        // X: Hour, Y: Duration, Z: Count (Volume)
        xLabel = 'Hour of Day';
        yLabel = 'Duration (m)';
        zLabel = 'Trade Volume';
        aggregationType = 'count';
        filteredTrades.forEach(t => {
            xValues.push(getHour(t.entryTime));
            yValues.push(getDuration(t));
            zValues.push(1);
        });
    }

    // --- GRID GENERATION ---
    const xMin = Math.min(...xValues);
    const xMax = Math.max(...xValues) + 0.01; 
    const yMin = Math.min(...yValues);
    const yMax = Math.max(...yValues) + 0.01;

    const zGrid: number[][] = Array(gridSize).fill(0).map(() => Array(gridSize).fill(0));
    const countGrid: number[][] = Array(gridSize).fill(0).map(() => Array(gridSize).fill(0));
    
    const xStep = (xMax - xMin) / gridSize;
    const yStep = (yMax - yMin) / gridSize;

    // Populate Grid
    for (let i = 0; i < xValues.length; i++) {
        const xIdx = Math.min(Math.floor((xValues[i] - xMin) / xStep), gridSize - 1);
        const yIdx = Math.min(Math.floor((yValues[i] - yMin) / yStep), gridSize - 1);
        
        if (xIdx >= 0 && xIdx < gridSize && yIdx >= 0 && yIdx < gridSize) {
            countGrid[yIdx][xIdx] += 1;
            
            if (aggregationType === 'winrate') {
                // For winrate, zValues contains PnL. We add 1 if win, 0 if loss
                if (zValues[i] > 0) zGrid[yIdx][xIdx] += 1;
            } else if (aggregationType === 'count') {
                zGrid[yIdx][xIdx] += 1;
            } else {
                // Avg or Sum
                zGrid[yIdx][xIdx] += zValues[i];
            }
        }
    }

    // Normalize Grid based on Type
    for (let y = 0; y < gridSize; y++) {
        for (let x = 0; x < gridSize; x++) {
            if (countGrid[y][x] > 0) {
                if (aggregationType === 'avg') {
                    zGrid[y][x] /= countGrid[y][x];
                } else if (aggregationType === 'winrate') {
                    zGrid[y][x] = (zGrid[y][x] / countGrid[y][x]) * 100;
                }
                // 'count' and 'sum' already correct
            }
        }
    }

    // --- GAUSSIAN SMOOTHING ---
    const smoothGrid = (grid: number[][]) => {
        const kernel = [
            [0.05, 0.1, 0.05],
            [0.1,  0.4, 0.1],
            [0.05, 0.1, 0.05]
        ];
        const rows = grid.length;
        const cols = grid[0].length;
        const newGrid = grid.map(row => [...row]);

        for (let y = 1; y < rows - 1; y++) {
            for (let x = 1; x < cols - 1; x++) {
                let sum = 0;
                for (let ky = -1; ky <= 1; ky++) {
                    for (let kx = -1; kx <= 1; kx++) {
                         sum += grid[y + ky][x + kx] * kernel[ky + 1][kx + 1];
                    }
                }
                newGrid[y][x] = sum;
            }
        }
        return newGrid;
    };

    let finalZ = zGrid;
    for (let i = 0; i < smoothing; i++) {
        finalZ = smoothGrid(finalZ);
    }

    const xTicks = Array(gridSize).fill(0).map((_, i) => xMin + i * xStep);
    const yTicks = Array(gridSize).fill(0).map((_, i) => yMin + i * yStep);

    // Color Scales
    let colorscale: any = 'Viridis'; // Default
    
    // Thermal Look for PnL/Density
    const thermalScale = [
        [0, 'rgb(20, 30, 70)'],      // Deep Dark Blue
        [0.2, 'rgb(30, 60, 180)'],   // Medium Blue
        [0.4, 'rgb(0, 180, 180)'],   // Cyan
        [0.6, 'rgb(100, 255, 100)'], // Green
        [0.8, 'rgb(255, 220, 0)'],   // Yellow
        [1, 'rgb(255, 100, 0)']      // Orange
    ];

    // Red-Green for Win Rate or PnL Average
    const divergingScale = [
        [0, 'rgb(200, 0, 0)'], 
        [0.5, 'rgb(255, 255, 0)'],
        [1, 'rgb(0, 200, 0)']
    ];

    if (aggregationType === 'winrate') {
        colorscale = divergingScale;
    } else if (metric === 'trend_seq' || metric === 'efficiency_dur_vol') {
        colorscale = thermalScale;
    } else {
        colorscale = thermalScale;
    }

    const data = [{
        z: finalZ,
        x: xTicks,
        y: yTicks,
        type: 'surface',
        contours: {
            z: {
                show: true,
                usecolormap: true,
                highlightcolor: "#fff",
                project: { z: true }
            }
        },
        colorscale: colorscale,
        showscale: false,
        opacity: 0.95,
        lighting: {
            roughness: 0.5,
            fresnel: 0.5,
            ambient: 0.4
        }
    }];

    const surfaceLayout = {
      autosize: true,
      margin: { l: 0, r: 0, b: 0, t: 0 },
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      scene: {
        xaxis: { title: xLabel, color: '#aaa', gridcolor: '#333', zeroline: false },
        yaxis: { title: yLabel, color: '#aaa', gridcolor: '#333', zeroline: false },
        zaxis: { title: zLabel, color: '#aaa', gridcolor: '#333', zeroline: false },
        camera: { eye: { x: 1.5, y: 1.5, z: 0.8 } }, 
        aspectmode: 'cube',
        bgcolor: 'rgba(0,0,0,0)'
      }
    };

    Plotly.newPlot(chartRef.current, data as any, surfaceLayout as any, config);

    return () => {
        if(chartRef.current) {
            Plotly.purge(chartRef.current);
        }
    };
  }, [filteredTrades, metric, gridSize, smoothing, side]); 

  // Helper for buttons
  const Btn = ({ id, label, icon: Icon, colorClass }: any) => (
    <button 
        onClick={() => setMetric(id)} 
        className={`flex items-center gap-2 px-4 py-2 text-left text-xs font-medium transition-colors ${metric === id ? `bg-neutral-800 ${colorClass}` : 'text-neutral-400 hover:text-neutral-200'}`}
    >
        <Icon className="w-3 h-3" /> {label}
    </button>
  );

  return (
    <div className={`bg-surface rounded-xl border border-neutral-800 shadow-xl flex flex-col overflow-hidden relative ${className || 'h-[500px]'}`}>
       
       {/* Controls Header */}
       <div className="absolute top-4 left-4 z-10 flex flex-col gap-2">
            
            {/* Side Toggle */}
            <div className="flex bg-neutral-900/80 backdrop-blur-md rounded-lg p-1 border border-neutral-800 shadow-lg w-fit">
                {(['all', 'long', 'short'] as const).map((s) => (
                    <button
                        key={s}
                        onClick={() => setSide(s)}
                        className={`px-3 py-1.5 text-xs font-bold uppercase rounded-md transition-all ${
                            side === s 
                            ? 'bg-primary text-black shadow-sm' 
                            : 'text-neutral-400 hover:text-white hover:bg-neutral-800'
                        }`}
                    >
                        {s}
                    </button>
                ))}
            </div>

            {/* Metric Toggle */}
            <div className="flex flex-col bg-neutral-900/80 backdrop-blur-md rounded-lg border border-neutral-800 shadow-lg overflow-hidden w-64 max-h-[350px] overflow-y-auto custom-scrollbar">
                <Btn id="price_action" label="Price Action (Scatter)" icon={CandlestickChart} colorClass="text-purple-400" />
                <div className="h-px bg-neutral-800 my-1 mx-2"></div>
                {/* Restored Charts */}
                <Btn id="regime" label="Time Regime (Hr/Day)" icon={Calendar} colorClass="text-blue-400" />
                <Btn id="hourly_density" label="Hourly Density (Hr/PnL)" icon={Clock} colorClass="text-rose-400" />
                <Btn id="efficiency_hr_dur" label="Efficiency (Hr/Dur)" icon={Hourglass} colorClass="text-yellow-400" />
                <div className="h-px bg-neutral-800 my-1 mx-2"></div>
                {/* New Analysis Charts */}
                <Btn id="trend_seq" label="Trend (Seq)" icon={TrendingUp} colorClass="text-indigo-400" />
                <Btn id="efficiency_wr" label="Efficiency (WR%)" icon={Zap} colorClass="text-yellow-400" />
                <Btn id="volatility_pnl" label="Volatility (PnL%)" icon={Activity} colorClass="text-emerald-400" />
                <Btn id="efficiency_dur_vol" label="Efficiency (Dur/Vol)" icon={Layers} colorClass="text-cyan-400" />
                <Btn id="volume_dur" label="Volume (Dur)" icon={BarChart3} colorClass="text-orange-400" />
            </div>
       
            {/* Surface Controls */}
            {metric !== 'price_action' && (
                <div className="flex flex-col bg-neutral-900/80 backdrop-blur-md rounded-lg border border-neutral-800 shadow-lg p-3 gap-3 animate-in fade-in slide-in-from-top-2 w-64">
                     <div className="flex flex-col gap-1.5">
                        <div className="flex items-center gap-1.5 text-neutral-500">
                            <Grid className="w-3 h-3" />
                            <span className="text-[10px] uppercase font-bold tracking-wide">Grid Resolution</span>
                        </div>
                        <div className="flex gap-1">
                            {[20, 30, 50].map(r => (
                                <button key={r} onClick={() => setGridSize(r)} className={`flex-1 px-1 py-1 text-[10px] font-medium rounded transition-colors ${gridSize === r ? 'bg-primary text-black' : 'bg-neutral-800 text-neutral-400 hover:bg-neutral-700'}`}>{r === 20 ? 'Low' : r === 30 ? 'Med' : 'High'}</button>
                            ))}
                        </div>
                     </div>
                     <div className="flex flex-col gap-1.5">
                        <div className="flex items-center gap-1.5 text-neutral-500">
                            <Wind className="w-3 h-3" />
                            <span className="text-[10px] uppercase font-bold tracking-wide">Smoothing</span>
                        </div>
                        <div className="flex gap-1">
                             {[0, 2, 5].map(s => (
                                <button key={s} onClick={() => setSmoothing(s)} className={`flex-1 px-1 py-1 text-[10px] font-medium rounded transition-colors ${smoothing === s ? 'bg-primary text-black' : 'bg-neutral-800 text-neutral-400 hover:bg-neutral-700'}`}>{s === 0 ? 'Raw' : s === 2 ? 'Soft' : 'Silk'}</button>
                            ))}
                        </div>
                     </div>
                </div>
            )}
       </div>

       <div className="flex items-center justify-end p-4 border-b border-neutral-800 bg-neutral-900/30">
            <div className="flex items-center gap-2">
                <Box className="w-5 h-5 text-primary" />
                <h3 className="font-semibold text-neutral-200">3D Cluster Analysis</h3>
            </div>
       </div>

       <div ref={chartRef} className="flex-1 w-full relative">
            {filteredTrades.length === 0 && (
                <div className="absolute inset-0 flex flex-col items-center justify-center bg-neutral-900/50 backdrop-blur-sm z-20">
                    <AlertCircle className="w-12 h-12 text-neutral-500 mb-2" />
                    <p className="text-neutral-400 font-medium">No trades found for this filter.</p>
                </div>
            )}
       </div>
    </div>
  );
});

export default ThreeDScatter;