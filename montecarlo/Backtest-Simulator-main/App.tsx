import React, { useState, useMemo, useEffect, useRef, useCallback } from 'react';
import { 
  BarChart3, 
  TrendingUp, 
  AlertTriangle, 
  Activity, 
  Settings2,
  RefreshCw,
  FileText,
  Play,
  Pause,
  RotateCcw,
  ShieldAlert,
  Hash,
  Scale,
  Percent,
  GraduationCap,
  ArrowDown,
  Gauge,
  Trophy,
  Target,
  BrainCircuit
} from 'lucide-react';
import FileUpload from './components/FileUpload';
import StatsCard from './components/StatsCard';
import { 
    EquityChart, 
    DistributionChart, 
    VaRCurveChart, 
    HistoricalEquityChart, 
    ScatterPnLDuration,
    ScatterMAEMFE,
    UnderwaterChart,
    RollingStatsChart,
    SkewKurtosisChart,
    StreakProbabilityChart
} from './components/Charts';
import RegressionAnalysis from './components/RegressionAnalysis';
import ThreeDScatter from './components/ThreeDScatter';
import ProbabilityHeatmap from './components/ProbabilityHeatmap';
import PropFirmDashboard from './components/PropFirmDashboard';
import AIReportCard from './components/AIReportCard'; // Import AI Component
import { parseTradingViewCSV } from './utils/csvParser';
import { runSimulationBatch, calculateStatistics, createPRNG } from './utils/monteCarlo';
import { runPropFirmSimulation } from './utils/propFirmLogic';
import { calculateSQN } from './utils/analytics';
import { Trade, SimulationResult, SimulationConfig, Statistics, PropFirmConfig, PropFirmAggregateStats } from './types';

function App() {
  const [trades, setTrades] = useState<Trade[]>([]);
  
  // Configs
  const [mcConfig, setMcConfig] = useState<SimulationConfig>({
    initialEquity: 50000,
    numSimulations: 1000,
    tradesPerSimulation: 100,
    seed: 12345,
    convergenceTolerance: 0.1, 
    confidenceLevel: 95,
    riskModel: 'fixed_pnl'
  });

  const [propConfig, setPropConfig] = useState<PropFirmConfig>({
    accountSize: 50000,
    maxDailyLoss: 1000,
    maxTotalLoss: 2000,
    profitTargetEval: 3000,
    maxDailyProfitEval: 1500,
    expressPayoutThreshold: 150,
    expressPayoutsRequired: 5,
    expressDaysForPayout: 5,
    tradesPerDay: 3
  });

  // State
  const [mcResults, setMcResults] = useState<SimulationResult[]>([]);
  const [propStats, setPropStats] = useState<PropFirmAggregateStats | null>(null);
  
  const [status, setStatus] = useState<'idle' | 'running' | 'paused' | 'completed' | 'converged'>('idle');
  const [progress, setProgress] = useState(0);

  // Refs for Monte Carlo Loop
  const resultsRef = useRef<SimulationResult[]>([]);
  const timerRef = useRef<number | null>(null);
  const rngRef = useRef<(() => number) | null>(null);
  const prevMeanEquityRef = useRef<number>(0);

  const BATCH_SIZE = 50; 

  const handleFileUpload = useCallback((content: string) => {
    try {
      const parsedTrades = parseTradingViewCSV(content);
      if (parsedTrades.length === 0) {
        alert("No valid trades found in CSV.");
        return;
      }
      setTrades(parsedTrades);
      setMcConfig(prev => ({ ...prev, tradesPerSimulation: parsedTrades.length }));
      
      // Reset logic inline to avoid dependency
      if (timerRef.current) clearTimeout(timerRef.current);
      setStatus('idle');
      setMcResults([]);
      resultsRef.current = [];
      prevMeanEquityRef.current = 0;
      setPropStats(null);
      setProgress(0);
      rngRef.current = null;
    } catch (e) {
      alert("Error parsing CSV: " + (e as Error).message);
    }
  }, []);

  const resetAll = useCallback(() => {
    if (timerRef.current) clearTimeout(timerRef.current);
    setStatus('idle');
    setMcResults([]);
    resultsRef.current = [];
    prevMeanEquityRef.current = 0;
    setPropStats(null);
    setProgress(0);
    rngRef.current = null;
  }, []);

  const startSimulation = useCallback(() => {
    if (status === 'idle' || status === 'completed' || status === 'converged') {
      // Start fresh
      setStatus('running');
      setMcResults([]);
      setPropStats(null);
      resultsRef.current = [];
      prevMeanEquityRef.current = 0;
      setProgress(0);
      rngRef.current = createPRNG(mcConfig.seed);
    } else if (status === 'running') {
      setStatus('paused');
      if (timerRef.current) clearTimeout(timerRef.current);
    } else if (status === 'paused') {
      setStatus('running');
    }
  }, [status, mcConfig.seed]);

  // Logic to Trigger Prop Firm Sim
  // Wrapped in useCallback to avoid stale closures in the loop
  const triggerPropFirmSim = useCallback(() => {
     const stats = runPropFirmSimulation(trades, propConfig, 1000, mcConfig.seed);
     setPropStats(stats);
  }, [trades, propConfig, mcConfig.seed]);

  // Monte Carlo Loop
  useEffect(() => {
    if (status === 'running') {
      const runBatch = () => {
        const currentCount = resultsRef.current.length;
        if (currentCount >= mcConfig.numSimulations) {
          setStatus('completed');
          triggerPropFirmSim(); 
          return;
        }

        const remaining = mcConfig.numSimulations - currentCount;
        const count = Math.min(BATCH_SIZE, remaining);
        
        // Safety check for RNG
        if (!rngRef.current) rngRef.current = createPRNG(mcConfig.seed);

        const batchResults = runSimulationBatch(trades, mcConfig, count, currentCount, rngRef.current!);
        resultsRef.current = [...resultsRef.current, ...batchResults];
        
        // Convergence Check
        if (resultsRef.current.length > BATCH_SIZE * 2) {
          const totalEquity = resultsRef.current.reduce((sum, r) => sum + r.finalEquity, 0);
          const currentMean = totalEquity / resultsRef.current.length;
          
          if (prevMeanEquityRef.current !== 0 && mcConfig.convergenceTolerance > 0) {
            const percentDiff = Math.abs((currentMean - prevMeanEquityRef.current) / prevMeanEquityRef.current) * 100;
            if (percentDiff < mcConfig.convergenceTolerance) {
               setMcResults([...resultsRef.current]);
               setProgress(100);
               setStatus('converged');
               triggerPropFirmSim(); 
               return; 
            }
          }
          prevMeanEquityRef.current = currentMean;
        }

        setMcResults([...resultsRef.current]);
        setProgress(Math.round((resultsRef.current.length / mcConfig.numSimulations) * 100));
        timerRef.current = window.setTimeout(runBatch, 0);
      };

      // Start the loop
      runBatch();
    }

    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [status, mcConfig.numSimulations, mcConfig.convergenceTolerance, mcConfig.seed, trades, triggerPropFirmSim, mcConfig.riskModel, mcConfig.initialEquity]);

  // Calculate MC Stats via useMemo instead of useEffect (Derived State)
  const mcStats = useMemo(() => {
    if (mcResults.length === 0) return null;
    
    let simulatedYears = 1;
    // Simple duration estimation
    const timestamps = trades
        .map(t => new Date(t.entryTime).getTime())
        .filter(t => !isNaN(t))
        .sort((a, b) => a - b);

    if (timestamps.length > 1) {
        const durationYears = Math.max((timestamps[timestamps.length - 1] - timestamps[0]) / (1000 * 60 * 60 * 24 * 365.25), 0.0027);
        const tradesPerYear = trades.length / durationYears;
        simulatedYears = mcConfig.tradesPerSimulation / tradesPerYear;
    }

    return calculateStatistics(mcResults, mcConfig.initialEquity, simulatedYears, mcConfig.confidenceLevel);
  }, [mcResults, mcConfig.initialEquity, mcConfig.tradesPerSimulation, mcConfig.confidenceLevel, trades]);


  const historicalStats = useMemo(() => {
    if (trades.length === 0) return null;
    
    // Basic Stats
    const wins = trades.filter(t => t.pnl > 0);
    const losses = trades.filter(t => t.pnl <= 0);
    const totalPnl = trades.reduce((acc, t) => acc + t.pnl, 0);
    const winRate = (wins.length / trades.length) * 100;
    const profitFactor = Math.abs(wins.reduce((acc, t) => acc + t.pnl, 0)) / Math.abs(losses.reduce((acc, t) => acc + t.pnl, 0) || 1);
    
    // SQN
    const sqn = calculateSQN(trades);

    // Duration for Annualization
    const timestamps = trades
        .map(t => new Date(t.entryTime).getTime())
        .filter(t => !isNaN(t))
        .sort((a, b) => a - b);
    const durationYears = timestamps.length > 1 
        ? Math.max((timestamps[timestamps.length - 1] - timestamps[0]) / (1000 * 60 * 60 * 24 * 365.25), 0.0027) // min 1 day
        : 1;
    const tradesPerYear = trades.length / durationYears;
    const annualFactor = Math.sqrt(tradesPerYear);

    // Sharpe / Sortino (on $ PnL)
    const pnls = trades.map(t => t.pnl);
    const avgPnl = totalPnl / trades.length;
    
    const sumSqDiff = pnls.reduce((sum, p) => sum + Math.pow(p - avgPnl, 2), 0);
    const stdDev = Math.sqrt(sumSqDiff / trades.length);
    
    const sumSqDownside = pnls.reduce((sum, p) => sum + (p < 0 ? Math.pow(p, 2) : 0), 0);
    const downsideDev = Math.sqrt(sumSqDownside / trades.length);

    const sharpe = (stdDev === 0 ? 0 : avgPnl / stdDev) * annualFactor;
    const sortino = (downsideDev === 0 ? 0 : avgPnl / downsideDev) * annualFactor;

    return { totalPnl, winRate, profitFactor, sharpe, sortino, count: trades.length, sqn };
  }, [trades]);

  return (
    <div className="min-h-screen bg-background text-neutral-100 pb-20">
      <header className="border-b border-neutral-800 bg-surface/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-[1600px] mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Activity className="text-primary w-6 h-6" />
            <h1 className="text-xl font-bold tracking-tight">Monte Carlo <span className="text-primary">Mastery</span></h1>
          </div>
          <div className="flex items-center gap-4">
            <a href="/filterless-live.html" className="text-sm text-neutral-400 hover:text-white">
              Filterless Live
            </a>
            {trades.length > 0 && (
                <button onClick={resetAll} className="text-sm text-neutral-400 hover:text-white">Clear Data</button>
            )}
          </div>
        </div>
      </header>

      <main className="max-w-[1600px] mx-auto px-4 sm:px-6 lg:px-8 py-8">
        
        {trades.length === 0 ? (
          <div className="max-w-xl mx-auto mt-20">
            <div className="text-center mb-10">
              <h2 className="text-3xl font-bold mb-4">Quantitative Strategy Analysis</h2>
              <p className="text-neutral-400">Upload your TradingView CSV to unlock 3D clustering, regression analysis, Monte Carlo projections, and Prop Firm stress testing.</p>
            </div>
            <FileUpload onFileUpload={handleFileUpload} />
          </div>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
            
            {/* --- SIDEBAR CONFIGURATION --- */}
            <div className="lg:col-span-1 space-y-6 lg:sticky lg:top-20 lg:h-fit">
              
              {/* Main Controls */}
              <div className="bg-surface rounded-xl p-6 border border-neutral-800 shadow-lg">
                 <div className="flex items-center justify-between mb-4">
                    <h3 className="font-semibold text-lg flex items-center gap-2">
                        <Settings2 className="w-5 h-5 text-primary" /> Config
                    </h3>
                    <div className="px-2 py-1 bg-neutral-800 rounded text-xs font-mono text-emerald-400">
                        {trades.length} Trades
                    </div>
                 </div>

                 {/* Sim Control Button */}
                 <div className="mb-6">
                    <button 
                        onClick={startSimulation}
                        className={`w-full font-bold py-3 rounded-lg flex items-center justify-center gap-2 transition-all shadow-lg ${
                            status === 'running' 
                            ? 'bg-amber-500 hover:bg-amber-600 text-black' 
                            : 'bg-white hover:bg-neutral-200 text-black'
                        }`}
                    >
                        {status === 'running' ? <Pause className="w-4 h-4"/> : <Play className="w-4 h-4"/>}
                        {status === 'running' ? 'Pause Sim' : 'Run Simulations'}
                    </button>
                    {status !== 'idle' && (
                        <div className="mt-2">
                            <div className="w-full bg-neutral-800 rounded-full h-1.5">
                                <div className="bg-emerald-500 h-1.5 rounded-full transition-all duration-300" style={{width: `${progress}%`}}></div>
                            </div>
                            <div className="flex justify-between text-[10px] text-neutral-500 mt-1 uppercase font-semibold">
                                <span>Monte Carlo</span>
                                <span>{status === 'converged' ? 'Converged' : status === 'completed' ? 'Done' : `${progress}%`}</span>
                            </div>
                        </div>
                    )}
                 </div>

                 {/* MC Inputs */}
                 <div className="space-y-3 pb-4 border-b border-neutral-800">
                    <p className="text-xs font-bold text-neutral-500 uppercase">Monte Carlo Settings</p>
                    <div className="grid grid-cols-2 gap-2">
                        <div>
                            <label className="text-[10px] text-neutral-400 block mb-1">Start Equity</label>
                            <input type="number" value={mcConfig.initialEquity} onChange={e => setMcConfig({...mcConfig, initialEquity: Number(e.target.value)})} className="w-full bg-neutral-900 border border-neutral-800 rounded p-1.5 text-xs focus:border-primary outline-none"/>
                        </div>
                        <div>
                            <label className="text-[10px] text-neutral-400 block mb-1"># Sims</label>
                            <input type="number" value={mcConfig.numSimulations} onChange={e => setMcConfig({...mcConfig, numSimulations: Number(e.target.value)})} className="w-full bg-neutral-900 border border-neutral-800 rounded p-1.5 text-xs focus:border-primary outline-none"/>
                        </div>
                    </div>
                    <div>
                         <label className="text-[10px] text-neutral-400 block mb-1">Confidence ({mcConfig.confidenceLevel}%)</label>
                         <input type="range" min="50" max="99" value={mcConfig.confidenceLevel} onChange={e => setMcConfig({...mcConfig, confidenceLevel: Number(e.target.value)})} className="w-full h-1 bg-neutral-800 rounded-lg appearance-none cursor-pointer accent-primary"/>
                    </div>
                    
                    {/* Position Sizing Toggle */}
                    <div className="pt-2">
                         <label className="text-[10px] text-neutral-400 block mb-1">Position Sizing Model</label>
                         <div className="grid grid-cols-2 gap-1 bg-neutral-900 p-1 rounded">
                              <button 
                                onClick={() => setMcConfig({...mcConfig, riskModel: 'fixed_pnl'})}
                                className={`text-[10px] py-1 rounded transition-colors ${mcConfig.riskModel === 'fixed_pnl' ? 'bg-neutral-700 text-white' : 'text-neutral-500 hover:text-neutral-300'}`}
                              >
                                Fixed PnL
                              </button>
                              <button 
                                onClick={() => setMcConfig({...mcConfig, riskModel: 'percent_equity'})}
                                className={`text-[10px] py-1 rounded transition-colors ${mcConfig.riskModel === 'percent_equity' ? 'bg-neutral-700 text-white' : 'text-neutral-500 hover:text-neutral-300'}`}
                              >
                                Compounding
                              </button>
                         </div>
                    </div>
                 </div>

                 {/* Prop Inputs */}
                 <div className="space-y-3 pt-4">
                    <p className="text-xs font-bold text-neutral-500 uppercase">Prop Firm Rules</p>
                    <div>
                        <label className="text-[10px] text-neutral-400 block mb-1">Account Size</label>
                        <input type="number" value={propConfig.accountSize} onChange={e => setPropConfig({...propConfig, accountSize: Number(e.target.value)})} className="w-full bg-neutral-900 border border-neutral-800 rounded p-1.5 text-xs focus:border-primary outline-none"/>
                    </div>
                    <div className="grid grid-cols-2 gap-2">
                        <div>
                            <label className="text-[10px] text-neutral-400 block mb-1">Max Daily Loss</label>
                            <input type="number" value={propConfig.maxDailyLoss} onChange={e => setPropConfig({...propConfig, maxDailyLoss: Number(e.target.value)})} className="w-full bg-neutral-900 border border-neutral-800 rounded p-1.5 text-xs focus:border-primary outline-none"/>
                        </div>
                        <div>
                            <label className="text-[10px] text-neutral-400 block mb-1">Max Total Loss</label>
                            <input type="number" value={propConfig.maxTotalLoss} onChange={e => setPropConfig({...propConfig, maxTotalLoss: Number(e.target.value)})} className="w-full bg-neutral-900 border border-neutral-800 rounded p-1.5 text-xs focus:border-primary outline-none"/>
                        </div>
                    </div>
                    <div className="grid grid-cols-2 gap-2">
                        <div>
                            <label className="text-[10px] text-neutral-400 block mb-1">Eval Target</label>
                            <input type="number" value={propConfig.profitTargetEval} onChange={e => setPropConfig({...propConfig, profitTargetEval: Number(e.target.value)})} className="w-full bg-neutral-900 border border-neutral-800 rounded p-1.5 text-xs focus:border-primary outline-none"/>
                        </div>
                        <div>
                            <label className="text-[10px] text-neutral-400 block mb-1">Max Day Profit</label>
                            <input type="number" value={propConfig.maxDailyProfitEval} onChange={e => setPropConfig({...propConfig, maxDailyProfitEval: Number(e.target.value)})} className="w-full bg-neutral-900 border border-neutral-800 rounded p-1.5 text-xs focus:border-primary outline-none"/>
                        </div>
                    </div>
                 </div>

              </div>

              {/* Historical Mini Card (Reduced version since main cards are in grid now) */}
              <div className="bg-surface rounded-xl p-4 border border-neutral-800 space-y-3">
                 <h4 className="text-xs font-bold text-neutral-500 uppercase mb-2">Sim Status</h4>
                 <div className="flex justify-between text-sm">
                      <span className="text-neutral-400">Mode</span>
                      <span className="text-neutral-200">{mcConfig.riskModel === 'fixed_pnl' ? 'Fixed Risk' : '% Compounding'}</span>
                 </div>
                 <div className="flex justify-between text-sm">
                      <span className="text-neutral-400">Iterations</span>
                      <span className="text-neutral-200">{mcConfig.numSimulations}</span>
                 </div>
              </div>
            </div>

            {/* --- MAIN CONTENT --- */}
            <div className="lg:col-span-3 space-y-8">
              
              {/* SECTION 1: DATA ANALYSIS */}
              <section className="space-y-6">
                <div className="flex items-center gap-2 border-b border-neutral-800 pb-2">
                    <TrendingUp className="text-primary w-5 h-5" />
                    <h2 className="text-xl font-bold">1. Historical Analysis</h2>
                </div>
                
                {/* Historical Stats Grid */}
                {historicalStats && (
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        <StatsCard 
                            title="SQN (R-based)" 
                            value={historicalStats.sqn.available ? historicalStats.sqn.score.toFixed(2) : '--'} 
                            subValue={`${historicalStats.sqn.rating} · ${historicalStats.sqn.methodLabel}`} 
                            icon={Trophy}
                            color={
                              !historicalStats.sqn.available
                                ? 'warning'
                                : historicalStats.sqn.score > 2.5
                                  ? 'success'
                                  : historicalStats.sqn.score > 1.6
                                    ? 'warning'
                                    : 'danger'
                            }
                        />
                        <StatsCard 
                            title="Win Rate" 
                            value={`${historicalStats.winRate.toFixed(1)}%`} 
                            icon={Target}
                            color="default"
                        />
                        <StatsCard 
                            title="Profit Factor" 
                            value={historicalStats.profitFactor.toFixed(2)} 
                            icon={Scale}
                            color="default"
                        />
                        <StatsCard 
                            title="Sharpe Ratio" 
                            value={historicalStats.sharpe.toFixed(2)} 
                            icon={Activity}
                            color="default"
                        />
                    </div>
                )}

                {/* ROW 1: Equity & Distribution (Equal Height Side-by-Side) */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-stretch">
                    <HistoricalEquityChart trades={trades} />
                    <DistributionChart 
                        data={trades.map(t => t.pnl)} 
                        title="PnL Distribution (Simplified)" 
                        unit="$" 
                        color="#3b82f6" 
                    />
                </div>

                {/* ROW 2: Regression & MAE/MFE */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-stretch">
                    <RegressionAnalysis trades={trades} />
                    <ScatterMAEMFE trades={trades} />
                </div>
                
                {/* ROW 3: Rolling Stats (Full Width) */}
                <div className="w-full">
                     <RollingStatsChart trades={trades} />
                </div>

                {/* ROW 4: Underwater (Full Width) */}
                <div className="w-full">
                    <UnderwaterChart trades={trades} />
                </div>

                {/* ROW 5: Full Width 3D Scatter */}
                <div className="w-full">
                    <ThreeDScatter trades={trades} />
                </div>
              </section>

              {/* SECTION 2: MONTE CARLO */}
              <section className="space-y-6">
                 <div className="flex items-center gap-2 border-b border-neutral-800 pb-2">
                    <Activity className="text-amber-500 w-5 h-5" />
                    <h2 className="text-xl font-bold">2. Monte Carlo Simulation</h2>
                </div>
                
                {mcResults.length > 0 ? (
                    <>
                        {mcStats && (
                            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                                <div className="col-span-1">
                                     <StatsCard title="Median Equity" value={`$${Math.round(mcStats.medianEquity).toLocaleString()}`} icon={TrendingUp} color="success" />
                                </div>
                                <div className="col-span-1">
                                     <StatsCard 
                                        title="Median CAGR" 
                                        value={`${mcStats.cagrMedian.toFixed(1)}%`} 
                                        subValue={`vs S&P: ${mcStats.cagrVsSP500 > 0 ? '+' : ''}${mcStats.cagrVsSP500.toFixed(1)}%`}
                                        icon={Percent} 
                                        color={mcStats.cagrMedian > mcStats.sp500Benchmark ? 'success' : 'warning'} 
                                     />
                                </div>
                                <div className="col-span-1">
                                    <StatsCard title="Sim Sharpe" value={mcStats.sharpeRatio.toFixed(2)} icon={Scale} color="default" />
                                </div>
                                <div className="col-span-1">
                                    <StatsCard title="Sim Sortino" value={mcStats.sortinoRatio.toFixed(2)} icon={Scale} color="default" />
                                </div>
                                <div className="col-span-1">
                                    <StatsCard title="VaR 95%" value={`${mcStats.var95.toFixed(1)}%`} subValue="Risk" icon={ShieldAlert} color="warning" />
                                </div>
                                <div className="col-span-1">
                                    <StatsCard title="VaR 99%" value={`${mcStats.var99.toFixed(1)}%`} subValue="Extreme Risk" icon={ShieldAlert} color="danger" />
                                </div>
                                <div className="col-span-1">
                                    <StatsCard title={`VaR ${mcConfig.confidenceLevel}%`} value={`${mcStats.varCustom.toFixed(1)}%`} subValue="Custom" icon={ShieldAlert} color="warning" />
                                </div>
                                <div className="col-span-1">
                                    <StatsCard title="Max Drawdown" value={`${mcStats.worstDrawdown.toFixed(1)}%`} icon={BarChart3} color="danger" />
                                </div>
                            </div>
                        )}
                        
                        {/* Main Equity Curve */}
                        <div className="bg-surface rounded-xl p-4 border border-neutral-800 shadow-xl">
                            <h4 className="text-neutral-300 text-sm font-semibold mb-4 ml-2">Simulated Equity Projections {mcConfig.riskModel === 'percent_equity' && '(Compounding)'}</h4>
                            <EquityChart results={mcResults} limitLines={50} />
                        </div>

                        {/* Probability Heatmap */}
                        <div className="w-full">
                            <ProbabilityHeatmap results={mcResults} />
                        </div>

                        {/* Streak Probability Analysis (NEW) */}
                        {historicalStats && (
                            <div className="w-full">
                                <StreakProbabilityChart results={mcResults} winRate={historicalStats.winRate} />
                            </div>
                        )}

                        {/* Distributions Grid */}
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-stretch">
                            <DistributionChart 
                                data={mcResults.map(r => r.finalEquity)} 
                                title="Ending Equity Distribution" 
                                unit="$" 
                                color="#10b981" 
                            />
                            <DistributionChart 
                                data={mcResults.map(r => r.maxDrawdownPercent)} 
                                title="Max Drawdown Distribution" 
                                unit="%" 
                                color="#ef4444" 
                            />
                            {/* Full width row for VaR Curve to prevent odd-number gaps */}
                            <div className="md:col-span-2">
                                <VaRCurveChart drawdowns={mcResults.map(r => r.maxDrawdownPercent)} />
                            </div>
                            <DistributionChart
                                data={mcResults.map(r => r.maxConsecutiveLosses)}
                                title="Max Consecutive Losses Dist."
                                unit=""
                                color="#f59e0b"
                                bins={10}
                            />
                            <DistributionChart
                                data={mcResults.map(r => r.winRate)}
                                title="Win Rate Distribution"
                                unit="%"
                                color="#3b82f6"
                            />
                        </div>
                    </>
                ) : (
                    <div className="bg-neutral-900/50 border border-dashed border-neutral-800 rounded-xl h-40 flex items-center justify-center text-neutral-500">
                        <p>Run simulation to view Monte Carlo projections.</p>
                    </div>
                )}
              </section>

              {/* SECTION 3: PROP FIRM */}
              <section className="space-y-6">
                 <div className="flex items-center gap-2 border-b border-neutral-800 pb-2">
                    <GraduationCap className="text-emerald-500 w-5 h-5" />
                    <h2 className="text-xl font-bold">3. Prop Firm Stress Test</h2>
                </div>

                {propStats ? (
                    <PropFirmDashboard stats={propStats} />
                ) : (
                     <div className="bg-neutral-900/50 border border-dashed border-neutral-800 rounded-xl h-40 flex flex-col items-center justify-center text-neutral-500">
                        <p>Waiting for Monte Carlo to complete...</p>
                        <p className="text-xs mt-2 text-neutral-600">Will run automatically after main simulation.</p>
                    </div>
                )}
              </section>

               {/* SECTION 4: AI ANALYSIS */}
              <section className="space-y-6">
                 <div className="flex items-center gap-2 border-b border-neutral-800 pb-2">
                    <BrainCircuit className="text-purple-500 w-5 h-5" />
                    <h2 className="text-xl font-bold">4. AI Strategy Report</h2>
                </div>
                
                <div className="w-full">
                    <AIReportCard 
                        trades={trades}
                        historicalStats={historicalStats} 
                        mcStats={mcStats} 
                        mcConfig={mcConfig} 
                    />
                </div>
              </section>

            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
