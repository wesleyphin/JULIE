import { Trade, SimulationConfig, SimulationResult, Statistics } from '../types';

// Mulberry32 - A simple, fast, seedable PRNG
export const createPRNG = (seed: number) => {
  let state = seed;
  return () => {
    state |= 0; 
    state = state + 0x6D2B79F5 | 0;
    let t = Math.imul(state ^ state >>> 15, 1 | state);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
};

export const runSimulationBatch = (
  trades: Trade[],
  config: SimulationConfig,
  batchSize: number,
  startId: number,
  randomGenerator: () => number // Pass the seeded generator
): SimulationResult[] => {
  const { initialEquity, tradesPerSimulation, riskModel } = config;
  const results: SimulationResult[] = [];
  
  // Pre-process pool based on risk model
  // If fixed_pnl: use absolute pnl values
  // If percent_equity: use pnlPercent (decimal)
  const pnlPool = trades.map(t => riskModel === 'percent_equity' ? (t.pnlPercent / 100) : t.pnl);

  if (pnlPool.length === 0) return [];

  for (let i = 0; i < batchSize; i++) {
    let currentEquity = initialEquity;
    let peakEquity = initialEquity;
    let maxDrawdown = 0;
    let winCount = 0;
    let grossProfit = 0;
    let grossLoss = 0;
    const equityCurve = [initialEquity];
    let isRuined = false;

    let currentConsecutiveLosses = 0;
    let maxConsecutiveLosses = 0;
    
    // Store PnLs for Sharpe/Sortino calc
    const simulationPnls: number[] = [];

    for (let j = 0; j < tradesPerSimulation; j++) {
      // Use the seeded generator instead of Math.random()
      const randomIndex = Math.floor(randomGenerator() * pnlPool.length);
      const tradeValue = pnlPool[randomIndex];

      let tradePnl = 0;
      
      if (riskModel === 'percent_equity') {
         // Compound Growth: Eq = Eq * (1 + return)
         // tradeValue is percentage (e.g., 0.02 for 2%)
         const pnlAmount = currentEquity * tradeValue;
         currentEquity += pnlAmount;
         tradePnl = pnlAmount;
      } else {
         // Fixed Lots
         currentEquity += tradeValue;
         tradePnl = tradeValue;
      }
      
      equityCurve.push(currentEquity);
      simulationPnls.push(tradePnl);

      // Max Drawdown calc (Peak-to-Valley)
      if (currentEquity > peakEquity) {
        peakEquity = currentEquity;
      }
      const drawdown = peakEquity - currentEquity;
      if (drawdown > maxDrawdown) {
        maxDrawdown = drawdown;
      }

      if (tradePnl > 0) {
        winCount++;
        grossProfit += tradePnl;
        currentConsecutiveLosses = 0;
      } else {
        grossLoss += Math.abs(tradePnl);
        // Treat 0 as loss for consecutive purposes or just negative? Usually <= 0
        if (tradePnl <= 0) {
            currentConsecutiveLosses++;
            if (currentConsecutiveLosses > maxConsecutiveLosses) {
                maxConsecutiveLosses = currentConsecutiveLosses;
            }
        }
      }

      if (currentEquity <= 0) {
        isRuined = true;
        // Break early if ruined to save cycles? Or keep going to see debt depth?
        // Usually break early in realistic sims, but for curve completeness we might continue.
        // Let's break early to prevent NaN spirals if equity goes negative in % model
        if (riskModel === 'percent_equity' && currentEquity <= 1) break; 
      }
    }

    // Accurate Max Drawdown % Calculation
    let localPeak = initialEquity;
    let maxDDPercent = 0;
    
    for(const eq of equityCurve) {
        if (eq > localPeak) localPeak = eq;
        const dd = (localPeak - eq) / localPeak;
        if (dd > maxDDPercent) maxDDPercent = dd;
    }

    // --- Calculate Ratios (Per Trade Basis) ---
    // Average PnL per trade
    const avgPnl = (currentEquity - initialEquity) / equityCurve.length;
    
    let sumSqDiff = 0;
    let sumSqDownside = 0;
    
    for(const p of simulationPnls) {
        sumSqDiff += Math.pow(p - avgPnl, 2);
        // Sortino uses downside deviation relative to 0 (or target return)
        if (p < 0) sumSqDownside += Math.pow(p, 2); 
    }
    
    const stdDev = Math.sqrt(sumSqDiff / simulationPnls.length);
    const downsideDev = Math.sqrt(sumSqDownside / simulationPnls.length);
    
    // Ratios (per trade)
    const sharpeRatio = stdDev === 0 ? 0 : avgPnl / stdDev;
    const sortinoRatio = downsideDev === 0 ? 0 : avgPnl / downsideDev;

    results.push({
      id: startId + i,
      equityCurve,
      finalEquity: currentEquity,
      maxDrawdown,
      maxDrawdownPercent: maxDDPercent * 100,
      winRate: (winCount / simulationPnls.length) * 100,
      profitFactor: grossLoss === 0 ? grossProfit : grossProfit / grossLoss,
      maxConsecutiveLosses,
      isRuined,
      sharpeRatio,
      sortinoRatio
    });
  }

  return results;
};

// Kept for backward compatibility, defaults to random seed
export const runMonteCarlo = (
  trades: Trade[],
  config: SimulationConfig
): SimulationResult[] => {
  const rng = createPRNG(Date.now());
  return runSimulationBatch(trades, config, config.numSimulations, 0, rng);
};

export const calculateStatistics = (
  results: SimulationResult[], 
  initialEquity: number, 
  durationYears: number = 1,
  confidenceLevel: number = 95
): Statistics => {
  if (results.length === 0) {
    return {
      medianEquity: 0,
      bestEquity: 0,
      worstEquity: 0,
      medianDrawdown: 0,
      worstDrawdown: 0,
      ruinProbability: 0,
      cagr95: 0,
      cagr05: 0,
      cagrMedian: 0,
      var95: 0,
      var99: 0,
      customConfidenceLevel: confidenceLevel,
      varCustom: 0,
      cagrCustom: 0,
      sharpeRatio: 0,
      sortinoRatio: 0,
      sp500Benchmark: 10.5,
      cagrVsSP500: 0
    };
  }

  const SP500_AVG_CAGR = 10.5; // Historical average ~10.5%

  const finalEquities = results.map(r => r.finalEquity).sort((a, b) => a - b);
  // Sort drawdowns ascending (small DD to large DD)
  const drawdowns = results.map(r => r.maxDrawdownPercent).sort((a, b) => a - b);
  const ruinedCount = results.filter(r => r.isRuined).length;

  const getPercentile = (sortedArr: number[], percentile: number) => {
    const index = Math.floor(percentile * sortedArr.length);
    return sortedArr[Math.min(index, sortedArr.length - 1)];
  };

  const calculateCAGR = (finalValue: number) => {
    if (initialEquity <= 0 || finalValue <= 0 || durationYears <= 0) return 0;
    return (Math.pow(finalValue / initialEquity, 1 / durationYears) - 1) * 100;
  };

  const equity05 = getPercentile(finalEquities, 0.05);
  const equity50 = getPercentile(finalEquities, 0.5);
  const equity95 = getPercentile(finalEquities, 0.95);

  const var95 = getPercentile(drawdowns, 0.95);
  const var99 = getPercentile(drawdowns, 0.99);

  // Custom Stats
  const alpha = 1 - (confidenceLevel / 100);
  const varCustom = getPercentile(drawdowns, confidenceLevel / 100);
  const equityCustom = getPercentile(finalEquities, alpha);

  // Ratios (Annualized)
  const sortedSharpes = results.map(r => r.sharpeRatio).sort((a,b) => a-b);
  const sortedSortinos = results.map(r => r.sortinoRatio).sort((a,b) => a-b);
  
  const medianSharpe = getPercentile(sortedSharpes, 0.5);
  const medianSortino = getPercentile(sortedSortinos, 0.5);

  // Annualization Factor = sqrt(Trades Per Year)
  // Trades Per Year = TradesPerSimulation / SimulatedYears
  // results[0].equityCurve.length represents N trades + 1 initial point
  const tradesPerSim = results[0].equityCurve.length - 1;
  const tradesPerYear = durationYears > 0 ? tradesPerSim / durationYears : tradesPerSim;
  const annualFactor = Math.sqrt(tradesPerYear);

  const cagrMedian = calculateCAGR(equity50);

  return {
    medianEquity: equity50,
    bestEquity: finalEquities[finalEquities.length - 1],
    worstEquity: finalEquities[0],
    medianDrawdown: getPercentile(drawdowns, 0.5),
    worstDrawdown: drawdowns[drawdowns.length - 1],
    ruinProbability: (ruinedCount / results.length) * 100,
    cagr05: calculateCAGR(equity05), 
    cagrMedian: cagrMedian,
    cagr95: calculateCAGR(equity95),
    var95,
    var99,
    customConfidenceLevel: confidenceLevel,
    varCustom,
    cagrCustom: calculateCAGR(equityCustom),
    sharpeRatio: medianSharpe * annualFactor,
    sortinoRatio: medianSortino * annualFactor,
    sp500Benchmark: SP500_AVG_CAGR,
    cagrVsSP500: cagrMedian - SP500_AVG_CAGR
  };
};