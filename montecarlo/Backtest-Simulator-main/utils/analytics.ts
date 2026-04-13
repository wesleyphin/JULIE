import { Trade } from '../types';

export interface RegressionResult {
  feature: string;
  coefficient: number; // Slope
  correlation: number; // Pearson r
  importance: number; // |r|
  rSquared: number;
  tStat?: number;
}

export interface SegmentedAnalysis {
  all: RegressionResult[];
  long: RegressionResult[];
  short: RegressionResult[];
}

export interface RollingStat {
    index: number;
    winRate: number;
    sharpe: number;
    avgPnl: number;
}

export interface SQNResult {
  available: boolean;
  score: number;
  rating: string;
  tradeCount: number;
  effectiveTradeCount: number;
  methodLabel: string;
}

const mean = (arr: number[]) => arr.reduce((a, b) => a + b, 0) / arr.length;

const stdDev = (arr: number[], mu: number) => {
  if (arr.length <= 1) return 0;
  const variance = arr.reduce((acc, val) => acc + Math.pow(val - mu, 2), 0) / (arr.length - 1);
  return Math.sqrt(variance);
};

// Calculate Pearson Correlation and Simple Linear Regression Slope
const analyzeFeature = (x: number[], y: number[], featureName: string): RegressionResult => {
  const n = x.length;
  if (n < 2) return { feature: featureName, coefficient: 0, correlation: 0, importance: 0, rSquared: 0 };

  // Check variance
  const uniqueX = new Set(x);
  if (uniqueX.size < 2) return { feature: featureName, coefficient: 0, correlation: 0, importance: 0, rSquared: 0 };

  const muX = mean(x);
  const muY = mean(y);
  
  const stdX = stdDev(x, muX);
  const stdY = stdDev(y, muY);

  let covariance = 0;
  for (let i = 0; i < n; i++) {
    covariance += (x[i] - muX) * (y[i] - muY);
  }
  covariance /= (n - 1);

  const correlation = (stdX * stdY) === 0 ? 0 : covariance / (stdX * stdY);
  const slope = (stdX === 0) ? 0 : correlation * (stdY / stdX);

  return {
    feature: featureName,
    coefficient: slope,
    correlation,
    importance: Math.abs(correlation),
    rSquared: correlation * correlation
  };
};

const runAnalysisOnSubset = (trades: Trade[]): RegressionResult[] => {
  if (trades.length < 5) return [];

  const y = trades.map(t => t.pnl);

  // Feature 1: Entry Hour
  const xHour = trades.map(t => {
    const d = new Date(t.entryTime);
    return isNaN(d.getTime()) ? 12 : d.getHours();
  });

  // Feature 2: Day of Week (Monday=1...Sunday=7)
  const xDay = trades.map(t => {
    const d = new Date(t.entryTime);
    const day = d.getDay(); 
    return day === 0 ? 7 : day;
  });

  // Feature 3: Trade Sequence (Trend)
  const xSequence = trades.map((_, i) => i);

  // Feature 4: Previous Trade Result (Win/Loss streakiness)
  const xPrevResult = trades.map((_, i) => {
      if (i === 0) return 0;
      return trades[i-1].pnl > 0 ? 1 : -1;
  });

  const results = [
    analyzeFeature(xHour, y, "Entry Hour"),
    analyzeFeature(xDay, y, "Day of Week"),
    analyzeFeature(xSequence, y, "Sequence (Trend)"),
    analyzeFeature(xPrevResult, y, "Prev. Outcome")
  ];

  return results.sort((a, b) => b.importance - a.importance);
};

export const performRegressionAnalysis = (trades: Trade[]): SegmentedAnalysis => {
  const longs = trades.filter(t => t.type === 'Long');
  const shorts = trades.filter(t => t.type === 'Short');

  return {
    all: runAnalysisOnSubset(trades),
    long: runAnalysisOnSubset(longs),
    short: runAnalysisOnSubset(shorts)
  };
};

export const calculateDistributionStats = (values: number[]) => {
  const n = values.length;
  if (n < 2) return { skew: 0, kurtosis: 0, mean: 0, stdDev: 0 };

  const mu = values.reduce((a, b) => a + b, 0) / n;
  
  let m2 = 0;
  let m3 = 0;
  let m4 = 0;

  for (const val of values) {
    const delta = val - mu;
    m2 += Math.pow(delta, 2);
    m3 += Math.pow(delta, 3);
    m4 += Math.pow(delta, 4);
  }

  m2 /= n;
  m3 /= n;
  m4 /= n;

  const stdDev = Math.sqrt(m2);
  
  if (m2 === 0) return { skew: 0, kurtosis: 0, mean: mu, stdDev: 0 };

  const skew = m3 / Math.pow(m2, 1.5);
  const kurtosis = (m4 / Math.pow(m2, 2)) - 3; // Excess Kurtosis

  return { skew, kurtosis, mean: mu, stdDev };
};

// --- New Analytics for v2 ---

const SQN_EFFECTIVE_TRADE_CAP = 100;

export const calculateSQN = (trades: Trade[]): SQNResult => {
    const rMultiples = trades
      .map((trade) => trade.rMultiple)
      .filter((value): value is number => value != null && Number.isFinite(value));

    if (rMultiples.length < 2) {
      return {
        available: false,
        score: 0,
        rating: 'Risk Data Missing',
        tradeCount: rMultiples.length,
        effectiveTradeCount: 0,
        methodLabel: 'R-multiple stop/risk data required',
      };
    }

    const avgR = mean(rMultiples);
    const devR = stdDev(rMultiples, avgR);
    if (devR === 0) {
      return {
        available: false,
        score: 0,
        rating: 'N/A',
        tradeCount: rMultiples.length,
        effectiveTradeCount: Math.min(rMultiples.length, SQN_EFFECTIVE_TRADE_CAP),
        methodLabel: 'R-multiple series has zero variance',
      };
    }

    const effectiveTradeCount = Math.min(rMultiples.length, SQN_EFFECTIVE_TRADE_CAP);
    const sqn = (avgR / devR) * Math.sqrt(effectiveTradeCount);

    let rating = 'Poor';
    if (sqn < 1.6) rating = 'Poor';
    else if (sqn < 2.0) rating = 'Average';
    else if (sqn < 2.5) rating = 'Good';
    else if (sqn < 3.0) rating = 'Very Good';
    else if (sqn < 5.0) rating = 'Excellent';
    else if (sqn < 7.0) rating = 'Superb';
    else rating = 'Holy Grail';

    return {
      available: true,
      score: sqn,
      rating,
      tradeCount: rMultiples.length,
      effectiveTradeCount,
      methodLabel:
        rMultiples.length > SQN_EFFECTIVE_TRADE_CAP
          ? `R-multiple, capped at ${SQN_EFFECTIVE_TRADE_CAP} trades`
          : `R-multiple, ${effectiveTradeCount} trades`,
    };
};

export const calculateRollingStats = (trades: Trade[], windowSize: number = 50): RollingStat[] => {
    if (trades.length < windowSize) return [];

    const stats: RollingStat[] = [];
    const pnls = trades.map(t => t.pnl);

    for (let i = windowSize; i <= pnls.length; i++) {
        const window = pnls.slice(i - windowSize, i);
        
        // Win Rate
        const wins = window.filter(p => p > 0).length;
        const winRate = (wins / windowSize) * 100;

        // Sharpe (simplified, assuming risk free = 0)
        const avg = mean(window);
        const dev = stdDev(window, avg);
        const sharpe = dev === 0 ? 0 : avg / dev; // Not annualized here, just relative stability

        stats.push({
            index: i,
            winRate,
            sharpe,
            avgPnl: avg
        });
    }

    return stats;
};
