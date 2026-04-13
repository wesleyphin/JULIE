export interface Trade {
  id: string;
  entryTime: string;
  exitTime: string;
  type: 'Long' | 'Short';
  pnl: number;
  pnlPercent: number;
  entryPrice?: number;
  exitPrice?: number;
  mae?: number; // Maximum Adverse Excursion (Drawdown during trade, usually negative)
  mfe?: number; // Maximum Favorable Excursion (Run-up during trade, positive)
  riskAmount?: number;
  riskPoints?: number;
  rMultiple?: number;
  riskSource?: 'r_multiple' | 'points' | 'currency';
}

export interface SimulationConfig {
  initialEquity: number;
  numSimulations: number;
  tradesPerSimulation: number;
  seed: number;
  convergenceTolerance: number; 
  confidenceLevel: number;
  riskModel: 'fixed_pnl' | 'percent_equity'; // New: Position Sizing
}

// New Prop Firm Specific Types
export interface PropFirmConfig {
  accountSize: number;
  maxDailyLoss: number; // DLL (e.g., 1000)
  maxTotalLoss: number; // MLL (e.g., 2000)
  profitTargetEval: number; // Target to pass Eval (e.g., 3000)
  maxDailyProfitEval: number; // Max daily PnL contribution for Eval (e.g., 1500)
  expressPayoutThreshold: number; // Min daily profit to count as a "profitable day" (e.g., 150)
  expressPayoutsRequired: number; // Number of payouts to reach Live (e.g., 5)
  expressDaysForPayout: number; // Days > Threshold required for one payout (e.g., 5)
  tradesPerDay: number; // Estimated trades per day for simulation
}

export interface PropFirmAggregateStats {
  totalTraders: number;
  
  // Eval Phase
  evalAttempts: number;
  evalBlown: number;
  evalPassed: number;
  evalPassRate: number;
  
  // Express Phase
  expressAttempts: number;
  expressBlown: number;
  expressPayoutsTotal: number;
  expressPassRate: number; // Rate of getting at least one payout? Or reaching live?
  
  // Live Phase
  liveReached: number;
  liveReachedRate: number;

  // Efficiency Stats
  avgTrialsToExpress: number;
  avgTrialsToFirstPayout: number;
  avgTrialsToLive: number;
  blownAfterFirstPayout: number;
}

export interface SimulationResult {
  id: number;
  equityCurve: number[];
  finalEquity: number;
  maxDrawdown: number;
  maxDrawdownPercent: number;
  winRate: number;
  profitFactor: number;
  maxConsecutiveLosses: number;
  isRuined: boolean; // equity <= 0
  sharpeRatio: number;
  sortinoRatio: number;
}

export interface Statistics {
  medianEquity: number;
  bestEquity: number;
  worstEquity: number;
  medianDrawdown: number;
  worstDrawdown: number;
  ruinProbability: number;
  cagr95: number; // 95th percentile outcome (optimistic) percentage
  cagr05: number; // 5th percentile outcome (pessimistic) percentage
  cagrMedian: number; // Median outcome percentage
  var95: number; // Value at Risk (95% confidence Max Drawdown)
  var99: number; // Value at Risk (99% confidence Max Drawdown)
  
  // Custom Confidence Stats
  customConfidenceLevel: number;
  varCustom: number; // VaR at user defined confidence
  cagrCustom: number; // CAGR at the lower bound of confidence (e.g. 5th percentile for 95% conf)

  // Risk Ratios
  sharpeRatio: number;
  sortinoRatio: number;

  // Benchmark
  sp500Benchmark: number;
  cagrVsSP500: number;
}
