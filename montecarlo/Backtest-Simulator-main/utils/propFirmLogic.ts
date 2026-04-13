import { Trade, PropFirmConfig, PropFirmAggregateStats } from '../types';
import { createPRNG } from './monteCarlo';

// Helper to simulate a single day of trading
const simulateDay = (
  pnlPool: number[], 
  rng: () => number, 
  tradesPerDay: number
): number => {
  let dailyPnL = 0;
  // Determine number of trades for this day (randomized slightly around the avg)
  // Simple approach: Fixed number or Poisson-like. Let's use simple randomization 1 to 2x.
  const numTrades = Math.max(1, Math.round(tradesPerDay * (0.5 + rng()))); 
  
  for (let i = 0; i < numTrades; i++) {
    const idx = Math.floor(rng() * pnlPool.length);
    dailyPnL += pnlPool[idx];
  }
  return dailyPnL;
};

export const runPropFirmSimulation = (
  trades: Trade[],
  config: PropFirmConfig,
  numTraders: number = 1000,
  seed: number = 12345
): PropFirmAggregateStats => {
  const pnlPool = trades.map(t => t.pnl);
  const rng = createPRNG(seed);

  // Aggregators
  let evalAttempts = 0;
  let evalBlown = 0;
  let evalPassed = 0;
  
  let expressAttempts = 0;
  let expressBlown = 0;
  let expressPayoutsTotal = 0;
  
  let liveReached = 0;
  let blownAfterFirstPayout = 0;

  // Trial counters for averages
  const trialsToExpress: number[] = [];
  const trialsToFirstPayout: number[] = [];
  const trialsToLive: number[] = [];

  // Simulation Constants
  const MAX_CAREER_DAYS = 2000; // Cap to prevent infinite loops
  const MAX_RESTARTS = 100; // Cap restarts per trader

  for (let t = 0; t < numTraders; t++) {
    let phase: 'EVAL' | 'EXPRESS' | 'LIVE' = 'EVAL';
    let balance = config.accountSize;
    let currentPhaseAttempts = 0;
    
    // Eval Specific State
    let evalProfitProgress = 0;

    // Express Specific State
    let expressDaysAboveThreshold = 0;
    let expressPayoutsCount = 0;

    // Career Tracking
    let totalAttempts = 1; // Starts with 1
    let careerActive = true;
    let day = 0;
    
    // Track stats for this specific trader
    let hasReachedExpress = false;
    let hasReachedFirstPayout = false;
    let hasReachedLive = false;

    // If restart happened
    let attemptsToReachExpress = 0;
    let attemptsToReachFirstPayout = 0;
    let attemptsToReachLive = 0;

    evalAttempts++; 
    attemptsToReachExpress++;
    attemptsToReachFirstPayout++;
    attemptsToReachLive++;

    while (careerActive && day < MAX_CAREER_DAYS && totalAttempts <= MAX_RESTARTS) {
      day++;
      const dailyPnL = simulateDay(pnlPool, rng, config.tradesPerDay);

      // --- 1. Check Daily Loss Limit (DLL) ---
      if (dailyPnL <= -config.maxDailyLoss) {
        // Blown by DLL
        if (phase === 'EVAL') {
          evalBlown++;
          // Restart logic
          phase = 'EVAL'; 
          balance = config.accountSize; // Reset Balance
          evalProfitProgress = 0; // Reset Progress
          evalAttempts++;
          totalAttempts++;
          if (!hasReachedExpress) attemptsToReachExpress++;
          if (!hasReachedFirstPayout) attemptsToReachFirstPayout++;
          if (!hasReachedLive) attemptsToReachLive++;
        } else if (phase === 'EXPRESS') {
          expressBlown++;
          if (hasReachedFirstPayout) blownAfterFirstPayout++;
          // Restart from Step 1
          phase = 'EVAL'; 
          balance = config.accountSize; // Reset Balance
          evalProfitProgress = 0; // Reset Progress
          expressPayoutsCount = 0;
          expressDaysAboveThreshold = 0;
          evalAttempts++; 
          expressAttempts++; 
          totalAttempts++;
          
          if (!hasReachedFirstPayout) attemptsToReachFirstPayout++;
          if (!hasReachedLive) attemptsToReachLive++;
        }
        continue; // Next day (which is effectively day 1 of new attempt)
      }

      // --- 2. Update Balance ---
      balance += dailyPnL;

      // --- 3. Check Max Loss Limit (MLL) ---
      // Fixed drawdown from initial balance
      const lossLimitLevel = config.accountSize - config.maxTotalLoss;
      if (balance < lossLimitLevel) {
        // Blown by MLL
        if (phase === 'EVAL') {
          evalBlown++;
          phase = 'EVAL';
          balance = config.accountSize; // Reset Balance
          evalProfitProgress = 0; // Reset Progress
          evalAttempts++;
          totalAttempts++;
          if (!hasReachedExpress) attemptsToReachExpress++;
          if (!hasReachedFirstPayout) attemptsToReachFirstPayout++;
          if (!hasReachedLive) attemptsToReachLive++;
        } else if (phase === 'EXPRESS') {
          expressBlown++;
          if (hasReachedFirstPayout) blownAfterFirstPayout++;
          phase = 'EVAL';
          balance = config.accountSize; // Reset Balance
          evalProfitProgress = 0; // Reset Progress
          expressPayoutsCount = 0;
          expressDaysAboveThreshold = 0;
          evalAttempts++;
          expressAttempts++;
          totalAttempts++;

          if (!hasReachedFirstPayout) attemptsToReachFirstPayout++;
          if (!hasReachedLive) attemptsToReachLive++;
        }
        continue;
      }

      // --- 4. Process Phase Logic ---

      if (phase === 'EVAL') {
        // Profit Target Logic
        // We apply the "Consistency Rule" (Max Daily contribution) BUT we must account for losses (Net PnL).
        // If daily PnL is negative, it fully deducts from progress.
        // If daily PnL is positive, it is capped by maxDailyProfitEval.
        
        let dailyContribution = dailyPnL;
        if (dailyContribution > config.maxDailyProfitEval) {
            dailyContribution = config.maxDailyProfitEval;
        }
        
        evalProfitProgress += dailyContribution;

        if (evalProfitProgress >= config.profitTargetEval) {
          // Passed Eval
          evalPassed++;
          phase = 'EXPRESS';
          expressAttempts++;
          balance = config.accountSize; // Reset Balance for new Phase
          evalProfitProgress = 0;
          expressDaysAboveThreshold = 0;
          expressPayoutsCount = 0;
          
          if (!hasReachedExpress) {
            hasReachedExpress = true;
            trialsToExpress.push(attemptsToReachExpress);
          }
        }
      } 
      else if (phase === 'EXPRESS') {
        // "Five profitable days of $150 or more = first payout"
        if (dailyPnL >= config.expressPayoutThreshold) {
          expressDaysAboveThreshold++;
        }

        if (expressDaysAboveThreshold >= config.expressDaysForPayout) {
          // PAYOUT!
          expressPayoutsTotal++;
          expressPayoutsCount++;
          expressDaysAboveThreshold = 0; // Reset days for next payout
          
          if (!hasReachedFirstPayout) {
            hasReachedFirstPayout = true;
            trialsToFirstPayout.push(attemptsToReachFirstPayout);
          }

          // "After 5 payouts initiate phase 3 (Live)"
          if (expressPayoutsCount >= config.expressPayoutsRequired) {
             phase = 'LIVE';
             liveReached++;
             careerActive = false; // Stop simulation for this trader as they reached the goal
             
             if (!hasReachedLive) {
                hasReachedLive = true;
                trialsToLive.push(attemptsToReachLive);
             }
          }
        }
      }
    }
  }

  const avg = (arr: number[]) => arr.length > 0 ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;

  return {
    totalTraders: numTraders,
    evalAttempts,
    evalBlown,
    evalPassed,
    evalPassRate: evalAttempts > 0 ? (evalPassed / evalAttempts) * 100 : 0,
    
    expressAttempts,
    expressBlown,
    expressPayoutsTotal,
    expressPassRate: expressAttempts > 0 ? (liveReached / expressAttempts) * 100 : 0, // Roughly rate of finishing express

    liveReached,
    liveReachedRate: (liveReached / numTraders) * 100,

    avgTrialsToExpress: avg(trialsToExpress),
    avgTrialsToFirstPayout: avg(trialsToFirstPayout),
    avgTrialsToLive: avg(trialsToLive),
    blownAfterFirstPayout
  };
};