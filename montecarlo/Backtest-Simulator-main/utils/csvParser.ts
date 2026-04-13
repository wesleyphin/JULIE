import { Trade } from '../types';

const normalizeHeader = (value: string): string => value.toLowerCase().replace(/[^a-z0-9]+/g, '');

const getColumnIndex = (colMap: Record<string, number>, candidates: string[]): number | undefined => {
  for (const candidate of candidates) {
    if (colMap[candidate] !== undefined) return colMap[candidate];
  }
  const normalizedCandidates = new Set(candidates.map(normalizeHeader));
  for (const [key, idx] of Object.entries(colMap)) {
    if (normalizedCandidates.has(normalizeHeader(key))) return idx;
  }
  return undefined;
};

const parseNumeric = (raw?: string): number | undefined => {
  if (!raw) return undefined;
  const parsed = parseFloat(raw.replace(/[$,%]/g, '').replace(/,/g, ''));
  return Number.isFinite(parsed) ? parsed : undefined;
};

const extractRiskMetrics = (
  row: string[],
  indices: {
    rMultipleCol?: number;
    riskAmountCol?: number;
    riskPointsCol?: number;
    pnlPointsCol?: number;
  },
  pnl: number,
): Pick<Trade, 'riskAmount' | 'riskPoints' | 'rMultiple' | 'riskSource'> => {
  const explicitRMultiple = indices.rMultipleCol !== undefined ? parseNumeric(row[indices.rMultipleCol]) : undefined;
  if (explicitRMultiple !== undefined) {
    return {
      rMultiple: explicitRMultiple,
      riskSource: 'r_multiple',
    };
  }

  const riskPoints = indices.riskPointsCol !== undefined ? parseNumeric(row[indices.riskPointsCol]) : undefined;
  const pnlPoints = indices.pnlPointsCol !== undefined ? parseNumeric(row[indices.pnlPointsCol]) : undefined;
  if (riskPoints !== undefined && riskPoints > 0 && pnlPoints !== undefined) {
    return {
      riskPoints,
      rMultiple: pnlPoints / riskPoints,
      riskSource: 'points',
    };
  }

  const riskAmount = indices.riskAmountCol !== undefined ? parseNumeric(row[indices.riskAmountCol]) : undefined;
  if (riskAmount !== undefined && riskAmount > 0) {
    return {
      riskAmount,
      rMultiple: pnl / riskAmount,
      riskSource: 'currency',
    };
  }

  return {};
};

// Helper for splitting CSV rows respecting quotes
const parseCSVRow = (row: string): string[] => {
  const result: string[] = [];
  let current = '';
  let inQuotes = false;
  
  for (let i = 0; i < row.length; i++) {
    const char = row[i];
    if (char === '"') {
      inQuotes = !inQuotes;
    } else if (char === ',' && !inQuotes) {
      result.push(current.trim());
      current = '';
    } else {
      current += char;
    }
  }
  result.push(current.trim());
  return result;
};

export const parseTradingViewCSV = (csvContent: string): Trade[] => {
  const lines = csvContent.trim().split('\n');
  if (lines.length < 2) return [];

  // Remove BOM if present
  if (lines[0].charCodeAt(0) === 0xFEFF) {
    lines[0] = lines[0].slice(1);
  }

  // Find headers
  const headers = lines[0].split(',').map(h => h.trim().replace(/^"|"$/g, ''));
  
  // -- ERROR HANDLING FOR KNOWN NON-TRADE FORMATS --
  
  // 1. Detect Optimization Reports (Summary Data)
  if (headers.includes('Iteration_ID') && (headers.includes('Total_Trades') || headers.includes('WinRate'))) {
      throw new Error("Detected Strategy Optimization Report. This file contains summary statistics, but the simulator requires a list of individual trades. Please export the 'List of Trades' tab from your backtesting platform.");
  }

  // 2. Detect Market Data (OHLCV)
  // Check for common OHLCV columns without PnL columns
  const hasOHLC = headers.includes('open') && headers.includes('high') && headers.includes('low') && headers.includes('close');
  const hasBento = headers.includes('ts_event') && headers.includes('publisher_id');
  
  if (hasBento || (hasOHLC && !headers.some(h => h.toLowerCase().includes('p&l') || h.toLowerCase().includes('profit') || h.toLowerCase().includes('pnl')))) {
      throw new Error("Detected Market Data (OHLCV). This file contains price data, not trade executions. Please upload a 'List of Trades' CSV to run the simulation.");
  }

  // -- END ERROR HANDLING --

  // Mapping column names to indices
  const colMap: Record<string, number> = {};
  headers.forEach((h, i) => {
    colMap[h] = i;
  });

  // Detect Format Strategy
  if (colMap['net_pnl'] !== undefined) {
     return parseCustomSnakeCase(lines, colMap);
  } else if (colMap['EnteredAt'] !== undefined && colMap['PnL'] !== undefined) {
     return parseCustomGenericCsv(lines, colMap);
  } else if (colMap['Net P&L USD'] !== undefined || colMap['Profit USD'] !== undefined) {
     return parseStandardTradingView(lines, colMap);
  } else {
     // Try to be lenient and look for just PnL column
     const pnlCol = Object.keys(colMap).find(k => k.toLowerCase().includes('p&l') || k.toLowerCase().includes('profit') || k === 'PnL');
     if (pnlCol) {
         return parseStandardTradingView(lines, colMap); // Attempt standard
     }
     
     console.error("Unknown CSV format. Headers found:", headers);
     throw new Error(`Invalid CSV format. Could not detect PnL column (e.g. 'net_pnl', 'PnL', or 'Net P&L USD'). Headers found: ${headers.join(', ')}`);
  }
};

const parseCustomGenericCsv = (lines: string[], colMap: Record<string, number>): Trade[] => {
    const trades: Trade[] = [];
    const riskIndices = {
        rMultipleCol: getColumnIndex(colMap, ['R Multiple', 'R-Multiple', 'r_multiple', 'RMultiple']),
        riskAmountCol: getColumnIndex(colMap, ['Risk USD', 'Risk Amount', 'Initial Risk USD', 'risk_amount', 'risk_per_trade']),
        riskPointsCol: getColumnIndex(colMap, ['SL points', 'Sl points', 'Stop points', 'Risk points', 'sl_points', 'sl_dist', 'SLPoints']),
        pnlPointsCol: getColumnIndex(colMap, ['PnL points', 'Pnl points', 'pnl_points', 'PnLPoints']),
    };

    for (let i = 1; i < lines.length; i++) {
        const row = parseCSVRow(lines[i]);
        if (row.length < Object.keys(colMap).length * 0.5) continue;

        const id = row[colMap['Id']];
        const pnlRaw = row[colMap['PnL']];
        const entryTime = row[colMap['EnteredAt']] || '';
        const exitTime = row[colMap['ExitedAt']] || '';
        const entryPriceRaw = row[colMap['EntryPrice']];
        const exitPriceRaw = row[colMap['ExitPrice']];
        const typeRaw = row[colMap['Type']];

        if (!pnlRaw) continue;

        const pnl = parseFloat(pnlRaw.replace(/[$,]/g, ''));
        if (isNaN(pnl)) continue;

        const entryPrice = parseFloat(entryPriceRaw);
        const exitPrice = parseFloat(exitPriceRaw);

        let type: 'Long' | 'Short' = 'Long';
        if (typeRaw && typeRaw.toLowerCase().includes('short')) {
            type = 'Short';
        }

        let pnlPercent = 0;
        if (!isNaN(entryPrice) && !isNaN(exitPrice) && entryPrice !== 0) {
            if (type === 'Long') {
                pnlPercent = ((exitPrice - entryPrice) / entryPrice) * 100;
            } else {
                pnlPercent = ((entryPrice - exitPrice) / entryPrice) * 100;
            }
        }

        // Fallback MAE/MFE as they are not explicitly in this format
        let mfe: number | undefined = undefined;
        let mae: number | undefined = undefined;
        
        if (pnl >= 0) {
            mfe = pnl;
            mae = 0;
        } else {
            mfe = 0;
            mae = pnl;
        }

        const riskMetrics = extractRiskMetrics(row, riskIndices, pnl);

        trades.push({
            id: id || i.toString(),
            entryTime,
            exitTime,
            type,
            pnl,
            pnlPercent,
            entryPrice: isNaN(entryPrice) ? undefined : entryPrice,
            exitPrice: isNaN(exitPrice) ? undefined : exitPrice,
            mae,
            mfe,
            ...riskMetrics,
        });
    }

    return trades;
};

const parseCustomSnakeCase = (lines: string[], colMap: Record<string, number>): Trade[] => {
    const trades: Trade[] = [];
    const riskIndices = {
        rMultipleCol: getColumnIndex(colMap, ['r_multiple', 'R Multiple', 'R-Multiple']),
        riskAmountCol: getColumnIndex(colMap, ['risk_amount', 'risk_per_trade', 'Risk USD', 'Initial Risk USD']),
        riskPointsCol: getColumnIndex(colMap, ['sl_dist', 'sl_points', 'SL points', 'Stop points', 'Risk points']),
        pnlPointsCol: getColumnIndex(colMap, ['pnl_points', 'PnL points', 'Pnl points']),
    };
    
    for (let i = 1; i < lines.length; i++) {
        const row = parseCSVRow(lines[i]);
        if (row.length < Object.keys(colMap).length * 0.5) continue; // Skip malformed lines

        const pnlRaw = row[colMap['net_pnl']];
        const sideRaw = row[colMap['side']] || 'Long';
        const entryTime = row[colMap['entry_time']] || '';
        const exitTime = row[colMap['exit_time']] || entryTime;
        
        // Price data
        const entryPriceRaw = row[colMap['entry_price']];
        const exitPriceRaw = row[colMap['exit_price']];
        
        // MAE/MFE
        const runUpRaw = row[colMap['run_up_usd']] || row[colMap['run_up']];
        const drawDownRaw = row[colMap['drawdown_usd']] || row[colMap['drawdown']];

        if (!pnlRaw) continue;

        const pnl = parseFloat(pnlRaw.replace(/[$,]/g, ''));
        if (isNaN(pnl)) continue;

        // Type normalization
        let type: 'Long' | 'Short' = 'Long';
        const lowerSide = sideRaw.toLowerCase();
        if (lowerSide.includes('short') || lowerSide.includes('sell')) {
            type = 'Short';
        }

        const entryPrice = parseFloat(entryPriceRaw);
        const exitPrice = parseFloat(exitPriceRaw);

        // Calculate % if available
        let pnlPercent = 0;
        
        if (!isNaN(entryPrice) && !isNaN(exitPrice) && entryPrice !== 0) {
            if (type === 'Long') {
                pnlPercent = ((exitPrice - entryPrice) / entryPrice) * 100;
            } else {
                pnlPercent = ((entryPrice - exitPrice) / entryPrice) * 100;
            }
        }
        
        // Parse MAE/MFE
        let mfe: number | undefined;
        let mae: number | undefined;
        if (runUpRaw) {
             mfe = parseFloat(runUpRaw.replace(/[$,]/g, ''));
             if (isNaN(mfe)) mfe = undefined;
        }
        if (drawDownRaw) {
             // MAE is a drawdown, so typically we want it negative for charts. 
             // TV usually provides it as absolute USD.
             const val = parseFloat(drawDownRaw.replace(/[$,]/g, ''));
             if (!isNaN(val)) mae = -Math.abs(val); 
        }

        // Fallback: If no explicit MAE/MFE data, estimate from PnL (Min bound)
        if (mfe === undefined && mae === undefined) {
            if (pnl >= 0) {
                mfe = pnl; // Best case: Price reached exit target
                mae = 0;   // Best case: No drawdown
            } else {
                mfe = 0;   // Worst case: No profit excursion
                mae = pnl; // Worst case: Drawdown equal to loss
            }
        }

        const riskMetrics = extractRiskMetrics(row, riskIndices, pnl);

        trades.push({
            id: (i).toString(),
            entryTime,
            exitTime,
            type,
            pnl,
            pnlPercent,
            entryPrice: isNaN(entryPrice) ? undefined : entryPrice,
            exitPrice: isNaN(exitPrice) ? undefined : exitPrice,
            mae,
            mfe,
            ...riskMetrics,
        });
    }

    return trades;
};

const parseStandardTradingView = (lines: string[], colMap: Record<string, number>): Trade[] => {
    const uniqueTrades = new Map<string, Trade>();
    
    // Identify key columns
    const idCol = colMap['Trade #'] ?? colMap['Id'];
    const typeCol = colMap['Type'];
    const pnlCol = colMap['Net P&L USD'] !== undefined ? colMap['Net P&L USD'] : 
                   colMap['Profit USD'] !== undefined ? colMap['Profit USD'] :
                   colMap['PnL'];
                   
    const pnlPercentCol = colMap['Net P&L %'] !== undefined ? colMap['Net P&L %'] : colMap['Profit %'];
    
    // Date/Time variants
    const entryDateCol = colMap['Date/Time'] ?? colMap['Date and time'] ?? colMap['Entry Date/Time'] ?? colMap['EnteredAt'];
    const exitDateCol = colMap['Exit Date/Time'] ?? colMap['ExitedAt'];

    // Price variants
    const entryPriceCol = colMap['Price'] ?? colMap['Entry Price'] ?? colMap['EntryPrice'];
    const exitPriceCol = colMap['Exit Price'] ?? colMap['ExitPrice'];
    
    // MAE/MFE variants
    // "Run-up USD" / "Drawdown USD"
    const runUpCol = colMap['Run-up USD'];
    const drawDownCol = colMap['Drawdown USD'];
    const riskIndices = {
        rMultipleCol: getColumnIndex(colMap, ['R Multiple', 'R-Multiple', 'r_multiple']),
        riskAmountCol: getColumnIndex(colMap, ['Risk USD', 'Risk Amount', 'Initial Risk USD', 'risk_amount', 'risk_per_trade']),
        riskPointsCol: getColumnIndex(colMap, ['SL points', 'Sl points', 'Stop points', 'Risk points', 'sl_points', 'sl_dist']),
        pnlPointsCol: getColumnIndex(colMap, ['PnL points', 'Pnl points', 'pnl_points']),
    };

    if (pnlCol === undefined) return [];

    for (let i = 1; i < lines.length; i++) {
        const row = parseCSVRow(lines[i]);
        if (row.length < 2) continue;

        const tradeId = idCol !== undefined ? row[idCol] : i.toString();
        const pnlRaw = row[pnlCol];
        
        if (!pnlRaw) continue;

        if (!uniqueTrades.has(tradeId)) {
            const pnl = parseFloat(pnlRaw.replace(/[$,]/g, ''));
            
            // Percentage
            let pnlPercent = 0;
            if (pnlPercentCol !== undefined && row[pnlPercentCol]) {
                pnlPercent = parseFloat(row[pnlPercentCol].replace(/[%]/g, ''));
            }

            // Dates
            const entryTime = entryDateCol !== undefined ? row[entryDateCol] : '';
            const exitTime = exitDateCol !== undefined ? row[exitDateCol] : entryTime;

            // Type
            let type: 'Long' | 'Short' = 'Long';
            if (typeCol !== undefined) {
                const rawType = row[typeCol].toLowerCase();
                if (rawType.includes('short') || rawType.includes('sell')) {
                    type = 'Short';
                }
            }

            // Prices
            let entryPrice: number | undefined;
            let exitPrice: number | undefined;

            if (entryPriceCol !== undefined && row[entryPriceCol]) {
                const val = parseFloat(row[entryPriceCol].replace(/[$,]/g, ''));
                if (!isNaN(val)) entryPrice = val;
            }
            if (exitPriceCol !== undefined && row[exitPriceCol]) {
                const val = parseFloat(row[exitPriceCol].replace(/[$,]/g, ''));
                if (!isNaN(val)) exitPrice = val;
            }

            // Calculate percent if missing but prices exist
            if (pnlPercent === 0 && entryPrice && exitPrice) {
                 if (type === 'Long') {
                    pnlPercent = ((exitPrice - entryPrice) / entryPrice) * 100;
                 } else {
                    pnlPercent = ((entryPrice - exitPrice) / entryPrice) * 100;
                 }
            }
            
            // MAE / MFE
            let mfe: number | undefined;
            let mae: number | undefined;
            
            if (runUpCol !== undefined && row[runUpCol]) {
                const val = parseFloat(row[runUpCol].replace(/[$,]/g, ''));
                if (!isNaN(val)) mfe = val;
            }
            if (drawDownCol !== undefined && row[drawDownCol]) {
                 const val = parseFloat(row[drawDownCol].replace(/[$,]/g, ''));
                 if (!isNaN(val)) mae = -Math.abs(val); // Ensure negative
            }

            // Fallback: If no explicit MAE/MFE data, estimate from PnL (Min bound)
            if (mfe === undefined && mae === undefined) {
                if (pnl >= 0) {
                    mfe = pnl; // Price reached at least the profit taken
                    mae = 0;   // Assume perfect entry
                } else {
                    mfe = 0;   // Assume price went straight down
                    mae = pnl; // Price dropped to at least the loss taken
                }
            }

            if (isNaN(pnl)) continue;
            const riskMetrics = extractRiskMetrics(row, riskIndices, pnl);

            uniqueTrades.set(tradeId, {
                id: tradeId,
                entryTime,
                exitTime,
                type,
                pnl,
                pnlPercent: isNaN(pnlPercent) ? 0 : pnlPercent,
                entryPrice,
                exitPrice,
                mae,
                mfe,
                ...riskMetrics,
            });
        }
    }

    return Array.from(uniqueTrades.values()).sort((a, b) => {
        // Try numerical sort if IDs are numbers
        const idA = parseInt(a.id);
        const idB = parseInt(b.id);
        if (!isNaN(idA) && !isNaN(idB)) return idA - idB;
        return 0;
    });
};
