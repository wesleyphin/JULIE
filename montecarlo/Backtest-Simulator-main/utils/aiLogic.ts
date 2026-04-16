import { GoogleGenAI, Chat } from "@google/genai";
import { Statistics, Trade } from "../types";

// Initialize Gemini Client
// The API key is obtained exclusively from process.env.API_KEY per instructions.
const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

const buildReportContext = (
  historicalStats: any,
  mcStats: Statistics | null,
  mcConfig: any
): string => {
  const tradeCount = Number(historicalStats?.count ?? 0);
  const profitFactor = Number(historicalStats?.profitFactor ?? NaN);
  const sharpe = Number(historicalStats?.sharpe ?? NaN);
  const sortino = Number(historicalStats?.sortino ?? NaN);
  const riskModel = String(mcConfig?.riskModel ?? "unknown");

  const notes: string[] = [];

  if (tradeCount <= 0) {
    notes.push("- Trade count is unavailable, so confidence should be low.");
  } else if (tradeCount < 100) {
    notes.push("- Sample size is small. Avoid strong deploy/archive conclusions.");
  } else if (tradeCount < 500) {
    notes.push("- Sample size is moderate. Conclusions should still be conditional.");
  } else {
    notes.push("- Sample size is reasonably large, which supports stronger statistical language.");
  }

  if (riskModel === "fixed_pnl") {
    notes.push("- Monte Carlo uses `fixed_pnl`, which shuffles dollar PnL outcomes. This is useful for path-stress, but it is not enough by itself to infer prop-firm breach probability or account-level ruin unless per-trade risk is normalized to account equity.");
  } else if (riskModel === "percent_equity") {
    notes.push("- Monte Carlo uses `percent_equity`, so account-level path statements are more defensible than with fixed-dollar shuffling.");
  } else {
    notes.push("- Monte Carlo risk model is unclear. Treat any account-level conclusions cautiously.");
  }

  if (Number.isFinite(profitFactor) && profitFactor < 1.05) {
    notes.push("- Profit factor is near break-even. Small friction changes can dominate the edge.");
  } else if (Number.isFinite(profitFactor) && profitFactor < 1.20) {
    notes.push("- Profit factor is positive but still thin. Friction sensitivity matters.");
  }

  if (Number.isFinite(sharpe) && Number.isFinite(sortino) && sortino > sharpe * 1.5) {
    notes.push("- Sortino is materially above Sharpe. That can suggest upside skew, but it does not prove dependence on a few outliers by itself.");
  }

  if (!mcStats) {
    notes.push("- Monte Carlo statistics are missing, so any forward-risk view should be limited.");
  }

  return notes.join("\n");
};

export const generateStrategyReport = async (
  historicalStats: any,
  mcStats: Statistics | null,
  mcConfig: any
): Promise<string> => {
  // Use Gemini 3 Pro for complex reasoning and analysis tasks
  const modelId = "gemini-3-pro-preview";
  const contextNotes = buildReportContext(historicalStats, mcStats, mcConfig);

  const prompt = `
    You are an objective quantitative researcher.
    Your job is to evaluate trading strategies with balanced rigor: identify real risks, identify real strengths, and state uncertainty explicitly.
    
    Analyze the following strategy data:

    ### 1. Historical Backtest Stats
    ${JSON.stringify(historicalStats, null, 2)}

    ### 2. Monte Carlo Projections (Future Risk)
    ${JSON.stringify(mcStats, null, 2)}

    ### 3. Simulation Config
    ${JSON.stringify(mcConfig, null, 2)}

    ### 4. Context Notes
    ${contextNotes}

    ---
    
    ### Report Instructions:
    Provide a "Quant Report Card" using the following Markdown format. Be concise, professional, and evidence-weighted.

    ### Critical Rules:
    *   Separate supported conclusions from inference.
    *   Use conditional language when assumptions are weak or missing.
    *   Do not use absolute language such as "guaranteed", "certain", "will fail", or "mathematically guarantees" unless that conclusion follows directly from the provided data and assumptions.
    *   If Monte Carlo uses \`fixed_pnl\`, do not convert drawdown percentages into prop-firm breach probabilities or account-ruin claims unless account-level risk sizing is explicitly established.
    *   If the data is sufficient for a strong conclusion, say so. If not, say "not established from provided data".
    *   Distinguish research viability from live deployment viability.
    *   If metrics conflict, explain the plausible interpretations instead of defaulting to the most negative one.

    **Grade:** [A+ to F]
    **Confidence:** [Low / Medium / High]
    **Summary:** [1 to 2 sentence summary]

    ### 1. Supported Conclusions
    *   List the conclusions that are directly supported by the provided stats.

    ### 2. Uncertainties & Model Limits
    *   Explicitly state what cannot be concluded reliably from the provided data.
    *   Mention any limitations caused by the Monte Carlo risk model or missing assumptions.

    ### 3. Risk Profile
    *   **Skewness & Tails:** Analyze what the data suggests about skew and tail dependence, but do not overclaim.
    *   **Drawdown Reality:** Compare Historical Max DD vs Monte Carlo tail drawdowns. Explain what this does and does not establish.
    *   **Ratios:** Analyze Sharpe vs Sortino and note alternative explanations if relevant.

    ### 4. Strategy Logic
    *   **Win Rate vs R:R:** Does the Win Rate justify the Profit Factor? Is this closer to a high-hit-rate system or a lower-hit-rate trend system?
    *   **Consistency:** Is the SQN healthy (>2.0), and how strong is that conclusion given the sample?

    ### 5. Deployment Context
    *   **Research Verdict:** [KEEP TESTING / OPTIMIZE / ARCHIVE]
    *   **Live/Prop Verdict:** [NOT ENOUGH EVIDENCE / PAPER TRADE / CONDITIONAL / READY]
    *   If discussing prop-firm viability, state the assumptions required. If they are missing, say viability is not established from provided data.
    *   Mention specific daily-loss or trailing-drawdown risks only if the provided setup justifies that inference.

    ### 6. Verdict
    *   **Final Recommendation:** [KEEP TESTING / OPTIMIZE / PAPER TRADE / ARCHIVE]
    *   **Key Warning:** What is the main risk, stated without exaggeration?
  `;

  try {
    const response = await ai.models.generateContent({
      model: modelId,
      contents: prompt,
      config: {
        temperature: 0.2,
        systemInstruction: "You are a mathematically rigorous and objective quantitative researcher. You are explicit about uncertainty, avoid overstating conclusions, and distinguish between what is supported by the data and what is merely plausible. You use Markdown for formatting.",
      },
    });

    return response.text || "Report generation returned empty.";
  } catch (error) {
    console.error("AI Generation Error:", error);
    return `**Error Generating Report**\n\nFailed to connect to Gemini API. \n\nDebug Info:\n- Model: ${modelId}\n- Error: ${(error as Error).message}`;
  }
};

export const createOptimizerChat = (): Chat => {
    return ai.chats.create({
        model: "gemini-3-pro-preview",
        config: {
            temperature: 0.2,
            systemInstruction: "You are a pragmatic algorithmic trader. You verify your work by simulating code execution on data samples. You prioritize risk management. You can maintain a conversation to refine strategies and fix errors.",
        },
    });
};

export const buildOptimizerPrompt = (
    strategyFiles: { name: string; content: string }[],
    marketDataFiles: { name: string; content: string }[],
    historicalStats: any,
    mcStats: Statistics | null,
    regressionStats: string
): string => {
    // Format strategy files for the prompt
    const filesContext = strategyFiles.map(f => `
    --- STRATEGY FILE: ${f.name} ---
    \`\`\`
    ${f.content}
    \`\`\`
    `).join('\n');

    // Format Data Context
    const dataContext = marketDataFiles.length > 0
        ? marketDataFiles.map(f => `
    --- MARKET DATA: ${f.name} (First 50 lines) ---
    \`\`\`csv
    ${f.content}
    \`\`\`
        `).join('\n')
        : "No specific market data file provided. Assume standard OHLCV format.";

    return `
    You are an expert Algorithmic Trading Developer and Backtesting Engine. Your task is to rewrite the provided trading strategy code to improve its performance AND verify it by running a "mental backtest" on the provided data snippets.

    --- INPUTS ---

    ### 1. Current Performance Stats
    Win Rate: ${historicalStats?.winRate.toFixed(2)}%
    Profit Factor: ${historicalStats?.profitFactor.toFixed(2)}
    Sharpe Ratio: ${historicalStats?.sharpe.toFixed(2)}

    ### 2. Monte Carlo Risk Analysis (Future Projections)
    99% Value at Risk (VaR): ${mcStats?.var99.toFixed(2)}%
    Probability of Ruin: ${mcStats?.ruinProbability.toFixed(1)}%
    Max Consecutive Losses (Simulated): ${mcStats?.worstDrawdown.toFixed(1)}%

    ### 3. Regression & Correlation Insights (Crucial for Optimization)
    ${regressionStats}

    ### 4. Codebase
    ${filesContext}

    ### 5. Market Data Context
    ${dataContext}

    --- TASK ---

    1.  **Analyze & Validate:** Identify specific weaknesses. Confirm CSV columns.
    2.  **Optimize Logic:** Rewrite the strategy to mitigate risks (high VaR) and leverage correlations.
    3.  **EXECUTE SIMULATION:** Trace your *new Optimized Strategy* logic against the rows provided in the "Market Data" snippets. 
        *   Literally "run" the code in your head row-by-row for the provided data.
        *   Identify where entries and exits would occur in those snippets.
        *   Calculate the hypothetical PnL for those specific trades.
    4.  **Generate Harness:** Write a Python/Pandas backtest script.

    --- OUTPUT FORMAT ---

    Provide the output in Markdown:

    **1. Simulation Results (AI Predicted):**
    *   **Trace Analysis:** "Running logic on ${marketDataFiles.length > 0 ? marketDataFiles[0].name : 'data'}..."
    *   **Trades Found:** List the specific timestamps and signals found in the snippets.
    *   **Hypothetical PnL:** estimated result for these specific rows.
    *   **Data Validation:** Confirmed columns found (e.g. 'Close' vs 'Adj Close').

    **2. Optimized Backtest Code:**
    *   A complete, runnable \`backtest.py\` script.
    *   **MUST** use the exact column names found in the CSV snippets.
    \`\`\`python
    # Backtest Code ...
    \`\`\`

    **3. Optimized Strategy Code:**
    *   The fully rewritten strategy file(s).
    **File: [filename]**
    \`\`\`[language]
    [Code]
    \`\`\`

    **4. Summary of Improvements:**
    *   Bullet points on risk reduction.
    `;
};

// Deprecated: Wrapper for backward compatibility if needed, but UI now uses chat directly.
export const optimizeStrategy = async (
    strategyFiles: { name: string; content: string }[],
    marketDataFiles: { name: string; content: string }[],
    historicalStats: any,
    mcStats: Statistics | null,
    regressionStats: string
): Promise<string> => {
    try {
        const chat = createOptimizerChat();
        const prompt = buildOptimizerPrompt(strategyFiles, marketDataFiles, historicalStats, mcStats, regressionStats);
        const result = await chat.sendMessage({ message: prompt });
        return result.text || "No response";
    } catch (error) {
         return `Error: ${(error as Error).message}`;
    }
};
