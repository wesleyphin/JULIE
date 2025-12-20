# Gemini Optimization Logs - Holiday Detection Examples

This document shows the enhanced UI logs with bank holiday detection integration.

## Example 1: Normal Market Conditions

```
============================================================
🧠 GEMINI OPTIMIZATION - ASIA SESSION
============================================================
✅ HOLIDAY STATUS: NORMAL_LIQUIDITY
🎯 NEW MULTIPLIERS | SL: 1.0x | TP: 1.2x | CHOP: 1.1x
🌊 TREND REGIME: TRENDING
📝 REASONING: RISK: Strong ADX (32.5) supports TP expansion... | TREND: Clear breakout above IB High... | CHOP: Normal conditions...
============================================================
```

## Example 2: Pre-Holiday Period (1 Day Before)

```
============================================================
🧠 GEMINI OPTIMIZATION - LONDON SESSION
============================================================
📅 HOLIDAY STATUS: Bank Holiday in 1 day(s) - Reducing targets
🎯 NEW MULTIPLIERS | SL: 0.9x | TP: 0.65x | CHOP: 1.8x
🌊 TREND REGIME: CHOPPY
⚠️  HOLIDAY ADJUSTMENTS: Targets reduced ~40% (Pre-holiday illiquidity)
📝 REASONING: RISK: PRE_HOLIDAY detected. Price action tightens significantly. Reduced TP to 0.65x to avoid unreachable targets... | TREND: Elevated t1_body to 3.5x to filter low-volume traps... | CHOP: Dead market chop detected. Raised multiplier to 1.8x for mean-reversion...
============================================================
```

## Example 3: Holiday Today

```
============================================================
🧠 GEMINI OPTIMIZATION - NY SESSION
============================================================
🚨 HOLIDAY STATUS: HOLIDAY_TODAY - Market closed/dead volume
🎯 NEW MULTIPLIERS | SL: 1.0x | TP: 0.5x | CHOP: 2.5x
🌊 TREND REGIME: CHOPPY
⚠️  HOLIDAY ADJUSTMENTS: Extreme risk reduction (Market closed)
📝 REASONING: RISK: HOLIDAY_TODAY status. Market effectively closed. Minimal TP at 0.5x... | TREND: Standard filters... | CHOP: Extreme chop. Market is random walk. Chop multiplier 2.5x...
============================================================
```

## Example 4: Post-Holiday Recovery

```
============================================================
🧠 GEMINI OPTIMIZATION - ASIA SESSION
============================================================
🔄 HOLIDAY STATUS: POST_HOLIDAY_RECOVERY - Volatility expanding
🎯 NEW MULTIPLIERS | SL: 1.2x | TP: 0.9x | CHOP: 1.3x
🌊 TREND REGIME: TRENDING
⚠️  HOLIDAY ADJUSTMENTS: Stops widened +12% (Post-holiday volatility)
📝 REASONING: RISK: POST_HOLIDAY_RECOVERY. Volatility expanding ~12%. Widened SL to 1.2x to survive liquidity rush... | TREND: Maintained elevated filters (2.5x) until clear directional flow... | CHOP: Moderate multiplier 1.3x as volume returns...
============================================================
```

## Example 5: Pre-Holiday Period (3 Days Before)

```
============================================================
🧠 GEMINI OPTIMIZATION - NY SESSION
============================================================
📅 HOLIDAY STATUS: Bank Holiday in 3 day(s) - Reducing targets
🎯 NEW MULTIPLIERS | SL: 1.0x | TP: 0.75x | CHOP: 1.5x
🌊 TREND REGIME: CHOPPY
⚠️  HOLIDAY ADJUSTMENTS: Targets reduced ~40% (Pre-holiday illiquidity)
📝 REASONING: RISK: PRE_HOLIDAY_3_DAYS. Early institutional withdrawal. Moderate TP reduction to 0.75x... | TREND: Increased filters to 3.0x for trap avoidance... | CHOP: Favoring mean-reversion with 1.5x multiplier...
============================================================
```

## Log Structure

### Session Start Banner
```
============================================================
🧠 GEMINI OPTIMIZATION - {SESSION_NAME} SESSION
============================================================
```

### Holiday Status (First Log Line)
- `✅ HOLIDAY STATUS: NORMAL_LIQUIDITY` - No holidays nearby
- `🚨 HOLIDAY STATUS: HOLIDAY_TODAY - Market closed/dead volume` - Holiday today
- `📅 HOLIDAY STATUS: Bank Holiday in {N} day(s) - Reducing targets` - Approaching holiday
- `🔄 HOLIDAY STATUS: POST_HOLIDAY_RECOVERY - Volatility expanding` - Day after holiday

### Multipliers and Regime
```
🎯 NEW MULTIPLIERS | SL: {sl_mult}x | TP: {tp_mult}x | CHOP: {chop_mult}x
🌊 TREND REGIME: {TRENDING/CHOPPY}
```

### Holiday Adjustments (If Applicable)
Only shown when `HOLIDAY_STATUS != NORMAL_LIQUIDITY`:
- `⚠️  HOLIDAY ADJUSTMENTS: Extreme risk reduction (Market closed)`
- `⚠️  HOLIDAY ADJUSTMENTS: Targets reduced ~40% (Pre-holiday illiquidity)`
- `⚠️  HOLIDAY ADJUSTMENTS: Stops widened +12% (Post-holiday volatility)`

### LLM Reasoning
```
📝 REASONING: RISK: {...} | TREND: {...} | CHOP: {...}
```

### Session End Banner
```
============================================================
```

## Key Visual Indicators

| Emoji | Meaning |
|-------|---------|
| 🧠 | Gemini AI optimization running |
| ✅ | Normal market conditions |
| 📅 | Holiday approaching (1-3 days) |
| 🚨 | Holiday today - critical alert |
| 🔄 | Post-holiday recovery period |
| 🎯 | Multipliers applied to risk parameters |
| 🌊 | Market regime (trend vs chop) |
| ⚠️  | Holiday-specific adjustments active |
| 📝 | Detailed AI reasoning |

## Integration Points

The holiday context flows through the system as follows:

1. **Detection**: `NewsFilter.get_holiday_context(current_time)` → Status code
2. **Injection**: Status code passed to `GeminiSessionOptimizer.optimize_new_session()`
3. **LLM Processing**: Gemini adjusts multipliers based on status code and playbook rules
4. **UI Display**: Logs show both status and resulting adjustments
5. **Reasoning**: LLM explains holiday-specific decisions in natural language

This creates full transparency from detection → optimization → execution.
