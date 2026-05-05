# σ-ML Strategy — Ship Notes (iter 13 final)

## Final config

| Param | Value |
|---|---|
| Bracket TP | 1.5 × ATR_60 |
| Bracket SL | 1.0 × ATR_60 |
| Min TP floor | 8 points |
| Min SL floor | 5 points |
| Threshold | 0.75 (max-prob across LONG/SHORT) |
| Daily DD cap | $200 (wire via `circuit_breaker.max_daily_loss = 200`) |
| Horizon | 60 minutes |
| Window | ET 11:00 – 12:00 (PT 8 – 9) |
| Warmup | 4,800 bars (~5 trading days) |

## Models
- LONG: `artifacts/stdev_ml_hr11_12/iter12_both_sides/iter12_1.5_1.0_L.pkl`
- SHORT: `artifacts/stdev_ml_hr11_12/iter12_both_sides/iter12_1.5_1.0_S.pkl`

## Strategy module
- `stdev_ml_strategy.py` (StdevMlStrategy class)
- Inherits Strategy base; integrates via the same pattern as ManifoldStrategy.

## Honest performance — single-position evaluator (iter 13)

### VAL (2021-01-21 → 2025-01-19, 1,030 sessions)
| Year | trades | /d | WR | PnL | DD |
|---|---|---|---|---|---|
| 2021 | 489 | 4.0 | 72.8% | +$7,708 | −$83 |
| 2022 | 35 | 0.4 | 71.4% | +$927 | −$135 |
| 2023 | 88 | 0.6 | 87.5% | +$6,434 | −$103 |
| 2024 | 831 | 6.6 | 79.4% | +$90,728 | −$807 |
| 2025-Jan | 31 | 3.4 | 67.7% | +$2,420 | −$559 |

VAL aggregate: 1,474 trades / 1.18 per session / 77.3% WR / +$108,217 / max DD −$807.

### OOS_2026 (Jan 1 – Apr 30, 80 sessions)
| Month | days w/ trades | n | /d | WR | PnL |
|---|---|---|---|---|---|
| 2026-01 | 20 | 141 | 7.05 | 85.1% | +$15,459 |
| 2026-02 | 2  | 5   | 2.50 | 80.0% | +$491 |
| 2026-03 | 6  | 25  | 4.17 | 76.0% | +$2,121 |
| 2026-04 | 13 | 54  | 4.15 | 57.4% | +$3,054 |

OOS aggregate: 225 trades / 2.30 per session / 77.3% WR / $94 avg / +$21,125 / DD −$724.

## Caveats — read before live
1. **Regime-dependent.** Heavy fire in trending bull regimes (2024 ≈6/d), near-silent in chop (2022 ≈0.4/d). DD stays low even in silent periods because no trades = no losses.
2. **Calibration drift across years.** AUC: VAL 0.72 → OOS 0.62 (LONG); 0.69 → 0.55 (SHORT). Model still gets useful precision at top thresholds but the overall ranking degrades each year.
3. **Most months <2/d under single-position.** Many months in 2022-2023 fired on only a handful of calendar days. Don't expect uniform daily activity.
4. **`vwap_60` is a top feature and is price-LEVEL.** Could partly track year/regime. If futures markets enter new price territory the model has never seen, results may degrade further.
5. **Horizon = 60 min, window = 60 min.** Position can only open between 11:00 and 12:00 ET, and runs at most until ~13:00 ET. If a position is open at 12:00 ET, it continues to TP/SL/horizon-end.
6. **DD-cap wiring is the bot's responsibility.** The strategy itself does not enforce the $200 daily DD cap — set `CircuitBreaker(max_daily_loss=200)` for this strategy's account/sub-account.

## Wiring checklist
- [ ] Add `from stdev_ml_strategy import StdevMlStrategy` to the strategy registry
- [ ] Confirm bot warmup buffer ≥ 4,800 bars before activating
- [ ] Set `CircuitBreaker.max_daily_loss = 200` for this strategy
- [ ] Confirm bar timestamps are tz-aware (ET) — strategy uses `df.index.hour` directly
- [ ] Verify ATR_60 from feature build matches the bot's expected ATR scale
- [ ] Run paper-trade for ≥10 sessions before going live

## What this strategy is NOT
- Not a "smooth, predictable per-day income" strategy. It's a regime detector with bracket trading.
- Not a primary breadwinner — expected annualized PnL based on VAL is highly bracket-dependent and dominated by 2024-style years.
- Not validated past 2026-04. Performance beyond OOS is purely speculative.

## Iter history (for the next person who looks at this)
- Iter 1-6: σ-only features, AUC ceiling 0.55
- Iter 7-9: macro features looked great (AUC 0.67) — were a leak; yfinance daily close was assigned to midnight of the same trading day
- Iter 10: leak fixed → AUC drops to 0.52, no real macro signal
- Iter 11: shorter horizon (60min) + ATR brackets + ToD cross-sectional → AUC 0.74; but model is regime detector (676 fires Jan 2026, 4 fires Feb 2026)
- Iter 12: added SHORT side + combined → marginally better
- Iter 13: re-evaluated with single-position constraint → ~73% deflation from iter 12 numbers, but still profitable
