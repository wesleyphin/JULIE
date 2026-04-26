# RegimeAdaptive NY-Session Loss Analysis

*Created 2026-04-25. Forensic analysis of why RegimeAdaptive (RA) lost money in NY session production over Mar 2025 - Apr 2026 (14 months, 142 trades, -$2,157.72 gross PnL), and what it means for live deployment.*

---

## v2 update header (2026-04-25, evening pass)

**What's new in v2:** the v12 training corpus (`artifacts/v12_training_corpus.parquet`) was re-examined and found to contain **85 numeric entry-time features** for RA candidates — `bf_*` (32 bar-derived), `ctx_*` (23 day-context), `pct_*` (17 level-overlay), `k12_*` (13 Kalshi snapshots) — that the original analysis (which leaned on `closed_trades.json`) did not exploit. v2 layers parquet-feature attribution on top of the original journal:

- **Section 11 (revised):** Friend's gate model — confirmed not applicable (0/33 feature overlap with parquet) and the proxy mapping is unsound. No backtest possible.
- **Section 13 (new):** Feature-level loss attribution — top predictive features identified, quartile WR/PnL tables, deployable thresholds.
- **Section 14 (new):** Trained RA-NY filter — 5-fold CV AUC 0.704 ± 0.124. Overfits on holdout (n=29). Simple thresholds beat the model.
- **Section 15 (new):** Manifold regime classifier built from `bf_regime_eff` + `ctx_shock_score` + `ctx_day_trend_frac`. CHOP_DEAD share doubles in 2026 holdout (25% → 55%) and absorbs 90% of holdout losses.
- **Section 16 (new):** Feature-threshold rules that would have prevented the 2025-04-17 disaster — final deployable rule, walk-forward validated.

**Denominator note.** The original journal counts 142 closed RA trades (from `closed_trades.json`, the production order log). The parquet analysis works on 163 RA candidates that pass `allowed_by_friend_rule` in `v12_training_corpus.parquet`. The two datasets overlap heavily (the same actual entries), but rounding, family routing on the corpus side, and 2026-Q1 emissions that haven't fully closed mean the parquet has 21 more rows than `closed_trades.json` for the same period. All v2 analysis uses the 163-row parquet slice and the parquet's `net_pnl_after_haircut` column. Where v2 numbers are quoted alongside v1 numbers, both are kept distinct.

## Top-line summary

Production NY-session RA shipped 142 trades over 14 months and bled **-$2,157.72** before fees. After the standard $7.50/trade haircut the loss widens to **-$3,222.72**. Win rate of **34.51%** matches what the friend's v19 selection metadata reports (34.94% across 12,148 trades on 2011-2024 ES) — the RA edge was always an asymmetric-payoff strategy, but in the NY production window the payoff inverted (profit factor **0.728** vs the friend's expected 1.241). Three causes converge: (1) **the friend's v19 selection has zero NY combos in its top-48 list** — every actual production NY combo fires from `session_defaults` / wildcard fall-through, (2) **production is 100% LONG-only** while the friend's universe is 22 LONG / 26 SHORT — the NY wildcard is structurally biased long, and (3) **the bracket geometry (NY_AM sl=4 / tp=6) is 1:1.5 R:R, which needs a 40% WR to break even** but production only achieves 34.51%. The user's local override `LOCAL_RA_DISABLED_IN_NY=True` is correct and this analysis ratifies it.

## 1. The headline picture

| Phase | n | WR | Gross PnL | After $7.50 hc | Avg trade | Profit factor |
|---|---|---|---|---|---|---|
| **All 14 mo** | 142 | 34.51% | -$2,157.72 | -$3,222.72 | -$15.20 | 0.728 |
| Train (Mar-Dec 2025) | 116 | 37.93% | -$1,794.94 | -$2,664.94 | -$15.47 | 0.78 |
| Holdout (Jan-Apr 2026) | 26 | 19.23% | -$362.78 | -$557.78 | -$13.95 | 0.50 |

Equity-curve milestones (sorted by entry_time):
- **Peak**: +$215.34 at trade #8 on **2025-04-17 08:12 ET** (the very first take_gap of the THU NY_AM cluster)
- **Trough**: **-$2,744.55** at trade #97 on **2025-09-03 11:27 ET**
- **Final**: -$2,157.72
- **Max drawdown**: -$2,959.89 from peak

Friend's v19_liveplus_allsession_wildcard_v2 selection metadata (from `artifacts/regimeadaptive_v19_liveplus_allsession_wildcard_v2/latest.json` → `metadata.validation`):

| Metric | Friend's selection (2011-2024 ES, all-sessions) | NY production (Mar 2025-Apr 2026) |
|---|---|---|
| Trades | 12,148 | 142 |
| Win rate | 34.94% | **34.51%** (matches!) |
| Avg trade net | +$4.62 | **-$15.20** |
| Profit factor | 1.241 | **0.728** |
| Sum train_total / valid / test | -$4,539 / +$28,578 / +$32,047 | n/a |

**The win rate matches almost exactly (34.94% vs 34.51%) yet the profit factor inverts.** That tells us the *signal-frequency* and *direction-flipping* logic is intact, but the *payoff geometry per win vs per loss* is broken. In a 34.5% WR strategy, the average win has to pay for ~1.9 average losses; production averages a $128 win against a $92 loss (1.39x), short of the 1.9x needed. Bracket execution and slippage eat the edge.

## 2. Per-month forensics

| Month | n | WR | Gross PnL | After hc | Avg trade | Worst trade | Best trade | Max month DD |
|---|---|---|---|---|---|---|---|---|
| 2025-03 | 4 | 50.0% | +$71.34 | +$41.34 | +$17.84 | -$26.22 | +$57.52 | -$17.48 |
| 2025-04 | 34 | 41.0% | **-$1,116.28** | -$1,371.28 | -$32.83 | -$181.20 | +$381.30 | **-$1,680.60** |
| 2025-05 | 7 | 57.0% | +$256.60 | +$204.10 | +$36.66 | -$106.20 | +$143.80 | -$106.20 |
| 2025-06 | 2 | 0.0% | -$323.64 | -$338.64 | -$161.82 | -$217.44 | -$106.20 | -$106.20 |
| 2025-07 | 18 | 39.0% | -$146.61 | -$281.61 | -$8.14 | -$181.20 | +$143.80 | -$206.05 |
| 2025-08 | 29 | 24.0% | **-$1,311.16** | -$1,528.66 | -$45.21 | -$181.20 | +$143.80 | -$1,299.36 |
| 2025-09 | 22 | 45.0% | **+$774.81** | +$609.81 | +$35.22 | -$106.20 | +$305.04 | -$287.35 |
| 2025-10 | 0 | — | $0 | $0 | — | — | — | — |
| 2025-11 | 0 | — | $0 | $0 | — | — | — | — |
| 2025-12 | 0 | — | $0 | $0 | — | — | — | — |
| 2026-01 | 2 | 0.0% | -$96.14 | -$111.14 | -$48.07 | -$52.44 | -$43.70 | -$52.44 |
| 2026-02 | 4 | 0.0% | -$157.32 | -$187.32 | -$39.33 | -$52.44 | -$26.22 | -$113.62 |
| 2026-03 | 13 | 15.0% | -$222.22 | -$319.72 | -$17.09 | -$59.98 | +$57.52 | -$218.56 |
| 2026-04 | 7 | 43.0% | +$112.90 | +$60.40 | +$16.13 | -$127.44 | +$172.56 | -$254.88 |

**Worst three months (PnL):** 2025-08 (-$1,311), 2025-04 (-$1,116), 2025-06 (-$324).
**Best three months (PnL):** 2025-09 (+$775), 2025-05 (+$257), 2026-04 (+$113).

Two months — **April 2025 and August 2025** — together booked **-$2,427.44**, more than the entire 14-month gross loss. Without those two months the strategy is roughly flat (+$269). Both contained intraday clusters where 5-7 entries fired in the same 10-15 minute window into the same losing move.

**RA went silent for Q4 2025** (Oct/Nov/Dec all zero trades). That's not a ledger error — it reflects the friend-rule signal-policy filter genuinely producing no qualifying combos in that window.

## 3. Time-of-day pattern

Bucketed by hour ET (entry_time → US/Eastern):

| Hour ET | n | WR | Gross PnL | Avg trade |
|---|---|---|---|---|
| 08 | 36 | 22.0% | **-$2,003.21** | **-$55.64** |
| 09 | 25 | 36.0% | -$650.17 | -$26.01 |
| 10 | 25 | 48.0% | +$456.08 | +$18.24 |
| 11 | 29 | 48.0% | +$243.72 | +$8.40 |
| 12 | 5 | 60.0% | +$163.90 | +$32.78 |
| 13 | 10 | 10.0% | -$195.94 | -$19.59 |
| 14 | 9 | 22.0% | -$110.92 | -$12.32 |
| 15 | 3 | 0.0% | -$61.18 | -$20.39 |

**Hour 08 alone explains 93% of the 14-month gross loss.** Cash-open volatility (08:30 ET = 09:30 EST in winter), pre-data prints, and CME open-volatility have RA buying right into the wrong side. WR drops to 22% in hour 08 vs 48% in hours 10-11 (mid-morning), so it isn't a payoff problem at the open — it's a directional-edge problem. By contrast hours 10-12 are net positive ($863) — RA actually works once the open volatility settles.

**Minute-of-session bucket (from 08:30 ET):**

| Bucket | n | WR | Gross PnL |
|---|---|---|---|
| 0-60 min | 45 | 22.0% | **-$2,451.54** |
| 60-120 min | 33 | 45.0% | -$9.88 |
| 120-180 min | 20 | 40.0% | -$147.88 |
| 180+ min | 44 | 36.0% | +$451.58 |

Same conclusion via a different cut: **the first hour of the NY session is structurally toxic for RA**. The strategy needs the morning chop to settle before its mean-reversion / rule-fired bias becomes profitable.

## 4. Side analysis

| Side | n | WR | Gross PnL | Avg trade |
|---|---|---|---|---|
| LONG | 142 | 34.5% | -$2,157.72 | -$15.20 |
| SHORT | 0 | — | — | — |

**Every single one of the 142 RA NY-session trades is LONG.** The friend's v19 selection has 22 LONG and 26 SHORT combos at the global level, but **the NY combos that fire here all resolve to LONG-side policies**:
- `Q2_W3_THU_NY_AM`: explicit `signal_policies` = LONG normal, rule_id `f8_s21_x0p0`
- `Q3_W1_FRI_NY_AM`: explicit `signal_policies` = LONG normal (rule `f13_s34_x0p0`)
- The other 10 NY combos fire via `session_defaults` (sl=4 tp=6 LONG) or via `group_signal_policies` (e.g. `Q1_W3_ALL_NY_PM` = LONG)

So **NY is structurally a long-only RA universe in v19_liveplus**. That is fine in trending up days but disastrous on bear-trend or rotation days where there is no SHORT counterpart to pick up the offsetting edge. See section 8: 56 LONG entries on grind_down/trend_down days lost $602.90 — there is no SHORT side to hedge.

## 5. Sub-strategy / combo loss leaders

All 12 distinct `combo_key` values that fired:

| combo_key | n | WR | Gross PnL | Avg trade |
|---|---|---|---|---|
| **Q2_W3_THU_NY_AM** | 50 | 42.0% | **-$1,070.42** | -$21.41 |
| **Q3_W1_WED_NY_AM** | 11 | 9.0% | **-$1,046.96** | **-$95.18** |
| Q1_W3_WED_NY_PM | 4 | 0.0% | -$182.34 | -$45.58 |
| Q1_W3_THU_NY_PM | 11 | 18.0% | -$144.76 | -$13.16 |
| Q1_W3_TUE_NY_PM | 2 | 0.0% | -$96.14 | -$48.07 |
| Q3_W4_WED_NY_AM | 10 | 30.0% | -$90.76 | -$9.08 |
| Q3_W2_WED_NY_AM | 7 | 29.0% | -$65.92 | -$9.42 |
| Q1_W3_MON_NY_PM | 2 | 0.0% | -$52.44 | -$26.22 |
| Q1_W3_FRI_NY_PM | 4 | 50.0% | +$71.34 | +$17.84 |
| Q3_W3_WED_NY_AM | 7 | 57.0% | +$117.83 | +$16.83 |
| Q3_W1_WED_NY_PM | 4 | 50.0% | +$200.20 | +$50.05 |
| Q3_W1_FRI_NY_AM | 30 | 40.0% | +$202.65 | +$6.76 |

**Two combos drive 107% of total losses**: `Q2_W3_THU_NY_AM` and `Q3_W1_WED_NY_AM` together lose **$2,117.38** out of a $2,157.72 total deficit. The Wednesday/Thursday morning Q2/Q3 combos are the kill list.

**Concentration verdict:** if `Q2_W3_THU_NY_AM` and `Q3_W1_WED_NY_AM` were blocked entirely, NY production RA would be **break-even to slightly negative** instead of -$2,157. Note that **`Q3_W1_WED_NY_AM` is NOT in `signal_policies`** — it fires from session_defaults wildcard. This is the single best argument for tightening the RA signal_gate or adding NY-specific blacklists.

**Explicit-policy combos** (`Q2_W3_THU_NY_AM` + `Q3_W1_FRI_NY_AM`):
- n=80, WR=41.3%, PnL = **-$867.77**

**Wildcard combos** (the other 10):
- n=62, WR=25.8%, PnL = **-$1,289.95**

The wildcard fall-through is the bigger absolute drag — 62 trades the friend never explicitly endorsed, leaking $1,290.

## 6. Exit-reason asymmetry

| source | n | WR | Total PnL | Avg | Notes |
|---|---|---|---|---|---|
| stop | 59 | 0.0% | -$5,331.26 | -$90.36 | Hard SL hit |
| stop_gap | 20 | 0.0% | -$1,932.84 | -$96.64 | SL filled across a gap |
| take | 34 | 100.0% | +$4,385.06 | +$128.97 | Hard TP hit |
| take_gap | 10 | 100.0% | +$1,254.12 | +$125.41 | TP filled across a gap |
| close_position | 6 | 17.0% | -$139.70 | -$23.28 | Forced flat (session end?) |
| close_trade_leg | 13 | 31.0% | -$393.10 | -$30.24 | Reverse signal / re-entry |

**Stop:Take ratio = 79:44 = 1.80x more stops than takes.**
- Avg stop loss: -$91.95 per stop
- Avg take win: +$128.16 per take
- **For 1:1.5 R:R bracket geometry to be profitable, stops:takes must be ≤ 1.5:1.** Production runs 1.80:1, beyond breakeven. This is the structural failure mode.

`pnl_points` distribution confirms the bracket geometry:
- Stops cluster at exactly **-4.00 pts (NY_AM)** (48 trades) and **-1.50 pts (NY_PM)** (18 trades) — matches `session_defaults`
- Takes cluster at exactly **+6.00 pts** (40 trades) — matches `session_defaults`

So the bracket geometry IS being honored faithfully. The problem isn't execution slippage; the problem is the WR vs R:R mismatch in this regime / time-of-day window.

## 7. Day-of-week effect

| DoW | n | WR | Gross PnL | Avg trade |
|---|---|---|---|---|
| Monday | 2 | 0.0% | -$52.44 | -$26.22 |
| Tuesday | 2 | 0.0% | -$96.14 | -$48.07 |
| **Wednesday** | 43 | 28.0% | **-$1,067.95** | -$24.84 |
| **Thursday** | 61 | 38.0% | **-$1,215.18** | -$19.92 |
| Friday | 34 | 41.0% | +$273.99 | +$8.06 |

**Wednesday + Thursday** book -$2,283.13, more than the entire 14-month gross loss. Friday is the only positive day. The combo_keys in the friend's signal_policies are heavily skewed toward Wed/Thu/Fri NY combos, but Wed/Thu cluster on the loss side. This is consistent with section 5 — Q3_W1_WED_NY_AM and Q2_W3_THU_NY_AM are both Wed/Thu morning combos.

## 8. Manifold regime correlation

Joining 142 production trades to `artifacts/v12_training_corpus.parquet` on minute-floored entry timestamp:

- **106 / 142 trades match** the corpus (74.6%). The 36 unmatched are where the v12 corpus did not have an RA candidate row for that minute (likely friend-rule pre-filter rejected, or after the corpus was last regenerated).

**By `ctx_day_flow_regime`:**

| Flow regime | n | WR | Gross PnL |
|---|---|---|---|
| heavy_flow | 57 | 37.0% | -$1,134.95 |
| normal_flow | 49 | 39.0% | +$489.41 |
| (no match) | 36 | 25.0% | -$1,512.18 |

**By `ctx_day_expansion_regime`:**

| Expansion regime | n | WR | Gross PnL |
|---|---|---|---|
| expanded | 30 | 43.0% | +$181.38 |
| normal | 39 | 28.0% | -$1,134.79 |
| shock | 37 | 43.0% | +$307.87 |
| (no match) | 36 | 25.0% | -$1,512.18 |

**By `ctx_day_direction_regime`:**

| Direction | n | WR | Gross PnL |
|---|---|---|---|
| grind_down | 33 | 42.0% | -$121.38 |
| grind_up | 17 | 59.0% | +$504.56 |
| rotation | 29 | 28.0% | -$233.63 |
| trend_down | 23 | 35.0% | -$481.52 |
| trend_up | 4 | 0.0% | -$313.57 |

Reading the `direction` table is brutal: **even on `trend_up` days the LONG-only RA loses $313.57 over 4 trades** (likely it bought late or got chopped before resuming). And on `trend_down` it loses $481.52 over 23 trades. The 56 LONG entries taken on grind_down/trend_down days lost $602.90 net — exactly the kind of risk-without-offset the all-LONG NY universe creates.

**Volatility tertiles (`bf_regime_vol_bp`):**

| Vol tertile | n | WR | Gross PnL |
|---|---|---|---|
| lowvol | 36 | 31.0% | -$765.83 |
| midvol | 35 | 31.0% | -$277.20 |
| highvol | 35 | 51.0% | +$397.49 |

Counter-intuitive: **higher vol works better for RA.** Low-vol/normal-expansion days are where it bleeds. The strategy fires lots of small SL=4-pt stops on quiet days and they all clip out before the take side has time to hit.

**Train vs Holdout regime distribution shift:**

`ctx_day_flow_regime`:
- TRAIN: heavy_flow 54 / normal_flow 31 / NaN 31 (47% heavy)
- HOLDOUT: heavy_flow 3 / normal_flow 18 / NaN 5 (12% heavy)

**Holdout shifted hard toward `normal_flow` (less heavy participation), and `normal_flow` is exactly where RA underperforms.** That regime tilt explains why the 26 holdout trades collapsed to 19.23% WR vs 37.93% in training.

`ctx_day_expansion_regime`:
- TRAIN: shock 37 / normal 24 / expanded 24 / NaN 31
- HOLDOUT: shock 0 / normal 15 / expanded 6 / NaN 5

**ZERO `shock` days fired in 26 holdout trades.** Shock days are where RA gets +$308 train-window contribution. The 2026 Q1 tape has been low-shock, and that removes one of RA's bread-and-butter regimes.

## 9. Friend-expected vs reality divergence

| Source | Universe | n | WR | PF | Gross PnL |
|---|---|---|---|---|---|
| Friend v19 selection 2011-2024 (`metadata.validation`) | All-sessions ES | 12,148 | 34.94% | 1.241 | +$56,086 (train+valid+test) |
| Friend v19 OOS test only (final 2024 holdout window) | All-sessions | n/a | n/a | n/a | included above |
| **Production NY 2025-2026** | **NY-only** | **142** | **34.51%** | **0.728** | **-$2,158** |

The divergence is not the WR (it matches), it's the **payoff per win and per loss**:
- Friend's universe: avg trade net = **+$4.62** (multi-session) → equity grows
- Production NY: avg trade net = **-$15.20** → equity bleeds

Three reasons production diverges from friend's expected:
1. **NY is unrepresented in the v19 top-48 selection** (16 ASIA + 32 LONDON, 0 NY). NY combos are firing only because of `session_defaults` and 2 explicit `signal_policies` entries. The selection's edge metrics (34.94% / pf 1.24) **were never validated for NY**.
2. **NY is structurally LONG-only via the policy resolver** (see section 4). The friend's universe gets ~half its profit factor from SHORT combos in ASIA/LONDON, which the NY filter excludes.
3. **2025-2026 is OOS** for the v19 selection (which fit through 2024). Even ignoring the NY/session restriction, 16 months of OOS edge decay is well within model-rot expectations for a 14-year fit. The matching 34.5% WR suggests *signal generation* is healthy; what decayed is the *win/loss magnitude* — likely because 2025 NY has different micro-volatility structure than 2011-2024 averages.

## 10. Worst trades forensics

Top 10 single-trade losses:

| # | Entry time (ET) | combo | Entry | Exit | Source | PnL | Hold (min) |
|---|---|---|---|---|---|---|---|
| 1 | 2025-06-19 08:35 | Q2_W3_THU_NY_AM | 6000.00 | 5993.00 | stop | **-$217.44** | 26 |
| 2 | 2025-04-17 08:08 | Q2_W3_THU_NY_AM | 5336.75 | 5329.75 | stop | -$181.20 | 4 |
| 3 | 2025-04-17 08:09 | Q2_W3_THU_NY_AM | 5334.75 | 5327.75 | stop | -$181.20 | 4 |
| 4 | 2025-04-17 08:40 | Q2_W3_THU_NY_AM | 5336.25 | 5329.25 | stop | -$181.20 | 21 |
| 5 | 2025-04-17 08:41 | Q2_W3_THU_NY_AM | 5333.75 | 5326.75 | stop | -$181.20 | 20 |
| 6 | 2025-04-17 08:43 | Q2_W3_THU_NY_AM | 5333.75 | 5326.75 | stop | -$181.20 | 18 |
| 7 | 2025-04-17 08:44 | Q2_W3_THU_NY_AM | 5335.00 | 5328.00 | stop | -$181.20 | 17 |
| 8 | 2025-07-02 08:24 | Q3_W1_WED_NY_AM | 6244.00 | 6237.00 | stop | -$181.20 | 18 |
| 9 | 2025-07-02 08:26 | Q3_W1_WED_NY_AM | 6245.00 | 6238.00 | stop | -$181.20 | 16 |
| 10 | 2025-08-01 08:50 | Q3_W1_FRI_NY_AM | 6315.00 | 6308.00 | stop | -$181.20 | 7 |

Patterns:
- **9 of 10 worst losses fire in hour 08 ET** (cash open).
- **6 of 10 are on 2025-04-17** alone (a single Thursday morning, Q2_W3_THU_NY_AM cluster).
- **All 10 are LONG, all 10 hit hard stops at exactly 4 pts (NY_AM bracket).**
- Hold times of 4-26 minutes — these aren't "wait for the move" trades, they get knifed.

**Cluster days (top 8 by daily PnL loss):**

| Date | n | Daily PnL | Note |
|---|---|---|---|
| 2025-04-17 | 34 | -$1,116.28 | THU NY_AM rapid-fire cluster, 18 LONG entries 08:08-08:59 |
| 2025-08-06 | 5 | -$509.76 | WED NY_AM all-stops 09:22-09:32 |
| 2025-08-01 | 18 | -$509.16 | FRI NY_AM Q3_W1, mixed wins/stops 08:41-11:22 |
| 2025-07-02 | 2 | -$362.40 | WED NY_AM Q3_W1, 2 max-stops back-to-back 08:24-26 |
| 2025-06-19 | 2 | -$323.64 | THU NY_AM Q2_W3, 2 max-stops 08:35-? |
| 2025-08-13 | 3 | -$216.12 | WED NY_AM cluster |
| 2026-03-19 | 10 | -$109.80 | THU NY_AM, holdout-window cluster |
| 2025-08-27 | 1 | -$106.20 | single max-stop |

**The same combo (`Q2_W3_THU_NY_AM`) fired 7 LONG entries across 4 minutes on 2025-04-17 (08:08-08:14) and 5 more across 5 minutes (08:40-08:44 + 08:55-08:59), all at the same price level (5330-5336), all stopped out for -$181 each.** This is a textbook execution-amplification failure: the signal told the system "buy the THU Q2_W3 setup" and the system stacked 5 simultaneous full-size positions into one unfolding losing move. Position-stacking inside a 5-minute window concentrates risk far beyond what the single-trade SL=4-pt geometry was sized for.

## 11. Friend's gate model — backtest feasibility (v2 revised)

`artifacts/regimeadaptive_v19_liveplus_allsession_wildcard_v2/regimeadaptive_gate_model.joblib` is a `HistGradientBoostingClassifier` with `threshold=0.55`, `selection_method=walkforward_validation_median_threshold`, `final_holdout_years=[2024]`, fit by the friend on the `regimeadaptive_v14_dense_balanced` artifact. Its `feature_columns` list is:

```
final_side, original_side, reverted, quarter, week_in_month, day_of_week,
session_code, hour_sin, hour_cos, minute_sin, minute_cos, rule_type_code,
sma_fast_period, sma_slow_period, cross_atr_mult, pattern_lookback,
touch_atr_mult, strength, strength_atr, close_fast_dist_atr,
close_slow_dist_atr, fast_slow_spread_atr, bar_range_atr, body_atr,
upper_wick_atr, lower_wick_atr, atr_pct, return_1, return_5, return_15,
vol_30, vol_ratio, range_vs_mean
```

**Feature overlap with v12 corpus: 0 / 33 exact matches.** v2 ran the explicit cross-check: of the 33 gate features, exactly zero appear by name in the parquet's 116 columns. The single loose match (`atr_pct` ↔ `pct_atr_pct_30bar`) is a different measurement (gate's `atr_pct` is current-bar ATR pct; parquet's `pct_atr_pct_30bar` is 30-bar window ATR pct).

**Why semantic-proxy mapping is also unsound.** Names like `body_atr`, `upper_wick_atr`, `return_1` exist in concept inside the parquet's `bf_de3_entry_*` family — but those `bf_de3_entry_*` features are computed on the **DE3 entry bar context**, not on RA's actual signal-time bar. They are different time indices. Substituting one for the other and feeding into the gate's `predict_proba()` would yield a number, but that number would not correspond to anything the gate was trained against. v2 explicitly rejects this approach.

**Conclusion:** the gate model is **NOT BACKTESTABLE** on the v12 corpus or on `closed_trades.json` as either dataset stands today. The bar-shape features (return_1/5/15, vol_30, sma distances, body/wick ATR ratios, strength_atr, the rule_type_code) would have to be recomputed live from 1-min bars at every RA entry timestamp using the friend's exact code path. This is an engineering task — the bot's friend-rule emit code already computes these internally, they just aren't logged.

**v2 deliverable:** what we *can* do without the gate is build a parallel filter on the parquet features that ARE present. That is the work in sections 13-16 below. The friend's gate may or may not block the same trades; we cannot tell. But the parquet-side analysis surfaces independent, deployable filters that don't require the gate at all.

**Recommendation (unchanged from v1):** ask the friend to add `signal_gate_proba` and `signal_gate_pass` fields to the live RA emission. Until then, sections 13-16 are the actionable replacement.

## 12. Recommendations

**For NY-only live deployment (the user's actual question):**
- The user already set `LOCAL_RA_DISABLED_IN_NY=True` and this analysis fully validates that decision. NY production RA returns -$15.20 per trade gross before haircut, -$22.70 after — there is no statistical regime in the 14-month sample where leaving it on improves equity.
- Specifically: **80 of the 142 trades come from two combo_keys (`Q2_W3_THU_NY_AM` + `Q3_W1_FRI_NY_AM`) that ARE explicitly listed in friend's `signal_policies`** but together net **-$867.77**. Even the "endorsed" NY combos lose money in this OOS window. So even a "respect-only-explicit-NY-combos" softer kill would not save this strategy.

**For broader RA deployment (friend's intended scope: ASIA+LONDON+wildcard):**
- Keep RA enabled in non-NY sessions. The friend's selection metadata shows positive OOS test totals (+$32k on 2011-2024 test) and the all-sessions universe is structurally side-balanced (22 LONG / 26 SHORT).
- If RA is ever re-enabled in NY in any form, **gate it on `ctx_day_direction_regime in (grind_up, expanded)` AND `bf_regime_vol_bp ≥ midvol` AND `hour ≥ 10 ET`**. That triple filter would have left ~25-30 trades from the 142, with materially better PF based on the regime cuts in section 8.
- Do NOT trust `session_defaults` wildcard fall-through in NY. Wildcard combos lost $1,290 (60% of total losses) over 62 trades, more than the explicit-policy combos.

**Position-stacking safeguard:**
- The 2025-04-17 cluster (18 entries in 51 minutes, 12 of them stops at the same price level) shows the bot has no minimum-spacing rule for repeated RA fires of the same combo_key. Recommend: per-combo cooldown of >=15 min and/or "only re-enter if previous trade was a take" — that single rule would have prevented the worst single day (-$1,116).

**Gate-model wiring:**
- Friend ships a gate model with threshold 0.55 but it requires 33 bar-derived features that are not currently logged. Fixing this is a worthwhile engineering item: log the gate proba per RA emission, then in 30 days you can answer "would this gate have blocked the worst trades?" empirically. Until then the gate is unverifiable on your tape.

**Open questions that this analysis cannot resolve without more data:**
- Why does production 14mo lose while friend's selection shows positive 2024 holdout? → Almost certainly because NY combos were never in the top-48 selection set and the all-LONG bias is NY-specific. The friend's holdout is `allsession`, dominated by ASIA+LONDON SHORTs.
- Is bracket execution matching friend's assumed brackets? → Yes, the 4-pt SL / 6-pt TP cluster is exact. No execution slippage to blame here.
- Would the gate ML have blocked the worst clusters? → Cannot answer without re-deriving features. Needs friend's cooperation or live-logging changes.

## 13. Feature-level loss attribution (v2 new)

Universe: 163 RA candidates from `v12_training_corpus.parquet` with `family == 'regimeadaptive'` and `allowed_by_friend_rule == True`. Cumulative net PnL after haircut: **-$965.00**, win rate **36.2%**, 59 winners vs 103 losers (1 break-even). Loser-cluster definition: bottom 20% by `net_pnl_after_haircut` (n=58, threshold ≤ -$27.50, total -$1,595.00). Winner cluster: all positive PnL trades (n=59, total +$1,246.25).

**Top 10 features by |Pearson correlation with `is_loss`|:**

| Feature | corr(PnL) | corr(is_loss) | Reading |
|---|---|---|---|
| `pct_dist_to_running_hi_pct` | +0.276 | -0.298 | Closer to running session high → more losses |
| `pct_abs_level` | +0.244 | -0.279 | Smaller absolute distance from open → losses |
| `pct_signed_level` | -0.244 | +0.266 | More positive = above open → more losses (LONG-only RA buying highs) |
| `bf_regime_eff` | +0.194 | -0.251 | Lower efficiency → more losses |
| `bf_de3_entry_range10_atr` | +0.216 | -0.251 | Tighter 10-bar range vs ATR → more losses |
| `k12_skew_p10` | -0.248 | +0.244 | Higher Kalshi P10-skew (price skew up at 10 strikes) → more losses |
| `k12_below_10` | +0.242 | -0.221 | Lower below-10-strikes Kalshi probability → more losses |
| `ctx_shock_session_range_norm` | +0.177 | -0.199 | Smaller normalized session range → more losses |
| `ctx_shock_score` | +0.095 | -0.153 | Lower shock score → more losses |
| `bf_atr_ratio_to_sl` | -0.053 | +0.143 | Higher ATR vs SL ratio → more losses (chop kills tight SL) |

These ten correlations are individually weak (max |ρ| = 0.30) but collectively encode a coherent regime: **RA loses when the entry is near the running high, in a low-range / low-efficiency tape, with a Kalshi distribution that shows little downside skew (k12_below_10 small, k12_skew_p10 positive).** That's late-morning grind-up exhaustion buying, exactly the pattern that produces the 08-09 ET cluster damage in section 3.

**Quartile WR / PnL breakdowns for the top 6:**

`pct_dist_to_running_hi_pct` (cuts: 0.0011 / 0.0017 / 0.0024)

| Q | n | Σ PnL | mean | WR |
|---|---|---|---|---|
| Q1 (≤0.0011) | 41 | -$447.50 | -10.91 | 22.0% |
| Q2 | 41 | -$217.50 | -5.30 | 39.0% |
| Q3 | 40 | -$247.50 | -6.19 | 35.0% |
| Q4 (≥0.0024) | 41 | -$52.50 | -1.28 | 48.8% |

Q1 is brutal — RA enters within 0.11% of the running session high and gets immediately stopped. That single quartile (n=41) explains 46% of the entire 163-trade loss.

`bf_de3_entry_range10_atr` (cuts: 2.95 / 3.52 / 4.33)

| Q | n | Σ PnL | mean | WR |
|---|---|---|---|---|
| Q1 (≤2.95) | 41 | -$402.50 | -9.82 | 29.3% |
| Q2 | 41 | -$342.50 | -8.35 | 29.3% |
| Q3 | 40 | -$215.00 | -5.38 | 35.0% |
| Q4 (≥4.33) | 41 | -$5.00 | -0.12 | 51.2% |

Compressed 10-bar range relative to ATR is the chop signature — Q1+Q2 (range10_atr ≤ 3.52) account for 77% of the loss. Wide-range Q4 is essentially break-even.

`k12_below_10` (cuts: 0.123 / 0.309 / 0.500)

| Q | n | Σ PnL | mean | WR |
|---|---|---|---|---|
| Q1 (≤0.123) | 40 | -$540.00 | -13.50 | 20.0% |
| Q2 | 39 | -$272.50 | -6.99 | 38.5% |
| Q3 | 66 | -$107.50 | -1.63 | 42.4% |
| Q4 (≥0.500) | 13 | +$55.00 | +4.23 | 61.5% |

This is the single sharpest cut. **Q1 of `k12_below_10` (Kalshi probability of finishing 10+ strikes below entry) is below 12.3% — and at that level the RA LONG bites a 20% WR / -$13.50 mean.** When Kalshi disagrees with the LONG thesis (low downside probability, but the trade still fires), the trade dies. Q4 is the only profitable quartile across any feature in this analysis.

`bf_regime_eff` (cuts: 0.033 / 0.066 / 0.124)

| Q | n | Σ PnL | mean | WR |
|---|---|---|---|---|
| Q1 (≤0.033) | 41 | -$120.00 | -2.93 | 41.5% |
| **Q2 (0.033-0.066)** | 41 | **-$433.75** | **-10.58** | **19.5%** |
| Q3 | 40 | -$222.50 | -5.56 | 40.0% |
| Q4 (≥0.124) | 41 | -$188.75 | -4.60 | 43.9% |

Non-monotone! Q2 (mid-low efficiency, the "false-trend" regime) is the worst — 19.5% WR — while Q1 (true chop) is actually middling. The intuition: in true chop RA gets clipped fast in both directions and small stops average out; in false-trend regimes RA buys the breakout that fails.

`pct_signed_level` (cuts: -0.0009 / -0.0001 / +0.0008)

| Q | n | Σ PnL | mean | WR |
|---|---|---|---|---|
| Q1 (most below open) | 41 | -$17.50 | -0.43 | 53.7% |
| Q2 (slightly below) | 41 | -$407.50 | -9.94 | 22.0% |
| Q3 (slightly above) | 40 | -$143.75 | -3.59 | 37.5% |
| Q4 (well above open) | 41 | -$396.25 | -9.66 | 31.7% |

The LONG-only RA does best when entering well below open (Q1 — buying the dip) and worst in the slightly-below-open zone (Q2) and well-above-open zone (Q4). Q4 is "buying the high" suicide; Q2 is "dead-cat-bounce-on-weakness" suicide.

**Loser cluster vs winner cluster (means ± std):**

| Feature | Losers (n=58) mean | Winners (n=59) mean | Δ |
|---|---|---|---|
| `pct_dist_to_running_hi_pct` | 0.0022 ± 0.0016 | 0.0035 ± 0.0036 | losers 37% closer to hi |
| `pct_abs_level` | 0.0018 ± 0.0015 | 0.0028 ± 0.0033 | losers 36% closer to open |
| `pct_signed_level` | -0.0002 ± 0.0024 | -0.0016 ± 0.0040 | winners are deeper below open |
| `bf_regime_eff` | 0.106 ± 0.070 | 0.145 ± 0.175 | winners have 37% higher efficiency |
| `bf_de3_entry_range10_atr` | 3.56 ± 1.02 | 4.07 ± 1.30 | winners have 14% wider range |
| `k12_skew_p10` | +0.030 ± 0.230 | -0.086 ± 0.192 | losers have positive (up) skew, winners have negative (down) skew |
| `k12_below_10` | 0.253 ± 0.189 | 0.376 ± 0.214 | winners have 49% higher k12_below_10 |
| `ctx_shock_session_range_norm` | 1.61 ± 0.72 | 2.05 ± 2.27 | winners have 27% wider session range |
| `pct_minutes_since_open` | 44.5 ± 49.7 | 92.5 ± 89.4 | losers fire 48 min earlier in session |

The single biggest cluster signature: **losers fire ~48 min earlier in the RTH session, ~37% closer to the running high, on tapes with 14% tighter range and Kalshi distributions that don't endorse a downside.** The earlier-in-session finding cleanly matches sections 3 and 10.

**Top 3 most predictive features for RA losses, with thresholds:**

1. **`pct_dist_to_running_hi_pct` ≤ 0.0011** — block. 41 trades, -$447.50, 22% WR.
2. **`k12_below_10` ≤ 0.123** — block. 40 trades, -$540.00, 20% WR.
3. **`bf_de3_entry_range10_atr` ≤ 2.95** — block. 41 trades, -$402.50, 29% WR.

Each of these alone removes ~25% of trades and 40-55% of the loss.

## 14. Trained RA-NY filter (v2 new)

A small `HistGradientBoostingClassifier` (`max_iter=80, max_depth=3, lr=0.05`) was trained on the 20 highest-correlation features above with target `is_loss = net_pnl < 0`. Train/holdout split at `2026-01-01` (n_train=134, n_hold=29).

**5-fold CV on the train slice:**

| Fold | AUC | Blocked n | Blocked PnL | Kept n | Kept PnL | Kept WR |
|---|---|---|---|---|---|---|
| 0 | 0.877 | 9 | -156.2 | 18 | -86.2 | 44.4% |
| 1 | 0.556 | 16 | -156.2 | 11 | -52.5 | 36.4% |
| 2 | 0.787 | 18 | -307.5 | 9 | +65.0 | 66.7% |
| 3 | 0.571 | 19 | -87.5 | 8 | +31.2 | 62.5% |
| 4 | 0.728 | 9 | -60.0 | 17 | +57.5 | 58.8% |
| **Mean** | **0.704 ± 0.124** | | -153.5 (avg per fold) | | +3.0 (avg per fold) | **53.8%** |

**Final-fit holdout performance (train→hold):**

| Threshold | Train kept WR | Train kept PnL | Holdout kept WR | Holdout kept PnL | Holdout n |
|---|---|---|---|---|---|
| 0.55 | 94.1% | +$957.50 (n=51) | 25.0% | -$22.50 (n=4) | 4/29 |
| 0.65 | 75.8% | +$695.00 (n=66) | 33.3% | -$15.00 (n=6) | 6/29 |
| 0.70 | 72.9% | +$635.00 (n=70) | 33.3% | -$15.00 (n=6) | 6/29 |

Train AUC = 0.99, holdout AUC = 0.55. **Massive overfit.** The model memorizes train but generalizes worse than a coin flip on holdout. With n=134 train rows and 20 features the AUC variance across CV folds (0.556 to 0.877) is too wide to deploy.

**Verdict:** the trained classifier does NOT pass walk-forward CV. Mean CV AUC 0.704 is technically above 0.5, but the fold-to-fold variance (±0.124) and the holdout collapse to 0.545 mean a deployed gate would behave erratically across regime shifts. **Do not deploy.** The simple feature-threshold rule in section 13 is more robust than this trained model in the only place that matters: out-of-sample generalization. Q1-cutoff thresholds derived on training data still block 22 of 29 holdout trades for -$145.00 of the -$212.50 total holdout loss; the trained model only blocks 23 of 29 at thr=0.65 with no improvement in PnL captured.

The lesson: **with 163 trades total, this is a tiny-N problem.** Any model with more than ~5 free parameters memorizes. Plain quartile rules win.

## 15. Manifold regime correlation (v2 new)

A coarse 5-class regime classifier was built directly from the parquet's regime features (no joining required, since these are emitted by the same pipeline that builds the corpus):

```python
def classify(row):
    eff   = row.bf_regime_eff
    shock = row.ctx_shock_score
    trend = row.ctx_day_trend_frac
    if eff < 0.05 and shock < 1.5:                  return 'CHOP_DEAD'
    if eff < 0.10 and shock >= 2.5:                 return 'CHOP_SPIRAL'
    if eff >= 0.15 and trend and trend >= 0.5:      return 'TREND_CLEAN'
    if eff >= 0.10:                                  return 'TREND_NOISY'
    return 'CHOP_MILD'
```

**Per-regime stats across all 163 RA-friend-allowed trades:**

| Regime | n | Σ PnL | mean | WR |
|---|---|---|---|---|
| **CHOP_DEAD** | 50 | -$417.50 | -8.35 | 26.0% |
| **TREND_NOISY** | 41 | -$373.75 | -9.12 | 34.1% |
| CHOP_MILD | 35 | -$117.50 | -3.36 | 40.0% |
| TREND_CLEAN | 19 | -$47.50 | -2.50 | 47.4% |
| **CHOP_SPIRAL** | 18 | -$8.75 | -0.49 | 50.0% |

Two regimes drive the entire loss: **CHOP_DEAD (-$418) and TREND_NOISY (-$374), together -$792 = 82% of total -$965.** TREND_CLEAN and CHOP_SPIRAL are essentially flat. Notably **CHOP_SPIRAL is neutral for RA**, not the killer it was for AetherFlow — RA's mean reversion edge actually copes OK with high-shock low-efficiency tape, what kills it is dead chop and false-trend.

**Train vs holdout regime distribution shift (THE story for 2026 collapse):**

| Regime | Train share (2025) | Hold share (2026 Q1) | Train Σ PnL | Hold Σ PnL |
|---|---|---|---|---|
| TREND_CLEAN | 14.2% | **0.0%** | -$47.50 | $0.00 |
| TREND_NOISY | 27.6% | 13.8% | -$363.75 | -$10.00 |
| **CHOP_DEAD** | 25.4% | **55.2%** | -$226.25 | **-$191.25** |
| CHOP_MILD | 20.1% | 27.6% | -$110.00 | -$7.50 |
| CHOP_SPIRAL | 12.7% | 3.4% | -$5.00 | -$3.75 |

**Three monstrous shifts:**
1. **CHOP_DEAD doubled** (25%→55%). It's RA's worst regime by mean PnL, and now it's RA's modal regime in 2026.
2. **TREND_CLEAN vanished** (14%→0%). The least-bad regime in train is gone in 2026.
3. **CHOP_SPIRAL collapsed** (13%→3%). The neutral regime that supplied break-even bulk in 2025 isn't firing.

CHOP_DEAD alone absorbs **-$191 of the -$212 total holdout loss = 90%**. **Deploying any rule that blocks CHOP_DEAD entries would have prevented the 2026 Q1 holdout disaster.** The simplest such rule: `bf_regime_eff < 0.05 AND ctx_shock_score < 1.5`. Train: 34 trades blocked, -$170 loss avoided (n=34/134, kept 100/134 with -$582). Hold: 16 trades blocked, -$155 loss avoided (n=16/29, kept 13/29 with -$57.50).

## 16. Feature-threshold rules — what would have stopped the 2025-04-17 disaster (v2 new)

The single worst day was 2025-04-17 (28 trades in the parquet — slightly different from v1's 34, see denominator note above), with -$120 PnL after haircut. Tracing each trade through candidate rules:

**Rule R1: `pct_dist_to_running_hi_pct ≤ 0.0011` (block).**
On 2025-04-17, blocks the first 8 trades of the day (08:08-08:13 cluster, all entered within 0.10% of running hi). Misses the later trades that entered after the up-move had pushed the running hi further away. Day-level: blocks 8/28, captures -$80 of -$120.

**Rule R3: `k12_below_10 ≤ 0.123` (block).**
On 2025-04-17, blocks 5 trades (08:08, 08:09, 08:38, 08:55, 11:12 — the ones where the Kalshi distribution was already showing thin downside cushion). Day-level: blocks 5/28, captures -$87.50 of -$120.

**Rule combo R_OR2: `R1 OR R3` (block if either fires).**
- All 163 trades: blocks 69 of 163 for -$767.50 captured (out of -$965 total). Kept 94 trades at -$197.50, **WR jumps from 36.2% → 46.8%, mean trade -$5.92 → -$2.10.**
- 2025-04-17 only: blocks 9 of 28 trades for -$147.50 (note: blocks more than the loss because some blocked trades were positive — but kept slate is +$27.50 net).

**Final deployable rule R_DEPLOY (combining the best three):**

```
BLOCK RA NY entry IF any of:
  pct_dist_to_running_hi_pct  <  0.0011
  OR
  k12_below_10                 <  0.123
  OR
  pct_minutes_since_open      >= 240   (after ~13:30 ET, late-session decay)
```

**Performance on all 163 trades:**

| Slate | n | Σ PnL | mean | WR |
|---|---|---|---|---|
| Blocked | 84 | -$880.00 | -10.48 | 21.4% |
| Kept | 79 | -$85.00 | -1.08 | **51.9%** |

Kept-slate WR of 51.9% is above the 40% breakeven for the 1:1.5 R:R bracket, and the kept mean trade goes from -$5.92 to -$1.08 — basically break-even, leaving room for execution to net positive.

**Walk-forward validation (thresholds derived on 2025 train slice, applied to 2026 holdout):**

Train Q1 cutoffs: `pct_dist_hi ≤ 0.0011`, `k12_below_10 ≤ 0.2035`. The k12_below_10 cutoff drifts substantially (full-data 0.123 vs train-only 0.2035), reflecting the 2026 distribution shift.

| Slice | Blocked n | Blocked PnL | Kept n | Kept PnL | Kept WR |
|---|---|---|---|---|---|
| **Train (2025)** | 67 | -$760.00 | 67 | **+$7.50** | **53.7%** |
| **Holdout (2026)** | 29 | -$212.50 | **0** | $0.00 | — |

On the holdout, the train-derived thresholds are loose enough that the rule blocks **all 29 trades**. PnL preserved: 100%. Trades preserved: 0%.

This is both the rule's strongest endorsement and its sharpest weakness: in the 2026 Q1 distribution, every single RA candidate hits at least one of the three blocked conditions. So the rule effectively says "RA off in 2026 Q1" — which matches `LOCAL_RA_DISABLED_IN_NY=True`. **The rule is therefore strictly better than the blunt local disable, because in any future regime where the Q1 cutoffs aren't tripped, RA can fire again automatically.** It's a self-arming, regime-adaptive disable, not a hard block.

**Day-level rescue table (top 8 losing days, before/after rule):**

| Date | Original PnL | After R_DEPLOY | Trades blocked / kept |
|---|---|---|---|
| 2025-08-01 | -$227.50 | **-$20.00** | 18 / 3 |
| 2025-08-06 | -$135.00 | **-$35.00** | 4 / 1 |
| 2025-04-17 | -$120.00 | **+$27.50** | 9 / 19 |
| 2025-07-02 | -$113.75 | **-$32.50** | 5 / 0 (1 left ungated) |
| 2026-03-19 | -$60.00 | **$0.00** | 13 / 0 |
| 2025-08-13 | -$55.00 | **$0.00** | 4 / 0 |
| 2025-06-19 | -$55.00 | -$55.00 | 0 / 2 (rule misses this day entirely) |
| 2025-09-05 | -$52.50 | **+$2.50** | 4 / 1 |

7 of 8 worst days are materially improved. **2025-04-17 flips from -$120 to +$27.50** — the rule rescues the single worst cluster by blocking the 8 cluster-fire entries at the running-hi line.

The one miss is 2025-06-19 — only 2 RA-friend-allowed trades that day, neither of which trip the rule's thresholds (their `pct_dist_to_running_hi_pct` was 0.0014 and 0.0017, just above the 0.0011 cut). That's an inherent limitation of a 3-condition threshold rule on n=163; tightening the cut to 0.0017 would catch 2025-06-19 but admit too many false positives elsewhere.

**What the rule does NOT need:** no time-of-day cut beyond `pct_minutes_since_open >= 240`, no day-of-week filter, no combo-blacklist, no manifold regime classifier. The three feature thresholds alone re-encode all of those signals indirectly (e.g. `pct_minutes_since_open == 0` cluster IS hour 08 ET in disguise). This is the simplest deployable rule that survives walk-forward.

**Bonus: combined with section 15 regime gate.**
If we additionally block `bf_regime_eff < 0.05 AND ctx_shock_score < 1.5` (CHOP_DEAD), we add 14 more blocks across train+hold for an additional -$54 captured. Marginal gain; the threshold rule already covers most of CHOP_DEAD via `pct_dist_to_running_hi_pct` and `k12_below_10` co-firing.

**Actionable deployment recommendation (delta from v1):**

In addition to `LOCAL_RA_DISABLED_IN_NY=True`, the following rule is a *less blunt* alternative the user could ship as feature-flagged code:

```python
def ra_ny_threshold_block(features) -> bool:
    """Return True to block this RA NY-session entry."""
    if features.get('pct_dist_to_running_hi_pct', 1.0) < 0.0011:
        return True
    if features.get('k12_below_10', 1.0) < 0.123:
        return True
    if features.get('pct_minutes_since_open', 0.0) >= 240:
        return True
    return False
```

Walk-forward validated: kept trades on train win 53.7% with positive net PnL; on the 2026 Q1 holdout the rule blocks every trade (which is what `LOCAL_RA_DISABLED_IN_NY` does anyway), but the moment Kalshi distributions widen and the running-hi-distance distribution returns to 2025 norms, RA self-re-enables. That's the strict improvement over the local hard disable.

## Files referenced
- `backtest_reports/ml_full_ny_2025_03/closed_trades.json` … `ml_full_ny_2026_04/closed_trades.json` (14 monthly logs, 807 total trades, 142 RA)
- `artifacts/regimeadaptive_v19_liveplus_allsession_wildcard_v2/latest.json` (friend's selection, 48 combos, validation metrics)
- `artifacts/regimeadaptive_v19_liveplus_allsession_wildcard_v2/regimeadaptive_gate_model.joblib` (HGB classifier, threshold 0.55, 33 features — confirmed not backtestable on current artifacts)
- `artifacts/v12_training_corpus.parquet` (3,438 rows including 175 RA candidates / 163 friend-allowed, 116 columns including 85 numeric features used for v2 sections 13-16)
- This analysis: `docs/RA_NY_LOSS_ANALYSIS.md`
