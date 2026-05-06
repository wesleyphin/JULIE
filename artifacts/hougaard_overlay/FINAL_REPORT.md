# Hougaard Overlay Backtest — Final Report

**Date**: 2026-05-05
**Question**: Does using Hougaard's "Highs & Lows" patterns as a conditional bias overlay (gate-opener via threshold tilt) improve PnL or trade frequency on StdevMl iter12 (1.5/1.0)?

**Honest answer**: **No. Don't ship the gate-opener overlay.** The bias signal is real at the *daily session* level but does not separate good from bad trades among marginal-confidence ML signals — which is exactly what the gate-opener relies on. Detail below.

---

## TL;DR

| Test | Result | Verdict |
|---|---|---|
| 2011-2019 stability of Hougaard claims | Edge survives (Sc B 1.47x lift, Sc C 70%/66%) | ✅ Real signal |
| iter12 calibration | Reliable above 0.65 prob, overconfident below on OOS | ⚠ Don't tilt below 0.65 |
| Tier A: aligned trades better at base=0.75? | VAL: PF Δ **−0.78** (worse). OOS: PF Δ +6.21 (better, n=177) | ❌ Mixed, dominated by VAL miss |
| Tier A bear-regime only | VAL bear: aligned PF **5.17** vs neutral **0.74** (huge!) | ✅ Conditional edge in bear regime |
| Control: aligned vs neutral in marginal zone | VAL @ delta -0.05: aligned PF 4.21 vs neutral 4.66 (worse) | ❌ **Bias adds no edge in the zone the overlay targets** |
| Tier B marginal (overlay-added trades) | All deltas profitable, but PF 1.32-4.70, smaller PnL than naive thr-lower | ❌ Overlay strictly worse than just lowering threshold |

**Recommendation**: Kill the gate-opener overlay. Two surviving angles worth considering separately:
1. **Bear-regime sizing boost** on already-strong (≥0.75) aligned signals — n is small but effect is dramatic.
2. **Daily session bias logging** for journal/observability — not as an active gate, just situational awareness.

---

## 1. Data setup

- **Backtest model**: `iter12_1.5_1.0` (LONG + SHORT HGBClassifier, 60 features each)
- **VAL period**: 2021-01-21 → 2025-01-19 (Biden term, 4 years, 53,732 bars)
- **OOS period**: 2026-01-01 → 2026-04-30 (4 months, 8,523 bars)
- **AUC**: VAL 0.725 (LONG) / 0.690 (SHORT); OOS 0.624 / 0.552
- **Daily DD cap**: $200/day (production setting)
- **Threshold sweep**: base ∈ {0.65, 0.70, 0.75}, delta ∈ {0, -0.02, -0.05, -0.08, -0.12}
- **Hougaard context**: 2,400 sessions tagged. 8.7% Scenario B active, 44.4% Scenario C active, 53.1% any bias

---

## 2. Stability check — 2011-2019 (out-of-sample for the claims)

The original analysis used 2021-2026 (74% bull). Re-ran on 2011-2019 (73% bull) for a regime-variety check.

| Metric | 2011-2019 | 2021-2026 |
|---|---|---|
| Scenario B Mon_low<Fri_low | 51.0% vs 34.6% baseline (1.47x) | 44.4% vs 23.4% (1.90x) |
| Scenario B Mon_close<Fri_low | **21.1% vs 11.5% (1.83x)** | (not measured original) |
| Scenario B bear regime amp | 60.0% vs bull 47.2% | 68.6% vs bull 34.1% |
| Scenario B high-vol amp | **64.8% vs low-vol 43.6%** | (not measured original) |
| Scenario C Tue-breaks-Mon-high | 70.4% week extends | 78.3% |
| Scenario C Tue-breaks-Mon-low | 65.8% week extends | 71.4% |

**Both scenarios survive in 2011-2019 with smaller but still meaningful lifts. Bear-regime amplification confirmed in both windows. New finding: high-vol regime amplifies the same way (64.8% vs 43.6%).**

→ The Hougaard claims have real statistical content as session-level bias, and that bias is regime-conditional.

---

## 3. iter12 calibration (reliability diagram)

| Prob bin | VAL n | VAL actual WR | OOS n | OOS actual WR |
|---|---|---|---|---|
| 0.30-0.37 | 1,154 | 44.3% | 234 | 31.6% |
| 0.37-0.44 | 5,696 | 51.4% | 912 | 37.2% |
| 0.44-0.51 | 10,909 | 55.5% | 1,862 | 41.0% |
| 0.51-0.58 | 11,422 | 58.9% | 1,879 | 43.9% |
| 0.58-0.65 | 9,279 | 64.4% | 1,440 | 47.8% |
| 0.65-0.72 | 6,793 | **71.9%** | 818 | 59.4% |
| 0.72-0.79 | 4,760 | **77.5%** | 488 | 68.2% |
| 0.79-0.86 | 2,559 | 81.5% | 492 | **77.6%** |
| 0.86-0.93 | 920 | 84.6% | 335 | **84.8%** |

**Insight**: VAL is well-calibrated across the board. OOS is well-calibrated only above ~0.65 — below that, the model is *overconfident* (predicts 0.45 prob, actual 41% win — close, but cumulative losses skew the strategy economics). This is the boundary where overlay tilts should respect.

---

## 4. Tier A — aligned vs opposed vs neutral at base=0.75

### Headline (full populations)

| Period | Bucket | n | WR | PF | avg PnL | Δ vs neutral |
|---|---|---|---|---|---|---|
| **VAL** | aligned | 1,167 | 77.6% | 5.79 | $+65 | **PF −0.78** |
| VAL | opposed | 1,980 | 83.4% | 6.98 | $+61 | PF +0.41 |
| VAL | neutral | 2,794 | 80.9% | 6.57 | $+73 | (baseline) |
| **OOS** | aligned | 177 | 88.7% | 11.94 | $+97 | **PF +6.21** |
| OOS | opposed | 318 | 77.0% | 5.06 | $+93 | PF −0.67 |
| OOS | neutral | 586 | 77.5% | 5.73 | $+81 | (baseline) |

VAL says aligned trades are *worse* than neutral. OOS says aligned trades are *much better*. **VAL has 23x more samples** so weight VAL heavily — aligned alignment is a slight negative on average.

### Regime-stratified (the interesting cut)

| Period | Regime | Aligned n / PF | Neutral n / PF | Δ PF |
|---|---|---|---|---|
| VAL | bull | 1,079 / 5.81 | 2,771 / 6.64 | −0.83 |
| **VAL** | **bear** | **88 / 5.17** | **23 / 0.74** | **+4.42** |
| VAL | high_vol | 153 / 5.10 | 398 / 5.41 | −0.31 |
| VAL | low_vol | 1,014 / 5.94 | 2,396 / 6.82 | −0.88 |
| OOS | bull | 163 / 9.78 | 561 / 5.62 | +4.16 |
| OOS | bear | 14 / inf (100% WR) | 25 / 7.31 | +inf |
| OOS | high_vol | 6 / 2.93 | 22 / 6.18 | −3.24 |
| OOS | low_vol | 171 / 14.21 | 564 / 5.70 | +8.51 |

**Key insight**: VAL bear-regime aligned trades have **PF 5.17 with avg $+26**, while neutral bear-regime trades have **PF 0.74 with avg $-5 (LOSING)**. Sample is small (88 vs 23) but the effect direction is dramatic and matches the original conditional-probability finding (60.0% Scenario-B in bear vs 47.2% bull).

OOS confirms: bear-regime aligned trades 100% WR (n=14). But sample tiny.

→ **The Hougaard alignment edge is concentrated in bear regime, not bull.** This matters because the production environment is currently bull (74-76%).

---

## 5. The decisive test — Control comparison

The overlay is only useful if **aligned trades in the marginal-confidence zone (between base+delta and base) are better than neutral trades in that same zone**. Otherwise, the overlay is just a fancy way to lower the threshold, and a uniform threshold lower works better.

### VAL @ base=0.75

| delta | bucket | n | WR | PF | avg | PnL |
|---|---|---|---|---|---|---|
| −0.05 | aligned | 953 | 73.3% | 4.21 | $+55 | $+52,538 |
| −0.05 | opposed | 1,231 | 75.9% | 4.42 | $+48 | $+59,641 |
| −0.05 | **neutral** | **1,624** | **75.4%** | **4.66** | **$+57** | **$+92,881** |
| −0.08 | aligned | 1,636 | 72.4% | 4.08 | $+56 | $+90,960 |
| −0.08 | opposed | 2,054 | 74.1% | 4.07 | $+48 | $+97,804 |
| −0.08 | **neutral** | **2,871** | **75.6%** | **4.70** | **$+61** | **$+175,672** |
| −0.12 | aligned | 2,723 | 70.2% | 3.83 | $+54 | $+146,587 |
| −0.12 | opposed | 3,257 | 72.8% | 3.74 | $+47 | $+153,324 |
| −0.12 | **neutral** | **4,713** | **74.3%** | **4.47** | **$+60** | **$+283,602** |

### OOS @ base=0.75

| delta | bucket | n | WR | PF | avg | PnL |
|---|---|---|---|---|---|---|
| −0.05 | aligned | 41 | 56.1% | 1.55 | $+44 | $+1,812 |
| −0.05 | opposed | 115 | 60.9% | 2.21 | $+67 | $+7,734 |
| −0.05 | **neutral** | **147** | **66.0%** | **2.95** | **$+73** | **$+10,774** |
| −0.08 | aligned | 93 | 59.1% | 1.98 | $+68 | $+6,368 |
| −0.08 | opposed | 200 | 63.0% | 2.37 | $+75 | $+14,953 |
| −0.08 | **neutral** | **232** | **72.0%** | **4.02** | **$+101** | **$+23,403** |

**Aligned bucket is the WORST in every cell, both VAL and OOS, at every delta.**

→ The Hougaard bias does NOT separate good from bad trades among marginal-confidence ML signals. The hypothesis that powered the gate-opener proposal is falsified. Lowering the threshold uniformly produces *better* trades than tilting on alignment.

---

## 6. Tier B — overlay vs naive threshold lowering

| Config | n | PF | PnL | max DD |
|---|---|---|---|---|
| **VAL** baseline @ 0.75 | 5,941 | 6.51 | $+399k | $-816 |
| VAL overlay 0.75 / Δ−0.12 | 7,408 | 5.80 | $+476k | $-1,244 |
| VAL baseline @ 0.65 (= naive lower by 0.10) | 14,439 | 4.98 | $+871k | $-1,931 |
| **OOS** baseline @ 0.75 | 1,081 | 5.98 | $+94.3k | $-779 |
| OOS overlay 0.75 / Δ−0.12 | 1,174 | 4.94 | $+100.5k | $-1,570 |
| OOS baseline @ 0.65 | 1,638 | 3.73 | $+137.4k | $-2,896 |

**Overlay deltas of −0.05 to −0.12 produce small PnL gains over baseline (+$6k OOS, +$77k VAL) at the cost of meaningfully larger drawdowns.** Naive threshold lowering produces ~5× more PnL gain at the cost of even larger DD. Risk-adjusted, neither option is compelling.

---

## 7. Where to go from here (ranked)

### A. Don't ship the threshold-tilt overlay. ❌
The control comparison falsifies the core hypothesis. Marginal-confidence aligned trades are not better than marginal-confidence neutral trades.

### B. Investigate bear-regime sizing boost. ⚠ (moderate priority)
Hypothesis: at full base threshold (0.75) and in bear regime, Hougaard-aligned trades have 7× the PF of neutral trades (VAL: 5.17 vs 0.74). Don't *open the gate*; instead, *size up* trades that are already passing the gate when bias is aligned and regime is bear.

Mechanism: existing gate fires → check `(bias_dir == signal_dir) AND not bull_regime` → if true, +20% size.

Validation needed:
- This is 88 VAL trades and 14 OOS — far too few for a confident size-tilt rec
- Need to test on full 2011-2019 window via the same harness (covers 2018 vol spike, 2022 bear)
- Would need to extend backtest infra to that window, costs maybe a half-day

### C. Ship the offline context engine for journaling/observability only. ✅ (low risk)
The engine works, the data is real, the bias direction is well-defined. Just write `bias_direction`, `bias_strength`, `active_scenarios` into the daily journal alongside actual trade outcomes for forward-looking analysis. **No gating change.** Pure logging.

### D. Don't bother with Scenario A. ❌
Both windows show ~1.1x lift. Not worth the engineering surface.

---

## 8. Files produced

```
tools/
  hougaard_context_offline.py             ← per-session context engine (reusable)
  hougaard_overlay_backtest.py            ← Tier A + Tier B + calibration
  hougaard_overlay_regime_drilldown.py    ← regime-stratified Tier A + control test

artifacts/hougaard_overlay/
  context_table.parquet                   ← 2,400 session rows, 2017-2026
  calibration_diagram.txt                 ← reliability of iter12 by prob bin
  tier_a_stratification.csv               ← buckets at base=0.75 (VAL+OOS)
  tier_a_summary.txt
  tier_a_regime_drilldown.csv             ← buckets × regime
  tier_b_sweep.csv                        ← full delta sweep
  tier_b_summary.txt                      ← human-readable comparison
  tier_b_marginal_trades_VAL.csv          ← every overlay-added trade w/ context
  tier_b_marginal_trades_OOS.csv
  tier_b_per_regime.csv                   ← OOS marginal slice (small sample)
  control_comparison.csv                  ← aligned/opposed/neutral in marginal zone
  regime_drilldown_summary.txt
  FINAL_REPORT.md                         ← this file
```

---

## 9. Caveats

1. **OOS is only 4 months and predominantly bull regime**. The bear-regime alignment edge can't be confirmed there. The case rests on VAL bear (n=88) and the original 2011-2019 conditional probability analysis.
2. **iter12 already encodes a lot of session-level context** (cross-sectional ToD features, σ regimes, vwap_60). The bias may not be additive because the model is already getting most of it. This is consistent with the control comparison finding.
3. **DD cap interactions**: both overlay and threshold-lower hit daily DD muting more often as more trades fire. The PF differences may be partially DD-cap artifacts. Not investigated further given the headline result.
4. **No transaction cost modeling**: PT_USD = $5 is the contract value but slippage/fees not subtracted. Marginal trades have lower PF and would be hit harder by realistic costs — strengthens the "don't ship" verdict.
