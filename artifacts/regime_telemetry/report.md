# Regime Telemetry Report

_Generated: 2026-05-01T09:01:37.325068Z_

- Total trades parsed: **206**
- Matched to placement: **100**
- Dead-tape clipped: **41**

## Per-(strategy, regime) realized stats

| Strategy | Regime | N | L/S | WR% | Avg PnL | MFE p50 | MFE p75 | MAE p50 | MAE p75 |
|---|---|---|---|---|---|---|---|---|---|
| DynamicEngine3 | calm_trend | 47 | 32/15 | 46.8% | $-12.95 | 4.50 | 10.25 | 8.00 | 13.50 |
| DynamicEngine3 | dead_tape | 49 | 44/5 | 46.9% | $-3.07 | 3.75 | 7.75 | 4.25 | 11.50 |
| DynamicEngine3 | neutral | 18 | 9/9 | 44.4% | $-25.20 | 7.75 | 14.75 | 8.25 | 15.12 |
| DynamicEngine3 | unknown | 90 | 72/18 | 58.9% | $+25.34 | 0.75 | 8.50 | 4.00 | 14.00 |
| FibH1214_fib_236 | dead_tape | 1 | 0/1 | 100.0% | $+13.13 | 0.00 | 0.00 | 0.00 | 0.00 |
| FibH1214_fib_382 | dead_tape | 1 | 0/1 | 100.0% | $+14.38 | 0.00 | 0.00 | 0.00 | 0.00 |

## Dead-tape skip-guard counterfactual (per strategy)

| Strategy | N | Actual $ | CF @12.5/10 $ | Δ @12.5/10 | CF @25/10 $ | Δ @25/10 | Verdict |
|---|---|---|---|---|---|---|---|
| DynamicEngine3 | 40 | $-87.30 | $-623.75 | $-536.45 | $-586.25 | $-498.95 | skip-guard would HURT |
| FibH1214_fib_382 | 1 | $+14.38 | $-7.50 | $-21.88 | $-7.50 | $-21.88 | skip-guard would HURT |
