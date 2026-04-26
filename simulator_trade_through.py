"""Honest trade-through simulator utilities.

This module is the SINGLE SOURCE OF TRUTH for two correctness fixes documented
in docs/STRATEGY_ARCHITECTURE_JOURNAL.md Section 8.25 / 8.26:

1) **Contract pinning.**  A signal is generated on a specific ES outright
   contract (the front-month at signal time, e.g. ESH6, ESM6). The walk-forward
   bar source MUST be limited to that one contract. The historical bug was a
   merged / forward-filled multi-symbol bar series that pulled ESM6 highs into
   the bar window for an ESH6 trade, awarding phantom TPs because ESM6 trades
   $50 above ESH6 due to dividend carry.

2) **Trade-through TP rule.**  An any-touch rule (`bar.high >= tp_price`) treats
   a single 1-tick wick as a fill. Real broker brackets do not. The conservative
   rule used here:

       TP fill confirmed iff:
         bar.high >= tp_price + 1 tick
         AND ( bar.close >= tp_price - 1 tick   # bar held near/through TP
               OR next_bar.high >= tp_price )   # next bar trade-through

   ES tick = 0.25 points. SL stays *any-touch* (asymmetric / conservative —
   losses count even on wicks; this matches how real broker brackets fill
   stops, and is the prudent assumption for accurate PnL).

The contract pinning is calibrated by close-price match to the signal price
(within 0.5 pt) at the signal bar. If the match is ambiguous, it falls back to
a static front-month roll calendar.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import pandas as pd

ES_TICK = 0.25
ES_POINT_VALUE = 5.0  # MES = $5/pt; ES is $50/pt. Default $5 for MES sim.

# Front-month roll calendar (approximate; ES rolls ~2nd Thursday before
# 3rd-Friday expiry). Used as fallback when price-match pinning is ambiguous.
# Boundaries chosen conservatively (a few business days before expiry).
ROLL_CALENDAR = [
    (pd.Timestamp("2024-12-15"), "ESH5"),
    (pd.Timestamp("2025-03-13"), "ESM5"),
    (pd.Timestamp("2025-06-12"), "ESU5"),
    (pd.Timestamp("2025-09-11"), "ESZ5"),
    (pd.Timestamp("2025-12-11"), "ESH6"),
    (pd.Timestamp("2026-03-12"), "ESM6"),
    (pd.Timestamp("2026-06-11"), "ESU6"),
    (pd.Timestamp("2026-09-10"), "ESZ6"),
    (pd.Timestamp("2026-12-10"), "ESH7"),
]


def front_month_by_calendar(ts: pd.Timestamp) -> str:
    """Return the front-month ES contract for the given timestamp,
    using the ROLL_CALENDAR fallback (no price information)."""
    if ts.tzinfo is not None:
        ts_naive = ts.tz_convert("UTC").tz_localize(None)
    else:
        ts_naive = ts
    best = ROLL_CALENDAR[0][1]
    for cut, sym in ROLL_CALENDAR:
        if ts_naive >= cut:
            best = sym
        else:
            break
    return best


def pin_contract(
    parquet_df: pd.DataFrame,
    ts: pd.Timestamp,
    signal_price: float,
    price_tol: float = 0.5,
) -> str:
    """Determine the active ES contract at signal time.

    Strategy:
      - At ts, look up the bar(s) for every available symbol.
      - Choose the symbol whose close price is within ``price_tol`` of
        ``signal_price`` AND has nonzero volume.
      - On tie, prefer front-month by calendar.
      - On no match, fall back to front-month by calendar.
    """
    if parquet_df.index.tz is None and ts.tzinfo is not None:
        # parquet must be tz-aware to compare; caller responsibility but
        # we'll do it here defensively.
        ts = ts.tz_convert("UTC").tz_localize(None)
    elif parquet_df.index.tz is not None and ts.tzinfo is None:
        ts = ts.tz_localize("US/Eastern")
    elif parquet_df.index.tz is not None and ts.tzinfo is not None:
        ts = ts.tz_convert(parquet_df.index.tz)

    # Fast lookup: rows with this exact timestamp
    try:
        rows = parquet_df.loc[ts]
    except KeyError:
        return front_month_by_calendar(ts)
    if isinstance(rows, pd.Series):
        # Single row at this ts → only one symbol available; trust it
        return str(rows["symbol"]) if "symbol" in rows.index else front_month_by_calendar(ts)

    # Filter by price match
    candidates = []
    for _, r in rows.iterrows():
        try:
            close = float(r["close"])
            sym = str(r["symbol"])
            vol = float(r.get("volume", 0) or 0)
        except Exception:
            continue
        if vol <= 0:
            continue
        if abs(close - float(signal_price)) <= price_tol:
            candidates.append(sym)

    if len(candidates) == 1:
        return candidates[0]
    fallback = front_month_by_calendar(ts)
    if not candidates:
        return fallback
    if fallback in candidates:
        return fallback
    # Multiple candidates, neither is fallback front-month → return the one
    # closest to fallback by suffix order
    return candidates[0]


def get_walk_forward_bars(
    parquet_df: pd.DataFrame,
    ts: pd.Timestamp,
    contract: str,
    horizon_bars: int = 30,
) -> pd.DataFrame:
    """Return the next ``horizon_bars`` bars for ``contract`` strictly after ts
    (exclusive of the entry bar itself). Returned df is a contiguous slice with
    columns [open, high, low, close, volume, symbol] indexed by timestamp."""
    if parquet_df.index.tz is not None and ts.tzinfo is None:
        ts = ts.tz_localize("US/Eastern")
    elif parquet_df.index.tz is not None and ts.tzinfo is not None:
        ts = ts.tz_convert(parquet_df.index.tz)

    # Subset by symbol first (cheap if pre-grouped, otherwise full filter)
    if "symbol" in parquet_df.columns:
        sym_df = parquet_df[parquet_df["symbol"] == contract]
    else:
        sym_df = parquet_df
    # Bars strictly after ts
    fwd = sym_df.loc[sym_df.index > ts].head(horizon_bars)
    return fwd


@dataclass
class TradeOutcome:
    exit_reason: str  # 'take' | 'stop' | 'stop_pessimistic' | 'horizon' | 'no_data'
    exit_price: float
    exit_bar: int  # index into walk-forward bars, 0-based; -1 if no_data
    entry_price: float
    pnl_points: float
    # Excursion tracking added 2026-04-25 (early-exit audit follow-up). MFE/MAE
    # are accumulated up to and INCLUDING the exit bar so downstream code can
    # post-process BE-arm corrections (BE-arm fires once MFE crosses
    # tp_dist * trigger_pct; for DE3 tp_dist=25, trigger_pct=0.40 -> 10 pts).
    # Both are positive-magnitude (pts in the favorable / adverse direction).
    mfe_points: float = 0.0
    mae_points: float = 0.0


def simulate_trade_through(
    bars: pd.DataFrame,
    side: str,
    entry_price: float,
    tp_price: float,
    sl_price: float,
    tick_size: float = ES_TICK,
) -> TradeOutcome:
    """Conservative trade-through simulation.

    TP fill confirmed only if:
        bar.high >= tp_price + 1 tick (LONG)  /  bar.low <= tp_price - 1 tick (SHORT)
        AND (bar.close >= tp_price - 1 tick OR next_bar.high >= tp_price)  [LONG]
            (bar.close <= tp_price + 1 tick OR next_bar.low  <= tp_price)  [SHORT]

    SL stays any-touch.

    Conservative tie-break: if a single bar's H/L could hit both TP and SL,
    SL wins (pessimistic).

    MFE/MAE are tracked across walked bars (positive-magnitude points in the
    favorable / adverse direction relative to entry_price). They include the
    exit bar so callers can ask "did we ever reach >= 10 pts in our favor
    before getting stopped?"  This is NOT a model of intra-bar path; it is the
    extreme of bar.high / bar.low across the walked window.
    """
    if bars is None or len(bars) == 0:
        return TradeOutcome(
            "no_data", float(entry_price), -1, float(entry_price), 0.0, 0.0, 0.0
        )

    side_u = str(side).upper()
    bars = bars.reset_index(drop=False)
    n = len(bars)
    mfe_points = 0.0
    mae_points = 0.0
    for j in range(n):
        bar = bars.iloc[j]
        h = float(bar["high"])
        l = float(bar["low"])
        c = float(bar["close"])
        next_bar = bars.iloc[j + 1] if j + 1 < n else None

        # Update excursions BEFORE checking exit so the exit bar's wick is
        # included (a bar that wicks to MFE>=10 pts then reverses to SL still
        # has its MFE seen — important for BE-arm modeling).
        if side_u == "LONG":
            mfe_points = max(mfe_points, h - entry_price)
            mae_points = max(mae_points, entry_price - l)
        elif side_u == "SHORT":
            mfe_points = max(mfe_points, entry_price - l)
            mae_points = max(mae_points, h - entry_price)

        if side_u == "LONG":
            sl_hit = math.isfinite(sl_price) and l <= sl_price
            tp_touch = math.isfinite(tp_price) and h >= (tp_price + tick_size)
            tp_confirm = False
            if tp_touch:
                close_held = c >= (tp_price - tick_size)
                next_through = (
                    next_bar is not None and float(next_bar["high"]) >= tp_price
                )
                tp_confirm = close_held or next_through
            tp_hit = tp_touch and tp_confirm
        elif side_u == "SHORT":
            sl_hit = math.isfinite(sl_price) and h >= sl_price
            tp_touch = math.isfinite(tp_price) and l <= (tp_price - tick_size)
            tp_confirm = False
            if tp_touch:
                close_held = c <= (tp_price + tick_size)
                next_through = (
                    next_bar is not None and float(next_bar["low"]) <= tp_price
                )
                tp_confirm = close_held or next_through
            tp_hit = tp_touch and tp_confirm
        else:
            return TradeOutcome(
                "no_data", float(entry_price), -1, float(entry_price), 0.0, 0.0, 0.0
            )

        if sl_hit and tp_hit:
            # Tie-break: SL wins (pessimistic / conservative)
            pnl = -(abs(entry_price - sl_price))
            return TradeOutcome(
                "stop_pessimistic", float(sl_price), j, float(entry_price),
                float(pnl), float(mfe_points), float(mae_points),
            )
        if sl_hit:
            pnl = -(abs(entry_price - sl_price))
            return TradeOutcome(
                "stop", float(sl_price), j, float(entry_price),
                float(pnl), float(mfe_points), float(mae_points),
            )
        if tp_hit:
            pnl = abs(tp_price - entry_price)
            return TradeOutcome(
                "take", float(tp_price), j, float(entry_price),
                float(pnl), float(mfe_points), float(mae_points),
            )

    # Horizon exit on last bar's close
    last = bars.iloc[n - 1]
    exit_price = float(last["close"])
    if side_u == "LONG":
        pnl = exit_price - entry_price
    else:
        pnl = entry_price - exit_price
    return TradeOutcome(
        "horizon", exit_price, n - 1, float(entry_price),
        float(pnl), float(mfe_points), float(mae_points),
    )


def simulate_trade(
    parquet_df: pd.DataFrame,
    entry_ts,
    side: str,
    entry_price: float,
    tp: float,
    sl: float,
    contract: Optional[str] = None,
    horizon_bars: int = 30,
) -> dict:
    """End-to-end: pin contract (if not given), pull walk-forward bars,
    simulate. Returns a plain dict with the outcome plus contract used."""
    ts = pd.Timestamp(entry_ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("US/Eastern")
    if contract is None:
        contract = pin_contract(parquet_df, ts, entry_price)
    bars = get_walk_forward_bars(parquet_df, ts, contract, horizon_bars=horizon_bars)
    outcome = simulate_trade_through(
        bars, side=side, entry_price=entry_price, tp_price=tp, sl_price=sl
    )
    return {
        "contract": contract,
        "exit_reason": outcome.exit_reason,
        "exit_price": outcome.exit_price,
        "exit_bar": outcome.exit_bar,
        "entry_price": outcome.entry_price,
        "pnl_points": outcome.pnl_points,
        "mfe_points": outcome.mfe_points,
        "mae_points": outcome.mae_points,
        "n_walk_bars": int(len(bars)),
    }
