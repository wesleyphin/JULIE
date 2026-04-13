import argparse
import datetime as dt
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest_symbol_context import apply_symbol_mode
from config import CONFIG
from regime_sltp_params import PARAMS as REGIME_SLTP_PARAMS
from regime_sltp_params import REVERTED_COMBOS as REGIME_REVERTED_COMBOS


NY_TZ = ZoneInfo("America/New_York")
TICK_SIZE = 0.25
SESSION_NAMES = ("ASIA", "LONDON", "NY_AM", "NY_PM", "CLOSED")
DAY_NAMES = ("MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN")
YEARLY_Q = ("Q1", "Q2", "Q3", "Q4")
MONTHLY_Q = ("W1", "W2", "W3", "W4")
SESSION_CODE = {name: idx for idx, name in enumerate(SESSION_NAMES)}
DAY_CODE = {name: idx for idx, name in enumerate(DAY_NAMES)}
YEARLY_Q_CODE = {name: idx for idx, name in enumerate(YEARLY_Q)}
MONTHLY_Q_CODE = {name: idx for idx, name in enumerate(MONTHLY_Q)}
COMBO_SPACE = len(YEARLY_Q) * len(MONTHLY_Q) * len(DAY_NAMES) * len(SESSION_NAMES)
OPPOSITE_SIGNAL_THRESHOLD = 3


@dataclass(frozen=True)
class StrategyParams:
    sma_fast: int
    sma_slow: int
    cross_atr_mult: float
    use_reversion: bool


def _round_points_to_tick(points: float) -> float:
    ticks = max(1, int(math.ceil(abs(float(points)) / TICK_SIZE)))
    return ticks * TICK_SIZE


def _resolve_sl_tp_conflict(
    side: int,
    bar_open: float,
    bar_close: float,
    stop_price: float,
    take_price: float,
) -> tuple[float, str]:
    is_green = float(bar_close) >= float(bar_open)
    stop_first = is_green if side > 0 else not is_green
    return (stop_price, "stop") if stop_first else (take_price, "take")


def _parse_datetime(raw: str, is_end: bool) -> pd.Timestamp:
    text = str(raw or "").strip()
    if not text:
        raise ValueError("Datetime text is empty")
    ts = pd.Timestamp(text)
    if ts.tzinfo is None:
        ts = ts.tz_localize(NY_TZ)
    else:
        ts = ts.tz_convert(NY_TZ)
    if len(text) <= 10:
        if is_end:
            ts = ts.replace(hour=23, minute=59, second=59, microsecond=999999)
        else:
            ts = ts.replace(hour=0, minute=0, second=0, microsecond=0)
    return ts


def _parse_int_list(raw: str) -> list[int]:
    out: list[int] = []
    for item in str(raw or "").split(","):
        text = item.strip()
        if text:
            out.append(int(text))
    return out


def _parse_float_list(raw: str) -> list[float]:
    out: list[float] = []
    for item in str(raw or "").split(","):
        text = item.strip()
        if text:
            out.append(float(text))
    return out


def _parse_reversion_list(raw: str) -> list[bool]:
    truthy = {"1", "true", "yes", "on", "reversion", "revert"}
    falsy = {"0", "false", "no", "off", "noreversion", "no_reversion"}
    out: list[bool] = []
    for item in str(raw or "").split(","):
        text = item.strip().lower()
        if not text:
            continue
        if text in truthy:
            out.append(True)
        elif text in falsy:
            out.append(False)
        else:
            raise ValueError(f"Unsupported reversion token: {item}")
    return out


def _combo_id(yq: int, mq: int, dow: int, sess: int) -> int:
    return (((int(yq) * len(MONTHLY_Q)) + int(mq)) * len(DAY_NAMES) + int(dow)) * len(SESSION_NAMES) + int(sess)


def _combo_key_from_id(combo_id: int) -> str:
    sess = combo_id % len(SESSION_NAMES)
    rem = combo_id // len(SESSION_NAMES)
    dow = rem % len(DAY_NAMES)
    rem //= len(DAY_NAMES)
    mq = rem % len(MONTHLY_Q)
    yq = rem // len(MONTHLY_Q)
    return f"{YEARLY_Q[yq]}_{MONTHLY_Q[mq]}_{DAY_NAMES[dow]}_{SESSION_NAMES[sess]}"


def _load_bars(source: Path, symbol_mode: str, symbol_method: str) -> tuple[pd.DataFrame, str]:
    df = pd.read_parquet(source, columns=["open", "high", "low", "close", "volume", "symbol"])
    if df.index.tz is None:
        df.index = df.index.tz_localize(NY_TZ)
    else:
        df.index = df.index.tz_convert(NY_TZ)
    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"]).sort_index()
    filtered, label, _ = apply_symbol_mode(df, symbol_mode, symbol_method)
    filtered = filtered.drop(columns=["symbol"], errors="ignore").copy()
    return filtered, label


def _build_combo_arrays(index: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    months = index.month.to_numpy(dtype=np.int16)
    days = index.day.to_numpy(dtype=np.int16)
    dow = index.dayofweek.to_numpy(dtype=np.int8)
    hours = index.hour.to_numpy(dtype=np.int8)

    yq = np.where(months <= 3, 0, np.where(months <= 6, 1, np.where(months <= 9, 2, 3))).astype(np.int8)
    mq = np.where(days <= 7, 0, np.where(days <= 14, 1, np.where(days <= 21, 2, 3))).astype(np.int8)

    sess = np.full(len(index), SESSION_CODE["CLOSED"], dtype=np.int8)
    sess[(hours >= 18) | (hours < 3)] = SESSION_CODE["ASIA"]
    sess[(hours >= 3) & (hours < 8)] = SESSION_CODE["LONDON"]
    sess[(hours >= 8) & (hours < 12)] = SESSION_CODE["NY_AM"]
    sess[(hours >= 12) & (hours < 17)] = SESSION_CODE["NY_PM"]

    combo_id_arr = (((yq.astype(np.int16) * len(MONTHLY_Q)) + mq.astype(np.int16)) * len(DAY_NAMES) + dow.astype(np.int16)) * len(SESSION_NAMES) + sess.astype(np.int16)
    return combo_id_arr.astype(np.int16), sess


def _build_regime_lookup() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    revert_lookup = np.zeros(COMBO_SPACE, dtype=bool)
    long_sl = np.full(COMBO_SPACE, 2.0, dtype=np.float32)
    long_tp = np.full(COMBO_SPACE, 3.0, dtype=np.float32)
    short_sl = np.full(COMBO_SPACE, 2.0, dtype=np.float32)
    short_tp = np.full(COMBO_SPACE, 3.0, dtype=np.float32)

    for combo_key in REGIME_REVERTED_COMBOS:
        parts = str(combo_key).split("_")
        if len(parts) != 4:
            continue
        combo_idx = _combo_id(
            YEARLY_Q_CODE[parts[0]],
            MONTHLY_Q_CODE[parts[1]],
            DAY_CODE[parts[2]],
            SESSION_CODE[parts[3]],
        )
        revert_lookup[combo_idx] = True

    for combo_key, payload in REGIME_SLTP_PARAMS.items():
        parts = str(combo_key).split("_")
        if len(parts) != 4:
            continue
        combo_idx = _combo_id(
            YEARLY_Q_CODE[parts[0]],
            MONTHLY_Q_CODE[parts[1]],
            DAY_CODE[parts[2]],
            SESSION_CODE[parts[3]],
        )
        long_params = payload.get("LONG", {})
        short_params = payload.get("SHORT", {})
        long_sl[combo_idx] = _round_points_to_tick(max(float(long_params.get("sl", 2.0)), 2.0))
        long_tp[combo_idx] = _round_points_to_tick(max(float(long_params.get("tp", 3.0)), 3.0))
        short_sl[combo_idx] = _round_points_to_tick(max(float(short_params.get("sl", 2.0)), 2.0))
        short_tp[combo_idx] = _round_points_to_tick(max(float(short_params.get("tp", 3.0)), 3.0))

    return revert_lookup, long_sl, long_tp, short_sl, short_tp


def _build_holiday_mask(index: pd.DatetimeIndex, session_codes: np.ndarray) -> np.ndarray:
    exec_cfg = CONFIG.get("BACKTEST_EXECUTION", {}) or {}
    if not bool(exec_cfg.get("enforce_us_holiday_closure", True)):
        return np.zeros(len(index), dtype=bool)
    start_day = pd.Timestamp(index.min().date())
    end_day = pd.Timestamp(index.max().date())
    holidays = USFederalHolidayCalendar().holidays(start=start_day, end=end_day)
    closed_dates = {ts.date() for ts in holidays}
    for raw in exec_cfg.get("extra_closed_dates_et", []) or []:
        try:
            closed_dates.add(pd.Timestamp(str(raw)).date())
        except Exception:
            continue
    blocked_sessions = {str(item).upper() for item in (exec_cfg.get("holiday_closure_sessions_et", []) or []) if item}
    if not blocked_sessions or "ALL" in blocked_sessions:
        blocked = np.ones(len(index), dtype=bool)
    else:
        blocked = np.isin(
            session_codes,
            [SESSION_CODE[name] for name in blocked_sessions if name in SESSION_CODE],
        )
    dates = index.date
    return np.array([day in closed_dates for day in dates], dtype=bool) & blocked


def _rolling_cache(close: np.ndarray, windows: Iterable[int]) -> dict[int, np.ndarray]:
    series = pd.Series(close)
    cache: dict[int, np.ndarray] = {}
    for window in sorted({int(w) for w in windows if int(w) > 0}):
        cache[window] = series.rolling(window).mean().to_numpy(dtype=np.float64)
    return cache


def _atr_array(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    prev_close = np.empty_like(close)
    prev_close[0] = close[0]
    prev_close[1:] = close[:-1]
    tr = np.maximum.reduce([
        np.abs(high - low),
        np.abs(high - prev_close),
        np.abs(low - prev_close),
    ])
    return pd.Series(tr).rolling(int(period)).mean().to_numpy(dtype=np.float64)


def _build_signal_package(
    combo_ids: np.ndarray,
    session_codes: np.ndarray,
    close: np.ndarray,
    sma_fast: np.ndarray,
    sma_slow: np.ndarray,
    atr: np.ndarray,
    params: StrategyParams,
    revert_lookup: np.ndarray,
    long_sl_lookup: np.ndarray,
    long_tp_lookup: np.ndarray,
    short_sl_lookup: np.ndarray,
    short_tp_lookup: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    signal_side = np.zeros(len(close), dtype=np.int8)
    valid = (
        np.isfinite(sma_fast)
        & np.isfinite(sma_slow)
        & np.isfinite(atr)
        & (session_codes != SESSION_CODE["CLOSED"])
    )
    cross_thresh = atr * float(params.cross_atr_mult)
    long_mask = valid & (sma_fast > sma_slow) & (close < (sma_fast - cross_thresh))
    short_mask = valid & (sma_fast < sma_slow) & (close > (sma_fast + cross_thresh))
    signal_side[long_mask] = 1
    signal_side[short_mask] = -1

    original_side = signal_side.copy()
    if params.use_reversion:
        revert_mask = revert_lookup[combo_ids]
        signal_side = np.where(revert_mask & (signal_side != 0), -signal_side, signal_side).astype(np.int8)

    sl = np.zeros(len(close), dtype=np.float32)
    tp = np.zeros(len(close), dtype=np.float32)
    long_rows = signal_side > 0
    short_rows = signal_side < 0
    sl[long_rows] = long_sl_lookup[combo_ids[long_rows]]
    tp[long_rows] = long_tp_lookup[combo_ids[long_rows]]
    sl[short_rows] = short_sl_lookup[combo_ids[short_rows]]
    tp[short_rows] = short_tp_lookup[combo_ids[short_rows]]
    return signal_side, sl, tp, original_side


def _contracts_for_drawdown(equity: float, peak: float, default_contracts: int) -> int:
    exec_cfg = CONFIG.get("BACKTEST_EXECUTION", {}) or {}
    base_contracts = int(default_contracts or exec_cfg.get("drawdown_size_scaling_base_contracts", 5) or 5)
    target_contracts = base_contracts

    # RegimeAdaptive-specific growth ratchet:
    # size increases only after new realized equity highs, capped by the
    # remaining strategy budget, then drawdown scaling can trim it back.
    if bool(exec_cfg.get("regimeadaptive_growth_size_scaling_enabled", False)):
        growth_step_usd = float(exec_cfg.get("regimeadaptive_growth_profit_step_usd", 0.0) or 0.0)
        growth_cap = int(exec_cfg.get("regimeadaptive_growth_size_scaling_max_contracts", base_contracts) or base_contracts)
        growth_anchor = str(exec_cfg.get("regimeadaptive_growth_anchor", "peak") or "peak").strip().lower()
        anchor_equity = float(peak if growth_anchor == "peak" else equity)
        if growth_step_usd > 0.0 and growth_cap > base_contracts:
            realized_profit = max(0.0, anchor_equity)
            growth_steps = int(realized_profit / growth_step_usd)
            target_contracts = min(growth_cap, base_contracts + max(0, growth_steps))
        else:
            target_contracts = max(base_contracts, growth_cap)

    if not bool(exec_cfg.get("drawdown_size_scaling_enabled", True)):
        return int(target_contracts)
    start_usd = float(exec_cfg.get("drawdown_size_scaling_start_usd", 0.0) or 0.0)
    max_usd = float(exec_cfg.get("drawdown_size_scaling_max_usd", 2000.0) or 2000.0)
    min_contracts = int(exec_cfg.get("drawdown_size_scaling_min_contracts", 1) or 1)
    if max_usd <= start_usd:
        return int(target_contracts)
    realized_dd = max(0.0, float(peak - equity))
    if realized_dd <= start_usd:
        return int(target_contracts)
    if realized_dd >= max_usd:
        return min_contracts
    span = max_usd - start_usd
    dd_above = realized_dd - start_usd
    contract_range = max(0, int(target_contracts) - min_contracts)
    if contract_range <= 0:
        return int(target_contracts)
    step_usd = span / float(contract_range)
    bucket = int(dd_above / step_usd) if step_usd > 0 else contract_range
    bucket = min(max(bucket, 0), contract_range)
    return max(min_contracts, int(target_contracts) - bucket)


def _simulate(
    df: pd.DataFrame,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    combo_ids: np.ndarray,
    session_codes: np.ndarray,
    holiday_mask: np.ndarray,
    signal_side: np.ndarray,
    signal_sl: np.ndarray,
    signal_tp: np.ndarray,
    original_side: np.ndarray,
    params: StrategyParams,
    contracts: int,
    point_value: float,
    fee_per_contract_rt: float,
) -> dict:
    exec_cfg = CONFIG.get("BACKTEST_EXECUTION", {}) or {}
    gap_fills = bool(exec_cfg.get("gap_fills", True))
    no_entry_window = bool(exec_cfg.get("enforce_no_new_entries_window", True))
    no_entry_start = int(exec_cfg.get("no_new_entries_start_hour_et", 16) or 16)
    no_entry_end = int(exec_cfg.get("no_new_entries_end_hour_et", 18) or 18)
    force_flat_enabled = bool(exec_cfg.get("force_flat_at_time", True))
    force_flat_hour = int(exec_cfg.get("force_flat_hour_et", 16) or 16)
    force_flat_minute = int(exec_cfg.get("force_flat_minute_et", 0) or 0)
    early_exit_cfg = (CONFIG.get("EARLY_EXIT", {}) or {}).get("RegimeAdaptive", {}) or {}
    early_exit_enabled = bool(early_exit_cfg.get("enabled", False))
    early_exit_bars = int(early_exit_cfg.get("exit_if_not_green_by", 30) or 30)
    early_exit_crosses = int(early_exit_cfg.get("max_profit_crosses", 4) or 4)

    index = df.index
    opens = df["open"].to_numpy(dtype=np.float64)
    highs = df["high"].to_numpy(dtype=np.float64)
    lows = df["low"].to_numpy(dtype=np.float64)
    closes = df["close"].to_numpy(dtype=np.float64)
    hours = index.hour.to_numpy(dtype=np.int8)
    minutes = index.minute.to_numpy(dtype=np.int8)
    mask = np.asarray((index >= start_time) & (index <= end_time), dtype=bool)
    test_positions = np.flatnonzero(mask)
    if test_positions.size == 0:
        raise ValueError("No bars found in the requested range.")

    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0
    gross_profit = 0.0
    gross_loss = 0.0
    wins = 0
    losses = 0
    trades = 0
    session_counts: dict[str, int] = {}
    exit_reason_counts: dict[str, int] = {}
    trade_examples: list[dict] = []

    active = False
    pending_entry: Optional[dict] = None
    pending_exit = False
    opposite_signal_count = 0
    active_trade: dict = {}

    def record_close(exit_price: float, exit_time: pd.Timestamp, reason: str) -> None:
        nonlocal active, active_trade, equity, peak, max_drawdown
        nonlocal gross_profit, gross_loss, wins, losses, trades, pending_exit
        pnl_points = (
            float(exit_price) - float(active_trade["entry_price"])
            if active_trade["side"] > 0
            else float(active_trade["entry_price"]) - float(exit_price)
        )
        fee_paid = fee_per_contract_rt * int(active_trade["size"])
        pnl_net = pnl_points * point_value * int(active_trade["size"]) - fee_paid
        equity += pnl_net
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)
        gross_profit += max(pnl_net, 0.0)
        gross_loss += min(pnl_net, 0.0)
        wins += int(pnl_net > 0)
        losses += int(pnl_net <= 0)
        trades += 1
        exit_reason_counts[reason] = exit_reason_counts.get(reason, 0) + 1
        session_counts[active_trade["session"]] = session_counts.get(active_trade["session"], 0) + 1
        if len(trade_examples) < 25:
            trade_examples.append(
                {
                    "entry_time": active_trade["entry_time"].isoformat(),
                    "exit_time": exit_time.isoformat(),
                    "side": "LONG" if active_trade["side"] > 0 else "SHORT",
                    "combo_key": active_trade["combo_key"],
                    "reverted": active_trade["reverted"],
                    "pnl_net": round(pnl_net, 2),
                    "exit_reason": reason,
                }
            )
        active = False
        active_trade = {}
        pending_exit = False

    for i in test_positions:
        ts = index[i]
        bar_open = float(opens[i])
        bar_high = float(highs[i])
        bar_low = float(lows[i])
        bar_close = float(closes[i])
        holiday_closed_now = bool(holiday_mask[i])
        entry_window_blocked = bool(no_entry_window and (hours[i] >= no_entry_start) and (hours[i] < no_entry_end))
        force_flat_now = bool(force_flat_enabled and hours[i] == force_flat_hour and minutes[i] >= force_flat_minute)

        if holiday_closed_now and active:
            record_close(bar_open, ts, "holiday_flat")
            opposite_signal_count = 0
        if force_flat_now and active:
            record_close(bar_open, ts, "session_flat")
            opposite_signal_count = 0
        if pending_exit and active:
            record_close(bar_open, ts, "reverse")

        if pending_entry is not None:
            if holiday_closed_now or entry_window_blocked:
                pending_entry = None
            else:
                active_trade = {
                    "side": pending_entry["side"],
                    "entry_price": bar_open,
                    "entry_time": ts,
                    "sl_dist": pending_entry["sl_dist"],
                    "tp_dist": pending_entry["tp_dist"],
                    "size": _contracts_for_drawdown(equity, peak, contracts),
                    "bars_held": 0,
                    "profit_crosses": 0,
                    "was_green": None,
                    "combo_key": pending_entry["combo_key"],
                    "reverted": pending_entry["reverted"],
                    "session": pending_entry["session"],
                }
                active = True
                pending_entry = None
                opposite_signal_count = 0

        if active:
            entry_price = float(active_trade["entry_price"])
            sl_dist = float(active_trade["sl_dist"])
            tp_dist = float(active_trade["tp_dist"])
            if active_trade["side"] > 0:
                stop_price = entry_price - sl_dist
                take_price = entry_price + tp_dist
                if gap_fills and bar_open <= stop_price:
                    record_close(stop_price, ts, "stop_gap")
                elif gap_fills and bar_open >= take_price:
                    record_close(take_price, ts, "take_gap")
                elif active and bar_low <= stop_price and bar_high >= take_price:
                    exit_price, reason = _resolve_sl_tp_conflict(1, bar_open, bar_close, stop_price, take_price)
                    record_close(exit_price, ts, reason)
                elif active and bar_low <= stop_price:
                    record_close(stop_price, ts, "stop")
                elif active and bar_high >= take_price:
                    record_close(take_price, ts, "take")
            else:
                stop_price = entry_price + sl_dist
                take_price = entry_price - tp_dist
                if gap_fills and bar_open >= stop_price:
                    record_close(stop_price, ts, "stop_gap")
                elif gap_fills and bar_open <= take_price:
                    record_close(take_price, ts, "take_gap")
                elif active and bar_high >= stop_price and bar_low <= take_price:
                    exit_price, reason = _resolve_sl_tp_conflict(-1, bar_open, bar_close, stop_price, take_price)
                    record_close(exit_price, ts, reason)
                elif active and bar_high >= stop_price:
                    record_close(stop_price, ts, "stop")
                elif active and bar_low <= take_price:
                    record_close(take_price, ts, "take")

            if active and early_exit_enabled:
                active_trade["bars_held"] += 1
                is_green = bar_close > entry_price if active_trade["side"] > 0 else bar_close < entry_price
                was_green = active_trade.get("was_green")
                if was_green is not None and is_green != was_green:
                    active_trade["profit_crosses"] += 1
                active_trade["was_green"] = is_green
                if (
                    (active_trade["bars_held"] >= early_exit_bars and not is_green)
                    or (active_trade["profit_crosses"] > early_exit_crosses)
                ):
                    record_close(bar_close, ts, "early_exit")

        if holiday_closed_now:
            continue

        current_signal = int(signal_side[i])
        if current_signal == 0:
            continue

        sig_payload = {
            "side": current_signal,
            "sl_dist": float(signal_sl[i]),
            "tp_dist": float(signal_tp[i]),
            "combo_key": _combo_key_from_id(int(combo_ids[i])),
            "reverted": bool(params.use_reversion and current_signal != int(original_side[i])),
            "session": SESSION_NAMES[int(session_codes[i])],
        }
        if not active:
            if pending_entry is None:
                pending_entry = sig_payload
            opposite_signal_count = 0
            continue
        if active_trade["side"] == current_signal:
            opposite_signal_count = 0
            continue
        opposite_signal_count += 1
        if opposite_signal_count >= OPPOSITE_SIGNAL_THRESHOLD:
            pending_exit = True
            if pending_entry is None:
                pending_entry = sig_payload
            opposite_signal_count = 0

    if active:
        final_idx = int(test_positions[-1])
        record_close(float(closes[final_idx]), index[final_idx], "end_of_range")

    profit_factor = gross_profit / abs(gross_loss) if gross_loss < 0 else float("inf") if gross_profit > 0 else 0.0
    winrate = (wins / trades) * 100.0 if trades else 0.0
    pnl_to_dd = equity / max(max_drawdown, 1.0)
    return {
        "params": asdict(params),
        "equity": float(round(equity, 2)),
        "trades": int(trades),
        "wins": int(wins),
        "losses": int(losses),
        "winrate": float(round(winrate, 4)),
        "max_drawdown": float(round(max_drawdown, 2)),
        "gross_profit": float(round(gross_profit, 2)),
        "gross_loss": float(round(gross_loss, 2)),
        "profit_factor": None if not math.isfinite(profit_factor) else float(round(profit_factor, 4)),
        "pnl_to_max_dd": float(round(pnl_to_dd, 6)),
        "exit_reasons": exit_reason_counts,
        "sessions": session_counts,
        "trade_examples": trade_examples,
    }


def _rank_results(results: list[dict], rank_by: str) -> list[dict]:
    key = str(rank_by or "pnl_to_dd").lower()
    if key == "pnl":
        return sorted(
            results,
            key=lambda item: (
                float(item.get("equity", 0.0)),
                -float(item.get("max_drawdown", 0.0)),
                float(item.get("profit_factor") or 0.0),
            ),
            reverse=True,
        )
    if key == "profit_factor":
        return sorted(
            results,
            key=lambda item: (
                float(item.get("profit_factor") or 0.0),
                float(item.get("equity", 0.0)),
                -float(item.get("max_drawdown", 0.0)),
            ),
            reverse=True,
        )
    return sorted(
        results,
        key=lambda item: (
            float(item.get("pnl_to_max_dd", 0.0)),
            float(item.get("equity", 0.0)),
            float(item.get("profit_factor") or 0.0),
        ),
        reverse=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a lightweight RegimeAdaptive filterless backtest/sweep on the cleaned outright parquet."
    )
    parser.add_argument("--source", default="es_master_outrights.parquet")
    parser.add_argument("--symbol-mode", default="auto_by_day")
    parser.add_argument("--symbol-method", default="volume")
    parser.add_argument("--start", default="2011-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--sma-fast-values", default="")
    parser.add_argument("--sma-slow-values", default="")
    parser.add_argument("--cross-atr-mults", default="")
    parser.add_argument("--reversion-values", default="")
    parser.add_argument("--contracts", type=int, default=10)
    parser.add_argument("--rank-by", choices=["pnl_to_dd", "pnl", "profit_factor"], default="pnl_to_dd")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    tuning = CONFIG.get("REGIME_ADAPTIVE_TUNING", {}) or {}
    sma_fast_values = _parse_int_list(args.sma_fast_values) or [int(tuning.get("sma_fast", 20) or 20)]
    sma_slow_values = _parse_int_list(args.sma_slow_values) or [int(tuning.get("sma_slow", 200) or 200)]
    cross_atr_mults = _parse_float_list(args.cross_atr_mults) or [float(tuning.get("cross_atr_mult", 0.1) or 0.1)]
    reversion_values = _parse_reversion_list(args.reversion_values) or [bool(tuning.get("enable_signal_reversion", True))]

    start_time = _parse_datetime(args.start, is_end=False)
    end_time = _parse_datetime(args.end, is_end=True)
    source = Path(args.source).expanduser().resolve()
    if not source.is_file():
        raise SystemExit(f"Source parquet not found: {source}")

    df, symbol_label = _load_bars(source, args.symbol_mode, args.symbol_method)
    combo_ids, session_codes = _build_combo_arrays(df.index)
    holiday_mask = _build_holiday_mask(df.index, session_codes)
    revert_lookup, long_sl_lookup, long_tp_lookup, short_sl_lookup, short_tp_lookup = _build_regime_lookup()

    close = df["close"].to_numpy(dtype=np.float64)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    atr = _atr_array(high, low, close, int(tuning.get("atr_period", 20) or 20))
    rolling_means = _rolling_cache(close, list(sma_fast_values) + list(sma_slow_values))

    point_value = float((CONFIG.get("RISK", {}) or {}).get("POINT_VALUE", 5.0) or 5.0)
    fee_per_side = float((CONFIG.get("RISK", {}) or {}).get("FEES_PER_SIDE", 0.37) or 0.37)
    fee_per_contract_rt = fee_per_side * 2.0

    param_grid: list[StrategyParams] = []
    for sma_fast in sma_fast_values:
        for sma_slow in sma_slow_values:
            if sma_fast <= 0 or sma_slow <= 0 or sma_fast >= sma_slow:
                continue
            for cross_mult in cross_atr_mults:
                for use_reversion in reversion_values:
                    param_grid.append(
                        StrategyParams(
                            sma_fast=int(sma_fast),
                            sma_slow=int(sma_slow),
                            cross_atr_mult=float(cross_mult),
                            use_reversion=bool(use_reversion),
                        )
                    )
    if not param_grid:
        raise SystemExit("No valid parameter combinations were produced.")

    results: list[dict] = []
    for params in param_grid:
        signal_side, signal_sl, signal_tp, original_side = _build_signal_package(
            combo_ids,
            session_codes,
            close,
            rolling_means[params.sma_fast],
            rolling_means[params.sma_slow],
            atr,
            params,
            revert_lookup,
            long_sl_lookup,
            long_tp_lookup,
            short_sl_lookup,
            short_tp_lookup,
        )
        result = _simulate(
            df,
            start_time,
            end_time,
            combo_ids,
            session_codes,
            holiday_mask,
            signal_side,
            signal_sl,
            signal_tp,
            original_side,
            params,
            contracts=int(args.contracts),
            point_value=point_value,
            fee_per_contract_rt=fee_per_contract_rt,
        )
        results.append(result)
        print(
            f"{params} -> equity={result['equity']:.2f} trades={result['trades']} "
            f"winrate={result['winrate']:.2f}% max_dd={result['max_drawdown']:.2f} "
            f"pf={result['profit_factor']}"
        )

    ranked = _rank_results(results, args.rank_by)
    out_path = Path(args.out).expanduser() if str(args.out or "").strip() else (
        Path("reports") / f"regimeadaptive_filterless_sweep_{dt.datetime.now(NY_TZ).strftime('%Y%m%d_%H%M%S')}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": dt.datetime.now(NY_TZ).isoformat(),
        "source": str(source),
        "symbol": symbol_label,
        "range_start": start_time.isoformat(),
        "range_end": end_time.isoformat(),
        "bars": int(((df.index >= start_time) & (df.index <= end_time)).sum()),
        "parameter_grid": [asdict(item) for item in param_grid],
        "ranking": args.rank_by,
        "best": ranked[0],
        "top_results": ranked[: max(1, int(args.top_k))],
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
