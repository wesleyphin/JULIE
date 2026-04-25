import math

import numpy as np
import pandas as pd

import backtest_mes_et as bt
from config import CONFIG
from tools.backtest_aetherflow_direct import (
    _aetherflow_backtest_execution_cfg,
    _aetherflow_trade_day_key,
    _apply_aetherflow_drawdown_size_cap,
    _normalize_aetherflow_risk_governor,
    _normalize_aetherflow_drawdown_size_scaling,
    _resolve_aetherflow_signal_size,
)


def _session_name(ts: pd.Timestamp) -> str:
    hour = int(ts.hour)
    if hour >= 18 or hour < 3:
        return "ASIA"
    if 3 <= hour < 8:
        return "LONDON"
    if 8 <= hour < 12:
        return "NY_AM"
    if 12 <= hour < 17:
        return "NY_PM"
    return "OFF"


def simulate_same_side_add_ons(
    *,
    df: pd.DataFrame,
    signals: pd.DataFrame,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    use_horizon_time_stop: bool,
    max_same_side_legs: int,
    risk_governor_override: dict | None = None,
) -> dict:
    exec_cfg = _aetherflow_backtest_execution_cfg()
    gap_fills = bool(exec_cfg.get("gap_fills", True))
    no_entry_window = bool(exec_cfg.get("enforce_no_new_entries_window", True))
    no_entry_start = int(exec_cfg.get("no_new_entries_start_hour_et", 16) or 16)
    no_entry_start_minute = int(exec_cfg.get("no_new_entries_start_minute_et", 0) or 0)
    no_entry_end = int(exec_cfg.get("no_new_entries_end_hour_et", 18) or 18)
    no_entry_end_minute = int(exec_cfg.get("no_new_entries_end_minute_et", 0) or 0)
    force_flat_enabled = bool(exec_cfg.get("force_flat_at_time", True))
    force_flat_hour = int(exec_cfg.get("force_flat_hour_et", 16) or 16)
    force_flat_minute = int(exec_cfg.get("force_flat_minute_et", 0) or 0)

    risk_cfg = CONFIG.get("RISK", {}) or {}
    point_value = float(risk_cfg.get("POINT_VALUE", 5.0) or 5.0)
    fees_per_side = float(risk_cfg.get("FEES_PER_SIDE", 2.5) or 2.5)
    fee_per_contract_rt = fees_per_side * 2.0
    tick_size = float(getattr(bt, "TICK_SIZE", 0.25) or 0.25)
    aetherflow_cfg = CONFIG.get("AETHERFLOW_STRATEGY", {}) or {}
    drawdown_size_cfg = _normalize_aetherflow_drawdown_size_scaling()
    risk_governor_cfg = _normalize_aetherflow_risk_governor(
        risk_governor_override if isinstance(risk_governor_override, dict) and risk_governor_override else aetherflow_cfg.get("risk_governor", {})
    )
    no_entry_start_total = (int(no_entry_start) * 60) + int(no_entry_start_minute)
    no_entry_end_total = (int(no_entry_end) * 60) + int(no_entry_end_minute)
    max_same_side_legs = max(1, int(max_same_side_legs))

    index = pd.DatetimeIndex(df.index)
    opens = df["open"].to_numpy(dtype=float)
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)
    hours = index.hour.to_numpy(dtype=np.int16)
    minutes = index.minute.to_numpy(dtype=np.int16)
    in_range = np.asarray((index >= start_time) & (index <= end_time), dtype=bool)
    positions = np.flatnonzero(in_range)
    if positions.size == 0:
        raise RuntimeError("No bars in requested range.")

    holiday_dates = bt._build_closed_holiday_dates_et(index)
    date_arr = np.array(index.date, dtype=object)
    holiday_closed_arr = np.isin(date_arr, np.array(sorted(holiday_dates), dtype=object)) if holiday_dates else np.zeros(len(df), dtype=bool)
    holiday_session_block = np.asarray([_session_name(ts) in {"ASIA", "LONDON", "NY_AM", "NY_PM"} for ts in index], dtype=bool)
    holiday_flat_arr = holiday_closed_arr & holiday_session_block

    ordered = signals.copy()
    conf_series = pd.to_numeric(ordered.get("aetherflow_confidence"), errors="coerce") if "aetherflow_confidence" in ordered.columns else pd.Series(0.0, index=ordered.index)
    if not isinstance(conf_series, pd.Series):
        conf_series = pd.Series(conf_series, index=ordered.index)
    conf_series = conf_series.fillna(0.0)
    selection_series = pd.to_numeric(ordered.get("selection_score"), errors="coerce") if "selection_score" in ordered.columns else pd.Series(np.nan, index=ordered.index)
    if not isinstance(selection_series, pd.Series):
        selection_series = pd.Series(selection_series, index=ordered.index)
    ordered["selection_score"] = selection_series.fillna(conf_series)
    ordered["aetherflow_confidence"] = conf_series
    setup_strength_series = pd.to_numeric(ordered.get("setup_strength"), errors="coerce") if "setup_strength" in ordered.columns else pd.Series(0.0, index=ordered.index)
    if not isinstance(setup_strength_series, pd.Series):
        setup_strength_series = pd.Series(setup_strength_series, index=ordered.index)
    ordered["setup_strength"] = setup_strength_series.fillna(0.0)
    ordered = ordered.sort_values(
        by=["selection_score", "aetherflow_confidence", "setup_strength"],
        ascending=[False, False, False],
        kind="mergesort",
    )
    signal_lookup: dict[int, list[dict]] = {}
    for ts, row in zip(pd.DatetimeIndex(ordered.index), ordered.to_dict("records")):
        signal_lookup.setdefault(int(ts.value), []).append(dict(row))

    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0
    wins = 0
    losses = 0
    trades = 0
    gross_profit = 0.0
    gross_loss = 0.0
    trade_log: list[dict] = []
    exit_reason_counts: dict[str, int] = {}
    session_counts: dict[str, int] = {}
    break_even_armed_trade_count = 0
    break_even_stop_update_count = 0
    early_exit_close_count = 0
    risk_governor_blocked_days: set[str] = set()
    risk_governor_blocked_signals = 0
    risk_governor_state = {
        "trade_day": None,
        "realized_pnl": 0.0,
        "loss_trades": 0,
        "blocked": False,
        "blocked_reason": "",
        "blocked_at": None,
    }
    same_side_addon_entries = 0
    suppressed_due_to_position_limit = 0
    suppressed_due_to_opposite_side_conflict = 0
    max_concurrent_legs = 0
    active_trades: list[dict] = []
    pending_entries: list[dict] = []

    def _coerce_float(value, default: float = 0.0) -> float:
        try:
            out = float(value)
        except Exception:
            return float(default)
        return float(out) if math.isfinite(out) else float(default)

    def _coerce_int(value, default: int = 0) -> int:
        return int(round(_coerce_float(value, float(default))))

    def _coerce_bool(value, default: bool = False) -> bool:
        if value is None:
            return bool(default)
        try:
            if pd.isna(value):
                return bool(default)
        except Exception:
            pass
        if isinstance(value, str):
            text = value.strip().lower()
            if text in {"1", "true", "yes", "on"}:
                return True
            if text in {"0", "false", "no", "off", ""}:
                return False
        return bool(value)

    def _ensure_risk_governor_day(ts: pd.Timestamp) -> None:
        trade_day = _aetherflow_trade_day_key(pd.Timestamp(ts))
        if risk_governor_state["trade_day"] == trade_day:
            return
        risk_governor_state["trade_day"] = trade_day
        risk_governor_state["realized_pnl"] = 0.0
        risk_governor_state["loss_trades"] = 0
        risk_governor_state["blocked"] = False
        risk_governor_state["blocked_reason"] = ""
        risk_governor_state["blocked_at"] = None

    def _risk_governor_blocks_new_entries(ts: pd.Timestamp) -> bool:
        _ensure_risk_governor_day(ts)
        return bool(
            risk_governor_cfg.get("enabled", False)
            and risk_governor_cfg.get("block_new_entries_rest_of_day", True)
            and risk_governor_state.get("blocked", False)
        )

    def _minute_window_blocked(current_hour: int, current_minute: int) -> bool:
        if not no_entry_window:
            return False
        current_total = (int(current_hour) * 60) + int(current_minute)
        if no_entry_start_total == no_entry_end_total:
            return False
        if no_entry_start_total < no_entry_end_total:
            return bool(no_entry_start_total <= current_total < no_entry_end_total)
        return bool(current_total >= no_entry_start_total or current_total < no_entry_end_total)

    def align_stop_price_to_tick(price: float, side_num: int) -> float:
        if not math.isfinite(price) or tick_size <= 0.0:
            return float(price)
        scaled = float(price) / float(tick_size)
        if int(side_num) > 0:
            return round(math.floor(scaled + 1e-9) * float(tick_size), 10)
        return round(math.ceil(scaled - 1e-9) * float(tick_size), 10)

    def current_side() -> int:
        sides = [int(t.get("side_num", 0) or 0) for t in active_trades]
        sides.extend(int(t.get("candidate_side", 0) or 0) for t in pending_entries)
        sides = [side for side in sides if side != 0]
        if not sides:
            return 0
        return int(sides[0]) if len(set(sides)) == 1 else 99

    def current_leg_count() -> int:
        return int(len(active_trades) + len(pending_entries))

    def close_trade(trade: dict, exit_price: float, exit_time: pd.Timestamp, reason: str, exit_bar_index: int) -> None:
        nonlocal equity, peak, max_drawdown, wins, losses, trades, gross_profit, gross_loss, early_exit_close_count
        pnl_points = (
            float(exit_price) - float(trade["entry_price"])
            if int(trade["side_num"]) > 0
            else float(trade["entry_price"]) - float(exit_price)
        )
        fee_paid = float(fee_per_contract_rt) * int(trade["size"])
        pnl_dollars = pnl_points * float(point_value) * int(trade["size"])
        pnl_net = pnl_dollars - fee_paid
        equity += pnl_net
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)
        gross_profit += max(pnl_net, 0.0)
        gross_loss += min(pnl_net, 0.0)
        wins += int(pnl_net > 0.0)
        losses += int(pnl_net <= 0.0)
        trades += 1
        if str(reason) == "early_exit":
            early_exit_close_count += 1
        _ensure_risk_governor_day(exit_time)
        risk_governor_state["realized_pnl"] = float(risk_governor_state.get("realized_pnl", 0.0) or 0.0) + float(pnl_net)
        if pnl_net < 0.0:
            risk_governor_state["loss_trades"] = int(risk_governor_state.get("loss_trades", 0) or 0) + 1
        if (
            bool(risk_governor_cfg.get("enabled", False))
            and not bool(risk_governor_state.get("blocked", False))
            and float(risk_governor_cfg.get("daily_loss_stop_usd", 0.0) or 0.0) > 0.0
            and float(risk_governor_state.get("realized_pnl", 0.0) or 0.0) <= -float(risk_governor_cfg.get("daily_loss_stop_usd", 0.0) or 0.0)
            and int(risk_governor_state.get("loss_trades", 0) or 0) >= int(risk_governor_cfg.get("min_loss_trades", 0) or 0)
        ):
            risk_governor_state["blocked"] = True
            risk_governor_state["blocked_reason"] = f"daily_loss_stop_{int(round(float(risk_governor_cfg.get('daily_loss_stop_usd', 0.0) or 0.0)))}"
            risk_governor_state["blocked_at"] = pd.Timestamp(exit_time)
            risk_governor_blocked_days.add(str(pd.Timestamp(risk_governor_state["trade_day"]).date()))
        exit_reason_counts[reason] = exit_reason_counts.get(reason, 0) + 1
        session_counts[trade["session"]] = session_counts.get(trade["session"], 0) + 1
        trade_log.append(
            {
                "trade_id": int(trades),
                "strategy": "AetherFlowStrategy",
                "sub_strategy": None,
                "side": "LONG" if int(trade["side_num"]) > 0 else "SHORT",
                "entry_time": pd.Timestamp(trade["entry_time"]).isoformat(),
                "exit_time": pd.Timestamp(exit_time).isoformat(),
                "entry_price": float(trade["entry_price"]),
                "exit_price": float(exit_price),
                "size": int(trade["size"]),
                "pnl_points": float(round(pnl_points, 6)),
                "pnl_dollars": float(round(pnl_dollars, 2)),
                "pnl_net": float(round(pnl_net, 2)),
                "fee_paid": float(round(fee_paid, 2)),
                "sl_dist": float(trade["sl_dist"]),
                "tp_dist": float(trade["tp_dist"]),
                "mfe_points": float(round(trade.get("mfe_points", 0.0), 6)),
                "mae_points": float(round(trade.get("mae_points", 0.0), 6)),
                "entry_mode": "aetherflow_direct",
                "vol_regime": "BYPASS",
                "exit_reason": str(reason),
                "bars_held": int(trade.get("bars_held", 0)),
                "session": str(trade["session"]),
                "aetherflow_setup_family": str(trade.get("setup_family", "") or ""),
                "aetherflow_confidence": float(trade.get("confidence", 0.0) or 0.0),
                "aetherflow_regime": str(trade.get("regime_name", "") or ""),
                "aetherflow_horizon_bars": int(trade.get("horizon_bars", 0) or 0),
                "aetherflow_use_horizon_time_stop": bool(trade.get("use_horizon_time_stop", False)),
                "aetherflow_entry_mode": str(trade.get("entry_mode", "") or ""),
                "aetherflow_entry_intrabar_fill": bool(trade.get("entry_intrabar_fill", False)),
                "aetherflow_size_multiplier": float(trade.get("size_multiplier", 1.0) or 1.0),
                "drawdown_size_scaling_enabled": bool(trade.get("drawdown_size_scaling_enabled", False)),
                "drawdown_size_source": str(trade.get("drawdown_size_source", "") or ""),
                "drawdown_size_realized_dd_usd": float(trade.get("drawdown_size_realized_dd_usd", 0.0) or 0.0),
                "drawdown_size_progress": float(trade.get("drawdown_size_progress", 0.0) or 0.0),
                "drawdown_size_cap": int(trade.get("drawdown_size_cap", trade.get("size", 0)) or 0),
                "drawdown_size_requested": int(trade.get("drawdown_size_requested", trade.get("requested_size", trade.get("size", 0))) or 0),
                "drawdown_size_applied": bool(trade.get("drawdown_size_applied", False)),
                "drawdown_size_step_usd": float(trade.get("drawdown_size_step_usd", 0.0) or 0.0),
                "aetherflow_break_even_enabled": bool(trade.get("break_even_enabled", False)),
                "aetherflow_break_even_applied": bool(trade.get("break_even_applied", False)),
                "aetherflow_break_even_move_count": int(trade.get("break_even_move_count", 0) or 0),
                "aetherflow_break_even_last_stop_price": (
                    None if trade.get("break_even_last_stop_price") is None else float(trade.get("break_even_last_stop_price"))
                ),
                "aetherflow_early_exit_enabled": bool(trade.get("early_exit_enabled", False)),
                "aetherflow_profit_crosses": int(trade.get("profit_crosses", 0) or 0),
            }
        )
        if trade in active_trades:
            active_trades.remove(trade)

    def update_excursions(trade: dict, bar_high: float, bar_low: float) -> None:
        entry_price = float(trade["entry_price"])
        if int(trade["side_num"]) > 0:
            favorable = max(0.0, float(bar_high) - entry_price)
            adverse = max(0.0, entry_price - float(bar_low))
        else:
            favorable = max(0.0, entry_price - float(bar_low))
            adverse = max(0.0, float(bar_high) - entry_price)
        trade["mfe_points"] = max(float(trade.get("mfe_points", 0.0)), favorable)
        trade["mae_points"] = max(float(trade.get("mae_points", 0.0)), adverse)

    def apply_break_even_stop_update(trade: dict, new_stop_price: float, *, from_pending: bool) -> bool:
        nonlocal break_even_armed_trade_count, break_even_stop_update_count
        side_num = int(trade.get("side_num", 0) or 0)
        current_stop_price = float(trade.get("current_stop_price", math.nan))
        if side_num == 0 or not math.isfinite(current_stop_price):
            return False
        target_stop_price = align_stop_price_to_tick(float(new_stop_price), side_num)
        entry_price = float(trade.get("entry_price", math.nan))
        tp_dist = float(trade.get("tp_dist", math.nan))
        if math.isfinite(entry_price) and math.isfinite(tp_dist) and tp_dist > 0.0 and tick_size > 0.0:
            take_price = entry_price + tp_dist if side_num > 0 else entry_price - tp_dist
            if side_num > 0:
                target_stop_price = min(target_stop_price, align_stop_price_to_tick(take_price - tick_size, side_num))
            else:
                target_stop_price = max(target_stop_price, align_stop_price_to_tick(take_price + tick_size, side_num))
        improved = target_stop_price > current_stop_price + 1e-12 if side_num > 0 else target_stop_price < current_stop_price - 1e-12
        if not improved:
            if from_pending:
                trade["break_even_pending_stop_price"] = None
            return False
        already_applied = bool(trade.get("break_even_applied", False))
        trade["current_stop_price"] = float(target_stop_price)
        trade["break_even_last_stop_price"] = float(target_stop_price)
        trade["break_even_armed"] = True
        trade["break_even_applied"] = True
        trade["break_even_move_count"] = int(trade.get("break_even_move_count", 0) or 0) + 1
        if from_pending:
            trade["break_even_pending_stop_price"] = None
        if not already_applied:
            break_even_armed_trade_count += 1
        break_even_stop_update_count += 1
        return True

    def apply_pending_break_even_stop_update(trade: dict) -> None:
        pending_stop_price = trade.get("break_even_pending_stop_price")
        if pending_stop_price is None:
            return
        try:
            pending_stop_price = float(pending_stop_price)
        except Exception:
            trade["break_even_pending_stop_price"] = None
            return
        if not math.isfinite(pending_stop_price):
            trade["break_even_pending_stop_price"] = None
            return
        apply_break_even_stop_update(trade, pending_stop_price, from_pending=True)

    def stage_break_even_stop_update(trade: dict) -> None:
        if not bool(trade.get("break_even_enabled", False)):
            return
        side_num = int(trade.get("side_num", 0) or 0)
        entry_price = float(trade.get("entry_price", math.nan))
        current_stop_price = float(trade.get("current_stop_price", math.nan))
        tp_dist = float(trade.get("tp_dist", math.nan))
        if side_num == 0 or not math.isfinite(entry_price) or not math.isfinite(current_stop_price) or not math.isfinite(tp_dist) or tp_dist <= 0.0:
            return
        trigger_pct = max(0.0, float(trade.get("break_even_trigger_pct", 0.0) or 0.0))
        trail_pct = max(0.0, float(trade.get("break_even_trail_pct", 0.0) or 0.0))
        buffer_ticks = max(0, int(round(float(trade.get("break_even_buffer_ticks", 0) or 0))))
        activate_on_next_bar = bool(trade.get("break_even_activate_on_next_bar", True))
        mfe_points = max(0.0, float(trade.get("mfe_points", 0.0) or 0.0))
        if (tp_dist * trigger_pct) > 0.0 and mfe_points + 1e-9 < (tp_dist * trigger_pct):
            return
        locked_points = max(float(buffer_ticks) * float(tick_size), mfe_points * trail_pct if trail_pct > 0.0 else 0.0)
        candidate_stop_price = align_stop_price_to_tick(entry_price + locked_points if side_num > 0 else entry_price - locked_points, side_num)
        improved = candidate_stop_price > current_stop_price + 1e-12 if side_num > 0 else candidate_stop_price < current_stop_price - 1e-12
        if not improved:
            return
        trade["break_even_armed"] = True
        trade["break_even_locked_points"] = float(locked_points)
        if activate_on_next_bar:
            pending_stop_price = trade.get("break_even_pending_stop_price")
            try:
                pending_stop_price = None if pending_stop_price is None else float(pending_stop_price)
            except Exception:
                pending_stop_price = None
            better_pending = pending_stop_price is None or (candidate_stop_price > pending_stop_price + 1e-12 if side_num > 0 else candidate_stop_price < pending_stop_price - 1e-12)
            if better_pending:
                trade["break_even_pending_stop_price"] = float(candidate_stop_price)
            return
        apply_break_even_stop_update(trade, candidate_stop_price, from_pending=False)

    def check_early_exit(trade: dict, current_price: float) -> bool:
        if not bool(trade.get("early_exit_enabled", False)):
            return False
        side_num = int(trade.get("side_num", 0) or 0)
        entry_price = float(trade.get("entry_price", math.nan))
        if side_num == 0 or not math.isfinite(entry_price):
            return False
        is_green = current_price > entry_price if side_num > 0 else current_price < entry_price
        was_green = trade.get("was_green")
        if was_green is not None and bool(was_green) != bool(is_green):
            trade["profit_crosses"] = int(trade.get("profit_crosses", 0) or 0) + 1
        trade["was_green"] = bool(is_green)
        exit_if_not_green_by = max(0, int(trade.get("early_exit_exit_if_not_green_by", 0) or 0))
        max_profit_crosses = max(0, int(trade.get("early_exit_max_profit_crosses", 0) or 0))
        if exit_if_not_green_by > 0 and int(trade.get("bars_held", 0) or 0) >= exit_if_not_green_by and not is_green:
            return True
        return int(trade.get("profit_crosses", 0) or 0) > max_profit_crosses

    def activate_trade_from_signal(signal: dict, *, entry_time: pd.Timestamp, entry_bar_index: int, entry_price: float, intrabar_fill: bool) -> None:
        nonlocal same_side_addon_entries, max_concurrent_legs
        side_num = _coerce_int(signal.get("candidate_side", 0), 0)
        if side_num == 0:
            return
        atr14 = max(_coerce_float(signal.get("atr14", 1.0), 1.0), 1e-9)
        setup_sl_mult = _coerce_float(signal.get("setup_sl_mult", 1.0), 1.0)
        setup_tp_mult = _coerce_float(signal.get("setup_tp_mult", 2.0), 2.0)
        sl_mult = _coerce_float(signal.get("sl_mult_override", setup_sl_mult), setup_sl_mult)
        tp_mult = _coerce_float(signal.get("tp_mult_override", setup_tp_mult), setup_tp_mult)
        sl_dist = float(np.clip(sl_mult * atr14, 1.0, 8.0))
        tp_dist = float(np.clip(tp_mult * atr14, max(sl_dist * 1.2, 1.5), 16.0))
        resolved_size, size_multiplier = _resolve_aetherflow_signal_size(signal)
        requested_size = int(resolved_size)
        resolved_size, drawdown_size_diag = _apply_aetherflow_drawdown_size_cap(
            requested_size,
            max(0.0, float(peak) - float(equity)),
            drawdown_size_cfg,
        )
        if active_trades or pending_entries:
            same_side_addon_entries += 1
        active_trades.append(
            {
                "entry_time": entry_time,
                "entry_bar_index": int(entry_bar_index),
                "entry_price": float(entry_price),
                "entry_intrabar_fill": bool(intrabar_fill),
                "side_num": side_num,
                "sl_dist": sl_dist,
                "tp_dist": tp_dist,
                "sl_mult": float(sl_mult),
                "tp_mult": float(tp_mult),
                "size": int(resolved_size),
                "requested_size": int(requested_size),
                "size_multiplier": float(size_multiplier),
                **drawdown_size_diag,
                "bars_held": 0,
                "mfe_points": 0.0,
                "mae_points": 0.0,
                "setup_family": str(signal.get("setup_family", "") or ""),
                "confidence": _coerce_float(signal.get("aetherflow_confidence", 0.0), 0.0),
                "regime_name": str(signal.get("manifold_regime_name", "") or ""),
                "session": _session_name(entry_time),
                "horizon_bars": _coerce_int(signal.get("horizon_bars_override", signal.get("setup_horizon_bars", 0.0)), 0),
                "use_horizon_time_stop": _coerce_bool(signal.get("use_horizon_time_stop", use_horizon_time_stop), bool(use_horizon_time_stop)),
                "entry_mode": str(signal.get("entry_mode", "market_next_bar") or "market_next_bar"),
                "current_stop_price": float(entry_price) - sl_dist if side_num > 0 else float(entry_price) + sl_dist,
                "break_even_enabled": _coerce_bool(signal.get("break_even_enabled", False), False),
                "break_even_activate_on_next_bar": _coerce_bool(signal.get("break_even_activate_on_next_bar", True), True),
                "break_even_trigger_pct": _coerce_float(signal.get("break_even_trigger_pct", 0.0), 0.0),
                "break_even_buffer_ticks": max(0, _coerce_int(signal.get("break_even_buffer_ticks", 0), 0)),
                "break_even_trail_pct": _coerce_float(signal.get("break_even_trail_pct", 0.0), 0.0),
                "break_even_armed": False,
                "break_even_applied": False,
                "break_even_move_count": 0,
                "break_even_last_stop_price": None,
                "break_even_pending_stop_price": None,
                "break_even_locked_points": 0.0,
                "early_exit_enabled": _coerce_bool(signal.get("early_exit_enabled", False), False),
                "early_exit_exit_if_not_green_by": max(0, _coerce_int(signal.get("early_exit_exit_if_not_green_by", 0), 0)),
                "early_exit_max_profit_crosses": max(0, _coerce_int(signal.get("early_exit_max_profit_crosses", 0), 0)),
                "profit_crosses": 0,
                "was_green": None,
            }
        )
        max_concurrent_legs = max(max_concurrent_legs, len(active_trades))

    for i in positions:
        ts = index[i]
        bar_open = float(opens[i])
        bar_high = float(highs[i])
        bar_low = float(lows[i])
        bar_close = float(closes[i])
        holiday_closed_now = bool(holiday_flat_arr[i])
        entry_window_blocked = _minute_window_blocked(hours[i], minutes[i])
        force_flat_now = bool(force_flat_enabled and hours[i] == force_flat_hour and minutes[i] >= force_flat_minute)

        if pending_entries:
            still_pending: list[dict] = []
            for pending_entry in list(pending_entries):
                if holiday_closed_now or entry_window_blocked or _risk_governor_blocks_new_entries(ts):
                    continue
                signal_bar_index = _coerce_int(pending_entry.get("_signal_bar_index", i - 1), i - 1)
                bars_since_signal = max(0, int(i - signal_bar_index))
                entry_mode = str(pending_entry.get("_entry_mode", "market_next_bar") or "market_next_bar").strip().lower()
                if entry_mode in {"market", "market_next_bar", "next_bar_market"}:
                    activate_trade_from_signal(pending_entry, entry_time=ts, entry_bar_index=int(i), entry_price=float(bar_open), intrabar_fill=False)
                    continue
                if entry_mode == "pullback_limit":
                    entry_window_bars = max(1, _coerce_int(pending_entry.get("_entry_window_bars", 2), 2))
                    if bars_since_signal > entry_window_bars:
                        continue
                    side_num = _coerce_int(pending_entry.get("candidate_side", 0), 0)
                    if side_num == 0:
                        continue
                    limit_price = pending_entry.get("_entry_limit_price")
                    if limit_price is None:
                        atr14 = max(_coerce_float(pending_entry.get("atr14", 1.0), 1.0), 1e-9)
                        pullback_atr = max(0.0, _coerce_float(pending_entry.get("entry_pullback_atr", 0.0), 0.0))
                        if pullback_atr <= 0.0:
                            activate_trade_from_signal(pending_entry, entry_time=ts, entry_bar_index=int(i), entry_price=float(bar_open), intrabar_fill=False)
                            continue
                        pullback_points = float(pullback_atr) * float(atr14)
                        raw_limit_price = float(bar_open) - pullback_points if side_num > 0 else float(bar_open) + pullback_points
                        limit_price = align_stop_price_to_tick(raw_limit_price, side_num)
                        pending_entry["_entry_limit_price"] = float(limit_price)
                    fill_price = None
                    intrabar_fill = False
                    if side_num > 0:
                        if gap_fills and bar_open <= float(limit_price):
                            fill_price = float(bar_open)
                        elif bar_low <= float(limit_price):
                            fill_price = float(limit_price)
                            intrabar_fill = True
                    else:
                        if gap_fills and bar_open >= float(limit_price):
                            fill_price = float(bar_open)
                        elif bar_high >= float(limit_price):
                            fill_price = float(limit_price)
                            intrabar_fill = True
                    if fill_price is not None:
                        activate_trade_from_signal(pending_entry, entry_time=ts, entry_bar_index=int(i), entry_price=float(fill_price), intrabar_fill=bool(intrabar_fill))
                        continue
                    if bars_since_signal >= entry_window_bars:
                        continue
                    still_pending.append(pending_entry)
                    continue
                activate_trade_from_signal(pending_entry, entry_time=ts, entry_bar_index=int(i), entry_price=float(bar_open), intrabar_fill=False)
            pending_entries = still_pending

        for trade in list(active_trades):
            if bool(trade.get("entry_intrabar_fill", False)) and int(trade.get("entry_bar_index", -1)) == int(i):
                trade["entry_intrabar_fill"] = False
                continue
            update_excursions(trade, bar_high, bar_low)
            trade["bars_held"] = int(i - trade["entry_bar_index"] + 1)
            apply_pending_break_even_stop_update(trade)
            stage_break_even_stop_update(trade)
            side = "LONG" if int(trade["side_num"]) > 0 else "SHORT"
            if side == "LONG":
                stop_price = float(trade.get("current_stop_price", float(trade["entry_price"]) - float(trade["sl_dist"])))
                take_price = float(trade["entry_price"]) + float(trade["tp_dist"])
                if holiday_closed_now or force_flat_now:
                    close_trade(trade, bar_open, ts, "holiday_flat" if holiday_closed_now else "force_flat", i)
                    continue
                if gap_fills and bar_open <= stop_price:
                    close_trade(trade, bar_open, ts, "stop_gap", i)
                    continue
                if gap_fills and bar_open >= take_price:
                    close_trade(trade, bar_open, ts, "take_gap", i)
                    continue
                hit_stop = bar_low <= stop_price
                hit_take = bar_high >= take_price
            else:
                stop_price = float(trade.get("current_stop_price", float(trade["entry_price"]) + float(trade["sl_dist"])))
                take_price = float(trade["entry_price"]) - float(trade["tp_dist"])
                if holiday_closed_now or force_flat_now:
                    close_trade(trade, bar_open, ts, "holiday_flat" if holiday_closed_now else "force_flat", i)
                    continue
                if gap_fills and bar_open >= stop_price:
                    close_trade(trade, bar_open, ts, "stop_gap", i)
                    continue
                if gap_fills and bar_open <= take_price:
                    close_trade(trade, bar_open, ts, "take_gap", i)
                    continue
                hit_stop = bar_high >= stop_price
                hit_take = bar_low <= take_price
            if hit_stop and hit_take:
                exit_price, exit_reason = bt._resolve_sl_tp_conflict(side, bar_open, bar_close, stop_price, take_price)
                close_trade(trade, float(exit_price), ts, str(exit_reason), i)
                continue
            if hit_stop:
                close_trade(trade, stop_price, ts, "stop", i)
                continue
            if hit_take:
                close_trade(trade, take_price, ts, "take", i)
                continue
            if bool(trade.get("use_horizon_time_stop", use_horizon_time_stop)) and int(trade.get("horizon_bars", 0) or 0) > 0 and int(trade["bars_held"]) >= int(trade["horizon_bars"]):
                close_trade(trade, bar_close, ts, "horizon", i)
                continue
            if check_early_exit(trade, bar_close):
                close_trade(trade, bar_close, ts, "early_exit", i)

        if holiday_closed_now or entry_window_blocked or i >= (len(df) - 1):
            continue
        if _risk_governor_blocks_new_entries(ts):
            blocked_here = signal_lookup.get(int(ts.value), []) or []
            risk_governor_blocked_signals += int(len(blocked_here))
            continue

        for signal in list(signal_lookup.get(int(ts.value), []) or []):
            side_num = _coerce_int(signal.get("candidate_side", 0), 0)
            if side_num == 0:
                continue
            now_side = current_side()
            if now_side not in {0, side_num}:
                suppressed_due_to_opposite_side_conflict += 1
                continue
            if current_leg_count() >= max_same_side_legs:
                suppressed_due_to_position_limit += 1
                continue
            pending_entry = dict(signal)
            entry_mode = str(pending_entry.get("entry_mode", "market_next_bar") or "market_next_bar").strip().lower()
            pending_entry["_signal_bar_index"] = int(i)
            pending_entry["_entry_mode"] = entry_mode
            pending_entry["_entry_window_bars"] = max(1, _coerce_int(pending_entry.get("entry_window_bars", 1 if entry_mode in {"market", "market_next_bar", "next_bar_market"} else 2), 1))
            pending_entry["_entry_limit_price"] = None
            pending_entries.append(pending_entry)

    if active_trades:
        end_ts = index[positions[-1]]
        end_close = float(closes[positions[-1]])
        for trade in list(active_trades):
            close_trade(trade, end_close, end_ts, "end_of_range", int(positions[-1]))

    risk_metrics = bt._compute_backtest_risk_metrics(trade_log)
    summary = {
        "equity": float(round(equity, 2)),
        "trades": int(trades),
        "wins": int(wins),
        "losses": int(losses),
        "winrate": float((wins / trades) * 100.0) if trades > 0 else 0.0,
        "max_drawdown": float(round(max_drawdown, 2)),
        "gross_profit": float(round(gross_profit, 2)),
        "gross_loss": float(round(gross_loss, 2)),
        "trade_log": trade_log,
        "exit_reason_counts": exit_reason_counts,
        "session_counts": session_counts,
        "selection": {"strategies": ["AetherFlowStrategy"], "filters": []},
        "symbol_mode": "direct_aetherflow",
        "fast_mode": {"enabled": True, "bar_stride": 1, "skip_mfe_mae": False},
        "entry_window_block_enabled": bool(no_entry_window),
        "entry_window_block_hours_et": [int(no_entry_start), int(no_entry_end)],
        "force_flat_enabled": bool(force_flat_enabled),
        "force_flat_time_et": f"{force_flat_hour:02d}:{force_flat_minute:02d}",
        "bar_minutes": 1,
        "drawdown_size_scaling_enabled": bool(drawdown_size_cfg.get("enabled", False)),
        "drawdown_size_scaling_source": str(drawdown_size_cfg.get("source", "") or ""),
        "drawdown_size_scaling_start_usd": float(drawdown_size_cfg.get("start_usd", 0.0) or 0.0),
        "drawdown_size_scaling_max_usd": float(drawdown_size_cfg.get("max_usd", 0.0) or 0.0),
        "drawdown_size_scaling_base_contracts": int(drawdown_size_cfg.get("base_contracts", 0) or 0),
        "drawdown_size_scaling_min_contracts": int(drawdown_size_cfg.get("min_contracts", 0) or 0),
        "aetherflow_risk_governor_enabled": bool(risk_governor_cfg.get("enabled", False)),
        "aetherflow_risk_governor_daily_loss_stop_usd": float(risk_governor_cfg.get("daily_loss_stop_usd", 0.0) or 0.0),
        "aetherflow_risk_governor_min_loss_trades": int(risk_governor_cfg.get("min_loss_trades", 0) or 0),
        "aetherflow_risk_governor_blocked_days": int(len(risk_governor_blocked_days)),
        "aetherflow_risk_governor_blocked_signals": int(risk_governor_blocked_signals),
        "break_even_armed_trades": int(break_even_armed_trade_count),
        "break_even_stop_updates": int(break_even_stop_update_count),
        "early_exit_closes": int(early_exit_close_count),
        "same_side_addon_mode_enabled": True,
        "max_same_side_legs": int(max_same_side_legs),
        "max_concurrent_legs": int(max_concurrent_legs),
        "same_side_addon_entries": int(same_side_addon_entries),
        "suppressed_signals_due_to_position_limit": int(suppressed_due_to_position_limit),
        "suppressed_signals_due_to_opposite_side_conflict": int(suppressed_due_to_opposite_side_conflict),
        "report": "",
    }
    summary.update(risk_metrics)
    return summary
