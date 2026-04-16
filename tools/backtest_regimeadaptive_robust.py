import argparse
import csv
import datetime as dt
import json
import math
import os
import platform
import sys
from pathlib import Path

if sys.platform.startswith("win"):
    _platform_machine = str(os.environ.get("PROCESSOR_ARCHITECTURE", "") or "").strip()
    if _platform_machine:
        platform.machine = lambda: _platform_machine  # type: ignore[assignment]

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import CONFIG
from regimeadaptive_artifact import load_regimeadaptive_artifact
from regimeadaptive_gate import (
    build_gate_feature_frame_for_positions,
    load_regimeadaptive_gate_model,
)
from tools.regimeadaptive_filterless_runner import (
    COMBO_SPACE,
    NY_TZ,
    SESSION_NAMES,
    _atr_array,
    _build_combo_arrays,
    _build_holiday_mask,
    _combo_key_from_id,
    _contracts_for_drawdown,
    _load_bars,
    _resolve_sl_tp_conflict,
    _rolling_cache,
    _round_points_to_tick,
)


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


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(sub) for key, sub in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _parse_session_threshold_overrides(raw_items: list[str] | None) -> dict[str, float]:
    overrides: dict[str, float] = {}
    valid_sessions = {str(session).upper() for session in SESSION_NAMES}
    for raw_item in raw_items or []:
        text = str(raw_item or "").strip()
        if not text:
            continue
        session_text, separator, threshold_text = text.partition("=")
        if not separator:
            raise SystemExit(
                f"Invalid --gate-threshold-session-override value '{text}'. Use SESSION=THRESHOLD."
            )
        session_name = str(session_text or "").strip().upper()
        if session_name not in valid_sessions:
            raise SystemExit(
                f"Unknown RegimeAdaptive session '{session_name}' in --gate-threshold-session-override."
            )
        try:
            threshold = float(threshold_text)
        except Exception as exc:
            raise SystemExit(
                f"Invalid threshold '{threshold_text}' for session override '{text}'."
            ) from exc
        if not math.isfinite(threshold):
            raise SystemExit(f"Non-finite threshold '{threshold_text}' for session override '{text}'.")
        overrides[session_name] = float(threshold)
    return overrides


def _robustness_metrics(trade_log: list[dict], start_time: pd.Timestamp, end_time: pd.Timestamp) -> dict:
    if not trade_log:
        return {
            "daily_sharpe": 0.0,
            "negative_years": 0,
            "worst_year_pnl": 0.0,
            "best_year_pnl": 0.0,
            "yearly_pnl_std": 0.0,
            "worst_3y_pnl": 0.0,
            "worst_5y_pnl": 0.0,
            "yearly_pnl": {},
        }

    trades = pd.DataFrame(trade_log)
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True).dt.tz_convert(NY_TZ)
    trades["exit_time"] = pd.to_datetime(trades["exit_time"], utc=True).dt.tz_convert(NY_TZ)
    trades["pnl_net"] = trades["pnl_net"].astype(float)
    trades["year"] = trades["entry_time"].dt.year.astype(int)

    yearly = trades.groupby("year")["pnl_net"].sum().sort_index()
    day_index = pd.date_range(start_time.tz_convert(NY_TZ).normalize(), end_time.tz_convert(NY_TZ).normalize(), freq="D")
    daily = (
        trades.groupby(trades["exit_time"].dt.normalize())["pnl_net"]
        .sum()
        .sort_index()
        .reindex(day_index, fill_value=0.0)
    )
    daily_mean = float(daily.mean())
    daily_std = float(daily.std(ddof=0))
    daily_sharpe = (daily_mean / daily_std) * math.sqrt(252.0) if daily_std > 0.0 else 0.0
    roll3 = yearly.rolling(3).sum().dropna()
    roll5 = yearly.rolling(5).sum().dropna()
    return {
        "daily_sharpe": float(round(daily_sharpe, 4)),
        "negative_years": int((yearly < 0).sum()),
        "worst_year_pnl": float(round(float(yearly.min()), 2)) if not yearly.empty else 0.0,
        "best_year_pnl": float(round(float(yearly.max()), 2)) if not yearly.empty else 0.0,
        "yearly_pnl_std": float(round(float(yearly.std(ddof=0)) if len(yearly) > 1 else 0.0, 2)),
        "worst_3y_pnl": float(round(float(roll3.min()) if not roll3.empty else float(yearly.sum()), 2)),
        "worst_5y_pnl": float(round(float(roll5.min()) if not roll5.empty else float(yearly.sum()), 2)),
        "yearly_pnl": {str(int(year)): float(round(float(value), 2)) for year, value in yearly.items()},
    }


def _artifact_rule_order(artifact) -> list[str]:
    rule_catalog = getattr(artifact, "rule_catalog", {}) or {}
    if not rule_catalog:
        return ["__base__"]
    ordered = sorted(str(rule_id) for rule_id in rule_catalog.keys())
    default_rule_id = getattr(artifact, "default_rule_id", None)
    if default_rule_id in ordered:
        ordered.remove(default_rule_id)
        ordered.insert(0, str(default_rule_id))
    return ordered


def _rolling_extrema_cache(values: np.ndarray, windows: list[int] | set[int], mode: str) -> dict[int, np.ndarray]:
    extrema: dict[int, np.ndarray] = {}
    series = pd.Series(values, copy=False)
    reducer = "max" if str(mode).lower() == "max" else "min"
    for window in sorted({max(1, int(raw or 1)) for raw in windows}):
        rolled = getattr(series.rolling(window, min_periods=window), reducer)().shift(1)
        extrema[window] = rolled.to_numpy(dtype=np.float64)
    return extrema


def _rule_type(rule_payload: dict) -> str:
    return str(rule_payload.get("rule_type", "pullback") or "pullback").strip().lower()


def _build_artifact_rule_lookup(artifact, rule_order: list[str]) -> np.ndarray:
    rule_index = {str(rule_id): idx for idx, rule_id in enumerate(rule_order)}
    lookup = np.full((COMBO_SPACE, 2), -1, dtype=np.int16)
    for combo_idx in range(COMBO_SPACE):
        combo_key = _combo_key_from_id(combo_idx)
        for side_idx, original_side in enumerate(("LONG", "SHORT")):
            if rule_order == ["__base__"]:
                lookup[combo_idx, side_idx] = 0
                continue
            rule_id = artifact.get_rule_id(combo_key, original_side=original_side)
            if rule_id in rule_index:
                lookup[combo_idx, side_idx] = int(rule_index[rule_id])
    return lookup


def _build_artifact_lookups(artifact) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    policy_lookup = np.zeros((COMBO_SPACE, 2), dtype=np.int8)
    early_exit_lookup = np.full((COMBO_SPACE, 2), -1, dtype=np.int8)
    min_gate_threshold_lookup = np.full((COMBO_SPACE, 2), np.nan, dtype=np.float32)
    long_sl = np.full(COMBO_SPACE, 2.0, dtype=np.float32)
    long_tp = np.full(COMBO_SPACE, 3.0, dtype=np.float32)
    short_sl = np.full(COMBO_SPACE, 2.0, dtype=np.float32)
    short_tp = np.full(COMBO_SPACE, 3.0, dtype=np.float32)

    for combo_idx in range(COMBO_SPACE):
        combo_key = _combo_key_from_id(combo_idx)
        session_name = combo_key.rsplit("_", 1)[-1]
        for side_idx, original_side in enumerate(("LONG", "SHORT")):
            policy = artifact.combo_policy(combo_key, original_side=original_side)
            if policy == "normal":
                policy_lookup[combo_idx, side_idx] = 1
            elif policy == "reversed":
                policy_lookup[combo_idx, side_idx] = -1
            early_exit_enabled = artifact.get_early_exit_enabled(combo_key, original_side=original_side)
            if early_exit_enabled is True:
                early_exit_lookup[combo_idx, side_idx] = 1
            elif early_exit_enabled is False:
                early_exit_lookup[combo_idx, side_idx] = 0
            min_gate_threshold = artifact.get_min_gate_threshold(combo_key, original_side=original_side)
            if min_gate_threshold is not None:
                min_gate_threshold_lookup[combo_idx, side_idx] = float(min_gate_threshold)
        long_sltp = artifact.get_sltp("LONG", combo_key, session_name)
        short_sltp = artifact.get_sltp("SHORT", combo_key, session_name)
        long_sl[combo_idx] = _round_points_to_tick(float(long_sltp.get("sl_dist", 2.0) or 2.0))
        long_tp[combo_idx] = _round_points_to_tick(float(long_sltp.get("tp_dist", 3.0) or 3.0))
        short_sl[combo_idx] = _round_points_to_tick(float(short_sltp.get("sl_dist", 2.0) or 2.0))
        short_tp[combo_idx] = _round_points_to_tick(float(short_sltp.get("tp_dist", 3.0) or 3.0))
    return policy_lookup, early_exit_lookup, min_gate_threshold_lookup, long_sl, long_tp, short_sl, short_tp


def _build_rule_strength_arrays(
    session_codes: np.ndarray,
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    rolling_cache: dict[int, np.ndarray],
    atr_cache: dict[int, np.ndarray],
    rolling_high_cache: dict[int, np.ndarray],
    rolling_low_cache: dict[int, np.ndarray],
    rule_payload: dict,
) -> tuple[np.ndarray, np.ndarray]:
    sma_fast = int(rule_payload.get("sma_fast", 20) or 20)
    sma_slow = int(rule_payload.get("sma_slow", 200) or 200)
    atr_period = int(rule_payload.get("atr_period", 20) or 20)
    cross_atr_mult = float(rule_payload.get("cross_atr_mult", 0.0) or 0.0)
    pattern_lookback = max(1, int(rule_payload.get("pattern_lookback", 8) or 8))
    touch_atr_mult = max(0.0, float(rule_payload.get("touch_atr_mult", 0.25) or 0.25))
    rule_type = _rule_type(rule_payload)

    sma_fast_arr = rolling_cache[sma_fast]
    sma_slow_arr = rolling_cache[sma_slow]
    atr = atr_cache[atr_period]
    valid = (
        np.isfinite(sma_fast_arr)
        & np.isfinite(sma_slow_arr)
        & np.isfinite(atr)
        & (session_codes != (len(SESSION_NAMES) - 1))
    )
    cross_thresh = atr * float(cross_atr_mult)
    long_strength = np.zeros(len(close), dtype=np.float32)
    short_strength = np.zeros(len(close), dtype=np.float32)
    trending_up = sma_fast_arr > sma_slow_arr
    trending_down = sma_fast_arr < sma_slow_arr

    if rule_type == "breakout":
        recent_high = rolling_high_cache.get(pattern_lookback)
        recent_low = rolling_low_cache.get(pattern_lookback)
        if recent_high is None:
            recent_high = _rolling_extrema_cache(high, {pattern_lookback}, "max")[pattern_lookback]
        if recent_low is None:
            recent_low = _rolling_extrema_cache(low, {pattern_lookback}, "min")[pattern_lookback]
        long_mask = valid & trending_up & np.isfinite(recent_high) & (close > (recent_high + cross_thresh))
        short_mask = valid & trending_down & np.isfinite(recent_low) & (close < (recent_low - cross_thresh))
        long_strength[long_mask] = np.asarray(
            (close[long_mask] - recent_high[long_mask] - cross_thresh[long_mask]),
            dtype=np.float32,
        )
        short_strength[short_mask] = np.asarray(
            (recent_low[short_mask] - close[short_mask] - cross_thresh[short_mask]),
            dtype=np.float32,
        )
        return long_strength, short_strength

    if rule_type == "continuation":
        recent_low = rolling_low_cache.get(pattern_lookback)
        recent_high = rolling_high_cache.get(pattern_lookback)
        if recent_low is None:
            recent_low = _rolling_extrema_cache(low, {pattern_lookback}, "min")[pattern_lookback]
        if recent_high is None:
            recent_high = _rolling_extrema_cache(high, {pattern_lookback}, "max")[pattern_lookback]
        touch_buffer = atr * float(touch_atr_mult)
        long_touch = np.isfinite(recent_low) & (recent_low <= (sma_fast_arr + touch_buffer))
        short_touch = np.isfinite(recent_high) & (recent_high >= (sma_fast_arr - touch_buffer))
        long_mask = valid & trending_up & long_touch & (close > (sma_fast_arr + cross_thresh))
        short_mask = valid & trending_down & short_touch & (close < (sma_fast_arr - cross_thresh))
        long_strength[long_mask] = np.asarray(
            (close[long_mask] - sma_fast_arr[long_mask] - cross_thresh[long_mask]),
            dtype=np.float32,
        )
        short_strength[short_mask] = np.asarray(
            (sma_fast_arr[short_mask] - close[short_mask] - cross_thresh[short_mask]),
            dtype=np.float32,
        )
        return long_strength, short_strength

    long_mask = valid & trending_up & (close < (sma_fast_arr - cross_thresh))
    short_mask = valid & trending_down & (close > (sma_fast_arr + cross_thresh))
    long_strength[long_mask] = np.asarray(
        (sma_fast_arr[long_mask] - close[long_mask] - cross_thresh[long_mask]),
        dtype=np.float32,
    )
    short_strength[short_mask] = np.asarray(
        (close[short_mask] - sma_fast_arr[short_mask] - cross_thresh[short_mask]),
        dtype=np.float32,
    )
    return long_strength, short_strength


def _build_signal_arrays(
    combo_ids: np.ndarray,
    long_strength: np.ndarray,
    short_strength: np.ndarray,
    policy_lookup: np.ndarray,
    early_exit_lookup: np.ndarray,
    long_sl_lookup: np.ndarray,
    long_tp_lookup: np.ndarray,
    short_sl_lookup: np.ndarray,
    short_tp_lookup: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(long_strength)
    signal_side = np.zeros(n, dtype=np.int8)
    long_mask = long_strength > 0.0
    short_mask = short_strength > 0.0
    signal_side[long_mask] = 1
    signal_side[short_mask] = -1

    original_side = signal_side.copy()
    signal_side_index = np.where(original_side > 0, 0, 1)
    combo_policy = np.zeros(n, dtype=np.int8)
    combo_policy[original_side != 0] = policy_lookup[combo_ids[original_side != 0], signal_side_index[original_side != 0]]
    signal_early_exit = np.full(n, -1, dtype=np.int8)
    signal_early_exit[original_side != 0] = early_exit_lookup[combo_ids[original_side != 0], signal_side_index[original_side != 0]]
    signal_side = np.where(combo_policy == 0, 0, signal_side).astype(np.int8)
    reversed_mask = (combo_policy < 0) & (signal_side != 0)
    signal_side = np.where(reversed_mask, -signal_side, signal_side).astype(np.int8)

    sl = np.zeros(n, dtype=np.float32)
    tp = np.zeros(n, dtype=np.float32)
    long_rows = signal_side > 0
    short_rows = signal_side < 0
    sl[long_rows] = long_sl_lookup[combo_ids[long_rows]]
    tp[long_rows] = long_tp_lookup[combo_ids[long_rows]]
    sl[short_rows] = short_sl_lookup[combo_ids[short_rows]]
    tp[short_rows] = short_tp_lookup[combo_ids[short_rows]]
    return signal_side, signal_early_exit, sl, tp, original_side


def _build_multirule_signal_arrays(
    combo_ids: np.ndarray,
    session_codes: np.ndarray,
    rule_lookup: np.ndarray,
    policy_lookup: np.ndarray,
    early_exit_lookup: np.ndarray,
    long_sl_lookup: np.ndarray,
    long_tp_lookup: np.ndarray,
    short_sl_lookup: np.ndarray,
    short_tp_lookup: np.ndarray,
    long_strength_matrix: np.ndarray,
    short_strength_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(combo_ids)
    rows = np.arange(n, dtype=np.int32)
    combo_long_rules = rule_lookup[combo_ids, 0]
    combo_short_rules = rule_lookup[combo_ids, 1]
    combo_long_policy = policy_lookup[combo_ids, 0]
    combo_short_policy = policy_lookup[combo_ids, 1]
    combo_long_ee = early_exit_lookup[combo_ids, 0]
    combo_short_ee = early_exit_lookup[combo_ids, 1]

    long_strength = np.zeros(n, dtype=np.float32)
    short_strength = np.zeros(n, dtype=np.float32)
    long_rule_mask = combo_long_rules >= 0
    short_rule_mask = combo_short_rules >= 0
    if np.any(long_rule_mask):
        long_strength[long_rule_mask] = long_strength_matrix[combo_long_rules[long_rule_mask], rows[long_rule_mask]]
    if np.any(short_rule_mask):
        short_strength[short_rule_mask] = short_strength_matrix[combo_short_rules[short_rule_mask], rows[short_rule_mask]]

    long_active = (combo_long_policy != 0) & (long_strength > 0.0)
    short_active = (combo_short_policy != 0) & (short_strength > 0.0)
    long_final = np.where(combo_long_policy > 0, 1, -1).astype(np.int8)
    short_final = np.where(combo_short_policy > 0, -1, 1).astype(np.int8)

    signal_side = np.zeros(n, dtype=np.int8)
    signal_early_exit = np.full(n, -1, dtype=np.int8)
    original_side = np.zeros(n, dtype=np.int8)
    selected_rule_index = np.full(n, -1, dtype=np.int16)
    chosen_strength = np.full(n, -1.0, dtype=np.float32)

    if np.any(long_active):
        signal_side[long_active] = long_final[long_active]
        signal_early_exit[long_active] = combo_long_ee[long_active]
        original_side[long_active] = 1
        selected_rule_index[long_active] = combo_long_rules[long_active]
        chosen_strength[long_active] = long_strength[long_active]

    if np.any(short_active):
        better_short = short_active & (short_strength > chosen_strength)
        if np.any(better_short):
            signal_side[better_short] = short_final[better_short]
            signal_early_exit[better_short] = combo_short_ee[better_short]
            original_side[better_short] = -1
            selected_rule_index[better_short] = combo_short_rules[better_short]
            chosen_strength[better_short] = short_strength[better_short]

        equal_conflict = short_active & long_active & np.isclose(short_strength, chosen_strength) & (short_final != signal_side)
        if np.any(equal_conflict):
            signal_side[equal_conflict] = 0
            signal_early_exit[equal_conflict] = -1
            original_side[equal_conflict] = 0
            selected_rule_index[equal_conflict] = -1
            chosen_strength[equal_conflict] = -1.0

    sl = np.zeros(n, dtype=np.float32)
    tp = np.zeros(n, dtype=np.float32)
    long_rows = signal_side > 0
    short_rows = signal_side < 0
    if np.any(long_rows):
        sl[long_rows] = long_sl_lookup[combo_ids[long_rows]]
        tp[long_rows] = long_tp_lookup[combo_ids[long_rows]]
    if np.any(short_rows):
        sl[short_rows] = short_sl_lookup[combo_ids[short_rows]]
        tp[short_rows] = short_tp_lookup[combo_ids[short_rows]]
    return signal_side, signal_early_exit, sl, tp, original_side, selected_rule_index


def _simulate(
    df: pd.DataFrame,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    combo_ids: np.ndarray,
    session_codes: np.ndarray,
    holiday_mask: np.ndarray,
    signal_side: np.ndarray,
    signal_early_exit: np.ndarray,
    signal_sl: np.ndarray,
    signal_tp: np.ndarray,
    original_side: np.ndarray,
    contracts: int,
    point_value: float,
    fee_per_contract_rt: float,
    selected_rule_index: np.ndarray | None = None,
    rule_order: list[str] | None = None,
    gate_prob: np.ndarray | None = None,
    gate_threshold: float | None = None,
    gate_thresholds: np.ndarray | None = None,
    hours: np.ndarray | None = None,
    minutes: np.ndarray | None = None,
    test_positions: np.ndarray | None = None,
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
    default_early_exit_enabled = bool(early_exit_cfg.get("enabled", False))
    early_exit_bars = int(early_exit_cfg.get("exit_if_not_green_by", 30) or 30)
    early_exit_crosses = int(early_exit_cfg.get("max_profit_crosses", 4) or 4)
    opposite_signal_threshold = 3

    index = df.index
    opens = df["open"].to_numpy(dtype=np.float64)
    highs = df["high"].to_numpy(dtype=np.float64)
    lows = df["low"].to_numpy(dtype=np.float64)
    closes = df["close"].to_numpy(dtype=np.float64)
    if hours is None:
        hours = np.fromiter((ts.hour for ts in index), dtype=np.int8, count=len(index))
    if minutes is None:
        minutes = np.fromiter((ts.minute for ts in index), dtype=np.int8, count=len(index))
    if test_positions is None:
        in_range = np.asarray((index >= start_time) & (index <= end_time), dtype=bool)
        test_positions = np.flatnonzero(in_range)
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
    contract_usage_counts: dict[int, int] = {}
    contract_sum = 0
    min_contracts_used = None
    max_contracts_used = None
    trade_log: list[dict] = []

    active = False
    pending_entry = None
    pending_exit = False
    opposite_signal_count = 0
    active_trade: dict = {}

    def _update_excursions(bar_high: float, bar_low: float) -> None:
        if not active:
            return
        entry_price = float(active_trade["entry_price"])
        if int(active_trade["side"]) > 0:
            favorable = max(0.0, float(bar_high) - entry_price)
            adverse = max(0.0, entry_price - float(bar_low))
        else:
            favorable = max(0.0, entry_price - float(bar_low))
            adverse = max(0.0, float(bar_high) - entry_price)
        active_trade["mfe_points"] = max(float(active_trade.get("mfe_points", 0.0)), favorable)
        active_trade["mae_points"] = max(float(active_trade.get("mae_points", 0.0)), adverse)

    def record_close(exit_price: float, exit_time: pd.Timestamp, reason: str) -> None:
        nonlocal active, active_trade, equity, peak, max_drawdown
        nonlocal gross_profit, gross_loss, wins, losses, trades, pending_exit
        pnl_points = (
            float(exit_price) - float(active_trade["entry_price"])
            if int(active_trade["side"]) > 0
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
        trade_log.append(
            {
                "trade_id": int(trades),
                "entry_time": active_trade["entry_time"].isoformat(),
                "signal_time": active_trade["signal_time"].isoformat() if active_trade.get("signal_time") is not None else None,
                "exit_time": exit_time.isoformat(),
                "side": "LONG" if int(active_trade["side"]) > 0 else "SHORT",
                "entry_price": float(active_trade["entry_price"]),
                "exit_price": float(exit_price),
                "size": int(active_trade["size"]),
                "sl_dist": float(active_trade["sl_dist"]),
                "tp_dist": float(active_trade["tp_dist"]),
                "pnl_points": float(round(pnl_points, 6)),
                "pnl_net": float(round(pnl_net, 2)),
                "mfe_points": float(round(active_trade.get("mfe_points", 0.0), 6)),
                "mae_points": float(round(active_trade.get("mae_points", 0.0), 6)),
                "bars_held": int(active_trade.get("bars_held", 0)),
                "combo_key": str(active_trade["combo_key"]),
                "reverted": bool(active_trade["reverted"]),
                "original_signal": str(active_trade.get("original_signal", "")),
                "rule_id": str(active_trade.get("rule_id", "") or ""),
                "gate_prob": (
                    float(round(float(active_trade.get("gate_prob", np.nan)), 6))
                    if active_trade.get("gate_prob") is not None and np.isfinite(float(active_trade.get("gate_prob", np.nan)))
                    else None
                ),
                "gate_threshold": (
                    float(round(float(active_trade.get("gate_threshold", np.nan)), 6))
                    if active_trade.get("gate_threshold") is not None and np.isfinite(float(active_trade.get("gate_threshold", np.nan)))
                    else None
                ),
                "session": str(active_trade["session"]),
                "exit_reason": str(reason),
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
            _update_excursions(bar_high, bar_low)
            record_close(bar_open, ts, "holiday_flat")
            opposite_signal_count = 0
        if force_flat_now and active:
            _update_excursions(bar_high, bar_low)
            record_close(bar_open, ts, "session_flat")
            opposite_signal_count = 0
        if pending_exit and active:
            _update_excursions(bar_high, bar_low)
            record_close(bar_open, ts, "reverse")

        if pending_entry is not None:
            if holiday_closed_now or entry_window_blocked:
                pending_entry = None
            else:
                size = _contracts_for_drawdown(equity, peak, contracts)
                active_trade = {
                    "side": int(pending_entry["side"]),
                    "entry_price": bar_open,
                    "entry_time": ts,
                    "signal_time": pending_entry.get("signal_time"),
                    "sl_dist": float(pending_entry["sl_dist"]),
                    "tp_dist": float(pending_entry["tp_dist"]),
                    "size": size,
                    "bars_held": 0,
                    "profit_crosses": 0,
                    "was_green": None,
                    "combo_key": pending_entry["combo_key"],
                    "reverted": bool(pending_entry["reverted"]),
                    "session": pending_entry["session"],
                    "mfe_points": 0.0,
                    "mae_points": 0.0,
                    "early_exit_enabled": pending_entry.get("early_exit_enabled"),
                    "original_signal": pending_entry.get("original_signal"),
                    "rule_id": pending_entry.get("rule_id"),
                    "gate_prob": pending_entry.get("gate_prob"),
                    "gate_threshold": pending_entry.get("gate_threshold"),
                }
                contract_usage_counts[size] = contract_usage_counts.get(size, 0) + 1
                contract_sum += size
                min_contracts_used = size if min_contracts_used is None else min(min_contracts_used, size)
                max_contracts_used = size if max_contracts_used is None else max(max_contracts_used, size)
                active = True
                pending_entry = None
                opposite_signal_count = 0

        if active:
            _update_excursions(bar_high, bar_low)
            entry_price = float(active_trade["entry_price"])
            sl_dist = float(active_trade["sl_dist"])
            tp_dist = float(active_trade["tp_dist"])
            if int(active_trade["side"]) > 0:
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

            trade_early_exit_enabled = active_trade.get("early_exit_enabled")
            if trade_early_exit_enabled is None:
                trade_early_exit_enabled = default_early_exit_enabled
            if active and bool(trade_early_exit_enabled):
                active_trade["bars_held"] += 1
                is_green = bar_close > entry_price if int(active_trade["side"]) > 0 else bar_close < entry_price
                was_green = active_trade.get("was_green")
                if was_green is not None and is_green != was_green:
                    active_trade["profit_crosses"] += 1
                active_trade["was_green"] = is_green
                if (
                    (int(active_trade["bars_held"]) >= early_exit_bars and not is_green)
                    or (int(active_trade["profit_crosses"]) > early_exit_crosses)
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
            "reverted": bool(current_signal != int(original_side[i]) and int(original_side[i]) != 0),
            "session": SESSION_NAMES[int(session_codes[i])],
            "signal_time": ts,
            "early_exit_enabled": None if int(signal_early_exit[i]) < 0 else bool(int(signal_early_exit[i])),
            "original_signal": "LONG" if int(original_side[i]) > 0 else "SHORT",
            "rule_id": (
                str(rule_order[int(selected_rule_index[i])])
                if selected_rule_index is not None
                and rule_order is not None
                and 0 <= int(selected_rule_index[i]) < len(rule_order)
                else ""
            ),
        }
        if gate_prob is not None and 0 <= int(i) < len(gate_prob):
            gate_prob_value = float(gate_prob[i])
            if np.isfinite(gate_prob_value):
                sig_payload["gate_prob"] = gate_prob_value
        threshold_value = None
        if gate_thresholds is not None and 0 <= int(i) < len(gate_thresholds):
            candidate_threshold = float(gate_thresholds[i])
            if np.isfinite(candidate_threshold):
                threshold_value = float(candidate_threshold)
        elif gate_threshold is not None and np.isfinite(float(gate_threshold)):
            threshold_value = float(gate_threshold)
        if threshold_value is not None:
            sig_payload["gate_threshold"] = threshold_value
        if not active:
            if pending_entry is None:
                pending_entry = sig_payload
            opposite_signal_count = 0
            continue
        if int(active_trade["side"]) == current_signal:
            opposite_signal_count = 0
            continue
        opposite_signal_count += 1
        if opposite_signal_count >= opposite_signal_threshold:
            pending_exit = True
            if pending_entry is None:
                pending_entry = sig_payload
            opposite_signal_count = 0

    if active:
        final_idx = int(test_positions[-1])
        _update_excursions(float(highs[final_idx]), float(lows[final_idx]))
        record_close(float(closes[final_idx]), index[final_idx], "end_of_range")

    profit_factor = gross_profit / abs(gross_loss) if gross_loss < 0 else float("inf") if gross_profit > 0 else 0.0
    avg_trade = equity / trades if trades else 0.0
    winrate = (wins / trades) * 100.0 if trades else 0.0
    robustness = _robustness_metrics(trade_log, start_time, end_time)
    return {
        "equity": float(round(equity, 2)),
        "trades": int(trades),
        "wins": int(wins),
        "losses": int(losses),
        "winrate": float(round(winrate, 4)),
        "avg_trade_net": float(round(avg_trade, 4)),
        "avg_contracts": float(round((contract_sum / trades), 4)) if trades else 0.0,
        "min_contracts_used": int(min_contracts_used or 0),
        "max_contracts_used": int(max_contracts_used or 0),
        "contract_usage": {str(key): int(value) for key, value in sorted(contract_usage_counts.items())},
        "max_drawdown": float(round(max_drawdown, 2)),
        "gross_profit": float(round(gross_profit, 2)),
        "gross_loss": float(round(gross_loss, 2)),
        "profit_factor": None if not math.isfinite(profit_factor) else float(round(profit_factor, 4)),
        "daily_sharpe": float(robustness["daily_sharpe"]),
        "negative_years": int(robustness["negative_years"]),
        "worst_year_pnl": float(robustness["worst_year_pnl"]),
        "best_year_pnl": float(robustness["best_year_pnl"]),
        "yearly_pnl_std": float(robustness["yearly_pnl_std"]),
        "worst_3y_pnl": float(robustness["worst_3y_pnl"]),
        "worst_5y_pnl": float(robustness["worst_5y_pnl"]),
        "yearly_pnl": dict(robustness["yearly_pnl"]),
        "exit_reasons": exit_reason_counts,
        "sessions": session_counts,
        "trade_log": trade_log,
    }


def _write_converted_csv(csv_path: Path, trade_log: list[dict]) -> None:
    cumulative = 0.0
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "Trade #",
                "Type",
                "Date and time",
                "Signal time",
                "Signal",
                "Price USD",
                "Position size (qty)",
                "Position size (value)",
                "Net P&L USD",
                "MFE points",
                "MAE points",
                "Cumulative P&L USD",
                "Exit time",
                "Exit price USD",
                "PnL points",
                "SL points",
                "TP points",
                "R Multiple",
                "Combo key",
                "Rule ID",
                "Gate Prob",
                "Gate Threshold",
                "Session",
                "Reverted",
                "Exit reason",
            ]
        )
        for idx, trade in enumerate(trade_log, start=1):
            entry_price = float(trade.get("entry_price", 0.0) or 0.0)
            qty = float(trade.get("size", 0.0) or 0.0)
            pnl_net = float(trade.get("pnl_net", 0.0) or 0.0)
            sl_points = float(trade.get("sl_dist", 0.0) or 0.0)
            tp_points = float(trade.get("tp_dist", 0.0) or 0.0)
            pnl_points = float(trade.get("pnl_points", 0.0) or 0.0)
            cumulative += pnl_net
            writer.writerow(
                [
                    idx,
                    "Trade",
                    str(trade.get("entry_time", "") or ""),
                    str(trade.get("signal_time", "") or ""),
                    str(trade.get("side", "") or ""),
                    entry_price,
                    qty,
                    round(entry_price * qty, 4),
                    round(pnl_net, 4),
                    float(trade.get("mfe_points", 0.0) or 0.0),
                    float(trade.get("mae_points", 0.0) or 0.0),
                    round(cumulative, 4),
                    str(trade.get("exit_time", "") or ""),
                    float(trade.get("exit_price", 0.0) or 0.0),
                    pnl_points,
                    sl_points,
                    tp_points,
                    round(pnl_points / sl_points, 6) if sl_points > 0.0 else None,
                    str(trade.get("combo_key", "") or ""),
                    str(trade.get("rule_id", "") or ""),
                    trade.get("gate_prob"),
                    trade.get("gate_threshold"),
                    str(trade.get("session", "") or ""),
                    bool(trade.get("reverted", False)),
                    str(trade.get("exit_reason", "") or ""),
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a direct RegimeAdaptive backtest from the robust artifact and export trades to CSV."
    )
    parser.add_argument("--source", default="es_master_outrights.parquet")
    parser.add_argument("--artifact", default="artifacts/regimeadaptive_robust/latest.json")
    parser.add_argument("--symbol-mode", default="auto_by_day")
    parser.add_argument("--symbol-method", default="volume")
    parser.add_argument("--start", default="2011-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--contracts", type=int, default=10)
    parser.add_argument("--fixed-contracts", action="store_true")
    parser.add_argument(
        "--default-unlisted-policy",
        default="skip",
        choices=["skip", "normal", "reversed"],
        help="Fallback policy for combo/side contexts not explicitly present in the artifact.",
    )
    parser.add_argument("--gate-threshold-override", type=float, default=None)
    parser.add_argument(
        "--gate-threshold-session-override",
        action="append",
        default=[],
        help="Repeatable SESSION=THRESHOLD overrides, e.g. LONDON=0.51",
    )
    parser.add_argument("--out-dir", default="backtest_reports")
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()
    if not source.is_file():
        raise SystemExit(f"Source parquet not found: {source}")
    artifact = load_regimeadaptive_artifact(
        str(args.artifact),
        default_policy_override=str(args.default_unlisted_policy or "skip"),
    )
    if artifact is None:
        raise SystemExit(f"Artifact could not be loaded: {args.artifact}")
    gate_session_threshold_overrides = _parse_session_threshold_overrides(
        list(args.gate_threshold_session_override or [])
    )
    base_rule = artifact.base_rule or {}

    start_time = _parse_datetime(args.start, is_end=False)
    end_time = _parse_datetime(args.end, is_end=True)
    df, symbol_label = _load_bars(source, str(args.symbol_mode), str(args.symbol_method))
    combo_ids, session_codes = _build_combo_arrays(df.index)
    holiday_mask = _build_holiday_mask(df.index, session_codes)

    close = df["close"].to_numpy(dtype=np.float64)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)

    rule_order = _artifact_rule_order(artifact)
    multirule_enabled = bool(getattr(artifact, "rule_catalog", {}) or {})
    if multirule_enabled:
        rule_payloads = {
            str(rule_id): dict(artifact.rule_catalog[str(rule_id)])
            for rule_id in rule_order
        }
        default_rule_id = getattr(artifact, "default_rule_id", None)
        base_rule = (
            dict(rule_payloads.get(str(default_rule_id), {}))
            if default_rule_id in rule_payloads
            else dict(next(iter(rule_payloads.values())))
        )
    else:
        rule_payloads = {"__base__": dict(base_rule)}

    sma_windows = sorted(
        {
            int(rule_payload.get("sma_fast", 20) or 20)
            for rule_payload in rule_payloads.values()
        }
        | {
            int(rule_payload.get("sma_slow", 200) or 200)
            for rule_payload in rule_payloads.values()
        }
    )
    atr_periods = sorted({int(rule_payload.get("atr_period", 20) or 20) for rule_payload in rule_payloads.values()})
    pattern_lookbacks = sorted(
        {
            max(1, int(rule_payload.get("pattern_lookback", 8) or 8))
            for rule_payload in rule_payloads.values()
            if _rule_type(rule_payload) in {"continuation", "breakout"}
        }
    )
    rolling = _rolling_cache(close, sma_windows)
    atr_cache = {period: _atr_array(high, low, close, int(period)) for period in atr_periods}
    rolling_high_cache = _rolling_extrema_cache(high, pattern_lookbacks, "max") if pattern_lookbacks else {}
    rolling_low_cache = _rolling_extrema_cache(low, pattern_lookbacks, "min") if pattern_lookbacks else {}

    (
        policy_lookup,
        early_exit_lookup,
        min_gate_threshold_lookup,
        long_sl_lookup,
        long_tp_lookup,
        short_sl_lookup,
        short_tp_lookup,
    ) = _build_artifact_lookups(artifact)
    selected_rule_index = None
    gate_rule_index = None
    gate_long_strength_matrix = None
    gate_short_strength_matrix = None
    if multirule_enabled:
        rule_lookup = _build_artifact_rule_lookup(artifact, rule_order)
        long_strength_matrix = np.zeros((len(rule_order), len(close)), dtype=np.float32)
        short_strength_matrix = np.zeros((len(rule_order), len(close)), dtype=np.float32)
        for idx, rule_id in enumerate(rule_order):
            rule_payload = rule_payloads[str(rule_id)]
            long_strength_matrix[idx], short_strength_matrix[idx] = _build_rule_strength_arrays(
                session_codes,
                close,
                high,
                low,
                rolling,
                atr_cache,
                rolling_high_cache,
                rolling_low_cache,
                rule_payload,
            )
        signal_side, signal_early_exit, signal_sl, signal_tp, original_side, selected_rule_index = _build_multirule_signal_arrays(
            combo_ids,
            session_codes,
            rule_lookup,
            policy_lookup,
            early_exit_lookup,
            long_sl_lookup,
            long_tp_lookup,
            short_sl_lookup,
            short_tp_lookup,
            long_strength_matrix,
            short_strength_matrix,
        )
        gate_rule_index = selected_rule_index
        gate_long_strength_matrix = long_strength_matrix
        gate_short_strength_matrix = short_strength_matrix
    else:
        base_long_strength, base_short_strength = _build_rule_strength_arrays(
            session_codes,
            close,
            high,
            low,
            rolling,
            atr_cache,
            rolling_high_cache,
            rolling_low_cache,
            base_rule,
        )
        signal_side, signal_early_exit, signal_sl, signal_tp, original_side = _build_signal_arrays(
            combo_ids,
            base_long_strength,
            base_short_strength,
            policy_lookup,
            early_exit_lookup,
            long_sl_lookup,
            long_tp_lookup,
            short_sl_lookup,
            short_tp_lookup,
        )
        gate_rule_index = np.zeros(len(close), dtype=np.int16)
        gate_long_strength_matrix = np.expand_dims(base_long_strength, axis=0)
        gate_short_strength_matrix = np.expand_dims(base_short_strength, axis=0)

    gate_prob = None
    gate_threshold = None
    gate_thresholds = None
    gate_model = load_regimeadaptive_gate_model(
        Path(artifact.path),
        getattr(artifact, "signal_gate", {}),
        threshold_override=args.gate_threshold_override,
        session_threshold_overrides=gate_session_threshold_overrides,
    )
    if gate_model is not None and gate_rule_index is not None and gate_long_strength_matrix is not None and gate_short_strength_matrix is not None:
        signal_positions = np.flatnonzero(signal_side != 0)
        if signal_positions.size:
            gate_features = build_gate_feature_frame_for_positions(
                pd.DatetimeIndex(df.index),
                df["open"].to_numpy(dtype=np.float64),
                high,
                low,
                close,
                combo_ids,
                signal_side,
                original_side,
                gate_rule_index,
                signal_positions,
                rule_order if multirule_enabled else ["__base__"],
                rule_payloads,
                rolling,
                atr_cache,
                gate_long_strength_matrix,
                gate_short_strength_matrix,
            )
            gate_probs = gate_model.predict_proba_frame(gate_features)
            gate_prob = np.full(len(close), np.nan, dtype=np.float32)
            gate_prob[signal_positions] = gate_probs.astype(np.float32)
            gate_threshold = float(gate_model.threshold)
            effective_thresholds = gate_model.thresholds_for_signal_contexts(
                session_codes[signal_positions],
                combo_ids[signal_positions],
                original_side[signal_positions],
            )
            signal_side_index = np.clip(np.abs(original_side[signal_positions]).astype(np.int16) - 1, 0, 1)
            min_gate_thresholds = min_gate_threshold_lookup[combo_ids[signal_positions], signal_side_index]
            finite_min_gate_thresholds = np.isfinite(min_gate_thresholds)
            if np.any(finite_min_gate_thresholds):
                effective_thresholds = effective_thresholds.copy()
                effective_thresholds[finite_min_gate_thresholds] = np.maximum(
                    effective_thresholds[finite_min_gate_thresholds],
                    min_gate_thresholds[finite_min_gate_thresholds].astype(np.float64),
                )
            gate_thresholds = np.full(len(close), np.nan, dtype=np.float32)
            gate_thresholds[signal_positions] = effective_thresholds.astype(np.float32)
            blocked_positions = signal_positions[gate_probs < effective_thresholds]
            if blocked_positions.size:
                signal_side[blocked_positions] = 0
                signal_early_exit[blocked_positions] = -1

    risk_cfg = CONFIG.get("RISK", {}) or {}
    point_value = float(risk_cfg.get("POINT_VALUE", 5.0) or 5.0)
    fee_per_side = float(risk_cfg.get("FEES_PER_SIDE", 0.37) or 0.37)
    fee_per_contract_rt = fee_per_side * 2.0

    exec_cfg = CONFIG.setdefault("BACKTEST_EXECUTION", {})
    prev_drawdown_enabled = exec_cfg.get("drawdown_size_scaling_enabled", True)
    prev_growth_enabled = exec_cfg.get("regimeadaptive_growth_size_scaling_enabled", False)
    if bool(args.fixed_contracts):
        exec_cfg["drawdown_size_scaling_enabled"] = False
        exec_cfg["regimeadaptive_growth_size_scaling_enabled"] = False

    try:
        result = _simulate(
            df,
            start_time,
            end_time,
            combo_ids,
            session_codes,
            holiday_mask,
            signal_side,
            signal_early_exit,
            signal_sl,
            signal_tp,
            original_side,
            int(args.contracts),
            point_value,
            fee_per_contract_rt,
            selected_rule_index=selected_rule_index,
            rule_order=rule_order if multirule_enabled else None,
            gate_prob=gate_prob,
            gate_threshold=gate_threshold,
            gate_thresholds=gate_thresholds,
        )
    finally:
        exec_cfg["drawdown_size_scaling_enabled"] = prev_drawdown_enabled
        exec_cfg["regimeadaptive_growth_size_scaling_enabled"] = prev_growth_enabled

    out_dir = Path(args.out_dir).expanduser()
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now(NY_TZ).strftime("%Y%m%d_%H%M%S")
    start_tag = start_time.strftime("%Y%m%d_%H%M")
    end_tag = end_time.strftime("%Y%m%d_%H%M")
    report_path = out_dir / f"backtest_regimeadaptive_robust_{symbol_label}_{start_tag}_{end_tag}_{timestamp}.json"
    csv_path = out_dir / f"converted_{report_path.stem}.csv"

    payload = {
        "created_at": dt.datetime.now(NY_TZ).isoformat(),
        "strategy": "RegimeAdaptiveRobustArtifact",
        "artifact_path": str(getattr(artifact, "path", args.artifact)),
        "source_data_path": str(source),
        "symbol": symbol_label,
        "range_start": start_time.isoformat(),
        "range_end": end_time.isoformat(),
        "base_rule": {
            "sma_fast": int(base_rule.get("sma_fast", 20) or 20),
            "sma_slow": int(base_rule.get("sma_slow", 200) or 200),
            "atr_period": int(base_rule.get("atr_period", 20) or 20),
            "cross_atr_mult": float(base_rule.get("cross_atr_mult", 0.1) or 0.1),
        },
        "rule_catalog": rule_payloads if multirule_enabled else None,
        "summary": {
            "default_unlisted_policy": str(getattr(artifact, "default_policy", args.default_unlisted_policy)),
            "selected_combo_count": int(np.sum(np.any(policy_lookup != 0, axis=1))),
            "reversed_side_policy_count": int(np.sum(policy_lookup < 0)),
            "skipped_side_policy_count": int(np.sum(policy_lookup == 0)),
            "gate_threshold": float(gate_model.threshold) if gate_model is not None else None,
            "gate_session_thresholds": getattr(gate_model, "session_thresholds", {}) if gate_model is not None else {},
            "gate_policy_thresholds": getattr(gate_model, "policy_thresholds", {}) if gate_model is not None else {},
            "selected_rule_count": int(
                len(
                    {
                        str(trade.get("rule_id", "") or "")
                        for trade in result.get("trade_log", []) or []
                        if str(trade.get("rule_id", "") or "").strip()
                    }
                )
            ),
        },
        "result": result,
        "trade_log": result.get("trade_log", []),
    }
    report_path.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=True), encoding="utf-8")
    _write_converted_csv(csv_path, result.get("trade_log", []) or [])

    print(
        json.dumps(
            {
                "report_path": str(report_path),
                "csv_path": str(csv_path),
                "equity": result.get("equity"),
                "trades": result.get("trades"),
                "wins": result.get("wins"),
                "losses": result.get("losses"),
                "winrate": result.get("winrate"),
                "max_drawdown": result.get("max_drawdown"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
