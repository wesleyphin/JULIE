import logging
from typing import Dict, Tuple, Optional

from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

from config import CONFIG
from volatility_filter import volatility_filter, VolRegime
from volume_profile import build_volume_profile


def _round_to_tick(value: float, tick_size: float) -> float:
    if tick_size <= 0:
        return value
    return round(value / tick_size) * tick_size


def _calc_atr(df: pd.DataFrame, window: int) -> Optional[float]:
    if df is None or len(df) < max(window, 2):
        return None
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr = pd.concat(
        [(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(window).mean().iloc[-1]
    return float(atr) if np.isfinite(atr) else None


def _calc_atr_series(df: pd.DataFrame, window: int) -> Optional[pd.Series]:
    if df is None or len(df) < max(window, 2):
        return None
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr = pd.concat(
        [(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(window).mean()
    return atr


def _to_ny_index(df: pd.DataFrame, max_bars: int) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    tail = df if len(df) <= max_bars else df.iloc[-max_bars:]
    idx = tail.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    idx = idx.tz_convert(ZoneInfo("America/New_York"))
    df_ny = tail.copy()
    df_ny.index = idx
    return df_ny


def asia_viability_gate(
    df: pd.DataFrame,
    ts=None,
    session: Optional[str] = None,
    cfg: Optional[Dict] = None,
) -> Tuple[bool, str]:
    gate_cfg = cfg if isinstance(cfg, dict) else CONFIG.get("ASIA_VIABILITY_GATE", {}) or {}
    if not gate_cfg.get("enabled", True):
        return True, "Asia gate disabled"
    if df is None or df.empty:
        return False, "Asia gate: no data"

    if ts is None:
        ts = df.index[-1]
    if session is None:
        session = volatility_filter.get_session(ts.hour)
    if str(session).upper() != "ASIA":
        return True, "Not ASIA"

    min_bars = int(gate_cfg.get("min_bars", 120) or 120)
    if len(df) < min_bars:
        return False, "Asia gate: insufficient bars"

    # Option A: ATR expansion
    atr_fast = int(gate_cfg.get("atr_ratio_fast", 5) or 5)
    atr_slow = int(gate_cfg.get("atr_ratio_slow", 60) or 60)
    ratio_min = float(gate_cfg.get("atr_ratio_min", 1.25) or 1.25)
    atr_fast_val = _calc_atr(df, atr_fast)
    atr_slow_val = _calc_atr(df, atr_slow)
    if atr_fast_val and atr_slow_val and atr_slow_val > 0:
        ratio = atr_fast_val / atr_slow_val
        if ratio >= ratio_min:
            return True, f"Asia gate A: ATR ratio {ratio:.2f} >= {ratio_min:.2f}"

    # Option B: Compression -> Release
    comp_window = int(gate_cfg.get("compression_atr_window", 30) or 30)
    comp_pctl = float(gate_cfg.get("compression_percentile", 20) or 20)
    comp_lookback = int(gate_cfg.get("compression_lookback", 200) or 200)
    release_atr_window = int(gate_cfg.get("compression_release_atr_window", 20) or 20)
    release_mult = float(gate_cfg.get("compression_range_mult", 1.2) or 1.2)
    atr30_series = _calc_atr_series(df, comp_window)
    if atr30_series is not None:
        atr30_series = atr30_series.dropna()
        if not atr30_series.empty:
            tail = atr30_series.iloc[-comp_lookback:] if len(atr30_series) > comp_lookback else atr30_series
            try:
                p20 = float(np.percentile(tail.to_numpy(), comp_pctl))
            except Exception:
                p20 = None
            atr30_val = float(atr30_series.iloc[-1]) if np.isfinite(atr30_series.iloc[-1]) else None
            if p20 is not None and atr30_val is not None and atr30_val < p20:
                atr20_val = _calc_atr(df, release_atr_window)
                bar_range = float(df["high"].iloc[-1] - df["low"].iloc[-1])
                if atr20_val and bar_range >= release_mult * atr20_val:
                    return True, "Asia gate B: compression->release"

    # Option C: Structural interaction
    tol = float(gate_cfg.get("interaction_tol_points", 0.25) or 0.25)
    ny_close_hour = int(gate_cfg.get("ny_close_hour", 16) or 16)
    ny_close_minute = int(gate_cfg.get("ny_close_minute", 0) or 0)
    vp_lookback = int(gate_cfg.get("vp_lookback_bars", 390) or 390)
    vp_value_area = float(gate_cfg.get("vp_value_area_pct", 0.7) or 0.7)
    tick_size = float(gate_cfg.get("tick_size", 0.25) or 0.25)
    max_history = int(gate_cfg.get("max_history_bars", 3000) or 3000)

    df_ny = _to_ny_index(df, max_history)
    if df_ny is None or df_ny.empty:
        return False, "Asia gate: missing NY index"

    ny_ts = ts
    if ny_ts.tzinfo is None:
        ny_ts = ny_ts.replace(tzinfo=ZoneInfo("UTC"))
    ny_ts = ny_ts.astimezone(ZoneInfo("America/New_York"))
    current_date = ny_ts.date()

    bar_high = float(df_ny["high"].iloc[-1])
    bar_low = float(df_ny["low"].iloc[-1])

    def _touch(level: Optional[float]) -> bool:
        if level is None:
            return False
        return (bar_low - tol) <= level <= (bar_high + tol)

    # Prior NY close
    if gate_cfg.get("use_ny_close", True):
        dates = pd.Index(df_ny.index.date).unique().sort_values()
        prev_dates = [d for d in dates if d < current_date]
        if prev_dates:
            prev_date = prev_dates[-1]
            day_mask = df_ny.index.date == prev_date
            day_df = df_ny.loc[day_mask]
            if not day_df.empty:
                target_time = pd.Timestamp(
                    year=prev_date.year,
                    month=prev_date.month,
                    day=prev_date.day,
                    hour=ny_close_hour,
                    minute=ny_close_minute,
                    tz=ZoneInfo("America/New_York"),
                )
                cutoff = day_df.loc[day_df.index <= target_time]
                ny_close = float(cutoff["close"].iloc[-1]) if not cutoff.empty else float(day_df["close"].iloc[-1])
                if _touch(ny_close):
                    return True, "Asia gate C: prior NY close interaction"

            # Prior day value area
            if gate_cfg.get("use_value_area", True) and not day_df.empty:
                vp = build_volume_profile(
                    day_df,
                    lookback=min(len(day_df), vp_lookback),
                    tick_size=tick_size,
                    value_area_pct=vp_value_area,
                )
                if vp:
                    if _touch(vp.get("vah")) or _touch(vp.get("val")) or _touch(vp.get("poc")):
                        return True, "Asia gate C: prior day value area interaction"

    # Asia session sweep
    if gate_cfg.get("use_asia_sweep", True):
        asia_start_hour = int(gate_cfg.get("asia_session_start_hour", 18) or 18)
        asia_end_hour = int(gate_cfg.get("asia_session_end_hour", 3) or 3)
        asia_start = ny_ts.replace(hour=asia_start_hour, minute=0, second=0, microsecond=0)
        if ny_ts.hour < asia_end_hour:
            asia_start = asia_start - pd.Timedelta(days=1)
        asia_df = df_ny.loc[(df_ny.index >= asia_start) & (df_ny.index <= ny_ts)]
        if len(asia_df) >= 2:
            prior_high = float(asia_df["high"].iloc[:-1].max())
            prior_low = float(asia_df["low"].iloc[:-1].min())
            current_high = float(asia_df["high"].iloc[-1])
            current_low = float(asia_df["low"].iloc[-1])
            if current_high >= prior_high + tol or current_low <= prior_low - tol:
                return True, "Asia gate C: Asia session sweep"

    return False, "Asia gate: no viable condition"


def _select_bracket(
    session: str,
    vol_regime: str,
    cfg: Dict,
) -> str:
    session_overrides = cfg.get("session_overrides", {}) or {}
    if session in session_overrides:
        return session_overrides[session]
    regime_map = cfg.get("vol_regime_brackets", {}) or {}
    if vol_regime in regime_map:
        return regime_map[vol_regime]
    return cfg.get("default_bracket", "NORMAL_TREND")


def apply_fixed_sltp(
    signal: Dict,
    df: pd.DataFrame,
    entry_price: float,
    ts=None,
    session: Optional[str] = None,
    vol_regime: Optional[str] = None,
) -> Tuple[bool, Dict]:
    """
    Apply fixed regime-based SL/TP brackets and pre-entry viability filters.

    Returns (ok, details). When ok is False, details["reason"] explains the block.
    """
    cfg = CONFIG.get("FIXED_SLTP_FRAMEWORK", {}) or {}
    if not cfg.get("enabled", False):
        return True, {}

    tick_size = float(cfg.get("tick_size", 0.25))
    min_cfg = CONFIG.get("SLTP_MIN", {}) or {}
    min_sl = float(min_cfg.get("sl", 1.25))
    min_tp = float(min_cfg.get("tp", 1.5))

    if ts is None:
        ts = df.index[-1]

    if session is None:
        session = volatility_filter.get_session(ts.hour)
    if vol_regime is None:
        vol_regime, _, _ = volatility_filter.get_regime(df, ts)

    if str(session).upper() == "ASIA":
        viable, reason = asia_viability_gate(df, ts=ts, session=session)
        if not viable:
            return False, {"reason": reason}

    bracket_name = _select_bracket(session, vol_regime, cfg)
    brackets = cfg.get("brackets", {}) or {}
    bracket = brackets.get(bracket_name, brackets.get(cfg.get("default_bracket", "")))
    if not bracket:
        return False, {"reason": f"FixedSLTP missing bracket: {bracket_name}"}

    sl_dist = float(bracket.get("SL", min_sl))
    tp_dist = float(bracket.get("TP", min_tp))

    sl_dist = max(sl_dist, min_sl)
    tp_dist = max(tp_dist, min_tp)

    sl_dist = _round_to_tick(sl_dist, tick_size)
    tp_dist = _round_to_tick(tp_dist, tick_size)

    viability = cfg.get("viability", {}) or {}
    if viability.get("enabled", True):
        atr_window = int(viability.get("atr_window", 20))
        atr_floor = float(viability.get("atr_floor", 0.7))
        lookback = int(viability.get("lookback_bars", 60))
        room_factor = float(viability.get("room_to_target_factor", 1.0))
        room_min_points = viability.get("room_to_target_min_points")
        room_k = viability.get("room_to_target_k")
        room_atr_window = int(viability.get("room_to_target_atr_window", atr_window) or atr_window)

        overrides = []
        session_overrides = viability.get("session_overrides", {}) or {}
        if session in session_overrides:
            overrides.append(session_overrides[session])
        regime_overrides = viability.get("vol_regime_overrides", {}) or {}
        if vol_regime in regime_overrides:
            overrides.append(regime_overrides[vol_regime])
        runtime_overrides = viability.get("runtime_overrides", {}) or {}
        if session in runtime_overrides:
            overrides.append(runtime_overrides[session])
        combo_key = f"{session}:{vol_regime}"
        if combo_key in runtime_overrides:
            overrides.append(runtime_overrides[combo_key])

        for override in overrides:
            if not isinstance(override, dict):
                continue
            if "atr_floor" in override:
                try:
                    atr_floor = float(override["atr_floor"])
                except Exception:
                    pass
            if "lookback_bars" in override:
                try:
                    lookback = int(override["lookback_bars"])
                except Exception:
                    pass
            if "room_to_target_factor" in override:
                try:
                    room_factor = float(override["room_to_target_factor"])
                except Exception:
                    pass
            if "room_to_target_min_points" in override:
                try:
                    room_min_points = float(override["room_to_target_min_points"])
                except Exception:
                    pass
            if "room_to_target_k" in override:
                try:
                    room_k = float(override["room_to_target_k"])
                except Exception:
                    pass
            if "room_to_target_atr_window" in override:
                try:
                    room_atr_window = int(override["room_to_target_atr_window"])
                except Exception:
                    pass

        atr_floor = max(0.1, min(atr_floor, 5.0))
        lookback = max(10, min(lookback, 500))
        room_factor = max(0.5, min(room_factor, 1.5))
        disable_room = bool(viability.get("disable_room_to_target", False))

        atr = _calc_atr(df, atr_window)
        if atr is not None and atr < atr_floor:
            return False, {"reason": f"FixedSLTP ATR too low ({atr:.2f} < {atr_floor:.2f})"}

        if len(df) >= lookback:
            lookback_df = df.iloc[-lookback:]
            recent_high = float(lookback_df["high"].max())
            recent_low = float(lookback_df["low"].min())
            side = signal.get("side")

            if side == "LONG":
                room_to_low = entry_price - recent_low
                room_to_high = recent_high - entry_price
                if room_to_low < sl_dist:
                    return False, {"reason": f"FixedSLTP invalidation too close ({room_to_low:.2f} < SL {sl_dist:.2f})"}
                required_tp_room = tp_dist * room_factor
                if room_min_points is not None or room_k is not None:
                    min_room = float(room_min_points or 0.0)
                    k_val = float(room_k or 0.0)
                    atr_room = _calc_atr(df, room_atr_window)
                    required_tp_room = min_room
                    if atr_room is not None:
                        required_tp_room = max(required_tp_room, k_val * atr_room)
                if (not disable_room) and room_to_high < required_tp_room:
                    return False, {"reason": f"FixedSLTP target room too small ({room_to_high:.2f} < TP {required_tp_room:.2f})"}
            elif side == "SHORT":
                room_to_high = recent_high - entry_price
                room_to_low = entry_price - recent_low
                if room_to_high < sl_dist:
                    return False, {"reason": f"FixedSLTP invalidation too close ({room_to_high:.2f} < SL {sl_dist:.2f})"}
                required_tp_room = tp_dist * room_factor
                if room_min_points is not None or room_k is not None:
                    min_room = float(room_min_points or 0.0)
                    k_val = float(room_k or 0.0)
                    atr_room = _calc_atr(df, room_atr_window)
                    required_tp_room = min_room
                    if atr_room is not None:
                        required_tp_room = max(required_tp_room, k_val * atr_room)
                if (not disable_room) and room_to_low < required_tp_room:
                    return False, {"reason": f"FixedSLTP target room too small ({room_to_low:.2f} < TP {required_tp_room:.2f})"}

    details = {
        "sl_dist": sl_dist,
        "tp_dist": tp_dist,
        "bracket": bracket_name,
        "vol_regime": vol_regime,
        "session": session,
    }
    return True, details


def log_fixed_sltp(details: Dict, strategy: Optional[str] = None) -> None:
    if not details:
        return
    strategy_label = f"{strategy} | " if strategy else ""
    logging.info(
        f"🎯 FixedSLTP {strategy_label}{details.get('bracket')} "
        f"SL={details.get('sl_dist', 0):.2f} TP={details.get('tp_dist', 0):.2f} "
        f"[{details.get('session')}|{details.get('vol_regime')}]"
    )
