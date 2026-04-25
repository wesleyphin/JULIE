import math
from typing import Any, Dict, Optional

import pandas as pd


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        raw = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(raw):
        return float(default)
    return float(raw)


def _bucket_ratio(
    value: Any,
    *,
    subdued: float = 0.80,
    elevated: float = 1.15,
    high: float = 1.45,
    extreme: float = 1.95,
) -> str:
    raw = _safe_float(value, float("nan"))
    if not math.isfinite(raw):
        return ""
    if raw >= extreme:
        return "extreme"
    if raw >= high:
        return "high"
    if raw >= elevated:
        return "elevated"
    if raw <= subdued:
        return "subdued"
    return "normal"


def _bucket_signed(
    value: Any,
    *,
    strong_neg: float = -1.60,
    neg: float = -0.50,
    pos: float = 0.50,
    strong_pos: float = 1.60,
) -> str:
    raw = _safe_float(value, float("nan"))
    if not math.isfinite(raw):
        return ""
    if raw <= strong_neg:
        return "strong_neg"
    if raw <= neg:
        return "neg"
    if raw < pos:
        return "balanced"
    if raw < strong_pos:
        return "pos"
    return "strong_pos"


def _bucket_score(
    value: Any,
    *,
    elevated: float = 0.80,
    high: float = 1.60,
    extreme: float = 2.60,
) -> str:
    raw = _safe_float(value, float("nan"))
    if not math.isfinite(raw):
        return ""
    if raw >= extreme:
        return "extreme"
    if raw >= high:
        return "high"
    if raw >= elevated:
        return "elevated"
    return "calm"


def _resolve_session_start(
    *,
    current_ts: pd.Timestamp,
    session_text: str,
) -> pd.Timestamp:
    session_clean = str(session_text or "").strip()
    start_hour = int((int(current_ts.hour) // 3) * 3)
    if "-" in session_clean:
        try:
            start_hour = int(session_clean.split("-", 1)[0])
        except Exception:
            start_hour = int((int(current_ts.hour) // 3) * 3)
    return current_ts.normalize() + pd.Timedelta(hours=int(start_hour))


def _align_ts_to_index(index: pd.Index, ts: pd.Timestamp) -> pd.Timestamp:
    if not isinstance(index, pd.DatetimeIndex):
        return ts
    if index.tz is None and ts.tzinfo is not None:
        return ts.tz_localize(None)
    if index.tz is not None and ts.tzinfo is None:
        return ts.tz_localize(index.tz)
    if index.tz is not None and ts.tzinfo is not None:
        return ts.tz_convert(index.tz)
    return ts


def _resolve_trading_day_start(current_ts: pd.Timestamp) -> pd.Timestamp:
    day_start = current_ts.normalize() + pd.Timedelta(hours=18)
    if int(current_ts.hour) < 18:
        day_start -= pd.Timedelta(days=1)
    return day_start


def _compute_range_and_volume(slice_df: pd.DataFrame) -> tuple[float, float]:
    if slice_df.empty:
        return float("nan"), float("nan")
    high_val = _safe_float(pd.to_numeric(slice_df["high"], errors="coerce").max(), float("nan"))
    low_val = _safe_float(pd.to_numeric(slice_df["low"], errors="coerce").min(), float("nan"))
    range_val = float(high_val - low_val) if math.isfinite(high_val) and math.isfinite(low_val) else float("nan")
    volume_val = float("nan")
    if "volume" in slice_df.columns:
        volume_val = _safe_float(pd.to_numeric(slice_df["volume"], errors="coerce").sum(), float("nan"))
    return range_val, volume_val


def _median(values: list[float]) -> float:
    clean = [float(v) for v in values if math.isfinite(_safe_float(v, float("nan")))]
    if not clean:
        return float("nan")
    return _safe_float(pd.Series(clean, dtype=float).median(), float("nan"))


def _classify_day_context(
    *,
    day_range_progress_ratio: float,
    day_volume_progress_ratio: float,
    gap_ratio: float,
    trend_frac: float,
    body_sign: int,
    first60_share: float,
    elapsed_minutes: int,
    recent_range_ratio: float,
    recent_volume_ratio: float,
    session_range_norm: float,
) -> Dict[str, Any]:
    if (
        math.isfinite(day_range_progress_ratio)
        and day_range_progress_ratio >= 1.70
    ) or (
        math.isfinite(recent_range_ratio)
        and recent_range_ratio >= 1.95
    ) or (
        math.isfinite(session_range_norm)
        and session_range_norm >= 1.85
    ):
        expansion_regime = "shock"
    elif (
        math.isfinite(day_range_progress_ratio)
        and day_range_progress_ratio >= 1.20
    ) or (
        math.isfinite(recent_range_ratio)
        and recent_range_ratio >= 1.20
    ) or (
        math.isfinite(session_range_norm)
        and session_range_norm >= 1.20
    ):
        expansion_regime = "expanded"
    elif (
        math.isfinite(day_range_progress_ratio)
        and day_range_progress_ratio <= 0.80
        and (not math.isfinite(recent_range_ratio) or recent_range_ratio <= 0.95)
    ):
        expansion_regime = "compressed"
    else:
        expansion_regime = "normal"

    if math.isfinite(day_volume_progress_ratio) and day_volume_progress_ratio >= 1.50:
        flow_regime = "heavy_flow"
    elif math.isfinite(recent_volume_ratio) and recent_volume_ratio >= 1.55:
        flow_regime = "heavy_flow"
    elif math.isfinite(day_volume_progress_ratio) and day_volume_progress_ratio <= 0.75:
        flow_regime = "thin_flow"
    else:
        flow_regime = "normal_flow"

    if math.isfinite(gap_ratio) and gap_ratio >= 0.80:
        gap_regime = "large_gap"
    elif math.isfinite(gap_ratio) and gap_ratio >= 0.25:
        gap_regime = "medium_gap"
    else:
        gap_regime = "small_gap"

    if math.isfinite(trend_frac) and trend_frac >= 0.60 and body_sign > 0:
        direction_regime = "trend_up"
    elif math.isfinite(trend_frac) and trend_frac >= 0.60 and body_sign < 0:
        direction_regime = "trend_down"
    elif (math.isfinite(trend_frac) and trend_frac <= 0.18) or body_sign == 0:
        direction_regime = "rotation"
    elif body_sign > 0:
        direction_regime = "grind_up"
    else:
        direction_regime = "grind_down"

    if elapsed_minutes <= 120 and (
        (math.isfinite(day_range_progress_ratio) and day_range_progress_ratio >= 1.05)
        or (math.isfinite(trend_frac) and trend_frac >= 0.42)
        or (math.isfinite(session_range_norm) and session_range_norm >= 1.15)
    ):
        opening_regime = "open_drive"
    elif (
        elapsed_minutes >= 240
        and math.isfinite(day_range_progress_ratio)
        and day_range_progress_ratio >= 1.05
        and math.isfinite(first60_share)
        and first60_share <= 0.30
    ):
        opening_regime = "late_drive"
    else:
        opening_regime = "distributed"

    day_type = f"{expansion_regime}|{direction_regime}|{opening_regime}"
    day_profile = f"{day_type}|{flow_regime}|{gap_regime}"
    return {
        "ctx_day_expansion_regime": str(expansion_regime),
        "ctx_day_direction_regime": str(direction_regime),
        "ctx_day_opening_regime": str(opening_regime),
        "ctx_day_flow_regime": str(flow_regime),
        "ctx_day_gap_regime": str(gap_regime),
        "ctx_day_type": str(day_type),
        "ctx_day_profile": str(day_profile),
    }


def compute_shock_context(
    df: pd.DataFrame,
    *,
    position: int,
    session_text: str,
    price_loc: Optional[float] = None,
    rvol_ratio: Optional[float] = None,
    recent_window: int = 3,
    base_window: int = 60,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "ctx_shock_recent_range_ratio": float("nan"),
        "ctx_shock_recent_volume_ratio": float("nan"),
        "ctx_shock_session_move_norm": float("nan"),
        "ctx_shock_session_range_norm": float("nan"),
        "ctx_shock_score": float("nan"),
        "ctx_shock_recent_range_bucket": "",
        "ctx_shock_recent_volume_bucket": "",
        "ctx_shock_session_move_bucket": "",
        "ctx_shock_session_range_bucket": "",
        "ctx_shock_direction_bucket": "",
        "ctx_shock_score_bucket": "",
        "ctx_day_range_progress_ratio": float("nan"),
        "ctx_day_volume_progress_ratio": float("nan"),
        "ctx_day_gap_ratio": float("nan"),
        "ctx_day_trend_frac": float("nan"),
        "ctx_day_first60_share": float("nan"),
        "ctx_day_expansion_regime": "",
        "ctx_day_direction_regime": "",
        "ctx_day_opening_regime": "",
        "ctx_day_flow_regime": "",
        "ctx_day_gap_regime": "",
        "ctx_day_type": "",
        "ctx_day_profile": "",
    }
    if not isinstance(df, pd.DataFrame) or df.empty:
        return out
    if position < 0 or position >= len(df):
        return out
    required_cols = {"open", "high", "low", "close"}
    if not required_cols.issubset(df.columns):
        return out

    index = df.index
    current_ts = pd.Timestamp(index[int(position)])
    if current_ts.tzinfo is None:
        current_ts = current_ts.tz_localize("America/New_York")
    else:
        current_ts = current_ts.tz_convert("America/New_York")

    recent_start = max(0, int(position) - int(max(1, recent_window)) + 1)
    base_start = max(0, int(position) - int(max(4, base_window)))

    high_recent = pd.to_numeric(df["high"].iloc[recent_start : int(position) + 1], errors="coerce")
    low_recent = pd.to_numeric(df["low"].iloc[recent_start : int(position) + 1], errors="coerce")
    close_now = _safe_float(df["close"].iloc[int(position)], float("nan"))
    if high_recent.empty or low_recent.empty or not math.isfinite(close_now):
        return out

    recent_range = _safe_float(high_recent.max(), float("nan")) - _safe_float(low_recent.min(), float("nan"))
    bar_ranges = (
        pd.to_numeric(df["high"].iloc[base_start:int(position)], errors="coerce")
        - pd.to_numeric(df["low"].iloc[base_start:int(position)], errors="coerce")
    )
    range_base = _safe_float(bar_ranges.replace([float("inf"), float("-inf")], pd.NA).dropna().median(), float("nan"))
    if not math.isfinite(range_base) or range_base <= 1e-9:
        range_base = max(0.25, abs(recent_range) / max(1.0, float(recent_window)))
    recent_range_ratio = _safe_float(recent_range / max(1e-9, range_base * max(1.0, float(recent_window))), float("nan"))

    recent_volume_ratio = float("nan")
    if "volume" in df.columns:
        recent_volume = _safe_float(
            pd.to_numeric(df["volume"].iloc[recent_start : int(position) + 1], errors="coerce").sum(),
            float("nan"),
        )
        volume_base = _safe_float(
            pd.to_numeric(df["volume"].iloc[base_start:int(position)], errors="coerce").dropna().median(),
            float("nan"),
        )
        if math.isfinite(recent_volume) and math.isfinite(volume_base) and volume_base > 1e-9:
            recent_volume_ratio = float(
                recent_volume / max(1e-9, volume_base * max(1.0, float(recent_window)))
            )

    session_start_ts = _resolve_session_start(current_ts=current_ts, session_text=str(session_text or ""))
    session_start_ts = _align_ts_to_index(index, session_start_ts)
    try:
        session_start_pos = int(index.searchsorted(session_start_ts))
    except Exception:
        session_start_pos = max(0, int(position) - 180)
    if session_start_pos > int(position):
        session_start_pos = max(0, int(position) - 180)
    session_slice = df.iloc[session_start_pos : int(position) + 1]
    if session_slice.empty:
        session_slice = df.iloc[max(0, int(position) - 180) : int(position) + 1]
    session_open = _safe_float(session_slice["open"].iloc[0], close_now)
    session_high = _safe_float(pd.to_numeric(session_slice["high"], errors="coerce").max(), close_now)
    session_low = _safe_float(pd.to_numeric(session_slice["low"], errors="coerce").min(), close_now)
    elapsed_bars = max(1, int(len(session_slice)))
    norm_scale = max(1e-9, range_base * max(1.0, math.sqrt(float(elapsed_bars))))
    session_move_norm = float((close_now - session_open) / norm_scale)
    session_range_norm = float((session_high - session_low) / norm_scale)

    price_loc_val = _safe_float(price_loc, float("nan"))
    rvol_val = _safe_float(rvol_ratio, float("nan"))
    shock_score = 0.0
    if math.isfinite(recent_range_ratio):
        shock_score += max(0.0, recent_range_ratio - 1.15)
    if math.isfinite(recent_volume_ratio):
        shock_score += 0.70 * max(0.0, recent_volume_ratio - 1.15)
    shock_score += 0.85 * max(0.0, session_range_norm - 1.10)
    shock_score += 0.95 * max(0.0, abs(session_move_norm) - 0.95)
    if math.isfinite(price_loc_val):
        shock_score += 0.35 * max(0.0, abs(price_loc_val - 0.5) - 0.22) * 4.0
    if math.isfinite(rvol_val):
        shock_score += 0.40 * max(0.0, rvol_val - 1.10)

    trading_day_start_ts = _resolve_trading_day_start(current_ts)
    trading_day_start_ts = _align_ts_to_index(index, trading_day_start_ts)
    try:
        trading_day_start_pos = int(index.searchsorted(trading_day_start_ts))
    except Exception:
        trading_day_start_pos = max(0, int(position) - 1440)
    if trading_day_start_pos > int(position):
        trading_day_start_pos = max(0, int(position) - 1440)
    trading_day_slice = df.iloc[trading_day_start_pos : int(position) + 1]
    if trading_day_slice.empty:
        trading_day_slice = df.iloc[max(0, int(position) - 1440) : int(position) + 1]
    day_open = _safe_float(trading_day_slice["open"].iloc[0], close_now)
    day_high = _safe_float(pd.to_numeric(trading_day_slice["high"], errors="coerce").max(), close_now)
    day_low = _safe_float(pd.to_numeric(trading_day_slice["low"], errors="coerce").min(), close_now)
    day_range = float(day_high - day_low) if math.isfinite(day_high) and math.isfinite(day_low) else float("nan")
    day_body = float(close_now - day_open) if math.isfinite(day_open) and math.isfinite(close_now) else float("nan")
    trend_frac = (
        float(abs(day_body) / max(day_range, 1e-9))
        if math.isfinite(day_body) and math.isfinite(day_range) and day_range > 1e-9
        else float("nan")
    )
    body_sign = 1 if math.isfinite(day_body) and day_body > 0.0 else (-1 if math.isfinite(day_body) and day_body < 0.0 else 0)

    elapsed_delta = current_ts - _resolve_trading_day_start(current_ts)
    elapsed_minutes = max(1, int(elapsed_delta.total_seconds() // 60))
    first60_end_ts = _align_ts_to_index(
        index,
        _resolve_trading_day_start(current_ts) + pd.Timedelta(minutes=60),
    )
    try:
        first60_end_pos = int(index.searchsorted(first60_end_ts, side="right")) - 1
    except Exception:
        first60_end_pos = min(int(position), int(trading_day_start_pos) + 60)
    first60_end_pos = max(int(trading_day_start_pos), min(int(position), int(first60_end_pos)))
    first60_slice = df.iloc[int(trading_day_start_pos) : int(first60_end_pos) + 1]
    first60_range, _first60_volume = _compute_range_and_volume(first60_slice)
    first60_share = (
        float(first60_range / max(day_range, 1e-9))
        if math.isfinite(first60_range) and math.isfinite(day_range) and day_range > 1e-9
        else float("nan")
    )

    prior_progress_ranges: list[float] = []
    prior_progress_volumes: list[float] = []
    prior_full_ranges: list[float] = []
    for day_offset in range(1, 21):
        prev_start_ts = _resolve_trading_day_start(current_ts) - pd.Timedelta(days=int(day_offset))
        prev_start_ts = _align_ts_to_index(index, prev_start_ts)
        prev_end_ts = _align_ts_to_index(index, prev_start_ts + pd.Timedelta(minutes=int(elapsed_minutes)))
        next_start_ts = _align_ts_to_index(index, prev_start_ts + pd.Timedelta(days=1))
        try:
            prev_start_pos = int(index.searchsorted(prev_start_ts))
            prev_end_pos = int(index.searchsorted(prev_end_ts, side="right")) - 1
            next_start_pos = int(index.searchsorted(next_start_ts))
        except Exception:
            continue
        if prev_start_pos >= len(df) or prev_end_pos < prev_start_pos:
            continue
        prev_end_pos = min(prev_end_pos, len(df) - 1)
        prev_progress_slice = df.iloc[prev_start_pos : prev_end_pos + 1]
        prev_progress_range, prev_progress_volume = _compute_range_and_volume(prev_progress_slice)
        if math.isfinite(prev_progress_range):
            prior_progress_ranges.append(float(prev_progress_range))
        if math.isfinite(prev_progress_volume):
            prior_progress_volumes.append(float(prev_progress_volume))
        if next_start_pos <= prev_start_pos:
            continue
        prev_full_slice = df.iloc[prev_start_pos:next_start_pos]
        prev_full_range, _prev_full_volume = _compute_range_and_volume(prev_full_slice)
        if math.isfinite(prev_full_range):
            prior_full_ranges.append(float(prev_full_range))

    prior_progress_range_med = _median(prior_progress_ranges)
    prior_progress_volume_med = _median(prior_progress_volumes)
    prior_full_range_med = _median(prior_full_ranges)
    day_range_progress_ratio = (
        float(day_range / max(prior_progress_range_med, 1e-9))
        if math.isfinite(day_range) and math.isfinite(prior_progress_range_med) and prior_progress_range_med > 1e-9
        else float("nan")
    )
    day_volume_val = _safe_float(
        pd.to_numeric(trading_day_slice["volume"], errors="coerce").sum(),
        float("nan"),
    ) if "volume" in trading_day_slice.columns else float("nan")
    day_volume_progress_ratio = (
        float(day_volume_val / max(prior_progress_volume_med, 1e-9))
        if math.isfinite(day_volume_val) and math.isfinite(prior_progress_volume_med) and prior_progress_volume_med > 1e-9
        else float("nan")
    )
    prev_close = float("nan")
    if int(trading_day_start_pos) > 0:
        prev_close = _safe_float(df["close"].iloc[int(trading_day_start_pos) - 1], float("nan"))
    gap_abs = (
        float(abs(day_open - prev_close))
        if math.isfinite(day_open) and math.isfinite(prev_close)
        else float("nan")
    )
    gap_ratio = (
        float(gap_abs / max(prior_full_range_med, 1e-9))
        if math.isfinite(gap_abs) and math.isfinite(prior_full_range_med) and prior_full_range_med > 1e-9
        else float("nan")
    )
    day_context = _classify_day_context(
        day_range_progress_ratio=day_range_progress_ratio,
        day_volume_progress_ratio=day_volume_progress_ratio,
        gap_ratio=gap_ratio,
        trend_frac=trend_frac,
        body_sign=int(body_sign),
        first60_share=first60_share,
        elapsed_minutes=int(elapsed_minutes),
        recent_range_ratio=recent_range_ratio,
        recent_volume_ratio=recent_volume_ratio,
        session_range_norm=session_range_norm,
    )

    out.update(
        {
            "ctx_shock_recent_range_ratio": float(recent_range_ratio)
            if math.isfinite(recent_range_ratio)
            else float("nan"),
            "ctx_shock_recent_volume_ratio": float(recent_volume_ratio)
            if math.isfinite(recent_volume_ratio)
            else float("nan"),
            "ctx_shock_session_move_norm": float(session_move_norm),
            "ctx_shock_session_range_norm": float(session_range_norm),
            "ctx_shock_score": float(shock_score),
            "ctx_shock_recent_range_bucket": _bucket_ratio(recent_range_ratio),
            "ctx_shock_recent_volume_bucket": _bucket_ratio(recent_volume_ratio),
            "ctx_shock_session_move_bucket": _bucket_signed(session_move_norm),
            "ctx_shock_session_range_bucket": _bucket_ratio(
                session_range_norm,
                subdued=0.70,
                elevated=1.00,
                high=1.35,
                extreme=1.85,
            ),
            "ctx_shock_direction_bucket": _bucket_signed(session_move_norm),
            "ctx_shock_score_bucket": _bucket_score(shock_score),
            "ctx_day_range_progress_ratio": float(day_range_progress_ratio)
            if math.isfinite(day_range_progress_ratio)
            else float("nan"),
            "ctx_day_volume_progress_ratio": float(day_volume_progress_ratio)
            if math.isfinite(day_volume_progress_ratio)
            else float("nan"),
            "ctx_day_gap_ratio": float(gap_ratio) if math.isfinite(gap_ratio) else float("nan"),
            "ctx_day_trend_frac": float(trend_frac) if math.isfinite(trend_frac) else float("nan"),
            "ctx_day_first60_share": float(first60_share) if math.isfinite(first60_share) else float("nan"),
        }
    )
    out.update(day_context)
    return out
