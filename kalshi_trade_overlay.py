import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


DEFAULT_OVERLAY_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "lookback_bars": 20000,
    "lookback_trade_days": 10,
    "min_trade_days": 6,
    "strike_window_size": 120,
    "min_curve_points": 8,
    "min_curve_range": 0.08,
    "min_unique_probabilities": 4,
    "max_target_window_points": 32.0,
    "max_target_window_tp_mult": 3.5,
    "momentum_probe_points": 5.0,
    "fade_absolute_threshold": {
        "background": 0.48,
        "balanced": 0.50,
        "forward_primary": 0.52,
    },
    "fade_adjacent_delta_threshold": {
        "background": 0.20,
        "balanced": 0.18,
        "forward_primary": 0.16,
    },
    "support_probability_floor": {
        "background": 0.46,
        "balanced": 0.52,
        "forward_primary": 0.57,
    },
    "entry_threshold": {
        "background": 0.45,
        "balanced": 0.50,
        "forward_primary": 0.55,
    },
    "momentum_retention_floor": {
        "background": 0.72,
        "balanced": 0.76,
        "forward_primary": 0.80,
    },
    "entry_block_buffer": {
        "background": 1.0,
        "balanced": 0.10,
        "forward_primary": 0.12,
    },
    "entry_size_floor": {
        "background": 0.85,
        "balanced": 0.60,
        "forward_primary": 0.45,
    },
    "forward_weight": {
        "background": 0.20,
        "balanced": 0.48,
        "forward_primary": 0.78,
    },
    "min_tp_multiplier": 0.55,
    "max_tp_multiplier": 1.75,
    "breakout_max_tp_multiplier": 2.5,
    "trail_enabled_roles": ["balanced", "forward_primary"],
    "trail_buffer_ticks": {
        "background": 6,
        "balanced": 4,
        "forward_primary": 4,
    },
    "recent_price_action": {
        "uncertain_mean_day_range": 85.0,
        "outrageous_mean_day_range": 100.0,
        "uncertain_max_day_range": 130.0,
        "outrageous_max_day_range": 150.0,
        "uncertain_flip_rate": 0.39,
        "outrageous_flip_rate": 0.42,
        "uncertain_large_bar_share": 0.14,
        "outrageous_large_bar_share": 0.17,
        "uncertain_mean_true_range": 1.55,
        "outrageous_mean_true_range": 1.85,
        "uncertain_min_score": 3,
        "outrageous_min_score": 5,
        "today_breakout_min_range_points": 70.0,
        "today_breakout_min_net_ratio": 0.60,
        "today_breakout_level_lookback_days": 3,
        "today_breakout_level_tolerance_points": 0.75,
        "today_chop_min_range_points": 30.0,
        "today_chop_min_flip_rate": 0.45,
    },
}


def _coerce_float(value: Any, default: float = math.nan) -> float:
    if value is None:
        return float(default)
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"none", "null", "nan"}:
            return float(default)
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    if math.isfinite(result):
        return float(result)
    return float(default)


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _clamp(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


def _safe_tick_round(value: float, tick_size: float) -> float:
    if not math.isfinite(value):
        return float("nan")
    tick = max(1e-9, float(tick_size))
    return round(round(float(value) / tick) * tick, 10)


def _trade_day_labels(index: pd.Index) -> pd.Index:
    shifted = pd.DatetimeIndex(index) - pd.Timedelta(hours=18)
    return pd.Index(shifted.date)


def _resolve_role_cfg(config: Dict[str, Any], key: str, role: str, default: float) -> float:
    bucket = config.get(key, {}) if isinstance(config.get(key, {}), dict) else {}
    return _coerce_float(bucket.get(role), default)


def _merge_overlay_config(overlay_cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    config = dict(DEFAULT_OVERLAY_CONFIG)
    overlay_cfg = overlay_cfg if isinstance(overlay_cfg, dict) else {}
    for key, value in overlay_cfg.items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            merged = dict(config.get(key, {}))
            merged.update(value)
            config[key] = merged
        else:
            config[key] = value
    return config


def analyze_recent_price_action(
    market_df: Optional[pd.DataFrame],
    overlay_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    config = _merge_overlay_config(overlay_cfg)
    result: Dict[str, Any] = {
        "enabled": bool(config.get("enabled", True)),
        "mode": "level",
        "role": "background",
        "forward_weight": _resolve_role_cfg(config, "forward_weight", "background", 0.20),
        "trade_days_considered": 0,
        "mean_day_range_points": None,
        "max_day_range_points": None,
        "mean_true_range_points": None,
        "mean_flip_rate": None,
        "mean_large_bar_share": None,
        "score": 0,
        "today_range_points": None,
        "today_net_move_points": None,
        "today_net_ratio": None,
        "today_flip_rate": None,
        "today_large_bar_share": None,
        "today_breach_up": False,
        "today_breach_down": False,
        "today_signal": None,
    }
    if not bool(config.get("enabled", True)):
        return result
    if not isinstance(market_df, pd.DataFrame) or market_df.empty:
        return result
    required_cols = {"open", "high", "low", "close"}
    if not required_cols.issubset(set(market_df.columns)):
        return result

    work = market_df.loc[:, ["open", "high", "low", "close"]].copy()
    lookback_bars = max(500, _coerce_int(config.get("lookback_bars"), 20000))
    if len(work) > lookback_bars:
        work = work.tail(lookback_bars)

    for column in ("open", "high", "low", "close"):
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.dropna()
    if work.empty:
        return result

    ret = work["close"].diff().fillna(0.0)
    prev_close = work["close"].shift(1)
    true_range = pd.concat(
        [
            (work["high"] - work["low"]).abs(),
            (work["high"] - prev_close).abs(),
            (work["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1).fillna(0.0)

    sign = np.sign(ret.to_numpy(dtype=float))
    flip = np.zeros(len(sign), dtype=float)
    for idx in range(1, len(sign)):
        if sign[idx] != 0.0 and sign[idx - 1] != 0.0 and sign[idx] != sign[idx - 1]:
            flip[idx] = 1.0

    work["trade_day"] = _trade_day_labels(work.index)
    work["ret"] = ret
    work["abs_ret"] = ret.abs()
    work["true_range"] = true_range
    work["flip"] = flip
    work["large_bar"] = (
        work["abs_ret"]
        >= np.maximum(1.5, work["abs_ret"].rolling(120, min_periods=30).median().fillna(0.5) * 2.5)
    ).astype(float)

    per_day = work.groupby("trade_day").agg(
        open=("open", "first"),
        close=("close", "last"),
        high=("high", "max"),
        low=("low", "min"),
        mean_true_range=("true_range", "mean"),
        flip_rate=("flip", "mean"),
        large_bar_share=("large_bar", "mean"),
    )
    per_day["day_range"] = per_day["high"] - per_day["low"]
    per_day = per_day.dropna()

    lookback_trade_days = max(3, _coerce_int(config.get("lookback_trade_days"), 10))
    if len(per_day) > lookback_trade_days:
        per_day = per_day.tail(lookback_trade_days)
    result["trade_days_considered"] = int(len(per_day))
    if len(per_day) < max(3, _coerce_int(config.get("min_trade_days"), 6)):
        return result

    mean_day_range = float(per_day["day_range"].mean())
    max_day_range = float(per_day["day_range"].max())
    mean_true_range = float(per_day["mean_true_range"].mean())
    mean_flip_rate = float(per_day["flip_rate"].mean())
    mean_large_bar_share = float(per_day["large_bar_share"].mean())

    result.update(
        {
            "mean_day_range_points": round(mean_day_range, 2),
            "max_day_range_points": round(max_day_range, 2),
            "mean_true_range_points": round(mean_true_range, 4),
            "mean_flip_rate": round(mean_flip_rate, 4),
            "mean_large_bar_share": round(mean_large_bar_share, 4),
        }
    )

    thresholds = config.get("recent_price_action", {})
    score = 0
    if mean_day_range >= _coerce_float(thresholds.get("uncertain_mean_day_range"), 85.0):
        score += 1
    if mean_day_range >= _coerce_float(thresholds.get("outrageous_mean_day_range"), 100.0):
        score += 1
    if max_day_range >= _coerce_float(thresholds.get("uncertain_max_day_range"), 130.0):
        score += 1
    if max_day_range >= _coerce_float(thresholds.get("outrageous_max_day_range"), 150.0):
        score += 1
    if mean_flip_rate >= _coerce_float(thresholds.get("uncertain_flip_rate"), 0.39):
        score += 1
    if mean_flip_rate >= _coerce_float(thresholds.get("outrageous_flip_rate"), 0.42):
        score += 1
    if mean_large_bar_share >= _coerce_float(thresholds.get("uncertain_large_bar_share"), 0.14):
        score += 1
    if mean_large_bar_share >= _coerce_float(thresholds.get("outrageous_large_bar_share"), 0.17):
        score += 1
    if mean_true_range >= _coerce_float(thresholds.get("uncertain_mean_true_range"), 1.55):
        score += 1
    if mean_true_range >= _coerce_float(thresholds.get("outrageous_mean_true_range"), 1.85):
        score += 1

    result["score"] = int(score)
    outrageous_min_score = max(1, _coerce_int(thresholds.get("outrageous_min_score"), 5))
    uncertain_min_score = max(1, _coerce_int(thresholds.get("uncertain_min_score"), 3))

    if score >= outrageous_min_score:
        role = "forward_primary"
        mode = "outrageous"
    elif score >= uncertain_min_score:
        role = "balanced"
        mode = "uncertain"
    else:
        role = "background"
        mode = "level"

    today_row = per_day.iloc[-1]
    prior_days = per_day.iloc[:-1]
    today_range = float(today_row["day_range"])
    today_net_move = float(today_row["close"] - today_row["open"])
    today_net_ratio = (
        float(abs(today_net_move) / today_range) if today_range > 1e-9 else 0.0
    )
    today_flip_rate = float(today_row["flip_rate"])
    today_large_bar_share = float(today_row["large_bar_share"])

    level_lookback = max(
        1,
        _coerce_int(thresholds.get("today_breakout_level_lookback_days"), 3),
    )
    level_tolerance = _coerce_float(
        thresholds.get("today_breakout_level_tolerance_points"), 0.75
    )
    breach_up = False
    breach_down = False
    if len(prior_days) >= 1:
        recent_prior = prior_days.tail(level_lookback)
        prior_high = float(recent_prior["high"].max())
        prior_low = float(recent_prior["low"].min())
        breach_up = bool(float(today_row["high"]) > prior_high + level_tolerance)
        breach_down = bool(float(today_row["low"]) < prior_low - level_tolerance)

    breakout_range = _coerce_float(
        thresholds.get("today_breakout_min_range_points"), 70.0
    )
    breakout_ratio = _coerce_float(
        thresholds.get("today_breakout_min_net_ratio"), 0.60
    )
    chop_range = _coerce_float(thresholds.get("today_chop_min_range_points"), 30.0)
    chop_flip = _coerce_float(thresholds.get("today_chop_min_flip_rate"), 0.45)

    today_signal = None
    if (
        today_range >= breakout_range
        and today_net_ratio >= breakout_ratio
        and (breach_up or breach_down)
    ):
        today_signal = "breakout"
        role = "forward_primary"
        mode = "breakout_outrageous"
    elif today_range >= chop_range and today_flip_rate >= chop_flip:
        today_signal = "chop"
        role = "forward_primary"
        mode = "chop_outrageous"

    result.update(
        {
            "today_range_points": round(today_range, 2),
            "today_net_move_points": round(today_net_move, 2),
            "today_net_ratio": round(today_net_ratio, 4),
            "today_flip_rate": round(today_flip_rate, 4),
            "today_large_bar_share": round(today_large_bar_share, 4),
            "today_breach_up": bool(breach_up),
            "today_breach_down": bool(breach_down),
            "today_signal": today_signal,
        }
    )

    result["role"] = role
    result["mode"] = mode
    result["forward_weight"] = _resolve_role_cfg(config, "forward_weight", role, 0.20)
    return result


def _aligned_probability(raw_probability: float, side: str) -> float:
    probability = _clamp(_coerce_float(raw_probability, 0.0), 0.0, 1.0)
    if str(side or "").upper() == "SHORT":
        return float(1.0 - probability)
    return float(probability)


def _extract_curve_markets(
    kalshi: Any,
    reference_es_price: float,
    overlay_cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    if kalshi is None or not getattr(kalshi, "enabled", False) or not getattr(kalshi, "is_healthy", False):
        return []
    if not hasattr(kalshi, "get_relative_markets_for_ui"):
        return []

    window_size = max(20, _coerce_int(overlay_cfg.get("strike_window_size"), 120))
    try:
        rows = kalshi.get_relative_markets_for_ui([float(reference_es_price)], window_size=window_size)
    except Exception:
        return []

    markets: List[Dict[str, Any]] = []
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        probability = _coerce_float(row.get("probability"), math.nan)
        strike_es = _coerce_float(row.get("strike_es"), math.nan)
        if not math.isfinite(strike_es):
            strike_spx = _coerce_float(row.get("strike"), math.nan)
            if math.isfinite(strike_spx):
                try:
                    strike_es = float(kalshi.spx_to_es(strike_spx))
                except Exception:
                    strike_es = strike_spx
        if not math.isfinite(probability) or not math.isfinite(strike_es):
            continue
        markets.append(
            {
                "strike_es": float(strike_es),
                "probability": _clamp(probability, 0.0, 1.0),
                "status": str(row.get("status", "") or ""),
            }
        )
    markets.sort(key=lambda item: item["strike_es"])
    return markets


def _is_curve_informative(markets: List[Dict[str, Any]], overlay_cfg: Dict[str, Any]) -> bool:
    min_curve_points = max(4, _coerce_int(overlay_cfg.get("min_curve_points"), 8))
    if len(markets) < min_curve_points:
        return False
    probabilities = [float(row["probability"]) for row in markets]
    if not probabilities:
        return False
    if (max(probabilities) - min(probabilities)) < _coerce_float(overlay_cfg.get("min_curve_range"), 0.08):
        return False
    unique_count = len({round(probability, 3) for probability in probabilities})
    if unique_count < max(3, _coerce_int(overlay_cfg.get("min_unique_probabilities"), 4)):
        return False
    return True


def _interpolated_aligned_probability(
    markets: List[Dict[str, Any]],
    price: float,
    side: str,
) -> Optional[float]:
    if not markets or not math.isfinite(price):
        return None
    below = [row for row in markets if row["strike_es"] <= float(price)]
    above = [row for row in markets if row["strike_es"] > float(price)]
    if below and above:
        low = below[-1]
        high = above[0]
        strike_span = float(high["strike_es"] - low["strike_es"])
        if strike_span <= 1e-9:
            raw_probability = float(low["probability"])
        else:
            frac = float(price - low["strike_es"]) / strike_span
            raw_probability = float(low["probability"]) * (1.0 - frac) + float(high["probability"]) * frac
    elif below:
        raw_probability = float(below[-1]["probability"])
    else:
        raw_probability = float(above[0]["probability"])
    return round(_aligned_probability(raw_probability, side), 4)


def _adjacent_fade_pair(
    forward_candidates: List[Dict[str, Any]],
    *,
    role: str,
    overlay_cfg: Dict[str, Any],
) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], str]:
    if not forward_candidates:
        return None, None, "no_forward_candidates"

    fade_abs_threshold = _resolve_role_cfg(overlay_cfg, "fade_absolute_threshold", role, 0.50)
    fade_delta_threshold = _resolve_role_cfg(overlay_cfg, "fade_adjacent_delta_threshold", role, 0.18)

    for idx in range(len(forward_candidates) - 1):
        current_row = forward_candidates[idx]
        next_row = forward_candidates[idx + 1]
        current_probability = float(current_row["aligned_probability"])
        next_probability = float(next_row["aligned_probability"])
        adjacent_drop = float(current_probability - next_probability)
        if current_probability >= fade_abs_threshold and adjacent_drop >= fade_delta_threshold:
            return current_row, next_row, "adjacent_drop"
        if current_probability >= fade_abs_threshold and next_probability < fade_abs_threshold:
            return current_row, next_row, "below_threshold"

    supportive_rows = [
        row for row in forward_candidates
        if float(row["aligned_probability"]) >= fade_abs_threshold
    ]
    if supportive_rows:
        support_row = supportive_rows[-1]
        next_index = min(len(forward_candidates) - 1, forward_candidates.index(support_row) + 1)
        fade_row = forward_candidates[next_index] if next_index != forward_candidates.index(support_row) else None
        return support_row, fade_row, "last_supportive"

    nearest_row = min(forward_candidates, key=lambda item: item["delta_points"])
    return nearest_row, None, "nearest_fallback"


def build_trade_plan(
    signal: Optional[Dict[str, Any]],
    current_price: float,
    kalshi: Any,
    *,
    price_action_profile: Optional[Dict[str, Any]] = None,
    overlay_cfg: Optional[Dict[str, Any]] = None,
    tick_size: float = 0.25,
) -> Dict[str, Any]:
    config = _merge_overlay_config(overlay_cfg)
    role = str((price_action_profile or {}).get("role") or "background")
    mode = str((price_action_profile or {}).get("mode") or "level")
    forward_weight = _coerce_float(
        (price_action_profile or {}).get("forward_weight"),
        _resolve_role_cfg(config, "forward_weight", role, 0.20),
    )
    result: Dict[str, Any] = {
        "enabled": bool(config.get("enabled", True)),
        "applied": False,
        "role": role,
        "mode": mode,
        "forward_weight": forward_weight,
        "curve_informative": False,
        "entry_probability": None,
        "probe_price": None,
        "probe_probability": None,
        "momentum_delta": None,
        "momentum_retention": None,
        "entry_support_score": None,
        "entry_threshold": _resolve_role_cfg(config, "entry_threshold", role, 0.45),
        "entry_blocked": False,
        "size_multiplier": 1.0,
        "tp_dist": None,
        "target_price": None,
        "tp_adjusted": False,
        "support_price": None,
        "fade_price": None,
        "anchor_price": None,
        "anchor_probability": None,
        "fade_reason": "",
        "support_span_points": None,
        "directional_distance_points": None,
        "sentiment_momentum": None,
        "trail_enabled": False,
        "trail_trigger_price": None,
        "trail_buffer_ticks": 0,
        "reason": "",
    }
    if not bool(config.get("enabled", True)):
        result["reason"] = "disabled"
        return result
    signal = signal if isinstance(signal, dict) else {}
    side = str(signal.get("side", "") or "").strip().upper()
    if side not in {"LONG", "SHORT"}:
        result["reason"] = "invalid_side"
        return result

    entry_price = _coerce_float(signal.get("entry_price", current_price), current_price)
    if not math.isfinite(entry_price) or entry_price <= 0.0:
        result["reason"] = "invalid_entry_price"
        return result
    tp_dist = _coerce_float(signal.get("tp_dist"), math.nan)
    if not math.isfinite(tp_dist) or tp_dist <= 0.0:
        result["reason"] = "missing_tp"
        return result

    markets = _extract_curve_markets(kalshi, entry_price, config)
    if not _is_curve_informative(markets, config):
        result["reason"] = "uninformative_curve"
        return result
    result["curve_informative"] = True

    entry_probability = _interpolated_aligned_probability(markets, entry_price, side)
    if entry_probability is None:
        result["reason"] = "entry_probability_unavailable"
        return result

    probe_points = max(float(tick_size), _coerce_float(config.get("momentum_probe_points"), 5.0))
    probe_price = float(entry_price + probe_points) if side == "LONG" else float(entry_price - probe_points)
    probe_probability = _interpolated_aligned_probability(markets, probe_price, side)
    if probe_probability is None:
        probe_probability = entry_probability
    momentum_delta = float(probe_probability - entry_probability)
    if entry_probability > 1e-9:
        momentum_retention = float(_clamp(probe_probability / entry_probability, 0.0, 1.5))
    else:
        momentum_retention = 0.0

    sentiment = {}
    try:
        sentiment = kalshi.get_sentiment(entry_price) if kalshi is not None else {}
    except Exception:
        sentiment = {}
    directional_distance = _coerce_float(sentiment.get("distance_es"), 0.0)
    if side == "SHORT":
        directional_distance = -directional_distance

    try:
        raw_momentum = kalshi.get_sentiment_momentum(entry_price, lookback=3) if kalshi is not None else None
    except Exception:
        raw_momentum = None
    signed_momentum = _coerce_float(raw_momentum, 0.0)
    if side == "SHORT":
        signed_momentum = -signed_momentum

    max_target_window_points = max(
        _coerce_float(config.get("max_target_window_points"), 32.0),
        tp_dist * _coerce_float(config.get("max_target_window_tp_mult"), 3.5),
    )
    forward_candidates: List[Dict[str, Any]] = []
    for row in markets:
        strike_es = float(row["strike_es"])
        delta = float(strike_es - entry_price)
        if side == "LONG":
            if delta <= max(float(tick_size), 1e-9):
                continue
            if delta > max_target_window_points:
                continue
        else:
            if delta >= -max(float(tick_size), 1e-9):
                continue
            if abs(delta) > max_target_window_points:
                continue
        aligned_probability = _aligned_probability(row["probability"], side)
        forward_candidates.append(
            {
                **row,
                "aligned_probability": float(aligned_probability),
                "delta_points": float(abs(delta)),
            }
        )

    support_row, fade_row, fade_reason = _adjacent_fade_pair(
        forward_candidates,
        role=role,
        overlay_cfg=config,
    )

    fade_strike_price = None
    support_span_points = 0.0
    if support_row is not None:
        support_price = float(support_row["strike_es"])
        result["support_price"] = round(support_price, 2)
        result["anchor_probability"] = round(float(support_row["aligned_probability"]), 4)
        fade_strike_price = support_price
        if fade_row is not None:
            result["fade_price"] = round(float(fade_row["strike_es"]), 2)
        support_span_points = abs(float(fade_strike_price) - float(entry_price))

    tp_direction = 1.0 if side == "LONG" else -1.0
    base_target_price = float(entry_price + (tp_direction * tp_dist))
    adjusted_target_price = base_target_price
    is_breakout = mode == "breakout_outrageous"
    if fade_strike_price is not None and math.isfinite(fade_strike_price):
        adjusted_target_price = float(
            base_target_price + (forward_weight * (float(fade_strike_price) - float(base_target_price)))
        )

    min_tp_multiplier = max(0.10, _coerce_float(config.get("min_tp_multiplier"), 0.55))
    max_tp_multiplier = max(min_tp_multiplier, _coerce_float(config.get("max_tp_multiplier"), 1.75))
    if is_breakout:
        max_tp_multiplier = max(
            max_tp_multiplier,
            _coerce_float(config.get("breakout_max_tp_multiplier"), 2.5),
        )
    adjusted_tp_dist = abs(float(adjusted_target_price) - float(entry_price))
    adjusted_tp_dist = _clamp(
        adjusted_tp_dist,
        max(float(tick_size), tp_dist * min_tp_multiplier),
        max(float(tick_size), tp_dist * max_tp_multiplier),
    )
    adjusted_tp_dist = _safe_tick_round(adjusted_tp_dist, tick_size)
    adjusted_target_price = float(entry_price + (tp_direction * adjusted_tp_dist))

    room_score = _clamp(float(support_span_points) / max(float(tp_dist), float(tick_size)), 0.0, 1.0)
    distance_score = _clamp(0.5 + (float(directional_distance) / 20.0), 0.0, 1.0)
    time_momentum_score = _clamp(0.5 + (float(signed_momentum) / 0.10), 0.0, 1.0)
    retention_floor = _resolve_role_cfg(config, "momentum_retention_floor", role, 0.76)
    retention_score = _clamp(momentum_retention / max(retention_floor, 1e-9), 0.0, 1.0)
    entry_support_score = (
        (0.35 * float(entry_probability))
        + (0.25 * float(probe_probability))
        + (0.20 * float(retention_score))
        + (0.10 * float(distance_score))
        + (0.10 * float(time_momentum_score))
    )
    entry_support_score = round(float(entry_support_score), 4)

    entry_threshold = _resolve_role_cfg(config, "entry_threshold", role, 0.45)
    block_buffer = _resolve_role_cfg(config, "entry_block_buffer", role, 1.0)
    size_floor = _resolve_role_cfg(config, "entry_size_floor", role, 0.85)
    entry_blocked = bool(
        role != "background"
        and (
            entry_support_score < (entry_threshold - block_buffer)
            or momentum_retention + 1e-9 < retention_floor
        )
    )
    size_multiplier = 1.0
    if entry_support_score < entry_threshold:
        ratio = _clamp(entry_support_score / max(entry_threshold, 1e-9), 0.0, 1.0)
        size_multiplier = float(size_floor + ((1.0 - size_floor) * ratio))
        size_multiplier = _clamp(size_multiplier, size_floor, 1.0)

    trail_enabled = bool(
        fade_strike_price is not None
        and role in set(config.get("trail_enabled_roles") or [])
        and support_span_points >= max(2.0 * float(tick_size), float(tp_dist) * 0.35)
    )

    trail_buffer_cfg = config.get("trail_buffer_ticks", {})
    if isinstance(trail_buffer_cfg, dict):
        trail_buffer_ticks = max(1, _coerce_int(trail_buffer_cfg.get(role), 4))
    else:
        trail_buffer_ticks = max(1, _coerce_int(trail_buffer_cfg, 4))

    result.update(
        {
            "applied": True,
            "entry_probability": float(entry_probability),
            "probe_price": round(float(probe_price), 2),
            "probe_probability": float(probe_probability),
            "momentum_delta": round(float(momentum_delta), 4),
            "momentum_retention": round(float(momentum_retention), 4),
            "entry_support_score": entry_support_score,
            "entry_threshold": float(entry_threshold),
            "entry_blocked": bool(entry_blocked),
            "size_multiplier": float(size_multiplier),
            "tp_dist": float(adjusted_tp_dist),
            "target_price": round(float(adjusted_target_price), 2),
            "tp_adjusted": abs(float(adjusted_tp_dist) - float(tp_dist)) > 1e-9,
            "anchor_price": round(float(fade_strike_price), 2) if fade_strike_price is not None else None,
            "support_span_points": round(float(support_span_points), 2),
            "directional_distance_points": round(float(directional_distance), 2),
            "sentiment_momentum": round(float(signed_momentum), 4),
            "fade_reason": str(fade_reason or ""),
            "trail_enabled": bool(trail_enabled),
            "trail_trigger_price": round(float(fade_strike_price), 2) if fade_strike_price is not None else None,
            "trail_buffer_ticks": int(trail_buffer_ticks),
            "reason": "ok",
        }
    )
    return result


def compute_tp_trail_stop(
    trade: Optional[Dict[str, Any]],
    *,
    market_price: float,
    bar_high: Optional[float] = None,
    bar_low: Optional[float] = None,
    tick_size: float = 0.25,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "triggered": False,
        "should_update": False,
        "stop_price": None,
    }
    if not isinstance(trade, dict):
        return result
    if not bool(trade.get("kalshi_tp_trail_enabled", False)):
        return result
    side = str(trade.get("side", "") or "").strip().upper()
    if side not in {"LONG", "SHORT"}:
        return result
    trigger_price = _coerce_float(trade.get("kalshi_tp_trigger_price"), math.nan)
    anchor_price = _coerce_float(
        trade.get("kalshi_tp_anchor_price", trade.get("kalshi_tp_trigger_price")),
        math.nan,
    )
    current_stop_price = _coerce_float(trade.get("current_stop_price"), math.nan)
    if not math.isfinite(trigger_price) or not math.isfinite(anchor_price) or not math.isfinite(current_stop_price):
        return result

    high_water = _coerce_float(bar_high, market_price)
    low_water = _coerce_float(bar_low, market_price)
    breached = (
        high_water >= trigger_price - 1e-9
        if side == "LONG"
        else low_water <= trigger_price + 1e-9
    )
    if not breached:
        return result

    buffer_ticks = max(1, _coerce_int(trade.get("kalshi_tp_trail_buffer_ticks"), 4))
    buffer_points = float(buffer_ticks) * max(1e-9, float(tick_size))
    candidate_stop = (
        float(anchor_price - buffer_points)
        if side == "LONG"
        else float(anchor_price + buffer_points)
    )
    candidate_stop = _safe_tick_round(candidate_stop, tick_size)
    improved = (
        candidate_stop > current_stop_price + 1e-9
        if side == "LONG"
        else candidate_stop < current_stop_price - 1e-9
    )
    result["triggered"] = True
    result["should_update"] = bool(improved)
    result["stop_price"] = float(candidate_stop) if math.isfinite(candidate_stop) else None
    return result
