import math
from typing import Any, Optional


DISTANCE_MODE_POINTS = "points"
DISTANCE_MODE_PERCENT_OF_ENTRY = "percent_of_entry"
DEFAULT_DISTANCE_MODE = DISTANCE_MODE_POINTS
DEFAULT_DISTANCE_ROUNDING = "ceil"


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return float(out)


def normalize_distance_mode(value: Any, default: str = DEFAULT_DISTANCE_MODE) -> str:
    text = str(value or "").strip().lower()
    if not text:
        text = str(default or DEFAULT_DISTANCE_MODE).strip().lower()
    if text in {"points", "point", "pts", "absolute", "fixed_points"}:
        return DISTANCE_MODE_POINTS
    if text in {
        "percent_of_entry",
        "percent",
        "pct",
        "pct_of_entry",
        "percent_move",
        "pct_move",
        "entry_percent",
    }:
        return DISTANCE_MODE_PERCENT_OF_ENTRY
    return normalize_distance_mode(default, DEFAULT_DISTANCE_MODE) if text != str(default).strip().lower() else DISTANCE_MODE_POINTS


def normalize_distance_rounding(value: Any, default: str = DEFAULT_DISTANCE_ROUNDING) -> str:
    text = str(value or "").strip().lower()
    if text in {"ceil", "ceiling", "up"}:
        return "ceil"
    if text in {"round", "nearest"}:
        return "round"
    if text in {"floor", "down"}:
        return "floor"
    text_default = str(default or DEFAULT_DISTANCE_ROUNDING).strip().lower()
    if text_default in {"ceil", "round", "floor"}:
        return text_default
    return DEFAULT_DISTANCE_ROUNDING


def distance_mode_uses_percent(value: Any) -> bool:
    return normalize_distance_mode(value) == DISTANCE_MODE_PERCENT_OF_ENTRY


def round_distance_points(
    points: Any,
    *,
    tick_size: float,
    rounding: Any = DEFAULT_DISTANCE_ROUNDING,
) -> float:
    points_val = abs(safe_float(points, 0.0))
    tick_val = abs(safe_float(tick_size, 0.0))
    if points_val <= 0.0:
        return 0.0
    if tick_val <= 0.0:
        return float(points_val)
    ticks = points_val / tick_val
    rounding_mode = normalize_distance_rounding(rounding)
    if rounding_mode == "floor":
        ticks_rounded = math.floor(ticks)
    elif rounding_mode == "round":
        ticks_rounded = int(round(ticks))
    else:
        ticks_rounded = math.ceil(ticks)
    ticks_rounded = max(1, int(ticks_rounded))
    return float(ticks_rounded * tick_val)


def convert_distance_to_points(
    distance_value: Any,
    *,
    distance_mode: Any = DEFAULT_DISTANCE_MODE,
    reference_price: Optional[Any] = None,
    tick_size: float = 0.25,
    rounding: Any = DEFAULT_DISTANCE_ROUNDING,
) -> float:
    raw_val = safe_float(distance_value, 0.0)
    if raw_val <= 0.0:
        return 0.0
    mode = normalize_distance_mode(distance_mode)
    if mode == DISTANCE_MODE_PERCENT_OF_ENTRY:
        ref_price = safe_float(reference_price, 0.0)
        if ref_price <= 0.0:
            return 0.0
        raw_points = ref_price * (raw_val / 100.0)
        return round_distance_points(raw_points, tick_size=tick_size, rounding=rounding)
    return round_distance_points(raw_val, tick_size=tick_size, rounding=rounding)


def convert_distance_pair_to_points(
    *,
    sl_value: Any,
    tp_value: Any,
    distance_mode: Any = DEFAULT_DISTANCE_MODE,
    reference_price: Optional[Any] = None,
    tick_size: float = 0.25,
    rounding: Any = DEFAULT_DISTANCE_ROUNDING,
) -> tuple[float, float]:
    return (
        convert_distance_to_points(
            sl_value,
            distance_mode=distance_mode,
            reference_price=reference_price,
            tick_size=tick_size,
            rounding=rounding,
        ),
        convert_distance_to_points(
            tp_value,
            distance_mode=distance_mode,
            reference_price=reference_price,
            tick_size=tick_size,
            rounding=rounding,
        ),
    )
