import math
from typing import Any, Dict, Optional, Tuple

LANE_LONG_REV = "Long_Rev"
LANE_SHORT_REV = "Short_Rev"
LANE_LONG_MOM = "Long_Mom"
LANE_SHORT_MOM = "Short_Mom"
LANE_NO_TRADE = "No_Trade"

LANE_ORDER = [
    LANE_LONG_REV,
    LANE_SHORT_REV,
    LANE_LONG_MOM,
    LANE_SHORT_MOM,
]

STRATEGY_TYPE_TO_LANE = {
    "long_rev": LANE_LONG_REV,
    "short_rev": LANE_SHORT_REV,
    "long_mom": LANE_LONG_MOM,
    "short_mom": LANE_SHORT_MOM,
}


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        if not math.isfinite(out):
            return float(default)
        return out
    except Exception:
        return float(default)


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def clip(value: float, lo: float, hi: float) -> float:
    return float(max(float(lo), min(float(hi), float(value))))


def safe_div(numer: float, denom: float, default: float = 0.0) -> float:
    d = float(denom)
    if abs(d) <= 1e-12:
        return float(default)
    return float(numer / d)


def normalize_signed(value: float, scale: float = 1.0, clip_abs: float = 3.0) -> float:
    s = float(scale) if abs(float(scale)) > 1e-12 else 1.0
    return clip(float(value) / s, -float(clip_abs), float(clip_abs))


def strategy_type_to_lane(strategy_type: Any) -> str:
    raw = str(strategy_type or "").strip().lower()
    return STRATEGY_TYPE_TO_LANE.get(raw, "")


def lane_to_side(lane: Any) -> str:
    lane_norm = str(lane or "").strip()
    if lane_norm.startswith("Long"):
        return "long"
    if lane_norm.startswith("Short"):
        return "short"
    return ""


def format_threshold(raw_thresh: Any) -> str:
    val = safe_float(raw_thresh, 0.0)
    rounded = round(val)
    if abs(val - rounded) <= 1e-9:
        return str(int(rounded))
    return f"{val:.2f}".rstrip("0").rstrip(".")


def normalize_family_tag(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    clean_chars = []
    for ch in text:
        if ch.isalnum() or ch in {"_", "-"}:
            clean_chars.append(ch)
        elif ch in {" ", "|", ":"}:
            clean_chars.append("_")
    out = "".join(clean_chars).strip("_")
    return out


def build_family_id(
    *,
    timeframe: Any,
    session: Any,
    strategy_type: Any,
    threshold: Any,
    family_tag: Any = "",
) -> str:
    tf = str(timeframe or "").strip()
    sess = str(session or "").strip()
    stype = str(strategy_type or "").strip()
    side = lane_to_side(strategy_type_to_lane(stype))
    thresh_txt = format_threshold(threshold)
    family_tag_txt = normalize_family_tag(family_tag)
    if not tf or not sess or not stype:
        return ""
    family_id = f"{tf}|{sess}|{side}|{stype}|T{thresh_txt}"
    if family_tag_txt:
        family_id = f"{family_id}|F{family_tag_txt}"
    return family_id


def parse_timeframe_minutes(value: Any) -> Optional[int]:
    raw = str(value or "").strip().lower()
    if not raw:
        return None
    if raw.endswith("min"):
        raw = raw[:-3]
    try:
        mins = int(float(raw))
    except Exception:
        return None
    if mins <= 0:
        return None
    return int(mins)


def parse_session_block(value: Any) -> Optional[Tuple[int, int]]:
    text = str(value or "").strip()
    if "-" not in text:
        return None
    left, right = text.split("-", 1)
    try:
        start = int(left)
        end = int(right)
    except Exception:
        return None
    if start < 0 or start > 23:
        return None
    if end <= start or end > 24:
        return None
    return (start, end)


def session_mid_hour(value: Any) -> Optional[float]:
    block = parse_session_block(value)
    if block is None:
        return None
    start, end = block
    return float(start + end) / 2.0


def session_distance_hours(a: Any, b: Any) -> Optional[float]:
    ma = session_mid_hour(a)
    mb = session_mid_hour(b)
    if ma is None or mb is None:
        return None
    return abs(float(ma) - float(mb))


def unique_variant_id(preferred: Any, family_id: str, idx: int) -> str:
    raw = str(preferred or "").strip()
    if raw:
        return raw
    if family_id:
        return f"{family_id}#m{idx}"
    return f"variant_{idx}"
