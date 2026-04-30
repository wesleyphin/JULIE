from __future__ import annotations

import json
import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from typing import Any, Dict, Optional

NY_TZ = ZoneInfo("America/New_York")
STATE_VERSION = 3
STATE_PATH = Path(__file__).with_name("bot_state.json")


def empty_sentiment_state() -> Dict[str, Any]:
    return {
        "enabled": False,
        "active": False,
        "healthy": False,
        "model_loaded": False,
        "quantized_8bit": False,
        "target_handle": None,
        "source": None,
        "last_poll_at": None,
        "last_analysis_at": None,
        "latest_post_id": None,
        "latest_post_created_at": None,
        "latest_post_url": None,
        "latest_post_text": None,
        "sentiment_label": None,
        "sentiment_score": None,
        "finbert_confidence": None,
        "trigger_side": None,
        "trigger_reason": None,
        "last_error": None,
        "metadata": {},
    }


def normalize_sentiment_state(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        return empty_sentiment_state()
    normalized = empty_sentiment_state()
    normalized.update({key: value.get(key) for key in normalized.keys()})
    return normalized


def trading_day_start(ts: datetime.datetime, tz: ZoneInfo = NY_TZ) -> datetime.datetime:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=tz)
    ts = ts.astimezone(tz)
    start = ts.replace(hour=18, minute=0, second=0, microsecond=0)
    if ts.hour < 18:
        start -= datetime.timedelta(days=1)
    return start


def serialize_dt(dt: Optional[datetime.datetime]) -> Optional[str]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=NY_TZ)
    return dt.isoformat()


def parse_dt(value: Optional[str]) -> Optional[datetime.datetime]:
    if not value:
        return None
    try:
        return datetime.datetime.fromisoformat(value)
    except Exception:
        return None


def load_bot_state(path: Path = STATE_PATH) -> Dict[str, Any]:
    try:
        if not path.exists():
            return {}
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _json_default(obj: Any) -> Any:
    """Coerce numpy scalar / array types to native Python types so json.dumps
    doesn't choke. Resolves the long-running 'Object of type bool is not JSON
    serializable' error (numpy.bool_ reproduces this exact message)."""
    try:
        import numpy as np
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_bot_state(state: Dict[str, Any], path: Path = STATE_PATH) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(state, indent=2, sort_keys=True, default=_json_default))
    tmp_path.replace(path)
