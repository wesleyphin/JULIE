from __future__ import annotations

import json
import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from typing import Any, Dict, Optional

NY_TZ = ZoneInfo("America/New_York")
STATE_VERSION = 2
STATE_PATH = Path(__file__).with_name("bot_state.json")


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


def save_bot_state(state: Dict[str, Any], path: Path = STATE_PATH) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(state, indent=2, sort_keys=True))
    tmp_path.replace(path)
