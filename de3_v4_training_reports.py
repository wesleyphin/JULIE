import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def resolve_path(raw_path: Any) -> Path:
    p = Path(str(raw_path or "").strip())
    if not p.is_absolute():
        p = Path(__file__).resolve().parent / p
    return p


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = dict(payload) if isinstance(payload, dict) else {}
    data.setdefault("created_at_utc", datetime.now(timezone.utc).isoformat())
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")

