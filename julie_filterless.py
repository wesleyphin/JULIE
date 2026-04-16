import asyncio
import os
from pathlib import Path

from process_singleton import acquire_singleton_lock

os.environ.setdefault("JULIE_FILTERLESS_ONLY", "1")
os.environ.setdefault("JULIE_DISABLE_STRATEGY_FILTERS", "1")

TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}


def _acquire_filterless_live_lock() -> object | None:
    enforce_singleton = str(
        os.environ.get("JULIE_ENFORCE_LIVE_SINGLETON", "1")
    ).strip().lower()
    if enforce_singleton not in TRUTHY_ENV_VALUES:
        return None
    lock_path = Path(__file__).resolve().parent / "logs" / str(
        os.environ.get("JULIE_LIVE_LOCK_FILENAME", "filterless_live.lock")
        or "filterless_live.lock"
    )
    lock_name = str(
        os.environ.get("JULIE_LIVE_LOCK_NAME", "filterless_live_bot")
        or "filterless_live_bot"
    )
    lock = acquire_singleton_lock(lock_path, name=lock_name)
    if lock is not None:
        return lock

    existing = ""
    try:
        existing = lock_path.read_text(encoding="utf-8").strip()
    except OSError:
        existing = ""
    print(f"Another filterless live bot instance is already running. Lock: {lock_path}")
    if existing:
        print(existing)
    raise SystemExit(0)


FILTERLESS_LIVE_LOCK = _acquire_filterless_live_lock()

from julie001 import run_bot


if __name__ == "__main__":
    asyncio.run(run_bot())
