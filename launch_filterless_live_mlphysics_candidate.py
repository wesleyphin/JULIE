import asyncio
import json
import os
import platform
import socket
import sys
from pathlib import Path


def _force_utf8_stdio() -> None:
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="backslashreplace")
            except Exception:
                pass


def _stabilize_windows_platform_queries() -> None:
    if os.name != "nt":
        return

    arch = (
        os.environ.get("PROCESSOR_ARCHITEW6432")
        or os.environ.get("PROCESSOR_ARCHITECTURE")
        or ""
    ).strip()
    if not arch:
        return

    normalized = {
        "AMD64": "x86_64",
        "X86": "x86",
        "ARM64": "arm64",
    }.get(arch.upper(), arch)

    winver = sys.getwindowsversion()
    release = "11" if winver.major >= 10 and winver.build >= 22000 else str(winver.major)
    version = f"{winver.major}.{winver.minor}.{winver.build}"
    uname_result = platform.uname_result(
        "Windows",
        socket.gethostname(),
        release,
        version,
        normalized,
    )

    platform.uname = lambda: uname_result
    platform.system = lambda: uname_result.system
    platform.machine = lambda: uname_result.machine
    platform.release = lambda: uname_result.release
    platform.version = lambda: uname_result.version
    platform.processor = lambda: normalized
    platform.win32_ver = lambda release="", version="", csd="", ptype="": (
        uname_result.release,
        uname_result.version,
        "",
        "",
    )


def _deep_merge(base: dict, overrides: dict) -> dict:
    out = dict(base)
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out.get(key) or {}, value)
        else:
            out[key] = value
    return out


_force_utf8_stdio()
_stabilize_windows_platform_queries()

ROOT = Path(__file__).resolve().parent
PROFILE_NAME = str(
    os.environ.get(
        "ML_PHYSICS_LIVE_PROFILE",
        "ml_physics_live_candidate_asia_thu_nypm_fri.json",
    )
    or "ml_physics_live_candidate_asia_thu_nypm_fri.json"
).strip()
PROFILE_PATH = ROOT / "configs" / PROFILE_NAME

os.environ.setdefault("JULIE_FILTERLESS_ONLY", "1")
os.environ.setdefault("JULIE_DISABLE_STRATEGY_FILTERS", "1")

from process_singleton import acquire_singleton_lock


LIVE_LOCK_PATH = ROOT / "logs" / "filterless_live.lock"
LIVE_LOCK = acquire_singleton_lock(LIVE_LOCK_PATH, name="filterless_live_bot")
if LIVE_LOCK is None:
    existing = ""
    try:
        existing = LIVE_LOCK_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        existing = ""
    print(
        "Another filterless live bot instance is already running. "
        f"Lock: {LIVE_LOCK_PATH}"
    )
    if existing:
        print(existing)
    raise SystemExit(0)

import config as app_config


if not PROFILE_PATH.exists():
    raise FileNotFoundError(f"Missing MLPhysics live candidate profile: {PROFILE_PATH}")

profile_overrides = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
effective_config = _deep_merge(dict(app_config.CONFIG), profile_overrides)
app_config.CONFIG.clear()
app_config.CONFIG.update(effective_config)
print(f"Loaded MLPhysics live candidate profile: {PROFILE_PATH.name}")

from julie001 import run_bot


if __name__ == "__main__":
    asyncio.run(run_bot())
