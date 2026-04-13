import asyncio
import os
import platform
import socket
import sys
from pathlib import Path


def _force_utf8_stdio() -> None:
    """
    Reconfigure stdio early so emoji-heavy logging does not crash under cp1252.

    This repo emits symbols like checkmarks and warning icons during import time,
    before runtime startup is complete. On some Windows machines, especially after
    migrating a workspace between PCs, redirected stdio can default to cp1252.
    """
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
    """
    Avoid Python 3.13 Windows WMI probes during large library imports.

    On this machine, `platform.uname()` / `platform.system()` / `platform.machine()`
    can block inside `platform._wmi_query()`, which delays or stalls filterless bot
    startup before `run_bot()` begins.
    """
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


_force_utf8_stdio()
_stabilize_windows_platform_queries()

os.environ.setdefault("JULIE_FILTERLESS_ONLY", "1")
os.environ.setdefault("JULIE_DISABLE_STRATEGY_FILTERS", "1")

from process_singleton import acquire_singleton_lock


LIVE_LOCK_PATH = Path(__file__).resolve().parent / "logs" / "filterless_live.lock"
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

from julie001 import run_bot


if __name__ == "__main__":
    asyncio.run(run_bot())
