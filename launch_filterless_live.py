import asyncio
import atexit
import json
import os
import platform
import socket
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from zoneinfo import ZoneInfo


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

from bot_state import load_bot_state
from config import CONFIG
from config_secrets import SECRETS
from julie001 import run_bot
from process_singleton import acquire_singleton_lock
from services.kalshi_provider import KalshiProvider
from tools.filterless_dashboard_bridge import DEFAULT_KALSHI_SNAPSHOT_PATH, write_json_atomic


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


NY_TZ = ZoneInfo("America/New_York")
ROOT = Path(__file__).resolve().parent
BRIDGE_SCRIPT = ROOT / "tools" / "filterless_dashboard_bridge.py"
BRIDGE_LOG_PATH = ROOT / "logs" / "dashboard_bridge.log"
BRIDGE_PROCESS: Optional[subprocess.Popen[Any]] = None
BRIDGE_LOG_HANDLE = None


def _coerce_price_from_state() -> Optional[float]:
    state = load_bot_state(ROOT / "bot_state.json")
    if not isinstance(state, dict):
        return None
    live_position = state.get("live_position")
    if isinstance(live_position, dict):
        for key in ("current_price", "avg_price", "entry_price"):
            value = live_position.get(key)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


def _build_kalshi_provider() -> Optional[KalshiProvider]:
    kalshi_cfg = CONFIG.get("KALSHI", {}) if isinstance(CONFIG, dict) else {}
    if not isinstance(kalshi_cfg, dict):
        return None
    provider_cfg = dict(kalshi_cfg)
    provider_cfg["key_id"] = str(SECRETS.get("KALSHI_KEY_ID", provider_cfg.get("key_id", "")) or "")
    provider_cfg["private_key_path"] = str(
        SECRETS.get("KALSHI_PRIVATE_KEY_PATH", provider_cfg.get("private_key_path", "")) or ""
    )
    provider = KalshiProvider(provider_cfg)
    return provider


def _kalshi_disabled_payload() -> Dict[str, Any]:
    return {
        "enabled": False,
        "healthy": False,
        "updated_at": datetime.now(NY_TZ).isoformat(),
        "basis_offset": 0.0,
        "probability_60m": None,
        "event_ticker": None,
        "spx_reference_price": None,
        "strikes": [],
    }


async def _kalshi_snapshot_loop(path: Path, interval_seconds: float = 10.0) -> None:
    provider = _build_kalshi_provider()
    while True:
        if provider is None or not getattr(provider, "enabled", False):
            write_json_atomic(path, _kalshi_disabled_payload())
            await asyncio.sleep(interval_seconds)
            continue
        price = _coerce_price_from_state()
        sentiment = provider.get_sentiment(price) if price is not None else {}
        strikes = provider._fetch_event_markets()  # noqa: SLF001 - intentionally surfacing full ladder to UI
        payload: Dict[str, Any] = {
            "enabled": True,
            "healthy": bool(getattr(provider, "is_healthy", False)),
            "updated_at": datetime.now(NY_TZ).isoformat(),
            "basis_offset": float(getattr(provider, "basis_offset", 0.0) or 0.0),
            "probability_60m": sentiment.get("probability"),
            "event_ticker": provider._current_event_ticker(),  # noqa: SLF001 - informational only
            "spx_reference_price": (float(price) - float(provider.basis_offset)) if price is not None else None,
            "strikes": strikes if isinstance(strikes, list) else [],
        }
        write_json_atomic(path, payload)
        await asyncio.sleep(interval_seconds)


def _start_bridge_process(path: Path) -> None:
    global BRIDGE_PROCESS, BRIDGE_LOG_HANDLE
    if BRIDGE_PROCESS is not None and BRIDGE_PROCESS.poll() is None:
        return

    BRIDGE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    BRIDGE_LOG_HANDLE = BRIDGE_LOG_PATH.open("a", encoding="utf-8", buffering=1)
    cmd = [
        sys.executable,
        str(BRIDGE_SCRIPT),
        "--follow",
        "--kalshi-snapshot-path",
        str(path),
    ]
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        BRIDGE_PROCESS = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=BRIDGE_LOG_HANDLE,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
            close_fds=True,
        )
    else:
        BRIDGE_PROCESS = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=BRIDGE_LOG_HANDLE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )


def _cleanup_bridge_process() -> None:
    global BRIDGE_PROCESS, BRIDGE_LOG_HANDLE
    process = BRIDGE_PROCESS
    if process is not None and process.poll() is None:
        try:
            if os.name == "nt":
                process.terminate()
            else:
                os.killpg(process.pid, signal.SIGTERM)
        except Exception:
            pass
    if BRIDGE_LOG_HANDLE is not None:
        try:
            BRIDGE_LOG_HANDLE.flush()
            BRIDGE_LOG_HANDLE.close()
        except Exception:
            pass
    BRIDGE_PROCESS = None
    BRIDGE_LOG_HANDLE = None


async def _run_all() -> None:
    kalshi_snapshot_path = DEFAULT_KALSHI_SNAPSHOT_PATH
    _start_bridge_process(kalshi_snapshot_path)
    tasks = [
        asyncio.create_task(_kalshi_snapshot_loop(kalshi_snapshot_path)),
        asyncio.create_task(run_bot()),
    ]
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)
    for task in done:
        exc = task.exception()
        if exc is not None:
            raise exc


if __name__ == "__main__":
    atexit.register(_cleanup_bridge_process)
    asyncio.run(_run_all())
