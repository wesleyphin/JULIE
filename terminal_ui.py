from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Dict

from rich.console import Console


class TerminalUI:
    """Lightweight fallback terminal UI used by julie_ui.py."""

    def __init__(self) -> None:
        self.console = Console()
        self._lock = threading.Lock()
        self._running = False
        self._events: deque[tuple[str, str, str]] = deque(maxlen=80)
        self._market_context: Dict[str, Any] = {}
        self._account_info: Dict[str, Any] = {}
        self._position: Dict[str, Any] = {"active": False}
        self._filters: Dict[str, Dict[str, Any]] = {}

    def start(self, refresh_rate: float = 1.0) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
        self.console.print("[bold cyan]Julie terminal monitor started[/bold cyan]")
        self.console.print(f"[dim]Refresh rate: {refresh_rate:.1f}s[/dim]")

    def stop(self) -> None:
        with self._lock:
            if not self._running:
                return
            self._running = False
        self.console.print("[bold yellow]Julie terminal monitor stopped[/bold yellow]")

    def add_event(self, category: str, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        cat = str(category or "INFO").upper()
        msg = str(message or "").strip()
        with self._lock:
            self._events.append((timestamp, cat, msg))
        self.console.print(f"[{timestamp}] [{cat}] {msg}")

    def update_filter_status(self, name: str, passed: bool, reason: str = "") -> None:
        payload = {
            "passed": bool(passed),
            "reason": str(reason or "").strip(),
            "updated_at": time.strftime("%H:%M:%S"),
        }
        with self._lock:
            self._filters[str(name)] = payload
        status = "PASS" if passed else "BLOCK"
        extra = f" | {payload['reason']}" if payload["reason"] else ""
        self.console.print(f"[FILTER] {name}: {status}{extra}")

    def update_account_info(self, payload: Dict[str, Any]) -> None:
        with self._lock:
            self._account_info.update(dict(payload or {}))
        self.console.print(f"[ACCOUNT] {self._account_info}")

    def update_market_context(self, payload: Dict[str, Any]) -> None:
        with self._lock:
            self._market_context.update(dict(payload or {}))
        self.console.print(f"[MARKET] {self._market_context}")

    def update_position(self, payload: Dict[str, Any]) -> None:
        with self._lock:
            self._position.update(dict(payload or {}))
        self.console.print(f"[POSITION] {self._position}")


_UI_SINGLETON: TerminalUI | None = None
_UI_LOCK = threading.Lock()


def get_ui() -> TerminalUI:
    global _UI_SINGLETON
    with _UI_LOCK:
        if _UI_SINGLETON is None:
            _UI_SINGLETON = TerminalUI()
        return _UI_SINGLETON
