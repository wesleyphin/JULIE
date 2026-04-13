from __future__ import annotations

import atexit
import json
import os
from pathlib import Path
from typing import Optional


class SingletonProcessLock:
    def __init__(self, lock_path: Path, *, name: str) -> None:
        self.lock_path = Path(lock_path)
        self.name = str(name or "process")
        self._handle = None

    def acquire(self) -> bool:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(self.lock_path, "a+", encoding="utf-8")
        try:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write("\n")
                handle.flush()
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.close()
            return False

        payload = {
            "pid": os.getpid(),
            "name": self.name,
            "cwd": str(Path.cwd()),
        }
        handle.seek(0)
        handle.truncate()
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        handle.flush()
        self._handle = handle
        atexit.register(self.release)
        return True

    def read_existing(self) -> str:
        try:
            return self.lock_path.read_text(encoding="utf-8").strip()
        except OSError:
            return ""

    def release(self) -> None:
        handle = self._handle
        self._handle = None
        if handle is None:
            return
        try:
            handle.flush()
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            handle.close()
        except OSError:
            pass


def acquire_singleton_lock(lock_path: Path, *, name: str) -> Optional[SingletonProcessLock]:
    lock = SingletonProcessLock(lock_path, name=name)
    if lock.acquire():
        return lock
    return None
