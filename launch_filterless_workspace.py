#!/usr/bin/env python3
"""
Launch the filterless live bot, dashboard bridge, and Monte Carlo UI together.
"""

from __future__ import annotations

import atexit
import argparse
import ctypes
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback
import webbrowser
from pathlib import Path
from typing import Optional

from process_singleton import acquire_singleton_lock


ROOT = Path(__file__).resolve().parent
MONTE_CARLO_DIR = ROOT / "montecarlo" / "Backtest-Simulator-main"
MONTE_CARLO_DIST_DIR = MONTE_CARLO_DIR / "dist"
BOT_SCRIPT = ROOT / "launch_filterless_live.py"
BRIDGE_SCRIPT = ROOT / "tools" / "filterless_dashboard_bridge.py"
STATIC_SERVER_SCRIPT = ROOT / "tools" / "filterless_static_server.py"
BOT_LOG_PATH = ROOT / "julie_bot_stdout_stderr.log"
BRIDGE_LOG_PATH = ROOT / "logs" / "filterless_dashboard_bridge.log"
FRONTEND_LOG_PATH = ROOT / "logs" / "filterless_frontend.log"
WORKSPACE_ERROR_LOG_PATH = ROOT / "logs" / "filterless_workspace_launcher.err.log"
WORKSPACE_STATUS_LOG_PATH = ROOT / "logs" / "filterless_workspace_launcher.status.log"
WORKSPACE_PID_PATH = ROOT / "logs" / "filterless_workspace_pids.json"
WORKSPACE_LOCK_PATH = ROOT / "logs" / "filterless_workspace.lock"
VITE_PORT = 3000
MONTE_CARLO_URL = f"http://localhost:{VITE_PORT}/"
FILTERLESS_URL = f"http://localhost:{VITE_PORT}/filterless-live.html"
DASHBOARD_STATE_PATH = MONTE_CARLO_DIR / "public" / "filterless_live_state.json"
BOT_STALE_TIMEOUT_SECONDS = 180.0
BRIDGE_STALE_TIMEOUT_SECONDS = 180.0
PROCESS_RESTART_DELAY_SECONDS = 2.0
MAX_RESTARTS_PER_PROCESS = 5
_WORKSPACE_JOB_HANDLE: object | None = None


if os.name == "nt":
    from ctypes import wintypes

    class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_void_p),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]


    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]


    class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]


def close_workspace_job_handle() -> None:
    global _WORKSPACE_JOB_HANDLE
    if os.name != "nt":
        return
    handle = _WORKSPACE_JOB_HANDLE
    _WORKSPACE_JOB_HANDLE = None
    if not handle:
        return
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel32.CloseHandle.restype = wintypes.BOOL
        kernel32.CloseHandle(handle)
    except Exception:
        pass


def ensure_workspace_job_handle() -> object | None:
    global _WORKSPACE_JOB_HANDLE
    if os.name != "nt":
        return None
    if _WORKSPACE_JOB_HANDLE:
        return _WORKSPACE_JOB_HANDLE

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.argtypes = (ctypes.c_void_p, wintypes.LPCWSTR)
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.SetInformationJobObject.argtypes = (
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    )
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL
    job_handle = kernel32.CreateJobObjectW(None, None)
    if not job_handle:
        raise OSError(ctypes.get_last_error(), "CreateJobObjectW failed")

    info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    info.BasicLimitInformation.LimitFlags = 0x00002000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    ok = kernel32.SetInformationJobObject(
        job_handle,
        9,  # JobObjectExtendedLimitInformation
        ctypes.byref(info),
        ctypes.sizeof(info),
    )
    if not ok:
        error = ctypes.get_last_error()
        kernel32.CloseHandle(job_handle)
        raise OSError(error, "SetInformationJobObject failed")

    _WORKSPACE_JOB_HANDLE = job_handle
    atexit.register(close_workspace_job_handle)
    return _WORKSPACE_JOB_HANDLE


def attach_process_to_workspace_job(name: str, process: subprocess.Popen) -> None:
    if os.name != "nt":
        return
    process_handle = getattr(process, "_handle", None)
    if not process_handle:
        return
    try:
        job_handle = ensure_workspace_job_handle()
    except OSError as exc:
        log_workspace_status(
            f"Workspace child job unavailable; {name} pid={process.pid} may outlive the shell: {exc}"
        )
        return

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    ok = kernel32.AssignProcessToJobObject(job_handle, wintypes.HANDLE(int(process_handle)))
    if ok:
        return

    error = ctypes.get_last_error()
    if error == 5:
        log_workspace_status(
            f"Workspace child job could not adopt {name} pid={process.pid}; "
            "falling back to normal shutdown handling"
        )
        return
    log_workspace_status(
        f"Workspace child job attach failed for {name} pid={process.pid}: "
        f"[WinError {error}] AssignProcessToJobObject"
    )


def log_workspace_status(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"{timestamp} | {message}"
    print(line, flush=True)
    WORKSPACE_STATUS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with WORKSPACE_STATUS_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the filterless bot, dashboard bridge, and Monte Carlo UI together."
    )
    parser.add_argument(
        "--account-id",
        type=str,
        default=None,
        help="Skip the interactive Topstep prompt and use this account ID for the bot.",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open the Monte Carlo and filterless dashboard pages automatically.",
    )
    parser.add_argument(
        "--skip-npm-install",
        action="store_true",
        help="Skip the npm install bootstrap even if node_modules is missing.",
    )
    parser.add_argument(
        "--browser-delay",
        type=float,
        default=4.0,
        help="Seconds to wait before opening browser tabs.",
    )
    return parser.parse_args()


def resolve_python() -> str:
    candidates: list[str] = []
    child_python = str(os.environ.get("FILTERLESS_CHILD_PYTHON", "") or "").strip()
    if child_python:
        candidates.append(child_python)
    env_python = str(os.environ.get("FILTERLESS_PYTHON", "") or "").strip()
    if env_python:
        candidates.append(env_python)
    if os.name == "nt":
        candidates.extend(
            [
                str(ROOT / ".venv" / "Scripts" / "python.exe"),
                str(ROOT / "venv" / "Scripts" / "python.exe"),
            ]
        )
    else:
        candidates.extend(
            [
                str(ROOT / ".venv" / "bin" / "python"),
                str(ROOT / "venv" / "bin" / "python"),
            ]
        )
    current_python = str(Path(sys.executable).expanduser())
    if current_python:
        candidates.append(current_python)
    path_python = shutil.which("python")
    if path_python:
        candidates.append(path_python)

    for candidate in candidates:
        if _python_candidate_usable(candidate):
            return candidate
    return sys.executable


def _python_candidate_usable(candidate: str) -> bool:
    text = str(candidate or "").strip()
    if not text:
        return False
    path = Path(text).expanduser()
    if not path.exists():
        return False
    try:
        result = subprocess.run(
            [str(path), "-V"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def resolve_npm() -> str:
    candidates = ["npm.cmd", "npm"] if os.name == "nt" else ["npm"]
    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    raise FileNotFoundError("npm was not found on PATH.")


def build_frontend_process(
    python_exe: str, skip_npm_install: bool
) -> tuple[str, list[str], Path, str]:
    try:
        npm_exe = resolve_npm()
    except FileNotFoundError:
        if not MONTE_CARLO_DIST_DIR.exists():
            raise FileNotFoundError(
                "npm was not found on PATH and the built Monte Carlo dist app is missing."
            )
        return (
            "Monte Carlo static server",
            [
                python_exe,
                str(STATIC_SERVER_SCRIPT),
                "--host",
                "127.0.0.1",
                "--port",
                str(VITE_PORT),
            ],
            ROOT,
            "static-dist",
        )

    ensure_node_modules(npm_exe, skip_npm_install)
    return (
        "Monte Carlo dev server",
        [
            npm_exe,
            "run",
            "dev",
            "--",
            "--host",
            "127.0.0.1",
            "--port",
            str(VITE_PORT),
            "--strictPort",
        ],
        MONTE_CARLO_DIR,
        "vite-dev",
    )


def ensure_paths() -> None:
    if not BOT_SCRIPT.exists():
        raise FileNotFoundError(f"Missing bot launcher: {BOT_SCRIPT}")
    if not BRIDGE_SCRIPT.exists():
        raise FileNotFoundError(f"Missing dashboard bridge: {BRIDGE_SCRIPT}")
    if not MONTE_CARLO_DIR.exists():
        raise FileNotFoundError(f"Missing Monte Carlo app directory: {MONTE_CARLO_DIR}")
    if not STATIC_SERVER_SCRIPT.exists():
        raise FileNotFoundError(f"Missing static server helper: {STATIC_SERVER_SCRIPT}")


def ensure_node_modules(npm_exe: str, skip_install: bool) -> None:
    node_modules = MONTE_CARLO_DIR / "node_modules"
    if node_modules.exists() or skip_install:
        return
    print("node_modules not found. Running `npm install` in Monte Carlo app...")
    subprocess.run([npm_exe, "install"], cwd=MONTE_CARLO_DIR, check=True)


def port_is_listening(port: int, host: str = "127.0.0.1") -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def process_match_tokens(name: str) -> list[str]:
    if name == "filterless bot":
        return ["launch_filterless_live.py"]
    if name == "filterless dashboard bridge":
        return ["filterless_dashboard_bridge.py"]
    if name == "Monte Carlo dev server":
        return ["vite", str(MONTE_CARLO_DIR).lower()]
    if name == "Monte Carlo static server":
        return ["filterless_static_server.py"]
    return [str(ROOT).lower()]


def load_workspace_processes() -> list[dict[str, object]]:
    if not WORKSPACE_PID_PATH.exists():
        return []
    try:
        payload = json.loads(WORKSPACE_PID_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    processes = payload.get("processes")
    if not isinstance(processes, list):
        return []
    return [entry for entry in processes if isinstance(entry, dict)]


def save_workspace_processes(processes: list[tuple[str, subprocess.Popen]]) -> None:
    WORKSPACE_PID_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "root": str(ROOT),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "processes": [
            {
                "name": name,
                "pid": process.pid,
                "match_tokens": process_match_tokens(name),
            }
            for name, process in processes
        ],
    }
    WORKSPACE_PID_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def clear_workspace_processes() -> None:
    try:
        WORKSPACE_PID_PATH.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass


def get_process_command_line(pid: int) -> str:
    if pid <= 0:
        return ""
    if os.name == "nt":
        query = (
            f"$p = Get-CimInstance Win32_Process -Filter \"ProcessId={pid}\"; "
            f"if ($null -ne $p) {{ [Console]::Out.Write(($p.CommandLine ?? '')) }}"
        )
        try:
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command", query],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return ""
        return (result.stdout or "").strip()
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return (result.stdout or "").strip()


def process_matches(pid: int, match_tokens: list[str]) -> bool:
    command_line = get_process_command_line(pid)
    if not command_line:
        return False
    lowered = command_line.lower()
    return all(token.lower() in lowered for token in match_tokens)


def terminate_process_tree(pid: int) -> None:
    if pid <= 0:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            text=True,
            check=False,
        )
        return
    try:
        os.kill(pid, 15)
    except OSError:
        pass


def cleanup_recorded_workspace_processes() -> None:
    recorded = load_workspace_processes()
    if not recorded:
        return
    for entry in recorded:
        try:
            pid = int(entry.get("pid") or 0)
        except (TypeError, ValueError):
            continue
        match_tokens = entry.get("match_tokens")
        if not isinstance(match_tokens, list):
            continue
        tokens = [str(token) for token in match_tokens if str(token).strip()]
        if tokens and process_matches(pid, tokens):
            terminate_process_tree(pid)
    clear_workspace_processes()


def cleanup_orphaned_workspace_processes() -> None:
    if os.name != "nt":
        return
    query = rf"""
$root = '{str(ROOT).replace("'", "''")}'.ToLower()
Get-CimInstance Win32_Process |
      Where-Object {{
        $_.ProcessId -ne {os.getpid()} -and
        $_.CommandLine -and
        (
          ($_.Name -like 'python*' -and $_.CommandLine.ToLower().Contains('launch_filterless_live.py')) -or
          ($_.Name -like 'python*' -and $_.CommandLine.ToLower().Contains('filterless_dashboard_bridge.py')) -or
          ($_.Name -eq 'node.exe' -and $_.CommandLine.ToLower().Contains('vite') -and $_.CommandLine.ToLower().Contains($root)) -or
          ($_.Name -like 'python*' -and $_.CommandLine.ToLower().Contains('filterless_static_server.py'))
        )
      }} |
  Select-Object -ExpandProperty ProcessId
"""
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", query],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return
    for line in (result.stdout or "").splitlines():
        try:
            pid = int(line.strip())
        except ValueError:
            continue
        terminate_process_tree(pid)


def ensure_workspace_slots() -> None:
    cleanup_recorded_workspace_processes()
    cleanup_orphaned_workspace_processes()
    if port_is_listening(VITE_PORT):
        raise RuntimeError(
            f"Port {VITE_PORT} is already in use. Close the existing localhost:{VITE_PORT} app and relaunch."
        )


def start_process(
    name: str,
    command: list[str],
    cwd: Path,
    *,
    stdout_path: Path | None = None,
    inherit_output: bool = False,
    extra_env: Optional[dict[str, str]] = None,
) -> tuple[subprocess.Popen, object | None]:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    if extra_env:
        env.update(extra_env)

    if inherit_output:
        process = subprocess.Popen(command, cwd=cwd, env=env)
        attach_process_to_workspace_job(name, process)
        return process, None

    handle = None
    if stdout_path is not None:
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        handle = stdout_path.open("a", encoding="utf-8")
        handle.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} | Starting {name} ===\n")
        handle.flush()

    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=handle if handle is not None else subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    attach_process_to_workspace_job(name, process)
    return process, handle


def open_browser_tabs(delay_seconds: float) -> None:
    def _wait_for_port(host: str, port: int, timeout_seconds: float) -> None:
        deadline = time.time() + max(0.0, timeout_seconds)
        while time.time() < deadline:
            try:
                with socket.create_connection((host, port), timeout=0.5):
                    return
            except OSError:
                time.sleep(0.5)

    def _open_url(url: str) -> None:
        try:
            if os.name == "nt":
                os.startfile(url)
                return
        except Exception:
            pass
        webbrowser.open(url, new=2)

    def _worker() -> None:
        time.sleep(max(0.0, delay_seconds))
        _wait_for_port("127.0.0.1", VITE_PORT, timeout_seconds=20.0)
        _open_url(MONTE_CARLO_URL)
        time.sleep(0.75)
        _open_url(FILTERLESS_URL)

    threading.Thread(target=_worker, daemon=True).start()


def stop_process(name: str, process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    print(f"Stopping {name}...")
    try:
        process.terminate()
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def file_age_seconds(path: Path) -> Optional[float]:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return max(0.0, time.time() - stat.st_mtime)


def record_managed_processes(managed: list[dict[str, object]]) -> None:
    payload = []
    for entry in managed:
        process = entry.get("process")
        if not isinstance(process, subprocess.Popen):
            continue
        payload.append((str(entry.get("name") or ""), process))
    save_workspace_processes(payload)


def restart_managed_process(entry: dict[str, object]) -> None:
    name = str(entry.get("name") or "process")
    process = entry.get("process")
    handle = entry.get("handle")
    if isinstance(process, subprocess.Popen):
        stop_process(name, process)
    if handle is not None:
        try:
            handle.close()
        except Exception:
            pass
    time.sleep(PROCESS_RESTART_DELAY_SECONDS)
    process, handle = start_process(
        name,
        [str(part) for part in entry.get("command", [])],
        Path(str(entry.get("cwd"))),
        stdout_path=Path(str(entry["stdout_path"])) if entry.get("stdout_path") else None,
        inherit_output=bool(entry.get("inherit_output")),
        extra_env=dict(entry.get("extra_env") or {}),
    )
    entry["process"] = process
    entry["handle"] = handle
    entry["launched_at"] = time.time()
    entry["restart_count"] = int(entry.get("restart_count") or 0) + 1


def choose_account_id(account_id_override: Optional[str]) -> Optional[str]:
    if account_id_override:
        selected = str(account_id_override).strip() or None
        if selected:
            log_workspace_status(f"Using account override: {selected}")
        return selected

    from config import CONFIG
    from client import ProjectXClient

    original_account_id = CONFIG.get("ACCOUNT_ID")
    CONFIG["ACCOUNT_ID"] = None

    try:
        client = ProjectXClient()
        client.login()
        log_workspace_status("Authenticated with ProjectX; prompting for account selection")
        print("\nSelect the Topstep account for this filterless workspace.\n")
        selected = client.fetch_accounts()
        if selected is None:
            log_workspace_status("Account selection returned no account")
            return None
        if isinstance(selected, list):
            selected_ids = [str(item).strip() for item in selected if str(item).strip()]
            if not selected_ids:
                log_workspace_status("Account selector returned an empty account list")
                return None
            chosen = selected_ids[0]
            log_workspace_status(
                f"Account selector returned multiple accounts; using first account {chosen}"
            )
            return chosen
        chosen = str(selected).strip() or None
        if chosen:
            log_workspace_status(f"Account selected: {chosen}")
        return chosen
    finally:
        CONFIG["ACCOUNT_ID"] = original_account_id


def main() -> int:
    args = parse_args()
    log_workspace_status("Workspace launch requested")
    ensure_paths()
    workspace_lock = acquire_singleton_lock(WORKSPACE_LOCK_PATH, name="filterless_workspace")
    if workspace_lock is None:
        existing = ""
        try:
            existing = WORKSPACE_LOCK_PATH.read_text(encoding="utf-8").strip()
        except OSError:
            existing = ""
        log_workspace_status(
            "Another filterless workspace instance is already running. "
            f"Lock: {WORKSPACE_LOCK_PATH}"
        )
        if existing:
            log_workspace_status(f"Existing lock payload: {existing}")
        return 1

    try:
        selected_account_id = choose_account_id(args.account_id)
    except Exception as exc:
        log_workspace_status(f"Failed to authenticate or select a Topstep account: {exc}")
        return 1
    if not selected_account_id:
        log_workspace_status("No Topstep account selected. Aborting launch.")
        return 1

    python_exe = resolve_python()
    log_workspace_status(f"Resolved Python: {python_exe}")
    try:
        frontend_name, frontend_command, frontend_cwd, frontend_mode = build_frontend_process(
            python_exe, args.skip_npm_install
        )
    except FileNotFoundError as exc:
        log_workspace_status(str(exc))
        return 1
    if frontend_mode == "vite-dev":
        log_workspace_status("Monte Carlo node_modules ready")
    try:
        log_workspace_status("Checking workspace slots and cleaning stale processes")
        ensure_workspace_slots()
        log_workspace_status("Workspace slot check passed")
    except RuntimeError as exc:
        log_workspace_status(str(exc))
        return 1

    managed: list[dict[str, object]] = []

    try:
        log_workspace_status("Starting filterless bot process")
        bot_process, bot_handle = start_process(
            "filterless bot",
            [python_exe, str(BOT_SCRIPT)],
            ROOT,
            stdout_path=BOT_LOG_PATH,
            extra_env={"JULIE_ACCOUNT_ID": selected_account_id},
        )
        log_workspace_status(f"Filterless bot started with pid={bot_process.pid}")
        managed.append(
            {
                "name": "filterless bot",
                "process": bot_process,
                "handle": bot_handle,
                "command": [python_exe, str(BOT_SCRIPT)],
                "cwd": ROOT,
                "stdout_path": BOT_LOG_PATH,
                "inherit_output": False,
                "extra_env": {"JULIE_ACCOUNT_ID": selected_account_id},
                "watch_path": BOT_LOG_PATH,
                "stale_seconds": BOT_STALE_TIMEOUT_SECONDS,
                "launched_at": time.time(),
                "restart_count": 0,
            }
        )
        record_managed_processes(managed)

        log_workspace_status("Starting filterless dashboard bridge process")
        bridge_process, bridge_handle = start_process(
            "filterless dashboard bridge",
            [python_exe, str(BRIDGE_SCRIPT), "--follow", "--poll-seconds", "1.0"],
            ROOT,
            stdout_path=BRIDGE_LOG_PATH,
        )
        log_workspace_status(f"Filterless dashboard bridge started with pid={bridge_process.pid}")
        managed.append(
            {
                "name": "filterless dashboard bridge",
                "process": bridge_process,
                "handle": bridge_handle,
                "command": [python_exe, str(BRIDGE_SCRIPT), "--follow", "--poll-seconds", "1.0"],
                "cwd": ROOT,
                "stdout_path": BRIDGE_LOG_PATH,
                "inherit_output": False,
                "extra_env": {},
                "watch_path": DASHBOARD_STATE_PATH,
                "stale_seconds": BRIDGE_STALE_TIMEOUT_SECONDS,
                "launched_at": time.time(),
                "restart_count": 0,
            }
        )
        record_managed_processes(managed)

        log_workspace_status(f"Starting {frontend_name.lower()} process")
        vite_process, vite_handle = start_process(
            frontend_name,
            frontend_command,
            frontend_cwd,
            stdout_path=FRONTEND_LOG_PATH,
        )
        log_workspace_status(f"{frontend_name} started with pid={vite_process.pid}")
        managed.append(
            {
                "name": frontend_name,
                "process": vite_process,
                "handle": vite_handle,
                "command": frontend_command,
                "cwd": frontend_cwd,
                "stdout_path": FRONTEND_LOG_PATH,
                "inherit_output": False,
                "extra_env": {},
                "watch_path": None,
                "stale_seconds": None,
                "launched_at": time.time(),
                "restart_count": 0,
            }
        )
        record_managed_processes(managed)
        log_workspace_status("All workspace processes launched")

        if not args.no_browser:
            open_browser_tabs(args.browser_delay)

        while True:
            restarted = False
            for entry in managed:
                name = str(entry.get("name") or "process")
                process = entry.get("process")
                if not isinstance(process, subprocess.Popen):
                    continue
                exit_code = process.poll()
                if exit_code is not None:
                    restart_count = int(entry.get("restart_count") or 0)
                    if restart_count < MAX_RESTARTS_PER_PROCESS:
                        print(f"{name} exited with code {exit_code}. Restarting ({restart_count + 1}/{MAX_RESTARTS_PER_PROCESS})...")
                        restart_managed_process(entry)
                        record_managed_processes(managed)
                        restarted = True
                        break
                    print(f"{name} exited with code {exit_code}. Restart limit reached; stopping remaining processes...")
                    return exit_code
            if restarted:
                continue

            now = time.time()
            for entry in managed:
                name = str(entry.get("name") or "process")
                process = entry.get("process")
                if not isinstance(process, subprocess.Popen) or process.poll() is not None:
                    continue
                watch_path = entry.get("watch_path")
                stale_seconds = entry.get("stale_seconds")
                launched_at = float(entry.get("launched_at") or now)
                if not isinstance(watch_path, Path) or stale_seconds in (None, 0):
                    continue
                stale_limit = float(stale_seconds)
                if now - launched_at < stale_limit:
                    continue
                age = file_age_seconds(watch_path)
                if age is None or age <= stale_limit:
                    continue
                restart_count = int(entry.get("restart_count") or 0)
                if restart_count >= MAX_RESTARTS_PER_PROCESS:
                    print(f"{name} watchdog detected stale output ({age:.0f}s) but restart limit is reached.")
                    return 1
                print(f"{name} watchdog detected stale output ({age:.0f}s). Restarting ({restart_count + 1}/{MAX_RESTARTS_PER_PROCESS})...")
                restart_managed_process(entry)
                record_managed_processes(managed)
                restarted = True
                break
            if restarted:
                continue
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nShutdown requested.")
        return 0
    finally:
        for entry in reversed(managed):
            process = entry.get("process")
            if isinstance(process, subprocess.Popen):
                stop_process(str(entry.get("name") or "process"), process)
        clear_workspace_processes()
        for entry in managed:
            handle = entry.get("handle")
            if handle is not None:
                try:
                    handle.close()
                except Exception:
                    pass
        close_workspace_job_handle()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        WORKSPACE_ERROR_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with WORKSPACE_ERROR_LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} | Workspace launcher crash ===\n")
            handle.write(traceback.format_exc())
            handle.write("\n")
        print(f"Workspace launcher crashed. See {WORKSPACE_ERROR_LOG_PATH}")
        raise SystemExit(1)
