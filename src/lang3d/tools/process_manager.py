"""FreeCAD GUI process lifecycle manager.

Provides graceful shutdown (WM_CLOSE first, force-kill as fallback),
PID tracking, and cleanup-on-exit guarantees.
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes
import logging
import subprocess
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)


def _is_pid_alive(pid: int) -> bool:
    """Check whether a process with the given PID is still running."""
    import sys

    if sys.platform != "win32":
        return False

    kernel32 = ctypes.windll.kernel32
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return False
    try:
        exit_code = ctypes.wintypes.DWORD()
        if kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return exit_code.value == 259  # STILL_ACTIVE
        return False
    finally:
        kernel32.CloseHandle(handle)


def _find_window_for_pid(pid: int) -> int | None:
    """Find the main window handle for a process by PID."""
    if not _is_pid_alive(pid):
        return None

    result_hwnd: list[int] = []

    def _enum_cb(hwnd: int, _lparam: Any) -> bool:
        window_pid = ctypes.wintypes.DWORD()
        ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(window_pid))
        if window_pid.value == pid and ctypes.windll.user32.IsWindowVisible(hwnd):
            result_hwnd.append(hwnd)
        return True

    try:
        WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
        ctypes.windll.user32.EnumWindows(WNDENUMPROC(_enum_cb), 0)
    except Exception:
        pass

    return result_hwnd[0] if result_hwnd else None


def _graceful_kill(pid: int, timeout: float = 5.0) -> bool:
    """Attempt graceful shutdown of a process.

    1. Send WM_CLOSE to the main window.
    2. Wait up to *timeout* seconds for the process to exit.
    3. Force-kill if still alive.

    Returns True if the process exited gracefully (within timeout).
    """
    WM_CLOSE = 0x0010

    if not _is_pid_alive(pid):
        return True

    # Step 1: Send WM_CLOSE
    hwnd = _find_window_for_pid(pid)
    if hwnd:
        ctypes.windll.user32.PostMessageW(hwnd, WM_CLOSE, 0, 0)

    # Step 2: Wait for graceful exit
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _is_pid_alive(pid):
            return True
        time.sleep(0.1)

    # Step 3: Force kill
    logger.warning("Force-killing PID %d after %.1fs timeout", pid, timeout)
    try:
        subprocess.run(
            ["taskkill", "/F", "/PID", str(pid)],
            capture_output=True,
            timeout=5,
            encoding="utf-8",
            errors="replace",
        )
    except Exception:
        pass

    return False


class FreeCADProcessManager:
    """Manages FreeCAD GUI process lifecycle."""

    def __init__(self) -> None:
        self._gui_pid: int | None = None
        self._gui_proc: subprocess.Popen | None = None
        self._lock = threading.RLock()

    def launch_gui(self, cmd: list[str]) -> subprocess.Popen:
        """Launch FreeCAD GUI, gracefully closing any existing instance first."""
        with self._lock:
            self.kill_existing()

            proc = subprocess.Popen(cmd)
            self._gui_pid = proc.pid
            self._gui_proc = proc
            logger.info("Launched FreeCAD GUI (PID %d)", proc.pid)
            return proc

    def kill_existing(self, timeout: float = 5.0) -> bool:
        """Gracefully stop the managed FreeCAD process.

        Returns True if the process was already gone or exited gracefully.
        """
        with self._lock:
            pid = self._gui_pid
            proc = self._gui_proc

            if pid is not None and _is_pid_alive(pid):
                graceful = _graceful_kill(pid, timeout=timeout)
                logger.info("FreeCAD PID %d killed (graceful=%s)", pid, graceful)

            self._gui_pid = None
            self._gui_proc = None

            # Also try to kill any leftover FreeCAD.exe processes
            try:
                result = subprocess.run(
                    ["taskkill", "/F", "/IM", "FreeCAD.exe"],
                    capture_output=True,
                    timeout=5,
                    encoding="utf-8",
                    errors="replace",
                )
                if result.returncode == 0:
                    logger.info("Cleaned up remaining FreeCAD.exe processes")
            except Exception:
                pass

            return True

    def is_running(self) -> bool:
        """Check if the managed process is still alive."""
        with self._lock:
            if self._gui_pid is None:
                return False
            return _is_pid_alive(self._gui_pid)

    @property
    def pid(self) -> int | None:
        """Return the PID of the managed process, or None."""
        return self._gui_pid

    def cleanup(self) -> None:
        """Ensure no zombie processes on module unload."""
        self.kill_existing(timeout=3.0)


# Module-level singleton
_process_manager = FreeCADProcessManager()
