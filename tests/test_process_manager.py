"""Tests for FreeCAD process manager."""

from __future__ import annotations

import subprocess
import time
from unittest.mock import MagicMock, patch, call

import pytest


# ---------------------------------------------------------------------------
# _is_pid_alive
# ---------------------------------------------------------------------------

class TestIsPidAlive:
    def test_current_process_alive(self):
        from lang3d.tools.process_manager import _is_pid_alive
        import os
        assert _is_pid_alive(os.getpid()) is True

    def test_nonexistent_pid(self):
        from lang3d.tools.process_manager import _is_pid_alive
        assert _is_pid_alive(99999999) is False


# ---------------------------------------------------------------------------
# _find_window_for_pid
# ---------------------------------------------------------------------------

class TestFindWindowForPid:
    def test_dead_pid_returns_none(self):
        from lang3d.tools.process_manager import _find_window_for_pid
        result = _find_window_for_pid(99999999)
        assert result is None


# ---------------------------------------------------------------------------
# _graceful_kill
# ---------------------------------------------------------------------------

class TestGracefulKill:
    def test_already_dead_pid(self):
        from lang3d.tools.process_manager import _graceful_kill
        result = _graceful_kill(99999999, timeout=0.1)
        assert result is True

    def test_graceful_shutdown_with_wm_close(self):
        from lang3d.tools.process_manager import _graceful_kill
        with patch("lang3d.tools.process_manager._find_window_for_pid", return_value=12345), \
             patch("lang3d.tools.process_manager._is_pid_alive", side_effect=[True, True, False]), \
             patch("lang3d.tools.process_manager.subprocess.run"), \
             patch("lang3d.tools.process_manager.ctypes.windll"):
            result = _graceful_kill(100, timeout=1.0)
            assert result is True

    def test_force_kill_after_timeout(self):
        from lang3d.tools.process_manager import _graceful_kill
        with patch("lang3d.tools.process_manager._find_window_for_pid", return_value=None), \
             patch("lang3d.tools.process_manager._is_pid_alive", return_value=True), \
             patch("lang3d.tools.process_manager.subprocess.run") as mock_run:
            result = _graceful_kill(100, timeout=0.1)
            mock_run.assert_called()
            assert result is False


# ---------------------------------------------------------------------------
# FreeCADProcessManager
# ---------------------------------------------------------------------------

class TestFreeCADProcessManager:
    def test_initial_state(self):
        from lang3d.tools.process_manager import FreeCADProcessManager
        mgr = FreeCADProcessManager()
        assert mgr.pid is None
        assert mgr.is_running() is False

    def test_launch_gui(self):
        from lang3d.tools.process_manager import FreeCADProcessManager
        with patch("lang3d.tools.process_manager._is_pid_alive", return_value=False), \
             patch("lang3d.tools.process_manager.subprocess.Popen") as mock_popen, \
             patch("lang3d.tools.process_manager.subprocess.run") as mock_run, \
             patch("lang3d.tools.process_manager._graceful_kill"):
            mock_proc = MagicMock()
            mock_proc.pid = 1234
            mock_popen.return_value = mock_proc
            mock_run.return_value = MagicMock(returncode=1)

            mgr = FreeCADProcessManager()
            proc = mgr.launch_gui(["freecad.exe"])

            assert proc.pid == 1234
            assert mgr.pid == 1234
            mock_popen.assert_called_once_with(["freecad.exe"])

    def test_is_running_after_launch(self):
        from lang3d.tools.process_manager import FreeCADProcessManager
        mgr = FreeCADProcessManager()
        mgr._gui_pid = 1234

        with patch("lang3d.tools.process_manager._is_pid_alive", return_value=True):
            assert mgr.is_running() is True

        with patch("lang3d.tools.process_manager._is_pid_alive", return_value=False):
            assert mgr.is_running() is False

    def test_kill_existing(self):
        from lang3d.tools.process_manager import FreeCADProcessManager
        with patch("lang3d.tools.process_manager._is_pid_alive", return_value=True), \
             patch("lang3d.tools.process_manager._graceful_kill", return_value=True) as mock_graceful, \
             patch("lang3d.tools.process_manager.subprocess.run", return_value=MagicMock(returncode=0)):
            mgr = FreeCADProcessManager()
            mgr._gui_pid = 1234
            mgr._gui_proc = MagicMock()

            result = mgr.kill_existing()

            assert mgr.pid is None
            mock_graceful.assert_called_once_with(1234, timeout=5.0)

    def test_kill_existing_no_process(self):
        from lang3d.tools.process_manager import FreeCADProcessManager
        with patch("lang3d.tools.process_manager.subprocess.run", return_value=MagicMock(returncode=1)):
            mgr = FreeCADProcessManager()
            result = mgr.kill_existing()
            assert result is True

    def test_cleanup_calls_kill(self):
        from lang3d.tools.process_manager import FreeCADProcessManager
        mgr = FreeCADProcessManager()
        with patch("lang3d.tools.process_manager.subprocess.run", return_value=MagicMock(returncode=1)), \
             patch.object(mgr, "kill_existing") as mock_kill:
            mgr.cleanup()
            mock_kill.assert_called_once_with(timeout=3.0)
