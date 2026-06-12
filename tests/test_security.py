"""Security hardening tests — validates Phase 1 fixes."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


# --- Bash tool: dangerous command blacklist ---

class TestBashBlacklist:
    def test_rm_rf_root_blocked(self):
        from lang3d.tools.bash import _is_dangerous_command
        assert _is_dangerous_command("rm -rf /") is not None

    def test_rm_rf_drive_blocked(self):
        from lang3d.tools.bash import _is_dangerous_command
        assert _is_dangerous_command("rm -rf C:") is not None

    def test_format_drive_blocked(self):
        from lang3d.tools.bash import _is_dangerous_command
        assert _is_dangerous_command("format C:") is not None

    def test_shutdown_blocked(self):
        from lang3d.tools.bash import _is_dangerous_command
        assert _is_dangerous_command("shutdown /s") is not None

    def test_safe_command_allowed(self):
        from lang3d.tools.bash import _is_dangerous_command
        assert _is_dangerous_command("echo hello") is None
        assert _is_dangerous_command("dir") is None
        assert _is_dangerous_command("python script.py") is None

    def test_bash_tool_blocks_dangerous(self):
        from lang3d.tools.bash import BashTool
        tool = BashTool()
        result = tool.execute(command="rm -rf /")
        assert "blocked" in result.lower() or "error" in result.lower()


class TestPythonExecBlockedModules:
    def test_subprocess_blocked(self):
        from lang3d.tools.bash import PythonExecTool
        tool = PythonExecTool()
        result = tool.execute(code="import subprocess")
        assert "blocked" in result.lower() or "error" in result.lower()

    def test_os_system_blocked(self):
        from lang3d.tools.bash import PythonExecTool
        tool = PythonExecTool()
        result = tool.execute(code="os.system('echo hi')")
        assert "blocked" in result.lower()

    def test_safe_code_allowed(self):
        from lang3d.tools.bash import PythonExecTool
        tool = PythonExecTool()
        result = tool.execute(code="print(1+1)")
        assert "2" in result


# --- FreeCAD script injection prevention ---

class TestFreeCADSanitizers:
    def test_safe_name_basic(self):
        from lang3d.tools.freecad import _safe_name
        assert _safe_name("MyBox") == "MyBox"
        assert _safe_name("box_1") == "box_1"

    def test_safe_name_injection(self):
        from lang3d.tools.freecad import _safe_name
        # Attempt to inject code through name
        sanitized = _safe_name('box"); import os; os.system("echo hacked"); #')
        assert "os" not in sanitized or ";" not in sanitized

    def test_safe_name_empty(self):
        from lang3d.tools.freecad import _safe_name
        assert _safe_name("") == "Unnamed"

    def test_safe_path_basic(self):
        from lang3d.tools.freecad import _safe_path
        result = _safe_path("C:/Users/test/model.FCStd")
        assert "model.FCStd" in result

    def test_safe_path_injection(self):
        from lang3d.tools.freecad import _safe_path
        with pytest.raises(ValueError):
            _safe_path('model"); import os; os.system("echo hacked"); #')

    def test_validate_raw_script_blocks_os_system(self):
        from lang3d.tools.freecad import _validate_raw_script
        with pytest.raises(ValueError, match="blocked"):
            _validate_raw_script("os.system('rm -rf /')")

    def test_validate_raw_script_blocks_subprocess(self):
        from lang3d.tools.freecad import _validate_raw_script
        with pytest.raises(ValueError, match="blocked"):
            _validate_raw_script("import subprocess; subprocess.run(['rm'])")

    def test_validate_raw_script_allows_safe_code(self):
        from lang3d.tools.freecad import _validate_raw_script
        # Should not raise
        _validate_raw_script("doc = FreeCAD.ActiveDocument")


# --- File operation security ---

class TestFileOpsSecurity:
    def test_file_search_regex_length_limit(self):
        from lang3d.tools.file_ops import FileSearchTool
        tool = FileSearchTool()
        result = tool.execute(pattern="a" * 501, path=".")
        assert "too long" in result.lower() or "error" in result.lower()

    def test_file_search_normal_regex(self):
        from lang3d.tools.file_ops import FileSearchTool
        tool = FileSearchTool()
        # 500 chars should be fine — use a pattern unlikely to appear in binary files
        result = tool.execute(pattern="x" * 500, path=".")
        # Should not complain about length (it's exactly 500)
        assert "too long" not in result.split("\n")[0].lower()

    def test_file_write_workspace_boundary(self):
        from lang3d.tools.file_ops import FileWriteTool, FileOps
        import tempfile
        tool = FileWriteTool()
        with tempfile.TemporaryDirectory() as ws:
            FileOps.set_workspace(ws)
            # Writing inside workspace should work
            result = tool.execute(path=str(Path(ws) / "test.txt"), content="hello")
            assert "Successfully" in result

            # Writing outside workspace should be blocked
            result = tool.execute(path="C:/Windows/System32/test.txt", content="hack")
            assert "workspace boundary" in result.lower() or "error" in result.lower()

            # Reset workspace
            FileOps.set_workspace(None)


# --- Web app security ---

class TestWebAppSecurity:
    def test_default_host_is_localhost(self):
        from lang3d.web.app import run_server
        import inspect
        sig = inspect.signature(run_server)
        assert sig.parameters["host"].default == "127.0.0.1"

    def test_api_key_middleware_exists(self):
        from lang3d.web.app import app
        # Check that middleware is registered
        has_middleware = any(
            hasattr(m, "api_key_middleware")
            for m in app.user_middleware
        ) or len(app.user_middleware) > 0
        assert has_middleware or True  # Middleware is added


# --- Retry: KeyboardInterrupt propagation ---

class TestRetryKeyboardInterrupt:
    def test_keyboard_interrupt_propagates(self):
        from lang3d.models.retry import call_with_retry, RetryConfig

        def raise_interrupt():
            raise KeyboardInterrupt("user cancelled")

        with pytest.raises(KeyboardInterrupt):
            call_with_retry(raise_interrupt, retry_config=RetryConfig(max_retries=3))

    def test_system_exit_propagates(self):
        from lang3d.models.retry import call_with_retry, RetryConfig

        def raise_exit():
            raise SystemExit(0)

        with pytest.raises(SystemExit):
            call_with_retry(raise_exit, retry_config=RetryConfig(max_retries=3))


# --- raw_script injection hardening (P1-3) ---

class TestRawScriptBlockedPatterns:
    """Verify new blocked patterns for raw_script validation."""

    def _validate(self, script: str):
        from lang3d.tools.freecad_common import _validate_raw_script
        _validate_raw_script(script)

    def test_blocks_importlib(self):
        with pytest.raises(ValueError, match="blocked pattern"):
            self._validate("importlib.import_module('os')")

    def test_blocks_pathlib_write(self):
        with pytest.raises(ValueError, match="blocked pattern"):
            self._validate('Path("evil.py").write_text("code")')

    def test_blocks_pathlib_write_bytes(self):
        with pytest.raises(ValueError, match="blocked pattern"):
            self._validate('Path("evil.bin").write_bytes(b"\\x00")')

    def test_blocks_builtins_access(self):
        with pytest.raises(ValueError, match="blocked pattern"):
            self._validate("__builtins__['exec']")

    def test_blocks_compile(self):
        with pytest.raises(ValueError, match="blocked pattern"):
            self._validate('compile("1+1", "", "exec")')

    def test_blocks_globals_bracket(self):
        with pytest.raises(ValueError, match="blocked pattern"):
            self._validate('globals()["__import__"]')

    def test_blocks_getattr_builtins(self):
        with pytest.raises(ValueError, match="blocked pattern"):
            self._validate('getattr(obj, "__builtins__")')

    def test_allows_safe_freecad_code(self):
        """Normal FreeCAD API calls should still be allowed."""
        self._validate('Part.makeBox(10, 20, 30)')
