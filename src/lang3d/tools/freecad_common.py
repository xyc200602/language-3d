"""Shared FreeCAD utilities used across multiple tool modules.

Extracts common functions:
- _find_freecad_python(): Locate the FreeCAD Python executable
- _run_freecad_script(): Execute a FreeCAD Python script via subprocess
- _safe_name(): Sanitize FreeCAD object/document names
- _safe_path(): Sanitize file paths for script generation
- _validate_raw_script(): Block dangerous patterns in raw scripts
"""

from __future__ import annotations

import re as _re
import subprocess
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Security helpers
# ---------------------------------------------------------------------------

def _safe_name(name: str) -> str:
    """Sanitize a FreeCAD object/document name to prevent script injection.

    Only allows alphanumeric, underscore, hyphen, and CJK characters.
    """
    if not name:
        return "Unnamed"
    sanitized = _re.sub(r"[^A-Za-z0-9_\-]", "_", name)
    if not sanitized or sanitized.startswith("_"):
        sanitized = "obj_" + sanitized
    return sanitized[:64]


def _safe_path(path: str) -> str:
    """Sanitize a file path to prevent script injection in f-string contexts.

    Rejects paths containing quotes or other dangerous characters.
    NOTE: The returned path is used inside r"..." raw strings in generated
    scripts, so backslashes must NOT be escaped (r-strings treat them literally).
    """
    p = str(path)
    if any(c in p for c in ('"', "'", "\n", "\r", ";", "`", "$")):
        raise ValueError(f"Path contains dangerous characters: {path!r}")
    return p


def _validate_raw_script(script: str) -> None:
    """Validate a raw_script operation to block dangerous FreeCAD API calls."""
    blocked_patterns = [
        r"\bos\.system\b",
        r"\bsubprocess\b",
        r"\bexec\s*\(",
        r"\beval\s*\(",
        r"\b__import__\b",
        r"\bopen\s*\([^)]*['\"]w",
        r"\bshutil\b",
        r"\bctypes\b",
        r"\bimportlib\b",
        r"\bPath\s*\([^)]*\)\s*\.write_text\b",
        r"\bPath\s*\([^)]*\)\s*\.write_bytes\b",
        r"\b__builtins__\b",
        r"\bcompile\s*\(",
        r"\bglobals\s*\(\s*\)\s*\[",
        r"\bgetattr\s*\([^)]*__builtins__",
    ]
    for pat in blocked_patterns:
        if _re.search(pat, script):
            raise ValueError(f"Raw script contains blocked pattern: {pat}")


# ---------------------------------------------------------------------------
# FreeCAD Python locator
# ---------------------------------------------------------------------------

_FREECAD_PYTHON: str | None = None


def _find_freecad_python() -> str | None:
    """Locate the FreeCAD Python executable.

    Searches:
    1. Cached result
    2. LANG3D_FREECAD_PYTHON env var
    3. Common install paths (Windows)
    4. PATH for FreeCAD-python
    """
    global _FREECAD_PYTHON
    if _FREECAD_PYTHON is not None:
        return _FREECAD_PYTHON

    import os

    # Check env var
    env_path = os.environ.get("LANG3D_FREECAD_PYTHON")
    if env_path and Path(env_path).exists():
        _FREECAD_PYTHON = env_path
        return env_path

    # Common Windows paths
    if sys.platform == "win32":
        search_paths = [
            Path("C:/Program Files/FreeCAD 1.0/bin/python.exe"),
            Path("C:/Program Files/FreeCAD 0.21/bin/python.exe"),
            Path("C:/Program Files/FreeCAD/bin/python.exe"),
            Path("C:/Program Files (x86)/FreeCAD/bin/python.exe"),
        ]
        # Also try to find FreeCAD dynamically in system + user install dirs.
        # FreeCAD 1.1 defaults to a per-user install under %LOCALAPPDATA%\Programs,
        # not C:\Program Files — without this glob the finder misses it.
        for base_env in ("PROGRAMFILES", "LOCALAPPDATA"):
            base = Path(os.environ.get(base_env, ""))
            if base.exists():
                for p in base.glob("Programs/FreeCAD*/bin/python.exe"):
                    search_paths.insert(0, p)
                for p in base.glob("FreeCAD*/bin/python.exe"):
                    search_paths.insert(0, p)

        for p in search_paths:
            if p.exists():
                _FREECAD_PYTHON = str(p)
                return _FREECAD_PYTHON

    # Linux/macOS: FreeCAD ships its Python interpreter at known paths.
    # Ubuntu/Debian package:  /usr/lib/freecad/bin/python  (or via `freecad -c`)
    # AppImage:               extracted /opt/FreeCAD*/bin/python
    # macOS:                  /Applications/FreeCAD.app/Contents/bin/python
    if sys.platform != "win32":
        linux_paths = [
            Path("/usr/lib/freecad/bin/python"),
            Path("/usr/lib/freecad-daily/bin/python"),
            Path("/usr/bin/python3"),  # if FreeCAD Python module is importable
        ]
        for p in linux_paths:
            if p.exists():
                _FREECAD_PYTHON = str(p)
                return _FREECAD_PYTHON
        # macOS app bundle
        for p in Path("/Applications").glob("FreeCAD*.app/Contents/bin/python"):
            if p.exists():
                _FREECAD_PYTHON = str(p)
                return _FREECAD_PYTHON

    return None


def _run_freecad_script(
    script_path: str,
    timeout: int = 120,
) -> str:
    """Execute a FreeCAD Python script via subprocess.

    Args:
        script_path: Path to the Python script to execute.
        timeout: Maximum execution time in seconds.

    Returns:
        stdout from the script execution.

    Raises:
        RuntimeError: If FreeCAD Python is not found or script fails.
    """
    fc_python = _find_freecad_python()
    if fc_python is None:
        raise RuntimeError(
            "FreeCAD Python not found. Set LANG3D_FREECAD_PYTHON or install FreeCAD."
        )

    result = subprocess.run(
        [fc_python, script_path],
        capture_output=True,
        text=True,
        timeout=timeout,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        error_msg = result.stderr or result.stdout or "Unknown error"
        raise RuntimeError(f"FreeCAD script error:\n{error_msg}")
    return result.stdout.strip()
