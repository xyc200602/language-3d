"""Python code sandbox execution tool."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from ..models.base import ToolDefinition
from .base import Tool


class PythonScriptTool(Tool):
    """Execute a Python script file."""

    name = "python_script"
    description = "Run a Python script file"

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the Python script to execute",
                    },
                    "args": {
                        "type": "string",
                        "description": "Command line arguments (optional)",
                    },
                },
                "required": ["path"],
            },
        )

    def execute(self, *, path: str, args: str = "", timeout: int = 60, **kwargs: Any) -> str:
        try:
            p = Path(path)
            if not p.exists():
                return f"Error: Script not found: {path}"

            cmd = [sys.executable, str(p)]
            if args:
                cmd.extend(args.split())

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                encoding="utf-8",
                errors="replace",
            )

            output = ""
            if result.stdout:
                output += result.stdout
            if result.stderr:
                if output:
                    output += "\n"
                output += f"STDERR:\n{result.stderr}"

            if result.returncode != 0:
                output += f"\nExit code: {result.returncode}"

            return output or "(no output)"
        except subprocess.TimeoutExpired:
            return f"Error: Script timed out after {timeout} seconds"
        except Exception as e:
            return f"Error: {e}"


def register_python_tools(registry: Any) -> None:
    registry.register(PythonScriptTool())
