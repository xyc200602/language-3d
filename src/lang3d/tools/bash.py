"""Shell command execution tool."""

from __future__ import annotations

import subprocess
import sys
from typing import Any

from ..models.base import ToolDefinition
from .base import Tool


class BashTool(Tool):
    name = "bash"
    description = "Execute a shell command and return its output"

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default 120)",
                    },
                    "cwd": {
                        "type": "string",
                        "description": "Working directory for the command",
                    },
                },
                "required": ["command"],
            },
        )

    def execute(
        self,
        *,
        command: str,
        timeout: int = 120,
        cwd: str | None = None,
        **kwargs: Any,
    ) -> str:
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
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

            # Truncate very long output
            if len(output) > 30000:
                output = output[:30000] + "\n... (output truncated)"

            return output or "(no output)"
        except subprocess.TimeoutExpired:
            return f"Error: Command timed out after {timeout} seconds"
        except Exception as e:
            return f"Error executing command: {e}"


class PythonExecTool(Tool):
    """Execute Python code in a subprocess."""

    name = "python_exec"
    description = "Execute Python code and return the output"

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default 30)",
                    },
                },
                "required": ["code"],
            },
        )

    def execute(self, *, code: str, timeout: int = 30, **kwargs: Any) -> str:
        try:
            result = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                timeout=timeout,
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
            return f"Error: Code execution timed out after {timeout} seconds"
        except Exception as e:
            return f"Error: {e}"


def register_bash_tools(registry: Any) -> None:
    """Register bash and python exec tools."""
    registry.register(BashTool())
    registry.register(PythonExecTool())
