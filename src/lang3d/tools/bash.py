"""Shell command execution tool."""

from __future__ import annotations

import re
import subprocess
import sys
from typing import Any

from ..models.base import ToolDefinition
from .base import Tool

# Dangerous command patterns that should be blocked
_DANGEROUS_PATTERNS = [
    r"\brm\s+-rf\s+/",
    r"\brm\s+-rf\s+[A-Za-z]:",
    r"\bformat\s+[A-Za-z]:",
    r"\bdd\s+if=",
    r"\bmkfs\b",
    r"\bshutdown\b",
    r"\breboot\b",
    r"\bhalt\b",
    r"\bpowershell\b.*-command.*Remove-Item",
    r"\breg\s+(delete|add)\b",
    r"\bnet\s+(user|localgroup)\b",
    r"\btaskkill\s+/f\s+/pid\s+0\b",
    r">/dev/sd",
    r"\bsudo\s+rm\b",
]


def _is_dangerous_command(command: str) -> str | None:
    """Check if a command matches a dangerous pattern. Returns the matched pattern or None."""
    cmd_lower = command.lower()
    for pattern in _DANGEROUS_PATTERNS:
        if re.search(pattern, cmd_lower):
            return pattern
    return None


# Dangerous Python modules that should not be importable in python_exec
_BLOCKED_MODULES = {
    "subprocess", "os.system", "shutil.rmtree",
    "ctypes", "winreg", "__import__", "importlib",
    "eval(", "exec(", "compile(",
}


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
        # Security: block dangerous commands
        dangerous = _is_dangerous_command(command)
        if dangerous:
            return f"Error: Command blocked (matches dangerous pattern: {dangerous})"
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
        # Security: block dangerous imports and patterns
        for mod in _BLOCKED_MODULES:
            if re.search(rf'\b{re.escape(mod)}\b', code):
                return f"Error: Code contains blocked module/reference: {mod}"
        # Block getattr with dunder access
        if re.search(r'getattr\s*\([^)]*[\'"]__\w+', code):
            return "Error: Code contains blocked pattern: getattr with dunder"
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
