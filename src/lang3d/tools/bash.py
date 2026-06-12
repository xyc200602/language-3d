"""Shell command execution tool."""

from __future__ import annotations

import ast
import logging
import re
import subprocess
import sys
from typing import Any

from ..models.base import ToolDefinition
from .base import Tool

logger = logging.getLogger(__name__)

# Dangerous command patterns — expanded to cover common bypasses
_DANGEROUS_PATTERNS = [
    # Recursive / forced deletion
    r"\brm\s+.*-[rR].*[fF]",
    r"\brm\s+.*-[fF].*[rR]",
    r"\brd\s+/[sS]\s+/[qQ]",
    # Disk / filesystem destruction
    r"\bformat\s+[A-Za-z]:",
    r"\bdd\s+if=",
    r"\bmkfs\b",
    # System control
    r"\bshutdown\b",
    r"\breboot\b",
    r"\bhalt\b",
    r"\binit\s+[06]\b",
    # Windows registry / user management
    r"\bpowershell\b.*-command.*Remove-Item",
    r"\breg\s+(delete|add)\b",
    r"\bnet\s+(user|localgroup)\b",
    r"\btaskkill\s+/f\s+/pid\s+0\b",
    # Device access
    r">/dev/sd",
    # Privilege escalation
    r"\bsudo\s+",
    r"\bsu\s+",
    r"\brunas\s+",
    # Shell injection / download-and-execute
    r"\bcurl\b.*\|\s*(bash|sh|python|python3)\b",
    r"\bwget\b.*\|\s*(bash|sh|python|python3)\b",
    r"\b(base64|xxd)\s+.*\|\s*(bash|sh)\b",
    # Reverse shells
    r"/dev/tcp/",
    r"\bnc\s+.*-e\b",
    r"\bncat\b",
    # Writing to system paths
    r">\s*/etc/",
    r">\s*/boot/",
    r">\s*/proc/",
    r">\s*/sys/",
    # Pipe chain with destructive commands
    r"\|\s*(bash|sh|zsh|fish)\s*$",
]


def _is_dangerous_command(command: str) -> str | None:
    """Check if a command matches a dangerous pattern. Returns the matched pattern or None."""
    cmd_lower = command.lower()
    for pattern in _DANGEROUS_PATTERNS:
        if re.search(pattern, cmd_lower):
            return pattern
    return None


# --- Python code safety via AST analysis ---

def _validate_python_code(code: str) -> list[str]:
    """Parse Python code AST and check for dangerous constructs.

    Returns a list of error strings. Empty list means safe.
    """
    errors: list[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"Syntax error: {e}"]

    _BLOCKED_IMPORTS = {
        "subprocess", "os", "shutil", "ctypes", "winreg",
        "importlib", "socket", "http", "urllib", "requests",
        "telnetlib", "ftplib", "smtplib", "xmlrpc",
    }
    _BLOCKED_BUILTINS = {
        "exec", "eval", "compile", "__import__",
        "breakpoint", "exit", "quit",
    }

    for node in ast.walk(tree):
        # Block import statements for dangerous modules
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_module = alias.name.split(".")[0]
                if root_module in _BLOCKED_IMPORTS:
                    errors.append(f"Blocked import: {alias.name}")
        if isinstance(node, ast.ImportFrom):
            if node.module:
                root_module = node.module.split(".")[0]
                if root_module in _BLOCKED_IMPORTS:
                    errors.append(f"Blocked import from: {node.module}")

        # Block dangerous builtins
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in _BLOCKED_BUILTINS:
                    errors.append(f"Blocked builtin call: {node.func.id}()")
            # Block method calls on dangerous modules (e.g., os.system(...))
            if isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name):
                    if node.func.value.id in _BLOCKED_IMPORTS:
                        errors.append(f"Blocked call: {node.func.value.id}.{node.func.attr}()")

        # Block dunder attribute access (e.g., __builtins__, __import__)
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__") and node.attr.endswith("__"):
                errors.append(f"Blocked dunder access: .{node.attr}")

        # Block dunder string-key lookups (e.g., d["__builtins__"])
        if isinstance(node, ast.Subscript):
            if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
                key = node.slice.value
                if key.startswith("__") and key.endswith("__"):
                    errors.append(f"Blocked dunder key access: {key}")

    return errors


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
            logger.warning("Blocked dangerous command (pattern: %s): %s", dangerous, command[:100])
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
        # Security: AST-based validation
        errors = _validate_python_code(code)
        if errors:
            logger.warning("Blocked unsafe Python code: %s", "; ".join(errors[:3]))
            return f"Error: Code contains blocked patterns: {'; '.join(errors[:5])}"
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
