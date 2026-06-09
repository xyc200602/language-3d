"""File operation tools."""

from __future__ import annotations

import glob as glob_module
import os
from pathlib import Path
from typing import Any

from ..models.base import ToolDefinition
from .base import Tool


class FileReadTool(Tool):
    name = "file_read"
    description = "Read the contents of a file"

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the file to read",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Line number to start reading from (0-based, optional)",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of lines to read (optional)",
                    },
                },
                "required": ["path"],
            },
        )

    def execute(self, *, path: str, offset: int = 0, limit: int = 2000, **kwargs: Any) -> str:
        try:
            p = Path(path)
            if not p.exists():
                return f"Error: File not found: {path}"
            if p.is_dir():
                return f"Error: Path is a directory: {path}"

            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
            total = len(lines)

            if offset < 0:
                offset = max(0, total + offset)

            selected = lines[offset : offset + limit]
            numbered = [f"{offset + i + 1:6d}\t{line}" for i, line in enumerate(selected)]

            result = "\n".join(numbered)
            if offset + limit < total:
                result += f"\n... ({total - offset - limit} more lines)"
            return result
        except Exception as e:
            return f"Error reading file: {e}"


class FileWriteTool(Tool):
    name = "file_write"
    description = "Write content to a file (creates parent directories if needed)"

    # Workspace boundary — if set, writes are restricted to this directory tree.
    _workspace: Path | None = None

    @classmethod
    def set_workspace(cls, workspace: str | Path | None) -> None:
        """Set the allowed workspace root. Writes outside this tree are rejected."""
        cls._workspace = Path(workspace).resolve() if workspace else None

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the file to write",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file",
                    },
                },
                "required": ["path", "content"],
            },
        )

    def execute(self, *, path: str, content: str, **kwargs: Any) -> str:
        try:
            p = Path(path).resolve()
            # Workspace boundary check
            if self._workspace is not None:
                try:
                    p.relative_to(self._workspace)
                except ValueError:
                    return f"Error: Path escapes workspace boundary: {path}"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return f"Successfully wrote {len(content)} characters to {path}"
        except Exception as e:
            return f"Error writing file: {e}"


class FileEditTool(Tool):
    name = "file_edit"
    description = "Replace exact text in a file"

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the file to edit",
                    },
                    "old_text": {
                        "type": "string",
                        "description": "Exact text to find and replace",
                    },
                    "new_text": {
                        "type": "string",
                        "description": "Replacement text",
                    },
                },
                "required": ["path", "old_text", "new_text"],
            },
        )

    def execute(self, *, path: str, old_text: str, new_text: str, **kwargs: Any) -> str:
        try:
            p = Path(path)
            if not p.exists():
                return f"Error: File not found: {path}"

            content = p.read_text(encoding="utf-8")
            count = content.count(old_text)
            if count == 0:
                return f"Error: Text not found in file"
            if count > 1:
                return f"Error: Text found {count} times, expected exactly 1. Please provide more context."

            new_content = content.replace(old_text, new_text, 1)
            p.write_text(new_content, encoding="utf-8")
            return f"Successfully replaced text in {path}"
        except Exception as e:
            return f"Error editing file: {e}"


class FileSearchTool(Tool):
    name = "file_search"
    description = "Search for a pattern in files (like grep)"

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search for",
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory to search in",
                    },
                    "glob": {
                        "type": "string",
                        "description": "File glob pattern (e.g. '*.py')",
                    },
                },
                "required": ["pattern", "path"],
            },
        )

    def execute(self, *, pattern: str, path: str, glob: str = "**/*", **kwargs: Any) -> str:
        import re

        try:
            # Security: limit regex length to prevent ReDoS
            if len(pattern) > 500:
                return "Error: Regex pattern too long (max 500 characters)"

            search_path = Path(path)
            if not search_path.exists():
                return f"Error: Path not found: {path}"

            regex = re.compile(pattern, re.IGNORECASE)
            results: list[str] = []
            files = search_path.glob(glob) if glob else [search_path]

            for fp in files:
                if not fp.is_file():
                    continue
                try:
                    text = fp.read_text(encoding="utf-8", errors="replace")
                    for i, line in enumerate(text.splitlines(), 1):
                        if regex.search(line):
                            results.append(f"{fp}:{i}: {line.strip()}")
                            if len(results) >= 50:
                                return "\n".join(results) + "\n... (truncated, too many results)"
                except Exception:
                    continue

            if not results:
                return "No matches found"
            return "\n".join(results)
        except Exception as e:
            return f"Error searching: {e}"


class FileGlobTool(Tool):
    name = "file_glob"
    description = "Find files matching a glob pattern"

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern (e.g. '**/*.py', 'src/**/*.ts')",
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory to search in",
                    },
                },
                "required": ["pattern", "path"],
            },
        )

    def execute(self, *, pattern: str, path: str, **kwargs: Any) -> str:
        try:
            search_path = Path(path)
            if not search_path.exists():
                return f"Error: Path not found: {path}"

            matches = sorted(search_path.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
            result_lines = [str(m) for m in matches[:100]]
            if not result_lines:
                return "No files matched the pattern"
            return "\n".join(result_lines)
        except Exception as e:
            return f"Error: {e}"


class ListDirTool(Tool):
    name = "list_dir"
    description = "List files and directories in a path"

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path to list",
                    },
                },
                "required": ["path"],
            },
        )

    def execute(self, *, path: str, **kwargs: Any) -> str:
        try:
            p = Path(path)
            if not p.exists():
                return f"Error: Path not found: {path}"
            if not p.is_dir():
                return f"Error: Not a directory: {path}"

            entries = []
            for entry in sorted(p.iterdir()):
                prefix = "DIR " if entry.is_dir() else "FILE"
                size = ""
                if entry.is_file():
                    try:
                        size = f" ({entry.stat().st_size} bytes)"
                    except OSError:
                        pass
                entries.append(f"{prefix}  {entry.name}{size}")

            return "\n".join(entries) if entries else "(empty directory)"
        except Exception as e:
            return f"Error: {e}"


def register_file_tools(registry: Any) -> None:
    """Register all file operation tools."""
    for tool_cls in [FileReadTool, FileWriteTool, FileEditTool, FileSearchTool, FileGlobTool, ListDirTool]:
        registry.register(tool_cls())
