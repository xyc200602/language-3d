"""Screen capture tools using mss + PIL."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ..models.base import ToolDefinition
from .base import Tool


def _find_windows_by_title(title_substring: str) -> list[tuple[int, str]]:
    """Find all windows whose title contains the given substring.

    Returns list of (hwnd, title) pairs.
    """
    import ctypes
    import ctypes.wintypes

    user32 = ctypes.windll.user32
    results: list[tuple[int, str]] = []

    # Callback for EnumWindows
    WNDENUMPROC = ctypes.WINFUNCTYPE(
        ctypes.c_bool, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM
    )

    def enum_callback(hwnd, _lparam):
        if user32.IsWindowVisible(hwnd):
            length = user32.GetWindowTextLengthW(hwnd)
            if length > 0:
                buf = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(hwnd, buf, length + 1)
                win_title = buf.value
                if title_substring.lower() in win_title.lower():
                    results.append((hwnd, win_title))
        return True

    user32.EnumWindows(WNDENUMPROC(enum_callback), 0)
    return results


class ScreenCaptureTool(Tool):
    name = "screen_capture"
    description = "Capture the screen or a specific region"

    def __init__(self, screenshot_dir: str = "") -> None:
        self.screenshot_dir = screenshot_dir

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "region": {
                        "type": "string",
                        "description": (
                            "Capture region: 'fullscreen', 'left', 'right', "
                            "or JSON string {left,top,width,height}"
                        ),
                    },
                    "monitor": {
                        "type": "integer",
                        "description": "Monitor index (0=all, 1=primary, 2=secondary)",
                    },
                },
                "required": [],
            },
        )

    def execute(self, *, region: str = "fullscreen", monitor: int = 1, **kwargs: Any) -> str:
        try:
            import mss
            import mss.tools
        except ImportError:
            return "Error: mss not installed. Run: pip install mss"

        try:
            save_dir = Path(self.screenshot_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            filename = f"screenshot_{int(time.time())}.png"
            filepath = save_dir / filename

            with mss.MSS() as sct:
                if region == "fullscreen":
                    capture = sct.monitors[monitor] if monitor < len(sct.monitors) else sct.monitors[0]
                elif region in ("left", "right", "top", "bottom"):
                    m = sct.monitors[monitor] if monitor < len(sct.monitors) else sct.monitors[0]
                    w, h = m["width"], m["height"]
                    halves = {
                        "left": {"left": m["left"], "top": m["top"], "width": w // 2, "height": h},
                        "right": {"left": m["left"] + w // 2, "top": m["top"], "width": w // 2, "height": h},
                        "top": {"left": m["left"], "top": m["top"], "width": w, "height": h // 2},
                        "bottom": {"left": m["left"], "top": m["top"] + h // 2, "width": w, "height": h // 2},
                    }
                    capture = halves[region]
                else:
                    try:
                        capture = json.loads(region)
                    except (json.JSONDecodeError, TypeError):
                        capture = sct.monitors[monitor]

                img = sct.grab(capture)
                mss.tools.to_png(img.rgb, img.size, output=str(filepath))

            return f"Screenshot saved to: {filepath}"
        except Exception as e:
            return f"Error capturing screen: {e}"


class WindowCaptureTool(Tool):
    """Capture a specific window by title substring (Windows only)."""

    name = "window_capture"
    description = "Capture a specific window by its title (partial match supported)"

    def __init__(self, screenshot_dir: str = "") -> None:
        self.screenshot_dir = screenshot_dir

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Window title substring to match (e.g. 'FreeCAD', 'Chrome')",
                    },
                },
                "required": ["title"],
            },
        )

    def execute(self, *, title: str, **kwargs: Any) -> str:
        try:
            import ctypes
            import ctypes.wintypes

            from PIL import ImageGrab

            # Find matching windows
            matches = _find_windows_by_title(title)
            if not matches:
                return f"Error: No window found matching '{title}'"

            user32 = ctypes.windll.user32
            screen_w = user32.GetSystemMetrics(0)  # SM_CXSCREEN
            screen_h = user32.GetSystemMetrics(1)  # SM_CYSCREEN

            # Try each matching window, skip zero-size/hidden windows
            for hwnd, full_title in matches:
                rect = ctypes.wintypes.RECT()
                user32.GetWindowRect(hwnd, ctypes.byref(rect))

                # Clamp to screen bounds
                left = max(0, rect.left)
                top = max(0, rect.top)
                right = min(screen_w, rect.right)
                bottom = min(screen_h, rect.bottom)
                w = right - left
                h = bottom - top

                if w < 10 or h < 10:
                    continue  # Skip tiny or hidden windows

                # Bring window to foreground
                user32.SetForegroundWindow(hwnd)
                time.sleep(0.5)

                # Save
                save_dir = Path(self.screenshot_dir)
                save_dir.mkdir(parents=True, exist_ok=True)
                filename = f"window_{int(time.time())}.png"
                filepath = save_dir / filename

                img = ImageGrab.grab(bbox=(left, top, right, bottom))
                img.save(str(filepath))

                size_kb = filepath.stat().st_size // 1024
                return (
                    f"Window captured: '{full_title}'\n"
                    f"Dimensions: {w}x{h}\n"
                    f"Saved to: {filepath} ({size_kb} KB)"
                )

            return f"Error: All matching windows for '{title}' have invalid dimensions"
        except Exception as e:
            return f"Error capturing window: {e}"


class ListWindowsTool(Tool):
    """List all visible windows (useful for finding window titles)."""

    name = "list_windows"
    description = "List all visible window titles on screen"

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={"type": "object", "properties": {}, "required": []},
        )

    def execute(self, **kwargs: Any) -> str:
        try:
            import ctypes
            import ctypes.wintypes

            user32 = ctypes.windll.user32
            windows: list[str] = []

            WNDENUMPROC = ctypes.WINFUNCTYPE(
                ctypes.c_bool, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM
            )

            def enum_callback(hwnd, _lparam):
                if user32.IsWindowVisible(hwnd):
                    length = user32.GetWindowTextLengthW(hwnd)
                    if length > 0:
                        buf = ctypes.create_unicode_buffer(length + 1)
                        user32.GetWindowTextW(hwnd, buf, length + 1)
                        if buf.value.strip():
                            windows.append(buf.value)
                return True

            user32.EnumWindows(WNDENUMPROC(enum_callback), 0)
            windows.sort(key=str.lower)

            lines = [f"Found {len(windows)} visible windows:"]
            for i, w in enumerate(windows, 1):
                lines.append(f"  {i}. {w}")
            return "\n".join(lines)
        except Exception as e:
            return f"Error listing windows: {e}"


def register_screen_tools(registry: Any, screenshot_dir: str = "") -> None:
    registry.register(ScreenCaptureTool(screenshot_dir=screenshot_dir))
    registry.register(WindowCaptureTool(screenshot_dir=screenshot_dir))
    registry.register(ListWindowsTool())
