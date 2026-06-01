"""GUI automation tools using PyAutoGUI for CAD menu interaction."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pyautogui

from ..models.base import ToolDefinition
from .base import Tool

# Safety: move mouse to corner raises FailSafeException
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.1


class GUIClickTool(Tool):
    """Click at absolute screen coordinates or relative to a window."""

    name = "gui_click"
    description = "Click at screen coordinates. Use with VLM analysis to identify element positions."

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "x": {
                        "type": "integer",
                        "description": "X coordinate on screen (pixels)",
                    },
                    "y": {
                        "type": "integer",
                        "description": "Y coordinate on screen (pixels)",
                    },
                    "button": {
                        "type": "string",
                        "description": "Mouse button: 'left', 'right', 'middle' (default: left)",
                    },
                    "clicks": {
                        "type": "integer",
                        "description": "Number of clicks (default: 1, use 2 for double-click)",
                    },
                    "pause": {
                        "type": "number",
                        "description": "Seconds to wait after clicking (default: 0.5)",
                    },
                },
                "required": ["x", "y"],
            },
        )

    def execute(
        self,
        *,
        x: int,
        y: int,
        button: str = "left",
        clicks: int = 1,
        pause: float = 0.5,
        **kwargs: Any,
    ) -> str:
        try:
            screen_w, screen_h = pyautogui.size()
            if not (0 <= x <= screen_w and 0 <= y <= screen_h):
                return f"Error: Coordinates ({x}, {y}) out of screen bounds (0-{screen_w}, 0-{screen_h})"

            pyautogui.click(x=x, y=y, button=button, clicks=clicks)
            time.sleep(pause)
            return f"Clicked at ({x}, {y}) button={button} clicks={clicks}"
        except pyautogui.FailSafeException:
            return "Error: FailSafe triggered - mouse moved to corner"
        except Exception as e:
            return f"Error clicking: {e}"


class GUITypeTool(Tool):
    """Type text or press keys at the current focused element."""

    name = "gui_type"
    description = "Type text or press keyboard keys at the currently focused UI element. Use after gui_click to focus an input field."

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to type (use empty string with keys for key presses only)",
                    },
                    "interval": {
                        "type": "number",
                        "description": "Seconds between each keystroke (default: 0.02)",
                    },
                    "pause": {
                        "type": "number",
                        "description": "Seconds to wait after typing (default: 0.3)",
                    },
                },
                "required": ["text"],
            },
        )

    def execute(
        self,
        *,
        text: str,
        interval: float = 0.02,
        pause: float = 0.3,
        **kwargs: Any,
    ) -> str:
        try:
            if text:
                pyautogui.typewrite(text, interval=interval)
            time.sleep(pause)
            return f"Typed: '{text[:50]}{'...' if len(text) > 50 else ''}' ({len(text)} chars)"
        except Exception as e:
            return f"Error typing: {e}"


class GUIHotkeyTool(Tool):
    """Press keyboard shortcuts / hotkeys (e.g. Ctrl+S, Alt+F)."""

    name = "gui_hotkey"
    description = "Press a keyboard shortcut / hotkey combination (e.g. Ctrl+S, Alt+F4, Ctrl+Shift+S). Use for menu shortcuts and UI navigation."

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "keys": {
                        "type": "string",
                        "description": "Hotkey combination, e.g. 'ctrl+s', 'alt+f4', 'ctrl+shift+s' (keys joined by '+')",
                    },
                    "pause": {
                        "type": "number",
                        "description": "Seconds to wait after pressing (default: 0.5)",
                    },
                },
                "required": ["keys"],
            },
        )

    def execute(self, *, keys: str, pause: float = 0.5, **kwargs: Any) -> str:
        try:
            key_list = [k.strip() for k in keys.split("+")]
            pyautogui.hotkey(*key_list)
            time.sleep(pause)
            return f"Pressed hotkey: {keys}"
        except Exception as e:
            return f"Error pressing hotkey: {e}"


class GUIPressKeyTool(Tool):
    """Press a single key (arrow keys, enter, escape, tab, etc.)."""

    name = "gui_press_key"
    description = "Press a single keyboard key (e.g. 'enter', 'escape', 'tab', 'down', 'up', 'left', 'right', 'delete', 'backspace')."

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Key name: 'enter', 'escape', 'tab', 'down', 'up', 'left', 'right', 'delete', 'backspace', 'space', etc.",
                    },
                    "presses": {
                        "type": "integer",
                        "description": "Number of times to press (default: 1)",
                    },
                    "pause": {
                        "type": "number",
                        "description": "Seconds to wait after pressing (default: 0.3)",
                    },
                },
                "required": ["key"],
            },
        )

    def execute(
        self,
        *,
        key: str,
        presses: int = 1,
        pause: float = 0.3,
        **kwargs: Any,
    ) -> str:
        try:
            for _ in range(presses):
                pyautogui.press(key)
                time.sleep(0.05)
            time.sleep(pause)
            return f"Pressed '{key}' {presses}x"
        except Exception as e:
            return f"Error pressing key: {e}"


class GUIScreenshotTool(Tool):
    """Capture a screen region and save, optionally return mouse position."""

    name = "gui_screenshot"
    description = "Take a screenshot of the full screen or a specific region. Returns file path, dimensions, and current mouse position."

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
                        "description": "Region as 'x,y,w,h' or 'fullscreen' (default: fullscreen)",
                    },
                    "pause": {
                        "type": "number",
                        "description": "Seconds to wait before capturing (default: 0.5)",
                    },
                },
                "required": [],
            },
        )

    def execute(
        self,
        *,
        region: str = "fullscreen",
        pause: float = 0.5,
        **kwargs: Any,
    ) -> str:
        try:
            time.sleep(pause)
            save_dir = Path(self.screenshot_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            filepath = save_dir / f"gui_{int(time.time())}.png"

            if region == "fullscreen":
                img = pyautogui.screenshot(str(filepath))
            else:
                parts = [int(p.strip()) for p in region.split(",")]
                if len(parts) != 4:
                    return "Error: region must be 'x,y,w,h' or 'fullscreen'"
                x, y, w, h = parts
                img = pyautogui.screenshot(str(filepath), region=(x, y, w, h))

            mouse_x, mouse_y = pyautogui.position()
            size_kb = filepath.stat().st_size // 1024
            w, h = img.size
            return (
                f"Screenshot saved: {filepath}\n"
                f"Dimensions: {w}x{h}, {size_kb}KB\n"
                f"Mouse position: ({mouse_x}, {mouse_y})"
            )
        except Exception as e:
            return f"Error taking screenshot: {e}"


class GUIMousePosTool(Tool):
    """Get current mouse cursor position."""

    name = "gui_mouse_pos"
    description = "Get the current mouse cursor position on screen."

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
            },
        )

    def execute(self, **kwargs: Any) -> str:
        try:
            x, y = pyautogui.position()
            screen_w, screen_h = pyautogui.size()
            return f"Mouse position: ({x}, {y}) | Screen: {screen_w}x{screen_h}"
        except Exception as e:
            return f"Error: {e}"


class GUIDragTool(Tool):
    """Drag mouse from one point to another (for 3D view rotation)."""

    name = "gui_drag"
    description = "Drag mouse from start to end coordinates. Useful for rotating 3D views in CAD software."

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "start_x": {
                        "type": "integer",
                        "description": "Start X coordinate",
                    },
                    "start_y": {
                        "type": "integer",
                        "description": "Start Y coordinate",
                    },
                    "end_x": {
                        "type": "integer",
                        "description": "End X coordinate",
                    },
                    "end_y": {
                        "type": "integer",
                        "description": "End Y coordinate",
                    },
                    "duration": {
                        "type": "number",
                        "description": "Drag duration in seconds (default: 0.5)",
                    },
                    "button": {
                        "type": "string",
                        "description": "Mouse button: 'left', 'middle', 'right' (default: left)",
                    },
                },
                "required": ["start_x", "start_y", "end_x", "end_y"],
            },
        )

    def execute(
        self,
        *,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        duration: float = 0.5,
        button: str = "left",
        **kwargs: Any,
    ) -> str:
        try:
            pyautogui.moveTo(start_x, start_y)
            pyautogui.drag(
                end_x - start_x,
                end_y - start_y,
                duration=duration,
                button=button,
            )
            return f"Dragged ({start_x},{start_y}) -> ({end_x},{end_y}) button={button} duration={duration}s"
        except Exception as e:
            return f"Error dragging: {e}"


class GUIScrollTool(Tool):
    """Scroll mouse wheel at current or specified position."""

    name = "gui_scroll"
    description = "Scroll the mouse wheel. Positive = scroll up, negative = scroll down. Useful for zooming in CAD software."

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "clicks": {
                        "type": "integer",
                        "description": "Scroll amount: positive=up, negative=down (e.g. 3 or -3)",
                    },
                    "x": {
                        "type": "integer",
                        "description": "X position to scroll at (optional, defaults to current mouse position)",
                    },
                    "y": {
                        "type": "integer",
                        "description": "Y position to scroll at (optional, defaults to current mouse position)",
                    },
                    "pause": {
                        "type": "number",
                        "description": "Seconds to wait after scrolling (default: 0.3)",
                    },
                },
                "required": ["clicks"],
            },
        )

    def execute(
        self,
        *,
        clicks: int,
        x: int | None = None,
        y: int | None = None,
        pause: float = 0.3,
        **kwargs: Any,
    ) -> str:
        try:
            kwargs_scroll: dict[str, Any] = {}
            if x is not None and y is not None:
                kwargs_scroll["x"] = x
                kwargs_scroll["y"] = y
            pyautogui.scroll(clicks, **kwargs_scroll)
            time.sleep(pause)
            pos = f" at ({x},{y})" if x is not None else ""
            direction = "up" if clicks > 0 else "down"
            return f"Scrolled {direction} {abs(clicks)} clicks{pos}"
        except Exception as e:
            return f"Error scrolling: {e}"


def register_gui_action_tools(registry: Any, screenshot_dir: str = "") -> None:
    """Register all GUI automation tools."""
    registry.register(GUIClickTool())
    registry.register(GUITypeTool())
    registry.register(GUIHotkeyTool())
    registry.register(GUIPressKeyTool())
    registry.register(GUIScreenshotTool(screenshot_dir=screenshot_dir))
    registry.register(GUIMousePosTool())
    registry.register(GUIDragTool())
    registry.register(GUIScrollTool())
