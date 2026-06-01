"""VLM (Vision-Language Model) tools for visual analysis.

Each tool accepts an optional 'detail' parameter to control the vision model:
  - "fast":     GLM-4V-Flash   (free, 0.2-3s, simple tasks)
  - "standard": GLM-4V-Plus    (best accuracy, 3-6s, default)
  - "detailed": GLM-4.6V-Flash (verbose, 20-27s, CAD verification)
  - "maximum":  GLM-4.6V       (most detailed, 40-50s, complex inspection)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ..models.base import ToolDefinition
from ..models.router import ModelRouter, VisionDetail
from .base import Tool

_DETAIL_DESCRIPTION = (
    "Vision detail level: 'fast' (quick, free), 'standard' (accurate, default), "
    "'detailed' (verbose, good for CAD), 'maximum' (most thorough). "
    "Use 'detailed' or 'maximum' for CAD model verification."
)


def _parse_detail(detail: str) -> VisionDetail | None:
    """Convert string to VisionDetail enum."""
    try:
        return VisionDetail(detail)
    except ValueError:
        return None


def _parse_verification_json(raw: str) -> dict[str, Any]:
    """Parse structured verification result from VLM output.

    Tries to extract a JSON object from the VLM response.
    Falls back to field-by-field extraction if JSON parsing fails.
    """
    import json
    import re

    # Try extracting JSON from response (may be wrapped in markdown code blocks)
    json_match = re.search(r'\{[^{}]*"match"[^{}]*\}', raw, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            return {
                "match": bool(data.get("match", False)),
                "observed": str(data.get("observed", "")),
                "differences": str(data.get("differences", "None")),
                "suggestion": str(data.get("suggestion", "None")),
                "fix_commands": str(data.get("fix_commands", "None")),
            }
        except (json.JSONDecodeError, ValueError):
            pass

    # Fallback: field-by-field extraction
    def _extract_field(name: str) -> str:
        pattern = rf'{name}[:\s]+(.*?)(?:\n|$)'
        m = re.search(pattern, raw, re.IGNORECASE)
        return m.group(1).strip() if m else "None"

    match_str = _extract_field("match")
    match_val = "yes" in match_str.lower() or match_str.lower() == "true"

    return {
        "match": match_val,
        "observed": _extract_field("observed"),
        "differences": _extract_field("differences"),
        "suggestion": _extract_field("suggestion"),
        "fix_commands": _extract_field("fix_commands"),
    }


class VLMAnalyzeTool(Tool):
    """Analyze an image using a vision-language model."""

    name = "vlm_analyze"
    description = "Analyze an image using a vision model (VLM). Describe what you see, identify UI elements, check 3D model status, etc."

    def __init__(self, router: ModelRouter) -> None:
        self.router = router

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "Path to the image file to analyze",
                    },
                    "prompt": {
                        "type": "string",
                        "description": "What to look for or analyze in the image",
                    },
                    "detail": {
                        "type": "string",
                        "description": _DETAIL_DESCRIPTION,
                    },
                },
                "required": ["image_path", "prompt"],
            },
        )

    def execute(self, *, image_path: str, prompt: str, detail: str = "detailed", **kwargs: Any) -> str:
        try:
            vd = _parse_detail(detail)
            result = self.router.vision(image_path, prompt, detail=vd)
            return result
        except Exception as e:
            return f"Error analyzing image: {e}"


class ScreenAnalyzeTool(Tool):
    """Capture screen and immediately analyze it with VLM."""

    name = "screen_analyze"
    description = "Capture the screen and analyze it with vision model in one step"

    def __init__(self, router: ModelRouter, screenshot_dir: str = "") -> None:
        self.router = router
        self.screenshot_dir = screenshot_dir

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "What to analyze on the screen",
                    },
                    "region": {
                        "type": "string",
                        "description": "Screen region: 'fullscreen', 'left', 'right' (default: fullscreen)",
                    },
                    "detail": {
                        "type": "string",
                        "description": _DETAIL_DESCRIPTION,
                    },
                },
                "required": ["prompt"],
            },
        )

    def execute(self, *, prompt: str, region: str = "fullscreen", detail: str = "detailed", **kwargs: Any) -> str:
        try:
            import mss
            import mss.tools

            save_dir = Path(self.screenshot_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            filepath = save_dir / f"analyze_{int(time.time())}.png"

            with mss.MSS() as sct:
                if region == "fullscreen":
                    capture = sct.monitors[0]
                else:
                    m = sct.monitors[1]
                    w, h = m["width"], m["height"]
                    if region == "left":
                        capture = {"left": m["left"], "top": m["top"], "width": w // 2, "height": h}
                    elif region == "right":
                        capture = {"left": m["left"] + w // 2, "top": m["top"], "width": w // 2, "height": h}
                    else:
                        capture = sct.monitors[0]

                img = sct.grab(capture)
                mss.tools.to_png(img.rgb, img.size, output=str(filepath))

            vd = _parse_detail(detail)
            result = self.router.vision(str(filepath), prompt, detail=vd)
            return f"[Screenshot: {filepath}]\n\n{result}"
        except ImportError:
            return "Error: mss not installed. Run: pip install mss"
        except Exception as e:
            return f"Error: {e}"


class WindowAnalyzeTool(Tool):
    """Capture a specific window and analyze it with VLM in one step."""

    name = "window_analyze"
    description = (
        "Capture a specific window by title and analyze it with VLM. "
        "Useful for checking CAD software state, verifying 3D models, etc."
    )

    def __init__(self, router: ModelRouter, screenshot_dir: str = "") -> None:
        self.router = router
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
                        "description": "Window title substring to match (e.g. 'FreeCAD', 'SolidWorks')",
                    },
                    "prompt": {
                        "type": "string",
                        "description": "What to analyze in the captured window",
                    },
                    "detail": {
                        "type": "string",
                        "description": _DETAIL_DESCRIPTION,
                    },
                },
                "required": ["title", "prompt"],
            },
        )

    def execute(self, *, title: str, prompt: str, detail: str = "detailed", **kwargs: Any) -> str:
        try:
            import ctypes
            import ctypes.wintypes

            from PIL import ImageGrab

            from .screen import _find_windows_by_title

            matches = _find_windows_by_title(title)
            if not matches:
                return f"Error: No window found matching '{title}'"

            user32 = ctypes.windll.user32
            screen_w = user32.GetSystemMetrics(0)
            screen_h = user32.GetSystemMetrics(1)

            # Find a window with valid dimensions
            for hwnd, full_title in matches:
                rect = ctypes.wintypes.RECT()
                user32.GetWindowRect(hwnd, ctypes.byref(rect))
                left = max(0, rect.left)
                top = max(0, rect.top)
                right = min(screen_w, rect.right)
                bottom = min(screen_h, rect.bottom)
                if right - left < 10 or bottom - top < 10:
                    continue

                user32.SetForegroundWindow(hwnd)
                time.sleep(0.5)

                save_dir = Path(self.screenshot_dir)
                save_dir.mkdir(parents=True, exist_ok=True)
                filepath = save_dir / f"win_analyze_{int(time.time())}.png"

                img = ImageGrab.grab(bbox=(left, top, right, bottom))
                img.save(str(filepath))

                vd = _parse_detail(detail)
                result = self.router.vision(str(filepath), prompt, detail=vd)
                size_kb = filepath.stat().st_size // 1024
                return (
                    f"[Window: '{full_title}', {right - left}x{bottom - top}, {size_kb}KB]\n\n{result}"
                )

            return f"Error: All matching windows for '{title}' have invalid dimensions"
        except Exception as e:
            return f"Error: {e}"


class CADVerifyTool(Tool):
    """Capture a CAD window and verify the 3D model matches expectations."""

    name = "cad_verify"
    description = (
        "Capture the CAD software window and verify if the 3D model matches the expected design. "
        "Returns structured verification result. Uses a detailed vision model for thorough analysis."
    )

    def __init__(self, router: ModelRouter, screenshot_dir: str = "") -> None:
        self.router = router
        self.screenshot_dir = screenshot_dir

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "expected": {
                        "type": "string",
                        "description": "Description of the expected 3D model (shape, dimensions, features)",
                    },
                    "window_title": {
                        "type": "string",
                        "description": "CAD window title (default: 'FreeCAD')",
                    },
                    "detail": {
                        "type": "string",
                        "description": _DETAIL_DESCRIPTION,
                    },
                },
                "required": ["expected"],
            },
        )

    def execute(self, *, expected: str, window_title: str = "FreeCAD", detail: str = "detailed", **kwargs: Any) -> str:
        try:
            import ctypes
            import ctypes.wintypes

            from PIL import ImageGrab

            from .screen import _find_windows_by_title

            matches = _find_windows_by_title(window_title)
            if not matches:
                return f"Error: No window found matching '{window_title}'. Is the CAD software open?"

            user32 = ctypes.windll.user32
            screen_w = user32.GetSystemMetrics(0)
            screen_h = user32.GetSystemMetrics(1)

            # Find a window with valid dimensions
            for hwnd, full_title in matches:
                rect = ctypes.wintypes.RECT()
                user32.GetWindowRect(hwnd, ctypes.byref(rect))
                left = max(0, rect.left)
                top = max(0, rect.top)
                right = min(screen_w, rect.right)
                bottom = min(screen_h, rect.bottom)
                if right - left < 10 or bottom - top < 10:
                    continue

                user32.SetForegroundWindow(hwnd)
                time.sleep(0.5)

                save_dir = Path(self.screenshot_dir)
                save_dir.mkdir(parents=True, exist_ok=True)
                filepath = save_dir / f"cad_verify_{int(time.time())}.png"

                img = ImageGrab.grab(bbox=(left, top, right, bottom))
                img.save(str(filepath))

                verify_prompt = (
                    "You are a 3D CAD model verification expert. "
                    "Analyze the CAD software screenshot and determine if the visible 3D model matches the expected description.\n\n"
                    f"Expected model: {expected}\n\n"
                    "You MUST respond with EXACTLY this JSON format (no markdown, no backticks, raw JSON only):\n"
                    '{"match": true/false, "observed": "what you see in the 3D viewport", '
                    '"differences": "any differences from expected, or None", '
                    '"suggestion": "what to fix if not matching, or None", '
                    '"fix_commands": "suggested fc_batch operations to fix issues, or None"}\n\n'
                    "Example of a mismatch response:\n"
                    '{"match": false, "observed": "A cube without any hole", '
                    '"differences": "Expected center hole is missing", '
                    '"suggestion": "Add a cylindrical cut operation", '
                    '"fix_commands": "None"}\n\n'
                    "Example of a match response:\n"
                    '{"match": true, "observed": "A 30x30x30mm cube with a center hole", '
                    '"differences": "None", '
                    '"suggestion": "None", '
                    '"fix_commands": "None"}'
                )

                vd = _parse_detail(detail)
                result = self.router.vision(str(filepath), verify_prompt, detail=vd)

                # Try to parse structured JSON from VLM response
                parsed = _parse_verification_json(result)

                return (
                    f"[CAD Verification - Window: '{full_title}']\n"
                    f"MATCH: {parsed['match']}\n"
                    f"OBSERVED: {parsed['observed']}\n"
                    f"DIFFERENCES: {parsed['differences']}\n"
                    f"SUGGESTION: {parsed['suggestion']}\n"
                    f"FIX_COMMANDS: {parsed['fix_commands']}\n"
                    f"\n--- Raw VLM output ---\n{result}"
                )

            return f"Error: All matching windows for '{window_title}' have invalid dimensions"
        except Exception as e:
            return f"Error: {e}"


def _parse_elements_json(raw: str) -> list[dict[str, Any]]:
    """Parse UI element list from VLM output.

    Expects a JSON array: [{"name": "...", "x": int, "y": int, "w": int, "h": int, "type": "..."}]
    Falls back to line-by-line parsing.
    """
    import json
    import re

    # Try extracting JSON array from response
    arr_match = re.search(r'\[.*?\]', raw, re.DOTALL)
    if arr_match:
        try:
            data = json.loads(arr_match.group())
            if isinstance(data, list):
                elements = []
                for item in data:
                    if isinstance(item, dict) and "name" in item:
                        elements.append({
                            "name": str(item.get("name", "")),
                            "x": int(item.get("x", 0)),
                            "y": int(item.get("y", 0)),
                            "w": int(item.get("w", 0)),
                            "h": int(item.get("h", 0)),
                            "type": str(item.get("type", "unknown")),
                        })
                if elements:
                    return elements
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # Fallback: parse numbered lines like "1. Menu Bar - x:10, y:5, w:200, h:25"
    elements = []
    for line in raw.split("\n"):
        m = re.match(
            r'\s*\d+\.\s*(.+?)\s*[-:]\s*x[:\s]*(\d+)\s*,\s*y[:\s]*(\d+)'
            r'(?:\s*,\s*w[:\s]*(\d+))?\s*(?:,\s*h[:\s]*(\d+))?',
            line,
        )
        if m:
            elements.append({
                "name": m.group(1).strip(),
                "x": int(m.group(2)),
                "y": int(m.group(3)),
                "w": int(m.group(4) or 0),
                "h": int(m.group(5) or 0),
                "type": "unknown",
            })

    return elements


class VLMLocateTool(Tool):
    """Locate UI elements in a window using VLM."""

    name = "vlm_locate"
    description = (
        "Capture a window and use VLM to locate UI elements (buttons, menus, input fields). "
        "Returns a list of elements with their names and screen coordinates. "
        "Use before gui_click to find the correct coordinates."
    )

    def __init__(self, router: ModelRouter, screenshot_dir: str = "") -> None:
        self.router = router
        self.screenshot_dir = screenshot_dir

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "window_title": {
                        "type": "string",
                        "description": "Window title substring (e.g. 'FreeCAD', 'Chrome')",
                    },
                    "target": {
                        "type": "string",
                        "description": "What kind of UI elements to locate (e.g. 'toolbar buttons', 'menu items', 'all interactive elements'). Default: 'all interactive elements'",
                    },
                    "detail": {
                        "type": "string",
                        "description": _DETAIL_DESCRIPTION,
                    },
                },
                "required": ["window_title"],
            },
        )

    def execute(
        self,
        *,
        window_title: str,
        target: str = "all interactive elements (buttons, menus, input fields, toolbars)",
        detail: str = "standard",
        **kwargs: Any,
    ) -> str:
        try:
            import ctypes
            import ctypes.wintypes

            from PIL import ImageGrab

            from .screen import _find_windows_by_title

            matches = _find_windows_by_title(window_title)
            if not matches:
                return f"Error: No window found matching '{window_title}'"

            user32 = ctypes.windll.user32
            screen_w = user32.GetSystemMetrics(0)
            screen_h = user32.GetSystemMetrics(1)

            for hwnd, full_title in matches:
                rect = ctypes.wintypes.RECT()
                user32.GetWindowRect(hwnd, ctypes.byref(rect))
                left = max(0, rect.left)
                top = max(0, rect.top)
                right = min(screen_w, rect.right)
                bottom = min(screen_h, rect.bottom)
                if right - left < 10 or bottom - top < 10:
                    continue

                user32.SetForegroundWindow(hwnd)
                time.sleep(0.5)

                save_dir = Path(self.screenshot_dir)
                save_dir.mkdir(parents=True, exist_ok=True)
                filepath = save_dir / f"vlm_locate_{int(time.time())}.png"

                img = ImageGrab.grab(bbox=(left, top, right, bottom))
                img.save(str(filepath))

                win_w = right - left
                win_h = bottom - top

                locate_prompt = (
                    "You are a UI element locator. Analyze this screenshot and identify "
                    f"the following: {target}.\n\n"
                    "The screenshot window position starts at screen coordinates: "
                    f"left={left}, top={top}. Window size: {win_w}x{win_h}.\n\n"
                    "For each element, report its CENTER position in ABSOLUTE screen coordinates "
                    "(window_offset + element_center_within_window).\n\n"
                    "You MUST respond with EXACTLY this JSON array (no markdown, no backticks, raw JSON only):\n"
                    '[{"name": "element name", "x": center_x_screen, "y": center_y_screen, '
                    '"w": approximate_width, "h": approximate_height, "type": "button/menu/input/toolbar/other"}]\n\n'
                    "Important:\n"
                    "- x,y must be ABSOLUTE screen coordinates (add window left/top offset)\n"
                    "- Report center of each element, not top-left corner\n"
                    "- Be precise with coordinates for accurate clicking\n"
                    "- List up to 20 most visible/relevant elements"
                )

                vd = _parse_detail(detail)
                result = self.router.vision(str(filepath), locate_prompt, detail=vd)

                elements = _parse_elements_json(result)

                if not elements:
                    return (
                        f"[VLM Locate - Window: '{full_title}', {win_w}x{win_h}]\n"
                        f"No elements parsed from VLM output.\n\n"
                        f"Raw VLM output:\n{result}"
                    )

                # Format output
                lines = [
                    f"[VLM Locate - Window: '{full_title}', {win_w}x{win_h}]",
                    f"Found {len(elements)} UI elements:",
                    "",
                ]
                for i, el in enumerate(elements, 1):
                    size_str = f", {el['w']}x{el['h']}" if el['w'] and el['h'] else ""
                    lines.append(
                        f"  {i}. {el['name']} ({el['type']}) "
                        f"@ ({el['x']}, {el['y']}){size_str}"
                    )

                lines.append("")
                lines.append("--- Raw VLM output ---")
                lines.append(result)

                return "\n".join(lines)

            return f"Error: All matching windows for '{window_title}' have invalid dimensions"
        except Exception as e:
            return f"Error: {e}"


def register_vlm_tools(registry: Any, router: ModelRouter, screenshot_dir: str = "") -> None:
    registry.register(VLMAnalyzeTool(router))
    registry.register(ScreenAnalyzeTool(router, screenshot_dir=screenshot_dir))
    registry.register(WindowAnalyzeTool(router, screenshot_dir=screenshot_dir))
    registry.register(CADVerifyTool(router, screenshot_dir=screenshot_dir))
    registry.register(VLMLocateTool(router, screenshot_dir=screenshot_dir))
