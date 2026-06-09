"""VLM (Vision-Language Model) tools for visual analysis.

Each tool accepts an optional 'detail' parameter to control the vision model:
  - "fast":     GLM-4V-Flash   (free, 0.2-3s, simple tasks)
  - "standard": GLM-4V-Plus    (best accuracy, 3-6s, default)
  - "detailed": GLM-4.6V-Flash (verbose, 20-27s, CAD verification)
  - "maximum":  GLM-4.6V       (most detailed, 40-50s, complex inspection)
"""

from __future__ import annotations

import os
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


def _normalize_verification(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize parsed verification data: unify boolean values, null strings."""
    # Normalize match field: accept true/True/yes/1/"true"
    raw_match = data.get("match", False)
    if isinstance(raw_match, str):
        raw_match = raw_match.lower() in ("true", "yes", "1")
    data["match"] = bool(raw_match)

    # Normalize optional text fields: treat null/"null"/"None" as None string
    for field in ("differences", "suggestion", "fix_commands"):
        val = data.get(field)
        if val is None or (isinstance(val, str) and val.lower() in ("null", "none", "")):
            data[field] = "None"
        else:
            data[field] = str(val)

    # Ensure observed is a string
    data["observed"] = str(data.get("observed", ""))

    return data


def _parse_verification_json(raw: str) -> dict[str, Any]:
    """Parse structured verification result from VLM output.

    Three-stage strategy:
    1. Extract JSON from markdown code blocks (```json ... ```)
    2. Extract bare JSON using bracket depth tracking (handles nested/multiline)
    3. Fallback to field-by-field regex extraction
    """
    import json
    import re

    # Strategy 1: Markdown code block
    md_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', raw, re.DOTALL)
    if md_match:
        try:
            data = json.loads(md_match.group(1))
            if isinstance(data, dict) and "match" in data:
                return _normalize_verification(data)
        except (json.JSONDecodeError, ValueError):
            pass

    # Strategy 2: Bracket depth tracking for bare JSON
    start = raw.find("{")
    if start >= 0:
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(raw)):
            ch = raw[i]
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = raw[start : i + 1]
                    try:
                        data = json.loads(candidate)
                        if isinstance(data, dict) and "match" in data:
                            return _normalize_verification(data)
                    except (json.JSONDecodeError, ValueError):
                        pass
                    break

    # Strategy 3: Field-by-field regex extraction
    def _extract_field(name: str) -> str:
        pattern = rf'["\']?{name}["\']?\s*[:：]\s*(.*?)(?:[,，\n}}]|$)'
        m = re.search(pattern, raw, re.IGNORECASE)
        if m:
            val = m.group(1).strip().strip('"').strip("'")
            return val if val else "None"
        return "None"

    match_str = _extract_field("match")
    match_val = match_str.lower() in ("true", "yes", "1")

    result = {
        "match": match_val,
        "observed": _extract_field("observed"),
        "differences": _extract_field("differences"),
        "suggestion": _extract_field("suggestion"),
        "fix_commands": _extract_field("fix_commands"),
    }

    # Extract confidence if present
    confidence_str = _extract_field("confidence")
    if confidence_str and confidence_str.lower() not in ("none", "null", ""):
        result["confidence"] = confidence_str.lower()

    return result


def _aggregate_angle_results(angle_results: list[dict[str, Any]]) -> bool:
    """Aggregate multi-angle verification results using confidence-weighted voting.

    Weights: high=2.0, medium=1.0, low=0.5
    Final MATCH if total MATCH weight > 50% of total possible weight.
    """
    CONFIDENCE_WEIGHTS: dict[str, float] = {
        "high": 2.0,
        "medium": 1.0,
        "low": 0.5,
    }

    total_weight = 0.0
    match_weight = 0.0

    for result in angle_results:
        confidence = str(result.get("confidence", "medium")).lower()
        weight = CONFIDENCE_WEIGHTS.get(confidence, 1.0)
        total_weight += weight
        if result.get("match", False):
            match_weight += weight

    if total_weight == 0:
        return False

    return match_weight / total_weight > 0.5


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
    """Verify a 3D model against expectations using VTK offscreen rendering + VLM.

    No GUI needed, no screenshots, no window management.
    Renders the model from multiple camera angles using VTK offscreen rendering,
    then sends the renders to a VLM for structural analysis.
    """

    name = "cad_verify"
    description = (
        "Verify if a 3D model (STL/FCStd file) matches the expected design. "
        "Uses VTK offscreen rendering for deterministic multi-angle views, "
        "then a VLM for structural analysis. No GUI or screenshot needed."
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
                    "stl_path": {
                        "type": "string",
                        "description": "Path to the STL file to verify. If omitted, searches workspace for the most recent STL.",
                    },
                    "detail": {
                        "type": "string",
                        "description": _DETAIL_DESCRIPTION,
                    },
                    "angles": {
                        "type": "string",
                        "description": (
                            "Camera angles for multi-view verification, comma-separated "
                            "(e.g. 'isometric,front,top'). Default: 'isometric,front,top,right'."
                        ),
                    },
                },
                "required": ["expected"],
            },
        )

    def execute(
        self,
        *,
        expected: str,
        stl_path: str = "",
        detail: str = "detailed",
        angles: str = "isometric,front,top,right",
        **kwargs: Any,
    ) -> str:
        try:
            # Resolve STL path
            resolved = self._resolve_stl(stl_path)
            if not resolved:
                return (
                    "Error: No STL file found. Use fc_batch export_stl first, "
                    "or provide stl_path parameter."
                )

            # Parse angles
            angle_list = [a.strip() for a in angles.split(",") if a.strip()] if angles else ["isometric"]

            # Render multi-angle PNGs via VTK offscreen
            png_paths = self._render_views(resolved, angle_list)

            # Send to VLM and aggregate
            verify_prompt = self._build_verify_prompt(expected)
            vd = _parse_detail(detail)

            angle_results: list[dict[str, Any]] = []
            for angle, png_path in zip(angle_list, png_paths):
                vlm_result = self.router.vision(png_path, verify_prompt, detail=vd)
                parsed = _parse_verification_json(vlm_result)
                parsed["angle"] = angle
                parsed["confidence"] = parsed.get("confidence", "medium")
                parsed["raw"] = vlm_result
                angle_results.append(parsed)

            # Aggregate via confidence-weighted voting
            if len(angle_results) == 1:
                final_match = angle_results[0]["match"]
                ar0 = angle_results[0]
                return (
                    f"[CAD Verification - VTK Offscreen]\n"
                    f"MATCH: {final_match}\n"
                    f"OBSERVED: {ar0['observed']}\n"
                    f"DIFFERENCES: {ar0['differences']}\n"
                    f"SUGGESTION: {ar0['suggestion']}\n"
                    f"FIX_COMMANDS: {ar0['fix_commands']}\n"
                    f"\n--- Raw VLM output ---\n{ar0['raw']}"
                )

            final_match = _aggregate_angle_results(angle_results)

            # Build multi-angle output
            lines = ["[CAD Verification (Multi-Angle) - VTK Offscreen]"]
            lines.append(f"FINAL MATCH: {final_match}")
            lines.append(f"STL: {resolved}")
            lines.append(f"Angles verified: {len(angle_results)}")
            lines.append("")
            for ar in angle_results:
                match_str = "MATCH" if ar["match"] else "MISMATCH"
                lines.append(
                    f"  [{ar['angle'].upper()}] {match_str} "
                    f"(confidence: {ar.get('confidence', 'medium')})"
                )
                lines.append(f"    Observed: {ar['observed'][:120]}")
                if ar["differences"] != "None":
                    lines.append(f"    Differences: {ar['differences'][:120]}")

            # Include the most detailed raw output
            best = max(angle_results, key=lambda a: len(a.get("raw", "")))
            lines.append("")
            lines.append(f"--- Raw VLM output (from {best['angle']}) ---")
            lines.append(best["raw"])

            return "\n".join(lines)

        except Exception as e:
            return f"Error: {e}"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_stl(self, stl_path: str) -> str | None:
        """Find the STL file to verify."""
        if stl_path and os.path.isfile(stl_path):
            return os.path.abspath(stl_path)

        # Search screenshot_dir and workspace for recent STL files
        search_dirs = [self.screenshot_dir]
        from ..config import load_config
        try:
            cfg = load_config()
            search_dirs.append(cfg.agent.workspace)
        except Exception:
            pass

        best_path: str | None = None
        best_mtime = 0.0
        for d in search_dirs:
            if not d or not os.path.isdir(d):
                continue
            for f in os.listdir(d):
                if f.lower().endswith(".stl") and not f.lower().endswith(".preview.stl"):
                    fp = os.path.join(d, f)
                    mtime = os.path.getmtime(fp)
                    if mtime > best_mtime:
                        best_mtime = mtime
                        best_path = fp
        return best_path

    def _render_views(self, stl_path: str, angle_list: list[str]) -> list[str]:
        """Render STL from multiple angles using VTK offscreen."""
        from .vtk_renderer import VTKOffscreenRenderer, VIEW_PRESETS

        save_dir = Path(self.screenshot_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        renderer = VTKOffscreenRenderer(width=1200, height=900)
        renderer.load_stl(stl_path)
        renderer.add_axes(length=25)

        # Validate angles against presets, fallback to isometric
        valid_angles = [a for a in angle_list if a in VIEW_PRESETS]
        if not valid_angles:
            valid_angles = ["isometric"]

        return renderer.render_all_views(
            str(save_dir), views=valid_angles, prefix=f"cad_verify_{int(time.time())}"
        )

    @staticmethod
    def _build_verify_prompt(expected: str) -> str:
        """Build the verification prompt."""
        return (
            "You are a 3D CAD model verification expert.\n\n"
            "Step 1: Describe what you see in the 3D render (shape, dimensions, features). "
            "Focus on topological features: shape types, number of holes/slots/bosses, "
            "overall proportions. Ignore viewing angle, lighting, or rendering artifacts.\n"
            "Step 2: Compare with the expected model description below. "
            "Match based on structural/topological similarity, not visual pixel-perfect match.\n"
            "Step 3: Give your conclusion.\n\n"
            f"Expected model: {expected}\n\n"
            "Typical match example: If expected is '80x60x8 rectangular plate with 4 mounting holes', "
            "and you see a rectangular plate with 4 circular through-holes, that is a MATCH.\n\n"
            "Respond with a JSON object (you may wrap it in ```json```):\n"
            '{"match": true/false, "observed": "what you see", '
            '"differences": "any differences or null", '
            '"suggestion": "fix suggestion or null", '
            '"fix_commands": "fc_batch operations or null", '
            '"confidence": "high/medium/low"}\n\n'
            "MATCHING RULES:\n"
            "- Be GENEROUS with matching. Only report mismatch if the shape/structure is clearly different.\n"
            "- Minor visual differences (viewing angle, lighting, edge rendering) should still match=true.\n"
            "- If the topology matches (same number of features, same shape types), report match=true.\n"
            "- Use null instead of \"None\" for empty fields."
        )


def _parse_elements_json(raw: str) -> list[dict[str, Any]]:
    """Parse UI element list from VLM output.

    Expects a JSON array: [{"name": "...", "x": int, "y": int, "w": int, "h": int, "type": "..."}]
    Falls back to line-by-line parsing.
    """
    import json
    import re

    # Try extracting JSON array from response
    # Use greedy match to handle nested arrays correctly (e.g., [[1,2],[3,4]])
    # Find the outermost balanced brackets
    start = raw.find("[")
    if start >= 0:
        depth = 0
        end = start
        for i in range(start, len(raw)):
            if raw[i] == "[":
                depth += 1
            elif raw[i] == "]":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        candidate = raw[start:end]
        try:
            data = json.loads(candidate)
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
