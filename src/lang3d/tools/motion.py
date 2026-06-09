"""Motion simulation tools: kinematic analysis, trajectory planning, VLM interpretation.

Tools:
  motion_range      - Analyze joint motion range and reachable workspace
  motion_trajectory  - Plan joint-space linear interpolation trajectory
  motion_vlm_analyze - Screenshot motion visualization + VLM analysis
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from ..knowledge.simulation import JOINT_TYPES, JointType
from ..models.base import ToolDefinition
from .base import Tool


# ---------------------------------------------------------------------------
# FreeCAD subprocess bridge (reuse simulation.py pattern)
# ---------------------------------------------------------------------------

def _find_freecad_python() -> str | None:
    """Find FreeCAD's bundled Python executable."""
    fc_path = os.environ.get("FREECAD_PATH")
    if fc_path:
        python = str(Path(fc_path) / "python.exe")
        if Path(python).exists():
            return python

    common_paths = [
        os.path.expanduser(r"~\AppData\Local\Programs\FreeCAD 1.1\bin"),
        os.path.expanduser(r"~\AppData\Local\Programs\FreeCAD 1.0\bin"),
        r"C:\Program Files\FreeCAD 1.1\bin",
        r"C:\Program Files\FreeCAD 1.0\bin",
        r"C:\Program Files\FreeCAD\bin",
    ]
    for p in common_paths:
        python = str(Path(p) / "python.exe")
        if Path(python).exists():
            return python

    return None


def _run_freecad_script(script: str, timeout: int = 120) -> str:
    """Execute a Python script using FreeCAD's bundled Python."""
    fc_python = _find_freecad_python()
    if not fc_python:
        raise RuntimeError(
            "FreeCAD not found. Install with: winget install FreeCAD\n"
            "Or set FREECAD_PATH to FreeCAD's bin directory."
        )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(script)
        script_path = f.name

    try:
        result = subprocess.run(
            [fc_python, script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode != 0:
            raise RuntimeError(f"FreeCAD script error:\n{result.stderr}")
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        raise RuntimeError("FreeCAD script timed out")
    finally:
        Path(script_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Script builders
# ---------------------------------------------------------------------------

def _build_forward_kinematics_script(
    document_path: str,
    joint_angles: dict[str, float],
) -> str:
    """Build FreeCAD script for forward kinematics computation."""
    angles_json = json.dumps(joint_angles)
    return f'''
import FreeCAD
import json
import math

doc = FreeCAD.openDocument(r"{document_path}")
if not doc:
    raise RuntimeError("Failed to open document")

# Find assembly objects with Placement
objects_info = []
for obj in doc.Objects:
    if hasattr(obj, "Placement"):
        pl = obj.Placement
        pos = pl.Base
        rot = pl.Rotation
        objects_info.append({{
            "name": obj.Name,
            "label": obj.Label,
            "position": [pos.x, pos.y, pos.z],
            "rotation_axis": list(rot.Axis) if hasattr(rot, "Axis") else [0, 0, 1],
            "rotation_angle": math.degrees(rot.Angle) if hasattr(rot, "Angle") else 0,
        }})

# Apply joint angles to named objects
angles = json.loads(r'{angles_json}')
for obj_name, angle_deg in angles.items():
    obj = doc.getObject(obj_name)
    if not obj:
        for o in doc.Objects:
            if o.Label == obj_name:
                obj = o
                break
    if obj and hasattr(obj, "Placement"):
        import FreeCAD as FC
        pl = obj.Placement
        base = pl.Base
        # Determine rotation axis from object properties or naming convention
        name_lower = obj_name.lower()
        axis = FC.Vector(0, 0, 1)  # default Z
        if "_x" in name_lower or "pitch" in name_lower or "shoulder" in name_lower:
            axis = FC.Vector(1, 0, 0)
        elif "_y" in name_lower or "yaw" in name_lower or "base" in name_lower:
            axis = FC.Vector(0, 1, 0)
        # else: default Z for "roll", "wrist", "elbow", "knee", etc.
        new_rot = FC.Rotation(axis, angle_deg)
        obj.Placement = FC.Placement(base, new_rot)

doc.recompute()

# Collect updated positions
for info in objects_info:
    obj = doc.getObject(info["name"])
    if obj and hasattr(obj, "Placement"):
        pos = obj.Placement.Base
        info["position_after"] = [pos.x, pos.y, pos.z]

print(json.dumps({{"objects": objects_info, "joint_angles": angles}}, indent=2))
'''


def _build_range_check_script(
    document_path: str,
    joint_name: str,
    joint_type: str,
    angle_range: list[float],
    steps: int,
) -> str:
    """Build FreeCAD script for joint range scanning."""
    min_val, max_val = angle_range
    return f'''
import FreeCAD
import json
import math

doc = FreeCAD.openDocument(r"{document_path}")
if not doc:
    raise RuntimeError("Failed to open document")

obj = doc.getObject("{joint_name}")
if not obj:
    # Try by label
    for o in doc.Objects:
        if o.Label == "{joint_name}":
            obj = o
            break

if not obj:
    print(json.dumps({{"error": "Object not found: {joint_name}"}}))
else:
    positions = []
    step_size = ({max_val} - {min_val}) / max({steps} - 1, 1)
    # Determine rotation axis based on joint type
    name_lower = "{joint_name}".lower()
    jtype = "{joint_type}".lower()
    import FreeCAD as FC
    if jtype == "prismatic":
        # Linear joint: translate along axis
        axis_vec = FC.Vector(0, 0, 1)
        if "_x" in name_lower:
            axis_vec = FC.Vector(1, 0, 0)
        elif "_y" in name_lower:
            axis_vec = FC.Vector(0, 1, 0)
    else:
        # Revolute / spherical: rotate around axis
        axis_vec = FC.Vector(0, 0, 1)  # default Z
        if "_x" in name_lower or "pitch" in name_lower or "shoulder" in name_lower:
            axis_vec = FC.Vector(1, 0, 0)
        elif "_y" in name_lower or "yaw" in name_lower or "base" in name_lower:
            axis_vec = FC.Vector(0, 1, 0)

    for i in range({steps}):
        angle = {min_val} + i * step_size
        if hasattr(obj, "Placement"):
            base = obj.Placement.Base
            if jtype == "prismatic":
                new_base = base.add(axis_vec * angle)
                obj.Placement = FC.Placement(new_base, obj.Placement.Rotation)
            else:
                new_rot = FC.Rotation(axis_vec, angle)
                obj.Placement = FC.Placement(base, new_rot)
            doc.recompute()
            pos = obj.Placement.Base
            positions.append({{
                "angle": round(angle, 2),
                "position": [round(pos.x, 4), round(pos.y, 4), round(pos.z, 4)],
            }})

    print(json.dumps({{
        "joint": "{joint_name}",
        "joint_type": "{joint_type}",
        "range": [{min_val}, {max_val}],
        "positions": positions,
    }}, indent=2))
'''


def _build_trajectory_script(
    document_path: str,
    start_angles: dict[str, float],
    end_angles: dict[str, float],
    steps: int,
) -> str:
    """Build FreeCAD script for trajectory interpolation."""
    start_json = json.dumps(start_angles)
    end_json = json.dumps(end_angles)
    return f'''
import FreeCAD
import json
import math

doc = FreeCAD.openDocument(r"{document_path}")
if not doc:
    raise RuntimeError("Failed to open document")

start = json.loads(r'{start_json}')
end = json.loads(r'{end_json}')
joints = list(start.keys())

waypoints = []
for i in range({steps} + 1):
    t = i / {steps}
    current = {{}}
    for j in joints:
        current[j] = start[j] + t * (end.get(j, start[j]) - start[j])

    # Apply current angles
    for j_name, angle in current.items():
        obj = doc.getObject(j_name)
        if not obj:
            for o in doc.Objects:
                if o.Label == j_name:
                    obj = o
                    break
        if obj and hasattr(obj, "Placement"):
            import FreeCAD as FC
            base = obj.Placement.Base
            name_lower = j_name.lower()
            axis = FC.Vector(0, 0, 1)
            if "_x" in name_lower or "pitch" in name_lower or "shoulder" in name_lower:
                axis = FC.Vector(1, 0, 0)
            elif "_y" in name_lower or "yaw" in name_lower or "base" in name_lower:
                axis = FC.Vector(0, 1, 0)
            new_rot = FC.Rotation(axis, angle)
            obj.Placement = FC.Placement(base, new_rot)

    doc.recompute()

    # Collect positions
    positions = {{}}
    for j_name in joints:
        obj = doc.getObject(j_name)
        if not obj:
            for o in doc.Objects:
                if o.Label == j_name:
                    obj = o
                    break
        if obj and hasattr(obj, "Placement"):
            pos = obj.Placement.Base
            positions[j_name] = [round(pos.x, 4), round(pos.y, 4), round(pos.z, 4)]

    waypoints.append({{
        "step": i,
        "t": round(t, 4),
        "angles": {{k: round(v, 2) for k, v in current.items()}},
        "positions": positions,
    }})

print(json.dumps({{"waypoints": waypoints, "total_steps": {steps}}}, indent=2))
'''


# ---------------------------------------------------------------------------
# Motion JSON parser
# ---------------------------------------------------------------------------

def _parse_motion_json(raw: str) -> dict[str, Any]:
    """Parse motion analysis result from FreeCAD script output."""
    import json as _json

    # Try extracting JSON from output
    json_match = re.search(r'\{[\s\S]*\}', raw)
    if json_match:
        try:
            data = _json.loads(json_match.group())
            return data
        except (_json.JSONDecodeError, ValueError):
            pass

    return {"raw": raw}


# ===========================================================================
# Tool: motion_range
# ===========================================================================

class MotionRangeTool(Tool):
    """Analyze joint motion range and reachable workspace."""

    name = "motion_range"
    description = (
        "Analyze joint motion range and reachable workspace for a FreeCAD assembly. "
        "Scans joint angles across a specified range and reports end-effector positions. "
        "Requires FreeCAD."
    )

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "document_path": {
                        "type": "string",
                        "description": "Path to .FCStd file with assembly",
                    },
                    "joint_name": {
                        "type": "string",
                        "description": "Name of the joint/object to scan",
                    },
                    "joint_type": {
                        "type": "string",
                        "enum": ["revolute", "prismatic", "fixed", "spherical"],
                        "description": "Joint type (default: revolute)",
                    },
                    "angle_range": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Motion range [min, max] in degrees or mm (default: [-180, 180])",
                    },
                    "steps": {
                        "type": "integer",
                        "description": "Number of scan steps (default: 36)",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Script timeout in seconds (default: 120)",
                    },
                },
                "required": ["document_path", "joint_name"],
            },
        )

    def execute(
        self,
        *,
        document_path: str,
        joint_name: str,
        joint_type: str = "revolute",
        angle_range: list[float] | None = None,
        steps: int = 36,
        timeout: int = 120,
        **kwargs: Any,
    ) -> str:
        if not Path(document_path).exists():
            return f"Error: Document not found: {document_path}"

        if not document_path.endswith(".FCStd"):
            return f"Error: File must be a .FCStd document, got: {document_path}"

        # Validate joint type
        jt = JOINT_TYPES.get(joint_type)
        if not jt:
            return f"Error: Unknown joint type '{joint_type}'. Valid: {list(JOINT_TYPES.keys())}"

        if angle_range is None:
            angle_range = jt.default_range

        if len(angle_range) != 2:
            return "Error: angle_range must be [min, max]"

        # Check FreeCAD availability
        fc_python = _find_freecad_python()
        if not fc_python:
            return "Error: FreeCAD not found. Install with: winget install FreeCAD"

        script = _build_range_check_script(
            document_path=document_path,
            joint_name=joint_name,
            joint_type=joint_type,
            angle_range=angle_range,
            steps=steps,
        )

        try:
            output = _run_freecad_script(script, timeout=timeout)
            parsed = _parse_motion_json(output)

            if "error" in parsed:
                return f"[Motion Range Analysis]\nError: {parsed['error']}"

            positions = parsed.get("positions", [])
            lines = [
                f"[Motion Range Analysis]",
                f"Document: {document_path}",
                f"Joint: {joint_name} ({joint_type})",
                f"Range: [{angle_range[0]}, {angle_range[1]}]",
                f"Steps: {len(positions)}",
                "",
                "--- End-Effector Positions ---",
            ]
            for p in positions:
                pos = p.get("position", [0, 0, 0])
                lines.append(f"  angle={p.get('angle', 0):>8.2f} -> pos=({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")

            if positions:
                xs = [p["position"][0] for p in positions]
                ys = [p["position"][1] for p in positions]
                zs = [p["position"][2] for p in positions]
                lines.append("")
                lines.append(f"Reachable workspace (approx):")
                lines.append(f"  X: [{min(xs):.4f}, {max(xs):.4f}]")
                lines.append(f"  Y: [{min(ys):.4f}, {max(ys):.4f}]")
                lines.append(f"  Z: [{min(zs):.4f}, {max(zs):.4f}]")

            return "\n".join(lines)

        except RuntimeError as e:
            return f"Error running motion range analysis: {e}"


# ===========================================================================
# Tool: motion_trajectory
# ===========================================================================

class MotionTrajectoryTool(Tool):
    """Plan joint-space linear interpolation trajectory."""

    name = "motion_trajectory"
    description = (
        "Plan a joint-space linear interpolation trajectory between two configurations. "
        "Returns waypoint positions for each step. Requires FreeCAD."
    )

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "document_path": {
                        "type": "string",
                        "description": "Path to .FCStd file with assembly",
                    },
                    "start_angles": {
                        "type": "object",
                        "description": "Starting joint angles: {object_name: angle_degrees}",
                    },
                    "end_angles": {
                        "type": "object",
                        "description": "Target joint angles: {object_name: angle_degrees}",
                    },
                    "steps": {
                        "type": "integer",
                        "description": "Number of interpolation steps (default: 10)",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Script timeout in seconds (default: 120)",
                    },
                },
                "required": ["document_path", "start_angles", "end_angles"],
            },
        )

    def execute(
        self,
        *,
        document_path: str,
        start_angles: dict[str, float],
        end_angles: dict[str, float],
        steps: int = 10,
        timeout: int = 120,
        **kwargs: Any,
    ) -> str:
        if not Path(document_path).exists():
            return f"Error: Document not found: {document_path}"

        if not document_path.endswith(".FCStd"):
            return f"Error: File must be a .FCStd document, got: {document_path}"

        if steps < 1:
            return "Error: steps must be >= 1"

        # Check FreeCAD availability
        fc_python = _find_freecad_python()
        if not fc_python:
            return "Error: FreeCAD not found. Install with: winget install FreeCAD"

        script = _build_trajectory_script(
            document_path=document_path,
            start_angles=start_angles,
            end_angles=end_angles,
            steps=steps,
        )

        try:
            output = _run_freecad_script(script, timeout=timeout)
            parsed = _parse_motion_json(output)

            waypoints = parsed.get("waypoints", [])
            lines = [
                f"[Motion Trajectory]",
                f"Document: {document_path}",
                f"Steps: {len(waypoints) - 1}",
                f"Start: {start_angles}",
                f"End: {end_angles}",
                "",
                "--- Waypoints ---",
            ]

            for wp in waypoints:
                step = wp.get("step", 0)
                t = wp.get("t", 0)
                angles = wp.get("angles", {})
                positions = wp.get("positions", {})
                angle_str = ", ".join(f"{k}={v:.1f}" for k, v in angles.items())
                pos_strs = []
                for jn, pos in positions.items():
                    pos_strs.append(f"{jn}: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
                lines.append(f"  step {step:>3d} (t={t:.2f}): angles=[{angle_str}]")
                for ps in pos_strs:
                    lines.append(f"          {ps}")

            return "\n".join(lines)

        except RuntimeError as e:
            return f"Error running trajectory planning: {e}"


# ===========================================================================
# Tool: motion_vlm_analyze
# ===========================================================================

class MotionVLMAnalyzeTool(Tool):
    """Capture motion visualization screenshot and analyze with VLM."""

    name = "motion_vlm_analyze"
    description = (
        "Capture FreeCAD window screenshot showing motion/assembly and analyze with VLM. "
        "Returns structured analysis of kinematic configuration, joint positions, and suggestions."
    )

    def __init__(self, router=None, screenshot_dir: str = "") -> None:
        self.router = router
        self.screenshot_dir = screenshot_dir

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "document_path": {
                        "type": "string",
                        "description": "Path to assembly document (for context)",
                    },
                    "analysis_type": {
                        "type": "string",
                        "enum": ["kinematic", "range", "trajectory"],
                        "description": "Type of motion analysis (default: kinematic)",
                    },
                    "window_title": {
                        "type": "string",
                        "description": "FreeCAD window title (default: 'FreeCAD')",
                    },
                    "detail": {
                        "type": "string",
                        "description": "VLM detail level: fast, standard, detailed, maximum (default: standard)",
                    },
                },
                "required": [],
            },
        )

    def execute(
        self,
        *,
        document_path: str = "",
        analysis_type: str = "kinematic",
        window_title: str = "FreeCAD",
        detail: str = "standard",
        **kwargs: Any,
    ) -> str:
        if not self.router:
            return "Error: VLM router not configured. Cannot analyze screenshot."

        try:
            import ctypes
            import ctypes.wintypes
            from PIL import ImageGrab
            from .screen import _find_windows_by_title

            matches = _find_windows_by_title(window_title)
            if not matches:
                return f"Error: No window found matching '{window_title}'. Open FreeCAD first."

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
                filepath = save_dir / f"motion_vlm_{int(time.time())}.png"

                img = ImageGrab.grab(bbox=(left, top, right, bottom))
                img.save(str(filepath))

                analyze_prompt = (
                    "You are a mechanical engineering expert analyzing a motion/assembly visualization.\n\n"
                    f"Analysis type: {analysis_type}\n"
                    f"Document: {document_path}\n\n"
                    "Analyze this FreeCAD assembly/motion screenshot and respond with EXACTLY this JSON format "
                    "(no markdown, no backticks, raw JSON only):\n"
                    '{"joint_count": number, "joint_types": ["type1", ...], '
                    '"configuration": "description of current assembly configuration", '
                    '"reachable_workspace": "description of workspace reachability", '
                    '"interference_risk": "any collision risks, or None", '
                    '"suggestion": "design improvement suggestion, or None"}\n\n'
                    "Consider:\n"
                    "- How many joints/links are visible?\n"
                    "- What types of joints (revolute, prismatic)?\n"
                    "- Is the current configuration feasible?\n"
                    "- Are there any obvious collision risks?\n"
                    "- What is the approximate reachable workspace?"
                )

                from ..models.router import VisionDetail

                try:
                    vd = VisionDetail(detail)
                except ValueError:
                    vd = None

                result = self.router.vision(str(filepath), analyze_prompt, detail=vd)

                return (
                    f"[Motion VLM Analysis - Window: '{full_title}']\n"
                    f"Analysis type: {analysis_type}\n"
                    f"\n--- VLM Output ---\n{result}"
                )

            return f"Error: All matching windows for '{window_title}' have invalid dimensions"
        except Exception as e:
            return f"Error: {e}"


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_motion_tools(
    registry: Any,
    router: Any = None,
    screenshot_dir: str = "",
) -> None:
    """Register all motion simulation tools."""
    registry.register(MotionRangeTool())
    registry.register(MotionTrajectoryTool())
    registry.register(MotionVLMAnalyzeTool(router=router, screenshot_dir=screenshot_dir))
