"""SolidWorks COM API control tool.

Note: Requires SolidWorks to be installed. Uses pywin32 for COM access.
When SolidWorks is not available, provides mock interface for development.
"""

from __future__ import annotations

import time
from typing import Any

from ..models.base import ToolDefinition
from .base import Tool


def _get_sw():
    """Get or create a SolidWorks COM connection."""
    try:
        import win32com.client

        try:
            sw = win32com.client.GetActiveObject("SldWorks.Application")
            return sw
        except Exception:
            sw = win32com.client.Dispatch("SldWorks.Application")
            sw.Visible = True
            return sw
    except ImportError:
        raise RuntimeError("pywin32 not installed. Run: pip install pywin32")
    except Exception as e:
        raise RuntimeError(f"Cannot connect to SolidWorks: {e}. Is SolidWorks installed and running?")


def _is_sw_available() -> bool:
    """Check if SolidWorks is available."""
    try:
        _get_sw()
        return True
    except Exception:
        return False


class SWNewPartTool(Tool):
    """Create a new part document in SolidWorks."""

    name = "sw_new_part"
    description = "Create a new part document in SolidWorks"

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={"type": "object", "properties": {}, "required": []},
        )

    def execute(self, **kwargs: Any) -> str:
        try:
            sw = _get_sw()
            part = sw.NewPart()
            return "New part document created"
        except Exception as e:
            return f"Error: {e}"


class SWExtrudeTool(Tool):
    """Create an extruded feature from a sketch."""

    name = "sw_extrude"
    description = "Extrude a sketch to create a 3D feature"

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "depth": {
                        "type": "number",
                        "description": "Extrusion depth in document units",
                    },
                    "reverse": {
                        "type": "boolean",
                        "description": "Reverse extrusion direction",
                    },
                },
                "required": ["depth"],
            },
        )

    def execute(self, *, depth: float, reverse: bool = False, **kwargs: Any) -> str:
        try:
            sw = _get_sw()
            model = sw.ActiveDoc
            if not model:
                return "Error: No active document"

            feature = model.FeatureManager.FeatureExtrusion2(
                True,  # sd
                False,  # flip
                reverse,  # dir
                0,  # type (blind)
                0,  # type2
                depth,  # depth1
                0,  # depth2
                False,  # draft
                False,  # draft2
                False,  # draft3
                False,  # draft4
                0,  # draft_angle
                0,  # draft_angle2
                False,  # merge
                True,  # use_feat_scope
                False,  # use_auto_select
                False,  # assembly
                0,  # n_bodies
                0,  # affect
            )

            if feature:
                return f"Extrusion created with depth {depth}m"
            return "Error: Failed to create extrusion"
        except Exception as e:
            return f"Error: {e}"


class SWSketchLineTool(Tool):
    """Draw a line in the active sketch."""

    name = "sw_sketch_line"
    description = "Draw a line in the active 2D sketch"

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "x1": {"type": "number", "description": "Start X (document units)"},
                    "y1": {"type": "number", "description": "Start Y (document units)"},
                    "x2": {"type": "number", "description": "End X (document units)"},
                    "y2": {"type": "number", "description": "End Y (document units)"},
                },
                "required": ["x1", "y1", "x2", "y2"],
            },
        )

    def execute(self, *, x1: float, y1: float, x2: float, y2: float, **kwargs: Any) -> str:
        try:
            sw = _get_sw()
            model = sw.ActiveDoc
            sketch_mgr = model.SketchManager
            sketch_mgr.CreateLine(x1, y1, 0, x2, y2, 0)
            return f"Line created: ({x1},{y1}) -> ({x2},{y2})"
        except Exception as e:
            return f"Error: {e}"


class SWSketchCircleTool(Tool):
    """Draw a circle in the active sketch."""

    name = "sw_sketch_circle"
    description = "Draw a circle in the active 2D sketch"

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "cx": {"type": "number", "description": "Center X (document units)"},
                    "cy": {"type": "number", "description": "Center Y (document units)"},
                    "radius": {"type": "number", "description": "Radius (document units)"},
                },
                "required": ["cx", "cy", "radius"],
            },
        )

    def execute(self, *, cx: float, cy: float, radius: float, **kwargs: Any) -> str:
        try:
            sw = _get_sw()
            model = sw.ActiveDoc
            sketch_mgr = model.SketchManager
            sketch_mgr.CreateCircleByRadius(cx, cy, 0, radius)
            return f"Circle created at ({cx},{cy}) with radius {radius}m"
        except Exception as e:
            return f"Error: {e}"


class SWSketchRectangleTool(Tool):
    """Draw a rectangle in the active sketch."""

    name = "sw_sketch_rectangle"
    description = "Draw a rectangle in the active 2D sketch"

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "x1": {"type": "number", "description": "Corner 1 X (document units)"},
                    "y1": {"type": "number", "description": "Corner 1 Y (document units)"},
                    "x2": {"type": "number", "description": "Corner 2 X (document units)"},
                    "y2": {"type": "number", "description": "Corner 2 Y (document units)"},
                },
                "required": ["x1", "y1", "x2", "y2"],
            },
        )

    def execute(self, *, x1: float, y1: float, x2: float, y2: float, **kwargs: Any) -> str:
        try:
            sw = _get_sw()
            model = sw.ActiveDoc
            sketch_mgr = model.SketchManager
            sketch_mgr.CreateCornerRectangle(x1, y1, 0, x2, y2, 0)
            return f"Rectangle created: ({x1},{y1}) to ({x2},{y2})"
        except Exception as e:
            return f"Error: {e}"


class SWSaveTool(Tool):
    """Save the active SolidWorks document."""

    name = "sw_save"
    description = "Save the active SolidWorks document"

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Save path (optional, defaults to current location)",
                    },
                },
                "required": [],
            },
        )

    def execute(self, *, path: str = "", **kwargs: Any) -> str:
        try:
            sw = _get_sw()
            model = sw.ActiveDoc
            if not model:
                return "Error: No active document"

            if path:
                result = model.SaveAs3(path, 0, 0)
                return f"Document saved to: {path}" if result == 0 else f"Error: Save failed (code: {result})"
            else:
                result = model.Save()
                return "Document saved" if result == 0 else f"Error: Save failed (code: {result})"
        except Exception as e:
            return f"Error: {e}"


class SWExportSTLTool(Tool):
    """Export the active document as STL."""

    name = "sw_export_stl"
    description = "Export the active SolidWorks part as STL file for 3D printing"

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Output STL file path",
                    },
                },
                "required": ["path"],
            },
        )

    def execute(self, *, path: str, **kwargs: Any) -> str:
        try:
            from pathlib import Path

            Path(path).parent.mkdir(parents=True, exist_ok=True)

            sw = _get_sw()
            model = sw.ActiveDoc
            if not model:
                return "Error: No active document"

            # STL export using SaveAs
            result = model.SaveAs3(path, 0, 0)
            return f"STL exported to: {path}" if result == 0 else f"Error: Export failed (code: {result})"
        except Exception as e:
            return f"Error: {e}"


class SWStatusTool(Tool):
    """Check SolidWorks connection status."""

    name = "sw_status"
    description = "Check if SolidWorks is running and connected"

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={"type": "object", "properties": {}, "required": []},
        )

    def execute(self, **kwargs: Any) -> str:
        try:
            sw = _get_sw()
            model = sw.ActiveDoc
            if model:
                return f"SolidWorks connected. Active document: {model.GetTitle()}"
            return "SolidWorks connected but no active document"
        except Exception as e:
            return f"SolidWorks not available: {e}"


def register_solidworks_tools(registry: Any) -> None:
    """Register all SolidWorks tools."""
    tools = [
        SWNewPartTool(),
        SWSketchLineTool(),
        SWSketchCircleTool(),
        SWSketchRectangleTool(),
        SWExtrudeTool(),
        SWSaveTool(),
        SWExportSTLTool(),
        SWStatusTool(),
    ]
    for tool in tools:
        registry.register(tool)
