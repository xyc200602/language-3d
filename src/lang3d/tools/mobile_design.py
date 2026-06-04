"""Mobile base design tools — chassis design and differential drive simulation.

Task 46: Provides tools for automated mobile base design from requirements
and differential drive trajectory simulation.
"""

from __future__ import annotations

import json
import math
from typing import Any

from ..knowledge.mobile_base import (
    DifferentialDriveKinematics,
    OmnidirectionalKinematics,
    design_mobile_base,
    MOBILE_BASE_TEMPLATES,
    DC_MOTOR_CATALOG,
)
from ..models.base import ToolDefinition
from .base import Tool


class MobileBaseDesignTool(Tool):
    """Design a complete mobile base from high-level requirements."""

    name = "mobile_base_design"
    description = (
        "Design a mobile robot base from requirements (payload, speed, grade, runtime). "
        "Returns chassis parameters, motor selection, battery sizing, kinematic properties, "
        "and a parts list. Supports differential 4W, differential 2W, and Mecanum drive types."
    )

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "payload_kg": {
                        "type": "number",
                        "description": "Payload mass in kg (default: 5.0)",
                    },
                    "max_speed_mm_s": {
                        "type": "number",
                        "description": "Maximum speed in mm/s (default: 500)",
                    },
                    "max_grade_deg": {
                        "type": "number",
                        "description": "Maximum climbable slope in degrees (default: 10)",
                    },
                    "runtime_hours": {
                        "type": "number",
                        "description": "Desired operating time in hours (default: 2.0)",
                    },
                    "drive_type": {
                        "type": "string",
                        "enum": ["differential_4w", "differential_2w", "mecanum"],
                        "description": "Drive type (default: differential_4w)",
                    },
                },
                "required": [],
            },
        )

    def execute(
        self,
        *,
        payload_kg: float = 5.0,
        max_speed_mm_s: float = 500.0,
        max_grade_deg: float = 10.0,
        runtime_hours: float = 2.0,
        drive_type: str = "differential_4w",
        **kwargs: Any,
    ) -> str:
        result = design_mobile_base(
            payload_kg=payload_kg,
            max_speed_mm_s=max_speed_mm_s,
            max_grade_deg=max_grade_deg,
            runtime_hours=runtime_hours,
            drive_type=drive_type,
        )
        return json.dumps(result, indent=2, ensure_ascii=False)


class DifferentialDriveSimTool(Tool):
    """Simulate differential drive trajectory."""

    name = "differential_drive_sim"
    description = (
        "Simulate a differential drive robot trajectory given left/right wheel "
        "velocities over time steps. Returns the path (x, y, theta) at each step."
    )

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "v_left_mm_s": {
                        "type": "number",
                        "description": "Left wheel velocity in mm/s",
                    },
                    "v_right_mm_s": {
                        "type": "number",
                        "description": "Right wheel velocity in mm/s",
                    },
                    "dt_s": {
                        "type": "number",
                        "description": "Time step in seconds (default: 0.1)",
                    },
                    "steps": {
                        "type": "integer",
                        "description": "Number of simulation steps (default: 50)",
                    },
                    "track_width_mm": {
                        "type": "number",
                        "description": "Track width in mm (default: 300)",
                    },
                },
                "required": ["v_left_mm_s", "v_right_mm_s"],
            },
        )

    def execute(
        self,
        *,
        v_left_mm_s: float,
        v_right_mm_s: float,
        dt_s: float = 0.1,
        steps: int = 50,
        track_width_mm: float = 300.0,
        **kwargs: Any,
    ) -> str:
        kin = DifferentialDriveKinematics(track_width_mm=track_width_mm)

        x, y, theta = 0.0, 0.0, 0.0
        path = [{"step": 0, "x": round(x, 2), "y": round(y, 2),
                 "theta_deg": round(math.degrees(theta), 2)}]

        v_linear, omega = kin.forward_kinematics(v_left_mm_s, v_right_mm_s)

        for i in range(1, steps + 1):
            x += v_linear * math.cos(theta) * dt_s
            y += v_linear * math.sin(theta) * dt_s
            theta += omega * dt_s
            path.append({
                "step": i,
                "x": round(x, 2),
                "y": round(y, 2),
                "theta_deg": round(math.degrees(theta), 2),
            })

        result = {
            "v_left_mm_s": v_left_mm_s,
            "v_right_mm_s": v_right_mm_s,
            "v_linear_mm_s": round(v_linear, 2),
            "omega_rad_s": round(omega, 4),
            "track_width_mm": track_width_mm,
            "dt_s": dt_s,
            "steps": steps,
            "final_position": path[-1],
            "path": path if steps <= 20 else path[::max(1, steps // 20)] + [path[-1]],
        }
        return json.dumps(result, indent=2, ensure_ascii=False)


def register_mobile_design_tools(registry: Any) -> None:
    """Register all mobile base design tools."""
    tools = [
        MobileBaseDesignTool(),
        DifferentialDriveSimTool(),
    ]
    for tool in tools:
        registry.register(tool)
