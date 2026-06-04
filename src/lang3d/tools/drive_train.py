"""Drive train design tool — auto-generate drive system assembly definitions.

Task 47: Generates the kinematic chain from motor → reducer → wheel for
differential drive, Mecanum, and other mobile base configurations.
"""

from __future__ import annotations

import json
from typing import Any

from ..knowledge.mobile_base import DC_MOTOR_CATALOG, DCMotorSpec
from ..models.base import ToolDefinition
from .base import Tool


def _generate_differential_drive_train(
    wheel_count: int = 4,
    motor_type: str = "JGA25-370",
    drive_side: str = "left",
) -> dict[str, Any]:
    """Generate drive train for one side of a differential drive base.

    Chain: motor → coupler → wheel (with motor_mount as structural support).
    """
    motor = DC_MOTOR_CATALOG.get(motor_type)
    motor_name = motor.name if motor else motor_type

    shaft_d = motor.shaft_diameter_mm if motor else 6.0

    prefix = "l" if drive_side == "left" else "r"

    parts = [
        {
            "name": f"motor_{prefix}",
            "category": "actuator",
            "description": f"{motor_name} {'左' if drive_side == 'left' else '右'}驱动电机",
            "material": "metal",
            "dimensions": {"length": 50, "width": 25, "height": 25},
        },
        {
            "name": f"coupler_{prefix}",
            "category": "coupling",
            "description": f"联轴器 ({shaft_d}mm 轴径)",
            "material": "Aluminum",
            "dimensions": {"diameter": 15, "height": 20},
        },
        {
            "name": f"wheel_{prefix}",
            "category": "wheel",
            "description": f"{'左' if drive_side == 'left' else '右'}驱动轮",
            "material": "TPU",
            "dimensions": {"diameter": 80, "width": 25},
        },
        {
            "name": f"motor_mount_{prefix}",
            "category": "structural",
            "description": f"{'左' if drive_side == 'left' else '右'}电机支架",
            "material": "PLA",
            "dimensions": {"length": 50, "width": 30, "height": 25},
        },
    ]

    joints = [
        {
            "type": "fixed",
            "parent": f"motor_mount_{prefix}",
            "child": f"motor_{prefix}",
            "parent_anchor": "top",
            "child_anchor": "bottom",
            "description": f"电机固定在支架上",
        },
        {
            "type": "fixed",
            "parent": f"motor_{prefix}",
            "child": f"coupler_{prefix}",
            "parent_anchor": "front",
            "child_anchor": "back",
            "description": f"联轴器连接电机轴",
        },
        {
            "type": "revolute",
            "parent": f"coupler_{prefix}",
            "child": f"wheel_{prefix}",
            "parent_anchor": "front",
            "child_anchor": "back",
            "axis": "y",
            "range_deg": [-360, 360],
            "description": f"轮子旋转",
        },
    ]

    return {"parts": parts, "joints": joints}


def drive_train_design(
    wheel_count: int = 4,
    drive_type: str = "differential",
    motor_type: str = "JGA25-370",
) -> dict[str, Any]:
    """Generate a complete drive train assembly definition.

    Creates the motor → coupler → wheel kinematic chain for each driven wheel.
    For differential drive with 4 wheels, typically 2 driven + 2 caster,
    or 4 driven (4WD).

    Returns an assembly definition with parts and joints.
    """
    all_parts: list[dict] = []
    all_joints: list[dict] = []

    if drive_type in ("differential", "differential_4w", "differential_2w"):
        # Determine driven wheels
        if drive_type == "differential_2w":
            driven_sides = ["left", "right"]
        else:
            driven_sides = ["left", "right"]

        for side in driven_sides:
            side_train = _generate_differential_drive_train(
                wheel_count=wheel_count,
                motor_type=motor_type,
                drive_side=side,
            )
            all_parts.extend(side_train["parts"])
            all_joints.extend(side_train["joints"])

    elif drive_type == "mecanum":
        # 4 independent drives for Mecanum
        for side in ["left", "right"]:
            side_train = _generate_differential_drive_train(
                wheel_count=4,
                motor_type=motor_type,
                drive_side=side,
            )
            all_parts.extend(side_train["parts"])
            all_joints.extend(side_train["joints"])

    motor = DC_MOTOR_CATALOG.get(motor_type)

    result = {
        "name": f"{drive_type}_drive_train",
        "drive_type": drive_type,
        "motor_type": motor_type,
        "motor_specs": {
            "name": motor.name,
            "voltage": motor.nominal_voltage,
            "rated_speed_rpm": motor.rated_speed_rpm,
            "rated_torque_kg_cm": motor.rated_torque_kg_cm,
        } if motor else {},
        "parts": all_parts,
        "joints": all_joints,
        "total_parts": len(all_parts),
        "total_joints": len(all_joints),
    }

    return result


class DriveTrainDesignTool(Tool):
    """Generate drive train assembly definition for mobile bases."""

    name = "drive_train_design"
    description = (
        "Generate a drive train assembly definition (motor → coupler → wheel) "
        "for mobile robot bases. Supports differential and Mecanum drive types. "
        "Returns parts list and joint definitions ready for assembly_solve."
    )

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "wheel_count": {
                        "type": "integer",
                        "description": "Number of wheels (default: 4)",
                    },
                    "drive_type": {
                        "type": "string",
                        "enum": ["differential", "differential_4w", "differential_2w", "mecanum"],
                        "description": "Drive type (default: differential)",
                    },
                    "motor_type": {
                        "type": "string",
                        "description": "Motor catalog key (default: JGA25-370)",
                    },
                },
                "required": [],
            },
        )

    def execute(
        self,
        *,
        wheel_count: int = 4,
        drive_type: str = "differential",
        motor_type: str = "JGA25-370",
        **kwargs: Any,
    ) -> str:
        result = drive_train_design(
            wheel_count=wheel_count,
            drive_type=drive_type,
            motor_type=motor_type,
        )
        return json.dumps(result, indent=2, ensure_ascii=False)


def register_drive_train_tools(registry: Any) -> None:
    """Register drive train design tools."""
    registry.register(DriveTrainDesignTool())
