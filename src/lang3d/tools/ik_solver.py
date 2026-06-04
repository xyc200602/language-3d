"""Inverse kinematics solver for robotic assemblies.

Provides two approaches:
  - Analytic IK for 3-DOF (or 4-DOF with wrist) planar/spherical arms
  - CCD (Cyclic Coordinate Descent) numerical solver for arbitrary chains

Both approaches forward-verify their solutions via FK (AssemblySolver).

Tools:
  ik_solve  - Solve inverse kinematics for a target end-effector position
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

from ..knowledge.mechanics import Assembly, Joint, Part
from ..models.base import ToolDefinition
from .assembly_solver import AssemblySolver
from .base import Tool


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class IKResult:
    """Result of an IK solve attempt."""
    joint_angles: dict[str, float]   # part_name -> angle_degrees
    end_effector: list[float]        # [x, y, z] achieved position
    error_mm: float                  # distance to target
    reachable: bool                  # True if error < tolerance
    method: str                      # "analytic" or "ccd"
    iterations: int                  # 0 for analytic, actual for CCD


@dataclass
class LinkSegment:
    """A link in the kinematic chain with its length."""
    name: str
    length: float
    joint_type: str
    parent_anchor: str
    axis: str  # "auto", "x", "y", "z"


# ---------------------------------------------------------------------------
# Chain extraction
# ---------------------------------------------------------------------------

def _extract_chain(assembly: Assembly) -> tuple[list[LinkSegment], float]:
    """Extract the kinematic chain and base height from assembly.

    Returns (links, base_height) where links are in order from base to tip,
    and base_height is the Z offset of the first revolute joint above origin.
    """
    solver = AssemblySolver(assembly)
    placements_home = solver.solve()

    # Find revolute joints in order
    revolute_joints = [j for j in assembly.joints if j.type == "revolute"]

    # Compute segment lengths from home-position Z differences
    links: list[LinkSegment] = []
    prev_z = 0.0
    for j in revolute_joints:
        child_pos = placements_home.get(j.child, {}).get("position", [0, 0, 0])
        z = child_pos[2]
        seg_len = abs(z - prev_z)
        links.append(LinkSegment(
            name=j.child,
            length=seg_len,
            joint_type=j.type,
            parent_anchor=j.parent_anchor,
            axis=j.axis,
        ))
        prev_z = z

    # Base height = first child Z
    base_height = 0.0
    if revolute_joints:
        first_child = revolute_joints[0].child
        if first_child in placements_home:
            base_height = placements_home[first_child]["position"][2]

    return links, base_height


# ---------------------------------------------------------------------------
# Analytic IK for 3-DOF spherical arm (base yaw + 2 planar links)
# ---------------------------------------------------------------------------

def _analytic_3dof(
    target: tuple[float, float, float],
    links: list[LinkSegment],
    base_height: float,
    joint_limits: dict[str, tuple[float, float]] | None = None,
) -> IKResult | None:
    """Analytic IK for a 3-DOF arm: base yaw + 2 pitch links.

    Assumes:
      - Joint 0: base rotation around Z (yaw)
      - Joint 1: shoulder pitch (rotation in vertical plane)
      - Joint 2: elbow pitch (rotation in vertical plane)

    For >3 revolute joints, uses the last two as the planar arm and
    keeps intermediate joints at 0.
    """
    if len(links) < 2:
        return None

    joint_limits = joint_limits or {}

    # Use the last two revolute links as the planar arm
    # (handles 2, 3, or 4+ DOF by keeping middle joints at 0)
    if len(links) == 2:
        base_link = links[0]
        link1 = links[0]
        link2 = links[1]
    else:
        base_link = links[0]
        link1 = links[1]
        link2 = links[2] if len(links) > 2 else links[1]

    L1 = link1.length
    L2 = link2.length
    if L1 < 0.1 or L2 < 0.1:
        return None

    tx, ty, tz = target

    # Base angle (yaw around Z)
    theta0 = math.degrees(math.atan2(ty, tx))

    # Distance from base axis to target in horizontal plane
    r = math.sqrt(tx * tx + ty * ty)

    # Height relative to shoulder
    z_rel = tz - base_height

    # Distance from shoulder to target
    D = math.sqrt(r * r + z_rel * z_rel)
    if D > L1 + L2:
        # Target unreachable — stretch towards it
        D = L1 + L2 - 0.01
        r_scale = r / max(math.sqrt(r * r + z_rel * z_rel), 1e-10)
        z_scale = z_rel / max(math.sqrt(r * r + z_rel * z_rel), 1e-10)
        r = D * r_scale
        z_rel = D * z_scale
        D = L1 + L2 - 0.01

    if D < abs(L1 - L2):
        D = abs(L1 - L2) + 0.01

    # Cosine law for elbow angle
    cos_elbow = (L1 * L1 + L2 * L2 - D * D) / (2 * L1 * L2)
    cos_elbow = max(-1.0, min(1.0, cos_elbow))
    elbow_angle = math.pi - math.acos(cos_elbow)

    # Shoulder angle
    alpha = math.atan2(z_rel, r)
    beta = math.acos(max(-1.0, min(1.0, (L1 * L1 + D * D - L2 * L2) / (2 * L1 * D))))
    shoulder_angle = alpha + beta

    # Convert to degrees
    theta1 = math.degrees(shoulder_angle)
    theta2 = math.degrees(elbow_angle)

    # Apply joint limits
    angles: dict[str, float] = {}

    # Base joint
    base_name = base_link.name
    lim = joint_limits.get(base_name, (-180, 180))
    theta0 = max(lim[0], min(lim[1], theta0))
    angles[base_name] = round(theta0, 2)

    # Set intermediate joints to 0 (for >3 DOF)
    if len(links) > 2:
        for i in range(1, len(links) - 2):
            angles[links[i].name] = 0.0

    # Shoulder
    shoulder_name = link1.name
    lim = joint_limits.get(shoulder_name, (-180, 180))
    theta1 = max(lim[0], min(lim[1], theta1))
    angles[shoulder_name] = round(theta1, 2)

    # Elbow
    elbow_name = link2.name
    lim = joint_limits.get(elbow_name, (-180, 180))
    theta2 = max(lim[0], min(lim[1], theta2))
    angles[elbow_name] = round(theta2, 2)

    # Extra wrist joints: keep at 0
    for link in links[3:]:
        angles.setdefault(link.name, 0.0)

    return IKResult(
        joint_angles=angles,
        end_effector=[tx, ty, tz],
        error_mm=0.0,  # will be computed by FK verify
        reachable=True,
        method="analytic",
        iterations=0,
    )


# ---------------------------------------------------------------------------
# CCD (Cyclic Coordinate Descent) numerical solver
# ---------------------------------------------------------------------------

def _ccd_solve(
    target: tuple[float, float, float],
    assembly: Assembly,
    links: list[LinkSegment],
    initial_angles: dict[str, float] | None = None,
    max_iterations: int = 100,
    tolerance_mm: float = 0.5,
    joint_limits: dict[str, tuple[float, float]] | None = None,
) -> IKResult:
    """CCD numerical IK solver with damping and random restarts.

    Iterates from the last joint to the first, rotating each joint
    to minimize the distance to the target. Uses damping factor to
    avoid oscillation and random restarts to escape local minima.
    """
    joint_limits = joint_limits or {}
    angles: dict[str, float] = dict(initial_angles) if initial_angles else {}

    # Initialize missing angles to 0
    for link in links:
        angles.setdefault(link.name, 0.0)

    solver = AssemblySolver(assembly)

    # Find end-effector name
    ee_name = links[-1].name if links else ""
    for j in assembly.joints:
        if j.parent == ee_name and j.type == "fixed":
            ee_name = j.child
            break

    best_error = float("inf")
    best_angles = dict(angles)
    damping = 0.5
    stagnation_count = 0

    for iteration in range(max_iterations):
        placements = solver.solve(joint_angles=angles)
        ee_pos = placements.get(ee_name, placements.get(links[-1].name if links else "", {}))
        ee_xyz = ee_pos.get("position", [0, 0, 0])

        error = math.sqrt(
            (target[0] - ee_xyz[0]) ** 2 +
            (target[1] - ee_xyz[1]) ** 2 +
            (target[2] - ee_xyz[2]) ** 2
        )

        if error < best_error:
            best_error = error
            best_angles = dict(angles)
            stagnation_count = 0
        else:
            stagnation_count += 1

        if error < tolerance_mm:
            return IKResult(
                joint_angles={k: round(v, 2) for k, v in best_angles.items()},
                end_effector=[round(x, 4) for x in ee_xyz],
                error_mm=round(error, 4),
                reachable=True,
                method="ccd",
                iterations=iteration + 1,
            )

        # Random restart if stuck
        if stagnation_count > 20:
            import random
            for link in links:
                lim = joint_limits.get(link.name, (-180, 180))
                angles[link.name] = random.uniform(lim[0], lim[1])
            stagnation_count = 0
            continue

        # CCD sweep: from tip to base
        for link in reversed(links):
            joint_pos_data = placements.get(link.name, {})
            joint_xyz = joint_pos_data.get("position", [0, 0, 0])

            ee_vec = (ee_xyz[0] - joint_xyz[0], ee_xyz[1] - joint_xyz[1], ee_xyz[2] - joint_xyz[2])
            target_vec = (target[0] - joint_xyz[0], target[1] - joint_xyz[1], target[2] - joint_xyz[2])

            rot_axis = _get_rot_axis(link)

            ee_proj = _project_to_plane(ee_vec, rot_axis)
            target_proj = _project_to_plane(target_vec, rot_axis)

            ee_len = _vec_length(ee_proj)
            tgt_len = _vec_length(target_proj)
            if ee_len < 1e-6 or tgt_len < 1e-6:
                continue

            cos_angle = _vec_dot(ee_proj, target_proj) / (ee_len * tgt_len)
            cos_angle = max(-1.0, min(1.0, cos_angle))
            delta = math.degrees(math.acos(cos_angle))

            cross = _vec_cross(ee_proj, target_proj)
            if _vec_dot(cross, rot_axis) < 0:
                delta = -delta

            # Apply damped delta
            delta *= damping
            current = angles.get(link.name, 0.0)
            new_angle = current + delta

            lim = joint_limits.get(link.name, (-180, 180))
            new_angle = max(lim[0], min(lim[1], new_angle))
            angles[link.name] = new_angle

            placements = solver.solve(joint_angles=angles)
            ee_pos = placements.get(ee_name, placements.get(links[-1].name if links else "", {}))
            ee_xyz = ee_pos.get("position", [0, 0, 0])

    # Use best angles found
    placements = solver.solve(joint_angles=best_angles)
    ee_pos = placements.get(ee_name, placements.get(links[-1].name if links else "", {}))
    ee_xyz = ee_pos.get("position", [0, 0, 0])
    final_error = math.sqrt(
        (target[0] - ee_xyz[0]) ** 2 +
        (target[1] - ee_xyz[1]) ** 2 +
        (target[2] - ee_xyz[2]) ** 2
    )

    return IKResult(
        joint_angles={k: round(v, 2) for k, v in best_angles.items()},
        end_effector=[round(x, 4) for x in ee_xyz],
        error_mm=round(final_error, 4),
        reachable=final_error < tolerance_mm,
        method="ccd",
        iterations=max_iterations,
    )


# ---------------------------------------------------------------------------
# Vector helpers for CCD
# ---------------------------------------------------------------------------

def _get_rot_axis(link: LinkSegment) -> tuple[float, float, float]:
    """Get rotation axis from link's axis field or anchor."""
    _EXPLICIT = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}
    if link.axis != "auto" and link.axis in _EXPLICIT:
        return _EXPLICIT[link.axis]
    from .assembly_solver import ANCHOR_DIRECTIONS
    return ANCHOR_DIRECTIONS.get(link.parent_anchor, (0, 0, 1))


def _project_to_plane(
    v: tuple[float, float, float], normal: tuple[float, float, float]
) -> tuple[float, float, float]:
    """Project vector v onto the plane defined by normal."""
    d = _vec_dot(v, normal) / (_vec_length(normal) ** 2 + 1e-10)
    return (v[0] - d * normal[0], v[1] - d * normal[1], v[2] - d * normal[2])


def _vec_dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _vec_cross(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _vec_length(v: tuple[float, float, float]) -> float:
    return math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)


# ---------------------------------------------------------------------------
# FK verification helper
# ---------------------------------------------------------------------------

def _fk_verify(
    assembly: Assembly,
    joint_angles: dict[str, float],
    end_effector_name: str,
) -> list[float]:
    """Run FK and return end-effector position."""
    solver = AssemblySolver(assembly)
    placements = solver.solve(joint_angles=joint_angles)
    ee = placements.get(end_effector_name, {})
    return ee.get("position", [0, 0, 0])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def solve_ik(
    assembly: Assembly,
    target: tuple[float, float, float],
    approach: str = "auto",
    tolerance_mm: float = 0.5,
    max_iterations: int = 200,
    initial_angles: dict[str, float] | None = None,
) -> IKResult:
    """Solve inverse kinematics for an assembly.

    Args:
        assembly: The assembly definition.
        target: Target end-effector position [x, y, z] in mm.
        approach: "analytic", "ccd", or "auto" (try analytic first, fall back to CCD).
        tolerance_mm: Acceptable error in mm.
        max_iterations: Max CCD iterations.
        initial_angles: Starting angles for CCD.

    Returns:
        IKResult with joint angles, error, and reachability.
    """
    links, base_height = _extract_chain(assembly)
    if not links:
        return IKResult(
            joint_angles={},
            end_effector=list(target),
            error_mm=float("inf"),
            reachable=False,
            method="none",
            iterations=0,
        )

    # Build joint limits from assembly
    joint_limits: dict[str, tuple[float, float]] = {}
    for j in assembly.joints:
        joint_limits[j.child] = j.range_deg

    # Find end-effector name (last child in chain, or its fixed child)
    ee_name = links[-1].name
    for j in assembly.joints:
        if j.parent == links[-1].name and j.type == "fixed":
            ee_name = j.child
            break

    result: IKResult | None = None

    if approach in ("analytic", "auto"):
        analytic = _analytic_3dof(target, links, base_height, joint_limits)
        if analytic is not None:
            # FK verify
            ee_actual = _fk_verify(assembly, analytic.joint_angles, ee_name)
            actual_error = math.sqrt(
                (target[0] - ee_actual[0]) ** 2 +
                (target[1] - ee_actual[1]) ** 2 +
                (target[2] - ee_actual[2]) ** 2
            )
            analytic.end_effector = [round(x, 4) for x in ee_actual]
            analytic.error_mm = round(actual_error, 4)
            analytic.reachable = actual_error < tolerance_mm
            result = analytic

    if (result is None or not result.reachable) and approach in ("ccd", "auto"):
        # Only use analytic result as CCD seed if it was reasonably close
        ccd_seed = None
        if result is not None and result.error_mm < 50:
            ccd_seed = result.joint_angles
        elif initial_angles:
            ccd_seed = initial_angles

        ccd_result = _ccd_solve(
            target=target,
            assembly=assembly,
            links=links,
            initial_angles=ccd_seed,
            max_iterations=max_iterations,
            tolerance_mm=tolerance_mm,
            joint_limits=joint_limits,
        )
        if result is None or ccd_result.error_mm < result.error_mm:
            result = ccd_result

    if result is None:
        return IKResult(
            joint_angles={},
            end_effector=list(target),
            error_mm=float("inf"),
            reachable=False,
            method="none",
            iterations=0,
        )

    return result


# ---------------------------------------------------------------------------
# Tool: ik_solve
# ---------------------------------------------------------------------------

class IKSolveTool(Tool):
    """Solve inverse kinematics for a target end-effector position."""

    name = "ik_solve"
    description = (
        "求解逆向运动学：给定目标末端位置，计算各关节角度。"
        "支持解析解（3-DOF）和 CCD 数值解（任意构型）。"
    )

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "assembly_name": {
                        "type": "string",
                        "description": "装配体名称（如 'robotic_arm'）或 JSON 字符串",
                    },
                    "target": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "目标末端位置 [x, y, z]（毫米）",
                    },
                    "approach": {
                        "type": "string",
                        "enum": ["auto", "analytic", "ccd"],
                        "description": "求解方法：auto（先解析后CCD）、analytic（仅解析）、ccd（仅CCD）",
                    },
                    "tolerance_mm": {
                        "type": "number",
                        "description": "可接受的误差（毫米），默认 0.5",
                    },
                    "max_iterations": {
                        "type": "integer",
                        "description": "CCD 最大迭代次数，默认 200",
                    },
                },
                "required": ["target"],
            },
        )

    def execute(
        self,
        *,
        assembly_name: str = "robotic_arm",
        target: list[float] | None = None,
        approach: str = "auto",
        tolerance_mm: float = 0.5,
        max_iterations: int = 200,
        **kwargs: Any,
    ) -> str:
        if target is None:
            return "错误：未指定目标位置 (target)"

        from .assembly_solver import _resolve_assembly
        assembly = _resolve_assembly(assembly_name, "")
        if assembly is None:
            return f"错误：未找到装配体定义 '{assembly_name}'"

        result = solve_ik(
            assembly=assembly,
            target=(target[0], target[1], target[2]),
            approach=approach,
            tolerance_mm=tolerance_mm,
            max_iterations=max_iterations,
        )

        lines = [
            f"[IK Solver] {assembly.name}",
            f"Target: ({target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f})",
            f"Method: {result.method}",
            f"Error: {result.error_mm:.4f} mm",
            f"Reachable: {'Yes' if result.reachable else 'No'}",
            f"Iterations: {result.iterations}",
            "",
            "--- Joint Angles ---",
        ]
        for name, angle in result.joint_angles.items():
            lines.append(f"  {name}: {angle:.2f} deg")

        lines.append("")
        lines.append(f"End Effector: ({result.end_effector[0]:.4f}, {result.end_effector[1]:.4f}, {result.end_effector[2]:.4f})")

        lines.append("")
        lines.append("--- JSON ---")
        lines.append(json.dumps({
            "joint_angles": result.joint_angles,
            "end_effector": result.end_effector,
            "error_mm": result.error_mm,
            "reachable": result.reachable,
            "method": result.method,
        }, ensure_ascii=False, indent=2))

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_ik_tools(registry: Any) -> None:
    """Register IK solver tools."""
    registry.register(IKSolveTool())
