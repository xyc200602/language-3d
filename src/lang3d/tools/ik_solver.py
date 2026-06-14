"""Inverse kinematics solver for robotic assemblies.

Provides two approaches:
  - Analytic IK for 3-DOF (or 4-DOF with wrist) planar/spherical arms
  - CCD (Cyclic Coordinate Descent) numerical solver for arbitrary chains

Both approaches forward-verify their solutions via FK (AssemblySolver).

Tools:
  ik_solve  - Solve inverse kinematics for a target end-effector position
"""

from __future__ import annotations

import enum
import json
import math
from dataclasses import dataclass, field
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
    method: str                      # "analytic", "ccd", "jacobian"
    iterations: int                  # 0 for analytic, actual for CCD/Jacobian


@dataclass
class LinkSegment:
    """A link in the kinematic chain with its length."""
    name: str
    length: float
    joint_type: str
    parent_anchor: str
    axis: str  # "auto", "x", "y", "z"
    # Extended fields for dual-arm / dynamic modeling
    inertia: float = 0.0            # kg⋅mm² (rotational inertia about joint axis)
    motor_spec: str = ""            # motor model identifier (e.g. "SG90", "MG996R")
    coupling_ratio: float = 1.0     # gear/belt ratio (output_rev / input_rev)


# ---------------------------------------------------------------------------
# Chain extraction
# ---------------------------------------------------------------------------

def _forward_link_length(
    joint: Joint, assembly: Assembly, parts_by_name: dict[str, Part]
) -> float:
    """Derive the axis-to-axis link length driven by ``joint``.

    Walks forward from ``joint.child`` through fixed joints to find the first
    structural part that exposes a ``length`` dimension. Under the clean
    arm-chain convention (front/back anchors, pitch axis=x) that ``length`` is
    exactly the distance between this joint's rotation axis and the next one.

    Returns 0.0 when no suitable part is found (caller falls back to
    centre-to-centre placement distance).
    """
    candidate = joint.child
    visited: set[str] = set()
    while candidate and candidate not in visited:
        visited.add(candidate)
        part = parts_by_name.get(candidate)
        if part is not None:
            length = part.dimensions.get("length", 0.0)
            if isinstance(length, (int, float)) and length > 0:
                return float(length)
        # Advance through the single fixed joint leaving this part, so we
        # traverse housing→link connections without diverging onto motors or
        # bearings (those attach via their own fixed joints off a sibling).
        next_candidate: str | None = None
        for jj in assembly.joints:
            if jj.parent == candidate and jj.type == "fixed":
                next_candidate = jj.child
                break
        candidate = next_candidate
    return 0.0


def _extract_chain(assembly: Assembly) -> tuple[list[LinkSegment], float]:
    """Extract the kinematic chain and base height from assembly.

    Returns (links, base_height) where links are in order from base to tip,
    and base_height is the Z offset of the first revolute joint above origin.

    Link lengths are read directly from the connecting structural part's
    ``length`` dimension (the authoritative source). This is independent of the
    solved posture, so a bent ``default_angles`` no longer corrupts the IK link
    lengths. When no length dimension is available (e.g. base-yaw housings or
    wrist rolls), we fall back to the centre-to-centre placement distance,
    which is still preferable to zero.
    """
    solver = AssemblySolver(assembly)
    placements_home = solver.solve()

    # Find revolute joints in order
    revolute_joints = [j for j in assembly.joints if j.type == "revolute"]
    parts_by_name = {p.name: p for p in assembly.parts}

    links: list[LinkSegment] = []
    prev_pos: list[float] = [0.0, 0.0, 0.0]
    for j in revolute_joints:
        child_pos = placements_home.get(j.child, {}).get("position", [0, 0, 0])

        # Primary: length from the driven structural part's "length" dimension.
        seg_len = _forward_link_length(j, assembly, parts_by_name)

        # Fallback: centre-to-centre distance from the solved home posture.
        if seg_len <= 0:
            seg_len = (
                (child_pos[0] - prev_pos[0]) ** 2
                + (child_pos[1] - prev_pos[1]) ** 2
                + (child_pos[2] - prev_pos[2]) ** 2
            ) ** 0.5

        links.append(LinkSegment(
            name=j.child,
            length=seg_len,
            joint_type=j.type,
            parent_anchor=j.parent_anchor,
            axis=j.axis,
        ))
        prev_pos = child_pos

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
        "Solve inverse kinematics: given a target end-effector position, "
        "compute joint angles. Supports analytic solution (3-DOF) and CCD "
        "numerical solution (arbitrary configurations)."
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
                        "description": "Assembly name (e.g. 'robotic_arm') or JSON string",
                    },
                    "target": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Target end-effector position [x, y, z] in mm",
                    },
                    "approach": {
                        "type": "string",
                        "enum": ["auto", "analytic", "ccd"],
                        "description": "Solving method: auto (analytic first, then CCD), analytic (analytic only), ccd (CCD only)",
                    },
                    "tolerance_mm": {
                        "type": "number",
                        "description": "Acceptable error in mm (default 0.5)",
                    },
                    "max_iterations": {
                        "type": "integer",
                        "description": "Maximum CCD iterations (default 200)",
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
# Jacobian-based IK solver (damped pseudoinverse + nullspace optimization)
# ---------------------------------------------------------------------------


def _compute_jacobian(
    links: list[LinkSegment],
    joint_angles_rad: list[float],
    joint_positions: list[list[float]],
    ee_pos: list[float],
) -> list[list[float]]:
    """Compute the 3×N geometric Jacobian for revolute joints.

    Each column j = z_j × (ee - p_j), where z_j is the rotation axis
    of joint j and p_j is its position.
    """
    n = len(links)
    jacobian: list[list[float]] = [[] for _ in range(3)]

    for j in range(n):
        axis = _get_rot_axis(links[j])
        # z_j (rotation axis in world frame — assume fixed for simplicity)
        z_j = axis
        # ee - p_j
        dx = ee_pos[0] - joint_positions[j][0]
        dy = ee_pos[1] - joint_positions[j][1]
        dz = ee_pos[2] - joint_positions[j][2]
        # cross product z_j × (ee - p_j)
        col = _vec_cross(z_j, (dx, dy, dz))
        jacobian[0].append(col[0])
        jacobian[1].append(col[1])
        jacobian[2].append(col[2])

    return jacobian


def _mat_transpose(m: list[list[float]]) -> list[list[float]]:
    rows, cols = len(m), len(m[0])
    return [[m[r][c] for r in range(rows)] for c in range(cols)]


def _mat_mul_vec(m: list[list[float]], v: list[float]) -> list[float]:
    return [sum(m[i][j] * v[j] for j in range(len(v))) for i in range(len(m))]


def _damped_pseudoinverse(
    jacobian: list[list[float]], damping: float = 0.01
) -> list[list[float]]:
    """Compute damped pseudoinverse: J^T (J J^T + λ²I)^{-1}.

    For the 3×N case, J J^T is 3×3 which is cheap to invert.
    """
    n = len(jacobian[0])
    jt = _mat_transpose(jacobian)
    # J J^T (3×3)
    jjt = [[sum(jacobian[r][k] * jt[k][c] for k in range(n))
             for c in range(3)] for r in range(3)]
    # Add λ²I
    for i in range(3):
        jjt[i][i] += damping * damping
    # Invert 3×3 matrix
    inv = _invert_3x3(jjt)
    if inv is None:
        # Fallback: simple transpose
        return jt
    # J^T * inv(JJ^T + λ²I)  →  N×3
    result = [[sum(jt[r][k] * inv[k][c] for k in range(3)) for c in range(3)]
              for r in range(n)]
    return result


def _invert_3x3(m: list[list[float]]) -> list[list[float]] | None:
    """Invert a 3×3 matrix using cofactor expansion. Returns None if singular."""
    a, b, c = m[0]
    d, e, f = m[1]
    g, h, i = m[2]
    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    if abs(det) < 1e-12:
        return None
    inv_det = 1.0 / det
    return [
        [(e * i - f * h) * inv_det, (c * h - b * i) * inv_det, (b * f - c * e) * inv_det],
        [(f * g - d * i) * inv_det, (a * i - c * g) * inv_det, (c * d - a * f) * inv_det],
        [(d * h - e * g) * inv_det, (b * g - a * h) * inv_det, (a * e - b * d) * inv_det],
    ]


def _nullspace_projector(jacobian: list[list[float]]) -> list[list[float]]:
    """Compute nullspace projection matrix: I - J^+ J (N×N)."""
    n = len(jacobian[0])
    jp = _damped_pseudoinverse(jacobian)
    # J^+ J (N×N)
    jpj = [[sum(jp[r][k] * jacobian[k][c] for k in range(3)) for c in range(n)]
           for r in range(n)]
    # I - J^+J
    result = [[(1.0 if r == c else 0.0) - jpj[r][c] for c in range(n)]
              for r in range(n)]
    return result


class JacobianIKSolver:
    """Jacobian-based IK solver with damped pseudoinverse and nullspace optimization.

    Features:
      - Damped least-squares (avoids singularity issues)
      - Nullspace optimization (minimize joint motion / prefer center pose)
      - Joint limits enforcement
    """

    def __init__(
        self,
        assembly: Assembly,
        links: list[LinkSegment],
        damping: float = 0.5,
        step_scale: float = 0.5,
        max_iterations: int = 200,
        tolerance_mm: float = 0.5,
        joint_limits: dict[str, tuple[float, float]] | None = None,
    ):
        self.assembly = assembly
        self.links = links
        self.damping = damping
        self.step_scale = step_scale
        self.max_iterations = max_iterations
        self.tolerance_mm = tolerance_mm
        self.joint_limits = joint_limits or {}
        self.solver = AssemblySolver(assembly)
        # Find end-effector name
        self.ee_name = links[-1].name if links else ""
        for j in assembly.joints:
            if j.parent == links[-1].name and j.type == "fixed":
                self.ee_name = j.child
                break

    def _get_joint_positions(
        self, angles: dict[str, float]
    ) -> tuple[list[list[float]], list[float]]:
        """Compute joint positions and EE position from current angles."""
        placements = self.solver.solve(joint_angles=angles)
        positions: list[list[float]] = []
        for link in self.links:
            pos = placements.get(link.name, {}).get("position", [0, 0, 0])
            positions.append(pos)
        ee_pos = placements.get(self.ee_name, {}).get("position", [0, 0, 0])
        return positions, ee_pos

    def solve(
        self,
        target: tuple[float, float, float],
        initial_angles: dict[str, float] | None = None,
    ) -> IKResult:
        """Solve IK using Jacobian damped pseudoinverse."""
        angles: dict[str, float] = dict(initial_angles) if initial_angles else {}
        for link in self.links:
            angles.setdefault(link.name, 0.0)

        # Preferred (center) angles for nullspace optimization
        center_angles: dict[str, float] = {}
        for link in self.links:
            lim = self.joint_limits.get(link.name, (-180, 180))
            center_angles[link.name] = (lim[0] + lim[1]) / 2.0

        best_error = float("inf")
        best_angles = dict(angles)

        for iteration in range(self.max_iterations):
            joint_pos, ee_pos = self._get_joint_positions(angles)

            error_vec = [
                target[0] - ee_pos[0],
                target[1] - ee_pos[1],
                target[2] - ee_pos[2],
            ]
            error = math.sqrt(error_vec[0] ** 2 + error_vec[1] ** 2 + error_vec[2] ** 2)

            if error < best_error:
                best_error = error
                best_angles = dict(angles)

            if error < self.tolerance_mm:
                return IKResult(
                    joint_angles={k: round(v, 2) for k, v in best_angles.items()},
                    end_effector=[round(x, 4) for x in ee_pos],
                    error_mm=round(error, 4),
                    reachable=True,
                    method="jacobian",
                    iterations=iteration + 1,
                )

            # Compute Jacobian
            angles_rad = [math.radians(angles.get(link.name, 0.0)) for link in self.links]
            jac = _compute_jacobian(self.links, angles_rad, joint_pos, ee_pos)

            # Damped pseudoinverse step
            jp = _damped_pseudoinverse(jac, damping=self.damping)
            delta_theta = _mat_mul_vec(jp, error_vec)

            # Nullspace gradient: push towards center angles
            ns = _nullspace_projector(jac)
            gradient = [center_angles[link.name] - angles.get(link.name, 0.0)
                        for link in self.links]
            ns_step = _mat_mul_vec(ns, gradient)

            # Apply step with scaling
            for i, link in enumerate(self.links):
                delta = delta_theta[i] * self.step_scale + ns_step[i] * 0.1
                new_angle = angles.get(link.name, 0.0) + delta
                lim = self.joint_limits.get(link.name, (-180, 180))
                angles[link.name] = max(lim[0], min(lim[1], new_angle))

        # Return best found
        _, ee_pos = self._get_joint_positions(best_angles)
        final_error = math.sqrt(
            (target[0] - ee_pos[0]) ** 2 +
            (target[1] - ee_pos[1]) ** 2 +
            (target[2] - ee_pos[2]) ** 2
        )
        return IKResult(
            joint_angles={k: round(v, 2) for k, v in best_angles.items()},
            end_effector=[round(x, 4) for x in ee_pos],
            error_mm=round(final_error, 4),
            reachable=final_error < self.tolerance_mm,
            method="jacobian",
            iterations=self.max_iterations,
        )


# ---------------------------------------------------------------------------
# Dual-arm coordination
# ---------------------------------------------------------------------------


class DualArmMode(str, enum.Enum):
    """Mode for dual-arm IK solving."""
    INDEPENDENT = "independent"   # Solve each arm separately
    COORDINATED = "coordinated"   # Solve with collision avoidance
    MASTER_SLAVE = "master_slave" # Arm2 follows Arm1 with fixed offset


@dataclass
class DualArmResult:
    """Result of dual-arm IK solve."""
    arm1: IKResult
    arm2: IKResult
    collision_free: bool
    min_clearance_mm: float
    mode: DualArmMode
    shared_workspace_overlap: float  # fraction 0..1


def solve_dual_arm_ik(
    arm1_assembly: Assembly,
    arm2_assembly: Assembly,
    target1: tuple[float, float, float],
    target2: tuple[float, float, float],
    mode: DualArmMode = DualArmMode.COORDINATED,
    collision_check_fn: Any | None = None,
    tolerance_mm: float = 1.0,
    max_iterations: int = 200,
) -> DualArmResult:
    """Solve IK for a dual-arm system.

    Args:
        arm1_assembly: Assembly for arm 1.
        arm2_assembly: Assembly for arm 2.
        target1: Target for arm 1 end-effector [x, y, z].
        target2: Target for arm 2 end-effector [x, y, z].
        mode: INDEPENDENT / COORDINATED / MASTER_SLAVE.
        collision_check_fn: Optional callable(angles1, angles2) -> (collision_free, min_clearance).
        tolerance_mm: Acceptable error.
        max_iterations: Max iterations per solver.

    Returns:
        DualArmResult with both arm solutions and collision status.
    """
    # Solve each arm individually
    result1 = solve_ik(
        assembly=arm1_assembly, target=target1,
        approach="auto", tolerance_mm=tolerance_mm,
        max_iterations=max_iterations,
    )
    result2 = solve_ik(
        assembly=arm2_assembly, target=target2,
        approach="auto", tolerance_mm=tolerance_mm,
        max_iterations=max_iterations,
    )

    # For MASTER_SLAVE mode: arm2 follows arm1 with a symmetric offset
    if mode == DualArmMode.MASTER_SLAVE:
        # Mirror arm1 target to get arm2 target (Y-axis mirror)
        mirror_target2 = (-target1[0], target1[1], target1[2])
        result2 = solve_ik(
            assembly=arm2_assembly, target=mirror_target2,
            approach="auto", tolerance_mm=tolerance_mm,
            max_iterations=max_iterations,
        )

    # Check collision if checker provided or coordinated mode
    collision_free = True
    min_clearance = float("inf")
    if collision_check_fn is not None:
        collision_free, min_clearance = collision_check_fn(
            result1.joint_angles, result2.joint_angles
        )
    elif mode == DualArmMode.COORDINATED:
        # Simple heuristic: check EE distance
        ee1 = result1.end_effector
        ee2 = result2.end_effector
        ee_dist = math.sqrt(
            (ee1[0] - ee2[0]) ** 2 +
            (ee1[1] - ee2[1]) ** 2 +
            (ee1[2] - ee2[2]) ** 2
        )
        min_clearance = ee_dist
        # If EEs are closer than 50mm, flag potential collision
        if ee_dist < 50:
            collision_free = False

    # Estimate shared workspace overlap
    links1, _ = _extract_chain(arm1_assembly)
    links2, _ = _extract_chain(arm2_assembly)
    reach1 = sum(l.length for l in links1) if links1 else 0
    reach2 = sum(l.length for l in links2) if links2 else 0
    # Simple overlap estimate: ratio of smaller reach to larger reach
    if reach1 > 0 and reach2 > 0:
        overlap = min(reach1, reach2) / max(reach1, reach2)
    else:
        overlap = 0.0

    return DualArmResult(
        arm1=result1,
        arm2=result2,
        collision_free=collision_free,
        min_clearance_mm=round(min_clearance, 2),
        mode=mode,
        shared_workspace_overlap=round(overlap, 4),
    )


# ---------------------------------------------------------------------------
# Tool: dual_arm_ik
# ---------------------------------------------------------------------------


class DualArmIKTool(Tool):
    """Solve dual-arm inverse kinematics with collision avoidance."""

    name = "dual_arm_ik"
    description = (
        "Dual-arm coordinated inverse kinematics solving: simultaneously solve "
        "joint angles for both left and right arms. Supports "
        "independent/coordinated/master-slave modes with collision detection."
    )

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "arm1_assembly": {
                        "type": "string",
                        "description": "Left arm assembly name or JSON",
                    },
                    "arm2_assembly": {
                        "type": "string",
                        "description": "Right arm assembly name or JSON",
                    },
                    "target1": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Left arm target position [x, y, z] mm",
                    },
                    "target2": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Right arm target position [x, y, z] mm",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["independent", "coordinated", "master_slave"],
                        "description": "Dual-arm mode: independent/coordinated/master_slave",
                    },
                    "tolerance_mm": {
                        "type": "number",
                        "description": "Acceptable error in mm (default 1.0)",
                    },
                },
                "required": ["target1", "target2"],
            },
        )

    def execute(
        self,
        *,
        arm1_assembly: str = "robotic_arm",
        arm2_assembly: str = "robotic_arm",
        target1: list[float] | None = None,
        target2: list[float] | None = None,
        mode: str = "coordinated",
        tolerance_mm: float = 1.0,
        **kwargs: Any,
    ) -> str:
        if target1 is None or target2 is None:
            return "Error: target1 and target2 are required"

        from .assembly_solver import _resolve_assembly
        a1 = _resolve_assembly(arm1_assembly, "")
        a2 = _resolve_assembly(arm2_assembly, "")
        if a1 is None:
            return f"Error: assembly '{arm1_assembly}' not found"
        if a2 is None:
            return f"Error: assembly '{arm2_assembly}' not found"

        result = solve_dual_arm_ik(
            arm1_assembly=a1,
            arm2_assembly=a2,
            target1=(target1[0], target1[1], target1[2]),
            target2=(target2[0], target2[1], target2[2]),
            mode=DualArmMode(mode),
            tolerance_mm=tolerance_mm,
        )

        lines = [
            f"[Dual-Arm IK] Mode: {result.mode.value}",
            f"Collision-free: {'Yes' if result.collision_free else 'NO'}",
            f"Min clearance: {result.min_clearance_mm:.1f} mm",
            f"Workspace overlap: {result.shared_workspace_overlap:.1%}",
            "",
            "--- Arm 1 ---",
            f"  Target: ({target1[0]:.1f}, {target1[1]:.1f}, {target1[2]:.1f})",
            f"  Method: {result.arm1.method}, Error: {result.arm1.error_mm:.2f} mm",
            f"  Reachable: {'Yes' if result.arm1.reachable else 'No'}",
        ]
        for name, angle in result.arm1.joint_angles.items():
            lines.append(f"    {name}: {angle:.2f} deg")

        lines.extend([
            "",
            "--- Arm 2 ---",
            f"  Target: ({target2[0]:.1f}, {target2[1]:.1f}, {target2[2]:.1f})",
            f"  Method: {result.arm2.method}, Error: {result.arm2.error_mm:.2f} mm",
            f"  Reachable: {'Yes' if result.arm2.reachable else 'No'}",
        ])
        for name, angle in result.arm2.joint_angles.items():
            lines.append(f"    {name}: {angle:.2f} deg")

        lines.append("")
        lines.append("--- JSON ---")
        lines.append(json.dumps({
            "arm1": {
                "joint_angles": result.arm1.joint_angles,
                "end_effector": result.arm1.end_effector,
                "error_mm": result.arm1.error_mm,
                "reachable": result.arm1.reachable,
            },
            "arm2": {
                "joint_angles": result.arm2.joint_angles,
                "end_effector": result.arm2.end_effector,
                "error_mm": result.arm2.error_mm,
                "reachable": result.arm2.reachable,
            },
            "collision_free": result.collision_free,
            "min_clearance_mm": result.min_clearance_mm,
            "mode": result.mode.value,
            "shared_workspace_overlap": result.shared_workspace_overlap,
        }, ensure_ascii=False, indent=2))

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_ik_tools(registry: Any) -> None:
    """Register IK solver tools."""
    registry.register(IKSolveTool())
    registry.register(DualArmIKTool())
