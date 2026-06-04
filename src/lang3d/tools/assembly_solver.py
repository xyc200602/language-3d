"""Assembly constraint solver — compute part positions from joint chains.

Given an Assembly with Joint constraints (parent_anchor / child_anchor / offset),
this solver walks the kinematic chain and computes a global Placement
(position + rotation) for every part, so that part_assemble can position
them without manually specifying coordinates.

Tools:
  assembly_solve  - Solve an Assembly definition and return part placements
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..knowledge.mechanics import Assembly, Joint, Part
from ..models.base import ToolDefinition
from .base import Tool

# ---------------------------------------------------------------------------
# Anchor direction vectors and offsets
# ---------------------------------------------------------------------------

# Each anchor face has a direction (outward normal) and a half-dimension key.
# For a part with bounding box (dx, dy, dz), the anchor point is at
# center + direction * half_extents.
ANCHOR_DIRECTIONS: dict[str, tuple[float, float, float]] = {
    "top":    (0, 0, 1),
    "bottom": (0, 0, -1),
    "left":   (-1, 0, 0),
    "right":  (1, 0, 0),
    "front":  (0, -1, 0),
    "back":   (0, 1, 0),
}

# Default dimension keys for estimating anchor offset from part dimensions.
# Maps anchor -> which dimension provides the half-extent along that axis.
ANCHOR_DIM_KEYS: dict[str, list[str]] = {
    "top":    ["height", "thickness"],
    "bottom": ["height", "thickness"],
    "left":   ["width", "diameter"],
    "right":  ["width", "diameter"],
    "front":  ["length", "depth"],
    "back":   ["length", "depth"],
}


def _half_extent(part: Part, anchor: str) -> float:
    """Estimate the half-extent of a part along the anchor direction."""
    dims = part.dimensions
    candidates = ANCHOR_DIM_KEYS.get(anchor, [])
    for key in candidates:
        if key in dims:
            return dims[key] / 2.0
    return 0.0


def _anchor_offset_for_part(part: Part, anchor: str) -> tuple[float, float, float]:
    """Compute the 3-D offset from the part center to the anchor point."""
    direction = ANCHOR_DIRECTIONS.get(anchor, (0, 0, 1))
    half = _half_extent(part, anchor)
    return (direction[0] * half, direction[1] * half, direction[2] * half)


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------

def _rotation_matrix_axis_angle(axis: tuple[float, float, float], angle_rad: float) -> list[list[float]]:
    """Rodrigues' rotation formula → 3×3 matrix."""
    x, y, z = axis
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    t = 1 - c
    return [
        [t * x * x + c,     t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c,     t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
    ]


def _mat_mul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    """3×3 matrix multiplication."""
    result = [[0.0] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                result[i][j] += a[i][k] * b[k][j]
    return result


def _mat_vec(m: list[list[float]], v: tuple[float, float, float]) -> tuple[float, float, float]:
    """Multiply 3×3 matrix by 3-vector."""
    return (
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    )


def _vec_add(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _identity_matrix() -> list[list[float]]:
    return [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


# Revolute joints rotate around the axis perpendicular to the parent anchor face.
# If the joint has an explicit axis field ("x"/"y"/"z"), use that instead.
_EXPLICIT_AXES: dict[str, tuple[float, float, float]] = {
    "x": (1, 0, 0),
    "y": (0, 1, 0),
    "z": (0, 0, 1),
}


def _revolute_axis(joint: Joint) -> tuple[float, float, float]:
    """Return the rotation axis for a revolute joint."""
    if joint.axis != "auto" and joint.axis in _EXPLICIT_AXES:
        return _EXPLICIT_AXES[joint.axis]
    # Fall back: infer from parent anchor normal
    d = ANCHOR_DIRECTIONS.get(joint.parent_anchor, (0, 0, 1))
    return d


# ---------------------------------------------------------------------------
# Core solver
# ---------------------------------------------------------------------------

class AssemblySolver:
    """Resolve part placements from assembly joint constraints."""

    def __init__(self, assembly: Assembly) -> None:
        self.assembly = assembly
        self._parts_by_name: dict[str, Part] = {p.name: p for p in assembly.parts}
        self._joints: list[Joint] = list(assembly.joints)

    def solve(
        self,
        joint_angles: dict[str, float] | None = None,
        base_position: tuple[float, float, float] = (0, 0, 0),
    ) -> dict[str, dict[str, Any]]:
        """Solve for every part's global Placement.

        Args:
            joint_angles: Optional mapping of {joint_description: angle_degrees}
                          or {child_part_name: angle_degrees}.
                          Defaults to all zeros.
            base_position: Global position for the root part.

        Returns:
            Dict mapping part_name → {"position": [x,y,z], "rotation": [ax,ay,az,angle_deg]}
        """
        if joint_angles is None:
            joint_angles = {}

        # Build adjacency: parent → list of (joint, child)
        children_of: dict[str, list[tuple[Joint, str]]] = {}
        for j in self._joints:
            children_of.setdefault(j.parent, []).append((j, j.child))

        # Find root: a part that is never a child
        child_set = {j.child for j in self._joints}
        all_names = set(self._parts_by_name.keys())
        roots = all_names - child_set
        if not roots:
            # Fallback: use the first part
            roots = {self.assembly.parts[0].name} if self.assembly.parts else set()

        placements: dict[str, dict[str, Any]] = {}

        # BFS/DFS from each root
        stack: list[tuple[str, tuple[float, float, float], list[list[float]]]] = []
        for root_name in roots:
            stack.append((root_name, base_position, _identity_matrix()))

        visited: set[str] = set()

        while stack:
            part_name, pos, rot_mat = stack.pop()
            if part_name in visited:
                continue
            visited.add(part_name)

            placements[part_name] = {
                "position": [round(pos[0], 4), round(pos[1], 4), round(pos[2], 4)],
                "rotation": _rot_mat_to_axis_angle_deg(rot_mat),
            }

            # Process children
            for joint, child_name in children_of.get(part_name, []):
                if child_name in visited:
                    continue
                child_part = self._parts_by_name.get(child_name)
                parent_part = self._parts_by_name.get(part_name)
                if not child_part or not parent_part:
                    continue

                child_pos, child_rot = self._compute_child_transform(
                    parent_part=parent_part,
                    child_part=child_part,
                    joint=joint,
                    parent_pos=pos,
                    parent_rot=rot_mat,
                    joint_angles=joint_angles,
                )
                stack.append((child_name, child_pos, child_rot))

        # Parts not in any joint get placed at origin
        for pname in self._parts_by_name:
            if pname not in placements:
                placements[pname] = {
                    "position": [base_position[0], base_position[1], base_position[2]],
                    "rotation": [0, 0, 1, 0],
                }

        return placements

    def _compute_child_transform(
        self,
        parent_part: Part,
        child_part: Part,
        joint: Joint,
        parent_pos: tuple[float, float, float],
        parent_rot: list[list[float]],
        joint_angles: dict[str, float],
    ) -> tuple[tuple[float, float, float], list[list[float]]]:
        """Compute global position and rotation for a child part."""
        # Parent anchor offset in parent's local frame
        parent_anchor_local = _anchor_offset_for_part(parent_part, joint.parent_anchor)
        # Transform to global frame using parent's rotation
        parent_anchor_global = _vec_add(
            parent_pos,
            _mat_vec(parent_rot, parent_anchor_local),
        )

        # Child anchor offset in child's local frame (points inward, negate)
        child_anchor_local = _anchor_offset_for_part(child_part, joint.child_anchor)
        # Negate because the child anchor should align with the parent anchor
        child_anchor_neg = (-child_anchor_local[0], -child_anchor_local[1], -child_anchor_local[2])

        # Get joint angle
        angle_deg = joint_angles.get(joint.child, 0.0)
        if angle_deg == 0.0:
            angle_deg = joint_angles.get(joint.description, 0.0)
        angle_rad = math.radians(angle_deg)

        # Compute child rotation
        if joint.type == "revolute":
            rot_axis_local = _revolute_axis(joint)
            rot_axis_global = _mat_vec(parent_rot, rot_axis_local)
            joint_rot = _rotation_matrix_axis_angle(rot_axis_global, angle_rad)
            child_rot = _mat_mul(joint_rot, parent_rot)
        elif joint.type == "prismatic":
            # Translation along the anchor direction, no rotation
            slide_dir = ANCHOR_DIRECTIONS.get(joint.parent_anchor, (0, 0, 1))
            slide_global = _mat_vec(parent_rot, slide_dir)
            offset = (slide_global[0] * angle_rad * 100,  # angle_rad used as distance scale
                      slide_global[1] * angle_rad * 100,
                      slide_global[2] * angle_rad * 100)
            parent_anchor_global = _vec_add(parent_anchor_global, offset)
            child_rot = [row[:] for row in parent_rot]
        else:
            # Fixed joint: inherit parent rotation
            child_rot = [row[:] for row in parent_rot]

        # Transform child anchor offset by child rotation
        child_anchor_in_parent_frame = _mat_vec(child_rot, child_anchor_neg)

        # Child center = parent anchor point + rotated child anchor offset
        child_center = _vec_add(
            parent_anchor_global,
            _mat_vec(child_rot, child_anchor_neg),
        )
        # Add explicit offset (rotated by child rotation so it moves with the joint)
        rotated_offset = _mat_vec(child_rot, joint.offset)
        child_center = _vec_add(
            child_center,
            rotated_offset,
        )

        return child_center, child_rot

    def get_joint_chain(self) -> list[dict[str, Any]]:
        """Return the ordered joint chain for this assembly."""
        chain = []
        for j in self._joints:
            chain.append({
                "type": j.type,
                "parent": j.parent,
                "child": j.child,
                "parent_anchor": j.parent_anchor,
                "child_anchor": j.child_anchor,
                "offset": list(j.offset),
                "range_deg": list(j.range_deg),
                "description": j.description,
            })
        return chain


def _rot_mat_to_axis_angle_deg(m: list[list[float]]) -> list[float]:
    """Convert a 3×3 rotation matrix to [ax, ay, az, angle_deg]."""
    # Trace
    trace = m[0][0] + m[1][1] + m[2][2]
    if trace >= 3.0 - 1e-10:
        return [0, 0, 1, 0]  # Identity
    if trace <= -1.0 + 1e-10:
        # 180 degree rotation
        angle = math.pi
        ax = math.sqrt(max(0, (m[0][0] + 1) / 2))
        ay = math.sqrt(max(0, (m[1][1] + 1) / 2))
        az = math.sqrt(max(0, (m[2][2] + 1) / 2))
        if abs(ax) < 1e-10 and abs(ay) < 1e-10:
            return [0, 0, 1, 180.0]
        return [ax, ay, az, 180.0]

    angle = math.acos(max(-1, min(1, (trace - 1) / 2)))
    s = 2 * math.sin(angle)
    if abs(s) < 1e-10:
        return [0, 0, 1, 0]

    ax = (m[2][1] - m[1][2]) / s
    ay = (m[0][2] - m[2][0]) / s
    az = (m[1][0] - m[0][1]) / s
    norm = math.sqrt(ax * ax + ay * ay + az * az)
    if norm < 1e-10:
        return [0, 0, 1, 0]
    ax /= norm
    ay /= norm
    az /= norm

    return [round(ax, 6), round(ay, 6), round(az, 6), round(math.degrees(angle), 4)]


# ---------------------------------------------------------------------------
# Tool: assembly_solve
# ---------------------------------------------------------------------------

class AssemblySolveTool(Tool):
    """Solve assembly constraints to compute part placements."""

    name = "assembly_solve"
    description = (
        "根据装配约束自动计算每个零件的全局位置和旋转。"
        "输入装配体名称和可选的关节角度，输出每个零件的 Placement。"
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
                        "description": "装配体名称（使用内置定义，如 'robotic_arm'）或 JSON 路径",
                    },
                    "assembly_json": {
                        "type": "string",
                        "description": "装配体定义的 JSON 字符串（可选，优先于 assembly_name）",
                    },
                    "joint_angles": {
                        "type": "object",
                        "description": "关节角度映射 {零件名或关节描述: 角度(度)}，默认全零",
                    },
                    "base_position": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "根零件的全局位置 [x, y, z]（默认 [0,0,0]）",
                    },
                },
                "required": [],
            },
        )

    def execute(
        self,
        *,
        assembly_name: str = "robotic_arm",
        assembly_json: str = "",
        joint_angles: dict[str, float] | None = None,
        base_position: list[float] | None = None,
        **kwargs: Any,
    ) -> str:
        assembly = _resolve_assembly(assembly_name, assembly_json)
        if assembly is None:
            return f"错误：未找到装配体定义 '{assembly_name}'"

        bp = tuple(base_position) if base_position else (0, 0, 0)
        solver = AssemblySolver(assembly)
        placements = solver.solve(joint_angles=joint_angles, base_position=bp)

        lines = [
            f"[Assembly Solver] {assembly.name}",
            f"零件数: {len(placements)}",
            f"关节数: {len(assembly.joints)}",
            "",
            "--- Part Placements ---",
        ]
        for pname, p in placements.items():
            pos = p["position"]
            rot = p["rotation"]
            lines.append(
                f"  {pname}: pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})  "
                f"rot=({rot[0]:.3f}, {rot[1]:.3f}, {rot[2]:.3f}, {rot[1]:.1f} deg)"
            )

        lines.append("")
        lines.append("--- JSON ---")
        lines.append(json.dumps(placements, ensure_ascii=False, indent=2))

        return "\n".join(lines)


def _resolve_assembly(name: str, json_str: str) -> Assembly | None:
    """Resolve an assembly by name or parse from JSON."""
    if json_str:
        try:
            return _parse_assembly_json(json_str)
        except Exception:
            pass

    # Built-in assemblies
    from ..knowledge.mechanics import ROBOTIC_ARM_ASSEMBLY

    builtins: dict[str, Assembly] = {
        "robotic_arm": ROBOTIC_ARM_ASSEMBLY,
        "3-dof_robotic_arm": ROBOTIC_ARM_ASSEMBLY,
    }
    key = name.lower().replace(" ", "_").replace("-", "_")
    if key in builtins:
        return builtins[key]

    # Try loading from file
    path = Path(name)
    if path.exists() and path.suffix in (".json", ".fcstd"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return _parse_assembly_dict(data)
        except Exception:
            pass

    return None


def _parse_assembly_json(json_str: str) -> Assembly:
    """Parse an Assembly from a JSON string."""
    data = json.loads(json_str)
    return _parse_assembly_dict(data)


def _parse_assembly_dict(data: dict[str, Any]) -> Assembly:
    """Parse an Assembly from a dict."""
    parts = [
        Part(
            name=p["name"],
            category=p.get("category", "custom"),
            description=p.get("description", ""),
            material=p.get("material", "PLA"),
            dimensions=p.get("dimensions", {}),
        )
        for p in data.get("parts", [])
    ]
    joints = [
        Joint(
            type=j.get("type", "fixed"),
            parent=j["parent"],
            child=j["child"],
            range_deg=tuple(j.get("range_deg", (-180, 180))),
            description=j.get("description", ""),
            parent_anchor=j.get("parent_anchor", "top"),
            child_anchor=j.get("child_anchor", "bottom"),
            offset=tuple(j.get("offset", (0, 0, 0))),
        )
        for j in data.get("joints", [])
    ]
    return Assembly(
        name=data.get("name", "Custom Assembly"),
        parts=parts,
        joints=joints,
        description=data.get("description", ""),
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_assembly_solver_tools(registry: Any) -> None:
    """Register assembly solver tools."""
    registry.register(AssemblySolveTool())
