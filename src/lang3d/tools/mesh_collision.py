"""Mesh-based collision detection for assemblies using trimesh + python-fcl.

Provides:
  - MeshCollisionChecker: trimesh bounding meshes + FCL collision queries
  - CollisionPair / CollisionResult: result dataclasses
  - MeshCollisionTool: Agent tool for mesh collision checking
  - register_mesh_collision_tools: registration function

This module co-exists with collision.py (capsule GJK) and does NOT
modify it.  All imports are guarded by try/except so the module degrades
gracefully if python-fcl is not installed.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

from ..knowledge.mechanics import Assembly, Part
from ..models.base import ToolDefinition
from .base import Tool

# Try importing optional dependencies
try:
    import trimesh
    import fcl as fcl_mod
    HAS_FCL = True
except ImportError:
    HAS_FCL = False


# ---------------------------------------------------------------------------
# Result data structures
# ---------------------------------------------------------------------------

@dataclass
class CollisionPair:
    """Result of a collision check between two parts."""

    part_a: str
    part_b: str
    is_collision: bool = False
    penetration_depth_mm: float = 0.0
    contact_points: int = 0
    notes: str = ""


@dataclass
class CollisionResult:
    """Complete collision detection result for an assembly."""

    collision_free: bool = True
    pairs: list[CollisionPair] = field(default_factory=list)
    parts_checked: int = 0
    pairs_checked: int = 0
    summary: str = ""


# ---------------------------------------------------------------------------
# Mesh creation helpers
# ---------------------------------------------------------------------------

def _part_to_box_mesh(part: Part) -> Any:
    """Create a trimesh box from a Part's dimensions.

    Falls back to a 20x20x20 mm cube if dimensions are incomplete.
    """
    d = part.dimensions
    if "outer_diameter" in d:
        size = d["outer_diameter"]
        h = d.get("height", size)
        mesh = trimesh.creation.box(extents=[size, size, h])
    elif "diameter" in d and "length" not in d:
        size = d["diameter"]
        h = d.get("height", size)
        mesh = trimesh.creation.box(extents=[size, size, h])
    elif "length" in d and "width" in d:
        l = d["length"]
        w = d["width"]
        h = d.get("height", 20)
        mesh = trimesh.creation.box(extents=[l, w, h])
    else:
        l = d.get("length", d.get("diameter", 20))
        w = d.get("width", l)
        h = d.get("height", d.get("thickness", 20))
        mesh = trimesh.creation.box(extents=[l, w, h])
    return mesh


# ---------------------------------------------------------------------------
# MeshCollisionChecker
# ---------------------------------------------------------------------------

class MeshCollisionChecker:
    """Assembly collision detection using trimesh bounding meshes + FCL.

    Each part is approximated by a box mesh from its dimensions.  FCL
    (Flexible Collision Library) is used for efficient pair-wise collision
    queries with penetration depth information.
    """

    def __init__(self) -> None:
        if not HAS_FCL:
            raise RuntimeError(
                "python-fcl (and trimesh) are required for MeshCollisionChecker. "
                "Install with: pip install python-fcl trimesh"
            )

    # -- public API ----------------------------------------------------------

    def create_bounding_mesh(self, part: Part) -> Any:
        """Return a trimesh.Trimesh bounding box for a part."""
        return _part_to_box_mesh(part)

    def check_assembly_collisions(
        self,
        assembly: Assembly,
        placements: dict[str, dict],
        skip_adjacent: bool = True,
    ) -> CollisionResult:
        """Check collisions across all part pairs in a solved assembly.

        Args:
            assembly: The Assembly definition.
            placements: ``{part_name: {"position": [x,y,z], "rotation": [...]}}``
            skip_adjacent: If True, skip pairs connected by a joint.
        """
        parts_by_name = {p.name: p for p in assembly.parts}
        adjacent = set()
        if skip_adjacent:
            for j in assembly.joints:
                adjacent.add((j.parent, j.child))
                adjacent.add((j.child, j.parent))

        names = [p.name for p in assembly.parts]
        pairs: list[CollisionPair] = []
        collision_free = True

        for i in range(len(names)):
            for k in range(i + 1, len(names)):
                a, b = names[i], names[k]
                if skip_adjacent and (a, b) in adjacent:
                    continue
                cp = self._check_pair(
                    a, b,
                    parts_by_name.get(a),
                    parts_by_name.get(b),
                    placements.get(a, {}),
                    placements.get(b, {}),
                )
                pairs.append(cp)
                if cp.is_collision:
                    collision_free = False

        n_pairs = len(pairs)
        n_collisions = sum(1 for p in pairs if p.is_collision)
        summary = (
            f"Collision check: {len(names)} parts, {n_pairs} pairs checked, "
            f"{n_collisions} collisions found. "
            f"Result: {'collision-free' if collision_free else 'COLLISIONS DETECTED'}"
        )

        return CollisionResult(
            collision_free=collision_free,
            pairs=pairs,
            parts_checked=len(names),
            pairs_checked=n_pairs,
            summary=summary,
        )

    # -- internals -----------------------------------------------------------

    def _check_pair(
        self,
        name_a: str,
        name_b: str,
        part_a: Part | None,
        part_b: Part | None,
        place_a: dict,
        place_b: dict,
    ) -> CollisionPair:
        """Check collision between two placed parts using FCL."""
        if part_a is None or part_b is None:
            return CollisionPair(
                part_a=name_a, part_b=name_b,
                notes="Missing part definition",
            )

        mesh_a = _part_to_box_mesh(part_a)
        mesh_b = _part_to_box_mesh(part_b)

        import numpy as np

        bvh_a = fcl_mod.BVHModel()
        bvh_a.beginModel(num_tris_=len(mesh_a.faces), num_vertices_=len(mesh_a.vertices))
        bvh_a.addSubModel(np.ascontiguousarray(mesh_a.vertices, dtype=np.float64),
                          np.ascontiguousarray(mesh_a.faces, dtype=np.int32))
        bvh_a.endModel()

        bvh_b = fcl_mod.BVHModel()
        bvh_b.beginModel(num_tris_=len(mesh_b.faces), num_vertices_=len(mesh_b.vertices))
        bvh_b.addSubModel(np.ascontiguousarray(mesh_b.vertices, dtype=np.float64),
                          np.ascontiguousarray(mesh_b.faces, dtype=np.int32))
        bvh_b.endModel()

        # Wrap BVHModels in CollisionObject with transforms
        t_a = self._make_fcl_transform(place_a)
        t_b = self._make_fcl_transform(place_b)
        co_a = fcl_mod.CollisionObject(bvh_a, t_a)
        co_b = fcl_mod.CollisionObject(bvh_b, t_b)

        request = fcl_mod.CollisionRequest()
        result = fcl_mod.CollisionResult()
        ret = fcl_mod.collide(co_a, co_b, request, result)

        is_col = ret > 0
        depth = 0.0
        if is_col:
            try:
                dreq = fcl_mod.DistanceRequest()
                dres = fcl_mod.DistanceResult()
                fcl_mod.distance(co_a, co_b, dreq, dres)
                depth = abs(dres.min_distance) if dres.min_distance < 0 else 0.0
            except Exception:
                depth = 0.0

        return CollisionPair(
            part_a=name_a,
            part_b=name_b,
            is_collision=is_col,
            penetration_depth_mm=round(depth, 3),
            contact_points=ret,
            notes="collision" if is_col else "clear",
        )

    @staticmethod
    def _make_fcl_transform(placement: dict) -> Any:
        """Build an FCL Transform from a placement dict.

        ``placement`` has keys ``position`` ([x,y,z]) and optionally
        ``rotation`` ([ax,ay,az,angle_deg]).
        """
        import numpy as np
        pos = placement.get("position", [0, 0, 0])
        return fcl_mod.Transform(np.array(pos, dtype=np.float64))



# ---------------------------------------------------------------------------
# Agent Tool: mesh_collision_check
# ---------------------------------------------------------------------------

class MeshCollisionTool(Tool):
    """Mesh-based collision detection tool for assemblies."""

    name = "mesh_collision_check"
    description = (
        "网格碰撞检测：使用 trimesh 包围盒 + FCL 库检测装配体零件间的碰撞。"
        "返回碰撞对列表和穿透深度信息。"
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
                        "description": "装配体名称（如 'robotic_arm'）或 JSON 定义",
                    },
                    "assembly_json": {
                        "type": "string",
                        "description": "自定义装配体 JSON 定义（可选）",
                    },
                    "joint_angles": {
                        "type": "object",
                        "description": "关节角度映射 {part_name: angle_deg}（可选）",
                    },
                    "skip_adjacent": {
                        "type": "boolean",
                        "description": "是否跳过相邻零件检查（默认 true）",
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
        joint_angles: dict | None = None,
        skip_adjacent: bool = True,
        **kwargs: Any,
    ) -> str:
        if not HAS_FCL:
            return (
                "[Mesh Collision] Error: python-fcl is not installed.\n"
                "Install with: pip install python-fcl trimesh"
            )

        # Resolve assembly: prefer JSON if provided
        from .assembly_solver import _resolve_assembly, _parse_assembly_json
        asm = None
        if assembly_json:
            try:
                asm = _parse_assembly_json(assembly_json)
            except Exception as e:
                return f"[Mesh Collision] Error parsing assembly JSON: {e}"
        if asm is None:
            asm = _resolve_assembly(assembly_name, "")
        if asm is None:
            return f"[Mesh Collision] Error: Unknown assembly '{assembly_name}'"

        # Solve positions using the anchor-based solver (AssemblySolver)
        from .assembly_solver import AssemblySolver
        solver = AssemblySolver(asm)
        placements = solver.solve(joint_angles=joint_angles or {})

        # Run collision check
        checker = MeshCollisionChecker()
        result = checker.check_assembly_collisions(
            asm, placements, skip_adjacent=skip_adjacent,
        )

        lines = [
            f"[Mesh Collision Check]",
            f"Assembly: {asm.name}",
            f"Parts: {result.parts_checked}",
            f"Pairs checked: {result.pairs_checked}",
            f"Collision-free: {'Yes' if result.collision_free else 'NO'}",
        ]

        for cp in result.pairs:
            status = "COLLISION" if cp.is_collision else "clear"
            lines.append(
                f"  {cp.part_a} <-> {cp.part_b}: {status}"
                + (f" (depth={cp.penetration_depth_mm:.2f}mm)" if cp.is_collision else "")
            )

        lines.append("")
        lines.append("--- JSON ---")
        lines.append(json.dumps({
            "collision_free": result.collision_free,
            "parts_checked": result.parts_checked,
            "pairs_checked": result.pairs_checked,
            "collision_pairs": [
                {
                    "part_a": cp.part_a,
                    "part_b": cp.part_b,
                    "is_collision": cp.is_collision,
                    "penetration_depth_mm": cp.penetration_depth_mm,
                    "contact_points": cp.contact_points,
                    "notes": cp.notes,
                }
                for cp in result.pairs
            ],
        }, ensure_ascii=False, indent=2))

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_mesh_collision_tools(registry: Any) -> None:
    """Register mesh collision tools.  Gracefully skips if FCL unavailable."""
    if HAS_FCL:
        registry.register(MeshCollisionTool())
