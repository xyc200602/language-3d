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


@dataclass
class InterferencePair:
    """Detailed interference information for a single part pair."""

    part_a: str
    part_b: str
    penetration_depth_mm: float = 0.0
    estimated_volume_mm3: float = 0.0
    clearance_mm: float = 0.0
    severity: str = "none"  # "none" | "clearance" | "light" | "moderate" | "severe"


@dataclass
class InterferenceReport:
    """Complete interference analysis report for an assembly."""

    collision_free: bool = True
    pairs: list[InterferencePair] = field(default_factory=list)
    worst_interference: InterferencePair | None = None
    parts_checked: int = 0
    pairs_checked: int = 0
    summary: str = ""


# ---------------------------------------------------------------------------
# Mesh creation helpers
# ---------------------------------------------------------------------------

def _part_to_box_mesh(part: Part, shrink_mm: float = 0.0) -> Any:
    """Create a trimesh collision mesh from a Part's dimensions.

    Cylindrical parts (those with ``diameter``/``outer_diameter``) get a
    cylinder mesh; everything else gets a box mesh.  Using the correct
    primitive avoids false positives: a cylinder approximated as a box
    has corners that extend up to ~21% beyond the actual surface, which
    causes phantom collisions between nearby round parts (motors, servos).

    Args:
        part: The part whose dimensions define the mesh.
        shrink_mm: If > 0, shrink the mesh by this many millimetres on
            every side.  Two flush-mounted parts gain a gap of
            ``2 * shrink_mm`` and are correctly reported as collision-free.
    """
    d = part.dimensions

    # Cylindrical parts → cylinder mesh
    dia = d.get("outer_diameter", d.get("diameter"))
    if dia is not None and "length" not in d and "width" not in d:
        h = d.get("height", dia)
        if shrink_mm > 0:
            dia = max(dia - 2.0 * shrink_mm, 0.1)
            h = max(h - 2.0 * shrink_mm, 0.1)
        return trimesh.creation.cylinder(radius=dia / 2.0, height=h)

    # Rectangular parts → box mesh
    if "length" in d and "width" in d:
        extents = [d["length"], d["width"], d.get("height", 20)]
    else:
        l = d.get("length", d.get("diameter", 20))
        w = d.get("width", l)
        h = d.get("height", d.get("thickness", 20))
        extents = [l, w, h]

    if shrink_mm > 0:
        extents = [max(e - 2.0 * shrink_mm, 0.1) for e in extents]

    return trimesh.creation.box(extents=extents)


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

    @staticmethod
    def _build_adjacent_pairs(assembly: Assembly) -> set[tuple[str, str]]:
        """Set of part-name pairs that are kinematically adjacent.

        Three notions of adjacency, all of which represent expected contact
        (not collisions) on a real robot:

        1. DIRECT (parent↔child): parts joined by a joint.
        2. TRANSITIVE 2-hop (grandparent↔grandchild): e.g. motor↔wheel through
           a suspension_link (motor→suspension→wheel). The motor body and wheel
           sit at the same axle and nominally overlap; flagging it as a
           collision would make the fix-loop break the suspension chain.
        3. SIBLINGS (same parent): e.g. motor_fl and standoff_fl both fixed to
           base_plate — mounted in the same region, corner tangency is expected.

        Without (2) and (3), the fix-loop "corrects" expected contact and
        breaks correctly-placed geometry (see docs/references/
        fix_loop_regression.md).
        """
        # Build the kinematic tree.
        children_by_parent: dict[str, list[str]] = {}
        for j in assembly.joints:
            children_by_parent.setdefault(j.parent, []).append(j.child)

        adjacent: set[tuple[str, str]] = set()
        # 1. Direct edges (both directions).
        for j in assembly.joints:
            adjacent.add((j.parent, j.child))
            adjacent.add((j.child, j.parent))
        # 2. Transitive 2-hop: grandparent↔grandchild (A→B→C ⇒ A adjacent C).
        for gp, mids in children_by_parent.items():
            for mid in mids:
                for gc in children_by_parent.get(mid, []):
                    adjacent.add((gp, gc))
                    adjacent.add((gc, gp))
        # 3. Siblings (same parent).
        for _parent, siblings in children_by_parent.items():
            for i in range(len(siblings)):
                for k in range(i + 1, len(siblings)):
                    adjacent.add((siblings[i], siblings[k]))
                    adjacent.add((siblings[k], siblings[i]))
        return adjacent

    def check_assembly_collisions(
        self,
        assembly: Assembly,
        placements: dict[str, dict],
        skip_adjacent: bool = True,
        min_penetration_mm: float = 0.0,
    ) -> CollisionResult:
        """Check collisions across all part pairs in a solved assembly.

        Args:
            assembly: The Assembly definition.
            placements: ``{part_name: {"position": [x,y,z], "rotation": [...]}}``
            skip_adjacent: If True, skip pairs connected by a joint.
            min_penetration_mm: Collision margin applied to each part's
                bounding box.  Each box is shrunk by this many millimetres
                on every side before the FCL collision test, so two
                flush-mounted parts (zero-depth face touch) gain a gap of
                ``2 * min_penetration_mm`` and are correctly reported as
                collision-free.  Real interferences deeper than
                ``2 * min_penetration_mm`` are still detected.  Set to 0.0
                to keep the legacy behaviour (any FCL contact counts).
        """
        parts_by_name = {p.name: p for p in assembly.parts}
        adjacent = self._build_adjacent_pairs(assembly) if skip_adjacent else set()

        names = [p.name for p in assembly.parts]
        pairs: list[CollisionPair] = []
        collision_free = True

        # Container / decorative parts: structural shells (chassis_body,
        # base_footprint) that INTENTIONALLY enclose internal parts (motors,
        # battery, wheels), PLUS suspension struts which are decorative
        # mechanical indicators (a real suspension is a rocker/A-arm linkage,
        # not a simple strut that can be collision-checked against the wheel
        # it attaches to). Collisions involving these parts are expected and
        # must not be reported — otherwise the fix-loop breaks the enclosure
        # / linkage the design requires. A part is skipped if its name marks
        # it as an enclosing body or a suspension element.
        _containers = {
            n for n in names
            if any(kw in n.lower() for kw in (
                "chassis_body", "body_shell", "housing", "suspension_",
            ))
        }

        for i in range(len(names)):
            for k in range(i + 1, len(names)):
                a, b = names[i], names[k]
                if skip_adjacent and (a, b) in adjacent:
                    continue
                # Skip container↔internal collisions (the shell encloses parts
                # by design — e.g. wheel mounted on chassis body side).
                if a in _containers or b in _containers:
                    continue
                cp = self._check_pair(
                    a, b,
                    parts_by_name.get(a),
                    parts_by_name.get(b),
                    placements.get(a, {}),
                    placements.get(b, {}),
                    collision_margin_mm=min_penetration_mm,
                )
                pairs.append(cp)
                if cp.is_collision:
                    collision_free = False

        n_pairs = len(pairs)
        n_collisions = sum(1 for p in pairs if p.is_collision)
        summary = (
            f"Collision check: {len(names)} parts, {n_pairs} pairs checked, "
            f"{n_collisions} collisions (margin {min_penetration_mm}mm). "
            f"Result: {'collision-free' if collision_free else 'COLLISIONS DETECTED'}"
        )

        return CollisionResult(
            collision_free=collision_free,
            pairs=pairs,
            parts_checked=len(names),
            pairs_checked=n_pairs,
            summary=summary,
        )

    def generate_interference_report(
        self,
        assembly: Assembly,
        placements: dict[str, dict],
        skip_adjacent: bool = True,
        clearance_threshold_mm: float = 2.0,
    ) -> InterferenceReport:
        """Generate a detailed interference report with volume estimates and severity.

        Uses AABB overlap volume as a fast (conservative) estimate of
        interference volume.  Severity is classified as:
          - "none"       : no collision, clearance > threshold
          - "clearance"  : no collision, but clearance < threshold
          - "light"      : penetration < 0.5 mm
          - "moderate"   : 0.5 <= penetration < 2.0 mm
          - "severe"     : penetration >= 2.0 mm
        """
        import numpy as np

        parts_by_name = {p.name: p for p in assembly.parts}
        adjacent = self._build_adjacent_pairs(assembly) if skip_adjacent else set()

        names = [p.name for p in assembly.parts]
        interference_pairs: list[InterferencePair] = []
        collision_free = True
        worst: InterferencePair | None = None

        _containers = {
            n for n in names
            if any(kw in n.lower() for kw in (
                "chassis_body", "body_shell", "housing", "suspension_",
            ))
        }

        for i in range(len(names)):
            for k in range(i + 1, len(names)):
                a, b = names[i], names[k]
                if skip_adjacent and (a, b) in adjacent:
                    continue
                # Skip container↔internal (the shell encloses parts by design).
                if a in _containers or b in _containers:
                    continue

                part_a = parts_by_name.get(a)
                part_b = parts_by_name.get(b)
                if part_a is None or part_b is None:
                    continue

                place_a = placements.get(a, {})
                place_b = placements.get(b, {})

                # Compute AABB overlap volume
                vol = self._aabb_overlap_volume(part_a, part_b, place_a, place_b)

                # Run collision check for depth
                cp = self._check_pair(a, b, part_a, part_b, place_a, place_b)

                depth = cp.penetration_depth_mm if cp.is_collision else 0.0
                clearance = 0.0 if cp.is_collision else max(0.0, -depth) if depth < 0 else 0.0

                severity = self._classify_severity(
                    cp.is_collision, depth, clearance, clearance_threshold_mm,
                )

                ip = InterferencePair(
                    part_a=a,
                    part_b=b,
                    penetration_depth_mm=round(depth, 4),
                    estimated_volume_mm3=round(vol, 4),
                    clearance_mm=round(clearance, 4),
                    severity=severity,
                )
                interference_pairs.append(ip)

                if cp.is_collision:
                    collision_free = False

                if worst is None or depth > worst.penetration_depth_mm:
                    worst = ip

        n_pairs = len(interference_pairs)
        n_col = sum(1 for ip in interference_pairs if ip.severity != "none" and ip.severity != "clearance")
        summary = (
            f"Interference report: {len(names)} parts, {n_pairs} pairs, "
            f"{n_col} interferences. "
            f"Result: {'interference-free' if collision_free else 'INTERFERENCES DETECTED'}"
        )

        return InterferenceReport(
            collision_free=collision_free,
            pairs=interference_pairs,
            worst_interference=worst,
            parts_checked=len(names),
            pairs_checked=n_pairs,
            summary=summary,
        )

    @staticmethod
    def _aabb_overlap_volume(
        part_a: Part,
        part_b: Part,
        place_a: dict,
        place_b: dict,
    ) -> float:
        """Estimate overlap volume from AABB intersection."""
        def part_aabb(part: Part, place: dict) -> tuple[list[float], list[float]]:
            d = part.dimensions
            pos = place.get("position", [0, 0, 0])
            if "length" in d and "width" in d:
                l, w = d["length"], d["width"]
                h = d.get("height", d.get("thickness", 20))
            elif "outer_diameter" in d:
                l = w = d["outer_diameter"]
                h = d.get("height", l)
            elif "diameter" in d:
                l = w = d["diameter"]
                h = d.get("height", l)
            else:
                l = w = h = 20.0
            lo = [pos[0] - l / 2, pos[1] - w / 2, pos[2] - h / 2]
            hi = [pos[0] + l / 2, pos[1] + w / 2, pos[2] + h / 2]
            return lo, hi

        lo_a, hi_a = part_aabb(part_a, place_a)
        lo_b, hi_b = part_aabb(part_b, place_b)

        overlap = 1.0
        for i in range(3):
            o_lo = max(lo_a[i], lo_b[i])
            o_hi = min(hi_a[i], hi_b[i])
            if o_hi <= o_lo:
                return 0.0
            overlap *= (o_hi - o_lo)
        return overlap

    @staticmethod
    def _classify_severity(
        is_collision: bool,
        depth: float,
        clearance: float,
        threshold: float,
    ) -> str:
        if is_collision:
            if depth >= 2.0:
                return "severe"
            elif depth >= 0.5:
                return "moderate"
            else:
                return "light"
        elif clearance > 0 and clearance < threshold:
            return "clearance"
        return "none"

    # -- internals -----------------------------------------------------------

    def _check_pair(
        self,
        name_a: str,
        name_b: str,
        part_a: Part | None,
        part_b: Part | None,
        place_a: dict,
        place_b: dict,
        collision_margin_mm: float = 0.0,
    ) -> CollisionPair:
        """Check collision between two placed parts using FCL.

        Args:
            collision_margin_mm: Shrink each part's box by this many mm per
                side before testing.  See :meth:`check_assembly_collisions`.
        """
        if part_a is None or part_b is None:
            return CollisionPair(
                part_a=name_a, part_b=name_b,
                notes="Missing part definition",
            )

        mesh_a = _part_to_box_mesh(part_a, shrink_mm=collision_margin_mm)
        mesh_b = _part_to_box_mesh(part_b, shrink_mm=collision_margin_mm)

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

        F7: previously the rotation component was silently dropped, so every
        part was treated as axis-aligned — rotating a part 90° would not
        change its collision footprint, producing false negatives.  Now we
        convert the axis-angle rotation to a 3×3 rotation matrix and pass
        it to ``fcl.Transform``.
        """
        import numpy as np
        pos = placement.get("position", [0, 0, 0])
        rot = placement.get("rotation", [0, 0, 1, 0])
        if rot and len(rot) >= 4 and abs(rot[3]) > 1e-9:
            # Axis-angle → rotation matrix (Rodrigues' formula)
            ax, ay, az, ang_deg = rot[0], rot[1], rot[2], rot[3]
            norm = math.sqrt(ax * ax + ay * ay + az * az)
            if norm < 1e-9:
                rot_mat = np.eye(3)
            else:
                ax, ay, az = ax / norm, ay / norm, az / norm
                c = math.cos(math.radians(ang_deg))
                s = math.sin(math.radians(ang_deg))
                t = 1.0 - c
                rot_mat = np.array([
                    [t * ax * ax + c,    t * ax * ay - s * az, t * ax * az + s * ay],
                    [t * ax * ay + s * az, t * ay * ay + c,    t * ay * az - s * ax],
                    [t * ax * az - s * ay, t * ay * az + s * ax, t * az * az + c],
                ], dtype=np.float64)
            return fcl_mod.Transform(rot_mat, np.array(pos, dtype=np.float64))
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
