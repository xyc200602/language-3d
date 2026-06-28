"""Joint motion collision detection, reachability analysis, and interference reporting.

Provides:
  - MotionCollisionChecker: sweep each revolute joint through its range,
    checking for collisions at each sample.
  - ReachabilityAnalyzer: grid-sample joint space and map end-effector positions
    to determine reachable workspace.
  - MotionCollisionTool / ReachabilityTool: Agent tools.

All heavy operations depend on python-fcl + trimesh and degrade gracefully
when they are not installed.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

from ..knowledge.mechanics import Assembly, Joint, Part
from ..models.base import ToolDefinition
from .base import Tool
from .mesh_collision import HAS_FCL, CollisionResult, MeshCollisionChecker

if HAS_FCL:
    from .assembly_solver import AssemblySolver


# ---------------------------------------------------------------------------
# Result data structures
# ---------------------------------------------------------------------------

@dataclass
class JointCollisionRange:
    """Collision analysis for a single joint swept through its range."""

    joint_name: str                       # child part name
    angle_min_deg: float = 0.0           # joint lower limit
    angle_max_deg: float = 0.0           # joint upper limit
    samples: int = 0                     # how many samples were tested
    collision_angles: list[float] = field(default_factory=list)
    collision_free_segments: list[tuple[float, float]] = field(default_factory=list)
    has_collision: bool = False


@dataclass
class MotionCollisionResult:
    """Full motion-collision sweep result for an assembly."""

    collision_free: bool = True
    joints_checked: int = 0
    joint_results: list[JointCollisionRange] = field(default_factory=list)
    summary: str = ""


@dataclass
class ReachabilityResult:
    """Reachability analysis result for an assembly."""

    reachable: bool = False
    target: tuple[float, float, float] = (0.0, 0.0, 0.0)
    end_effector_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    error_mm: float = float("inf")
    workspace_bbox: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
    ] = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    samples_total: int = 0
    summary: str = ""


# ---------------------------------------------------------------------------
# MotionCollisionChecker
# ---------------------------------------------------------------------------

class MotionCollisionChecker:
    """Sweep each revolute joint and check for collisions at each angle sample.

    For each revolute joint the checker:
      1. Discretises the joint's ``range_deg`` into *num_samples* equal steps.
      2. Calls ``AssemblySolver.solve()`` at each step with all other joints
         at their home (default) angles.
      3. Runs ``MeshCollisionChecker.check_assembly_collisions()`` on the
         resulting placements.
      4. Records which angles cause collisions and computes the maximal
         collision-free angular segments.
    """

    def __init__(self, num_samples: int = 10, min_penetration_mm: float = 2.0) -> None:
        if not HAS_FCL:
            raise RuntimeError(
                "python-fcl (and trimesh) are required. "
                "Install with: pip install python-fcl trimesh"
            )
        self._checker = MeshCollisionChecker()
        self.num_samples = num_samples
        # Collision margin (mm per side) applied to each part's mesh.
        # 2.0mm filters:
        #   - zero-depth face touches (flush-mounted parts)
        #   - cylinder discretisation artifacts (FCL uses triangle meshes)
        # while still catching real interferences deeper than 4mm.
        # Consistent with the "moderate" severity boundary (>= 0.5mm)
        # in generate_interference_report and the > 5mm "severe" threshold
        # in the static collision check.
        self.min_penetration_mm = min_penetration_mm

    # -- public API ----------------------------------------------------------

    def check_motion_collisions(
        self,
        assembly: Assembly,
        base_angles: dict[str, float] | None = None,
        skip_adjacent: bool = True,
    ) -> MotionCollisionResult:
        """Sweep every revolute joint and return a full collision report.

        Each joint is swept across its ``range_deg`` while the others stay at
        ``base_angles``.  A collision is attributed to the swept joint only if
        it is **new** relative to the baseline pose — i.e. the swept motion
        actually introduced it.  Baseline collisions (e.g. badly-offset gripper
        fingers touching at home pose) are reported once in
        ``MotionCollisionResult.baseline_collisions`` instead of being blamed
        on every joint.  Without this distinction, a single static
        interference would mark every revolute joint as colliding, masking the
        joints that are actually fine and hiding the real root cause.
        """
        solver = AssemblySolver(assembly)
        base = dict(base_angles or assembly.default_angles)

        # Baseline: collisions present at the home pose.  These are subtracted
        # from each swept pose so only *motion-induced* collisions are
        # attributed to the joint under test.
        baseline_placements = solver.solve(joint_angles=base)
        baseline_result = self._checker.check_assembly_collisions(
            assembly, baseline_placements,
            skip_adjacent=skip_adjacent,
            min_penetration_mm=self.min_penetration_mm,
        )
        baseline_keys: set[tuple[str, str]] = {
            tuple(sorted((c.part_a, c.part_b)))
            for c in baseline_result.pairs if c.is_collision
        }

        revolute_joints = [
            j for j in assembly.joints if j.type == "revolute"
        ]

        if not revolute_joints:
            return MotionCollisionResult(
                collision_free=True,
                joints_checked=0,
                summary="No revolute joints found.",
            )

        overall_free = True
        joint_results: list[JointCollisionRange] = []

        for joint in revolute_joints:
            jr = self._sweep_joint(
                assembly, solver, joint, base, skip_adjacent,
                self.min_penetration_mm,
                baseline_keys=baseline_keys,
            )
            joint_results.append(jr)
            if jr.has_collision:
                overall_free = False

        n_col = sum(1 for jr in joint_results if jr.has_collision)
        summary = (
            f"Motion collision check: {len(revolute_joints)} revolute joints, "
            f"{n_col} with motion-induced collisions "
            f"({len(baseline_keys)} baseline collisions excluded). "
            f"Result: {'collision-free' if overall_free else 'COLLISIONS DETECTED'}"
        )

        result = MotionCollisionResult(
            collision_free=overall_free,
            joints_checked=len(revolute_joints),
            joint_results=joint_results,
            summary=summary,
        )
        # Attach baseline info for downstream diagnostics (attribute set
        # dynamically to avoid changing the dataclass signature).
        result.baseline_collisions = sorted(baseline_keys)  # type: ignore[attr-defined]
        return result

    # -- internals -----------------------------------------------------------

    def _sweep_joint(
        self,
        assembly: Assembly,
        solver: "AssemblySolver",
        joint: Joint,
        base_angles: dict[str, float],
        skip_adjacent: bool,
        min_penetration_mm: float = 0.5,
        baseline_keys: set[tuple[str, str]] | None = None,
    ) -> JointCollisionRange:
        """Sweep ``joint`` across its range and record motion-induced collisions.

        ``baseline_keys`` is the set of part-pairs already colliding at the
        home pose.  When provided, only collisions **not** in this set are
        attributed to the swept joint — those are the new interferences the
        motion actually caused.  This is the difference between asking
        "does joint J cause self-collision?" (what we want) and "is there any
        collision while joint J moves?" (what the old code reported, which
        blamed every joint for unrelated baseline interferences).
        """
        lo, hi = joint.range_deg
        angles = [
            lo + (hi - lo) * i / max(self.num_samples - 1, 1)
            for i in range(self.num_samples)
        ]

        collision_angles: list[float] = []

        for angle in angles:
            test_angles = dict(base_angles)
            test_angles[joint.child] = angle

            placements = solver.solve(joint_angles=test_angles)
            result = self._checker.check_assembly_collisions(
                assembly, placements, skip_adjacent=skip_adjacent,
                min_penetration_mm=min_penetration_mm,
            )
            if not result.collision_free:
                # Subtract baseline collisions — only NEW contacts count.
                if baseline_keys is not None:
                    new_cols = [
                        c for c in result.pairs
                        if c.is_collision
                        and tuple(sorted((c.part_a, c.part_b))) not in baseline_keys
                    ]
                    if not new_cols:
                        continue  # all collisions were pre-existing
                collision_angles.append(round(angle, 2))

        # Compute collision-free segments
        free_segments = self._compute_free_segments(lo, hi, collision_angles)

        return JointCollisionRange(
            joint_name=joint.child,
            angle_min_deg=lo,
            angle_max_deg=hi,
            samples=self.num_samples,
            collision_angles=collision_angles,
            collision_free_segments=free_segments,
            has_collision=len(collision_angles) > 0,
        )

    @staticmethod
    def _compute_free_segments(
        lo: float,
        hi: float,
        collision_angles: list[float],
        tolerance: float = 0.01,
    ) -> list[tuple[float, float]]:
        """Return maximal angular intervals free of collision.

        Two adjacent sample angles that both collide cannot bound a free
        interval — the joint must move through collision to get from one to
        the other.  Previously the algorithm assumed the segment between any
        two collision samples was free, which let it return 8 "free"
        segments even when all 9 samples collided, masking total-joint-lock
        failures as partial-range usability.
        """
        if not collision_angles:
            return [(lo, hi)]

        sorted_col = sorted(collision_angles)
        segments: list[tuple[float, float]] = []
        current_start = lo

        for i, ca in enumerate(sorted_col):
            # A free segment leading up to ca exists only if current_start is
            # NOT itself a colliding sample.  When current_start == a previous
            # collision angle, the interval (prev_col, ca) is bracketed by
            # collisions on both ends and must be treated as blocked.
            current_is_collision = any(
                abs(current_start - c) <= tolerance for c in sorted_col
            )
            if not current_is_collision and ca - current_start > tolerance:
                segments.append((current_start, ca))
            current_start = ca

        # Final tail: only free if hi is not itself a collision sample.
        hi_is_collision = any(abs(hi - c) <= tolerance for c in sorted_col)
        if not hi_is_collision and hi - current_start > tolerance:
            segments.append((current_start, hi))

        return segments


# ---------------------------------------------------------------------------
# ReachabilityAnalyzer
# ---------------------------------------------------------------------------

class ReachabilityAnalyzer:
    """Analyse the reachable workspace of an assembly by grid sampling.

    For each grid point in joint-angle space the analyzer:
      1. Solves forward kinematics via ``AssemblySolver.solve()``.
      2. Records the end-effector (last child) position.
    After sampling it can answer whether a target point is reachable
    (within a tolerance) and report the workspace bounding box.
    """

    def __init__(self, samples_per_joint: int = 5) -> None:
        if not HAS_FCL:
            raise RuntimeError(
                "python-fcl (and trimesh) are required. "
                "Install with: pip install python-fcl trimesh"
            )
        self.samples_per_joint = samples_per_joint

    # -- public API ----------------------------------------------------------

    def analyze_reachability(
        self,
        assembly: Assembly,
        target: tuple[float, float, float],
        tolerance_mm: float = 5.0,
        base_angles: dict[str, float] | None = None,
    ) -> ReachabilityResult:
        """Check if *target* is reachable by the end effector."""
        solver = AssemblySolver(assembly)
        base = dict(base_angles or assembly.default_angles)

        revolute_joints = [
            j for j in assembly.joints if j.type == "revolute"
        ]

        if not revolute_joints:
            placements = solver.solve(joint_angles=base)
            ee = self._end_effector_pos(assembly, placements)
            err = self._distance(ee, target)
            return ReachabilityResult(
                reachable=err <= tolerance_mm,
                target=target,
                end_effector_position=ee,
                error_mm=round(err, 3),
                samples_total=1,
                summary=f"No revolute joints; single-position check. Error={err:.1f}mm",
            )

        # Find the last child (end effector) part name
        ee_name = revolute_joints[-1].child

        # Grid sample joint space
        joint_ranges = [
            (j.child, j.range_deg[0], j.range_deg[1])
            for j in revolute_joints
        ]

        positions: list[tuple[float, float, float]] = []
        best_err = float("inf")
        best_pos = (0.0, 0.0, 0.0)
        total_samples = 0

        # Build sample angles for each joint
        joint_samples = []
        for name, lo, hi in joint_ranges:
            n = self.samples_per_joint
            samples = [lo + (hi - lo) * i / max(n - 1, 1) for i in range(n)]
            joint_samples.append((name, samples))

        # Iterate over cartesian product of samples
        from itertools import product
        sample_lists = [s for _, s in joint_samples]
        names = [n for n, _ in joint_samples]

        for combo in product(*sample_lists):
            test_angles = dict(base)
            for name, angle in zip(names, combo):
                test_angles[name] = angle

            placements = solver.solve(joint_angles=test_angles)
            pos = placements.get(ee_name, {}).get("position", [0, 0, 0])
            p3 = (pos[0], pos[1], pos[2])
            positions.append(p3)
            total_samples += 1

            err = self._distance(p3, target)
            if err < best_err:
                best_err = err
                best_pos = p3

        # Compute workspace bounding box
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        zs = [p[2] for p in positions]
        bbox = ((min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs)))

        reachable = best_err <= tolerance_mm
        summary = (
            f"Reachability: target=({target[0]:.1f},{target[1]:.1f},{target[2]:.1f}), "
            f"best error={best_err:.1f}mm, "
            f"{'REACHABLE' if reachable else 'NOT REACHABLE'}, "
            f"samples={total_samples}"
        )

        return ReachabilityResult(
            reachable=reachable,
            target=target,
            end_effector_position=best_pos,
            error_mm=round(best_err, 3),
            workspace_bbox=bbox,
            samples_total=total_samples,
            summary=summary,
        )

    def compute_workspace_bbox(
        self,
        assembly: Assembly,
        base_angles: dict[str, float] | None = None,
    ) -> tuple[
        tuple[float, float, float],
        tuple[float, float, float],
    ]:
        """Return (min_xyz, max_xyz) of the end-effector workspace."""
        solver = AssemblySolver(assembly)
        base = dict(base_angles or assembly.default_angles)

        revolute_joints = [
            j for j in assembly.joints if j.type == "revolute"
        ]
        if not revolute_joints:
            placements = solver.solve(joint_angles=base)
            ee = self._end_effector_pos(assembly, placements)
            return ee, ee

        ee_name = revolute_joints[-1].child
        joint_ranges = [
            (j.child, j.range_deg[0], j.range_deg[1])
            for j in revolute_joints
        ]

        joint_samples = []
        for name, lo, hi in joint_ranges:
            n = self.samples_per_joint
            samples = [lo + (hi - lo) * i / max(n - 1, 1) for i in range(n)]
            joint_samples.append((name, samples))

        from itertools import product
        sample_lists = [s for _, s in joint_samples]
        names = [n for n, _ in joint_samples]

        positions: list[tuple[float, float, float]] = []
        for combo in product(*sample_lists):
            test_angles = dict(base)
            for name, angle in zip(names, combo):
                test_angles[name] = angle
            placements = solver.solve(joint_angles=test_angles)
            pos = placements.get(ee_name, {}).get("position", [0, 0, 0])
            positions.append((pos[0], pos[1], pos[2]))

        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        zs = [p[2] for p in positions]
        return ((min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs)))

    # -- internals -----------------------------------------------------------

    @staticmethod
    def _end_effector_pos(
        assembly: Assembly,
        placements: dict[str, dict],
    ) -> tuple[float, float, float]:
        """Return the position of the last child part (end effector)."""
        if assembly.joints:
            last = assembly.joints[-1].child
            pos = placements.get(last, {}).get("position", [0, 0, 0])
        elif assembly.parts:
            pos = placements.get(assembly.parts[-1].name, {}).get(
                "position", [0, 0, 0],
            )
        else:
            pos = [0, 0, 0]
        return (pos[0], pos[1], pos[2])

    @staticmethod
    def _distance(
        a: tuple[float, float, float],
        b: tuple[float, float, float],
    ) -> float:
        return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


# ---------------------------------------------------------------------------
# Agent Tool: motion_collision_check
# ---------------------------------------------------------------------------

class MotionCollisionTool(Tool):
    """Joint-motion collision sweep tool."""

    name = "motion_collision_check"
    description = (
        "关节运动碰撞检测：遍历每个旋转关节的角度范围，"
        "等间距采样并检测碰撞，返回碰撞角度和自由角度段。"
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
                        "description": "装配体名称（如 'robotic_arm'）",
                    },
                    "assembly_json": {
                        "type": "string",
                        "description": "自定义装配体 JSON 定义（可选）",
                    },
                    "joint_angles": {
                        "type": "object",
                        "description": "基准关节角度 {part_name: angle_deg}（可选）",
                    },
                    "num_samples": {
                        "type": "integer",
                        "description": "每个关节采样数（默认 10）",
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
        num_samples: int = 10,
        **kwargs: Any,
    ) -> str:
        if not HAS_FCL:
            return (
                "[Motion Collision] Error: python-fcl is not installed.\n"
                "Install with: pip install python-fcl trimesh"
            )

        from .assembly_solver import _resolve_assembly, _parse_assembly_json

        asm = None
        if assembly_json:
            try:
                asm = _parse_assembly_json(assembly_json)
            except Exception as e:
                return f"[Motion Collision] Error parsing assembly JSON: {e}"
        if asm is None:
            asm = _resolve_assembly(assembly_name, "")
        if asm is None:
            return f"[Motion Collision] Error: Unknown assembly '{assembly_name}'"

        checker = MotionCollisionChecker(num_samples=num_samples)
        result = checker.check_motion_collisions(
            asm, base_angles=joint_angles,
        )

        lines = [
            f"[Motion Collision Check]",
            f"Assembly: {asm.name}",
            f"Joints checked: {result.joints_checked}",
            f"Collision-free: {'Yes' if result.collision_free else 'NO'}",
        ]

        for jr in result.joint_results:
            status = "HAS COLLISIONS" if jr.has_collision else "clear"
            lines.append(
                f"  Joint '{jr.joint_name}': {status} "
                f"({jr.samples} samples, "
                f"{len(jr.collision_angles)} collisions)"
            )
            if jr.collision_free_segments:
                segs = ", ".join(
                    f"[{a:.1f}, {b:.1f}]" for a, b in jr.collision_free_segments
                )
                lines.append(f"    Free segments: {segs}")

        lines.append("")
        lines.append("--- JSON ---")
        lines.append(json.dumps({
            "collision_free": result.collision_free,
            "joints_checked": result.joints_checked,
            "joints": [
                {
                    "joint_name": jr.joint_name,
                    "has_collision": jr.has_collision,
                    "collision_angles": jr.collision_angles,
                    "collision_free_segments": jr.collision_free_segments,
                }
                for jr in result.joint_results
            ],
        }, ensure_ascii=False, indent=2))

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Agent Tool: reachability_check
# ---------------------------------------------------------------------------

class ReachabilityTool(Tool):
    """Workspace reachability analysis tool."""

    name = "reachability_check"
    description = (
        "可达性分析：通过网格采样关节空间，判断目标点是否可达，"
        "计算工作空间边界框。"
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
                        "description": "装配体名称（如 'robotic_arm'）",
                    },
                    "assembly_json": {
                        "type": "string",
                        "description": "自定义装配体 JSON 定义（可选）",
                    },
                    "target_x": {
                        "type": "number",
                        "description": "目标 X 坐标 (mm)",
                    },
                    "target_y": {
                        "type": "number",
                        "description": "目标 Y 坐标 (mm)",
                    },
                    "target_z": {
                        "type": "number",
                        "description": "目标 Z 坐标 (mm)",
                    },
                    "tolerance_mm": {
                        "type": "number",
                        "description": "可达性容差 (mm, 默认 5.0)",
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
        target_x: float = 100.0,
        target_y: float = 0.0,
        target_z: float = 100.0,
        tolerance_mm: float = 5.0,
        **kwargs: Any,
    ) -> str:
        if not HAS_FCL:
            return (
                "[Reachability] Error: python-fcl is not installed.\n"
                "Install with: pip install python-fcl trimesh"
            )

        from .assembly_solver import _resolve_assembly, _parse_assembly_json

        asm = None
        if assembly_json:
            try:
                asm = _parse_assembly_json(assembly_json)
            except Exception as e:
                return f"[Reachability] Error parsing assembly JSON: {e}"
        if asm is None:
            asm = _resolve_assembly(assembly_name, "")
        if asm is None:
            return f"[Reachability] Error: Unknown assembly '{assembly_name}'"

        analyzer = ReachabilityAnalyzer(samples_per_joint=5)
        target = (target_x, target_y, target_z)
        result = analyzer.analyze_reachability(
            asm, target=target, tolerance_mm=tolerance_mm,
        )

        lines = [
            f"[Reachability Check]",
            f"Assembly: {asm.name}",
            f"Target: ({target_x:.1f}, {target_y:.1f}, {target_z:.1f})",
            f"Reachable: {'Yes' if result.reachable else 'NO'}",
            f"Best position: ({result.end_effector_position[0]:.1f}, "
            f"{result.end_effector_position[1]:.1f}, "
            f"{result.end_effector_position[2]:.1f})",
            f"Error: {result.error_mm:.1f} mm",
            f"Samples: {result.samples_total}",
            f"Workspace bbox: "
            f"({result.workspace_bbox[0][0]:.1f}, {result.workspace_bbox[0][1]:.1f}, "
            f"{result.workspace_bbox[0][2]:.1f}) -> "
            f"({result.workspace_bbox[1][0]:.1f}, {result.workspace_bbox[1][1]:.1f}, "
            f"{result.workspace_bbox[1][2]:.1f})",
        ]

        lines.append("")
        lines.append("--- JSON ---")
        lines.append(json.dumps({
            "reachable": result.reachable,
            "target": list(result.target),
            "end_effector_position": list(result.end_effector_position),
            "error_mm": result.error_mm,
            "workspace_bbox": {
                "min": list(result.workspace_bbox[0]),
                "max": list(result.workspace_bbox[1]),
            },
            "samples_total": result.samples_total,
        }, ensure_ascii=False, indent=2))

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_motion_collision_tools(registry: Any) -> None:
    """Register motion collision and reachability tools."""
    if HAS_FCL:
        registry.register(MotionCollisionTool())
        registry.register(ReachabilityTool())
    # CollisionFixTool always registered (pure analysis, no FCL dependency)
    registry.register(CollisionFixTool())


# ---------------------------------------------------------------------------
# Collision fix suggestion — decision tree analysis + constraint propagation
# ---------------------------------------------------------------------------


@dataclass
class CollisionFixSuggestion:
    """A single fix suggestion for a collision detected during motion sweep."""

    joint_name: str
    fix_type: str  # "reduce_link_length" | "limit_joint_range" | "add_offset" | "increase_spacing"
    description: str  # Chinese description
    parameter: str  # parameter name to change
    current_value: float
    suggested_value: float
    confidence: float  # 0.0 - 1.0


@dataclass
class CollisionFixReport:
    """Full fix report for all collisions detected in a motion sweep."""

    total_collisions: int
    suggestions: list[CollisionFixSuggestion]
    constraint_updates: dict[str, dict[str, float]]
    summary: str


class CollisionFixSuggester:
    """Analyze MotionCollisionResult data and suggest fixes.

    Uses a decision tree based on collision angle distribution to recommend
    parameter changes. Pure analysis — does not depend on FCL.
    """

    def suggest_fixes(
        self,
        result: MotionCollisionResult,
        assembly: Assembly | None = None,
    ) -> CollisionFixReport:
        """Generate fix suggestions from a motion collision result.

        Args:
            result: The motion collision sweep result to analyze.
            assembly: Optional assembly for dimension lookups.

        Returns:
            A CollisionFixReport with suggestions and constraint updates.
        """
        suggestions: list[CollisionFixSuggestion] = []
        constraint_updates: dict[str, dict[str, float]] = {}

        colliding_joints = [jr for jr in result.joint_results if jr.has_collision]

        for jr in colliding_joints:
            total_range = jr.angle_max_deg - jr.angle_min_deg
            if total_range <= 0:
                continue

            collision_range = len(jr.collision_angles)
            collision_ratio = collision_range / max(jr.samples, 1)

            # Compute collision-free fraction
            free_span = 0.0
            for seg_start, seg_end in jr.collision_free_segments:
                free_span += seg_end - seg_start
            free_ratio = free_span / total_range

            # Find if collision is near center (zero) or extreme angles
            center_collision = any(
                abs(a) < total_range * 0.15 for a in jr.collision_angles
            )

            # Find extreme angle collision (front/back 20%)
            extreme_collision_angles = [
                a for a in jr.collision_angles
                if a > jr.angle_max_deg - total_range * 0.2
                or a < jr.angle_min_deg + total_range * 0.2
            ]
            extreme_ratio = len(extreme_collision_angles) / max(collision_range, 1)

            # Decision tree
            if collision_ratio > 0.5:
                # Large collision coverage → child link is too long
                sug = self._make_reduce_link_length(
                    jr, assembly, collision_ratio,
                )
                if sug:
                    suggestions.append(sug)
                    self._propagate_length_change(sug, assembly, constraint_updates)

            elif free_ratio < 0.1:
                # Almost no free segment → increase spacing between parts
                sug = self._make_increase_spacing(jr, assembly)
                if sug:
                    suggestions.append(sug)

            elif extreme_ratio > 0.8:
                # Collision mainly at extreme angles → limit joint range
                sug = self._make_limit_joint_range(jr)
                if sug:
                    suggestions.append(sug)
                    self._propagate_range_change(sug, constraint_updates)

            elif center_collision and collision_ratio < 0.4:
                # Collision near zero position → parts overlap at home
                sug = self._make_add_offset(jr)
                if sug:
                    suggestions.append(sug)

            else:
                # Default: suggest limiting range to free segments
                if jr.collision_free_segments:
                    sug = self._make_limit_joint_range(jr)
                    if sug:
                        suggestions.append(sug)
                        self._propagate_range_change(sug, constraint_updates)

        summary = self._build_summary(colliding_joints, suggestions)
        return CollisionFixReport(
            total_collisions=len(colliding_joints),
            suggestions=suggestions,
            constraint_updates=constraint_updates,
            summary=summary,
        )

    def _make_reduce_link_length(
        self,
        jr: JointCollisionRange,
        assembly: Assembly | None,
        collision_ratio: float,
    ) -> CollisionFixSuggestion | None:
        """Suggest reducing child link length by ~25%."""
        # Try to find the child part's length dimension
        child_length = 100.0  # default estimate
        dim_name = "length"
        if assembly:
            for part in assembly.parts:
                if part.name == jr.joint_name:
                    for key in ("length", "height", "width"):
                        if key in part.dimensions:
                            child_length = part.dimensions[key]
                            dim_name = key
                            break
                    break

        reduction = 0.25 * collision_ratio
        new_length = child_length * (1.0 - min(reduction, 0.5))

        return CollisionFixSuggestion(
            joint_name=jr.joint_name,
            fix_type="reduce_link_length",
            description=(
                f"关节 '{jr.joint_name}' 碰撞覆盖率高（{collision_ratio:.0%}），"
                f"建议缩短子连杆长度至 {new_length:.1f}mm（缩减{reduction:.0%}）"
            ),
            parameter=f"{jr.joint_name}.{dim_name}",
            current_value=child_length,
            suggested_value=round(new_length, 1),
            confidence=min(0.9, 0.5 + collision_ratio),
        )

    def _make_limit_joint_range(
        self,
        jr: JointCollisionRange,
    ) -> CollisionFixSuggestion | None:
        """Suggest limiting joint range to collision-free segments."""
        if not jr.collision_free_segments:
            return None

        # Find the largest free segment
        best_seg = max(jr.collision_free_segments, key=lambda s: s[1] - s[0])
        margin = 5.0  # 5-degree safety margin
        new_min = best_seg[0] + margin
        new_max = best_seg[1] - margin

        if new_max <= new_min:
            return None

        current_mid = (jr.angle_min_deg + jr.angle_max_deg) / 2
        new_mid = (new_min + new_max) / 2

        return CollisionFixSuggestion(
            joint_name=jr.joint_name,
            fix_type="limit_joint_range",
            description=(
                f"关节 '{jr.joint_name}' 在极端角度碰撞，"
                f"建议限制范围至 [{new_min:.1f}, {new_max:.1f}]°"
            ),
            parameter=f"{jr.joint_name}.range_deg",
            current_value=round(current_mid, 1),
            suggested_value=round(new_mid, 1),
            confidence=0.75,
        )

    def _make_add_offset(
        self,
        jr: JointCollisionRange,
    ) -> CollisionFixSuggestion | None:
        """Suggest adding an offset to avoid zero-position overlap."""
        # Find the collision-free segment closest to center
        best_seg = None
        best_dist = float("inf")
        for seg in jr.collision_free_segments:
            seg_mid = (seg[0] + seg[1]) / 2
            dist = abs(seg_mid)
            if dist < best_dist:
                best_dist = dist
                best_seg = seg

        if best_seg is None:
            return None

        offset = (best_seg[0] + best_seg[1]) / 2.0

        return CollisionFixSuggestion(
            joint_name=jr.joint_name,
            fix_type="add_offset",
            description=(
                f"关节 '{jr.joint_name}' 在零位附近碰撞，"
                f"建议将零位偏移至 {offset:.1f}°"
            ),
            parameter=f"{jr.joint_name}.home_offset_deg",
            current_value=0.0,
            suggested_value=round(offset, 1),
            confidence=0.7,
        )

    def _make_increase_spacing(
        self,
        jr: JointCollisionRange,
        assembly: Assembly | None,
    ) -> CollisionFixSuggestion | None:
        """Suggest increasing spacing between parent and child parts."""
        spacing = 15.0  # default extra spacing

        return CollisionFixSuggestion(
            joint_name=jr.joint_name,
            fix_type="increase_spacing",
            description=(
                f"关节 '{jr.joint_name}' 碰撞覆盖几乎全部范围，"
                f"建议在父子零件间增加 {spacing:.1f}mm 间距"
            ),
            parameter=f"{jr.joint_name}.spacing_mm",
            current_value=0.0,
            suggested_value=spacing,
            confidence=0.6,
        )

    def _propagate_length_change(
        self,
        suggestion: CollisionFixSuggestion,
        assembly: Assembly | None,
        updates: dict[str, dict[str, float]],
    ) -> None:
        """Propagate link length changes to downstream joint positions."""
        joint_name = suggestion.joint_name
        reduction = suggestion.current_value - suggestion.suggested_value

        # Update downstream joints: shift positions by the reduction
        if assembly:
            for joint in assembly.joints:
                if joint.parent == joint_name and joint.type != "fixed":
                    key = f"{joint.child}.offset_z"
                    current = updates.get(joint.child, {}).get("offset_z", 0.0)
                    updates.setdefault(joint.child, {})["offset_z"] = current - reduction

    def _propagate_range_change(
        self,
        suggestion: CollisionFixSuggestion,
        updates: dict[str, dict[str, float]],
    ) -> None:
        """Propagate joint range changes to default angles."""
        joint_name = suggestion.joint_name
        updates.setdefault(joint_name, {})["default_angle"] = suggestion.suggested_value

    def _build_summary(
        self,
        colliding: list[JointCollisionRange],
        suggestions: list[CollisionFixSuggestion],
    ) -> str:
        """Build a Chinese-language summary of the fix report."""
        if not colliding:
            return "无碰撞，无需修复。"

        lines = [f"检测到 {len(colliding)} 个关节存在碰撞："]
        for s in suggestions:
            lines.append(f"  - {s.description}")
            lines.append(f"    置信度: {s.confidence:.0%}, 参数: {s.parameter}")

        if suggestions:
            lines.append(f"共 {len(suggestions)} 条修复建议。")
        else:
            lines.append("无法自动生成修复建议，请手动检查。")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Auto-apply: turn structured suggestions into assembly mutations
    # ------------------------------------------------------------------

    def apply_fixes(
        self,
        assembly: Assembly,
        fixes: "CollisionFixReport",
        min_confidence: float = 0.6,
    ) -> Assembly:
        """Apply suggested fixes to an assembly (returns modified copy).

        Counterpart to :meth:`suggest_fixes`.  Mutates a deep-copied
        assembly so the caller's original is untouched.  Low-confidence
        suggestions are skipped.

        Args:
            assembly: The original assembly (not mutated).
            fixes: The fix report from :meth:`suggest_fixes`.
            min_confidence: Skip suggestions below this threshold.

        Returns:
            A new Assembly with adjusted ``part.dimensions``,
            ``joint.range_deg``, ``joint.offset``, and
            ``assembly.default_angles``.
        """
        import copy as _copy

        new_assembly = _copy.deepcopy(assembly)
        parts_by_name = {p.name: p for p in new_assembly.parts}

        for suggestion in fixes.suggestions:
            if suggestion.confidence < min_confidence:
                continue

            child_name = suggestion.joint_name

            if suggestion.fix_type == "reduce_link_length":
                # Shorten the child part's dominant dimension.
                part = parts_by_name.get(child_name)
                if part:
                    for key in ("length", "height", "width"):
                        if key in part.dimensions:
                            part.dimensions[key] = suggestion.suggested_value
                            break

            elif suggestion.fix_type == "limit_joint_range":
                # Restrict the joint's range_deg to the collision-free segment.
                # suggestion.suggested_value is the new midpoint; reconstruct
                # the range from the original span centred on the new midpoint.
                for joint in new_assembly.joints:
                    if joint.child == child_name and joint.range_deg:
                        original_span = (
                            joint.range_deg[1] - joint.range_deg[0]
                        )
                        new_mid = suggestion.suggested_value
                        half = original_span / 2.0
                        # Apply a 10% safety shrink so the new range sits
                        # comfortably inside the collision-free segment.
                        half *= 0.9
                        joint.range_deg = [
                            round(new_mid - half, 1),
                            round(new_mid + half, 1),
                        ]
                        break

            elif suggestion.fix_type == "add_offset":
                # Shift the joint's home angle so the arm starts in a
                # collision-free position rather than at zero.
                if hasattr(new_assembly, "default_angles") and \
                        new_assembly.default_angles is not None:
                    new_assembly.default_angles[child_name] = (
                        suggestion.suggested_value
                    )

            elif suggestion.fix_type == "increase_spacing":
                # Push the child part away from the parent by adding a
                # Z-axis offset to the connecting joint.
                for joint in new_assembly.joints:
                    if joint.child == child_name:
                        current = (
                            list(joint.offset) if joint.offset else [0.0, 0.0, 0.0]
                        )
                        current[2] += suggestion.suggested_value
                        joint.offset = current
                        break

        return new_assembly


# ---------------------------------------------------------------------------
# Agent Tool: collision_fix_suggest
# ---------------------------------------------------------------------------


class CollisionFixTool(Tool):
    """Analyze collision results and suggest fixes."""

    name = "collision_fix_suggest"
    description = (
        "碰撞修复建议：分析运动碰撞检测结果，根据碰撞角度分布自动建议"
        "参数修改方案（缩减连杆长度、限制关节范围、增加偏移、增大间距）。"
    )

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "assembly_json": {
                        "type": "string",
                        "description": "装配体 JSON 定义（可选，用于查找零件尺寸）",
                    },
                    "collision_result_json": {
                        "type": "string",
                        "description": "motion_collision_check 返回的碰撞结果 JSON",
                    },
                },
                "required": ["collision_result_json"],
            },
        )

    def execute(
        self,
        *,
        assembly_json: str = "",
        collision_result_json: str = "",
        **kwargs: Any,
    ) -> str:
        if not collision_result_json:
            return "[Collision Fix] 错误：未提供碰撞检测结果 JSON"

        try:
            collision_data = json.loads(collision_result_json)
        except json.JSONDecodeError as e:
            return f"[Collision Fix] 错误：无效的 JSON - {e}"

        # Build MotionCollisionResult from JSON
        result = MotionCollisionResult(
            collision_free=collision_data.get("collision_free", True),
            joints_checked=collision_data.get("joints_checked", 0),
            joint_results=[],
        )

        for j in collision_data.get("joints", []):
            jr = JointCollisionRange(
                joint_name=j["joint_name"],
                angle_min_deg=j.get("angle_min_deg", -180),
                angle_max_deg=j.get("angle_max_deg", 180),
                samples=j.get("samples", 0),
                collision_angles=j.get("collision_angles", []),
                collision_free_segments=[
                    tuple(seg) for seg in j.get("collision_free_segments", [])
                ],
                has_collision=j.get("has_collision", False),
            )
            result.joint_results.append(jr)

        # Parse optional assembly
        assembly = None
        if assembly_json:
            try:
                from .assembly_solver import _parse_assembly_json
                assembly = _parse_assembly_json(assembly_json)
            except Exception as _e:
                pass  # TODO: assembly JSON parse for collision check failed (no logger available)

        suggester = CollisionFixSuggester()
        report = suggester.suggest_fixes(result, assembly)

        lines = [
            f"[Collision Fix Suggest]",
            f"碰撞关节数: {report.total_collisions}",
            f"修复建议数: {len(report.suggestions)}",
            "",
            report.summary,
        ]

        if report.constraint_updates:
            lines.append("")
            lines.append("--- 约束更新 ---")
            for part_name, dims in report.constraint_updates.items():
                dims_str = ", ".join(f"{k}={v:.1f}" for k, v in dims.items())
                lines.append(f"  {part_name}: {dims_str}")

        lines.append("")
        lines.append("--- JSON ---")
        lines.append(json.dumps({
            "total_collisions": report.total_collisions,
            "suggestions": [
                {
                    "joint_name": s.joint_name,
                    "fix_type": s.fix_type,
                    "description": s.description,
                    "parameter": s.parameter,
                    "current_value": s.current_value,
                    "suggested_value": s.suggested_value,
                    "confidence": s.confidence,
                }
                for s in report.suggestions
            ],
            "constraint_updates": report.constraint_updates,
        }, ensure_ascii=False, indent=2))

        return "\n".join(lines)
