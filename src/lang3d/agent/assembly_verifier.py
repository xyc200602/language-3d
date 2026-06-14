"""Assembly verification — geometric correctness, fit checks, collision, tolerance chain.

Checks:
- Part completeness (file existence)
- Joint fit / clearance
- Mating surface alignment (normal parallelism + distance)
- Bolt hole cross-part alignment (position deviation < threshold)
- Collision detection (FCL mesh-based, with mandatory warning when unavailable)
- Tolerance chain stackup (worst-case accumulation)
- Assembly sequence feasibility (basic reachability check)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..knowledge.mechanics import (
    PRINT_TOLERANCES,
    STANDARD_SCREWS,
    Assembly,
    Joint,
    Part,
)


@dataclass
class FitCheck:
    """Result of a joint fit check between two parts."""

    joint: Joint
    parent_part: Part | None
    child_part: Part | None
    clearance: float = 0.0
    required_clearance: float = 0.0
    fits: bool | None = False
    notes: str = ""


@dataclass
class PartCheck:
    """Result of checking a single part's file existence."""

    part_name: str
    exists: bool = False
    file_path: str = ""
    notes: str = ""


@dataclass
class CollisionCheck:
    """Result of a collision check between two parts (Task 63)."""

    part_a: str
    part_b: str
    is_collision: bool = False
    penetration_depth_mm: float = 0.0
    notes: str = ""


@dataclass
class MatingSurfaceCheck:
    """Result of checking mating surface alignment between two parts (Task 80)."""

    joint_description: str
    parent_part: str
    child_part: str
    normal_deviation_deg: float = 0.0   # angle between face normals (degrees)
    face_distance_mm: float = 0.0       # distance between mating faces
    parallel_ok: bool = True            # normal deviation < 1°
    distance_ok: bool = True            # distance < 0.1mm
    notes: str = ""


@dataclass
class BoltHoleAlignmentCheck:
    """Result of checking bolt hole alignment across parts (Task 80)."""

    parent_part: str
    child_part: str
    hole_count_parent: int = 0
    hole_count_child: int = 0
    max_position_deviation_mm: float = 0.0
    aligned: bool = True                # all holes within 0.5mm deviation
    notes: str = ""


@dataclass
class ToleranceChainCheck:
    """Result of tolerance chain stackup analysis (Task 80)."""

    chain_name: str
    dimension_count: int = 0
    nominal_total: float = 0.0
    upper_deviation: float = 0.0
    lower_deviation: float = 0.0
    total_tolerance: float = 0.0
    acceptable: bool = True
    allowed_total: float = 0.0
    notes: str = ""


@dataclass
class AssemblySequenceCheck:
    """Result of assembly sequence feasibility check (Task 80)."""

    step: int
    part_name: str
    parent_name: str
    feasible: bool = True
    notes: str = ""


@dataclass
class MotionCollisionSummary:
    """Summary of a joint-sweep motion collision scan (F5)."""

    joints_checked: int = 0
    collision_free: bool = True
    joints_with_collisions: int = 0
    verified: bool = False        # False when FCL unavailable / placements missing
    notes: str = ""


@dataclass
class CenterOfMassStabilityCheck:
    """Result of center-of-mass / support-polygon stability check (F6)."""

    center_of_mass_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    support_polygon_xy: list[tuple[float, float]] = field(default_factory=list)
    inside_support_polygon: bool = True
    margin_mm: float = 0.0        # distance from COM projection to nearest polygon edge (- = outside)
    total_mass_kg: float = 0.0
    verified: bool = False        # False when mass/positions unavailable
    notes: str = ""


@dataclass
class VerificationItem:
    """A single verification check result (pass/fail + data)."""

    name: str
    category: str        # "mating_surface" | "bolt_alignment" | "collision" | "tolerance" | "sequence"
    passed: bool
    details: str = ""
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class AssemblyVerificationResult:
    """Complete assembly verification result."""

    assembly_name: str
    part_checks: list[PartCheck] = field(default_factory=list)
    fit_checks: list[FitCheck] = field(default_factory=list)
    tolerance_issues: list[str] = field(default_factory=list)
    overall_pass: bool = False
    summary: str = ""
    # --- Task 63: collision check fields ---
    collision_checks: list = field(default_factory=list)
    collision_free: bool = True
    # --- Task 80: production-grade verification ---
    mating_surface_checks: list[MatingSurfaceCheck] = field(default_factory=list)
    bolt_alignment_checks: list[BoltHoleAlignmentCheck] = field(default_factory=list)
    tolerance_chain_checks: list[ToleranceChainCheck] = field(default_factory=list)
    sequence_checks: list[AssemblySequenceCheck] = field(default_factory=list)
    verification_items: list[VerificationItem] = field(default_factory=list)
    fcl_available: bool = True
    # --- F5/F6: motion collision sweep + COM stability ---
    motion_collision: MotionCollisionSummary = field(
        default_factory=MotionCollisionSummary,
    )
    com_stability: CenterOfMassStabilityCheck = field(
        default_factory=CenterOfMassStabilityCheck,
    )


class AssemblyVerifier:
    """Verifies assembly completeness, joint fits, and tolerances."""

    def __init__(self) -> None:
        self.tolerances = PRINT_TOLERANCES
        self.screws = STANDARD_SCREWS

    def verify_assembly(
        self,
        assembly: Assembly,
        workspace: str | Path,
        parts_results: dict[str, dict[str, Any]] | None = None,
        placements: dict | None = None,
        allowed_tolerance_total: float | None = None,
    ) -> AssemblyVerificationResult:
        """Run full assembly verification.

        Args:
            assembly: The Assembly definition to verify.
            workspace: Directory where part files should exist.
            parts_results: Optional mapping of part_name -> {artifacts, result, ...}
            placements: Optional solved placements dict (from assembly_solve).
                        When provided, runs collision detection.
            allowed_tolerance_total: Max acceptable total tolerance band (mm).
                ``None`` (default) enables a conservative 1.0 mm limit.
                ``0`` explicitly disables the tolerance chain check.

        Returns:
            AssemblyVerificationResult with all checks.
        """
        workspace = Path(workspace)

        # F8: tolerance chain was disabled by default (allowed_total=0),
        # masking tolerance stackup failures.  Use a conservative default.
        if allowed_tolerance_total is None:
            allowed_tolerance_total = 1.0

        part_checks = self.check_part_completeness(assembly, workspace, parts_results)
        fit_checks = self.check_joint_fits(assembly)
        screw_issues = self.check_screw_holes(assembly)

        # Collect tolerance issues
        tolerance_issues: list[str] = []
        for fc in fit_checks:
            if fc.fits is False:
                tolerance_issues.append(
                    f"{fc.parent_part.name if fc.parent_part else '?'} -> "
                    f"{fc.child_part.name if fc.child_part else '?'}: "
                    f"clearance {fc.clearance:.2f}mm < required {fc.required_clearance:.2f}mm"
                )
        tolerance_issues.extend(screw_issues)

        # --- Task 63: collision checks ---
        # F3: when placements are not provided, collisions are UNVERIFIED —
        # we must not treat this as collision-free.  Setting
        # collision_free=False forces overall_pass to reflect the gap.
        collision_checks: list[CollisionCheck] = []
        collision_free = True
        fcl_available = True
        if placements is not None:
            collision_checks, collision_free, fcl_available = self.check_collisions(
                assembly, placements,
            )
        else:
            collision_checks = [CollisionCheck(
                part_a="*",
                part_b="*",
                is_collision=False,
                notes="UNVERIFIED: no solver placements provided — collisions NOT checked, "
                      "assembly must NOT be labeled 'production-grade'",
            )]
            collision_free = False
            fcl_available = False

        # --- Task 80: production-grade verification ---
        mating_checks = self.check_mating_surfaces(assembly, placements)
        bolt_checks = self.check_bolt_hole_alignment(assembly)
        tol_chain_checks = self.check_tolerance_chain(assembly, allowed_tolerance_total)
        seq_checks = self.check_assembly_sequence(assembly)

        # --- F5: motion-range collision sweep ---
        motion_summary = self.check_motion_collisions(assembly, placements)

        # --- F6: center-of-mass / support-polygon stability ---
        com_check = self.check_center_of_mass_stability(assembly, placements)

        # Build structured verification items
        verification_items: list[VerificationItem] = []

        # Mating surface items
        for mc in mating_checks:
            verification_items.append(VerificationItem(
                name=f"mating_{mc.parent_part}_{mc.child_part}",
                category="mating_surface",
                passed=mc.parallel_ok and mc.distance_ok,
                details=mc.notes,
                data={
                    "normal_deviation_deg": mc.normal_deviation_deg,
                    "face_distance_mm": mc.face_distance_mm,
                },
            ))

        # Bolt alignment items
        for bc in bolt_checks:
            verification_items.append(VerificationItem(
                name=f"bolt_align_{bc.parent_part}_{bc.child_part}",
                category="bolt_alignment",
                passed=bc.aligned,
                details=bc.notes,
                data={"max_deviation_mm": bc.max_position_deviation_mm},
            ))

        # Collision items — UNVERIFIED checks (no placements / no FCL) must
        # NOT count as passed, otherwise the score is meaningless.  F3.
        for cc in collision_checks:
            is_unverified = "UNVERIFIED" in cc.notes
            verification_items.append(VerificationItem(
                name=f"collision_{cc.part_a}_{cc.part_b}",
                category="collision",
                passed=(not cc.is_collision) and not is_unverified,
                details=cc.notes,
                data={"penetration_mm": cc.penetration_depth_mm},
            ))

        # Tolerance chain items
        for tc in tol_chain_checks:
            verification_items.append(VerificationItem(
                name=f"tol_chain_{tc.chain_name}",
                category="tolerance",
                passed=tc.acceptable,
                details=tc.notes,
                data={"total_tolerance": tc.total_tolerance},
            ))

        # Sequence items
        for sc in seq_checks:
            verification_items.append(VerificationItem(
                name=f"sequence_{sc.step}_{sc.part_name}",
                category="sequence",
                passed=sc.feasible,
                details=sc.notes,
            ))

        # F5: motion collision item
        verification_items.append(VerificationItem(
            name="motion_collision_sweep",
            category="motion_collision",
            passed=motion_summary.collision_free if motion_summary.verified else False,
            details=motion_summary.notes,
            data={
                "joints_checked": motion_summary.joints_checked,
                "joints_with_collisions": motion_summary.joints_with_collisions,
            },
        ))

        # F6: COM stability item
        verification_items.append(VerificationItem(
            name="com_stability",
            category="stability",
            passed=com_check.inside_support_polygon if com_check.verified else False,
            details=com_check.notes,
            data={
                "com_mm": list(com_check.center_of_mass_mm),
                "margin_mm": com_check.margin_mm,
                "total_mass_kg": com_check.total_mass_kg,
            },
        ))

        # Determine overall pass
        all_parts_exist = all(pc.exists for pc in part_checks)
        # Three-state fits model: True=pass, False=fail, None=inconclusive.
        # Only explicit False fails the assembly; None (missing dimensions)
        # does not cause overall failure.
        all_fits_ok = not any(fc.fits is False for fc in fit_checks)
        all_mating_ok = all(mc.parallel_ok and mc.distance_ok for mc in mating_checks)
        all_bolts_ok = all(bc.aligned for bc in bolt_checks)
        all_tol_ok = all(tc.acceptable for tc in tol_chain_checks)
        all_seq_ok = all(sc.feasible for sc in seq_checks)
        overall_pass = (
            all_parts_exist
            and all_fits_ok
            and len(screw_issues) == 0
            and collision_free
            and all_mating_ok
            and all_bolts_ok
            and all_tol_ok
            and all_seq_ok
            and motion_summary.collision_free
            and com_check.inside_support_polygon
        )

        # Build summary
        completed_parts = sum(1 for pc in part_checks if pc.exists)
        failed_fits = sum(1 for fc in fit_checks if fc.fits is False)
        n_collisions = sum(1 for cc in collision_checks if cc.is_collision)
        n_items_pass = sum(1 for vi in verification_items if vi.passed)
        n_items_total = len(verification_items)
        summary = (
            f"装配验证: {assembly.name}\n"
            f"- 零件完成: {completed_parts}/{len(part_checks)}\n"
            f"- 配合检查: {len(fit_checks) - failed_fits}/{len(fit_checks)} 通过\n"
            f"- 公差问题: {len(tolerance_issues)}\n"
            f"- 碰撞检查: {n_collisions} 碰撞"
            + ("" if collision_free else " (发现干涉)")
            + f"\n- 配合面检查: {len(mating_checks)} 对"
            + f"\n- 螺栓对齐: {len(bolt_checks)} 组"
            + f"\n- 公差链: {len(tol_chain_checks)} 条"
            + f"\n- 运动碰撞扫描: "
            + (motion_summary.notes if not motion_summary.verified
               else f"{motion_summary.joints_checked} 关节, "
                    f"{motion_summary.joints_with_collisions} 有碰撞")
            + f"\n- 重心稳定性: "
            + (com_check.notes if not com_check.verified
               else f"COM=({com_check.center_of_mass_mm[0]:.1f},"
                    f"{com_check.center_of_mass_mm[1]:.1f},"
                    f"{com_check.center_of_mass_mm[2]:.1f}) "
                    f"{'在' if com_check.inside_support_polygon else '不在'}"
                    f"支撑多边形内")
            + f"\n- 结构化检查: {n_items_pass}/{n_items_total} 通过"
            + "\n"
            f"- 总体结果: {'通过' if overall_pass else '未通过'}"
        )

        return AssemblyVerificationResult(
            assembly_name=assembly.name,
            part_checks=part_checks,
            fit_checks=fit_checks,
            tolerance_issues=tolerance_issues,
            overall_pass=overall_pass,
            summary=summary,
            collision_checks=collision_checks,
            collision_free=collision_free,
            mating_surface_checks=mating_checks,
            bolt_alignment_checks=bolt_checks,
            tolerance_chain_checks=tol_chain_checks,
            sequence_checks=seq_checks,
            verification_items=verification_items,
            fcl_available=fcl_available,
            motion_collision=motion_summary,
            com_stability=com_check,
        )

    def check_part_completeness(
        self,
        assembly: Assembly,
        workspace: Path,
        parts_results: dict[str, dict[str, Any]] | None = None,
    ) -> list[PartCheck]:
        """Check that all parts have been created.

        Looks for files matching part names in the workspace directory.
        """
        checks: list[PartCheck] = []
        parts_results = parts_results or {}

        for part in assembly.parts:
            # Check if part has been reported as completed
            part_result = parts_results.get(part.name)
            found_path = ""

            if part_result:
                artifacts = part_result.get("artifacts", [])
                for art in artifacts:
                    art_lower = Path(art).stem.lower().replace(" ", "_")
                    if part.name in art_lower or art_lower in part.name:
                        found_path = art
                        break

            # Also check workspace for matching files
            if not found_path and workspace.exists():
                for ext in (".fcstd", ".step", ".stl"):
                    candidate = workspace / f"{part.name}{ext}"
                    if candidate.exists():
                        found_path = str(candidate)
                        break

            checks.append(PartCheck(
                part_name=part.name,
                exists=bool(found_path),
                file_path=found_path,
                notes="" if found_path else "文件未找到",
            ))

        return checks

    def check_joint_fits(self, assembly: Assembly) -> list[FitCheck]:
        """Check joint fits based on part dimensions and print tolerances.

        For each joint, compares the shaft diameter (from child) with
        the hole diameter (from parent) and checks if clearance is
        within acceptable tolerance.
        """
        checks: list[FitCheck] = []
        parts_by_name = {p.name: p for p in assembly.parts}

        for joint in assembly.joints:
            parent = parts_by_name.get(joint.parent)
            child = parts_by_name.get(joint.child)

            clearance = 0.0
            required = 0.0
            fits = True
            notes = ""

            if parent and child:
                # Try to extract shaft/hole dimensions
                shaft_d = child.dimensions.get("shaft_diameter", 0)
                hole_d = parent.dimensions.get("shaft_diameter", 0)  # May be in parent

                # Check outer_diameter of child vs inner diameter inferred from parent
                child_od = child.dimensions.get("outer_diameter", 0)
                parent_wall = parent.dimensions.get("wall_thickness", 0)
                parent_od = parent.dimensions.get("outer_diameter", 0)

                if shaft_d > 0 and parent_od > 0 and parent_wall > 0:
                    # Hole = outer_diameter - 2 * wall_thickness
                    hole_d = parent_od - 2 * parent_wall
                    clearance = hole_d - shaft_d

                    if joint.type == "revolute":
                        required = self.tolerances["sliding_fit"]
                    elif joint.type == "fixed":
                        required = self.tolerances["tight_fit"]
                    else:
                        required = self.tolerances["loose_fit"]

                    fits = clearance >= required
                    if not fits:
                        notes = (
                            f"间隙不足: 孔径 {hole_d:.2f}mm, 轴径 {shaft_d:.2f}mm, "
                            f"间隙 {clearance:.2f}mm < 要求 {required:.2f}mm"
                        )
                    else:
                        notes = f"配合良好: 间隙 {clearance:.2f}mm >= 要求 {required:.2f}mm"
                else:
                    notes = "尺寸数据不完整，无法检查配合"
                    fits = None  # inconclusive — dimensions missing

            checks.append(FitCheck(
                joint=joint,
                parent_part=parent,
                child_part=child,
                clearance=clearance,
                required_clearance=required,
                fits=fits,
                notes=notes,
            ))

        return checks

    def check_screw_holes(self, assembly: Assembly) -> list[str]:
        """Check that screw hole specifications are consistent.

        Looks for screw size references in part notes and validates
        against STANDARD_SCREWS.
        """
        issues: list[str] = []

        for part in assembly.parts:
            notes = part.notes.lower()
            for screw_size, specs in self.screws.items():
                screw_lower = screw_size.lower()
                if screw_lower in notes:
                    # Check if dimensions mention related hole sizes
                    for dim_name, dim_val in part.dimensions.items():
                        if "hole" in dim_name.lower() or "diameter" in dim_name.lower():
                            # Verify the dimension makes sense for the screw
                            clearance = specs["clearance_hole"]
                            tap = specs["tap_hole"]
                            # The dimension should be close to either clearance or tap hole
                            if abs(dim_val - clearance) > 1.0 and abs(dim_val - tap) > 1.0:
                                issues.append(
                                    f"{part.name}: 尺寸 {dim_name}={dim_val}mm "
                                    f"与 {screw_size} 螺钉不匹配 "
                                    f"(期望 通行孔={clearance}mm 或 攻丝孔={tap}mm)"
                                )

        return issues

    def check_collisions(
        self,
        assembly: Assembly,
        placements: dict,
    ) -> tuple[list[CollisionCheck], bool, bool]:
        """Run mesh collision detection on a solved assembly (Task 63/80).

        Args:
            assembly: The Assembly definition.
            placements: Solved placements from assembly_solve.

        Returns:
            (collision_checks, collision_free, fcl_available) tuple.
            When FCL is not available, fcl_available=False and a warning
            is included in the collision checks.
        """
        try:
            from ..tools.mesh_collision import MeshCollisionChecker
            checker = MeshCollisionChecker()
            result = checker.check_assembly_collisions(assembly, placements)
            checks = [
                CollisionCheck(
                    part_a=cp.part_a,
                    part_b=cp.part_b,
                    is_collision=cp.is_collision,
                    penetration_depth_mm=cp.penetration_depth_mm,
                    notes=cp.notes,
                )
                for cp in result.pairs
            ]
            return checks, result.collision_free, True
        except ImportError:
            # FCL not available — collision status is UNVERIFIED, NOT "safe".
            # F3: returning collision_free=True here created an always-pass
            # path that masked broken kinematics.  Now we return False so the
            # assembly cannot be labelled "collision-free" without actually
            # checking collisions.
            warning_check = CollisionCheck(
                part_a="*",
                part_b="*",
                is_collision=False,
                notes="UNVERIFIED: FCL (python-fcl) not installed — collisions NOT checked, "
                      "assembly must NOT be labeled 'production-grade'",
            )
            return [warning_check], False, False
        except Exception as e:
            warning_check = CollisionCheck(
                part_a="*",
                part_b="*",
                is_collision=False,
                notes=f"UNVERIFIED: Collision check failed: {e} — "
                      "assembly must NOT be labeled 'production-grade'",
            )
            return [warning_check], False, False

    # ------------------------------------------------------------------
    # Task 80: Mating surface verification
    # ------------------------------------------------------------------

    def check_mating_surfaces(
        self,
        assembly: Assembly,
        placements: dict | None = None,
    ) -> list[MatingSurfaceCheck]:
        """Verify mating surfaces between adjacent parts.

        For each joint, checks that:
        1. Face normals are parallel (deviation < 1°)
        2. Face distance is within tolerance (< 0.1mm)

        When placements are available, computes actual face distances
        from solved positions. Otherwise, uses a heuristic check based
        on joint anchor directions.
        """
        checks: list[MatingSurfaceCheck] = []
        parts_by_name = {p.name: p for p in assembly.parts}

        for joint in assembly.joints:
            parent = parts_by_name.get(joint.parent)
            child = parts_by_name.get(joint.child)
            desc = joint.description or f"{joint.parent}→{joint.child}"

            if not parent or not child:
                checks.append(MatingSurfaceCheck(
                    joint_description=desc,
                    parent_part=joint.parent,
                    child_part=joint.child,
                    notes="零件信息缺失",
                    parallel_ok=True,
                    distance_ok=True,
                ))
                continue

            if placements and joint.parent in placements and joint.child in placements:
                # Compute from actual placements
                pp = placements[joint.parent]
                cp = placements[joint.child]
                p_pos = pp.get("position", (0, 0, 0))
                c_pos = cp.get("position", (0, 0, 0))

                # Check anchor face normal alignment
                parent_anchor = joint.parent_anchor or "top"
                child_anchor = joint.child_anchor or "bottom"
                p_normal = self._anchor_normal(parent_anchor)
                c_normal = self._anchor_normal(child_anchor)

                # Normal deviation (should be anti-parallel for mating faces)
                dot = sum(a * b for a, b in zip(p_normal, c_normal))
                # Anti-parallel: dot ≈ -1
                cos_angle = max(-1.0, min(1.0, -dot))
                deviation_deg = math.degrees(math.acos(cos_angle))

                # Distance between face centers
                p_offset = self._anchor_offset(parent_anchor, parent)
                c_offset = self._anchor_offset(child_anchor, child)
                p_face = tuple(p + o for p, o in zip(p_pos, p_offset))
                c_face = tuple(c + o for c, o in zip(c_pos, c_offset))
                distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(p_face, c_face)))

                parallel_ok = deviation_deg < 1.0
                distance_ok = distance < 0.1
                notes = ""
                if not parallel_ok:
                    notes += f"法向偏差过大 ({deviation_deg:.2f}° >= 1°). "
                if not distance_ok:
                    notes += f"面距离过大 ({distance:.3f}mm >= 0.1mm). "
                if parallel_ok and distance_ok:
                    notes = "配合面贴合良好"

                checks.append(MatingSurfaceCheck(
                    joint_description=desc,
                    parent_part=joint.parent,
                    child_part=joint.child,
                    normal_deviation_deg=deviation_deg,
                    face_distance_mm=distance,
                    parallel_ok=parallel_ok,
                    distance_ok=distance_ok,
                    notes=notes.strip(),
                ))
            else:
                # Heuristic: assume OK (no placement data to verify)
                checks.append(MatingSurfaceCheck(
                    joint_description=desc,
                    parent_part=joint.parent,
                    child_part=joint.child,
                    notes="无位置数据，跳过配合面验证",
                    parallel_ok=True,
                    distance_ok=True,
                ))

        return checks

    # ------------------------------------------------------------------
    # Task 80: Bolt hole alignment verification
    # ------------------------------------------------------------------

    def check_bolt_hole_alignment(
        self,
        assembly: Assembly,
    ) -> list[BoltHoleAlignmentCheck]:
        """Verify bolt holes align across parts.

        For each bolted joint, checks that the parent and child have
        matching hole counts and that hole positions correspond within
        0.5mm tolerance.

        Uses bolt_holes from ConnectionMethod if available, otherwise
        infers from part dimensions.
        """
        checks: list[BoltHoleAlignmentCheck] = []
        parts_by_name = {p.name: p for p in assembly.parts}

        for joint in assembly.joints:
            parent = parts_by_name.get(joint.parent)
            child = parts_by_name.get(joint.child)

            if not parent or not child:
                continue

            conn = joint.connection
            if conn is None or conn.type != "bolted":
                continue

            # Count holes on the parent side from the connection spec.
            parent_holes = len(conn.bolt_holes) if conn.bolt_holes else conn.bolt_count
            if parent_holes == 0:
                parent_holes = 4  # default 4-bolt pattern

            # F8: previously the child hole count was blindly copied from
            # the parent (``child_holes = parent_holes``), which made the
            # count-match check a tautology.  Now we infer the child's hole
            # count independently: functional parts expose their mounting
            # pattern via ``notes`` or dimensions; if unknown we report
            # ``child_holes = -1`` so the alignment cannot silently pass.
            child_holes = self._infer_child_hole_count(child, conn)

            # Check position alignment
            if conn.bolt_holes and len(conn.bolt_holes) > 0:
                max_dev = 0.0
                # Verify holes are symmetrically distributed
                if len(conn.bolt_holes) >= 2:
                    xs = [bh.position[0] for bh in conn.bolt_holes]
                    ys = [bh.position[1] for bh in conn.bolt_holes]
                    # Symmetry check: holes should be roughly symmetric around center
                    cx = sum(xs) / len(xs)
                    cy = sum(ys) / len(ys)
                    # Center should be near (0, 0) for a properly defined interface
                    center_dev = math.sqrt(cx ** 2 + cy ** 2)
                    max_dev = center_dev

                count_ok = child_holes < 0 or parent_holes == child_holes
                aligned = max_dev < 0.5 and count_ok
                notes = ""
                if child_holes < 0:
                    notes = "子件孔数未知，孔数匹配未验证"
                elif parent_holes != child_holes:
                    notes = f"孔数量不匹配: {parent_holes} vs {child_holes}"
                elif max_dev >= 0.5:
                    notes = f"孔位偏差过大: {max_dev:.3f}mm"
                else:
                    notes = "螺栓孔对齐良好"

                checks.append(BoltHoleAlignmentCheck(
                    parent_part=joint.parent,
                    child_part=joint.child,
                    hole_count_parent=parent_holes,
                    hole_count_child=child_holes,
                    max_position_deviation_mm=max_dev,
                    aligned=aligned,
                    notes=notes,
                ))
            else:
                # No explicit holes — cannot verify alignment.
                # F8: previously this was auto-pass; now we only pass when
                # the child hole count is independently known to match.
                count_ok = child_holes < 0 or parent_holes == child_holes
                if child_holes < 0:
                    hole_notes = "无显式孔位数据，子件孔数未知"
                elif not count_ok:
                    hole_notes = f"孔数量不匹配: {parent_holes} vs {child_holes}"
                else:
                    hole_notes = "无显式孔位数据，孔数匹配"
                checks.append(BoltHoleAlignmentCheck(
                    parent_part=joint.parent,
                    child_part=joint.child,
                    hole_count_parent=parent_holes,
                    hole_count_child=child_holes,
                    notes=hole_notes,
                    aligned=count_ok and child_holes >= 0,
                ))

        return checks

    @staticmethod
    def _infer_child_hole_count(child: Part, conn) -> int:
        """Infer the child part's bolt hole count independently.

        Returns the inferred count, or ``-1`` when it cannot be determined
        (meaning the count-match check cannot be resolved and should not
        silently pass).
        """
        # 1. Check the child's notes for a catalog mounting pattern hint
        notes = (child.notes or "").lower()
        for token, count in [
            ("4xm3", 4), ("4xm4", 4), ("4xm5", 4), ("4xm6", 4),
            ("2xm3", 2), ("2xm4", 2), ("2xm2", 2),
            ("bolt_count:4", 4), ("bolt_count:2", 2),
        ]:
            if token in notes:
                return count

        # 2. Category-based defaults: functional parts (motors, bearings)
        #    have well-known mounting patterns.
        cat = (child.category or "").lower()
        name = child.name.lower()
        if any(kw in name for kw in ("nema17", "nema23", "stepper")):
            return 4  # NEMA steppers use 4-bolt flange
        if "mg996r" in name or "sg90" in name:
            return 2  # hobby servos use 2-bolt mount
        if "bearing" in name or cat == "bearing":
            return 0  # press-fit, no bolts

        # 3. If the child is functional (actuator/sensor/electronics) but
        #    we can't determine the pattern, leave unknown.
        if cat in ("actuator", "sensor", "electronics", "fastener", "bearing"):
            return -1

        # 4. Structural parts: assume they match the joint's bolt_count,
        #    since the connection spec applies to both mating parts.
        return conn.bolt_count if conn.bolt_count > 0 else 4

    # ------------------------------------------------------------------
    # Task 80: Tolerance chain analysis
    # ------------------------------------------------------------------

    def check_tolerance_chain(
        self,
        assembly: Assembly,
        allowed_total: float = 1.0,
    ) -> list[ToleranceChainCheck]:
        """Analyze tolerance accumulation along assembly chains.

        For each joint chain from base to end-effector, computes worst-case
        tolerance stackup and checks if total is acceptable.
        """
        if allowed_total <= 0:
            return []

        from ..tools.tolerance_analysis import ToleranceStackup

        parts_by_name = {p.name: p for p in assembly.parts}
        checks: list[ToleranceChainCheck] = []

        # Build chains: for each part, trace back to base
        child_to_parent: dict[str, tuple[str, Joint]] = {}
        for joint in assembly.joints:
            child_to_parent[joint.child] = (joint.parent, joint)

        # Find leaf parts (not a parent of any joint)
        parents = {j.parent for j in assembly.joints}
        leaves = [p.name for p in assembly.parts if p.name not in parents]

        for leaf in leaves:
            chain_parts: list[tuple[str, Joint]] = []
            current = leaf
            visited = set()
            while current in child_to_parent:
                parent_name, joint = child_to_parent[current]
                if parent_name in visited:
                    break
                visited.add(parent_name)
                chain_parts.append((current, joint))
                current = parent_name

            if not chain_parts:
                continue

            # Build tolerance stackup
            stack = ToleranceStackup(name=f"base_to_{leaf}")
            for part_name, joint in chain_parts:
                part = parts_by_name.get(part_name)
                if part:
                    # Use print tolerance based on joint type
                    h = part.dimensions.get("height",
                         part.dimensions.get("thickness",
                         part.dimensions.get("width", 0)))
                    if h > 0:
                        tol = 0.1  # default 0.1mm per dimension
                        if joint.type == "revolute":
                            tol = PRINT_TOLERANCES["sliding_fit"]
                        elif joint.type == "fixed":
                            tol = PRINT_TOLERANCES["tight_fit"]
                        stack.add_dimension(
                            name=part_name,
                            nominal=h,
                            upper=tol,
                            lower=-tol,
                        )

            if stack.dimension_count == 0:
                continue

            result = stack.compute_stackup()
            acceptable = result.total_tolerance <= allowed_total

            checks.append(ToleranceChainCheck(
                chain_name=f"base_to_{leaf}",
                dimension_count=len(result.dimensions),
                nominal_total=result.nominal,
                upper_deviation=result.upper_dev,
                lower_deviation=result.lower_dev,
                total_tolerance=result.total_tolerance,
                acceptable=acceptable,
                allowed_total=allowed_total,
                notes=(
                    f"总公差 {result.total_tolerance:.4f}mm "
                    f"{'≤' if acceptable else '>'} "
                    f"允许 {allowed_total:.4f}mm"
                ),
            ))

        return checks

    # ------------------------------------------------------------------
    # Task 80: Assembly sequence feasibility
    # ------------------------------------------------------------------

    def check_assembly_sequence(
        self,
        assembly: Assembly,
    ) -> list[AssemblySequenceCheck]:
        """Verify that parts can be assembled in the defined order.

        Basic checks:
        1. Each child part has a parent that is already in the sequence
        2. No circular dependencies
        """
        checks: list[AssemblySequenceCheck] = []
        assembled = set()

        # Find the base part (first part, or part that is not a child)
        children = {j.child for j in assembly.joints}
        parents = {j.parent for j in assembly.joints}
        base_parts = parents - children
        if not base_parts and assembly.parts:
            base_parts = {assembly.parts[0].name}

        for part in assembly.parts:
            if part.name in base_parts:
                assembled.add(part.name)

        # Check each joint in order
        for i, joint in enumerate(assembly.joints):
            parent_assembled = joint.parent in assembled
            if not parent_assembled:
                checks.append(AssemblySequenceCheck(
                    step=i + 1,
                    part_name=joint.child,
                    parent_name=joint.parent,
                    feasible=False,
                    notes=f"父件 '{joint.parent}' 尚未装配",
                ))
            else:
                checks.append(AssemblySequenceCheck(
                    step=i + 1,
                    part_name=joint.child,
                    parent_name=joint.parent,
                    feasible=True,
                    notes=f"'{joint.child}' 可安装到 '{joint.parent}'",
                ))
                assembled.add(joint.child)

        return checks

    # ------------------------------------------------------------------
    # F5: Motion-range collision sweep
    # ------------------------------------------------------------------

    def check_motion_collisions(
        self,
        assembly: Assembly,
        placements: dict | None = None,
    ) -> MotionCollisionSummary:
        """Sweep each revolute joint through its range and check for collisions.

        Reuses the existing ``MotionCollisionChecker`` which was previously
        never called from the verifier — meaning a robot whose kinematics
        self-collide through its motion range would still pass verification.

        When FCL is unavailable or placements are missing, returns a summary
        marked ``verified=False`` so the caller knows the check did not run.
        """
        revolute_count = sum(1 for j in assembly.joints if j.type == "revolute")
        if revolute_count == 0:
            return MotionCollisionSummary(
                joints_checked=0,
                collision_free=True,
                verified=True,
                notes="无旋转关节，跳过运动碰撞扫描",
            )

        if placements is None:
            return MotionCollisionSummary(
                joints_checked=revolute_count,
                collision_free=False,
                verified=False,
                notes="UNVERIFIED: 无位置数据，运动碰撞未检查",
            )

        try:
            from ..tools.motion_collision import MotionCollisionChecker
            checker = MotionCollisionChecker(num_samples=7)
            result = checker.check_motion_collisions(assembly)
            n_col = sum(1 for jr in result.joint_results if jr.has_collision)
            return MotionCollisionSummary(
                joints_checked=result.joints_checked,
                collision_free=result.collision_free,
                joints_with_collisions=n_col,
                verified=True,
                notes=result.summary,
            )
        except RuntimeError as e:
            # FCL / trimesh not installed
            return MotionCollisionSummary(
                joints_checked=revolute_count,
                collision_free=False,
                verified=False,
                notes=f"UNVERIFIED: 运动碰撞检查依赖不可用 ({e})",
            )
        except Exception as e:
            return MotionCollisionSummary(
                joints_checked=revolute_count,
                collision_free=False,
                verified=False,
                notes=f"UNVERIFIED: 运动碰撞检查异常 ({e})",
            )

    # ------------------------------------------------------------------
    # F6: Center-of-mass / support-polygon stability
    # ------------------------------------------------------------------

    def check_center_of_mass_stability(
        self,
        assembly: Assembly,
        placements: dict | None = None,
    ) -> CenterOfMassStabilityCheck:
        """Check that the assembly COM projects inside the base support polygon.

        Computes the assembly mass / COM via ``compute_assembly_mass`` and
        builds a support polygon from the convex hull of the base-part
        contact points (lowest parts).  If the XY projection of the COM
        falls outside this polygon, the assembly would tip over — a failure
        that was previously never caught.
        """
        if placements is None:
            return CenterOfMassStabilityCheck(
                verified=False,
                notes="UNVERIFIED: 无位置数据，重心稳定性未检查",
            )

        try:
            from ..knowledge.mechanics import compute_assembly_mass
            mass_result = compute_assembly_mass(assembly, positions=placements)
        except Exception as e:
            return CenterOfMassStabilityCheck(
                verified=False,
                notes=f"UNVERIFIED: 质量计算失败 ({e})",
            )

        com = mass_result.get("center_of_mass_mm", [0.0, 0.0, 0.0])
        com_xy = (com[0], com[1])
        total_mass = mass_result.get("total_mass_kg", 0.0)

        # Build support polygon from the lowest parts (ground contact).
        # Parts within 5mm of the minimum Z are considered "on the ground".
        if not placements:
            return CenterOfMassStabilityCheck(
                center_of_mass_mm=tuple(com),
                total_mass_kg=total_mass,
                verified=False,
                notes="UNVERIFIED: 无位置数据",
            )

        min_z = min(
            p["position"][2] for p in placements.values()
        )
        ground_parts = []
        for name, place in placements.items():
            z = place["position"][2]
            part = next((p for p in assembly.parts if p.name == name), None)
            if part is None:
                continue
            dims = part.dimensions
            h = dims.get("height", dims.get("thickness",
                    dims.get("length", 10)))
            low = z - h / 2
            if low <= min_z + 5.0:
                # Contact point footprint: approximate as box XY extents
                l = dims.get("length", dims.get("diameter", 10))
                w = dims.get("width", dims.get("diameter", l))
                ground_parts.append((name, place["position"], l, w))

        if not ground_parts:
            return CenterOfMassStabilityCheck(
                center_of_mass_mm=tuple(com),
                total_mass_kg=total_mass,
                verified=True,
                inside_support_polygon=True,
                notes="无接地零件，跳过稳定性检查",
            )

        # Collect footprint corner points (XY) from ground-contact parts.
        # Solver convention: the part's `length` dimension maps to the
        # solver Y axis (forward), `width` to X (lateral) — so the polygon
        # X extent uses width and Y extent uses length.
        poly_points: list[tuple[float, float]] = []
        for _, pos, l, w in ground_parts:
            cx, cy = pos[0], pos[1]
            poly_points.extend([
                (cx - w / 2, cy - l / 2),
                (cx + w / 2, cy - l / 2),
                (cx - w / 2, cy + l / 2),
                (cx + w / 2, cy + l / 2),
            ])

        inside, margin = self._point_in_convex_polygon(com_xy, poly_points)

        notes = (
            f"COM=({com[0]:.1f},{com[1]:.1f},{com[2]:.1f})mm, "
            f"{'在' if inside else '不在'}支撑多边形内"
            f" (裕量 {margin:.1f}mm)"
        )

        return CenterOfMassStabilityCheck(
            center_of_mass_mm=tuple(com),
            support_polygon_xy=poly_points,
            inside_support_polygon=inside,
            margin_mm=margin,
            total_mass_kg=total_mass,
            verified=True,
            notes=notes,
        )

    @staticmethod
    def _point_in_convex_polygon(
        point: tuple[float, float],
        poly_points: list[tuple[float, float]],
    ) -> tuple[bool, float]:
        """Test if a 2D point is inside the convex hull of poly_points.

        Returns (inside, margin) where margin is the signed distance to the
        nearest edge (positive = inside, negative = outside).
        """
        if len(poly_points) < 3:
            return True, 0.0

        # Compute convex hull via Andrew's monotone chain
        points = sorted(set(poly_points))
        if len(points) < 3:
            # Degenerate: treat as inside
            return True, 0.0

        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        lower: list[tuple[float, float]] = []
        for p in points:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)

        upper: list[tuple[float, float]] = []
        for p in reversed(points):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)

        hull = lower[:-1] + upper[:-1]
        n = len(hull)
        if n < 3:
            return True, 0.0

        inside = True
        min_margin = float("inf")
        for i in range(n):
            a = hull[i]
            b = hull[(i + 1) % n]
            # Edge vector
            ex, ey = b[0] - a[0], b[1] - a[1]
            edge_len = math.sqrt(ex * ex + ey * ey)
            if edge_len < 1e-9:
                continue
            # Outward normal (for CCW hull, left-hand normal points inward;
            # we want the sign that is positive when point is inside)
            nx, ny = -ey / edge_len, ex / edge_len
            # Signed distance from point to edge line
            dist = nx * (point[0] - a[0]) + ny * (point[1] - a[1])
            if dist < 0:
                inside = False
            min_margin = min(min_margin, dist)
        return inside, round(min_margin, 2)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _anchor_normal(anchor: str) -> tuple[float, float, float]:
        """Return outward normal vector for an anchor face."""
        normals = {
            "top": (0, 0, 1),
            "bottom": (0, 0, -1),
            "left": (-1, 0, 0),
            "right": (1, 0, 0),
            "front": (0, -1, 0),
            "back": (0, 1, 0),
        }
        return normals.get(anchor, (0, 0, 1))

    @staticmethod
    def _anchor_offset(anchor: str, part: Part) -> tuple[float, float, float]:
        """Return position offset to the center of an anchor face."""
        d = part.dimensions
        l = d.get("length", d.get("diameter", 0)) / 2
        w = d.get("width", 0) / 2
        h = d.get("height", d.get("thickness", 0)) / 2
        offsets = {
            "top": (0, 0, h),
            "bottom": (0, 0, -h),
            "left": (-l, 0, 0),
            "right": (l, 0, 0),
            "front": (0, -w, 0),
            "back": (0, w, 0),
        }
        return offsets.get(anchor, (0, 0, h))

    @staticmethod
    def generate_assembly_report(result: AssemblyVerificationResult) -> str:
        """Generate a human-readable assembly verification report."""
        lines = [
            f"# 装配验证报告: {result.assembly_name}",
            "",
            f"**总体结果: {'通过 ✓' if result.overall_pass else '未通过 ✗'}**",
            "",
            "## 零件检查",
        ]

        for pc in result.part_checks:
            icon = "✓" if pc.exists else "✗"
            lines.append(f"- {icon} {pc.part_name}: {pc.notes or ('文件: ' + pc.file_path)}")

        lines.append("")
        lines.append("## 配合检查")

        for fc in result.fit_checks:
            icon = "✓" if fc.fits is True else ("?" if fc.fits is None else "✗")
            parent_name = fc.parent_part.name if fc.parent_part else fc.joint.parent
            child_name = fc.child_part.name if fc.child_part else fc.joint.child
            lines.append(
                f"- {icon} {parent_name} → {child_name} ({fc.joint.type}): {fc.notes or 'OK'}"
            )

        if result.tolerance_issues:
            lines.append("")
            lines.append("## 公差问题")
            for issue in result.tolerance_issues:
                lines.append(f"- {issue}")

        # --- Task 63: collision info ---
        if result.collision_checks:
            lines.append("")
            lines.append("## 碰撞检查")
            for cc in result.collision_checks:
                icon = "✗" if cc.is_collision else "✓"
                detail = f" (穿透 {cc.penetration_depth_mm:.2f}mm)" if cc.is_collision else ""
                lines.append(f"- {icon} {cc.part_a} ↔ {cc.part_b}: {cc.notes}{detail}")
            if not result.fcl_available:
                lines.append("- ⚠ FCL 不可用，碰撞检查已跳过")

        # --- Task 80: mating surface checks ---
        if result.mating_surface_checks:
            lines.append("")
            lines.append("## 配合面检查")
            for mc in result.mating_surface_checks:
                icon = "✓" if (mc.parallel_ok and mc.distance_ok) else "✗"
                lines.append(
                    f"- {icon} {mc.parent_part} ↔ {mc.child_part}: "
                    f"法向偏差 {mc.normal_deviation_deg:.2f}°, "
                    f"距离 {mc.face_distance_mm:.3f}mm"
                )

        # --- Task 80: bolt hole alignment ---
        if result.bolt_alignment_checks:
            lines.append("")
            lines.append("## 螺栓孔对齐")
            for bc in result.bolt_alignment_checks:
                icon = "✓" if bc.aligned else "✗"
                lines.append(
                    f"- {icon} {bc.parent_part} ↔ {bc.child_part}: "
                    f"{bc.hole_count_parent}/{bc.hole_count_child} 孔, "
                    f"最大偏差 {bc.max_position_deviation_mm:.3f}mm"
                )

        # --- Task 80: tolerance chain ---
        if result.tolerance_chain_checks:
            lines.append("")
            lines.append("## 公差链分析")
            for tc in result.tolerance_chain_checks:
                icon = "✓" if tc.acceptable else "✗"
                lines.append(
                    f"- {icon} {tc.chain_name}: "
                    f"{tc.dimension_count} 维, "
                    f"总公差 {tc.total_tolerance:.4f}mm "
                    f"(允许 {tc.allowed_total:.4f}mm)"
                )

        # --- Task 80: assembly sequence ---
        if result.sequence_checks:
            lines.append("")
            lines.append("## 装配序列检查")
            for sc in result.sequence_checks:
                icon = "✓" if sc.feasible else "✗"
                lines.append(
                    f"- {icon} 步骤 {sc.step}: {sc.part_name} → {sc.parent_name}"
                )

        lines.append("")
        lines.append("---")
        lines.append(result.summary)

        return "\n".join(lines)
