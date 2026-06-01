"""Assembly verification - fit checks, tolerance analysis, and completeness."""

from __future__ import annotations

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
    fits: bool = False
    notes: str = ""


@dataclass
class PartCheck:
    """Result of checking a single part's file existence."""

    part_name: str
    exists: bool = False
    file_path: str = ""
    notes: str = ""


@dataclass
class AssemblyVerificationResult:
    """Complete assembly verification result."""

    assembly_name: str
    part_checks: list[PartCheck] = field(default_factory=list)
    fit_checks: list[FitCheck] = field(default_factory=list)
    tolerance_issues: list[str] = field(default_factory=list)
    overall_pass: bool = False
    summary: str = ""


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
    ) -> AssemblyVerificationResult:
        """Run full assembly verification.

        Args:
            assembly: The Assembly definition to verify.
            workspace: Directory where part files should exist.
            parts_results: Optional mapping of part_name -> {artifacts, result, ...}

        Returns:
            AssemblyVerificationResult with all checks.
        """
        workspace = Path(workspace)

        part_checks = self.check_part_completeness(assembly, workspace, parts_results)
        fit_checks = self.check_joint_fits(assembly)
        screw_issues = self.check_screw_holes(assembly)

        # Collect tolerance issues
        tolerance_issues: list[str] = []
        for fc in fit_checks:
            if not fc.fits:
                tolerance_issues.append(
                    f"{fc.parent_part.name if fc.parent_part else '?'} -> "
                    f"{fc.child_part.name if fc.child_part else '?'}: "
                    f"clearance {fc.clearance:.2f}mm < required {fc.required_clearance:.2f}mm"
                )
        tolerance_issues.extend(screw_issues)

        # Determine overall pass
        all_parts_exist = all(pc.exists for pc in part_checks)
        all_fits_ok = all(fc.fits for fc in fit_checks)
        overall_pass = all_parts_exist and all_fits_ok and len(screw_issues) == 0

        # Build summary
        completed_parts = sum(1 for pc in part_checks if pc.exists)
        failed_fits = sum(1 for fc in fit_checks if not fc.fits)
        summary = (
            f"装配验证: {assembly.name}\n"
            f"- 零件完成: {completed_parts}/{len(part_checks)}\n"
            f"- 配合检查: {len(fit_checks) - failed_fits}/{len(fit_checks)} 通过\n"
            f"- 公差问题: {len(tolerance_issues)}\n"
            f"- 总体结果: {'通过' if overall_pass else '未通过'}"
        )

        return AssemblyVerificationResult(
            assembly_name=assembly.name,
            part_checks=part_checks,
            fit_checks=fit_checks,
            tolerance_issues=tolerance_issues,
            overall_pass=overall_pass,
            summary=summary,
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
            icon = "✓" if fc.fits else "✗"
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

        lines.append("")
        lines.append("---")
        lines.append(result.summary)

        return "\n".join(lines)
