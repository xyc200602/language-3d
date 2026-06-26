"""Shared pipeline context for export_engineering_package.

Avoids redundant computation across the 12-step export pipeline by computing
assembly positions, mass, and subsystem decomposition once and sharing them
with all downstream modules.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from ..knowledge.mechanics import Assembly, compute_assembly_mass

logger = logging.getLogger(__name__)


@dataclass
class AssemblyContext:
    """Shared intermediate results for the export pipeline.

    Created once at the start of export_engineering_package() and passed
    to all downstream modules so they don't need to recompute positions,
    mass, subsystems, etc.

    Two sets of positions are tracked:
      * ``positions`` — the *default pose* (with ``assembly.default_angles``
        applied).  Used for visualization, mass properties, and rendering.
      * ``home_positions`` — the *home pose* (all joint angles = 0).  Used
        for URDF export so that joint origins describe the ZERO state
        rather than baking in default_angles.  Without this split, a
        URDF for an arm with default_angles={'shoulder': -45} would have
        the -45° rotation baked into the shoulder joint's ``rpy``,
        meaning MuJoCo's qpos=0 corresponds to the bent pose instead of
        the home pose — making downstream simulation and control wrong.
    """

    assembly: Assembly
    positions: dict[str, Any] = field(default_factory=dict)
    home_positions: dict[str, Any] = field(default_factory=dict)
    mass_result: dict[str, Any] = field(default_factory=dict)
    subsystems: dict[str, list[str]] = field(default_factory=dict)
    kinematic_analysis: dict[str, Any] = field(default_factory=dict)
    solved: bool = False
    home_solved: bool = False
    mass_computed: bool = False
    subsystems_built: bool = False
    kinematic_analyzed: bool = False

    def ensure_positions(self) -> dict[str, Any]:
        """Compute assembly positions (default pose) if not already done."""
        if not self.solved:
            from .assembly_solver import AssemblySolver

            # Ensure default_angles exist for arm-like assemblies.
            # If the LLM didn't provide them, inject reasonable bend
            # angles so the arm isn't a straight line.
            self._ensure_default_angles()

            solver = AssemblySolver(self.assembly)
            try:
                self.positions = solver.solve()
            except (ValueError, RuntimeError) as e:
                logger.warning(
                    "Default-pose solve failed (%s) — using empty "
                    "positions. Downstream steps will use part defaults.",
                    e,
                )
                self.positions = {}
            # Adjust ground contact: find lowest part considering geometry
            self._adjust_ground_contact()
            self._validate_positions()
            self.solved = True
        return self.positions

    def ensure_home_positions(self) -> dict[str, Any]:
        """Compute positions with all joint angles = 0 (home pose).

        Used for URDF joint origin computation.  See class docstring for
        why this is distinct from ``ensure_positions()``.
        """
        if not self.home_solved:
            from .assembly_solver import AssemblySolver

            solver = AssemblySolver(self.assembly)
            # Explicit {} → solver does NOT fall back to default_angles,
            # so every revolute joint sits at 0°.
            #
            # P0-4: the all-zero solve can hit _clamp_child_offset ValueError
            # when LLM-generated offsets are fine in a bent default pose but
            # extreme when the arm is straight (e.g. wrist→gripper offset
            # projects to 433mm vertically).  Previously this crashed the
            # entire export — now we fall back to the default-pose positions
            # with a warning so the engineering package is still produced.
            try:
                self.home_positions = solver.solve(joint_angles={})
            except (ValueError, RuntimeError) as e:
                logger.warning(
                    "Home-pose (all-zero) solve failed (%s) — falling "
                    "back to default-pose positions for URDF export. "
                    "Joint origin RPY may bake in default_angles; this "
                    "is a data-quality issue, not a pipeline crash.",
                    e,
                )
                self.home_positions = self.ensure_positions()
            self.home_solved = True
        return self.home_positions

    def _ensure_default_angles(self) -> None:
        """If default_angles are missing, inject reasonable arm bend angles."""
        da = self.assembly.default_angles
        # If already set with non-zero values, keep them
        if da and any(v != 0 for v in da.values()):
            return

        # Check if this looks like an arm (has revolute joints)
        revolute_children = [
            j.child for j in self.assembly.joints if j.type == "revolute"
        ]
        if not revolute_children:
            return

        # Generate progressive bend angles: first joint -45°, then decreasing
        n = len(revolute_children)
        for i, child in enumerate(revolute_children):
            if child not in da:
                # Progressive angles: -45, -30, -15, 10 ...
                angle = -45 + i * 20
                if i == n - 1 and n > 2:
                    angle = 15  # wrist angles up slightly
                da[child] = angle

        self.assembly.default_angles = da

    def _adjust_ground_contact(self) -> None:
        """Shift all positions up so the lowest point sits at Z=0."""
        min_z = float("inf")
        for name, placement in self.positions.items():
            pos = placement["position"]
            z_center = pos[2]
            # Estimate the lowest extent of the part
            part = next(
                (p for p in self.assembly.parts if p.name == name), None
            )
            if part is None:
                continue
            dims = part.dimensions
            # For cylindrical parts, the lowest point after rotation
            # depends on the radius (diameter/2)
            rot = placement.get("rotation", [0, 0, 1, 0])
            angle_deg = abs(rot[3]) if len(rot) > 3 else 0
            if "diameter" in dims or "outer_diameter" in dims:
                d = dims.get("outer_diameter", dims.get("diameter", 10))
                radius = d / 2
                # If rotated significantly, the cylinder axis is not vertical
                # and the lowest point is at z_center - radius
                if angle_deg > 10:
                    low = z_center - radius
                else:
                    h = dims.get("height", dims.get("length", 10))
                    low = z_center - h / 2
            else:
                # Box part: half-height in Z
                h = dims.get("height", dims.get("thickness", 5))
                # If rotated, the bounding box changes, but approximate
                low = z_center - h / 2
            if low < min_z:
                min_z = low

        if min_z < 0:
            shift = -min_z
            for placement in self.positions.values():
                placement["position"][2] = round(
                    placement["position"][2] + shift, 4
                )

    def _validate_positions(self) -> list[str]:
        """Validate solved positions. Returns warnings."""
        import math
        warnings = []

        # 1. NaN/Inf
        for name, pdata in self.positions.items():
            pos = pdata.get("position", [0, 0, 0])
            if any(not math.isfinite(v) for v in pos):
                warnings.append(f"Part '{name}' has non-finite position: {pos}")

        # 2. Duplicate positions.
        # Parts joined by a joint legitimately share a position (e.g. a
        # suspension_link co-located with its motor at zero prismatic travel,
        # or two links at a coincident anchor). Skip such pairs — only flag
        # UNRELATED parts that overlap at the exact same point.
        _joint_pairs: set[tuple[str, str]] = set()
        for j in self.assembly.joints:
            _joint_pairs.add((j.parent, j.child))
            _joint_pairs.add((j.child, j.parent))
        seen: dict[str, str] = {}
        for name, pdata in self.positions.items():
            pos = pdata.get("position", [0, 0, 0])
            key = f"{pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f}"
            if key in seen:
                other = seen[key]
                if (name, other) not in _joint_pairs:
                    warnings.append(f"Parts '{name}' and '{other}' at same position")
            else:
                seen[key] = name

        # 3. Symmetry: parts with matching suffixes (_fl/_fr/_rl/_rr) should
        # have similar Z. IMPORTANT: group by the part's semantic role, not
        # just its corner suffix — otherwise standoffs (high, near the deck)
        # get lumped with wheels (low, near the ground) and trigger a false
        # 41.5mm "wheels Z variation" warning on every wheeled chassis. We
        # key the group on the prefix-before-the-corner (wheel_/motor_/...)
        # so each role is validated independently.
        corner_suffixes = ("_fl", "_fr", "_rl", "_rr")
        lr_suffixes = ("_l", "_r")
        groups: dict[str, list[str]] = {}
        for part in self.assembly.parts:
            grp = None
            if any(part.name.endswith(s) for s in corner_suffixes):
                # e.g. "wheel_fl" → prefix "wheel", "standoff_rr" → "standoff"
                prefix = part.name.rsplit("_", 1)[0]
                grp = prefix
            elif any(part.name.endswith(s) for s in lr_suffixes):
                prefix = part.name.rsplit("_", 1)[0]
                grp = prefix
            if grp is not None:
                groups.setdefault(grp, []).append(part.name)
        for grp_name, names in groups.items():
            if len(names) >= 2:
                zs = [self.positions[n]["position"][2] for n in names if n in self.positions]
                if zs and max(zs) - min(zs) > 20:
                    warnings.append(f"Symmetry group '{grp_name}' Z variation: {max(zs)-min(zs):.1f}mm")

        if warnings:
            import logging
            logging.getLogger(__name__).warning("Position validation: %s", "; ".join(warnings))
        return warnings

    def ensure_mass(self) -> dict[str, Any]:
        """Compute mass properties if not already done."""
        if not self.mass_computed:
            self.mass_result = compute_assembly_mass(
                self.assembly, positions=self.positions,
            )
            self.mass_computed = True
        return self.mass_result

    def ensure_subsystems(self) -> dict[str, list[str]]:
        """Build subsystem decomposition if not already done."""
        if not self.subsystems_built:
            from .export_package import _build_subsystems
            self.subsystems = _build_subsystems(self.assembly, self.positions)
            self.subsystems_built = True
        return self.subsystems

    def ensure_kinematic_analysis(self) -> dict[str, Any]:
        """Run closed-chain loop detection and differential-drive inference.

        Stamps the context with:
        - loops: list of closed kinematic loops (if any)
        - loop_count: number of detected loops
        - converged: True if ClosedChainSolver converged (or no loops)
        - error_mm: residual closure error in mm
        - differential_constraint: auto-detected differential-drive spec,
          or None if no wheel pair found.
        """
        if self.kinematic_analyzed:
            return self.kinematic_analysis

        from .assembly_solver import ClosedChainSolver, DifferentialConstraint

        solver = ClosedChainSolver(self.assembly)
        loops = solver.detect_loops()
        result: dict[str, Any] = {
            "loop_count": len(loops),
            "loops": [{"parts": loop} for loop in loops],
        }

        if loops:
            try:
                solve_result = solver.solve_closed_chain(
                    initial_angles=dict(self.assembly.default_angles),
                    max_iterations=50,
                    tolerance=0.5,
                )
                result["converged"] = bool(solve_result.get("converged"))
                result["iterations"] = int(solve_result.get("iterations", 0))
                result["error_mm"] = float(solve_result.get("error_mm", 0.0))
                if not solve_result.get("converged"):
                    logger.warning(
                        "Closed-chain solver did not converge: %d loops, "
                        "error=%.3fmm after %d iterations",
                        len(loops),
                        solve_result.get("error_mm", 0.0),
                        solve_result.get("iterations", 0),
                    )
            except Exception as e:
                logger.warning("Closed-chain solver failed: %s", e)
                result["converged"] = False
                result["error"] = str(e)
        else:
            result["converged"] = True
            result["iterations"] = 0
            result["error_mm"] = 0.0

        # Differential-drive auto-detection
        diff = self._detect_differential_constraint()
        if diff is not None:
            # Sanitize: turning_radius_mm defaults to inf — JSON-unsafe
            diff_dict = diff.to_dict()
            tr = diff_dict.get("turning_radius_mm")
            if tr is None or tr != tr or abs(tr) == float("inf"):
                diff_dict["turning_radius_mm"] = None
            result["differential_constraint"] = diff_dict
            result["differential_left_wheel"] = diff.left_wheel
            result["differential_right_wheel"] = diff.right_wheel
            result["differential_track_width_mm"] = diff.track_width_mm
            v_left, v_right = diff.speed_ratio(0.0)
            result["differential_straight_ratio"] = [round(v_left, 4), round(v_right, 4)]

        self.kinematic_analysis = result
        self.kinematic_analyzed = True
        return result

    def _detect_differential_constraint(self) -> "DifferentialConstraint | None":
        """Auto-detect a differential-drive wheel pair from the assembly.

        Looks for wheel pairs with naming patterns wheel_fl/wheel_fr,
        wheel_rl/wheel_rr, wheel_l/wheel_r, or mecanum_fl/fr.
        Returns a DifferentialConstraint with track width derived from
        solved positions, or None if no pair is found.
        """
        from .assembly_solver import DifferentialConstraint

        pair_patterns = [
            ("wheel_fl", "wheel_fr"),
            ("wheel_rl", "wheel_rr"),
            ("wheel_l", "wheel_r"),
            ("mecanum_fl", "mecanum_fr"),
            ("mecanum_rl", "mecanum_rr"),
        ]
        part_names = {p.name for p in self.assembly.parts}
        for left, right in pair_patterns:
            if left in part_names and right in part_names:
                # Derive track width from solved positions
                left_pos = self.positions.get(left, {}).get("position")
                right_pos = self.positions.get(right, {}).get("position")
                if left_pos and right_pos:
                    dx = right_pos[0] - left_pos[0]
                    dy = right_pos[1] - left_pos[1]
                    track = (dx * dx + dy * dy) ** 0.5
                    if track < 1.0:
                        track = 200.0  # fallback
                else:
                    track = 200.0
                return DifferentialConstraint(
                    left_wheel=left,
                    right_wheel=right,
                    track_width_mm=round(track, 1),
                    description=f"auto-detected {left}/{right} pair",
                )
        return None

    def get_com(self) -> list[float]:
        """Get center of mass, computing mass if needed."""
        mass = self.ensure_mass()
        return list(mass["center_of_mass_mm"])
