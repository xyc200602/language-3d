"""Assembly-level visual verification with VLM closed-loop correction.

Implements the CADCodeVerify (ICLR 2025) iterative verification pattern:
solve assembly → render multi-angle screenshots → VLM evaluates visual
correctness → detect problems → generate correction feedback → re-solve
→ loop (max 3 rounds).
"""

from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from ..knowledge.mechanics import Assembly, Joint

logger = logging.getLogger(__name__)


class ProblemType(str, Enum):
    COLLISION = "collision"
    FLOATING = "floating"
    WRONG_ORIENTATION = "wrong_orientation"
    UNREASONABLE_LAYOUT = "unreasonable_layout"


class Severity(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class LayoutProblem:
    """A detected layout problem from VLM visual verification."""

    problem_type: ProblemType
    severity: Severity
    description: str
    affected_parts: list[str] = field(default_factory=list)
    suggestion: str = ""


@dataclass
class AssemblyVisualVerificationResult:
    """Result of an assembly visual verification round."""

    passed: bool = False
    problems: list[LayoutProblem] = field(default_factory=list)
    vlm_response: str = ""
    round_number: int = 0
    corrections_applied: list[dict[str, Any]] = field(default_factory=list)


def _build_assembly_prompt(
    assembly: Assembly,
    expected_layout: str = "",
    screenshot_paths: list[str] | None = None,
) -> str:
    """Build the VLM prompt for assembly-level verification.

    Follows the CADCodeVerify pattern: describe what you see, compare with
    expectations, structured output.
    """
    parts_summary = "\n".join(
        f"  - {p.name} ({p.category}): {p.description}"
        for p in assembly.parts
    )
    joints_summary = "\n".join(
        f"  - {j.parent} ← {j.type} → {j.child}"
        for j in assembly.joints
    )

    prompt = (
        "You are a 3D CAD assembly verification expert.\n\n"
        "Step 1: Describe what you see in the 3D viewport images.\n"
        "Look at the spatial arrangement of all parts carefully.\n\n"
        "Step 2: Check for the following assembly problems:\n"
        "  - **Collision**: Parts overlapping or intersecting\n"
        "  - **Floating**: Parts not connected to anything (unsupported)\n"
        "  - **Wrong orientation**: Parts rotated incorrectly\n"
        "  - **Unreasonable layout**: Parts too far apart or in illogical positions\n\n"
        "Step 3: Compare with the expected layout description.\n\n"
        f"Assembly: {assembly.name} ({len(assembly.parts)} parts)\n"
        f"Parts:\n{parts_summary}\n\n"
        f"Joints:\n{joints_summary}\n\n"
    )

    if expected_layout:
        prompt += f"Expected layout: {expected_layout}\n\n"

    prompt += (
        'Respond with a JSON object:\n'
        '{"passed": true/false, '
        '"problems": [{"type": "collision|floating|wrong_orientation|unreasonable_layout", '
        '"severity": "high|medium|low", '
        '"description": "...", '
        '"affected_parts": ["part1", "part2"], '
        '"suggestion": "..."}], '
        '"overall_assessment": "..."}\n'
    )
    return prompt


def _parse_layout_problems(vlm_response: str) -> list[LayoutProblem]:
    """Parse VLM JSON response into LayoutProblem list."""
    problems: list[LayoutProblem] = []

    # Try to extract JSON from response
    json_str = vlm_response
    # Try code block extraction
    if "```json" in json_str:
        start = json_str.index("```json") + 7
        end = json_str.index("```", start)
        json_str = json_str[start:end].strip()
    elif "```" in json_str:
        start = json_str.index("```") + 3
        end = json_str.index("```", start)
        json_str = json_str[start:end].strip()

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        # Try to find { ... } in the response
        start = vlm_response.find("{")
        end = vlm_response.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                data = json.loads(vlm_response[start:end])
            except json.JSONDecodeError:
                logger.warning("Failed to parse VLM response as JSON")
                return problems
        else:
            return problems

    raw_problems = data.get("problems", [])
    for rp in raw_problems:
        try:
            problems.append(LayoutProblem(
                problem_type=ProblemType(rp.get("type", "unreasonable_layout")),
                severity=Severity(rp.get("severity", "medium")),
                description=rp.get("description", ""),
                affected_parts=rp.get("affected_parts", []),
                suggestion=rp.get("suggestion", ""),
            ))
        except (ValueError, KeyError):
            continue

    return problems


def _generate_constraint_corrections(
    problems: list[LayoutProblem],
    assembly: Assembly,
) -> list[dict[str, Any]]:
    """Convert visual problems into constraint correction suggestions.

    Returns a list of correction dicts with keys:
    - joint_index: index in assembly.joints
    - correction_type: "offset" | "angle" | "attachment"
    - value: the corrected value
    - reason: why this correction was made
    """
    corrections: list[dict[str, Any]] = []

    for problem in problems:
        if problem.problem_type == ProblemType.COLLISION:
            # Move parts apart by adjusting offset
            for part_name in problem.affected_parts:
                for i, joint in enumerate(assembly.joints):
                    if joint.child == part_name or joint.parent == part_name:
                        corrections.append({
                            "joint_index": i,
                            "correction_type": "offset",
                            "value": 5.0,  # mm offset to resolve collision
                            "reason": f"Collision: {problem.description}",
                        })
                        break

        elif problem.problem_type == ProblemType.FLOATING:
            # Suggest fixed joint for floating parts
            for part_name in problem.affected_parts:
                # Check if this part has any joint
                has_joint = any(
                    j.child == part_name or j.parent == part_name
                    for j in assembly.joints
                )
                if not has_joint:
                    corrections.append({
                        "part_name": part_name,
                        "correction_type": "add_joint",
                        "value": "fixed",
                        "reason": f"Floating part: {problem.description}",
                    })

        elif problem.problem_type == ProblemType.WRONG_ORIENTATION:
            for part_name in problem.affected_parts:
                for i, joint in enumerate(assembly.joints):
                    if joint.child == part_name:
                        corrections.append({
                            "joint_index": i,
                            "correction_type": "angle",
                            "value": 90.0,  # degree rotation
                            "reason": f"Wrong orientation: {problem.description}",
                        })
                        break

    return corrections


def apply_corrections(
    assembly: Assembly,
    corrections: list[dict[str, Any]],
) -> Assembly:
    """Apply constraint corrections to an assembly and return a modified copy.

    Does not mutate the original assembly.
    """
    # Deep copy assembly data
    import copy
    new_parts = [copy.deepcopy(p) for p in assembly.parts]
    new_joints = [copy.deepcopy(j) for j in assembly.joints]

    for corr in corrections:
        if "joint_index" in corr:
            idx = corr["joint_index"]
            if idx < len(new_joints):
                joint = new_joints[idx]
                ctype = corr["correction_type"]
                if ctype == "offset":
                    # Adjust position by adding Z offset
                    current_offset = joint.offset or (0, 0, 0)
                    joint.offset = (
                        current_offset[0],
                        current_offset[1],
                        current_offset[2] + corr["value"],
                    )
                elif ctype == "angle":
                    if not hasattr(joint, "angle"):
                        joint.angle = corr["value"]
                    else:
                        joint.angle = corr["value"]
        elif corr.get("correction_type") == "add_joint":
            # Add a fixed joint for floating parts
            part_name = corr["part_name"]
            # Find the nearest structural part to attach to
            parent = "base_plate"
            for p in new_parts:
                if p.category == "structural" and p.name != part_name:
                    parent = p.name
                    break
            new_joints.append(Joint(
                type="fixed",
                parent=parent,
                child=part_name,
                parent_anchor="top",
                child_anchor="bottom",
            ))

    return Assembly(
        name=assembly.name,
        parts=new_parts,
        joints=new_joints,
    )


def verify_assembly_visual(
    assembly: Assembly,
    positions: dict[str, dict],
    model_backend: Any = None,
    expected_layout: str = "",
    max_iterations: int = 3,
    detail_level: str = "detailed",
) -> AssemblyVisualVerificationResult:
    """Main entry: run closed-loop assembly visual verification.

    For each iteration:
    1. Render assembly → screenshots (if FreeCAD + GUI available)
    2. Send screenshots + prompt to VLM
    3. Parse response into LayoutProblem list
    4. If passed or max iterations reached, return result
    5. Otherwise, generate corrections and re-solve

    Args:
        assembly: The assembly to verify
        positions: Solved positions from assembly solver
        model_backend: VLM model backend (GLMBackend, etc.)
        expected_layout: Text description of expected layout
        max_iterations: Maximum verification iterations (default 3)
        detail_level: VLM analysis detail level

    Returns:
        AssemblyVisualVerificationResult with final state
    """
    iteration_history: list[AssemblyVisualVerificationResult] = []
    current_assembly = assembly
    current_positions = positions

    for round_num in range(1, max_iterations + 1):
        logger.info("Assembly visual verification round %d/%d", round_num, max_iterations)

        # Build prompt
        prompt = _build_assembly_prompt(current_assembly, expected_layout)

        # Attempt VLM verification if backend is available
        vlm_response = ""
        screenshots: list[str] = []

        if model_backend is not None:
            try:
                # Render assembly screenshots using matplotlib
                # Use a persistent temp dir that lives until after VLM calls
                render_tmpdir = tempfile.mkdtemp(prefix="vlm_verify_")
                try:
                    screenshots = _render_to_dir(
                        current_assembly, current_positions, render_tmpdir,
                    )
                    if screenshots:
                        # Send the isometric view to VLM (most informative)
                        # If detailed, also send front and top views
                        views_to_check = [screenshots[0]]  # isometric
                        if detail_level in ("detailed", "maximum") and len(screenshots) >= 3:
                            views_to_check = screenshots[:3]  # iso + front + top

                        all_responses = []
                        for ss_path in views_to_check:
                            resp = model_backend.vision(
                                ss_path,
                                prompt,
                                max_tokens=4096 if detail_level == "detailed" else 2048,
                            )
                            all_responses.append(resp)
                        vlm_response = "\n\n---\n\n".join(all_responses)
                    else:
                        # No screenshots — fall back to heuristic
                        logger.warning("No screenshots generated, using heuristic verification")
                        vlm_response = _heuristic_verification(current_assembly, current_positions)
                finally:
                    # Clean up temp files
                    import shutil
                    shutil.rmtree(render_tmpdir, ignore_errors=True)
            except Exception as e:
                logger.warning("VLM verification failed: %s", e)
                vlm_response = f'{{"passed": false, "problems": [], "overall_assessment": "VLM error: {e}"}}'
        else:
            # No backend: use heuristic-only verification
            vlm_response = _heuristic_verification(current_assembly, current_positions)

        # Parse problems
        problems = _parse_layout_problems(vlm_response)
        passed = len(problems) == 0

        # Check if the JSON says passed and no problems
        if '"passed": true' in vlm_response or '"passed":true' in vlm_response:
            if not problems:
                passed = True

        result = AssemblyVisualVerificationResult(
            passed=passed,
            problems=problems,
            vlm_response=vlm_response,
            round_number=round_num,
        )
        iteration_history.append(result)

        if passed:
            logger.info("Assembly visual verification PASSED on round %d", round_num)
            result.corrections_applied = []
            return result

        # Generate and apply corrections for next iteration
        corrections = _generate_constraint_corrections(problems, current_assembly)
        result.corrections_applied = corrections

        if round_num < max_iterations and corrections:
            logger.info(
                "Applying %d corrections for round %d",
                len(corrections),
                round_num + 1,
            )
            current_assembly = apply_corrections(current_assembly, corrections)

            # Re-solve assembly positions
            try:
                from ..tools.assembly_solver import AssemblySolver
                solver = AssemblySolver(current_assembly)
                current_positions = solver.solve()
            except Exception as e:
                logger.warning("Re-solve failed: %s", e)
                break

    # Return last result (max iterations reached)
    if iteration_history:
        return iteration_history[-1]
    return AssemblyVisualVerificationResult(
        passed=False,
        vlm_response="No iterations completed",
        round_number=0,
    )


def _render_to_dir(
    assembly: Assembly,
    positions: dict[str, dict],
    output_dir: str,
) -> list[str]:
    """Render assembly using matplotlib and save multi-angle screenshots.

    Uses matplotlib (no FreeCAD GUI required) for headless rendering.
    Saves PNGs to output_dir. Returns list of screenshot file paths.
    """
    screenshots: list[str] = []

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        import numpy as np

        # Subsystem colors (RGBA)
        _SUBSYS_COLORS = {
            "arm_left": (0.85, 0.30, 0.20, 0.8),
            "arm_right": (0.20, 0.75, 0.30, 0.8),
            "ipc": (0.75, 0.55, 0.10, 0.8),
            "sensor_tower": (0.60, 0.20, 0.75, 0.8),
        }

        def _get_subsystem(name: str) -> str:
            if name.startswith("arm_l_"): return "arm_left"
            if name.startswith("arm_r_"): return "arm_right"
            if "ipc" in name: return "ipc"
            if name.startswith(("sensor_", "imu_", "lidar_", "camera_")): return "sensor_tower"
            return "chassis"

        def _box_faces(l, w, h, pos):
            x, y, z = pos
            v = np.array([
                [x-l/2,y-w/2,z],[x+l/2,y-w/2,z],[x+l/2,y+w/2,z],[x-l/2,y+w/2,z],
                [x-l/2,y-w/2,z+h],[x+l/2,y-w/2,z+h],[x+l/2,y+w/2,z+h],[x-l/2,y+w/2,z+h],
            ])
            return [
                [v[0],v[1],v[2],v[3]], [v[4],v[5],v[6],v[7]],
                [v[0],v[1],v[5],v[4]], [v[2],v[3],v[7],v[6]],
                [v[0],v[3],v[7],v[4]], [v[1],v[2],v[6],v[5]],
            ]

        def _cyl_faces(r, h, pos, n=12):
            x, y, z = pos
            bottom, top = [], []
            for i in range(n):
                a = 2*np.pi*i/n
                bx, by = x+r*np.cos(a), y+r*np.sin(a)
                bottom.append([bx,by,z])
                top.append([bx,by,z+h])
            faces = []
            for i in range(n):
                j = (i+1) % n
                faces.append([bottom[i],bottom[j],top[j],top[i]])
            faces.append(bottom)
            faces.append(top)
            return faces

        def _render_view(ax, elev, azim, title):
            ax.cla()
            ax.set_title(title, fontsize=14, fontweight='bold')
            for p in assembly.parts:
                dims = p.dimensions
                pos = positions.get(p.name, {}).get("position", [0,0,0])
                sub = _get_subsystem(p.name)
                color = _SUBSYS_COLORS.get(sub, (0.20, 0.40, 0.80, 0.7))

                if "diameter" in dims or "outer_diameter" in dims:
                    r = dims.get("diameter", dims.get("outer_diameter", 10)) / 2
                    h = dims.get("height", 10)
                    faces = _cyl_faces(r, h, pos)
                else:
                    l = dims.get("length", 10)
                    w = dims.get("width", 10)
                    h = dims.get("height", 10)
                    faces = _box_faces(l, w, h, pos)

                poly = Poly3DCollection(faces, alpha=0.75)
                poly.set_facecolor(color[:3])
                poly.set_edgecolor((0.2, 0.2, 0.2, 0.3))
                poly.set_linewidth(0.3)
                ax.add_collection3d(poly)

            ax.view_init(elev=elev, azim=azim)
            ax.set_xlim(-200, 200)
            ax.set_ylim(-150, 150)
            ax.set_zlim(-80, 280)
            ax.set_xlabel('X (mm)')
            ax.set_ylabel('Y (mm)')
            ax.set_zlabel('Z (mm)')

        views = [
            (25, 45, "isometric"),
            (0, 0, "front"),
            (90, 0, "top"),
            (0, 270, "right"),
        ]

        for elev, azim, vname in views:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            _render_view(ax, elev, azim, f"Assembly - {vname.title()} View")
            img_path = str(Path(output_dir) / f"assembly_{vname}.png")
            fig.savefig(img_path, dpi=100, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            if Path(img_path).exists():
                    screenshots.append(img_path)

    except Exception as e:
        logger.warning("Matplotlib render failed: %s", e)

    return screenshots


def _heuristic_verification(
    assembly: Assembly,
    positions: dict[str, dict],
) -> str:
    """Run heuristic checks when VLM is not available.

    Checks for basic issues: floating parts, extreme positions, etc.
    """
    problems = []

    # Check for parts without positions
    positioned_parts = set(positions.keys())
    for part in assembly.parts:
        if part.name not in positioned_parts:
            problems.append({
                "type": "floating",
                "severity": "high",
                "description": f"Part '{part.name}' has no position",
                "affected_parts": [part.name],
                "suggestion": f"Add a joint for {part.name}",
            })

    # Check for extreme positions (> 500mm from origin)
    for name, pos_data in positions.items():
        pos = pos_data.get("position", [0, 0, 0])
        dist = sum(x**2 for x in pos) ** 0.5
        if dist > 500:
            problems.append({
                "type": "unreasonable_layout",
                "severity": "medium",
                "description": f"Part '{name}' is {dist:.0f}mm from origin",
                "affected_parts": [name],
                "suggestion": f"Review constraints for {name}",
            })

    # Check for parts at same position (potential collision)
    pos_map: dict[str, str] = {}
    for name, pos_data in positions.items():
        pos = pos_data.get("position", [0, 0, 0])
        key = f"{pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f}"
        if key in pos_map:
            problems.append({
                "type": "collision",
                "severity": "high",
                "description": f"'{name}' and '{pos_map[key]}' at same position",
                "affected_parts": [name, pos_map[key]],
                "suggestion": "Adjust joint offsets to separate parts",
            })
        else:
            pos_map[key] = name

    passed = len(problems) == 0
    return json.dumps({
        "passed": passed,
        "problems": problems,
        "overall_assessment": f"Heuristic check: {len(problems)} issues found",
    })
