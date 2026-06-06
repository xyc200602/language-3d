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
                # Try to render and capture screenshots
                screenshots = _render_and_capture(
                    current_assembly, current_positions,
                )
                if screenshots:
                    # Send each screenshot to VLM
                    all_responses = []
                    for ss_path in screenshots:
                        resp = model_backend.vision(
                            ss_path,
                            prompt,
                            max_tokens=4096 if detail_level == "detailed" else 2048,
                        )
                        all_responses.append(resp)
                    vlm_response = "\n\n---\n\n".join(all_responses)
                else:
                    # No screenshots, use text-only analysis
                    vlm_response = model_backend.vision(
                        "",  # Empty image path will fail; just use prompt
                        prompt,
                    )
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


def _render_and_capture(
    assembly: Assembly,
    positions: dict[str, dict],
) -> list[str]:
    """Render assembly in FreeCAD GUI and capture multi-angle screenshots.

    Returns list of screenshot file paths.
    """
    screenshots: list[str] = []

    try:
        from ..tools.freecad import (
            FCOpenGUITool,
            FCSetCameraTool,
            build_assembly_script,
            _shape_type_for_part,
            _subsystem_for_part,
            _run_freecad_script,
        )

        # Build and execute assembly render script
        parts_info = [
            {
                "name": p.name,
                "shape_type": _shape_type_for_part(p),
                "dimensions": p.dimensions,
                "subsystem": _subsystem_for_part(p.name),
            }
            for p in assembly.parts
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            render_path = Path(tmpdir) / "assembly"
            script = build_assembly_script(
                assembly_parts=parts_info,
                positions=positions,
                output_path=str(render_path),
            )
            _run_freecad_script(script, timeout=120)

            fcstd_path = render_path.with_suffix(".FCStd")
            if not fcstd_path.exists():
                return screenshots

            # Open in GUI and capture screenshots
            open_tool = FCOpenGUITool()
            open_tool.execute(file_path=str(fcstd_path), wait_seconds=3)

            import time
            time.sleep(2)

            from ..tools.screen import _find_windows_by_title
            windows = _find_windows_by_title("FreeCAD")
            if not windows:
                return screenshots

            # Capture screenshots from multiple angles
            camera_tool = FCSetCameraTool()
            try:
                from ..tools.screen import ScreenCaptureTool
                capture_tool = ScreenCaptureTool()

                for view_name in ["isometric", "front", "top", "right"]:
                    try:
                        camera_tool.execute(
                            file_path=str(fcstd_path),
                            view=view_name,
                        )
                        time.sleep(2)
                        # Capture screenshot
                        screenshot_path = str(Path(tmpdir) / f"assembly_{view_name}.png")
                        capture_tool.execute(
                            region="fullscreen",
                            save_path=screenshot_path,
                        )
                        if Path(screenshot_path).exists():
                            screenshots.append(screenshot_path)
                    except Exception:
                        continue
            finally:
                # Close FreeCAD
                from ..tools.freecad import FCCloseGUITool
                FCCloseGUITool().execute()

    except Exception as e:
        logger.warning("Render and capture failed: %s", e)

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
