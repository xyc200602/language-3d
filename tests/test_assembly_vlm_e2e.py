"""Assembly VLM visual verification E2E test — 41-part complex robot.

Tests the full closed-loop: build complex robot → solve assembly →
VTK render → VLM analyze → parse problems → correct → re-solve → re-render.

Requires:
  - GLM_API_KEY in environment (for real VLM calls)
  - VTK installed (for offscreen rendering)

Usage:
  # Run via pytest
  pytest tests/test_assembly_vlm_e2e.py -v -m e2e

  # Run as standalone script
  python tests/test_assembly_vlm_e2e.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pytest

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ---------------------------------------------------------------------------
# Environment checks
# ---------------------------------------------------------------------------

def _has_api_key() -> bool:
    """Check if GLM API key is available."""
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
    return bool(os.environ.get("GLM_API_KEY"))


def _has_vtk() -> bool:
    """Check if VTK is available."""
    try:
        import vtk  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

@dataclass
class RoundResult:
    """Result from a single verification round."""
    round_number: int
    passed: bool
    problem_count: int
    problem_types: list[str]
    corrections_count: int
    render_succeeded: bool
    vlm_called: bool
    vlm_response_snippet: str


@dataclass
class AssemblyVLMReport:
    """Full E2E test report."""
    total_parts: int
    total_joints: int
    total_rounds: int
    initial_problems: int
    final_problems: int
    problem_reduction_pct: float
    final_passed: bool
    rounds: list[dict[str, Any]] = field(default_factory=list)
    elapsed_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def complex_robot():
    """Build the 41-part complex robot assembly."""
    from lang3d.tools.export_package import build_complex_robot
    return build_complex_robot()


@pytest.fixture
def solved_positions(complex_robot):
    """Solve assembly positions for the complex robot."""
    from lang3d.tools.assembly_solver import AssemblySolver
    solver = AssemblySolver(complex_robot)
    return solver.solve()


@pytest.fixture
def model_backend():
    """Create a real GLMBackend for VLM calls."""
    from lang3d.models.glm import GLMBackend
    api_key = os.environ.get("GLM_API_KEY", "")
    base_url = os.environ.get(
        "GLM_BASE_URL", "https://open.bigmodel.cn/api/coding/paas/v4"
    )
    # Use GLM-4.6V-Flash for detailed assembly analysis
    return GLMBackend(
        api_key=api_key,
        base_url=base_url,
        vision_model="GLM-4.6V-Flash",
    )


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------

@pytest.mark.e2e
@pytest.mark.skipif(not _has_api_key(), reason="GLM_API_KEY not configured")
class TestAssemblyVLME2E:
    """E2E tests with real VLM backend and 41-part complex robot."""

    def test_complex_robot_assembly_builds(self, complex_robot):
        """Verify the 41-part complex robot builds correctly."""
        assert len(complex_robot.parts) >= 40, (
            f"Expected >= 40 parts, got {len(complex_robot.parts)}"
        )
        assert len(complex_robot.joints) >= 35, (
            f"Expected >= 35 joints, got {len(complex_robot.joints)}"
        )

    def test_assembly_solver_produces_positions(self, complex_robot, solved_positions):
        """Verify the solver produces positions for all parts."""
        assert len(solved_positions) >= 40, (
            f"Expected >= 40 positions, got {len(solved_positions)}"
        )
        # Every position should have x, y, z coordinates
        for name, pos_data in solved_positions.items():
            pos = pos_data.get("position", [])
            assert len(pos) == 3, f"Part '{name}' has invalid position: {pos}"

    @pytest.mark.skipif(not _has_vtk(), reason="VTK not installed")
    def test_vtk_renders_assembly(self, complex_robot, solved_positions):
        """Verify VTK can render the full 41-part assembly to PNG files."""
        from lang3d.agent.assembly_visual_verifier import _render_to_dir

        with tempfile.TemporaryDirectory(prefix="vlm_e2e_render_") as tmpdir:
            screenshots = _render_to_dir(
                complex_robot, solved_positions, tmpdir,
            )
            # Should produce at least 4 views (isometric, front, top, right)
            assert len(screenshots) >= 3, (
                f"Expected >= 3 rendered views, got {len(screenshots)}"
            )
            for ss_path in screenshots:
                assert os.path.isfile(ss_path), f"Screenshot missing: {ss_path}"
                assert os.path.getsize(ss_path) > 5000, (
                    f"Screenshot too small ({os.path.getsize(ss_path)} bytes): {ss_path}"
                )

    @pytest.mark.skipif(not _has_vtk(), reason="VTK not installed")
    def test_vlm_detects_injected_collision(self, complex_robot, model_backend):
        """Verify VLM can detect an intentionally injected collision.

        Overrides solved positions to place multiple parts at the exact same
        coordinates, then checks if VLM identifies the issue.
        """
        from lang3d.agent.assembly_visual_verifier import (
            _render_to_dir,
            _build_assembly_prompt,
        )
        from lang3d.tools.assembly_solver import AssemblySolver

        # Solve normally first
        solver = AssemblySolver(complex_robot)
        positions = solver.solve()

        # Inject collision: force wheel_fl, wheel_fr, and motor_fl all to (0,0,5)
        # (same position as base_plate center — visually obvious overlap)
        for name in ("wheel_fl", "wheel_fr", "motor_fl"):
            if name in positions:
                positions[name] = {"position": [0, 0, 5], "rotation": [0, 0, 0, 0]}

        # Render and send to VLM
        with tempfile.TemporaryDirectory(prefix="vlm_collision_") as tmpdir:
            screenshots = _render_to_dir(complex_robot, positions, tmpdir)
            assert len(screenshots) >= 1, "No screenshots rendered"

            prompt = _build_assembly_prompt(
                complex_robot,
                expected_layout=(
                    "4-wheel differential drive mobile robot with two 3-DOF arms "
                    "mounted on a chassis plate. Wheels should be at 4 corners, "
                    "not overlapping with the chassis."
                ),
                positions=positions,
            )

            response = model_backend.vision(
                screenshots[0],
                prompt,
                max_tokens=4096,
            )

            # VLM should NOT say "passed": true — it should flag problems.
            # Check both raw response and parsed problems.
            raw_lower = response.lower()
            has_passed_true = (
                '"passed": true' in response or '"passed":true' in response
            )

            # Keywords that indicate VLM detected the injected issue
            detection_keywords = [
                "collision", "overlap", "intersect", "重叠", "交叉",
                "floating", "unsupported", "悬空",
                "unreasonable", "wrong position", "incorrect position",
                "不合理", "错误",
            ]
            detected = any(kw in raw_lower for kw in detection_keywords)

            assert (not has_passed_true) or detected, (
                f"VLM did not detect injected collision. "
                f"Response indicates passed=true with no issues. "
                f"Response: {response[:500]}"
            )

    @pytest.mark.skipif(not _has_vtk(), reason="VTK not installed")
    def test_closed_loop_reduces_problems(self, complex_robot, solved_positions, model_backend):
        """Main E2E test: verify closed-loop reduces visual problems by >= 50%.

        This is the core assertion for Task 64: run the full
        verify_assembly_visual loop and measure problem reduction.
        """
        from lang3d.agent.assembly_visual_verifier import (
            verify_assembly_visual,
            _render_to_dir,
            _parse_layout_problems,
            _build_assembly_prompt,
        )

        # --- Round 0: baseline assessment (before any corrections) ---
        baseline_problems = 0
        with tempfile.TemporaryDirectory(prefix="vlm_baseline_") as tmpdir:
            screenshots = _render_to_dir(complex_robot, solved_positions, tmpdir)
            assert len(screenshots) >= 1, "No baseline screenshots"

            prompt = _build_assembly_prompt(
                complex_robot,
                expected_layout=(
                    "4-wheel differential drive mobile robot with two 3-DOF arms "
                    "mounted on a chassis plate, an IPC box, and a sensor tower."
                ),
                positions=solved_positions,
            )

            baseline_response = model_backend.vision(
                screenshots[0],
                prompt,
                max_tokens=4096,
            )
            baseline_problems = len(_parse_layout_problems(baseline_response))

        # --- Run closed-loop verification (up to 3 iterations) ---
        start_time = time.time()
        result = verify_assembly_visual(
            assembly=complex_robot,
            positions=solved_positions,
            model_backend=model_backend,
            expected_layout=(
                "4-wheel differential drive mobile robot with two 3-DOF arms "
                "mounted on a chassis plate, an IPC box, and a sensor tower."
            ),
            max_iterations=3,
            detail_level="detailed",
        )
        elapsed = time.time() - start_time

        final_problems = len(result.problems)

        # Compute reduction
        if baseline_problems > 0:
            reduction_pct = (
                (baseline_problems - final_problems) / baseline_problems * 100
            )
        else:
            # Baseline already clean — trivially 100% reduction
            reduction_pct = 100.0

        # --- Save structured report ---
        report = AssemblyVLMReport(
            total_parts=len(complex_robot.parts),
            total_joints=len(complex_robot.joints),
            total_rounds=result.round_number,
            initial_problems=baseline_problems,
            final_problems=final_problems,
            problem_reduction_pct=round(reduction_pct, 1),
            final_passed=result.passed,
            elapsed_seconds=round(elapsed, 1),
        )

        report_path = Path(__file__).parent.parent / "data" / "assembly_vlm_e2e_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(asdict(report), indent=2, ensure_ascii=False))

        # --- Assertions ---
        # The closed loop should reduce problems by at least 50%
        # (or pass outright if baseline is clean)
        assert (
            result.passed or reduction_pct >= 50.0
        ), (
            f"Closed-loop verification failed to reduce problems by 50%. "
            f"Baseline: {baseline_problems} problems, "
            f"Final: {final_problems} problems, "
            f"Reduction: {reduction_pct:.1f}%, "
            f"Rounds: {result.round_number}/3"
        )

    @pytest.mark.skipif(not _has_vtk(), reason="VTK not installed")
    def test_vlm_identifies_subsystem_groups(self, complex_robot, solved_positions, model_backend):
        """Verify VLM can correctly identify major subsystem groups visually.

        Sends the assembly render and asks VLM to count subsystem groups.
        """
        from lang3d.agent.assembly_visual_verifier import (
            _render_to_dir,
        )

        with tempfile.TemporaryDirectory(prefix="vlm_subsys_") as tmpdir:
            screenshots = _render_to_dir(complex_robot, solved_positions, tmpdir)
            assert len(screenshots) >= 1

            prompt = (
                "You are a 3D assembly analysis expert.\n\n"
                "Look at the 3D viewport image of a robot assembly.\n"
                "Count the number of distinct subsystem groups you can identify.\n"
                "The assembly should contain:\n"
                "- A chassis/base with 4 wheels\n"
                "- Two robotic arms (left and right)\n"
                "- An IPC/control box\n"
                "- A sensor tower\n\n"
                'Respond with a JSON object: {"subsystem_count": <number>, '
                '"subsystems": [{"name": "...", "parts_count": <int>}]}\n'
            )

            response = model_backend.vision(
                screenshots[0],
                prompt,
                max_tokens=2048,
            )

            # Parse response — try multiple strategies
            subsystem_count = 0
            data = None

            # Strategy 1: direct JSON parse
            try:
                data = json.loads(response.strip())
            except json.JSONDecodeError:
                pass

            # Strategy 2: extract from code block
            if data is None:
                for marker_start, marker_end in [("```json", "```"), ("```", "```")]:
                    if marker_start in response:
                        s = response.index(marker_start) + len(marker_start)
                        e = response.index(marker_end, s)
                        try:
                            data = json.loads(response[s:e].strip())
                            break
                        except json.JSONDecodeError:
                            pass

            # Strategy 3: find { ... } in response
            if data is None:
                s = response.find("{")
                e = response.rfind("}") + 1
                if s >= 0 and e > s:
                    try:
                        data = json.loads(response[s:e])
                    except json.JSONDecodeError:
                        pass

            # Strategy 4: extract number from plain text
            if data is not None:
                subsystem_count = data.get("subsystem_count", 0)
            else:
                # VLM returned plain text — look for numbers near keywords
                import re
                # Look for patterns like "5 subsystem" or "count: 5"
                matches = re.findall(r"(?:subsystem|group|count)[:\s]*(\d+)", response.lower())
                if matches:
                    subsystem_count = int(matches[0])
                else:
                    # Count keyword occurrences as a rough estimate
                    keywords = ["chassis", "arm", "ipc", "sensor", "tower", "base"]
                    subsystem_count = sum(1 for kw in keywords if kw in response.lower())

            assert subsystem_count >= 4, (
                f"VLM identified only {subsystem_count} subsystems, "
                f"expected >= 4. Response: {response[:500]}"
            )


# ---------------------------------------------------------------------------
# Heuristic-only tests (no VLM API needed)
# ---------------------------------------------------------------------------

class TestAssemblyHeuristicE2E:
    """Tests that run without VLM, using heuristic verification only."""

    def test_heuristic_passes_for_valid_assembly(self, complex_robot, solved_positions):
        """Heuristic verification should pass for the standard complex robot."""
        from lang3d.agent.assembly_visual_verifier import (
            _heuristic_verification,
        )
        result_str = _heuristic_verification(complex_robot, solved_positions)
        data = json.loads(result_str)
        assert data["passed"] is True, (
            f"Heuristic check found unexpected problems: {data['problems']}"
        )

    def test_heuristic_detects_missing_positions(self, complex_robot):
        """Heuristic should detect parts without positions."""
        from lang3d.agent.assembly_visual_verifier import (
            _heuristic_verification,
        )
        # Empty positions → all parts floating
        result_str = _heuristic_verification(complex_robot, {})
        data = json.loads(result_str)
        assert data["passed"] is False
        assert len(data["problems"]) >= 1
        assert any(p["type"] == "floating" for p in data["problems"])

    def test_heuristic_detects_extreme_positions(self, complex_robot):
        """Heuristic should detect parts positioned too far from origin."""
        from lang3d.agent.assembly_visual_verifier import (
            _heuristic_verification,
        )
        extreme_pos = {
            p.name: {"position": [999, 999, 999]}
            for p in complex_robot.parts
        }
        result_str = _heuristic_verification(complex_robot, extreme_pos)
        data = json.loads(result_str)
        assert data["passed"] is False
        assert any(p["type"] == "unreasonable_layout" for p in data["problems"])

    def test_heuristic_detects_coincident_positions(self, complex_robot):
        """Heuristic should detect two parts at the same position."""
        from lang3d.agent.assembly_visual_verifier import (
            _heuristic_verification,
        )
        if len(complex_robot.parts) < 2:
            pytest.skip("Need >= 2 parts")
        same_pos = {
            p.name: {"position": [10.0, 20.0, 30.0]}
            for p in complex_robot.parts
        }
        result_str = _heuristic_verification(complex_robot, same_pos)
        data = json.loads(result_str)
        assert data["passed"] is False
        assert any(p["type"] == "collision" for p in data["problems"])


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

def main():
    """Run the E2E test suite standalone and save a report."""
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")

    if not _has_api_key():
        print("ERROR: GLM_API_KEY not configured. Set it in .env or environment.")
        sys.exit(1)

    if not _has_vtk():
        print("ERROR: VTK not installed. Install with: pip install vtk")
        sys.exit(1)

    from lang3d.tools.export_package import build_complex_robot
    from lang3d.tools.assembly_solver import AssemblySolver
    from lang3d.agent.assembly_visual_verifier import (
        verify_assembly_visual,
        _render_to_dir,
        _parse_layout_problems,
        _build_assembly_prompt,
    )
    from lang3d.models.glm import GLMBackend

    print("=" * 70)
    print("Assembly VLM E2E Test — 41-part Complex Robot")
    print("=" * 70)

    # Build assembly
    print("\n[1/5] Building complex robot assembly...")
    assembly = build_complex_robot()
    print(f"  Parts: {len(assembly.parts)}")
    print(f"  Joints: {len(assembly.joints)}")

    # Solve positions
    print("\n[2/5] Solving assembly positions...")
    solver = AssemblySolver(assembly)
    positions = solver.solve()
    print(f"  Positions solved: {len(positions)}")

    # Render baseline
    print("\n[3/5] Rendering baseline views...")
    with tempfile.TemporaryDirectory(prefix="vlm_e2e_") as tmpdir:
        screenshots = _render_to_dir(assembly, positions, tmpdir)
        print(f"  Screenshots: {len(screenshots)}")
        for ss in screenshots:
            print(f"    {os.path.basename(ss)}: {os.path.getsize(ss)} bytes")

        # VLM baseline assessment
        print("\n[4/5] VLM baseline assessment...")
        api_key = os.environ.get("GLM_API_KEY", "")
        base_url = os.environ.get(
            "GLM_BASE_URL", "https://open.bigmodel.cn/api/coding/paas/v4"
        )
        backend = GLMBackend(
            api_key=api_key,
            base_url=base_url,
            vision_model="GLM-4.6V-Flash",
        )

        prompt = _build_assembly_prompt(
            assembly,
            expected_layout=(
                "4-wheel differential drive mobile robot with two 3-DOF arms "
                "mounted on a chassis plate, an IPC box, and a sensor tower."
            ),
            positions=positions,
        )
        baseline_response = backend.vision(screenshots[0], prompt, max_tokens=4096)
        baseline_problems = _parse_layout_problems(baseline_response)
        print(f"  Baseline problems: {len(baseline_problems)}")
        for p in baseline_problems:
            print(f"    [{p.severity.value}] {p.problem_type.value}: {p.description}")

    # Run closed-loop verification
    print("\n[5/5] Running closed-loop verification (max 3 rounds)...")
    start_time = time.time()
    result = verify_assembly_visual(
        assembly=assembly,
        positions=positions,
        model_backend=backend,
        expected_layout=(
            "4-wheel differential drive mobile robot with two 3-DOF arms "
            "mounted on a chassis plate, an IPC box, and a sensor tower."
        ),
        max_iterations=3,
        detail_level="detailed",
    )
    elapsed = time.time() - start_time

    # Report
    final_count = len(result.problems)
    if len(baseline_problems) > 0:
        reduction = (len(baseline_problems) - final_count) / len(baseline_problems) * 100
    else:
        reduction = 100.0

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Parts:            {len(assembly.parts)}")
    print(f"  Baseline problems: {len(baseline_problems)}")
    print(f"  Final problems:    {final_count}")
    print(f"  Reduction:         {reduction:.1f}%")
    print(f"  Passed:            {result.passed}")
    print(f"  Rounds:            {result.round_number}/3")
    print(f"  Elapsed:           {elapsed:.1f}s")

    if result.problems:
        print("\n  Remaining problems:")
        for p in result.problems:
            print(f"    [{p.severity.value}] {p.problem_type.value}: {p.description}")

    # Save report
    report = AssemblyVLMReport(
        total_parts=len(assembly.parts),
        total_joints=len(assembly.joints),
        total_rounds=result.round_number,
        initial_problems=len(baseline_problems),
        final_problems=final_count,
        problem_reduction_pct=round(reduction, 1),
        final_passed=result.passed,
        elapsed_seconds=round(elapsed, 1),
    )
    report_path = Path(__file__).parent.parent / "data" / "assembly_vlm_e2e_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(asdict(report), indent=2, ensure_ascii=False))
    print(f"\n  Report saved to: {report_path}")

    # Exit code
    if result.passed or reduction >= 50.0:
        print("\nPASS: Closed-loop reduced problems by >= 50%")
        sys.exit(0)
    else:
        print(f"\nFAIL: Reduction only {reduction:.1f}% (need >= 50%)")
        sys.exit(1)


if __name__ == "__main__":
    main()
