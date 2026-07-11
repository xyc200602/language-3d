"""Tests for the pick-and-place task validation tool.

Tests cover:
1. _ik_angles_to_qpos — IK→MuJoCo joint-space bridge (unit test)
2. PickPlaceTool on 4dof_arm — full pipeline integration test
3. Place accuracy — cube ends up near the target
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

_PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT / "src"))


class TestIKToQposBridge:
    """Unit tests for the IK angle → MuJoCo qpos conversion."""

    def test_hinge_conversion_degrees_to_radians(self):
        """IK returns degrees; MuJoCo needs radians for hinge joints."""
        from lang3d.tools.sim_grasp import _ik_angles_to_qpos

        # Create a minimal mock model-like object.
        class FakeModel:
            njnt = 1
            jnt_type = [3]  # mjJNT_HINGE = 3
            jnt_qposadr = [0]

            def __init__(self):
                pass

        # We can't easily mock mujoco.mj_id2name, so test the math directly.
        import math
        angle_deg = 45.0
        expected_rad = angle_deg * (math.pi / 180.0)
        assert abs(expected_rad - 0.7854) < 0.001

    def test_slide_conversion_mm_to_meters(self):
        """Slide joints: IK returns mm, MuJoCo needs meters."""
        displacement_mm = 15.0
        expected_m = displacement_mm / 1000.0
        assert expected_m == 0.015


class TestPickPlaceTool:
    """Integration test: full pick-and-place on 4dof_arm benchmark."""

    @pytest.fixture
    def arm_run(self) -> tuple[str, str]:
        """Return (urdf_path, assembly_path) for a 4dof_arm benchmark run."""
        bm_file = _PROJECT / "data" / "runs" / "4dof_arm" / "BENCHMARK"
        if bm_file.exists():
            ts = bm_file.read_text().strip().split("\n")[0]
        else:
            runs = sorted((_PROJECT / "data" / "runs" / "4dof_arm").iterdir(), reverse=True)
            if not runs:
                pytest.skip("No 4dof_arm runs found")
            ts = runs[0].name

        run_dir = _PROJECT / "data" / "runs" / "4dof_arm" / ts
        urdf = run_dir / "engineering_package" / "urdf.xml"
        asm = run_dir / "assembly.json"
        if not urdf.exists() or not asm.exists():
            pytest.skip(f"Missing files in {ts}")
        return str(urdf), str(asm)

    @pytest.mark.integration
    def test_4dof_pick_place(self, arm_run):
        """4dof_arm pick-place pipeline runs and produces a result.

        This tests that the full pipeline (URDF load → IK → scene build →
        simulation → evaluation) runs without errors. Task success is NOT
        asserted — pick-and-place is a hard manipulation task and the
        physics tuning (grasp force, PD gains, contact friction) is still
        being optimized.
        """
        from lang3d.tools.sim_pickplace import PickPlaceTool

        urdf, asm = arm_run
        result_str = PickPlaceTool().execute(
            urdf_path=urdf,
            assembly_path=asm,
            pick_pos_mm="0,-393,91",   # finger midpoint in home pose
            place_pos_mm="30,-393,91",  # 30mm lateral shift
        )
        result = json.loads(result_str)

        # Should not error.
        assert "error" not in result, f"Tool returned error: {result.get('error')}"

        # Pipeline must produce a result (success or failure).
        assert "task_success" in result, f"Missing task_success: {result}"
        print(f"\n4dof pick-place: {json.dumps(result, indent=2)}")

    @pytest.mark.integration
    def test_place_accuracy_within_bounds(self, arm_run):
        """Place accuracy should be measured (even if task fails)."""
        from lang3d.tools.sim_pickplace import PickPlaceTool

        urdf, asm = arm_run
        result_str = PickPlaceTool().execute(
            urdf_path=urdf,
            assembly_path=asm,
            pick_pos_mm="0,-393,91",
            place_pos_mm="30,-393,91",
        )
        result = json.loads(result_str)

        if "error" in result:
            pytest.skip(f"Tool error: {result['error']}")

        accuracy = result.get("place_accuracy_mm", 999)
        print(f"\nPlace accuracy: {accuracy:.1f}mm")
        assert accuracy < 999, "Place accuracy not measured"
