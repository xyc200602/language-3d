"""Tests for the MuJoCo simulation validation tool.

These tests validate the sim_mujoco tool that loads generated URDFs into
MuJoCo to verify the NL→CAD→URDF pipeline produces physically realisable
robots. Tests skip if mujoco is not installed.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

# Make src/ importable when running tests directly
SRC = Path(__file__).parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _mujoco_available() -> bool:
    try:
        import mujoco  # noqa: F401
        return True
    except ImportError:
        return False


# Path to the example engineering package produced by a previous run of
# export_engineering_package.  These tests use it as a real-world fixture
# rather than mocking the URDF.
#
# Path history:
#   - Pre-2026-06-18: data/e2e_results/4dof_arm_20260615_021554/...
#   - Post-2026-06-18 refactor: data/runs/<case>/<ts>/engineering_package/urdf.xml
#     (the legacy directory was archived to data/archive_pre_refactor/)
#
# We search both layouts so the test still works against either an old run
# (kept under archive_pre_refactor/) or a freshly produced one.
_URDF_CANDIDATES = [
    # New canonical layout (preferred)
    Path(__file__).parent.parent / "data" / "runs",
    # Legacy archive (still readable; populated by the 2026-06-18 refactor)
    Path(__file__).parent.parent / "data" / "archive_pre_refactor" / "e2e_results",
    # Legacy pre-archive location (for checkouts older than the refactor)
    Path(__file__).parent.parent / "data" / "e2e_results",
]


def _find_example_urdf() -> Path | None:
    """Locate a real 4dof_arm URDF under any of the candidate roots."""
    import glob
    for root in _URDF_CANDIDATES:
        if not root.exists():
            continue
        # Prefer urdf.xml inside engineering_package (canonical new layout)
        for p in root.rglob("urdf.xml"):
            if "4dof_arm" in str(p) or "4dof" in p.parent.parent.name:
                return p
        # Fall back to the old ros2_package layout
        for p in root.rglob("*.urdf"):
            if "4dof" in p.name.lower():
                return p
    return None


_EXAMPLE_URDF = _find_example_urdf()


# ----------------------------------------------------------------------------
# Pure-function tests (no mujoco required)
# ----------------------------------------------------------------------------


class TestRewriteMeshPaths:
    """Tests for the mesh path auto-fix logic."""

    def test_absolute_path_left_alone(self, tmp_path: Path) -> None:
        """Existing absolute paths should not be modified."""
        from lang3d.tools.sim_mujoco import _rewrite_mesh_paths

        stl = tmp_path / "cube.stl"
        stl.write_bytes(b"dummy")
        urdf = tmp_path / "robot.urdf"
        urdf.write_text(
            f'<robot><link name="a"><visual><geometry>'
            f'<mesh filename="{stl.as_posix()}" />'
            f'</geometry></visual></link></robot>'
        )
        new_text, warnings = _rewrite_mesh_paths(urdf.read_text(), urdf)
        assert stl.as_posix() in new_text
        # No warning because it was already absolute and exists
        assert not any("Mesh not found" in w for w in warnings)

    def test_relative_path_resolved_from_parent(self, tmp_path: Path) -> None:
        """``meshes/X.stl`` should resolve against URDF parent directories."""
        from lang3d.tools.sim_mujoco import _rewrite_mesh_paths

        pkg = tmp_path / "pkg"
        (pkg / "meshes").mkdir(parents=True)
        stl = pkg / "meshes" / "base.stl"
        stl.write_bytes(b"dummy")
        urdf_dir = pkg / "urdf"
        urdf_dir.mkdir()
        urdf = urdf_dir / "robot.urdf"
        urdf.write_text(
            '<robot><link name="a"><visual><geometry>'
            '<mesh filename="meshes/base.stl" />'
            '</geometry></visual></link></robot>'
        )
        new_text, warnings = _rewrite_mesh_paths(urdf.read_text(), urdf)
        # Path should now be absolute
        assert str(stl.resolve()) in new_text.replace("/", "\\").replace("\\", "/") \
            or stl.as_posix() in new_text
        # A "Rewrote" informational warning is added
        assert any("Rewrote" in w for w in warnings)

    def test_missing_mesh_produces_warning(self, tmp_path: Path) -> None:
        """Missing STL files should produce a 'Mesh not found' warning."""
        from lang3d.tools.sim_mujoco import _rewrite_mesh_paths

        urdf = tmp_path / "robot.urdf"
        urdf.write_text(
            '<robot><link name="a"><visual><geometry>'
            '<mesh filename="meshes/nonexistent.stl" />'
            '</geometry></visual></link></robot>'
        )
        new_text, warnings = _rewrite_mesh_paths(urdf.read_text(), urdf)
        assert any("Mesh not found" in w for w in warnings)
        # Original path preserved when not found
        assert "meshes/nonexistent.stl" in new_text

    def test_package_uri_skipped(self, tmp_path: Path) -> None:
        """``package://`` URIs should be flagged, not rewritten."""
        from lang3d.tools.sim_mujoco import _rewrite_mesh_paths

        urdf = tmp_path / "robot.urdf"
        urdf.write_text(
            '<robot><link name="a"><visual><geometry>'
            '<mesh filename="package://my_pkg/meshes/base.stl" />'
            '</geometry></visual></link></robot>'
        )
        _, warnings = _rewrite_mesh_paths(urdf.read_text(), urdf)
        assert any("package://" in w or "URI" in w for w in warnings)


# ----------------------------------------------------------------------------
# Tool registration test (no mujoco required)
# ----------------------------------------------------------------------------


def test_tool_registration() -> None:
    """sim_mujoco should be discoverable via the registry."""
    from lang3d.tools.base import ToolRegistry
    from lang3d.tools.sim_mujoco import SimMujocoTool, register_sim_tools

    registry = ToolRegistry()
    register_sim_tools(registry)
    tool = registry.get("sim_mujoco")
    assert tool is not None
    assert isinstance(tool, SimMujocoTool)
    assert tool.name == "sim_mujoco"

    definition = tool.get_definition()
    assert definition.name == "sim_mujoco"
    assert "urdf_path" in definition.parameters["properties"]


def test_tool_missing_urdf_returns_error() -> None:
    """Calling without urdf_path should return an error string."""
    from lang3d.tools.sim_mujoco import SimMujocoTool

    result = SimMujocoTool().execute(urdf_path="")
    assert "Error" in result or "required" in result.lower()


def test_tool_nonexistent_file_returns_error(tmp_path: Path) -> None:
    """Non-existent URDF file should produce a clear error message."""
    from lang3d.tools.sim_mujoco import SimMujocoTool

    bogus = tmp_path / "does_not_exist.urdf"
    result = SimMujocoTool().execute(urdf_path=str(bogus))
    assert "not found" in result.lower() or "FAIL" in result


def test_tool_invalid_urdf_reports_parse_error(tmp_path: Path) -> None:
    """A malformed URDF should produce a parse error, not a crash."""
    pytest.importorskip("mujoco")
    from lang3d.tools.sim_mujoco import SimMujocoTool

    bogus = tmp_path / "broken.urdf"
    bogus.write_text("<robot><link name='a'><badly-formed")
    result = SimMujocoTool().execute(urdf_path=str(bogus))
    assert "FAIL" in result or "parse" in result.lower() or "Error" in result


# ----------------------------------------------------------------------------
# Real-load tests — require mujoco and the example URDF
# ----------------------------------------------------------------------------

needs_mujoco_and_example = pytest.mark.skipif(
    not (_mujoco_available() and _EXAMPLE_URDF.exists()),
    reason="mujoco not installed or example URDF not available",
)


@needs_mujoco_and_example
class TestRealLoad:
    """Tests that load the real 4dof_arm example URDF into MuJoCo."""

    def test_load_existing_4dof_arm(self) -> None:
        """Loading the example URDF should succeed with auto mesh-fix."""
        from lang3d.tools.sim_mujoco import SimMujocoTool

        result = SimMujocoTool().execute(
            urdf_path=str(_EXAMPLE_URDF),
            mode="report",  # scan only, no physics
        )
        # Should report successful load
        assert "加载结果: 成功" in result
        # 11 URDF links → 7 MuJoCo bodies (4 fixed-link merges)
        assert "MuJoCo body 数: 7" in result
        assert "fixed 合并: 4" in result
        # 6 joints total
        assert "关节数: 6" in result

    def test_all_bodies_have_positive_mass(self) -> None:
        """Every non-world body should have positive mass."""
        from lang3d.tools.sim_mujoco import _load_model, _scan_bodies

        load = _load_model(str(_EXAMPLE_URDF))
        assert load["ok"], f"Load failed: {load['error']}"
        try:
            bodies = _scan_bodies(load["model"])
            for b in bodies:
                if b["id"] == 0:
                    continue  # world body
                assert b["mass_kg"] > 0, f"Body {b['name']} has zero mass"
                assert not b["mass_warning"], f"Body {b['name']}: {b['mass_warning']}"
        finally:
            import os
            for f in load["temp_files"]:
                try:
                    os.unlink(f)
                except OSError:
                    pass

    def test_joints_have_correct_types(self) -> None:
        """The 4-DOF arm should have 4 HINGE + 2 SLIDE joints."""
        from lang3d.tools.sim_mujoco import _load_model, _scan_joints

        load = _load_model(str(_EXAMPLE_URDF))
        assert load["ok"]
        try:
            joints = _scan_joints(load["model"])
            types = [j["type"] for j in joints]
            assert types.count("HINGE") == 4
            assert types.count("SLIDE") == 2
        finally:
            import os
            for f in load["temp_files"]:
                try:
                    os.unlink(f)
                except OSError:
                    pass

    def test_validate_mode_completes(self) -> None:
        """Full validate mode should run and return a verdict line."""
        from lang3d.tools.sim_mujoco import SimMujocoTool

        result = SimMujocoTool().execute(
            urdf_path=str(_EXAMPLE_URDF),
            mode="validate",
            duration_sec=0.2,  # short to keep test fast
        )
        # Should have a structured verdict
        assert "结构验证:" in result
        # The 4dof_arm URDF is structurally valid (loads, meshes resolve,
        # masses are positive, joints can move)
        assert "PASS" in result


@needs_mujoco_and_example
def test_mesh_auto_fix_writes_temp_urdf() -> None:
    """When mesh paths need rewriting, a temp URDF should be created and cleaned up."""
    from lang3d.tools.sim_mujoco import _load_model

    load = _load_model(str(_EXAMPLE_URDF))
    assert load["ok"]
    # The example URDF has meshes in urdf/ subdir which require rewriting
    # so temp_files should be non-empty.
    assert len(load["temp_files"]) >= 1
    temp_path = load["temp_files"][0]
    assert Path(temp_path).exists()
    # Clean up
    import os
    for f in load["temp_files"]:
        os.unlink(f)
    assert not Path(temp_path).exists()


# ----------------------------------------------------------------------------
# Grasp tool tests
# ----------------------------------------------------------------------------


def test_grasp_tool_registration() -> None:
    """sim_grasp should be discoverable via the registry."""
    from lang3d.tools.base import ToolRegistry
    from lang3d.tools.sim_mujoco import SimGraspTool, register_sim_tools

    registry = ToolRegistry()
    register_sim_tools(registry)
    tool = registry.get("sim_grasp")
    assert tool is not None
    assert isinstance(tool, SimGraspTool)
    assert tool.name == "sim_grasp"

    definition = tool.get_definition()
    assert "urdf_path" in definition.parameters["properties"]
    assert "cube_size_mm" in definition.parameters["properties"]


def test_grasp_tool_missing_urdf_returns_error() -> None:
    """Missing urdf_path should produce an error string."""
    from lang3d.tools.sim_mujoco import SimGraspTool

    result = SimGraspTool().execute(urdf_path="")
    assert "Error" in result or "required" in result.lower()


def test_grasp_tool_nonexistent_file_returns_error(tmp_path: Path) -> None:
    """Non-existent URDF should produce a clear error."""
    from lang3d.tools.sim_mujoco import SimGraspTool

    bogus = tmp_path / "nope.urdf"
    result = SimGraspTool().execute(urdf_path=str(bogus))
    assert "not found" in result.lower() or "Error" in result


def test_grasp_tool_no_gripper_reports_clearly(tmp_path: Path) -> None:
    """A URDF with no slide joints should report 'no gripper' clearly."""
    pytest.importorskip("mujoco")
    from lang3d.tools.sim_mujoco import SimGraspTool

    # URDF with only hinge joints (no slide = no gripper)
    urdf = tmp_path / "no_gripper.urdf"
    urdf.write_text("""<?xml version="1.0"?>
<robot name="no_gripper">
  <link name="base"><inertial><mass value="0.1"/><inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial></link>
  <link name="arm"><inertial><mass value="0.1"/><inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial></link>
  <joint name="j1" type="revolute">
    <parent link="base"/><child link="arm"/>
    <origin xyz="0 0 0.05" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.5" upper="1.5" effort="5" velocity="5"/>
  </joint>
</robot>""")
    result = SimGraspTool().execute(urdf_path=str(urdf))
    assert "NO GRIPPER" in result or "slide joints" in result.lower()


@needs_mujoco_and_example
class TestGraspRealLoad:
    """Grasp tests on the real 4dof_arm example URDF.

    These tests verify the TOOL runs correctly and produces a structured
    result.  The grasp itself may FAIL because the example gripper's
    geometry (long fingers with L-shaped tips) doesn't generate enough
    clamping friction on a cube — that's a real finding about the
    assembly, not a tool bug.
    """

    def test_grasp_runs_and_returns_structured_result(self) -> None:
        """Tool should run end-to-end and return JSON with grasp fields."""
        from lang3d.tools.sim_mujoco import SimGraspTool

        result = SimGraspTool().execute(
            urdf_path=str(_EXAMPLE_URDF),
            cube_size_mm=20.0,
            cube_mass_g=20.0,
            grasp_force_n=10.0,
            duration_sec=1.5,  # short to keep test fast
        )
        # Should contain the structured report markers
        assert "[sim_grasp]" in result
        assert "夹爪识别" in result or "NO GRIPPER" in result
        assert "--- JSON ---" in result

        # Should identify the 2 slide joints as the gripper
        if "夹爪识别" in result:
            assert "gripper_finger_left" in result
            assert "gripper_finger_right" in result

    def test_grasp_returns_verdict_field(self) -> None:
        """Result must include a clear PASS/FAIL verdict."""
        from lang3d.tools.sim_mujoco import SimGraspTool

        result = SimGraspTool().execute(
            urdf_path=str(_EXAMPLE_URDF),
            cube_size_mm=20.0,
            duration_sec=1.0,
        )
        assert "总体结论" in result
        assert "PASS" in result or "FAIL" in result

    def test_grasp_json_contains_required_fields(self) -> None:
        """JSON summary must include lift_c_m, geometry_ok, etc."""
        import json as _json
        from lang3d.tools.sim_mujoco import SimGraspTool

        result = SimGraspTool().execute(
            urdf_path=str(_EXAMPLE_URDF),
            cube_size_mm=20.0,
            duration_sec=1.0,
        )
        # Extract JSON block.  The report may contain trailing text after
        # the JSON object (the multi-gripper aggregate verdict), so parse
        # only up to the matching closing brace via the raw decoder.
        json_start = result.find("--- JSON ---")
        assert json_start >= 0
        after_marker = result[json_start + len("--- JSON ---"):].lstrip()
        _decoder = _json.JSONDecoder()
        summary, _ = _decoder.raw_decode(after_marker)

        for required_key in (
            "urdf_path", "finger_separation_mm", "cube_size_mm",
            "lifted", "geometry_ok", "held_against_gravity",
            "phase_a_contacts_max", "note",
        ):
            assert required_key in summary, f"Missing key: {required_key}"


# ---------------------------------------------------------------------------
# record_motion — physics rollout for web animation
# ---------------------------------------------------------------------------


def test_record_motion_returns_frame_series() -> None:
    """record_motion produces one frame per (duration * fps) timestep.

    Each frame carries per-body world pose (m + quaternion).  Uses the
    example 4dof arm so the joint/mesh structure is real.
    """
    from lang3d.tools.sim_mujoco import record_motion

    result = record_motion(_EXAMPLE_URDF, duration_sec=1.0, fps=20)
    assert result["ok"] is True
    assert isinstance(result["bodies"], list) and len(result["bodies"]) > 0
    # 1.0s @ 20fps -> ~20 frames (allow a couple of slack for sampling edges).
    assert len(result["frames"]) >= 15
    f0 = result["frames"][0]
    assert "t" in f0 and "poses" in f0
    # Each pose = [px, py, pz, qw, qx, qy, qz] (7 floats), parallel to bodies.
    assert len(f0["poses"]) == len(result["bodies"])
    assert len(f0["poses"][0]) == 7


def test_record_motion_missing_urdf_reports_error(tmp_path) -> None:
    """A non-existent URDF returns ok=False with an error message."""
    from lang3d.tools.sim_mujoco import record_motion

    result = record_motion(str(tmp_path / "nope.xml"), duration_sec=0.5)
    assert result["ok"] is False
    assert "error" in result
    assert result["frames"] == []


def test_wheeled_base_drives_in_record_motion() -> None:
    """Regression guard for the "能动" expectation: a wheeled robot's base
    must TRANSLATE during the rollout, not stay welded to the world.

    Before the floating-base + wheel-drive fix, the chassis was bolted to
    the ground (fixed base_footprint→base_plate joint) and only the arms
    articulated — the robot could never drive.  This test loads a real
    4-wheel dual-arm URDF and asserts the base_footprint body moves
    measurably between the first and last frame."""
    import glob
    from lang3d.tools.sim_mujoco import record_motion

    urdfs = sorted(glob.glob(
        "data/runs/4wheel_dual_arm/*/engineering_package/ros2_package/"
        "*/urdf/*.urdf",
    ))
    if not urdfs:
        pytest.skip("no 4wheel_dual_arm run URDF available")
    result = record_motion(urdfs[-1], duration_sec=2.0, fps=10)
    assert result["ok"] is True
    bodies = result["bodies"]
    if "base_footprint" not in bodies:
        pytest.skip("URDF has no base_footprint (arm-only?)")
    bp_idx = bodies.index("base_footprint")
    frames = result["frames"]
    assert len(frames) >= 5
    p0 = frames[0]["poses"][bp_idx][:3]
    pN = frames[-1]["poses"][bp_idx][:3]
    # Horizontal translation in metres (exclude z which the upright-stabilize
    # keeps near 0).  Must move >5mm — the pre-fix robot moved 0.0mm.
    horiz = ((pN[0] - p0[0]) ** 2 + (pN[1] - p0[1]) ** 2) ** 0.5
    assert horiz > 0.005, (
        f"base did not translate (horiz={horiz*1000:.1f}mm) — "
        f"wheeled base is not driving"
    )

