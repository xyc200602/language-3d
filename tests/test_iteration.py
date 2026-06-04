"""Tests for iterative design optimization tools.

Tests cover:
- Impact analysis for each change type
- Change application (payload, reach, material, actuator, controller)
- Incremental artifact regeneration
- Design comparison
- Full iteration workflow
- Tool registration and execution
"""

from __future__ import annotations

import pytest

from lang3d.knowledge.mechanics import ROBOTIC_ARM_ASSEMBLY
from lang3d.tools.base import ToolRegistry
from lang3d.tools.iteration import (
    ChangeDiff,
    DesignSnapshot,
    ImpactAnalysis,
    IterationTool,
    RequirementChange,
    analyze_impact,
    apply_change,
    compare_designs,
    create_snapshot,
    iterate_design,
    register_iteration_tools,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def snapshot():
    """Create a basic design snapshot."""
    return create_snapshot(
        ROBOTIC_ARM_ASSEMBLY,
        ["MG996R", "MG996R", "DS3218", "SG90"],
        ["AS5600", "LIMIT_SWITCH_MICRO", "MPU6050"],
        "esp32",
    )


@pytest.fixture
def full_snapshot():
    """Create a snapshot with all artifacts generated."""
    return create_snapshot(
        ROBOTIC_ARM_ASSEMBLY,
        ["MG996R", "MG996R", "DS3218", "SG90"],
        ["AS5600", "LIMIT_SWITCH_MICRO", "MPU6050"],
        "esp32",
        generate_all=True,
    )


# ============================================================================
# Test: Impact analysis
# ============================================================================

class TestImpactAnalysis:
    """Test impact analysis for different change types."""

    def test_payload_change_impact(self, snapshot):
        change = RequirementChange("payload", 200, 500)
        impact = analyze_impact(snapshot, change)
        assert impact.change.change_type == "payload"
        assert len(impact.affected_joints) > 0
        assert len(impact.affected_parts) > 0
        assert "firmware" in impact.affected_artifacts
        assert "bom" in impact.affected_artifacts

    def test_payload_change_severity_minor(self, snapshot):
        change = RequirementChange("payload", 200, 220)
        impact = analyze_impact(snapshot, change)
        assert impact.severity == "minor"

    def test_payload_change_severity_moderate(self, snapshot):
        change = RequirementChange("payload", 200, 260)
        impact = analyze_impact(snapshot, change)
        assert impact.severity == "moderate"

    def test_payload_change_severity_major(self, snapshot):
        change = RequirementChange("payload", 200, 600)
        impact = analyze_impact(snapshot, change)
        assert impact.severity == "major"

    def test_reach_change_impact(self, snapshot):
        change = RequirementChange("reach", 300, 400)
        impact = analyze_impact(snapshot, change)
        assert len(impact.affected_joints) > 0
        assert "firmware" in impact.affected_artifacts
        assert "bom" in impact.affected_artifacts
        assert "ik" in impact.affected_artifacts

    def test_reach_change_severity_major(self, snapshot):
        change = RequirementChange("reach", 100, 200)
        impact = analyze_impact(snapshot, change)
        assert impact.severity == "major"

    def test_material_change_impact(self, snapshot):
        change = RequirementChange("material", "PLA", "ABS")
        impact = analyze_impact(snapshot, change)
        assert len(impact.affected_parts) == len(ROBOTIC_ARM_ASSEMBLY.parts)
        assert impact.affected_joints == []
        assert "print" in impact.affected_artifacts
        assert "bom" in impact.affected_artifacts
        assert impact.severity == "minor"

    def test_actuator_change_impact(self, snapshot):
        change = RequirementChange("actuator", "MG996R", "DS3218")
        impact = analyze_impact(snapshot, change)
        assert "firmware" in impact.affected_artifacts
        assert "wiring" in impact.affected_artifacts

    def test_controller_change_impact(self, snapshot):
        change = RequirementChange("controller", "esp32", "arduino")
        impact = analyze_impact(snapshot, change)
        assert "firmware" in impact.affected_artifacts
        assert "wiring" in impact.affected_artifacts
        assert impact.severity == "minor"

    def test_unknown_change_type(self, snapshot):
        change = RequirementChange("unknown", "a", "b")
        impact = analyze_impact(snapshot, change)
        assert impact.severity == "unknown"
        assert not impact.auto_fixable

    def test_payload_notes_present(self, snapshot):
        change = RequirementChange("payload", 200, 300)
        impact = analyze_impact(snapshot, change)
        assert len(impact.notes) > 0
        assert "200" in impact.notes[0] or "300" in impact.notes[0]


# ============================================================================
# Test: Change application
# ============================================================================

class TestApplyPayloadChange:
    """Test payload change application."""

    def test_payload_updates_metadata(self, snapshot):
        change = RequirementChange("payload", 200, 500)
        new_snap, diff = apply_change(snapshot, change)
        assert new_snap.metadata["payload_g"] == 500

    def test_payload_reselects_actuators_if_needed(self, snapshot):
        # Large payload increase should trigger actuator re-selection
        change = RequirementChange("payload", 200, 2000)
        new_snap, diff = apply_change(snapshot, change)
        # Check that analysis happened
        assert len(diff.artifacts_regenerated) > 0

    def test_payload_generates_artifacts(self, snapshot):
        change = RequirementChange("payload", 200, 300)
        new_snap, diff = apply_change(snapshot, change)
        assert "bom" in diff.artifacts_regenerated
        assert "firmware" in diff.artifacts_regenerated

    def test_payload_diff_has_summary(self, snapshot):
        change = RequirementChange("payload", 200, 500)
        new_snap, diff = apply_change(snapshot, change)
        assert "200" in diff.summary or "500" in diff.summary


class TestApplyReachChange:
    """Test reach change application."""

    def test_reach_scales_dimensions(self, snapshot):
        change = RequirementChange("reach", 100, 150)
        new_snap, diff = apply_change(snapshot, change)
        assert len(diff.part_changes) > 0
        # Check that dimensions actually changed
        for pc in diff.part_changes:
            if "old_dimensions" in pc:
                assert pc["old_dimensions"] != pc["new_dimensions"]

    def test_reach_generates_all_artifacts(self, snapshot):
        change = RequirementChange("reach", 100, 120)
        new_snap, diff = apply_change(snapshot, change)
        assert "bom" in diff.artifacts_regenerated
        assert "firmware" in diff.artifacts_regenerated
        assert "ik" in diff.artifacts_regenerated

    def test_reach_scale_factor(self, snapshot):
        change = RequirementChange("reach", 100, 200)
        new_snap, diff = apply_change(snapshot, change)
        assert "scale=2.00" in diff.summary or "2.00x" in diff.summary


class TestApplyMaterialChange:
    """Test material change application."""

    def test_material_updates_all_parts(self, snapshot):
        change = RequirementChange("material", "PLA", "ABS")
        new_snap, diff = apply_change(snapshot, change)
        assert len(diff.part_changes) == len(ROBOTIC_ARM_ASSEMBLY.parts)
        for pc in diff.part_changes:
            assert pc["new_material"] == "ABS"

    def test_material_generates_print_and_bom(self, snapshot):
        change = RequirementChange("material", "PLA", "PETG")
        new_snap, diff = apply_change(snapshot, change)
        assert "print_config" in diff.artifacts_regenerated
        assert "bom" in diff.artifacts_regenerated

    def test_material_does_not_regenerate_firmware(self, snapshot):
        change = RequirementChange("material", "PLA", "ABS")
        new_snap, diff = apply_change(snapshot, change)
        assert "firmware" not in diff.artifacts_regenerated


class TestApplyActuatorChange:
    """Test actuator change application."""

    def test_actuator_replacement(self, snapshot):
        change = RequirementChange("actuator", "MG996R", "DS3218")
        new_snap, diff = apply_change(snapshot, change)
        assert "DS3218" in new_snap.actuator_ids
        assert len(diff.actuator_changes) > 0

    def test_actuator_generates_firmware_wiring(self, snapshot):
        change = RequirementChange("actuator", "MG996R", "DS3218")
        new_snap, diff = apply_change(snapshot, change)
        assert "firmware" in diff.artifacts_regenerated
        assert "wiring" in diff.artifacts_regenerated


class TestApplyControllerChange:
    """Test controller change application."""

    def test_controller_updated(self, snapshot):
        change = RequirementChange("controller", "esp32", "arduino")
        new_snap, diff = apply_change(snapshot, change)
        assert new_snap.controller == "arduino"

    def test_controller_generates_firmware_wiring(self, snapshot):
        change = RequirementChange("controller", "esp32", "arduino")
        new_snap, diff = apply_change(snapshot, change)
        assert "firmware" in diff.artifacts_regenerated
        assert "wiring" in diff.artifacts_regenerated


# ============================================================================
# Test: Incremental regeneration
# ============================================================================

class TestIncrementalRegeneration:
    """Test that only affected artifacts are regenerated."""

    def test_material_does_not_touch_firmware(self, snapshot):
        change = RequirementChange("material", "PLA", "ABS")
        new_snap, diff = apply_change(snapshot, change)
        assert "firmware" not in diff.artifacts_regenerated
        assert "wiring" not in diff.artifacts_regenerated

    def test_controller_does_not_touch_print(self, snapshot):
        change = RequirementChange("controller", "esp32", "arduino")
        new_snap, diff = apply_change(snapshot, change)
        assert "print" not in diff.artifacts_regenerated

    def test_payload_does_not_touch_print(self, snapshot):
        change = RequirementChange("payload", 200, 500)
        new_snap, diff = apply_change(snapshot, change)
        assert "print_config" not in diff.artifacts_regenerated

    def test_reach_touches_most_artifacts(self, snapshot):
        change = RequirementChange("reach", 100, 150)
        new_snap, diff = apply_change(snapshot, change)
        # Reach affects assembly, IK, firmware, print, BOM, quality
        assert len(diff.artifacts_regenerated) >= 5


# ============================================================================
# Test: Design comparison
# ============================================================================

class TestDesignComparison:
    """Test comparing two design snapshots."""

    def test_identical_snapshots_no_changes(self, snapshot):
        other = snapshot.copy()
        diff = compare_designs(snapshot, other)
        assert len(diff.part_changes) == 0
        assert len(diff.actuator_changes) == 0

    def test_detects_material_change(self, snapshot):
        other = snapshot.copy()
        other.assembly.parts[0].material = "ABS"
        diff = compare_designs(snapshot, other)
        assert len(diff.part_changes) > 0

    def test_detects_actuator_change(self, snapshot):
        other = snapshot.copy()
        other.actuator_ids = ["DS3218"] + snapshot.actuator_ids[1:]
        diff = compare_designs(snapshot, other)
        assert len(diff.actuator_changes) > 0

    def test_detects_cost_change(self, full_snapshot):
        other = full_snapshot.copy()
        if other.bom:
            other.bom["cost_summary"]["total_cost_cny"] += 50
        diff = compare_designs(full_snapshot, other)
        assert diff.cost_change_cny == 50.0


# ============================================================================
# Test: Full iteration
# ============================================================================

class TestFullIteration:
    """Test full iteration workflow."""

    def test_iterate_payload(self, snapshot):
        change = RequirementChange("payload", 200, 400)
        new_snap, impact, diff = iterate_design(snapshot, change)
        assert new_snap.metadata["payload_g"] == 400
        assert impact.severity in ("minor", "moderate", "major")
        assert len(diff.artifacts_regenerated) > 0

    def test_iterate_material(self, snapshot):
        change = RequirementChange("material", "PLA", "ABS")
        new_snap, impact, diff = iterate_design(snapshot, change)
        for part in new_snap.assembly.parts:
            assert part.material == "ABS"

    def test_iterate_preserves_original(self, snapshot):
        original_materials = [p.material for p in snapshot.assembly.parts]
        change = RequirementChange("material", "PLA", "ABS")
        iterate_design(snapshot, change)
        # Original should be unchanged
        for i, p in enumerate(snapshot.assembly.parts):
            assert p.material == original_materials[i]

    def test_multiple_iterations(self, snapshot):
        # Apply material change then actuator change
        change1 = RequirementChange("material", "PLA", "ABS")
        snap2, _, _ = iterate_design(snapshot, change1)
        assert snap2.assembly.parts[0].material == "ABS"

        change2 = RequirementChange("controller", "esp32", "arduino")
        snap3, _, diff3 = iterate_design(snap2, change2)
        assert snap3.controller == "arduino"
        assert snap3.assembly.parts[0].material == "ABS"  # Previous change preserved


# ============================================================================
# Test: DesignSnapshot
# ============================================================================

class TestDesignSnapshot:
    """Test DesignSnapshot creation and operations."""

    def test_create_basic_snapshot(self, snapshot):
        assert snapshot.assembly.name == ROBOTIC_ARM_ASSEMBLY.name
        assert len(snapshot.actuator_ids) == 4
        assert len(snapshot.sensor_ids) == 3
        assert snapshot.controller == "esp32"

    def test_create_full_snapshot(self, full_snapshot):
        assert full_snapshot.bom is not None
        assert full_snapshot.firmware is not None
        assert full_snapshot.wiring is not None
        assert full_snapshot.print_config is not None
        assert full_snapshot.quality_report is not None

    def test_full_snapshot_bom_has_cost(self, full_snapshot):
        assert full_snapshot.bom["cost_summary"]["total_cost_cny"] > 0

    def test_full_snapshot_firmware_has_files(self, full_snapshot):
        assert "robot_arm.ino" in full_snapshot.firmware

    def test_snapshot_deep_copy(self, snapshot):
        other = snapshot.copy()
        other.actuator_ids[0] = "CHANGED"
        assert snapshot.actuator_ids[0] != "CHANGED"

    def test_full_snapshot_has_placements(self, full_snapshot):
        assert "placements" in full_snapshot.metadata

    def test_full_snapshot_has_torque_analysis(self, full_snapshot):
        assert "torque_analysis" in full_snapshot.metadata

    def test_full_snapshot_has_power_budget(self, full_snapshot):
        assert "power_budget" in full_snapshot.metadata


# ============================================================================
# Test: Tool registration and execution
# ============================================================================

class TestIterationTool:
    """Test IterationTool."""

    def test_registered(self):
        registry = ToolRegistry()
        register_iteration_tools(registry)
        assert "iteration_design" in registry.list_tools()

    def test_execute_payload_change(self):
        tool = IterationTool()
        result = tool.execute(
            change_type="payload",
            old_value=200,
            new_value=500,
        )
        assert "迭代设计" in result
        assert "payload" in result.lower() or "负载" in result

    def test_execute_material_change(self):
        tool = IterationTool()
        result = tool.execute(
            change_type="material",
            old_value="PLA",
            new_value="ABS",
        )
        assert "ABS" in result
        assert "PLA" in result

    def test_execute_controller_change(self):
        tool = IterationTool()
        result = tool.execute(
            change_type="controller",
            old_value="esp32",
            new_value="arduino",
        )
        assert "arduino" in result.lower()

    def test_execute_missing_values(self):
        tool = IterationTool()
        result = tool.execute(change_type="payload")
        assert "错误" in result

    def test_execute_reach_change(self):
        tool = IterationTool()
        result = tool.execute(
            change_type="reach",
            old_value=100,
            new_value=200,
        )
        assert "200" in result

    def test_execute_actuator_change(self):
        tool = IterationTool()
        result = tool.execute(
            change_type="actuator",
            old_value="MG996R",
            new_value="DS3218",
        )
        assert "DS3218" in result

    def test_execute_shows_regenerated_artifacts(self):
        tool = IterationTool()
        result = tool.execute(
            change_type="material",
            old_value="PLA",
            new_value="ABS",
        )
        assert "已重新生成" in result
