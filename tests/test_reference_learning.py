"""Tests for open-source robot reference learning (Task 81).

Covers:
- Thor profile accuracy (updated from MG996R to NEMA17 + GT2 belt)
- PAROL6 profile data integrity
- New connection patterns (belt drive, gear mesh, slotted mount, joint housing)
- New interface rules (gt2_belt_drive_housing, mr105_bearing_seat, etc.)
- Updated assembly statistics
- PAROL6 few-shot example validity
- Statistics helper function
"""

from __future__ import annotations

import json
import pytest

from lang3d.knowledge.assembly_patterns import (
    CONNECTION_PATTERNS,
    INTERFACE_RULES,
    ASSEMBLY_STATISTICS,
    ROBOT_PROFILES,
    ConnectionPattern,
    RobotAssemblyProfile,
    generate_assembly_stats_summary,
    get_connection_pattern,
    get_recommended_bolt_size,
    get_robot_profile,
    list_profiles,
)


# ============================================================================
# 1. Thor profile (updated)
# ============================================================================

class TestThorProfile:

    def test_thor_profile_exists(self):
        profile = get_robot_profile("thor")
        assert profile is not None
        assert profile.name == "Thor"

    def test_thor_dof(self):
        profile = get_robot_profile("thor")
        assert profile.dof == 6

    def test_thor_uses_steppers_not_servos(self):
        """Updated Thor uses NEMA17 steppers, not MG996R servos."""
        profile = get_robot_profile("thor")
        assert "NEMA17" in profile.actuators_used[0]
        assert "MG996R" not in str(profile.actuators_used)

    def test_thor_37_printable_parts(self):
        """Thor has 37 unique printable structural parts."""
        profile = get_robot_profile("thor")
        assert profile.structural_parts == 37

    def test_thor_belt_drive(self):
        """Thor uses GT2 belt drive transmission."""
        profile = get_robot_profile("thor")
        assert "belt_drive" in profile.connection_methods
        assert profile.connection_methods["belt_drive"] > 0

    def test_thor_gear_mesh(self):
        """Thor uses 3D-printed gear pairs."""
        profile = get_robot_profile("thor")
        assert "gear_mesh" in profile.connection_methods

    def test_thor_key_dimensions(self):
        profile = get_robot_profile("thor")
        assert profile.key_dimensions["height_stretched_mm"] == 625
        assert profile.key_dimensions["payload_kg"] == 0.75

    def test_thor_freecad_source(self):
        """Thor CAD source is FreeCAD native."""
        profile = get_robot_profile("thor")
        assert "FreeCAD" in profile.notes

    def test_thor_total_parts(self):
        """37 structural + 6 functional + 12 fasteners = 55."""
        profile = get_robot_profile("thor")
        assert profile.total_parts == 55


# ============================================================================
# 2. PAROL6 profile
# ============================================================================

class TestPAROL6Profile:

    def test_parol6_profile_exists(self):
        profile = get_robot_profile("parol6")
        assert profile is not None
        assert profile.name == "PAROL6"

    def test_parol6_dof(self):
        profile = get_robot_profile("parol6")
        assert profile.dof == 6

    def test_parol6_uses_steppers(self):
        profile = get_robot_profile("parol6")
        assert "NEMA17" in profile.actuators_used[0]

    def test_parol6_belt_drive(self):
        profile = get_robot_profile("parol6")
        assert "belt_drive" in profile.connection_methods
        assert profile.connection_methods["belt_drive"] == 6

    def test_parol6_reach(self):
        profile = get_robot_profile("parol6")
        assert profile.key_dimensions["reach_mm"] == 400

    def test_parol6_structural_parts(self):
        profile = get_robot_profile("parol6")
        assert profile.structural_parts == 30

    def test_parol6_project_url(self):
        profile = get_robot_profile("parol6")
        assert "PAROL6-Desktop-robot-arm" in profile.project_url

    def test_parol6_materials_include_petg(self):
        """PAROL6 uses PETG for high-stress parts."""
        profile = get_robot_profile("parol6")
        assert "PETG" in profile.materials

    def test_parol6_notes_mentions_trinamic(self):
        """PAROL6 uses Trinamic drivers."""
        profile = get_robot_profile("parol6")
        assert "Trinamic" in profile.notes


# ============================================================================
# 3. New connection patterns
# ============================================================================

class TestNewConnectionPatterns:

    def test_belt_drive_pattern_exists(self):
        pattern = get_connection_pattern("housing", "stepper_motor", "bolted")
        # There should be a belt_drive pattern
        belt_patterns = [p for p in CONNECTION_PATTERNS
                         if "belt" in p.name]
        assert len(belt_patterns) >= 2  # belt_drive_joint + belt_tensioner

    def test_belt_drive_joint_pattern(self):
        pattern = None
        for p in CONNECTION_PATTERNS:
            if p.name == "belt_drive_joint_assembly":
                pattern = p
                break
        assert pattern is not None
        assert "thor" in pattern.source_projects
        assert "parol6" in pattern.source_projects
        assert pattern.typical_bolt_size == "M3"
        assert pattern.typical_bolt_count == 4

    def test_gear_transmission_pattern(self):
        pattern = None
        for p in CONNECTION_PATTERNS:
            if p.name == "gear_transmission_3d_printed":
                pattern = p
                break
        assert pattern is not None
        assert "thor" in pattern.source_projects
        assert pattern.connection_method == "gear_mesh"

    def test_belt_tensioner_pattern(self):
        pattern = None
        for p in CONNECTION_PATTERNS:
            if p.name == "belt_tensioner_slotted_mount":
                pattern = p
                break
        assert pattern is not None
        assert "parol6" in pattern.source_projects

    def test_joint_housing_bearing_seats_pattern(self):
        pattern = None
        for p in CONNECTION_PATTERNS:
            if p.name == "joint_housing_bearing_seats":
                pattern = p
                break
        assert pattern is not None
        assert pattern.connection_method == "bolted"
        assert len(pattern.source_projects) >= 2

    def test_all_patterns_have_names(self):
        for p in CONNECTION_PATTERNS:
            assert p.name, f"ConnectionPattern missing name"
            assert p.connection_method, f"Pattern {p.name} missing method"

    def test_total_pattern_count(self):
        """Should have at least 12 patterns (8 original + 4 new)."""
        assert len(CONNECTION_PATTERNS) >= 12


# ============================================================================
# 4. New interface rules
# ============================================================================

class TestNewInterfaceRules:

    def test_gt2_belt_drive_housing_rule(self):
        assert "gt2_belt_drive_housing" in INTERFACE_RULES
        rule = INTERFACE_RULES["gt2_belt_drive_housing"]
        assert rule["part_type"] == "housing"
        assert rule["constraint"] == "bolted"

    def test_mr105_bearing_seat_rule(self):
        assert "mr105_bearing_seat" in INTERFACE_RULES
        rule = INTERFACE_RULES["mr105_bearing_seat"]
        assert rule["constraint"] == "press_fit"

    def test_joint_housing_split_rule(self):
        assert "joint_housing_split" in INTERFACE_RULES
        rule = INTERFACE_RULES["joint_housing_split"]
        assert rule["part_type"] == "housing"

    def test_gt2_pulley_mount_rule(self):
        assert "gt2_pulley_mount" in INTERFACE_RULES
        rule = INTERFACE_RULES["gt2_pulley_mount"]
        assert rule["constraint"] == "set_screw"

    def test_gt2_housing_has_slots(self):
        """Belt drive housing should have slotted holes for tensioning."""
        rule = INTERFACE_RULES["gt2_belt_drive_housing"]
        features = rule["features"]
        slot_features = [f for f in features if f["type"] == "slot"]
        assert len(slot_features) > 0

    def test_mr105_bore_diameter(self):
        """MR105 OD=10mm, bore should be slightly smaller for press fit."""
        rule = INTERFACE_RULES["mr105_bearing_seat"]
        features = rule["features"]
        bore = [f for f in features if f["type"] == "bore"][0]
        assert bore["diameter"] < 10.0  # interference fit


# ============================================================================
# 5. Updated assembly statistics
# ============================================================================

class TestUpdatedStatistics:

    def test_belt_drive_in_statistics(self):
        stats = ASSEMBLY_STATISTICS["connection_method_distribution"]
        assert "belt_drive" in stats
        assert stats["belt_drive"] > 0

    def test_gear_mesh_in_statistics(self):
        stats = ASSEMBLY_STATISTICS["connection_method_distribution"]
        assert "gear_mesh" in stats

    def test_transmission_distribution(self):
        """New transmission_distribution section exists."""
        assert "transmission_distribution" in ASSEMBLY_STATISTICS
        td = ASSEMBLY_STATISTICS["transmission_distribution"]
        assert "timing_belt" in td
        assert td["timing_belt"] >= 0.4  # Most common

    def test_belt_drive_parameters(self):
        """New belt_drive_parameters section exists."""
        assert "belt_drive_parameters" in ASSEMBLY_STATISTICS
        bp = ASSEMBLY_STATISTICS["belt_drive_parameters"]
        assert bp["belt_pitch_mm"] == 2.0  # GT2
        assert "reduction_ratios" in bp

    def test_m3_bolt_dominance(self):
        """M3 should be even more dominant after Thor/PAROL6 analysis."""
        stats = ASSEMBLY_STATISTICS["bolt_size_distribution"]
        assert stats["M3"] >= 0.60

    def test_structural_part_ratio(self):
        """Structural parts ratio should be >= 0.55."""
        stats = ASSEMBLY_STATISTICS["part_class_distribution"]
        assert stats["structural"] >= 0.55

    def test_new_bolt_count_entries(self):
        """New bolt count entries for belt-drive robots."""
        counts = ASSEMBLY_STATISTICS["typical_bolt_counts"]
        assert "housing_split_bolts" in counts
        assert "motor_to_housing_belt" in counts

    def test_petg_in_material_distribution(self):
        stats = ASSEMBLY_STATISTICS["material_distribution_3d_printed"]
        assert "PETG" in stats

    def test_brass_in_material_distribution(self):
        stats = ASSEMBLY_STATISTICS["material_distribution_3d_printed"]
        assert "brass" in stats

    def test_distributions_sum_to_approx_1(self):
        """Distribution percentages should approximately sum to 1.0."""
        cmd = ASSEMBLY_STATISTICS["connection_method_distribution"]
        total = sum(cmd.values())
        assert abs(total - 1.0) < 0.05  # Allow small rounding

        bsd = ASSEMBLY_STATISTICS["bolt_size_distribution"]
        total = sum(bsd.values())
        assert abs(total - 1.0) < 0.05


# ============================================================================
# 6. PAROL6 few-shot example
# ============================================================================

class TestPAROL6FewShotExample:

    def test_example_is_valid_json(self):
        from lang3d.tools.assembly_generator import EXAMPLE_6DOF_BELT_DRIVE_ARM
        data = json.loads(EXAMPLE_6DOF_BELT_DRIVE_ARM)
        assert "name" in data
        assert "parts" in data
        assert "joints" in data

    def test_example_has_20_parts(self):
        from lang3d.tools.assembly_generator import EXAMPLE_6DOF_BELT_DRIVE_ARM
        data = json.loads(EXAMPLE_6DOF_BELT_DRIVE_ARM)
        assert len(data["parts"]) == 20

    def test_example_joint_count(self):
        """20 parts → 19 joints for connected tree."""
        from lang3d.tools.assembly_generator import EXAMPLE_6DOF_BELT_DRIVE_ARM
        data = json.loads(EXAMPLE_6DOF_BELT_DRIVE_ARM)
        assert len(data["joints"]) == 19

    def test_example_6_motors(self):
        from lang3d.tools.assembly_generator import EXAMPLE_6DOF_BELT_DRIVE_ARM
        data = json.loads(EXAMPLE_6DOF_BELT_DRIVE_ARM)
        motors = [p for p in data["parts"] if p["category"] == "actuator"]
        assert len(motors) == 6

    def test_example_4_bearings(self):
        from lang3d.tools.assembly_generator import EXAMPLE_6DOF_BELT_DRIVE_ARM
        data = json.loads(EXAMPLE_6DOF_BELT_DRIVE_ARM)
        bearings = [p for p in data["parts"] if p["category"] == "bearing"]
        assert len(bearings) == 4

    def test_example_has_belt_drive_connection(self):
        """Motor-to-housing connections should use bolted with M3."""
        from lang3d.tools.assembly_generator import EXAMPLE_6DOF_BELT_DRIVE_ARM
        data = json.loads(EXAMPLE_6DOF_BELT_DRIVE_ARM)
        bolted_motor_joints = [
            j for j in data["joints"]
            if j.get("connection_method") == "bolted"
            and "motor" in j.get("child", "")
        ]
        assert len(bolted_motor_joints) >= 5

    def test_example_has_press_fit_bearings(self):
        from lang3d.tools.assembly_generator import EXAMPLE_6DOF_BELT_DRIVE_ARM
        data = json.loads(EXAMPLE_6DOF_BELT_DRIVE_ARM)
        press_fit = [j for j in data["joints"]
                     if j.get("connection_method") == "press_fit"]
        assert len(press_fit) >= 3

    def test_example_default_angles_not_all_zero(self):
        from lang3d.tools.assembly_generator import EXAMPLE_6DOF_BELT_DRIVE_ARM
        data = json.loads(EXAMPLE_6DOF_BELT_DRIVE_ARM)
        angles = data.get("default_angles", {})
        assert len(angles) > 0
        assert not all(v == 0 for v in angles.values())

    def test_example_nema17_dimensions(self):
        """All motor parts should have NEMA17-standard dimensions."""
        from lang3d.tools.assembly_generator import EXAMPLE_6DOF_BELT_DRIVE_ARM
        data = json.loads(EXAMPLE_6DOF_BELT_DRIVE_ARM)
        for part in data["parts"]:
            if part["category"] == "actuator":
                dims = part["dimensions"]
                assert dims["length"] == pytest.approx(42.3, abs=0.1)
                assert dims["width"] == pytest.approx(42.3, abs=0.1)

    def test_example_connected_tree(self):
        """All parts should be reachable from base via joints."""
        from lang3d.tools.assembly_generator import EXAMPLE_6DOF_BELT_DRIVE_ARM
        data = json.loads(EXAMPLE_6DOF_BELT_DRIVE_ARM)
        part_names = {p["name"] for p in data["parts"]}

        # Build adjacency
        children_map: dict[str, set[str]] = {}
        for j in data["joints"]:
            children_map.setdefault(j["parent"], set()).add(j["child"])

        # BFS from base
        visited = {"base"}
        queue = ["base"]
        while queue:
            current = queue.pop(0)
            for child in children_map.get(current, set()):
                if child not in visited:
                    visited.add(child)
                    queue.append(child)

        assert visited == part_names, f"Unreachable parts: {part_names - visited}"


# ============================================================================
# 7. Profile listing and helpers
# ============================================================================

class TestProfileHelpers:

    def test_list_profiles_includes_parol6(self):
        profiles = list_profiles()
        assert "parol6" in profiles

    def test_list_profiles_includes_thor(self):
        profiles = list_profiles()
        assert "thor" in profiles

    def test_total_profile_count(self):
        """Should have at least 5 profiles now (bcn3d, thor, parol6, leo, anymal)."""
        profiles = list_profiles()
        assert len(profiles) >= 5

    def test_stats_summary_includes_parol6(self):
        summary = generate_assembly_stats_summary()
        assert "PAROL6" in summary

    def test_stats_summary_includes_thor(self):
        summary = generate_assembly_stats_summary()
        assert "Thor" in summary

    def test_stats_summary_includes_belt_drive(self):
        summary = generate_assembly_stats_summary()
        # Should mention belt_drive in statistics
        assert "belt_drive" in summary

    def test_get_recommended_bolt_size_for_housing_motor(self):
        bolt_size, count = get_recommended_bolt_size("housing", "stepper_motor")
        assert bolt_size == "M3"
        assert count == 4


# ============================================================================
# 8. Cross-project pattern consistency
# ============================================================================

class TestCrossProjectConsistency:

    def test_thor_parol6_share_belt_drive_pattern(self):
        """Both Thor and PAROL6 should be in belt_drive source projects."""
        for p in CONNECTION_PATTERNS:
            if p.name == "belt_drive_joint_assembly":
                assert "thor" in p.source_projects
                assert "parol6" in p.source_projects
                return
        pytest.fail("belt_drive_joint_assembly pattern not found")

    def test_bcn3d_thor_parol6_share_nema17_pattern(self):
        """All three should use NEMA17 mounting pattern."""
        for p in CONNECTION_PATTERNS:
            if p.name == "nema17_to_bracket_bolted":
                assert "bcn3d_moveo" in p.source_projects
                assert "thor" in p.source_projects
                assert "parol6" in p.source_projects
                return
        pytest.fail("nema17_to_bracket_bolted pattern not found")

    def test_all_profiles_have_connection_methods(self):
        for name, profile in ROBOT_PROFILES.items():
            assert len(profile.connection_methods) > 0, \
                f"{name} has no connection methods"

    def test_all_profiles_have_actuators(self):
        for name, profile in ROBOT_PROFILES.items():
            assert len(profile.actuators_used) > 0, \
                f"{name} has no actuators"
