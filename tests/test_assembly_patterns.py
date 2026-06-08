"""Tests for the assembly pattern knowledge base (Task 70).

Covers:
1. Robot profiles data integrity
2. Connection pattern matching
3. Interface feature rules
4. Statistical distributions
5. Helper functions
6. Integration with assembly_generator realistic example
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# 1. Robot profiles
# ---------------------------------------------------------------------------

class TestRobotProfiles:

    def test_all_profiles_loaded(self):
        from lang3d.knowledge.assembly_patterns import ROBOT_PROFILES
        assert len(ROBOT_PROFILES) >= 5
        expected = {"bcn3d_moveo", "thor", "parol6", "leo_rover", "anymal_b"}
        assert set(ROBOT_PROFILES.keys()) == expected

    def test_bcn3d_moveo_profile(self):
        from lang3d.knowledge.assembly_patterns import ROBOT_PROFILES
        p = ROBOT_PROFILES["bcn3d_moveo"]
        assert p.dof == 5
        assert p.total_parts == 37
        assert p.structural_parts == 22
        assert p.functional_parts == 7
        assert "bolted" in p.connection_methods
        assert "press_fit" in p.connection_methods
        assert "NEMA17-42BYGH" in p.actuators_used

    def test_thor_profile(self):
        from lang3d.knowledge.assembly_patterns import ROBOT_PROFILES
        p = ROBOT_PROFILES["thor"]
        assert p.dof == 6
        assert p.total_parts == 55
        assert p.structural_parts == 37
        assert "NEMA17" in p.actuators_used[0]
        assert p.key_dimensions["reach_mm"] == 280
        assert p.key_dimensions["height_stretched_mm"] == 625
        assert p.key_dimensions["payload_kg"] == 0.75

    def test_leo_rover_profile(self):
        from lang3d.knowledge.assembly_patterns import ROBOT_PROFILES
        p = ROBOT_PROFILES["leo_rover"]
        assert p.dof == 2
        assert p.structural_parts == 18
        assert "bolted" in p.connection_methods
        assert p.key_dimensions["wheel_diameter_mm"] == 125
        assert p.key_dimensions["track_width_mm"] == 354
        assert p.key_dimensions["ground_clearance_mm"] == 108
        assert "Buehler" in p.actuators_used[0]

    def test_anymal_b_profile(self):
        from lang3d.knowledge.assembly_patterns import ROBOT_PROFILES
        p = ROBOT_PROFILES["anymal_b"]
        assert p.dof == 12
        assert p.total_parts == 45
        assert "bolted" in p.connection_methods
        assert p.key_dimensions["weight_kg"] == 30.0

    def test_all_profiles_have_urls(self):
        from lang3d.knowledge.assembly_patterns import ROBOT_PROFILES
        for name, p in ROBOT_PROFILES.items():
            assert p.project_url, f"{name} missing project_url"
            assert "http" in p.project_url

    def test_all_profiles_have_notes(self):
        from lang3d.knowledge.assembly_patterns import ROBOT_PROFILES
        for name, p in ROBOT_PROFILES.items():
            assert p.notes, f"{name} missing notes"

    def test_parts_add_up(self):
        """structural + functional + fastener should be ≤ total_parts."""
        from lang3d.knowledge.assembly_patterns import ROBOT_PROFILES
        for name, p in ROBOT_PROFILES.items():
            classified = p.structural_parts + p.functional_parts + p.fastener_parts
            assert classified <= p.total_parts, \
                f"{name}: classified={classified} > total={p.total_parts}"


# ---------------------------------------------------------------------------
# 2. Connection patterns
# ---------------------------------------------------------------------------

class TestConnectionPatterns:

    def test_patterns_exist(self):
        from lang3d.knowledge.assembly_patterns import CONNECTION_PATTERNS
        assert len(CONNECTION_PATTERNS) >= 8

    def test_all_patterns_have_sources(self):
        from lang3d.knowledge.assembly_patterns import CONNECTION_PATTERNS
        for p in CONNECTION_PATTERNS:
            assert p.source_projects, f"Pattern '{p.name}' missing source_projects"

    def test_nema17_to_bracket_pattern(self):
        from lang3d.knowledge.assembly_patterns import get_connection_pattern
        p = get_connection_pattern("bracket", "stepper_motor")
        assert p is not None
        assert p.connection_method == "bolted"
        assert p.typical_bolt_size == "M3"
        assert p.typical_bolt_count == 4

    def test_bearing_press_fit_pattern(self):
        from lang3d.knowledge.assembly_patterns import get_connection_pattern
        p = get_connection_pattern("housing", "bearing")
        assert p is not None
        assert p.connection_method == "press_fit"

    def test_wheel_press_fit_pattern(self):
        from lang3d.knowledge.assembly_patterns import get_connection_pattern
        p = get_connection_pattern("wheel_hub", "dc_motor")
        assert p is not None
        assert p.connection_method == "press_fit"

    def test_plate_to_standoff_pattern(self):
        from lang3d.knowledge.assembly_patterns import get_connection_pattern
        p = get_connection_pattern("plate", "plate")
        assert p is not None
        assert p.typical_bolt_size == "M3"

    def test_pattern_by_method(self):
        from lang3d.knowledge.assembly_patterns import get_connection_pattern
        p = get_connection_pattern("bracket", "stepper_motor", method="bolted")
        assert p is not None
        assert p.connection_method == "bolted"

    def test_no_match_returns_none(self):
        from lang3d.knowledge.assembly_patterns import get_connection_pattern
        p = get_connection_pattern("nonexistent", "also_nonexistent")
        assert p is None


# ---------------------------------------------------------------------------
# 3. Interface feature rules
# ---------------------------------------------------------------------------

class TestInterfaceRules:

    def test_rules_exist(self):
        from lang3d.knowledge.assembly_patterns import INTERFACE_RULES
        assert len(INTERFACE_RULES) >= 7

    def test_nema17_mount_rule(self):
        from lang3d.knowledge.assembly_patterns import get_interface_rules
        rule = get_interface_rules("bracket")
        assert rule is not None
        assert "features" in rule
        # Should have through holes and clearance hole
        features = rule["features"]
        assert any(f["type"] == "through_hole" for f in features)
        assert any(f["type"] == "clearance_hole" for f in features)

    def test_bearing_seat_rule(self):
        from lang3d.knowledge.assembly_patterns import get_interface_rules
        rule = get_interface_rules("housing")
        assert rule is not None
        features = rule["features"]
        assert any(f["type"] == "bore" for f in features)

    def test_wheel_hub_rule(self):
        from lang3d.knowledge.assembly_patterns import get_interface_rules
        rule = get_interface_rules("wheel_hub")
        assert rule is not None

    def test_plate_rule(self):
        from lang3d.knowledge.assembly_patterns import get_interface_rules
        rule = get_interface_rules("plate")
        assert rule is not None


# ---------------------------------------------------------------------------
# 4. Statistical distributions
# ---------------------------------------------------------------------------

class TestAssemblyStatistics:

    def test_statistics_exist(self):
        from lang3d.knowledge.assembly_patterns import ASSEMBLY_STATISTICS
        assert "connection_method_distribution" in ASSEMBLY_STATISTICS
        assert "bolt_size_distribution" in ASSEMBLY_STATISTICS
        assert "part_class_distribution" in ASSEMBLY_STATISTICS

    def test_connection_methods_sum_to_one(self):
        from lang3d.knowledge.assembly_patterns import ASSEMBLY_STATISTICS
        total = sum(ASSEMBLY_STATISTICS["connection_method_distribution"].values())
        assert abs(total - 1.0) < 0.01, f"Connection methods sum to {total}"

    def test_bolt_sizes_sum_to_one(self):
        from lang3d.knowledge.assembly_patterns import ASSEMBLY_STATISTICS
        total = sum(ASSEMBLY_STATISTICS["bolt_size_distribution"].values())
        assert abs(total - 1.0) < 0.01, f"Bolt sizes sum to {total}"

    def test_part_classes_sum_to_one(self):
        from lang3d.knowledge.assembly_patterns import ASSEMBLY_STATISTICS
        total = sum(ASSEMBLY_STATISTICS["part_class_distribution"].values())
        assert abs(total - 1.0) < 0.01, f"Part classes sum to {total}"

    def test_bolted_is_dominant(self):
        from lang3d.knowledge.assembly_patterns import ASSEMBLY_STATISTICS
        dist = ASSEMBLY_STATISTICS["connection_method_distribution"]
        assert dist["bolted"] > 0.5  # Bolted should be >50%

    def test_m3_is_dominant_bolt(self):
        from lang3d.knowledge.assembly_patterns import ASSEMBLY_STATISTICS
        dist = ASSEMBLY_STATISTICS["bolt_size_distribution"]
        assert dist["M3"] > 0.4  # M3 should be >40%

    def test_structural_is_majority(self):
        from lang3d.knowledge.assembly_patterns import ASSEMBLY_STATISTICS
        dist = ASSEMBLY_STATISTICS["part_class_distribution"]
        assert dist["structural"] > 0.4  # Structural should be >40%


# ---------------------------------------------------------------------------
# 5. Helper functions
# ---------------------------------------------------------------------------

class TestHelperFunctions:

    def test_get_robot_profile(self):
        from lang3d.knowledge.assembly_patterns import get_robot_profile
        p = get_robot_profile("bcn3d_moveo")
        assert p is not None
        assert p.name == "BCN3D MOVEO"

    def test_get_robot_profile_not_found(self):
        from lang3d.knowledge.assembly_patterns import get_robot_profile
        assert get_robot_profile("nonexistent") is None

    def test_list_profiles(self):
        from lang3d.knowledge.assembly_patterns import list_profiles
        profiles = list_profiles()
        assert len(profiles) >= 4
        assert "bcn3d_moveo" in profiles

    def test_get_recommended_bolt_nema17(self):
        from lang3d.knowledge.assembly_patterns import get_recommended_bolt_size
        size, count = get_recommended_bolt_size("bracket", "stepper_motor")
        assert size == "M3"
        assert count == 4

    def test_get_recommended_bolt_bearing(self):
        from lang3d.knowledge.assembly_patterns import get_recommended_bolt_size
        size, count = get_recommended_bolt_size("housing", "bearing")
        # Press fit, no bolts
        assert size == "M3"  # Default fallback

    def test_get_recommended_bolt_unknown(self):
        from lang3d.knowledge.assembly_patterns import get_recommended_bolt_size
        size, count = get_recommended_bolt_size("unknown", "thing")
        assert size == "M3"  # Default
        assert count == 2    # Default

    def test_generate_stats_summary(self):
        from lang3d.knowledge.assembly_patterns import generate_assembly_stats_summary
        summary = generate_assembly_stats_summary()
        assert "Connection Method Distribution" in summary
        assert "Bolt Size Distribution" in summary
        assert "Robot Profiles" in summary
        assert "BCN3D MOVEO" in summary
        assert "Thor" in summary


# ---------------------------------------------------------------------------
# 6. Integration: realistic example in assembly_generator
# ---------------------------------------------------------------------------

class TestRealisticExample:

    def test_realistic_example_is_valid_json(self):
        import json
        from lang3d.tools.assembly_generator import EXAMPLE_5DOF_ARM_REALISTIC
        data = json.loads(EXAMPLE_5DOF_ARM_REALISTIC)
        assert data["name"] == "5dof_printed_arm"

    def test_realistic_example_joint_count(self):
        import json
        from lang3d.tools.assembly_generator import EXAMPLE_5DOF_ARM_REALISTIC
        data = json.loads(EXAMPLE_5DOF_ARM_REALISTIC)
        assert len(data["joints"]) == len(data["parts"]) - 1

    def test_realistic_example_has_connection_methods(self):
        import json
        from lang3d.tools.assembly_generator import EXAMPLE_5DOF_ARM_REALISTIC
        data = json.loads(EXAMPLE_5DOF_ARM_REALISTIC)
        bolted_joints = [j for j in data["joints"] if j.get("connection_method") == "bolted"]
        press_fit_joints = [j for j in data["joints"] if j.get("connection_method") == "press_fit"]
        assert len(bolted_joints) >= 4   # Motor mountings
        assert len(press_fit_joints) >= 2  # Bearings

    def test_realistic_example_has_nema17_dimensions(self):
        import json
        from lang3d.tools.assembly_generator import EXAMPLE_5DOF_ARM_REALISTIC
        data = json.loads(EXAMPLE_5DOF_ARM_REALISTIC)
        motors = [p for p in data["parts"] if "NEMA17" in p["description"]]
        assert len(motors) >= 4  # 5 DOF arm should have 5 motors
        for m in motors:
            dims = m["dimensions"]
            assert abs(dims["length"] - 42.3) < 0.5
            assert abs(dims["width"] - 42.3) < 0.5

    def test_realistic_example_has_608_bearings(self):
        import json
        from lang3d.tools.assembly_generator import EXAMPLE_5DOF_ARM_REALISTIC
        data = json.loads(EXAMPLE_5DOF_ARM_REALISTIC)
        bearings = [p for p in data["parts"] if p["category"] == "bearing"]
        assert len(bearings) >= 2
        for b in bearings:
            assert abs(b["dimensions"]["diameter"] - 22) < 1

    def test_realistic_example_materials_match_moveo(self):
        """Structural parts should be PLA (like BCN3D MOVEO)."""
        import json
        from lang3d.tools.assembly_generator import EXAMPLE_5DOF_ARM_REALISTIC
        data = json.loads(EXAMPLE_5DOF_ARM_REALISTIC)
        structural = [p for p in data["parts"] if p["category"] == "structural"]
        pla_parts = [p for p in structural if p["material"] == "PLA"]
        assert len(pla_parts) >= 5  # Most structural parts should be PLA
