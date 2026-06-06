"""Tests for part_feature_engine module."""

from __future__ import annotations

import pytest

from lang3d.knowledge.mechanics import Part
from lang3d.tools.export_package import _freecad_ops_for_part, build_complex_robot
from lang3d.tools.part_feature_engine import (
    FeatureConfig,
    _classify,
    generate_ops,
    infer_features,
)

# Valid op types that _build_script() supports.
_VALID_OPS = {
    "new_doc", "make_box", "make_cylinder", "make_sphere", "make_cone",
    "boolean", "cylinder_with_hole", "plate_with_holes",
    "move", "rotate",
    "fillet", "chamfer",
    "sweep", "loft",
    "polar_pattern", "linear_pattern", "mirror",
    "shell", "draft",
    "create_sketch", "extrude_sketch", "revolve_sketch", "pocket",
    "save", "export_stl", "export_step",
    "status", "object_info", "volume_check", "compute_mass",
    "delete_object", "raw_script",
}


# ============================================================================
# 1. TestAllPartsProduceOps — 41 parts all generate valid ops
# ============================================================================


class TestAllPartsProduceOps:
    """Every part in the 41-part robot must produce a valid op sequence."""

    @pytest.fixture()
    def robot_parts(self):
        return build_complex_robot().parts

    def test_all_41_parts_produce_ops(self, robot_parts):
        for part in robot_parts:
            ops = generate_ops(part)
            assert len(ops) >= 2, f"{part.name}: too few ops ({len(ops)})"
            assert ops[0]["type"] == "new_doc", f"{part.name}: must start with new_doc"
            assert ops[-1]["type"] == "export_stl", f"{part.name}: must end with export_stl"

    def test_all_op_types_valid(self, robot_parts):
        for part in robot_parts:
            ops = generate_ops(part)
            for op in ops:
                assert op["type"] in _VALID_OPS, (
                    f"{part.name}: invalid op type '{op['type']}'"
                )


# ============================================================================
# 2. TestFeatureInference — each family infers correct features
# ============================================================================


class TestFeatureInference:
    """Each part family should infer the expected features."""

    def test_plate_has_holes_and_chamfers(self):
        p = Part("base_plate", "structural", "test",
                 dimensions=dict(length=300, width=200, height=5))
        cfg = infer_features(p)
        assert len(cfg.mounting_holes) > 0
        assert cfg.mounting_holes[0]["pattern"] == "grid"
        assert len(cfg.chamfers) > 0

    def test_top_plate_has_smaller_holes(self):
        p = Part("top_plate", "structural", "test",
                 dimensions=dict(length=280, width=180, height=3))
        cfg = infer_features(p)
        # M3 holes (diameter 3.0) vs M4 holes (diameter 4.0) for base
        assert cfg.mounting_holes[0]["diameter_mm"] < 4.0

    def test_standoff_has_bore(self):
        p = Part("standoff_fl", "structural", "test",
                 dimensions=dict(length=8, diameter=6, height=50))
        cfg = infer_features(p)
        assert cfg.bore is not None
        assert cfg.bore["through"] is True

    def test_wheel_has_keyway(self):
        p = Part("wheel_fl", "structural", "test",
                 dimensions=dict(diameter=65, height=26))
        cfg = infer_features(p)
        assert cfg.bore is not None
        assert cfg.bore.get("keyway") is True
        assert cfg.bore["diameter_mm"] == 6.0

    def test_arm_base_has_bearing_seats_and_shell_and_fillets(self):
        p = Part("arm_l_base", "joint", "test",
                 dimensions=dict(outer_diameter=80, height=40))
        cfg = infer_features(p)
        assert len(cfg.bearing_seats) > 0
        assert cfg.shell is not None
        assert len(cfg.fillets) > 0

    def test_arm_shoulder_has_features(self):
        p = Part("arm_l_shoulder", "joint", "test",
                 dimensions=dict(outer_diameter=60, height=35))
        cfg = infer_features(p)
        assert len(cfg.bearing_seats) > 0
        assert cfg.shell is not None

    def test_arm_elbow_has_features(self):
        p = Part("arm_l_elbow", "joint", "test",
                 dimensions=dict(outer_diameter=50, height=30))
        cfg = infer_features(p)
        assert len(cfg.bearing_seats) > 0

    def test_arm_wrist_has_features(self):
        p = Part("arm_l_wrist", "joint", "test",
                 dimensions=dict(outer_diameter=40, height=25))
        cfg = infer_features(p)
        assert len(cfg.bearing_seats) > 0

    def test_link_has_cable_channel(self):
        p = Part("arm_l_upper_link", "structural", "test",
                 dimensions=dict(length=150, width=40, height=30))
        cfg = infer_features(p)
        assert len(cfg.cable_channels) > 0

    def test_forearm_has_cable_channel(self):
        p = Part("arm_l_forearm", "structural", "test",
                 dimensions=dict(length=120, width=35, height=25))
        cfg = infer_features(p)
        assert len(cfg.cable_channels) > 0

    def test_battery_box_uses_inner_box(self):
        p = Part("battery_box", "battery", "test",
                 dimensions=dict(length=150, width=60, height=40))
        ops = generate_ops(p)
        types = {o["type"] for o in ops}
        assert "make_box" in types
        assert "boolean" in types
        bool_cuts = [o for o in ops if o["type"] == "boolean" and o["operation"] == "cut"]
        assert len(bool_cuts) >= 2  # inner box cut + grid holes compound cut

    def test_encoder_has_bore(self):
        p = Part("encoder_fl", "sensor", "test",
                 dimensions=dict(diameter=12, height=5))
        cfg = infer_features(p)
        assert cfg.bore is not None

    def test_motor_has_holes_and_bore(self):
        p = Part("motor_fl", "actuator", "test",
                 dimensions=dict(length=40, width=30, height=25))
        cfg = infer_features(p)
        assert len(cfg.mounting_holes) > 0
        assert cfg.bore is not None

    def test_gripper_has_chamfers(self):
        p = Part("arm_l_gripper", "structural", "test",
                 dimensions=dict(length=60, width=30, height=20))
        cfg = infer_features(p)
        assert len(cfg.chamfers) > 0

    def test_fan_has_bore_and_polar_holes(self):
        p = Part("ipc_fan", "structural", "test",
                 dimensions=dict(diameter=40, height=10))
        cfg = infer_features(p)
        assert cfg.bore is not None
        assert len(cfg.mounting_holes) > 0
        assert cfg.mounting_holes[0]["pattern"] == "polar"

    def test_sensor_tower_post_has_bore(self):
        p = Part("sensor_tower_post", "structural", "test",
                 dimensions=dict(diameter=20, height=120))
        cfg = infer_features(p)
        assert cfg.bore is not None
        assert cfg.bore["diameter_mm"] == 5.0

    def test_sensor_mount_has_holes(self):
        p = Part("imu_mount", "sensor", "test",
                 dimensions=dict(length=25, width=15, height=5))
        cfg = infer_features(p)
        assert len(cfg.mounting_holes) > 0

    def test_lidar_mount_has_holes(self):
        p = Part("lidar_mount", "sensor", "test",
                 dimensions=dict(diameter=80, height=40))
        cfg = infer_features(p)
        assert len(cfg.mounting_holes) > 0

    def test_pcb_has_holes_and_chamfers(self):
        p = Part("motor_driver_board", "controller", "test",
                 dimensions=dict(length=70, width=50, height=10))
        cfg = infer_features(p)
        assert len(cfg.mounting_holes) > 0
        assert len(cfg.chamfers) > 0

    def test_bracket_has_fillets(self):
        p = Part("ipc_bracket", "structural", "test",
                 dimensions=dict(length=120, width=80, height=40))
        cfg = infer_features(p)
        assert len(cfg.chamfers) > 0 or len(cfg.fillets) > 0

    def test_unknown_part_empty_config(self):
        p = Part("mystery_part", "custom", "test",
                 dimensions=dict(length=10, width=10, height=10))
        cfg = infer_features(p)
        assert len(cfg.mounting_holes) == 0
        assert cfg.bore is None
        assert cfg.shell is None


# ============================================================================
# 3. TestOperationGeneration — ops contain correct operation types
# ============================================================================


class TestOperationGeneration:
    """Verify that generated ops contain the expected operation types."""

    def _op_types(self, part: Part) -> set[str]:
        return {op["type"] for op in generate_ops(part)}

    def test_plate_uses_plate_with_holes(self):
        p = Part("base_plate", "structural", "test",
                 dimensions=dict(length=300, width=200, height=5))
        assert "plate_with_holes" in self._op_types(p)

    def test_standoff_uses_cylinder_with_hole(self):
        p = Part("standoff_fl", "structural", "test",
                 dimensions=dict(length=8, diameter=6, height=50))
        assert "cylinder_with_hole" in self._op_types(p)

    def test_wheel_uses_cylinder_with_hole(self):
        p = Part("wheel_fl", "structural", "test",
                 dimensions=dict(diameter=65, height=26))
        assert "cylinder_with_hole" in self._op_types(p)
        # Should also have keyway via make_box + boolean cut
        types = self._op_types(p)
        assert "make_box" in types
        assert "boolean" in types

    def test_motor_uses_make_box(self):
        p = Part("motor_fl", "actuator", "test",
                 dimensions=dict(length=40, width=30, height=25))
        types = self._op_types(p)
        assert "make_box" in types
        assert "boolean" in types  # shaft bore cut
        assert "chamfer" in types

    def test_arm_joint_uses_cylinder_with_hole(self):
        p = Part("arm_l_base", "joint", "test",
                 dimensions=dict(outer_diameter=80, height=40))
        types = self._op_types(p)
        assert "cylinder_with_hole" in types
        assert "shell" in types
        assert "fillet" in types
        assert "boolean" in types  # bearing seat + mount holes

    def test_arm_joint_uses_polar_pattern(self):
        p = Part("arm_l_base", "joint", "test",
                 dimensions=dict(outer_diameter=80, height=40))
        types = self._op_types(p)
        assert "polar_pattern" in types

    def test_link_uses_make_box(self):
        p = Part("arm_l_upper_link", "structural", "test",
                 dimensions=dict(length=150, width=40, height=30))
        types = self._op_types(p)
        assert "make_box" in types
        assert "boolean" in types  # cable channel via box + boolean cut
        assert "fillet" in types

    def test_battery_box_uses_inner_box(self):
        p = Part("battery_box", "battery", "test",
                 dimensions=dict(length=150, width=60, height=40))
        types = self._op_types(p)
        assert "make_box" in types
        assert "boolean" in types

    def test_encoder_uses_cylinder_with_hole(self):
        p = Part("encoder_fl", "sensor", "test",
                 dimensions=dict(diameter=12, height=5))
        assert "cylinder_with_hole" in self._op_types(p)

    def test_fan_uses_cylinder_with_hole_and_polar(self):
        p = Part("ipc_fan", "structural", "test",
                 dimensions=dict(diameter=40, height=10))
        types = self._op_types(p)
        assert "cylinder_with_hole" in types
        assert "polar_pattern" in types

    def test_gripper_uses_make_box_and_chamfer(self):
        p = Part("arm_l_gripper", "structural", "test",
                 dimensions=dict(length=60, width=30, height=20))
        types = self._op_types(p)
        assert "make_box" in types
        assert "chamfer" in types

    def test_bracket_uses_fillets(self):
        p = Part("ipc_bracket", "structural", "test",
                 dimensions=dict(length=120, width=80, height=40))
        types = self._op_types(p)
        assert "make_box" in types
        assert "fillet" in types

    def test_sensor_tower_post_uses_cylinder_with_hole(self):
        p = Part("sensor_tower_post", "structural", "test",
                 dimensions=dict(diameter=20, height=120))
        assert "cylinder_with_hole" in self._op_types(p)


# ============================================================================
# 4. TestBackwardCompatibility — _freecad_ops_for_part still works
# ============================================================================


class TestBackwardCompatibility:
    """The old _freecad_ops_for_part() entry point must still work."""

    def test_returns_valid_ops(self):
        p = Part("base_plate", "structural", "test",
                 dimensions=dict(length=300, width=200, height=5))
        ops = _freecad_ops_for_part(p)
        assert ops[0]["type"] == "new_doc"
        assert ops[-1]["type"] == "export_stl"

    def test_unknown_part_falls_back_to_primitive(self):
        p = Part("mystery", "custom", "test",
                 dimensions=dict(length=10, width=10, height=10))
        ops = _freecad_ops_for_part(p)
        assert ops[0]["type"] == "new_doc"
        assert ops[-1]["type"] == "export_stl"
        # Should be a simple make_box (has length+width)
        body_ops = [o for o in ops if o["type"] in ("make_box", "make_cylinder")]
        assert len(body_ops) >= 1

    def test_unknown_cylindrical_part_falls_back(self):
        p = Part("weird_cyl", "custom", "test",
                 dimensions=dict(diameter=20, height=50))
        ops = _freecad_ops_for_part(p)
        body_ops = [o for o in ops if o["type"] in ("make_cylinder", "cylinder_with_hole")]
        assert len(body_ops) >= 1

    def test_unknown_fallback_no_dimensions(self):
        p = Part("nada", "custom", "test",
                 dimensions=dict(height=10))
        ops = _freecad_ops_for_part(p)
        assert ops[0]["type"] == "new_doc"
        assert ops[-1]["type"] == "export_stl"


# ============================================================================
# 5. TestProportionalFeatures — features scale with part size
# ============================================================================


class TestProportionalFeatures:
    """Feature dimensions should scale proportionally with part size."""

    def test_plate_margin_scales(self):
        # Larger plate should have larger margin
        small = Part("base_plate", "structural", "test",
                     dimensions=dict(length=100, width=80, height=5))
        large = Part("base_plate", "structural", "test",
                     dimensions=dict(length=400, width=300, height=5))
        cfg_s = infer_features(small)
        cfg_l = infer_features(large)
        assert cfg_l.mounting_holes[0]["margin"] > cfg_s.mounting_holes[0]["margin"]

    def test_bearing_seat_scales_with_joint_size(self):
        # arm_base (OD=80) should have larger bore than arm_wrist (OD=40)
        base = Part("arm_l_base", "joint", "test",
                    dimensions=dict(outer_diameter=80, height=40))
        wrist = Part("arm_l_wrist", "joint", "test",
                     dimensions=dict(outer_diameter=40, height=25))
        cfg_base = infer_features(base)
        cfg_wrist = infer_features(wrist)
        assert cfg_base.bearing_seats[0]["bore_diameter"] > cfg_wrist.bearing_seats[0]["bore_diameter"]
        assert cfg_base.bearing_seats[0]["shoulder_diameter"] > cfg_wrist.bearing_seats[0]["shoulder_diameter"]

    def test_shell_thickness_scales(self):
        # arm_base should have thicker shell than wrist
        base = Part("arm_l_base", "joint", "test",
                    dimensions=dict(outer_diameter=80, height=40))
        wrist = Part("arm_l_wrist", "joint", "test",
                     dimensions=dict(outer_diameter=40, height=25))
        cfg_base = infer_features(base)
        cfg_wrist = infer_features(wrist)
        assert cfg_base.shell["thickness_mm"] > cfg_wrist.shell["thickness_mm"]

    def test_cable_channel_scales_with_link_length(self):
        short = Part("arm_l_forearm", "structural", "test",
                     dimensions=dict(length=120, width=35, height=25))
        long = Part("arm_l_upper_link", "structural", "test",
                    dimensions=dict(length=150, width=40, height=30))
        cfg_s = infer_features(short)
        cfg_l = infer_features(long)
        # Channel length (end_offset - start_offset) should be longer for longer link
        len_s = cfg_s.cable_channels[0]["end_offset"] - cfg_s.cable_channels[0]["start_offset"]
        len_l = cfg_l.cable_channels[0]["end_offset"] - cfg_l.cable_channels[0]["start_offset"]
        assert len_l > len_s

    def test_mounting_hole_spacing_scales_with_plate_size(self):
        small = Part("base_plate", "structural", "test",
                     dimensions=dict(length=100, width=80, height=5))
        large = Part("base_plate", "structural", "test",
                     dimensions=dict(length=400, width=300, height=5))
        ops_s = generate_ops(small)
        ops_l = generate_ops(large)
        # Find plate_with_holes op and check margins
        margin_s = next(o for o in ops_s if o["type"] == "plate_with_holes")["margin"]
        margin_l = next(o for o in ops_l if o["type"] == "plate_with_holes")["margin"]
        assert margin_l > margin_s


# ============================================================================
# 6. TestBodyNameChain — verify FreeCAD object naming is consistent
# ============================================================================


class TestBodyNameChain:
    """Verify that boolean/shell/pocket ops chain correctly via unique names.

    The core correctness property: every operation that references a body by
    name must reference an object that actually exists in the FreeCAD
    document at that point in the script execution.
    """

    def _simulate_doc_names(self, ops: list[dict]) -> set[str]:
        """Simulate FreeCAD document object names after running *ops*.

        Returns the set of names that exist in the document after all ops.
        Raises AssertionError if any op references a non-existent object.
        """
        existing: set[str] = set()

        for op in ops:
            t = op["type"]

            if t == "new_doc":
                existing.clear()

            elif t in ("make_box", "make_cylinder", "make_sphere", "make_cone"):
                name = op["name"]
                assert name not in existing, f"Duplicate name: {name}"
                existing.add(name)

            elif t == "cylinder_with_hole":
                name = op["name"]
                assert name not in existing, f"Duplicate name: {name}"
                existing.add(name)

            elif t == "plate_with_holes":
                name = op["name"]
                assert name not in existing, f"Duplicate name: {name}"
                existing.add(name)

            elif t == "boolean":
                obj1 = op["object1"]
                obj2 = op["object2"]
                result = op["result_name"]
                assert obj1 in existing, f"boolean: object1 '{obj1}' not found"
                assert obj2 in existing, f"boolean: object2 '{obj2}' not found"
                assert result not in existing, f"Duplicate result: {result}"
                existing.add(result)

            elif t == "delete_object":
                obj = op["object"]
                assert obj in existing, f"delete_object: '{obj}' not found"
                existing.discard(obj)

            elif t == "move":
                obj = op["object"]
                assert obj in existing, f"move: '{obj}' not found"

            elif t == "rotate":
                obj = op["object"]
                assert obj in existing, f"rotate: '{obj}' not found"

            elif t == "fillet":
                obj = op["object"]
                assert obj in existing, f"fillet: '{obj}' not found"

            elif t == "chamfer":
                obj = op["object"]
                assert obj in existing, f"chamfer: '{obj}' not found"

            elif t == "shell":
                obj = op.get("object", "")
                result = op.get("result_name", "Shell")
                if obj:
                    assert obj in existing, f"shell: '{obj}' not found"
                assert result not in existing, f"Duplicate shell result: {result}"
                existing.add(result)

            elif t == "polar_pattern":
                obj = op.get("object", "")
                result = op.get("result_name", "PolarPattern")
                if obj:
                    assert obj in existing, f"polar_pattern: '{obj}' not found"
                assert result not in existing, f"Duplicate pattern result: {result}"
                existing.add(result)

            elif t == "linear_pattern":
                obj = op.get("object", "")
                result = op.get("result_name", "LinearPattern")
                if obj:
                    assert obj in existing, f"linear_pattern: '{obj}' not found"
                assert result not in existing, f"Duplicate pattern result: {result}"
                existing.add(result)

            elif t == "create_sketch":
                name = op.get("name", "Sketch")
                assert name not in existing, f"Duplicate sketch: {name}"
                existing.add(name)

            elif t == "extrude_sketch":
                sketch = op.get("sketch", "Sketch")
                result = op.get("name", "Extrusion")
                assert sketch in existing, f"extrude_sketch: sketch '{sketch}' not found"
                assert result not in existing, f"Duplicate extrude result: {result}"
                existing.add(result)

            elif t == "pocket":
                sketch = op.get("sketch", "Sketch")
                target = op.get("target", "")
                result = op.get("name", "PocketResult")
                assert sketch in existing, f"pocket: sketch '{sketch}' not found"
                if target:
                    assert target in existing, f"pocket: target '{target}' not found"
                assert result not in existing, f"Duplicate pocket result: {result}"
                existing.add(result)

            elif t in ("export_stl", "export_step", "save",
                       "status", "object_info", "volume_check", "compute_mass"):
                pass  # no name changes

        return existing

    def test_arm_base_body_chain(self):
        """arm_l_base: cylinder_with_hole → bearing seat cut → polar holes → shell → fillet."""
        p = Part("arm_l_base", "joint", "test",
                 dimensions=dict(outer_diameter=80, height=40))
        ops = generate_ops(p)
        self._simulate_doc_names(ops)

    def test_motor_body_chain(self):
        """motor_fl: make_box → grid holes (fused) → shaft bore → chamfer."""
        p = Part("motor_fl", "actuator", "test",
                 dimensions=dict(length=40, width=30, height=25))
        ops = generate_ops(p)
        self._simulate_doc_names(ops)

    def test_wheel_body_chain(self):
        """wheel_fl: cylinder_with_hole → keyway cut."""
        p = Part("wheel_fl", "structural", "test",
                 dimensions=dict(diameter=65, height=26))
        ops = generate_ops(p)
        self._simulate_doc_names(ops)

    def test_arm_link_body_chain(self):
        """arm_l_upper_link: make_box → end holes → pocket → fillet."""
        p = Part("arm_l_upper_link", "structural", "test",
                 dimensions=dict(length=150, width=40, height=30))
        ops = generate_ops(p)
        self._simulate_doc_names(ops)

    def test_battery_box_body_chain(self):
        """battery_box: plate_with_holes → shell → fillet."""
        p = Part("battery_box", "battery", "test",
                 dimensions=dict(length=150, width=60, height=40))
        ops = generate_ops(p)
        self._simulate_doc_names(ops)

    def test_fan_body_chain(self):
        """ipc_fan: cylinder_with_hole → polar holes → cut."""
        p = Part("ipc_fan", "structural", "test",
                 dimensions=dict(diameter=40, height=10))
        ops = generate_ops(p)
        self._simulate_doc_names(ops)

    def test_gripper_body_chain(self):
        """arm_l_gripper: make_box → bottom hole → channel → chamfer."""
        p = Part("arm_l_gripper", "structural", "test",
                 dimensions=dict(length=60, width=30, height=20))
        ops = generate_ops(p)
        self._simulate_doc_names(ops)

    def test_bracket_body_chain(self):
        """ipc_bracket: make_box → two_faces holes → fillet."""
        p = Part("ipc_bracket", "structural", "test",
                 dimensions=dict(length=120, width=80, height=40))
        ops = generate_ops(p)
        self._simulate_doc_names(ops)

    def test_all_41_parts_have_valid_body_chain(self):
        """Every part in the 41-part robot must pass the name-chain simulation."""
        for part in build_complex_robot().parts:
            ops = generate_ops(part)
            self._simulate_doc_names(ops)  # asserts internally

    def test_export_stl_references_final_body(self):
        """export_stl must have explicit 'object' pointing to the final body."""
        for part in build_complex_robot().parts:
            ops = generate_ops(part)
            export_op = ops[-1]
            assert export_op["type"] == "export_stl"
            assert "object" in export_op, (
                f"{part.name}: export_stl missing explicit 'object' key"
            )
            # The referenced object must still exist in the simulated document
            final_names = self._simulate_doc_names(ops)
            assert export_op["object"] in final_names, (
                f"{part.name}: export_stl object '{export_op['object']}' "
                f"not in final document {final_names}"
            )

    def test_document_is_clean_at_export(self):
        """At export time, the document should contain only 1 solid (the final body)
        plus possibly the export_stl target.  No leftover tool objects."""
        for part in build_complex_robot().parts:
            ops = generate_ops(part)
            names = self._simulate_doc_names(ops)
            # Filter to objects that would have a Shape (exclude Sketcher objects)
            solid_like = {n for n in names if "sketch" not in n.lower()
                          and "kwsk" not in n.lower() and "chsk" not in n.lower()}
            assert len(solid_like) <= 3, (
                f"{part.name}: too many objects at export ({solid_like})"
            )

    def test_no_duplicate_result_names_in_any_part(self):
        """No two object-creating ops should produce the same name.

        ``new_doc`` uses ``name`` for the document (not an object), so it
        is excluded from the check.
        """
        _OBJECT_CREATING_OPS = {
            "make_box", "make_cylinder", "make_sphere", "make_cone",
            "cylinder_with_hole", "plate_with_holes",
            "boolean", "polar_pattern", "linear_pattern", "mirror",
            "shell", "draft",
            "create_sketch", "extrude_sketch", "revolve_sketch", "pocket",
        }
        for part in build_complex_robot().parts:
            ops = generate_ops(part)
            created = set()
            for op in ops:
                if op["type"] not in _OBJECT_CREATING_OPS:
                    continue
                for key in ("name", "result_name"):
                    if key in op:
                        n = op[key]
                        assert n not in created, (
                            f"{part.name}: duplicate created name '{n}'"
                        )
                        created.add(n)
