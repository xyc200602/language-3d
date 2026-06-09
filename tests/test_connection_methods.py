"""Tests for ConnectionMethod and assembly connection types.

Covers:
1. ConnectionMethod dataclass and properties
2. Required constraints per connection type
3. Required additional parts per connection type
4. Human-readable descriptions
5. Joint with ConnectionMethod integration
6. BoltHole dataclass
"""

from __future__ import annotations

import pytest

from lang3d.knowledge.mechanics import (
    Assembly,
    BoltHole,
    ConnectionMethod,
    Joint,
    Part,
)


# ---------------------------------------------------------------------------
# 1. ConnectionMethod dataclass basics
# ---------------------------------------------------------------------------

class TestConnectionMethodBasics:

    def test_create_bolted_connection(self):
        conn = ConnectionMethod(
            type="bolted",
            bolt_size="M4",
            bolt_count=4,
            torque_nm=2.5,
        )
        assert conn.type == "bolted"
        assert conn.bolt_size == "M4"
        assert conn.bolt_count == 4
        assert conn.torque_nm == 2.5

    def test_create_press_fit_connection(self):
        conn = ConnectionMethod(
            type="press_fit",
            interference_mm=0.1,
        )
        assert conn.type == "press_fit"
        assert conn.interference_mm == 0.1

    def test_create_snap_fit_connection(self):
        conn = ConnectionMethod(
            type="snap_fit",
            snap_count=2,
            snap_force_n=15.0,
        )
        assert conn.type == "snap_fit"
        assert conn.snap_count == 2

    def test_create_adhesive_connection(self):
        conn = ConnectionMethod(
            type="adhesive",
            adhesive_type="epoxy",
            bond_area_mm2=500.0,
        )
        assert conn.type == "adhesive"
        assert conn.adhesive_type == "epoxy"
        assert conn.bond_area_mm2 == 500.0

    def test_create_welded_connection(self):
        conn = ConnectionMethod(
            type="welded",
            weld_type="fillet",
        )
        assert conn.type == "welded"
        assert conn.weld_type == "fillet"

    def test_create_magnetic_connection(self):
        conn = ConnectionMethod(type="magnetic")
        assert conn.type == "magnetic"


# ---------------------------------------------------------------------------
# 2. Required constraints per connection type
# ---------------------------------------------------------------------------

class TestRequiredConstraints:

    def test_bolted_requires_coincident_and_concentric(self):
        conn = ConnectionMethod(type="bolted")
        assert "coincident" in conn.required_constraints
        assert "concentric" in conn.required_constraints

    def test_press_fit_requires_concentric_and_distance(self):
        conn = ConnectionMethod(type="press_fit")
        assert "concentric" in conn.required_constraints
        assert "distance" in conn.required_constraints

    def test_snap_fit_requires_coincident(self):
        conn = ConnectionMethod(type="snap_fit")
        assert "coincident" in conn.required_constraints

    def test_adhesive_requires_coincident(self):
        conn = ConnectionMethod(type="adhesive")
        assert "coincident" in conn.required_constraints

    def test_welded_requires_coincident(self):
        conn = ConnectionMethod(type="welded")
        assert "coincident" in conn.required_constraints

    def test_magnetic_requires_coincident(self):
        conn = ConnectionMethod(type="magnetic")
        assert "coincident" in conn.required_constraints

    def test_unknown_type_defaults_to_coincident(self):
        conn = ConnectionMethod(type="unknown_type")
        assert conn.required_constraints == ["coincident"]


# ---------------------------------------------------------------------------
# 3. Required additional parts per connection type
# ---------------------------------------------------------------------------

class TestRequiredParts:

    def test_bolted_requires_fasteners(self):
        conn = ConnectionMethod(type="bolted")
        parts = conn.required_parts
        assert "bolt" in parts
        assert "nut" in parts
        assert "washer" in parts

    def test_press_fit_no_additional_parts(self):
        conn = ConnectionMethod(type="press_fit")
        assert conn.required_parts == []

    def test_snap_fit_no_additional_parts(self):
        conn = ConnectionMethod(type="snap_fit")
        assert conn.required_parts == []

    def test_adhesive_requires_adhesive_material(self):
        conn = ConnectionMethod(type="adhesive")
        assert "adhesive" in conn.required_parts

    def test_welded_no_additional_parts(self):
        conn = ConnectionMethod(type="welded")
        assert conn.required_parts == []

    def test_magnetic_requires_magnet(self):
        conn = ConnectionMethod(type="magnetic")
        assert "magnet" in conn.required_parts


# ---------------------------------------------------------------------------
# 4. Human-readable descriptions
# ---------------------------------------------------------------------------

class TestConnectionDescription:

    def test_bolted_description(self):
        conn = ConnectionMethod(type="bolted", bolt_size="M5", bolt_count=4, torque_nm=5.0)
        desc = conn.describe()
        assert "M5" in desc
        assert "4" in desc
        assert "5.0" in desc
        assert "螺栓" in desc

    def test_press_fit_description(self):
        conn = ConnectionMethod(type="press_fit", interference_mm=0.12)
        desc = conn.describe()
        assert "0.12" in desc
        assert "压入" in desc

    def test_snap_fit_description(self):
        conn = ConnectionMethod(type="snap_fit", snap_count=3, snap_force_n=20.0)
        desc = conn.describe()
        assert "3" in desc
        assert "卡扣" in desc

    def test_adhesive_description(self):
        conn = ConnectionMethod(type="adhesive", adhesive_type="epoxy", bond_area_mm2=300)
        desc = conn.describe()
        assert "epoxy" in desc
        assert "黏结" in desc

    def test_welded_description(self):
        conn = ConnectionMethod(type="welded", weld_type="fillet")
        desc = conn.describe()
        assert "fillet" in desc
        assert "焊接" in desc

    def test_magnetic_description(self):
        conn = ConnectionMethod(type="magnetic")
        desc = conn.describe()
        assert "磁吸" in desc


# ---------------------------------------------------------------------------
# 5. Joint with ConnectionMethod integration
# ---------------------------------------------------------------------------

class TestJointWithConnection:

    def test_joint_with_bolted_connection(self):
        joint = Joint(
            type="fixed",
            parent="base_plate",
            child="motor_bracket",
            description="电机支架螺栓固定",
            parent_anchor="top",
            child_anchor="bottom",
            connection=ConnectionMethod(
                type="bolted",
                bolt_size="M4",
                bolt_count=4,
                torque_nm=2.0,
            ),
        )
        assert joint.connection is not None
        assert joint.connection.type == "bolted"
        assert joint.connection.bolt_count == 4

    def test_joint_with_press_fit(self):
        joint = Joint(
            type="fixed",
            parent="joint_housing",
            child="bearing_608",
            description="轴承压入座孔",
            parent_anchor="center",
            child_anchor="center",
            connection=ConnectionMethod(
                type="press_fit",
                interference_mm=0.05,
            ),
        )
        assert joint.connection.type == "press_fit"

    def test_joint_without_connection_backward_compatible(self):
        """Joints without connection field should still work."""
        joint = Joint(
            type="revolute",
            parent="shoulder",
            child="elbow",
            axis="y",
        )
        assert joint.connection is None

    def test_revolute_joint_with_bolted_servo(self):
        """A revolute joint where the servo is bolted to the bracket."""
        joint = Joint(
            type="revolute",
            parent="bracket",
            child="servo_arm",
            axis="z",
            range_deg=(-180, 180),
            connection=ConnectionMethod(
                type="bolted",
                bolt_size="M3",
                bolt_count=2,
            ),
        )
        assert joint.type == "revolute"
        assert joint.connection.bolt_size == "M3"

    def test_assembly_with_connection_methods(self):
        assembly = Assembly(
            name="test_assembly",
            parts=[
                Part(name="plate", category="structural", description="板",
                     dimensions={"length": 100, "width": 80, "height": 5}),
                Part(name="motor", category="actuator", description="电机",
                     dimensions={"length": 40, "width": 30, "height": 25}),
            ],
            joints=[
                Joint(
                    type="fixed",
                    parent="plate",
                    child="motor",
                    connection=ConnectionMethod(type="bolted", bolt_size="M4", bolt_count=4),
                ),
            ],
        )
        assert len(assembly.joints) == 1
        assert assembly.joints[0].connection.type == "bolted"
        assert assembly.joints[0].connection.required_parts == ["bolt", "nut", "washer"]


# ---------------------------------------------------------------------------
# 6. BoltHole dataclass
# ---------------------------------------------------------------------------

class TestBoltHole:

    def test_create_bolt_hole(self):
        hole = BoltHole(
            position=(10.0, 10.0, 0.0),
            diameter=3.4,
            depth=5.0,
            bolt_size="M3",
        )
        assert hole.position == (10.0, 10.0, 0.0)
        assert hole.diameter == 3.4
        assert hole.bolt_size == "M3"

    def test_bolt_hole_defaults(self):
        hole = BoltHole(position=(0.0, 0.0, 0.0))
        assert hole.diameter == 3.0
        assert hole.depth == 0.0
        assert hole.bolt_size == "M3"

    def test_through_hole_has_zero_depth(self):
        hole = BoltHole(position=(0, 0, 0), diameter=4.5, depth=0)
        assert hole.depth == 0  # Through hole

    def test_connection_with_bolt_holes(self):
        conn = ConnectionMethod(
            type="bolted",
            bolt_size="M5",
            bolt_count=4,
            bolt_holes=[
                BoltHole(position=(10, 10, 0), diameter=5.5, bolt_size="M5"),
                BoltHole(position=(50, 10, 0), diameter=5.5, bolt_size="M5"),
                BoltHole(position=(10, 40, 0), diameter=5.5, bolt_size="M5"),
                BoltHole(position=(50, 40, 0), diameter=5.5, bolt_size="M5"),
            ],
        )
        assert len(conn.bolt_holes) == 4
        assert all(h.bolt_size == "M5" for h in conn.bolt_holes)
