"""Tests for the parametric wheeled mobile-base generator.

Covers ``knowledge/mobile_base_gen.build_wheeled_base`` — the wheeled-base
counterpart to ``arm_topology.build_arm_example``. The key invariant these
tests pin: for every wheel count and payload, the generated chassis solves
with **all wheels flat on the ground** (zero Z variation) and the
axis/anchor/no_distribute conventions preserved — the exact properties the
LLM kept breaking when given EXAMPLE_4W_ROBOT as advisory text.
"""

from __future__ import annotations

import json
from collections import defaultdict, deque

import pytest

from lang3d.knowledge.mobile_base_gen import (
    build_wheeled_base,
    parse_drive_type,
)
from lang3d.knowledge.mechanics import Assembly, Joint, Part
from lang3d.tools.assembly_solver import AssemblySolver


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load(**kwargs) -> dict:
    return json.loads(build_wheeled_base(**kwargs))


def _to_assembly(data: dict) -> Assembly:
    parts = [
        Part(name=p["name"], category=p["category"], description=p["description"],
             material=p.get("material", "PLA"), dimensions=p["dimensions"])
        for p in data["parts"]
    ]
    jf = ("type", "parent", "child", "axis", "parent_anchor", "child_anchor",
          "offset", "range_deg", "no_distribute", "distribution_group")
    joints = [Joint(**{k: v for k, v in j.items() if k in jf}) for j in data["joints"]]
    return Assembly(name=data["name"], parts=parts, joints=joints)


def _connected_tree(data: dict) -> bool:
    parts = {p["name"] for p in data["parts"]}
    joints = data["joints"]
    if len(joints) != len(parts) - 1:
        return False
    adj: dict[str, list[str]] = defaultdict(list)
    for j in joints:
        adj[j["parent"]].append(j["child"])
        adj[j["child"]].append(j["parent"])
    seen = {"base_plate"}
    q = deque(["base_plate"])
    while q:
        c = q.popleft()
        for nb in adj[c]:
            if nb not in seen:
                seen.add(nb)
                q.append(nb)
    return seen == parts


def _wheel_z_variation(data: dict) -> float:
    """Solve the assembly and return the max-min Z spread of all wheels.

    Wheels must be flat: variation ≈ 0. A non-zero value means at least one
    wheel is off the ground — the failure mode this generator exists to
    prevent."""
    asm = _to_assembly(data)
    pos = AssemblySolver(asm).solve()
    wheel_names = [p["name"] for p in data["parts"] if p["name"].startswith("wheel_")]
    zs = [pos[n]["position"][2] for n in wheel_names if n in pos]
    if not zs:
        return 0.0
    return max(zs) - min(zs)


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


class TestBuildWheeledBase:
    @pytest.mark.parametrize("wheel_count", [2, 4])
    def test_valid_json(self, wheel_count):
        data = _load(wheel_count=wheel_count)
        assert data["name"].startswith(f"{wheel_count}w")
        assert "parts" in data and "joints" in data

    @pytest.mark.parametrize("wheel_count", [2, 4])
    def test_unique_part_names(self, wheel_count):
        names = [p["name"] for p in _load(wheel_count=wheel_count)["parts"]]
        assert len(names) == len(set(names))

    @pytest.mark.parametrize("wheel_count", [2, 4])
    def test_connected_tree(self, wheel_count):
        assert _connected_tree(_load(wheel_count=wheel_count))

    @pytest.mark.parametrize("wheel_count", [2, 4])
    def test_joint_count_is_parts_minus_one(self, wheel_count):
        data = _load(wheel_count=wheel_count)
        assert len(data["joints"]) == len(data["parts"]) - 1

    @pytest.mark.parametrize("wheel_count", [2, 4])
    def test_has_core_parts(self, wheel_count):
        names = {p["name"] for p in _load(wheel_count=wheel_count)["parts"]}
        assert "base_plate" in names
        assert "top_plate" in names
        assert "battery_box" in names
        # one motor + wheel + standoff per corner
        for c in ["fl", "fr", "rl", "rr"][:wheel_count]:
            assert f"motor_{c}" in names
            assert f"wheel_{c}" in names
            assert f"standoff_{c}" in names


# ---------------------------------------------------------------------------
# THE critical invariant: wheels flat + correct joint conventions
# ---------------------------------------------------------------------------


class TestWheelConventions:
    """These are the conventions the LLM kept breaking. The generator
    guarantees them structurally so they can't be mutated away."""

    @pytest.mark.parametrize("wheel_count", [2, 4])
    def test_wheels_flat_zero_z_variation(self, wheel_count):
        """All wheels must solve to the same Z (flat on the ground).

        This is the headline failure the generator fixes: the LLM-produced
        dual-arm had 42.5mm Z variation; the generator gives 0.0mm."""
        data = _load(wheel_count=wheel_count)
        assert _wheel_z_variation(data) < 1.0  # < 1mm tolerance for float noise

    @pytest.mark.parametrize("wheel_count", [2, 4])
    def test_wheel_joint_axis_is_y(self, wheel_count):
        """axis=y is what makes the cylinder lie horizontal and roll.
        axis=z (the LLM's frequent mistake) stands the wheel vertical."""
        data = _load(wheel_count=wheel_count)
        for j in data["joints"]:
            if j["child"].startswith("wheel_"):
                assert j["axis"] == "y", f"{j['child']} axis={j['axis']}"

    @pytest.mark.parametrize("wheel_count", [2, 4])
    def test_wheel_child_anchor_is_center(self, wheel_count):
        """child_anchor=center places the wheel centre on the motor face.
        A face anchor would offset the wheel sideways into the motor body."""
        data = _load(wheel_count=wheel_count)
        for j in data["joints"]:
            if j["child"].startswith("wheel_"):
                assert j["child_anchor"] == "center"

    @pytest.mark.parametrize("wheel_count", [2, 4])
    def test_wheel_no_distribute_true(self, wheel_count):
        """no_distribute=True is mandatory: without it the sibling-distribution
        algorithm treats the 4 wheels as a group and spreads them across the
        base face, flinging them hundreds of mm from their motors."""
        data = _load(wheel_count=wheel_count)
        for j in data["joints"]:
            if j["child"].startswith("wheel_"):
                assert j.get("no_distribute") is True

    @pytest.mark.parametrize("wheel_count", [2, 4])
    def test_motor_distribution_group(self, wheel_count):
        """Motors share a distribution_group so the solver's 2x2 grid puts
        them at the four corners of the base underside."""
        data = _load(wheel_count=wheel_count)
        motor_groups = {
            j["distribution_group"] for j in data["joints"]
            if j["child"].startswith("motor_")
        }
        assert motor_groups == {"motors"}


# ---------------------------------------------------------------------------
# Parametric sizing (payload drives dimensions)
# ---------------------------------------------------------------------------


class TestParametricSizing:
    def test_heavier_payload_gets_bigger_wheels(self):
        """A 50kg robot must have larger wheels than a 2kg one — the
        hard-coded example gave both 65mm."""
        light = _load(payload_kg=2.0)
        heavy = _load(payload_kg=50.0)
        lw = next(p["dimensions"]["diameter"] for p in light["parts"]
                  if p["name"] == "wheel_fl")
        hw = next(p["dimensions"]["diameter"] for p in heavy["parts"]
                  if p["name"] == "wheel_fl")
        assert hw > lw

    def test_heavier_payload_gets_wider_stance(self):
        light = _load(payload_kg=2.0)
        heavy = _load(payload_kg=50.0)
        lb = next(p["dimensions"]["width"] for p in light["parts"]
                  if p["name"] == "base_plate")
        hb = next(p["dimensions"]["width"] for p in heavy["parts"]
                  if p["name"] == "base_plate")
        assert hb > lb

    def test_arm_mount_metadata_present(self):
        """The _arm_mount block tells the composer where arms attach."""
        data = _load()
        assert "_arm_mount" in data
        assert data["_arm_mount"]["deck"] == "top_plate"
        assert "offset_mm" in data["_arm_mount"]


# ---------------------------------------------------------------------------
# Drive-type inference
# ---------------------------------------------------------------------------


class TestParseDriveType:
    @pytest.mark.parametrize("desc,expected", [
        ("麦克纳姆轮机器人", "mecanum"),
        ("mecanum base", "mecanum"),
        ("全向移动底盘", "mecanum"),
        ("omnidirectional", "mecanum"),
        ("4轮差速底盘", "differential"),
        ("双轮差速机器人", "differential"),
        ("一个机器人", "differential"),  # default
    ])
    def test_inference(self, desc, expected):
        assert parse_drive_type(desc) == expected


# ---------------------------------------------------------------------------
# Solver integration (the assembly must actually solve)
# ---------------------------------------------------------------------------


class TestSolverIntegration:
    @pytest.mark.parametrize("wheel_count", [2, 4])
    def test_all_parts_positioned_no_nan(self, wheel_count):
        asm = _to_assembly(_load(wheel_count=wheel_count))
        pos = AssemblySolver(asm).solve()
        assert len(pos) == len(asm.parts)
        for name, pose in pos.items():
            p = pose.get("position", [])
            assert len(p) == 3
            for c in p:
                assert c == c, f"{name} has NaN"
                assert abs(c) != float("inf")

    def test_wheels_below_base_plate(self):
        """Wheels must hang below the deck (negative Z relative to base)."""
        asm = _to_assembly(_load(wheel_count=4))
        pos = AssemblySolver(asm).solve()
        base_z = pos["base_plate"]["position"][2]
        for c in ["fl", "fr", "rl", "rr"]:
            wz = pos[f"wheel_{c}"]["position"][2]
            assert wz < base_z, f"wheel_{c} at Z={wz} not below base Z={base_z}"
