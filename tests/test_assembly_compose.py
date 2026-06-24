"""Tests for the deterministic dual-arm assembly composer.

Covers ``agent/assembly_compose.compose_dual_arm_assembly`` — the ArtiCAD-
style deterministic assembler that bolts two mirrored arms onto a wheeled
chassis with no LLM involvement. The key invariants: the result is one
connected tree, both arms are present and symmetric, the wheels stay flat,
and the arm internal topology (DOF, gripper) is preserved through the copy.
"""

from __future__ import annotations

import json
from collections import defaultdict, deque

import pytest

from lang3d.agent.assembly_compose import compose_dual_arm_assembly
from lang3d.knowledge.arm_topology import build_arm_example
from lang3d.knowledge.mechanics import Assembly, Joint, Part
from lang3d.knowledge.mobile_base_gen import build_wheeled_base
from lang3d.tools.assembly_solver import AssemblySolver


def _compose(arm_dof: int = 4, **chassis_kw) -> dict:
    chassis = build_wheeled_base(wheel_count=4, **chassis_kw)
    arm = build_arm_example(arm_dof)
    return json.loads(compose_dual_arm_assembly(chassis, arm, arm_dof))


def _to_assembly(data: dict) -> Assembly:
    parts = [
        Part(name=p["name"], category=p["category"], description=p["description"],
             material=p.get("material", "PLA"), dimensions=p["dimensions"])
        for p in data["parts"]
    ]
    jf = ("type", "parent", "child", "axis", "parent_anchor", "child_anchor",
          "offset", "range_deg", "no_distribute", "distribution_group",
          "mimic_joint", "mimic_multiplier", "mimic_offset")
    joints = [Joint(**{k: v for k, v in j.items() if k in jf}) for j in data["joints"]]
    return Assembly(name=data["name"], parts=parts, joints=joints,
                    default_angles=data.get("default_angles", {}))


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


class TestComposition:
    def test_produces_valid_json(self):
        d = _compose(4)
        assert d["name"].startswith("dual_arm_4dof")
        assert "parts" in d and "joints" in d

    def test_connected_tree(self):
        assert _connected_tree(_compose(4))

    def test_joint_count_is_parts_minus_one(self):
        d = _compose(4)
        assert len(d["joints"]) == len(d["parts"]) - 1

    def test_has_both_arms(self):
        names = {p["name"] for p in _compose(4)["parts"]}
        assert "arm_l_base_plate" in names
        assert "arm_r_base_plate" in names

    def test_has_chassis(self):
        names = {p["name"] for p in _compose(4)["parts"]}
        for c in ["fl", "fr", "rl", "rr"]:
            assert f"wheel_{c}" in names
            assert f"motor_{c}" in names


class TestArmSymmetry:
    """The headline fix: arms must be symmetric (left/right mirror)."""

    def test_arm_bases_symmetric_in_x(self):
        d = _compose(4)
        asm = _to_assembly(d)
        pos = AssemblySolver(asm).solve()
        l = pos["arm_l_base_plate"]["position"]
        r = pos["arm_r_base_plate"]["position"]
        # Symmetric about X=0: l[0] ≈ -r[0]
        assert abs(l[0] + r[0]) < 1.0, f"X asymmetry: l={l[0]} r={r[0]}"
        assert abs(l[1] - r[1]) < 1.0  # same Y
        assert abs(l[2] - r[2]) < 1.0  # same Z

    def test_arm_grippers_present_both_sides(self):
        names = {p["name"] for p in _compose(4)["parts"]}
        for side in ("l", "r"):
            assert f"arm_{side}_gripper_finger_left" in names
            assert f"arm_{side}_gripper_finger_right" in names

    def test_right_arm_offsets_mirrored(self):
        """The right arm's finger X offsets must be negated (mirror)."""
        d = _compose(4)
        for j in d["joints"]:
            if j["child"] == "arm_r_gripper_finger_left" and "offset" in j:
                left_off = next(
                    jj["offset"][0] for jj in d["joints"]
                    if jj["child"] == "arm_l_gripper_finger_left"
                )
                assert j["offset"][0] == pytest.approx(-left_off, abs=0.01)


class TestTopologyPreserved:
    """Each copied arm keeps its internal DOF and joint conventions."""

    @pytest.mark.parametrize("dof", [3, 4, 5, 6])
    def test_each_arm_has_correct_dof(self, dof):
        d = _compose(dof)
        # Count revolute joints in each arm (excluding wheels + chassis).
        for side in ("l", "r"):
            arm_rev = [
                j for j in d["joints"]
                if j.get("type") == "revolute"
                and j["child"].startswith(f"arm_{side}_")
            ]
            assert len(arm_rev) == dof, (
                f"arm_{side} has {len(arm_rev)} revolute joints, expected {dof}"
            )

    def test_arm_z_yaw_is_only_top_bottom_joint(self):
        """Each arm's base yaw must still be the only z-axis top/bottom joint."""
        d = _compose(4)
        for side in ("l", "r"):
            z_joints = [
                j for j in d["joints"]
                if j.get("axis") == "z"
                and j.get("parent_anchor") == "top"
                and j["child"].startswith(f"arm_{side}_")
            ]
            assert len(z_joints) == 1, f"arm_{side} has {len(z_joints)} z-yaw joints"


class TestWheelsStayFlat:
    """Composing arms must NOT break the chassis wheels — they stay flat."""

    def test_wheel_z_variation_near_zero(self):
        d = _compose(4)
        asm = _to_assembly(d)
        pos = AssemblySolver(asm).solve()
        zs = [pos[f"wheel_{c}"]["position"][2] for c in ["fl", "fr", "rl", "rr"]]
        assert max(zs) - min(zs) < 1.0

    def test_wheels_below_base(self):
        d = _compose(4)
        asm = _to_assembly(d)
        pos = AssemblySolver(asm).solve()
        base_z = pos["base_plate"]["position"][2]
        for c in ["fl", "fr", "rl", "rr"]:
            assert pos[f"wheel_{c}"]["position"][2] < base_z


class TestSolverIntegration:
    def test_all_parts_positioned(self):
        d = _compose(6)
        asm = _to_assembly(d)
        pos = AssemblySolver(asm).solve()
        assert len(pos) == len(d["parts"])

    def test_no_nan_positions(self):
        d = _compose(4)
        asm = _to_assembly(d)
        pos = AssemblySolver(asm).solve()
        for name, pose in pos.items():
            for c in pose.get("position", []):
                assert c == c, f"{name} NaN"
