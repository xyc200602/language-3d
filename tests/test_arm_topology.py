"""Tests for the parametric single-arm topology generator (2-7 DOF).

Covers ``knowledge/arm_topology.build_arm_example`` and ``parse_dof``,
which replace the legacy hand-written 4/5/6-DOF example JSONs and the
keyword-substring DOF detector. The key invariant these tests pin: for
*every* DOF from 2 to 7, the generated assembly is a connected tree with
exactly one base z-yaw, a working gripper, and solver-positionable parts
— i.e. the parametric scaffold is at least as valid as the hard-coded
4-DOF exemplar that previously served as the catch-all fallback.
"""

from __future__ import annotations

import json
from collections import defaultdict, deque

import pytest

from lang3d.knowledge.arm_topology import (
    build_arm_example,
    parse_dof,
    zigzag_angles,
    _joint_schedule,
)
from lang3d.knowledge.mechanics import Assembly, Joint, Part
from lang3d.tools.assembly_solver import AssemblySolver


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load(n_dof: int) -> dict:
    return json.loads(build_arm_example(n_dof))


def _to_assembly(data: dict) -> Assembly:
    """Rebuild an Assembly from the generated JSON (drops connection_detail
    fields the Joint dataclass does not carry)."""
    parts = [
        Part(
            name=p["name"], category=p["category"], description=p["description"],
            material=p.get("material", "PLA"), dimensions=p["dimensions"],
        )
        for p in data["parts"]
    ]
    joint_fields = (
        "type", "parent", "child", "axis", "parent_anchor", "child_anchor",
        "offset", "range_deg", "mimic_joint", "mimic_multiplier", "mimic_offset",
    )
    joints = [
        Joint(**{k: v for k, v in j.items() if k in joint_fields})
        for j in data["joints"]
    ]
    return Assembly(
        name=data["name"], parts=parts, joints=joints,
        default_angles=data.get("default_angles", {}),
    )


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


# ---------------------------------------------------------------------------
# DOF schedule
# ---------------------------------------------------------------------------


class TestJointSchedule:
    def test_base_yaw_is_always_first(self):
        for n in range(2, 8):
            roles = [r[0] for r in _joint_schedule(n)]
            assert roles[0] == "yaw", f"DOF={n}: base yaw not first"

    def test_exactly_one_yaw(self):
        """Only the base is a z-yaw; the wrist never duplicates it."""
        for n in range(2, 8):
            roles = [r[0] for r in _joint_schedule(n)]
            assert roles.count("yaw") == 1, f"DOF={n}: multiple yaws {roles}"

    def test_dof_count_matches_roles(self):
        for n in range(2, 8):
            assert len(_joint_schedule(n)) == n

    def test_clamps_out_of_range(self):
        assert len(_joint_schedule(1)) == 2   # clamped up
        assert len(_joint_schedule(99)) == 7  # clamped down


# ---------------------------------------------------------------------------
# zig-zag sequence
# ---------------------------------------------------------------------------


class TestZigzagAngles:
    def test_signs_alternate(self):
        """A zig-zag must alternate sign so the arm extends, not curls."""
        for n in (2, 3, 4, 5, 6, 7):
            seq = zigzag_angles(n)
            for i in range(1, len(seq)):
                assert seq[i] * seq[i - 1] < 0, f"n={n}: not alternating at {i}"

    def test_single_joint(self):
        assert zigzag_angles(1) == [-35.0]

    def test_empty(self):
        assert zigzag_angles(0) == []

    def test_extends_for_many_joints(self):
        """>4 joints must still get a full-length sequence (no wrap loss)."""
        seq = zigzag_angles(7)
        assert len(seq) == 7
        assert all(abs(v) > 0 for v in seq)


# ---------------------------------------------------------------------------
# parse_dof
# ---------------------------------------------------------------------------


class TestParseDof:
    @pytest.mark.parametrize("desc,expected", [
        ("6自由度机械臂", 6),
        ("设计一个4自由度机械臂", 4),
        ("7-dof arm", 7),
        ("5dof robot", 5),
        ("两轴机械臂", 2),
        ("三自由度机械臂", 3),
        ("六轴工业臂", 6),
        ("N关节机械臂 with 3", None),  # bare digit without DOF unit → None
    ])
    def test_parses_explicit_dof(self, desc, expected):
        assert parse_dof(desc) == expected

    @pytest.mark.parametrize("desc", [
        "5V power supply",
        "6 wheels robot",
        "a robot with 4 legs",
        "the arm reaches 300mm",
    ])
    def test_no_false_positive_on_unrelated_numbers(self, desc):
        """Regression: the old keyword detector matched '5'/'6' anywhere and
        misclassified '5V' / '6 wheels' as 5-DOF / 6-DOF arms."""
        assert parse_dof(desc) is None

    def test_none_when_unstated(self):
        assert parse_dof("a robotic arm") is None
        assert parse_dof("") is None

    def test_out_of_range_returns_none(self):
        assert parse_dof("1自由度") is None
        assert parse_dof("8自由度机械臂") is None


# ---------------------------------------------------------------------------
# Generated assembly structure (the core validation)
# ---------------------------------------------------------------------------


class TestBuildArmExample:
    @pytest.mark.parametrize("n", [2, 3, 4, 5, 6, 7])
    def test_valid_json(self, n):
        data = _load(n)
        assert data["name"].startswith(f"{n}dof")
        assert "parts" in data and "joints" in data

    @pytest.mark.parametrize("n", [2, 3, 4, 5, 6, 7])
    def test_unique_part_names(self, n):
        names = [p["name"] for p in _load(n)["parts"]]
        assert len(names) == len(set(names))

    @pytest.mark.parametrize("n", [2, 3, 4, 5, 6, 7])
    def test_joint_refs_valid(self, n):
        data = _load(n)
        names = {p["name"] for p in data["parts"]}
        for j in data["joints"]:
            assert j["parent"] in names
            assert j["child"] in names

    @pytest.mark.parametrize("n", [2, 3, 4, 5, 6, 7])
    def test_connected_tree(self, n):
        assert _connected_tree(_load(n)), f"DOF={n}: not a connected tree"

    @pytest.mark.parametrize("n", [2, 3, 4, 5, 6, 7])
    def test_exactly_one_base_z_yaw(self, n):
        """The base yaw must be the ONLY revolute joint with axis=z,
        anchored top/bottom on the base_plate."""
        data = _load(n)
        z_revs = [
            j for j in data["joints"]
            if j.get("type") == "revolute" and j.get("axis") == "z"
        ]
        assert len(z_revs) == 1, f"DOF={n}: {len(z_revs)} z-yaw joints"
        assert z_revs[0]["parent"] == "base_plate"
        assert z_revs[0]["parent_anchor"] == "top"

    @pytest.mark.parametrize("n", [2, 3, 4, 5, 6, 7])
    def test_arm_joints_use_front_back(self, n):
        """Every non-base joint uses front/back so link length participates
        in positioning (the IK axis-to-axis distance requirement).

        The base yaw is the ONLY top/bottom joint (sits on the plate,
        axis=z). All other arm-segment joints (pitch shoulder/elbow, wrist
        roll) use front/back so each link's ``length`` extends the arm
        horizontally — this is the convention the solver and the
        ``_fix_arm_chain_anchors`` sanitizer both expect."""
        data = _load(n)
        for j in data["joints"]:
            if j["parent"] == "base_plate":
                continue  # base joint is top/bottom
            if j["child"] == "gripper_servo":
                continue  # gripper servo mounts top/bottom on gripper_base
            if j["type"] == "prismatic":
                continue  # gripper fingers use front/back too, handled separately
            assert j["parent_anchor"] == "front", (
                f"DOF={n}: joint {j['parent']}->{j['child']} "
                f"anchor={j['parent_anchor']}"
            )

    @pytest.mark.parametrize("n", [2, 3, 4, 5, 6, 7])
    def test_gripper_present(self, n):
        names = {p["name"] for p in _load(n)["parts"]}
        for g in ("gripper_base", "gripper_servo",
                  "gripper_finger_left", "gripper_finger_right"):
            assert g in names, f"DOF={n}: missing {g}"

    @pytest.mark.parametrize("n", [2, 3, 4, 5, 6, 7])
    def test_pitch_joints_have_default_angles(self, n):
        """Pitch (axis=x) joints must carry a non-zero default angle so the
        arm folds into a pose instead of lying flat."""
        data = _load(n)
        pitch_children = {
            j["child"] for j in data["joints"]
            if j.get("type") == "revolute" and j.get("axis") == "x"
        }
        for child in pitch_children:
            assert child in data["default_angles"], (
                f"DOF={n}: pitch joint {child} has no default_angle"
            )
            assert abs(data["default_angles"][child]) > 1e-6

    @pytest.mark.parametrize("n", [2, 3, 4, 5, 6, 7])
    def test_part_count_grows_with_dof(self, n):
        """More DOF → more parts (each extra joint adds a servo+link pair)."""
        assert len(_load(n)["parts"]) >= len(_load(n - 1)["parts"]) if n > 2 else True

    @pytest.mark.parametrize("n", [2, 3, 4, 5, 6, 7])
    def test_solver_positions_all_parts_no_nan(self, n):
        """The generated arm must be solvable by AssemblySolver with every
        part positioned and no NaN/Inf coordinates. This is the offline
        equivalent of the e2e position check — it runs without an LLM key."""
        asm = _to_assembly(_load(n))
        positions = AssemblySolver(asm).solve()
        assert len(positions) == len(asm.parts), f"DOF={n}: missing positions"
        for name, pose in positions.items():
            pos = pose.get("position", [])
            assert len(pos) == 3, f"DOF={n}: {name} bad position shape"
            for c in pos:
                assert c == c, f"DOF={n}: {name} has NaN position"
                assert abs(c) != float("inf")
        # base_plate at origin
        assert positions["base_plate"]["position"] == [0.0, 0.0, 0.0]
