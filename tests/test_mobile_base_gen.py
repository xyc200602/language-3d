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


def _wheel_ground_clearance(data: dict) -> float:
    """Solve and return the minimum wheel-BOTTOM Z (ground contact).

    The ground is Z=0. A wheel resting on the ground has bottom_z ≈ 0.
    The pre-2026-06-24 Z-stack buried wheels 55mm underground (bottom_z<0)
    because base_plate was the Z=0 root and wheels hung below it. The
    Husky-style base_footprint root fixes this: wheel bottoms land ~0.
    Returns the minimum over all wheels.
    """
    asm = _to_assembly(data)
    pos = AssemblySolver(asm).solve()
    bottoms = []
    for p in data["parts"]:
        if not p["name"].startswith("wheel_"):
            continue
        z = pos[p["name"]]["position"][2]
        radius = p["dimensions"]["diameter"] / 2.0
        bottoms.append(z - radius)
    return min(bottoms) if bottoms else 0.0


def _base_plate_bottom_z(data: dict) -> float:
    """Solve and return the base_plate BOTTOM Z (deck underside)."""
    asm = _to_assembly(data)
    pos = AssemblySolver(asm).solve()
    z = pos["base_plate"]["position"][2]
    half_h = next(
        p["dimensions"]["height"] / 2.0
        for p in data["parts"] if p["name"] == "base_plate"
    )
    return z - half_h


def _wheel_top_z(data: dict) -> float:
    """Solve and return the maximum wheel-TOP Z (highest point of any wheel)."""
    asm = _to_assembly(data)
    pos = AssemblySolver(asm).solve()
    tops = []
    for p in data["parts"]:
        if p["name"].startswith("wheel_"):
            z = pos[p["name"]]["position"][2]
            tops.append(z + p["dimensions"]["diameter"] / 2.0)
    return max(tops) if tops else 0.0


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
        assert "battery_box" in names
        # Husky structure: wheel + motor per corner (no top_plate/standoff/suspension).
        # Arms mount directly on base_plate (the single deck).
        assert "top_plate" not in names
        for c in ["fl", "fr", "rl", "rr"][:wheel_count]:
            assert f"wheel_{c}" in names
            assert f"motor_{c}" in names


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
    def test_wheel_joint_axis_is_x(self, wheel_count):
        """axis=x is what makes the wheel roll along the body's LONG edge.

        The body's long edge (base_length) maps to Y (front/back).  A wheel
        that rolls along Y (forward/back) must have its axle perpendicular
        to Y, i.e. along X.  axis=z would stand the wheel vertical (wrong);
        axis=y (the old value) made it roll along X (the SHORT edge / the
        width), which is sideways relative to the long body — the
        '轮子不应该顺着长边吗' defect."""
        data = _load(wheel_count=wheel_count)
        for j in data["joints"]:
            if j["child"].startswith("wheel_"):
                assert j["axis"] == "x", f"{j['child']} axis={j['axis']}"

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


class TestHuskyStructure:
    """Guard the Husky-style chassis topology (real_ugv_chassis_engineering.md).

    The pre-rewrite structure had a 4-layer chain (base_plate→motor→suspension
    →wheel) plus a separate chassis_body shell and standoffs. That was wrong —
    real UGVs (Husky, Leo Rover) mount wheels DIRECTLY on base_link via a
    continuous/revolute joint, with no suspension link and no separate shell.
    These tests pin the correct structure so it can't regress."""

    def test_wheel_parent_is_base_plate(self):
        """Every wheel's revolute joint parent must be base_plate (not motor,
        not suspension). Husky: wheel continuous joint → base_link."""
        data = _load(wheel_count=4)
        for j in data["joints"]:
            if j["child"].startswith("wheel_"):
                assert j["parent"] == "base_plate", (
                    f"{j['child']} parent is {j['parent']}, expected base_plate"
                )
                assert j["type"] == "revolute"
                assert j["axis"] == "x"  # rolls along Y (the long edge)

    def test_has_suspension_parts(self):
        """Each wheel has a visible suspension strut (project requirement +
        user direction). The strut is a short cylinder between the wheel axle
        and the chassis underside, modeled as a real mechanical part — not
        just metadata. Previously the chassis was rigid (Husky-style, no
        suspension), but the project expectation explicitly requires
        suspension to be modeled."""
        data = _load(wheel_count=4)
        names = {p["name"] for p in data["parts"]}
        susp = {n for n in names if n.startswith("suspension_")}
        assert len(susp) == 4, f"expected 4 suspension struts, got {len(susp)}: {susp}"

    def test_has_chassis_body_shell(self):
        """The chassis has a visible 3D body shell (chassis_body) that rises
        above the deck and encloses the drivetrain — like Husky's base_link
        (267mm tall). Without it the robot reads as "a flat plate on wheels".
        The shell is narrower than the deck so wheels stay visible at the
        sides, and sits ON the deck top so it doesn't intersect the wheels."""
        data = _load(wheel_count=4)
        names = {p["name"] for p in data["parts"]}
        assert "chassis_body" in names, "chassis_body shell missing"
        body = next(p for p in data["parts"] if p["name"] == "chassis_body")
        deck = next(p for p in data["parts"] if p["name"] == "base_plate")
        # Body must be taller than the thin deck (it's a real shell, not a plate)
        assert body["dimensions"]["height"] > deck["dimensions"]["height"] * 2, (
            "chassis_body should be a tall shell, not a thin plate"
        )

    def test_no_standoffs(self):
        """No standoffs — the body itself is the structure (top_plate mounts
        directly on base_plate with a Z offset)."""
        data = _load(wheel_count=4)
        names = {p["name"] for p in data["parts"]}
        assert not any("standoff" in n.lower() for n in names)

    def test_motor_inside_base(self):
        """Motors are real COTS parts mounted INSIDE the body (fixed to
        base_plate), not in the wheel drive chain. Husky drives via belt
        internally; the motor is visible but doesn't carry the wheel joint."""
        data = _load(wheel_count=4)
        for j in data["joints"]:
            if j["child"].startswith("motor_"):
                assert j["parent"] == "base_plate"
                assert j["type"] == "fixed"
        # And no wheel's parent is a motor (motor is not in the drive chain).
        for j in data["joints"]:
            if j["child"].startswith("wheel_"):
                assert not j["parent"].startswith("motor_"), (
                    "wheel must not hang off a motor (drive chain simplified)"
                )

    def test_motor_has_real_model(self):
        """Motor description carries a real COTS model number (AGENTS.md §1.2:
        functional parts need real specs, not invented boxes)."""
        data = _load(wheel_count=4)
        m = next(p for p in data["parts"] if p["name"] == "motor_fl")
        assert any(tok in m["description"] for tok in ("JGA25", "GA25", "NEMA", "TT"))


class TestGroundContact:
    """The Husky-style Z-stack (added 2026-06-24): wheels rest ON the ground
    (bottom Z ≈ 0) and the deck sits ABOVE the wheels.

    The pre-fix Z-stack was inverted: base_plate was the Z=0 root and wheels
    hung below it via the motors, burying wheels 55mm underground (bottom
    Z = -55). The fix adds a virtual base_footprint root at Z=0 and raises
    the real base_plate above the wheel axle. These tests pin both halves of
    that contract so it cannot regress."""

    @pytest.mark.parametrize("wheel_count", [2, 4])
    def test_wheels_rest_on_ground(self, wheel_count):
        """Every wheel's bottom must be at (or within 5mm of) Z=0.

        This is the headline physical-correctness check: a wheel whose
        bottom is below 0 is buried; above ~10mm is floating. ±5mm accounts
        for the solver's auto joint-face clearance jitter."""
        data = _load(wheel_count=wheel_count)
        clearance = _wheel_ground_clearance(data)
        assert -1.0 <= clearance <= 8.0, (
            f"wheel bottom Z = {clearance:.2f}mm — not resting on ground "
            f"(Z=0). Buried (<-1) or floating (>8)."
        )

    @pytest.mark.parametrize("payload_kg", [2.0, 5.0, 20.0])
    def test_wheels_grounded_across_payloads(self, payload_kg):
        """Ground contact must hold for every payload tier (different wheel
        diameters / clearances), not just the default 5kg."""
        data = _load(wheel_count=4, payload_kg=payload_kg)
        clearance = _wheel_ground_clearance(data)
        assert -1.0 <= clearance <= 8.0, (
            f"payload={payload_kg}kg wheel bottom Z = {clearance:.2f}mm"
        )

    @pytest.mark.parametrize("wheel_count", [2, 4])
    def test_deck_sits_above_axle(self, wheel_count):
        """The base_plate underside must be ABOVE the wheel axle.

        Wheels poke outward from the chassis corners (in XY they sit outside
        the base plate), so a Z-overlap between deck and wheel-top is not a
        physical collision. What matters is that the deck sits ABOVE the
        axle so the deck isn't at ground level (the old bug) and the wheel
        can spin freely beneath the deck overhang."""
        data = _load(wheel_count=wheel_count)
        deck_bottom = _base_plate_bottom_z(data)
        wheel_axle = _wheel_top_z(data) - (
            next(p["dimensions"]["diameter"] for p in data["parts"]
                 if p["name"].startswith("wheel_")) / 2.0
        )
        assert deck_bottom >= wheel_axle, (
            f"deck bottom Z={deck_bottom:.1f} is below the wheel axle "
            f"Z={wheel_axle:.1f} — deck would block the wheel"
        )

    @pytest.mark.parametrize("wheel_count", [2, 4])
    def test_base_footprint_is_ground_root(self, wheel_count):
        """base_footprint is the virtual ground-contact root at Z=0.

        Its presence is what lets the rest of the Z-stack reference the
        ground plane; without it base_plate would be the root and wheels
        would sink below Z=0 again."""
        data = _load(wheel_count=wheel_count)
        names = {p["name"] for p in data["parts"]}
        assert "base_footprint" in names
        # It must be the true root (no joint has it as a child).
        children = {j["child"] for j in data["joints"]}
        assert "base_footprint" not in children, (
            "base_footprint must be the root, but a joint parents onto it"
        )
        # And it must be the parent of base_plate.
        bp_joints = [
            j for j in data["joints"]
            if j["parent"] == "base_footprint" and j["child"] == "base_plate"
        ]
        assert len(bp_joints) == 1, "expected exactly one base_footprint→base_plate joint"


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
        """The _arm_mount block tells the composer where arms attach.

        Arms mount on the chassis_body TOP FACE (the visible shell), not on
        the thin base_plate deck beneath it — this matches real wheeled
        dual-arm robots (HSR/TIAGo/Fetch), where the arm base is co-planar
        with the chassis top surface. Mounting on the lower deck had the
        shell enclose the arm bases (穿模), so this contract is load-bearing
        for collision-free assembly."""
        data = _load()
        assert "_arm_mount" in data
        assert data["_arm_mount"]["deck"] == "chassis_body"
        assert "offset_mm" in data["_arm_mount"]
        # deck_top_z is the shell's top-face height above the deck center;
        # the composer uses it to raise the arm base onto the shell top.
        assert "deck_top_z" in data["_arm_mount"]
        assert data["_arm_mount"]["deck_top_z"] > 0


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
