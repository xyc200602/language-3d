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
    # profile="mobile": arms mounted on a wheeled chassis use the compact
    # scale (shorter links, smaller base) — the desktop profile is rejected
    # by the proportion guard, which is the correct behaviour.
    arm = build_arm_example(arm_dof, profile="mobile")
    # configure_avoidance=False: these tests check composition STRUCTURE
    # (topology, symmetry, finger gap). The collision-aware configurator is
    # a heavy per-joint solver+FCL search (~minutes) and is exercised by its
    # own dedicated test below (TestDualArmAvoidance). Skipping it here keeps
    # the structural suite fast while still validating the avoidance logic.
    return json.loads(compose_dual_arm_assembly(
        chassis, arm, arm_dof, configure_avoidance=False,
    ))


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

    def test_right_arm_internal_offsets_not_mirrored(self):
        """The right arm's INTERNAL offsets must NOT be negated.

        The arm mounts on the chassis as a rigid, un-mirrored subtree — the
        lateral placement (left vs right) is handled by the mounting joint's
        ±spread offset, not by mirroring the arm's internal geometry.
        Mirroring the gripper finger offsets collapsed the right arm's
        gripper: the ±16mm finger spread flipped sign, driving both fingers
        inward to a 12.4mm overlap (0mm gap) instead of a 37.6mm gap. Real
        dual-arm robots (TIAGo++, HSR) have both grippers oriented the same
        way (opening forward), not mirrored. This test pins the fix.
        """
        d = _compose(4)
        for j in d["joints"]:
            if j["child"] == "arm_r_gripper_finger_left" and "offset" in j:
                left_off = next(
                    jj["offset"][0] for jj in d["joints"]
                    if jj["child"] == "arm_l_gripper_finger_left"
                )
                assert j["offset"][0] == pytest.approx(left_off, abs=0.01), (
                    "right arm finger offset must EQUAL left arm's (not negated) "
                    f"— got {j['offset'][0]}, expected {left_off}"
                )

    @pytest.mark.parametrize("dof", [3, 4, 5, 6])
    def test_both_grippers_have_adequate_finger_gap(self, dof):
        """Both arms' grippers must have ≥35mm finger gap (verifier threshold).

        This is the headline regression: before the mirror fix, arm_r's
        gripper fingers collapsed to 12.4mm apart, failing VLM verification
        and dead-looping the fixer. Both arms must now solve to the same
        healthy gap.
        """
        d = _compose(dof)
        asm = _to_assembly(d)
        pos = AssemblySolver(asm).solve()
        for side in ("l", "r"):
            fl = pos[f"arm_{side}_gripper_finger_left"]["position"]
            fr = pos[f"arm_{side}_gripper_finger_right"]["position"]
            center_dist = abs(fl[0] - fr[0])
            finger_width = 14  # from arm_topology._FINGER
            gap = center_dist - finger_width
            assert gap >= 35.0, (
                f"arm_{side} finger gap {gap:.1f}mm < 35mm "
                f"(centers {center_dist:.1f}mm apart)"
            )


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


class TestNoBodyInterpenetration:
    """Regression guard for the穿模 defect: arms must NOT pierce the
    chassis_body shell.

    Before this contract, arms mounted on the thin base_plate deck (Z≈101)
    while the chassis_body shell rose from Z≈98 to Z≈143 above it — so the
    shell ENCLOSED the arm base_plate and base_yaw_servo (visible穿模). The
    fix mounts arms on the chassis_body TOP FACE so they rise above the
    shell. These tests pin that contract: no arm part's bottom may dip
    below the shell top, and no arm part may partially AABB-overlap the
    shell. (Full containment is the battery's case — the battery is
    intentionally hidden inside the shell — so we only flag PARTIAL
    overlaps.)"""

    @staticmethod
    def _aabb(part, pos):
        d = part["dimensions"]
        cx, cy, cz = pos[part["name"]]["position"]
        if "diameter" in d:  # upright cylinder
            r = d["diameter"] / 2.0
            h = d["height"]
            return (cx - r, cx + r, cy - r, cy + r, cz - h / 2.0, cz + h / 2.0)
        lx, ly, lz = d.get("length", 0), d.get("width", 0), d.get("height", 0)
        return (cx - lx / 2.0, cx + lx / 2.0,
                cy - ly / 2.0, cy + ly / 2.0,
                cz - lz / 2.0, cz + lz / 2.0)

    @staticmethod
    def _overlap(a, b):
        ox = max(0, min(a[1], b[1]) - max(a[0], b[0]))
        oy = max(0, min(a[3], b[3]) - max(a[2], b[2]))
        oz = max(0, min(a[5], b[5]) - max(a[4], b[4]))
        return ox * oy * oz

    @staticmethod
    def _contained(small, big):
        return (
            small[0] >= big[0] and small[1] <= big[1]
            and small[2] >= big[2] and small[3] <= big[3]
            and small[4] >= big[4] and small[5] <= big[5]
        )

    @pytest.mark.parametrize("dof", [3, 4, 5, 6])
    def test_arms_clear_of_chassis_body(self, dof):
        """No arm part may PARTIALLY AABB-overlap the chassis_body shell.

        The arm bases must sit on the shell TOP face (above its max Z), and
        every higher link rises further from there. Any partial overlap means
        an arm part pierced the shell."""
        d = _compose(dof)
        asm = _to_assembly(d)
        pos = AssemblySolver(asm).solve()
        parts = {p["name"]: p for p in d["parts"]}
        body_box = self._aabb(parts["chassis_body"], pos)
        body_top = body_box[5]
        bad = []
        for name, part in parts.items():
            if not name.startswith("arm_"):
                continue
            box = self._aabb(part, pos)
            # The arm base bottom must be at or above the shell top.
            if box[4] < body_top - 0.5:
                bad.append((name, f"z_bottom={box[4]:.1f} < body_top={body_top:.1f}"))
                continue
            ov = self._overlap(body_box, box)
            if ov > 0 and not self._contained(box, body_box):
                bad.append((name, f"partial overlap {ov:.0f}mm^3"))
        assert not bad, (
            f"arm parts pierce the chassis_body shell: {bad}"
        )

    def test_arm_base_rests_on_shell_top(self):
        """The arm base_plate bottom must coincide with the shell top face
        (±2mm tolerance for solver float jitter), proving the arm is mounted
        ON the shell rather than buried under it."""
        d = _compose(4)
        asm = _to_assembly(d)
        pos = AssemblySolver(asm).solve()
        parts = {p["name"]: p for p in d["parts"]}
        body_top = self._aabb(parts["chassis_body"], pos)[5]
        for side in ("l", "r"):
            base = parts[f"arm_{side}_base_plate"]
            cz = pos[f"arm_{side}_base_plate"]["position"][2]
            base_h = base["dimensions"]["height"]
            base_bottom = cz - base_h / 2.0
            assert abs(base_bottom - body_top) < 2.0, (
                f"arm_{side}_base_plate bottom Z={base_bottom:.1f}, "
                f"shell top Z={body_top:.1f} — arm not resting on shell top"
            )


class TestProportionGuard:
    """The proportion guard (Change C) prevents desktop-scale arms from being
    bolted onto a mobile chassis, and lets correctly-scaled mobile arms through.

    Real wheeled dual-arm robots keep arm-reach within ~1-3× the chassis
    diagonal and the arm base ≤ ~60% of the deck area. A desktop-profile arm
    (200×150 base, 360mm reach) violates both and must be rejected so the
    caller regenerates at mobile scale — NOT silently rescaled."""

    def test_mobile_profile_arm_passes_guard(self):
        """A profile='mobile' arm (70×50 base, ~280mm reach on 108mm deck)
        must compose without raising — it is the correct scale."""
        chassis = build_wheeled_base(wheel_count=4, payload_kg=5.0)
        arm = build_arm_example(4, profile="mobile")
        # Must not raise.
        result = compose_dual_arm_assembly(chassis, arm, arm_dof=4)
        assert "parts" in json.loads(result)

    def test_proportion_guard_rejects_oversized_arm(self):
        """The proportion guard must reject an arm whose base exceeds 60% of
        the deck area. We synthesise an arm with a deliberately oversized
        base plate (bigger than the chassis deck) and confirm it raises."""
        import pytest
        import json as _json
        chassis = build_wheeled_base(wheel_count=4, payload_kg=2.0)
        arm = _json.loads(build_arm_example(4, profile="mobile"))
        # Inflate the arm base to be bigger than the chassis base.
        for p in arm["parts"]:
            if p["name"] == "base_plate":
                p["dimensions"]["length"] = 300
                p["dimensions"]["width"] = 300
        with pytest.raises(ValueError, match="profile='mobile'"):
            compose_dual_arm_assembly(chassis, _json.dumps(arm), arm_dof=4)

    def test_mobile_arm_base_under_60pct_of_deck(self):
        """The mobile arm's base must occupy < 60% of the deck area (the
        structural proportion that makes a real robot look right)."""
        d = _compose(4)
        deck = next(p for p in d["parts"] if p["name"] == "base_plate")
        arm_base = next(p for p in d["parts"] if p["name"] == "arm_l_base_plate")
        deck_area = deck["dimensions"]["length"] * deck["dimensions"]["width"]
        arm_area = arm_base["dimensions"]["length"] * arm_base["dimensions"]["width"]
        assert arm_area < deck_area * 0.6, (
            f"arm base {arm_area:.0f}mm² >= 60% of deck {deck_area:.0f}mm²"
        )

    def test_mobile_servos_same_as_desktop(self):
        """Functional parts (servos) must be IDENTICAL across profiles — the
        profile only changes structural link/base sizes, never real COTS servos.
        This pins AGENTS.md §1.2: no rescaling functional parts."""
        from src.lang3d.knowledge.arm_topology import _servo_spec
        # The servo specs are module-level constants shared by both profiles.
        # A mobile arm's shoulder servo == a desktop arm's shoulder servo.
        mobile = json.loads(build_arm_example(4, profile="mobile"))
        desktop = json.loads(build_arm_example(4))
        m_servo = next(p for p in mobile["parts"] if p["name"] == "base_yaw_servo")
        d_servo = next(p for p in desktop["parts"] if p["name"] == "base_yaw_servo")
        assert m_servo["dimensions"] == d_servo["dimensions"], (
            "servo dims differ between profiles — functional parts must not change"
        )
        # And the dims match the real catalog spec.
        model, dia, h = _servo_spec(0)
        assert m_servo["dimensions"] == {"diameter": dia, "height": h}
        assert model == "MG996R"


class TestDualArmAvoidance:
    """Multi-arm autonomous collision avoidance — the project expectation:
    "多机械臂的话自主避障".

    The collision-aware configurator (``configure_avoidance=True``, the
    production default) configures a dual-arm assembly into a collision-free
    coordinated pose by (1) splaying the base yaws outward symmetrically and
    (2) tightening each joint's range_deg to its collision-free reachable
    region.  This is heavy (solver+FCL per step) so it has its own dedicated
    test rather than running on every structural compose."""

    def test_configured_pose_is_collision_free(self):
        """With avoidance configured, a full motion-sweep finds ZERO
        motion-induced collisions across all arm joints."""
        from lang3d.tools.motion_collision import HAS_FCL
        if not HAS_FCL:
            pytest.skip("python-fcl not installed")
        from lang3d.tools.motion_collision import MotionCollisionChecker

        chassis = build_wheeled_base(wheel_count=4, payload_kg=5.0)
        arm = build_arm_example(4, profile="mobile")
        d = json.loads(compose_dual_arm_assembly(
            chassis, arm, arm_dof=4, configure_avoidance=True,
        ))
        asm = _to_assembly(d)
        mc = MotionCollisionChecker(num_samples=7)
        res = mc.check_motion_collisions(assembly=asm, skip_adjacent=True)
        colliding = [jr.joint_name for jr in res.joint_results if jr.has_collision]
        assert colliding == [], (
            f"avoidance-configured dual-arm still collides on: {colliding}"
        )

    def test_base_yaws_splayed_outward(self):
        """The configurator must splay the base yaws symmetrically outward
        (left negative, right positive), not leave them at 0 (stacked)."""
        from lang3d.tools.motion_collision import HAS_FCL
        if not HAS_FCL:
            pytest.skip("python-fcl not installed")
        chassis = build_wheeled_base(wheel_count=4, payload_kg=5.0)
        arm = build_arm_example(4, profile="mobile")
        d = json.loads(compose_dual_arm_assembly(
            chassis, arm, arm_dof=4, configure_avoidance=True,
        ))
        angles = d["default_angles"]
        l_yaw = angles["arm_l_base_yaw_servo"]
        r_yaw = angles["arm_r_base_yaw_servo"]
        assert l_yaw < 0, f"left yaw must splay toward -X (outward), got {l_yaw}"
        assert r_yaw > 0, f"right yaw must splay toward +X (outward), got {r_yaw}"
        # Symmetric (mirror).
        assert abs(l_yaw + r_yaw) < 1.0

    def test_ranges_narrowed_from_bare_servo_spec(self):
        """At least the base-yaw ranges must be NARROWER than the bare ±180°
        servo spec — proving the configurator applied collision-aware soft
        limits (a yaw that physically hits the other arm is not a usable
        ±180°)."""
        from lang3d.tools.motion_collision import HAS_FCL
        if not HAS_FCL:
            pytest.skip("python-fcl not installed")
        chassis = build_wheeled_base(wheel_count=4, payload_kg=5.0)
        arm = build_arm_example(4, profile="mobile")
        d = json.loads(compose_dual_arm_assembly(
            chassis, arm, arm_dof=4, configure_avoidance=True,
        ))
        by_name = {j["child"]: j for j in d["joints"]}
        for side in ("l", "r"):
            jn = f"arm_{side}_base_yaw_servo"
            lo, hi = by_name[jn]["range_deg"]
            span = hi - lo
            assert span < 360.0, (
                f"{jn} range [{lo},{hi}] span {span}° not narrowed from ±180°"
            )

