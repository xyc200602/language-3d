"""Unit tests for the targeted modification engine (agent/modifier.py).

Covers:
- ModificationRequest classification (Chinese + English keywords)
- Part-level scaling / anchor flip / offset bump
- Subsystem-level scaling + collision routing
- apply_targeted_fix_from_vlm deterministically resolving common problems
- modifications_diff reporting
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT / "src"))

from lang3d.agent.modifier import (  # noqa: E402
    ModificationRequest,
    apply_modification,
    apply_targeted_fix_from_vlm,
    classify_modification,
    modifications_diff,
)
from lang3d.agent.assembly_visual_verifier import (  # noqa: E402
    LayoutProblem,
    ProblemType,
    Severity,
    classify_problem_text,
)
from lang3d.knowledge.mechanics import Assembly, Joint, Part  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def arm_assembly() -> Assembly:
    """A minimal 4-DOF-arm-like assembly with gripper fingers."""
    parts = [
        Part(name="base_plate", category="structural", description="base",
             dimensions={"length": 100, "width": 100, "height": 5}),
        Part(name="shoulder_link", category="link", description="shoulder",
             dimensions={"length": 120, "width": 20, "height": 20}),
        Part(name="elbow_link", category="link", description="elbow",
             dimensions={"length": 100, "width": 18, "height": 18}),
        Part(name="gripper_finger_left", category="effector", description="l",
             dimensions={"length": 30, "width": 4, "height": 15}),
        Part(name="gripper_finger_right", category="effector", description="r",
             dimensions={"length": 30, "width": 4, "height": 15}),
    ]
    joints = [
        Joint(type="fixed", parent="base_plate", child="shoulder_link"),
        Joint(type="revolute", parent="shoulder_link", child="elbow_link",
              parent_anchor="top", child_anchor="bottom"),
        Joint(type="prismatic", parent="elbow_link",
              child="gripper_finger_left", offset=(0.0, -2.0, 0.0)),
        Joint(type="prismatic", parent="elbow_link",
              child="gripper_finger_right", offset=(0.0, 2.0, 0.0)),
    ]
    return Assembly(name="test_arm", parts=parts, joints=joints,
                    description="a 4dof arm")


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


class TestClassify:
    def test_enlarge_subsystem_chinese(self, arm_assembly):
        req = classify_modification("把机械臂放大两倍", arm_assembly)
        assert req.scope == "subsystem"
        assert req.target == "arm"
        assert req.intent == "enlarge"
        assert req.params["factor"] == pytest.approx(2.0)

    def test_enlarge_subsystem_english(self, arm_assembly):
        req = classify_modification("make the gripper 50% larger", arm_assembly)
        assert req.scope == "subsystem"
        assert req.target == "gripper"
        assert req.intent == "enlarge"
        assert req.params["factor"] == pytest.approx(1.5)

    def test_enlarge_specific_part(self, arm_assembly):
        req = classify_modification("enlarge gripper_finger_left by 30%", arm_assembly)
        assert req.scope == "part"
        assert req.target == "gripper_finger_left"
        assert req.params["factor"] == pytest.approx(1.3)

    def test_redo(self, arm_assembly):
        req = classify_modification("完全重做机械臂", arm_assembly)
        assert req.scope == "whole"
        assert req.intent == "redo"

    def test_fix_collision(self, arm_assembly):
        req = classify_modification("fix finger overlap on gripper", arm_assembly)
        assert req.scope == "subsystem"
        assert req.target == "gripper"
        assert req.intent == "fix_collision"

    def test_fix_orientation(self, arm_assembly):
        req = classify_modification("flip shoulder_link orientation", arm_assembly)
        assert req.scope == "part"
        assert req.target == "shoulder_link"
        assert req.intent == "fix_orientation"

    def test_unknown_falls_back_to_whole(self, arm_assembly):
        req = classify_modification("paint it red", arm_assembly)
        assert req.scope == "whole"


# ---------------------------------------------------------------------------
# Part-level modifiers
# ---------------------------------------------------------------------------


class TestModifyPart:
    def test_scale_part_enlarge(self, arm_assembly):
        req = ModificationRequest(
            scope="part", intent="enlarge", target="gripper_finger_left",
            params={"factor": 2.0}, source="user", raw_text="x",
        )
        new = apply_modification(arm_assembly, req)
        finger_before = next(p for p in arm_assembly.parts
                             if p.name == "gripper_finger_left")
        finger_after = next(p for p in new.parts
                            if p.name == "gripper_finger_left")
        assert finger_after.dimensions["length"] == pytest.approx(
            finger_before.dimensions["length"] * 2.0
        )
        assert finger_after.dimensions["width"] == pytest.approx(
            finger_before.dimensions["width"] * 2.0
        )

    def test_flip_anchor(self, arm_assembly):
        req = ModificationRequest(
            scope="part", intent="fix_orientation", target="elbow_link",
            source="user", raw_text="x",
        )
        new = apply_modification(arm_assembly, req)
        joint_before = next(j for j in arm_assembly.joints
                            if j.child == "elbow_link")
        joint_after = next(j for j in new.joints if j.child == "elbow_link")
        assert joint_after.parent_anchor == joint_before.child_anchor
        assert joint_after.child_anchor == joint_before.parent_anchor

    def test_part_not_found_noop(self, arm_assembly):
        req = ModificationRequest(
            scope="part", intent="enlarge", target="nonexistent",
            params={"factor": 2.0}, source="user", raw_text="x",
        )
        new = apply_modification(arm_assembly, req)
        # Should return the assembly unchanged
        assert len(new.parts) == len(arm_assembly.parts)


# ---------------------------------------------------------------------------
# Subsystem-level modifiers
# ---------------------------------------------------------------------------


class TestModifySubsystem:
    def test_scale_gripper(self, arm_assembly):
        req = ModificationRequest(
            scope="subsystem", intent="enlarge", target="gripper",
            params={"factor": 1.5}, source="user", raw_text="x",
        )
        new = apply_modification(arm_assembly, req)
        # Both fingers should be scaled
        for name in ("gripper_finger_left", "gripper_finger_right"):
            after = next(p for p in new.parts if p.name == name)
            before = next(p for p in arm_assembly.parts if p.name == name)
            assert after.dimensions["length"] == pytest.approx(
                before.dimensions["length"] * 1.5
            )

    def test_subsystem_no_match(self, arm_assembly):
        req = ModificationRequest(
            scope="subsystem", intent="enlarge", target="nonexistent",
            params={"factor": 1.5}, source="user", raw_text="x",
        )
        new = apply_modification(arm_assembly, req)
        # Unchanged
        assert len(new.parts) == len(arm_assembly.parts)


# ---------------------------------------------------------------------------
# VLM feedback integration
# ---------------------------------------------------------------------------


class TestApplyTargetedFixFromVLM:
    def test_finger_overlap_fix(self, arm_assembly):
        problems = [
            "Parts 'gripper_finger_left' and 'gripper_finger_right' overlap "
            "by 38x41x23mm in their rotated world bounding boxes"
        ]
        new, applied = apply_targeted_fix_from_vlm(arm_assembly, problems)
        assert applied
        # Verify finger offset has been bumped
        left_after = next(j for j in new.joints if j.child == "gripper_finger_left")
        left_before = next(j for j in arm_assembly.joints
                           if j.child == "gripper_finger_left")
        # Left finger offset should be more negative on X (further from  # 2026-06-22 axis convention change: Y -> X
        # center) — the lateral/gap axis.
        assert left_after.offset[0] < left_before.offset[0]  # 2026-06-22 axis convention change: [1] -> [0]

    def test_gripper_invisible_fix(self, arm_assembly):
        # "Gripper invisible" is a rendering problem, NOT a geometry one.
        # Scaling the fingers up was deliberately removed because it caused a
        # runaway growth loop (60→96→153.6mm, see assembly_visual_verifier
        # GRIPPER_INVISIBLE handler). The correct behaviour is to NOT mutate
        # geometry here and let the LLM-regeneration fallback rewrite the
        # assembly. So this problem alone yields no deterministic fix.
        problems = [
            "End effector is a solid block, not a gripper with parallel prongs",
        ]
        new, applied = apply_targeted_fix_from_vlm(arm_assembly, problems)
        assert applied is False
        # Fingers must NOT have been scaled (the old runaway-growth path).
        finger_after = next(p for p in new.parts if p.name == "gripper_finger_left")
        finger_before = next(p for p in arm_assembly.parts
                             if p.name == "gripper_finger_left")
        assert finger_after.dimensions["length"] == finger_before.dimensions["length"]

    def test_multiple_problems_combined(self, arm_assembly):
        """The real production case: gripper invisible AND fingers overlap.

        The finger overlap IS deterministically fixable (finger_spread), so
        ``applied`` is True and the joints change. The gripper-invisible
        half does not scale the fingers (see test_gripper_invisible_fix)."""
        problems = [
            "End effector is a solid block",
            "Parts 'gripper_finger_left' and 'gripper_finger_right' "
            "overlap by 38x41x23mm",
        ]
        new, applied = apply_targeted_fix_from_vlm(arm_assembly, problems)
        assert applied
        diff = modifications_diff(arm_assembly, new)
        # The finger_spread correction moves the finger joints (offset change).
        assert len(diff["joints_changed"]) >= 1
        # Fingers are NOT scaled (no parts_changed from gripper-invisible).
        finger_names = {"gripper_finger_left", "gripper_finger_right"}
        scaled = [c for c in diff["parts_changed"] if c["name"] in finger_names]
        assert scaled == []

    def test_unknown_problem_no_action(self, arm_assembly):
        problems = ["the model is painted the wrong color"]
        new, applied = apply_targeted_fix_from_vlm(arm_assembly, problems)
        assert applied is False
        # Assembly unchanged
        assert len(new.parts) == len(arm_assembly.parts)

    def test_empty_problems(self, arm_assembly):
        new, applied = apply_targeted_fix_from_vlm(arm_assembly, [])
        assert applied is False

    def test_runaway_scaling_is_rejected(self, arm_assembly):
        """REGRESSION (production bug 2026-06-18):

        apply_targeted_fix_from_vlm was repeatedly applying ×1.6 scaling
        on every VLM round.  After 2 rounds a 30mm finger became 393mm
        (×13) and the whole assembly blew past the VTK camera frustum,
        producing blank renders.  The sanity guard must reject any
        correction that would grow a part by >3× or the total assembly
        by >2×.
        """
        # Simulate the runaway scenario: pre-scale the fingers once, then
        # demand another fix.  The guard should refuse.
        from lang3d.knowledge.mechanics import Assembly, Part, Joint
        big_parts = [
            Part(name="base_plate", category="structural", description="b",
                 dimensions={"length": 200, "width": 150, "height": 20}),
            Part(name="shoulder_link", category="link", description="s",
                 dimensions={"length": 120, "width": 25, "height": 20}),
            # Fingers already scaled to 5× original (runaway state)
            Part(name="gripper_finger_left", category="effector", description="",
                 dimensions={"length": 300, "width": 50, "height": 140}),
            Part(name="gripper_finger_right", category="effector", description="",
                 dimensions={"length": 300, "width": 50, "height": 140}),
        ]
        big_joints = [
            Joint(type="fixed", parent="base_plate", child="shoulder_link"),
            Joint(type="prismatic", parent="shoulder_link",
                  child="gripper_finger_left", offset=(0.0, -25.0, 0.0)),
            Joint(type="prismatic", parent="shoulder_link",
                  child="gripper_finger_right", offset=(0.0, 25.0, 0.0)),
        ]
        big_asm = Assembly(name="runaway", parts=big_parts, joints=big_joints)

        problems = [
            "End effector is a solid block, not a gripper",
        ]
        new, applied = apply_targeted_fix_from_vlm(big_asm, problems)
        # The guard should refuse — fingers are already huge.
        assert applied is False
        # Verify nothing changed
        new_finger = next(p for p in new.parts
                          if p.name == "gripper_finger_left")
        assert new_finger.dimensions["length"] == 300.0

    def test_runaway_joint_offset_is_rejected(self, arm_assembly):
        """REGRESSION: offsets >200mm must trigger the sanity guard.

        Production case: finger_spread pushed fingers to ±210mm offset,
        which put them outside the camera frustum.
        """
        from lang3d.knowledge.mechanics import Assembly, Joint
        # Plant a joint with an offset already near the limit
        runaway_joints = [
            Joint(type="prismatic", parent="shoulder_link",
                  child="gripper_finger_left", offset=(0.0, -180.0, 0.0)),
            Joint(type="prismatic", parent="shoulder_link",
                  child="gripper_finger_right", offset=(0.0, 180.0, 0.0)),
        ]
        runaway_asm = Assembly(
            name="runaway_offset",
            parts=arm_assembly.parts,
            joints=runaway_joints,
            description="",
        )
        # Asking for more spread on top of already-large offsets
        problems = [
            "Parts 'gripper_finger_left' and 'gripper_finger_right' "
            "overlap by 38x41x23mm"
        ]
        new, applied = apply_targeted_fix_from_vlm(runaway_asm, problems)
        # Either not applied (idempotent guard inside finger_spread),
        # or applied but sanity-checked — either way, the resulting
        # assembly must not have offsets beyond 200mm.
        for j in new.joints:
            off = j.offset or (0, 0, 0)
            for c in off:
                assert abs(c) <= 200.0, (
                    f"Joint {j.parent}->{j.child} offset {off} exceeds 200mm"
                )


# ---------------------------------------------------------------------------
# Problem text classifier
# ---------------------------------------------------------------------------


class TestClassifyProblemText:
    def test_finger_overlap(self, arm_assembly):
        p = classify_problem_text(
            "Parts 'gripper_finger_left' and 'gripper_finger_right' "
            "overlap by 38x41x23mm",
            arm_assembly,
        )
        assert p.problem_type == ProblemType.FINGER_OVERLAP
        assert "gripper_finger_left" in p.affected_parts
        assert "gripper_finger_right" in p.affected_parts
        assert p.correction["penetration_mm"] == 38.0

    def test_plate_overlap(self, arm_assembly):
        p = classify_problem_text(
            "Parts 'base_plate' and 'top_plate' are 88mm apart but have "
            "combined extent 580mm — likely overlapping",
            arm_assembly,
        )
        assert p.problem_type == ProblemType.PLATE_OVERLAP

    def test_gripper_invisible(self, arm_assembly):
        p = classify_problem_text(
            "End effector is a solid block, not a gripper",
            arm_assembly,
        )
        assert p.problem_type == ProblemType.GRIPPER_INVISIBLE

    def test_missing_part(self, arm_assembly):
        p = classify_problem_text(
            "No base plate found — critical parts missing",
            arm_assembly,
        )
        assert p.problem_type == ProblemType.MISSING_PART

    def test_unknown_falls_back(self, arm_assembly):
        p = classify_problem_text("some random complaint", arm_assembly)
        assert p.problem_type == ProblemType.UNREASONABLE_LAYOUT


# ---------------------------------------------------------------------------
# Diff
# ---------------------------------------------------------------------------


class TestModificationsDiff:
    def test_diff_detects_dim_changes(self, arm_assembly):
        req = ModificationRequest(
            scope="part", intent="enlarge", target="gripper_finger_left",
            params={"factor": 2.0}, source="user", raw_text="x",
        )
        new = apply_modification(arm_assembly, req)
        diff = modifications_diff(arm_assembly, new)
        changed_names = [c["name"] for c in diff["parts_changed"]]
        assert "gripper_finger_left" in changed_names

    def test_diff_detects_offset_changes(self, arm_assembly):
        problems = [
            "Parts 'gripper_finger_left' and 'gripper_finger_right' "
            "overlap by 38x41x23mm"
        ]
        new, _ = apply_targeted_fix_from_vlm(arm_assembly, problems)
        diff = modifications_diff(arm_assembly, new)
        # At least one joint should be flagged
        assert len(diff["joints_changed"]) >= 1


class TestFunctionalPartScalingGuard:
    """AGENTS.md §1.2: real COTS functional parts (servos/motors) must never be
    rescaled. The guard in _scale_part / _apply_reach / scale_part refuses to
    resize them; structural parts (links, fingers) still resize."""

    def test_servo_not_rescaled(self):
        """A servo (category=actuator) passed to _scale_part must come back
        UNCHANGED — its catalog dimensions are authoritative."""
        from lang3d.agent.modifier import _scale_part, _is_functional_part
        from lang3d.knowledge.mechanics import Part
        servo = Part(
            name="shoulder_servo", category="actuator",
            description="肩舵机MG996R", material="Steel",
            dimensions={"diameter": 41, "height": 43},
        )
        assert _is_functional_part(servo) is True
        scaled = _scale_part(servo, factor=2.0)
        assert scaled.dimensions == {"diameter": 41, "height": 43}, (
            "servo was rescaled — real COTS parts must keep catalog dims"
        )

    def test_structural_link_is_rescaled(self):
        """A structural link (category=link) must still resize."""
        from lang3d.agent.modifier import _scale_part, _is_functional_part
        from lang3d.knowledge.mechanics import Part
        link = Part(
            name="shoulder_link", category="link",
            description="肩连杆", material="Aluminum",
            dimensions={"length": 120, "width": 25, "height": 15},
        )
        assert _is_functional_part(link) is False
        scaled = _scale_part(link, factor=2.0)
        assert scaled.dimensions["length"] == 240

    def test_gripper_finger_is_rescaled(self):
        """Gripper fingers are designable structural parts (even though
        arm_topology categorises them 'mechanical') — they must resize, since
        the VLM-closeup fix enlarges invisible fingers."""
        from lang3d.agent.modifier import _is_functional_part
        from lang3d.knowledge.mechanics import Part
        finger = Part(
            name="gripper_finger_left", category="mechanical",
            description="左手指", material="PLA",
            dimensions={"length": 60, "width": 14, "height": 28},
        )
        assert _is_functional_part(finger) is False, (
            "fingers are structural/designable, not COTS actuators"
        )

    def test_servo_detected_by_description_model(self):
        """Even without category=actuator, a part whose description names a
        real servo model is treated as functional (defense for parts where the
        generator set an unusual category)."""
        from lang3d.agent.modifier import _is_functional_part
        from lang3d.knowledge.mechanics import Part
        servo = Part(
            name="wrist_motor", category="joint",
            description="腕部DS3218舵机", material="Steel",
            dimensions={"diameter": 40, "height": 39},
        )
        assert _is_functional_part(servo) is True

    def test_diff_no_changes(self, arm_assembly):
        # Apply a no-op modification (nonexistent part)
        req = ModificationRequest(
            scope="part", intent="enlarge", target="nonexistent",
            params={"factor": 2.0}, source="user", raw_text="x",
        )
        new = apply_modification(arm_assembly, req)
        diff = modifications_diff(arm_assembly, new)
        assert diff["parts_changed"] == []
        assert diff["joints_changed"] == []
