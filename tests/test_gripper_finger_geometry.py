"""Regression tests for gripper finger geometry.

These guard the two bugs that made the 4dof_arm VLM verification loop fail
for 3 rounds straight (FAILED_MAX_ROUNDS):

  1. Finger gap was placed on the LENGTH axis (X), so two 60 mm bars spaced
     32 mm apart overlapped by ~28 mm along their length.
  2. The VLM ``finger_spread`` correction never actually ran — wrong offset
     component + an idempotency guard that was always false.

The tests use the PRODUCTION finger dimensions (length=60, width=14) which
no existing test covered — that is precisely why the bug slipped through.
See AGENTS.md §5 (visual verification) and §3.2 (gripper is a hot zone).
"""

from __future__ import annotations

import json

import pytest

from lang3d.tools.assembly_generator import (
    _geometric_prevalidation,
    _normalize_gripper_fingers,
    _parse_assembly_json,
)
from lang3d.tools.assembly_solver import AssemblySolver


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _production_gripper_json() -> str:
    """A gripper assembly using the real 4dof_arm finger dimensions.

    Mirrors data/runs/4dof_arm/.../assembly.json:
      gripper_base  : 28 x 50 x 32 mm
      fingers       : 60 x 14 x 28 mm  (length=60 is the long bar)
    """
    return json.dumps({
        "name": "gripper_production",
        "parts": [
            {
                "name": "gripper_base",
                "category": "mechanical",
                "description": "gripper base",
                "material": "PLA",
                "dimensions": {"length": 28, "width": 50, "height": 32},
            },
            {
                "name": "gripper_finger_left",
                "category": "mechanical",
                "description": "left finger",
                "material": "PLA",
                "dimensions": {"length": 60, "width": 14, "height": 28},
            },
            {
                "name": "gripper_finger_right",
                "category": "mechanical",
                "description": "right finger",
                "material": "PLA",
                "dimensions": {"length": 60, "width": 14, "height": 28},
            },
        ],
        "joints": [
            {
                "type": "prismatic", "parent": "gripper_base",
                "child": "gripper_finger_left", "axis": "x",
                "parent_anchor": "center", "child_anchor": "center",
                "range_deg": [0, 20],
            },
            {
                "type": "prismatic", "parent": "gripper_base",
                "child": "gripper_finger_right", "axis": "x",
                "parent_anchor": "center", "child_anchor": "center",
                "range_deg": [0, 20],
            },
        ],
    })


def _solve_and_validate(asm):
    """Run solver + geometric prevalidation, return problem list."""
    positions = AssemblySolver(asm).solve()
    parts_dict = [{"name": p.name, "dimensions": p.dimensions} for p in asm.parts]
    joints_dict = [{"parent": j.parent, "child": j.child} for j in asm.joints]
    return _geometric_prevalidation(parts_dict, positions, joints_dict)


# ---------------------------------------------------------------------------
# Production-size finger non-intersection (the core regression)
# ---------------------------------------------------------------------------


class TestProductionFingersDoNotIntersect:
    """With real dimensions (length=60, width=14), the sanitised + solved
    fingers must NOT intersect in the geometric prevalidation."""

    def test_no_finger_overlap_at_production_size(self):
        asm = _parse_assembly_json(_production_gripper_json())
        asm = _normalize_gripper_fingers(asm)
        problems = _solve_and_validate(asm)

        finger_problems = [
            p for p in problems
            if "finger" in p.lower() and "overlap" in p.lower()
        ]
        assert finger_problems == [], (
            "Production-size fingers still intersect after sanitize+solve:\n"
            + "\n".join(finger_problems)
        )

    def test_finger_gap_exceeds_width(self):
        """Centre-to-centre X distance must clear the finger width."""  # 2026-06-22 axis convention change: Y -> X
        asm = _parse_assembly_json(_production_gripper_json())
        asm = _normalize_gripper_fingers(asm)

        left = next(j for j in asm.joints if "finger_left" in j.child)
        right = next(j for j in asm.joints if "finger_right" in j.child)
        finger = next(p for p in asm.parts if "finger_left" in p.name)
        width = float(finger.dimensions.get("width", 14.0))

        gap = abs(right.offset[0] - left.offset[0])  # 2026-06-22 axis convention change: gap on X, not Y
        assert gap > width, (
            f"X gap ({gap:.1f}mm) must exceed finger width ({width}mm) "
            f"or the AABBs overlap on the lateral axis"
        )

    def test_fingers_separated_on_width_axis(self):
        """The two fingers must differ on X (lateral axis), not just on Y."""  # 2026-06-22 axis convention change
        asm = _parse_assembly_json(_production_gripper_json())
        asm = _normalize_gripper_fingers(asm)

        left = next(j for j in asm.joints if "finger_left" in j.child)
        right = next(j for j in asm.joints if "finger_right" in j.child)

        # They separate on X...  # 2026-06-22 axis convention change: Y -> X
        assert left.offset[0] != right.offset[0], (
            "fingers must have different X offsets (the gap axis)"  # 2026-06-22 axis convention change
        )
        # ...and the prismatic axis matches (X), so closing actually grips.  # 2026-06-22 axis convention change
        assert left.axis == "x" and right.axis == "x", (  # 2026-06-22 axis convention change: 'y' -> 'x'
            f"prismatic axis must be 'x', got left={left.axis}, right={right.axis}"
        )


# ---------------------------------------------------------------------------
# finger_spread correction actually fires
# ---------------------------------------------------------------------------


class TestFingerSpreadCorrection:
    """The VLM-loop finger_spread correction must actually modify offsets.

    Before the fix it never ran: it compared abs(cur[1]) < target where
    cur[1] was the forward offset (-44), so the guard was always false and
    the correction was a no-op for 3 VLM rounds.
    """

    def test_correction_pushes_intersecting_fingers_apart(self):
        """Start with fingers that DO intersect (tiny X gap) and confirm  # 2026-06-22 axis convention change
        finger_spread widens them past the non-intersection threshold."""
        asm = _parse_assembly_json(_production_gripper_json())
        asm = _normalize_gripper_fingers(asm)

        # Sabotage: force the fingers back together on X (simulate the  # 2026-06-22 axis convention change: Y -> X
        # pre-fix state where gap landed on the wrong axis).
        left = next(j for j in asm.joints if "finger_left" in j.child)
        right = next(j for j in asm.joints if "finger_right" in j.child)
        left.offset = (-3.0, left.offset[1], 0.0)   # 6mm gap < 14mm width  # 2026-06-22 axis convention change
        right.offset = (3.0, right.offset[1], 0.0)  # 2026-06-22 axis convention change

        # Sanity check: they DO intersect before correction.
        pre_problems = _solve_and_validate(asm)
        assert any("finger" in p.lower() and "overlap" in p.lower()
                   for p in pre_problems), "test setup: fingers should intersect"

        # Apply the finger_spread correction the same way the verifier does.
        from lang3d.agent.assembly_visual_verifier import (
            _apply_finger_spread_to_joint,
        )
        _apply_finger_spread_to_joint(left, asm.parts)
        _apply_finger_spread_to_joint(right, asm.parts)

        # After correction: no intersection.
        post_problems = _solve_and_validate(asm)
        finger_overlap = [p for p in post_problems
                          if "finger" in p.lower() and "overlap" in p.lower()]
        assert finger_overlap == [], (
            "finger_spread did not resolve intersection:\n"
            + "\n".join(finger_overlap)
        )

    def test_correction_is_idempotent(self):
        """Applying finger_spread twice must not push fingers further —
        once they're past the threshold, the guard skips them."""
        asm = _parse_assembly_json(_production_gripper_json())
        asm = _normalize_gripper_fingers(asm)

        from lang3d.agent.assembly_visual_verifier import (
            _apply_finger_spread_to_joint,
        )
        left = next(j for j in asm.joints if "finger_left" in j.child)
        right = next(j for j in asm.joints if "finger_right" in j.child)
        _apply_finger_spread_to_joint(left, asm.parts)
        _apply_finger_spread_to_joint(right, asm.parts)

        x_after_first = left.offset[0]  # 2026-06-22 axis convention change: spread pushes X, not Y

        _apply_finger_spread_to_joint(left, asm.parts)
        x_after_second = left.offset[0]  # 2026-06-22 axis convention change

        assert x_after_first == x_after_second, (
            f"finger_spread is not idempotent: X went {x_after_first} → {x_after_second}"  # 2026-06-22 axis convention change
        )


# ---------------------------------------------------------------------------
# Forward direction depends on parent-chain attachment (the wrist_link fix)
# ---------------------------------------------------------------------------


def _gripper_with_parent_json(parent_anchor: str, child_anchor: str) -> str:
    """A gripper assembly that has a parent link (wrist_link) attached.

    Mirrors the 4dof_arm topology: wrist_link -> gripper_base -> fingers.
    The wrist_link attaches to gripper_base on ``child_anchor`` face; the
    fingers must point the OTHER way so they don't stab back into the arm.
    """
    return json.dumps({
        "name": "gripper_with_parent",
        "parts": [
            {"name": "wrist_link", "category": "link",
             "description": "wrist", "material": "PLA",
             "dimensions": {"length": 60, "width": 20, "height": 14}},
            {"name": "gripper_base", "category": "mechanical",
             "description": "gripper base", "material": "PLA",
             "dimensions": {"length": 28, "width": 50, "height": 32}},
            {"name": "gripper_finger_left", "category": "mechanical",
             "description": "left finger", "material": "PLA",
             "dimensions": {"length": 60, "width": 14, "height": 28}},
            {"name": "gripper_finger_right", "category": "mechanical",
             "description": "right finger", "material": "PLA",
             "dimensions": {"length": 60, "width": 14, "height": 28}},
        ],
        "joints": [
            {"type": "fixed", "parent": "wrist_link", "child": "gripper_base",
             "parent_anchor": parent_anchor, "child_anchor": child_anchor},
            {"type": "prismatic", "parent": "gripper_base",
             "child": "gripper_finger_left", "axis": "x",
             "parent_anchor": "center", "child_anchor": "center",
             "range_deg": [0, 20]},
            {"type": "prismatic", "parent": "gripper_base",
             "child": "gripper_finger_right", "axis": "x",
             "parent_anchor": "center", "child_anchor": "center",
             "range_deg": [0, 20]},
        ],
    })


class TestForwardDirection:
    """The finger forward direction must follow the parent-chain attachment.

    Regression for the wrist_link/finger intersection: when the arm attaches
    on the gripper base's 'back' face, fingers must point to 'front' (-Y),
    not back into the arm (+Y).  Forward is along Y (solver front/back axis).
    """

    def test_fingers_point_away_when_parent_on_back(self):
        """Parent (wrist) on 'back' face -> fingers point -Y (front)."""  # 2026-06-22 axis convention change: +X -> -Y
        asm = _parse_assembly_json(_gripper_with_parent_json("front", "back"))
        asm = _normalize_gripper_fingers(asm)
        left = next(j for j in asm.joints if "finger_left" in j.child)
        # forward_y must be negative (fingers away from the back-mounted arm)  # 2026-06-22 axis convention change
        assert left.offset[1] < 0, (  # 2026-06-22 axis convention change: offset[0] > 0 -> offset[1] < 0
            f"fingers must point -Y when parent is on 'back' face; "
            f"got offset[1]={left.offset[1]} (this drives fingers into the arm)"
        )

    def test_fingers_flip_when_parent_on_front(self):
        """Parent (wrist) on 'front' face -> fingers point +Y (back)."""  # 2026-06-22 axis convention change: -X -> +Y
        asm = _parse_assembly_json(_gripper_with_parent_json("back", "front"))
        asm = _normalize_gripper_fingers(asm)
        left = next(j for j in asm.joints if "finger_left" in j.child)
        assert left.offset[1] > 0, (  # 2026-06-22 axis convention change: offset[0] < 0 -> offset[1] > 0
            f"fingers must point +Y when parent is on 'front' face; "
            f"got offset[1]={left.offset[1]}"
        )

    def test_no_intersection_with_wrist_link(self):
        """End-to-end: the 4dof topology (wrist->base->fingers, back-mount)
        must NOT produce a wrist_link/finger intersection after solving."""
        asm = _parse_assembly_json(_gripper_with_parent_json("front", "back"))
        asm = _normalize_gripper_fingers(asm)
        problems = _solve_and_validate(asm)
        wrist_finger = [
            p for p in problems
            if "wrist" in p.lower() and "finger" in p.lower()
            and "overlap" in p.lower()
        ]
        assert wrist_finger == [], (
            "wrist_link/finger still intersect after forward-direction fix:\n"
            + "\n".join(wrist_finger)
        )
