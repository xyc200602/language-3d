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
        """Centre-to-centre Y distance must clear the finger width."""
        asm = _parse_assembly_json(_production_gripper_json())
        asm = _normalize_gripper_fingers(asm)

        left = next(j for j in asm.joints if "finger_left" in j.child)
        right = next(j for j in asm.joints if "finger_right" in j.child)
        finger = next(p for p in asm.parts if "finger_left" in p.name)
        width = float(finger.dimensions.get("width", 14.0))

        gap = abs(right.offset[1] - left.offset[1])
        assert gap > width, (
            f"Y gap ({gap:.1f}mm) must exceed finger width ({width}mm) "
            f"or the AABBs overlap on the width axis"
        )

    def test_fingers_separated_on_width_axis(self):
        """The two fingers must differ on Y (width axis), not just on X."""
        asm = _parse_assembly_json(_production_gripper_json())
        asm = _normalize_gripper_fingers(asm)

        left = next(j for j in asm.joints if "finger_left" in j.child)
        right = next(j for j in asm.joints if "finger_right" in j.child)

        # They separate on Y...
        assert left.offset[1] != right.offset[1], (
            "fingers must have different Y offsets (the gap axis)"
        )
        # ...and the prismatic axis matches (Y), so closing actually grips.
        assert left.axis == "y" and right.axis == "y", (
            f"prismatic axis must be 'y', got left={left.axis}, right={right.axis}"
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
        """Start with fingers that DO intersect (tiny Y gap) and confirm
        finger_spread widens them past the non-intersection threshold."""
        asm = _parse_assembly_json(_production_gripper_json())
        asm = _normalize_gripper_fingers(asm)

        # Sabotage: force the fingers back together on Y (simulate the
        # pre-fix state where gap landed on the wrong axis).
        left = next(j for j in asm.joints if "finger_left" in j.child)
        right = next(j for j in asm.joints if "finger_right" in j.child)
        left.offset = (left.offset[0], -3.0, 0.0)   # 6mm gap < 14mm width
        right.offset = (right.offset[0], 3.0, 0.0)

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

        y_after_first = left.offset[1]

        _apply_finger_spread_to_joint(left, asm.parts)
        y_after_second = left.offset[1]

        assert y_after_first == y_after_second, (
            f"finger_spread is not idempotent: Y went {y_after_first} → {y_after_second}"
        )
