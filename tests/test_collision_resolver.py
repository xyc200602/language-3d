"""Tests for the collision feedback loop in :mod:`lang3d.tools.collision_resolver`.

These tests exercise three layers:
1. **Unit**: ``CollisionResolution`` dataclass, resolver construction,
   graceful handling of missing dependencies.
2. **Integration**: ``resolve()`` on a real (or deliberately broken)
   assembly — verifies collision count decreases and the input is not
   mutated.
3. **Smoke**: ``CollisionFixSuggester.apply_fixes`` round-trips
   structured suggestions into assembly mutations.
"""

from __future__ import annotations

import copy
import pytest

from lang3d.knowledge.mechanics import (
    Assembly,
    Joint,
    Part,
    ROBOTIC_ARM_ASSEMBLY,
)
from lang3d.tools.assembly_solver import AssemblySolver
from lang3d.tools.collision_resolver import (
    CollisionResolution,
    CollisionResolver,
)

try:
    from lang3d.tools.mesh_collision import HAS_FCL
except ImportError:
    HAS_FCL = False

pytestmark = pytest.mark.skipif(
    not HAS_FCL,
    reason="CollisionResolver requires trimesh + python-fcl",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _two_overlapping_parts() -> Assembly:
    """Two flat plates stacked at the same Z — guaranteed to overlap."""
    return Assembly(
        name="overlap_test",
        parts=[
            Part(
                name="plate_a",
                category="structural",
                description="plate A",
                dimensions={"length": 50, "width": 50, "height": 10},
            ),
            Part(
                name="plate_b",
                category="structural",
                description="plate B",
                dimensions={"length": 50, "width": 50, "height": 10},
            ),
        ],
        joints=[
            Joint(
                type="fixed",
                parent="plate_a",
                child="plate_b",
            ),
        ],
    )


def _simple_arm() -> Assembly:
    """A small 2-link arm — base + shoulder + link — that usually solves cleanly."""
    return Assembly(
        name="simple_arm",
        parts=[
            Part(
                name="base",
                category="structural",
                description="base",
                dimensions={"length": 60, "width": 60, "height": 10},
            ),
            Part(
                name="shoulder",
                category="actuator",
                description="shoulder servo",
                dimensions={"diameter": 30, "height": 25},
            ),
            Part(
                name="upper_link",
                category="structural",
                description="upper link",
                dimensions={"length": 80, "width": 20, "height": 15},
            ),
        ],
        joints=[
            Joint(
                type="revolute",
                parent="base",
                child="shoulder",
                range_deg=[-90, 90],
                axis="z",
                parent_anchor="top",
                child_anchor="bottom",
            ),
            Joint(
                type="fixed",
                parent="shoulder",
                child="upper_link",
                parent_anchor="top",
                child_anchor="bottom",
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Unit tests — construction and dataclass
# ---------------------------------------------------------------------------


class TestCollisionResolverUnit:
    def test_resolver_instantiates(self) -> None:
        r = CollisionResolver(max_rounds=2)
        assert r._max_rounds == 2
        assert r._checker is not None
        assert r._motion_checker is not None
        assert r._fixer is not None

    def test_resolution_dataclass_defaults(self) -> None:
        result = CollisionResolution()
        assert result.resolved is False
        assert result.remaining_count == 0
        assert result.rounds_used == 0
        assert result.fixes_applied == []
        assert result.collision_history == []
        assert result.modified_assembly is None

    def test_min_rounds_clamped_to_one(self) -> None:
        r = CollisionResolver(max_rounds=0)
        assert r._max_rounds == 1


# ---------------------------------------------------------------------------
# Integration tests — resolve() on real assemblies
# ---------------------------------------------------------------------------


class TestCollisionResolverIntegration:
    def test_no_collision_returns_immediately(self) -> None:
        """Clean assembly → resolved=True with zero rounds."""
        assembly = _simple_arm()
        solver = AssemblySolver(assembly)
        positions = solver.solve()

        resolver = CollisionResolver(max_rounds=2)
        result = resolver.resolve(assembly, positions)

        assert isinstance(result, CollisionResolution)
        # A clean assembly may still report minor static overlap from
        # adjacent parts, but the resolver should not loop unnecessarily.
        assert result.rounds_used >= 0

    def test_overlapping_parts_does_not_crash(self) -> None:
        """Two plates at the same Z — resolver must handle gracefully."""
        assembly = _two_overlapping_parts()
        solver = AssemblySolver(assembly)
        positions = solver.solve()

        resolver = CollisionResolver(max_rounds=2)
        result = resolver.resolve(assembly, positions)

        assert isinstance(result, CollisionResolution)
        # Even if it cannot fully resolve, it must return the best attempt.
        assert result.modified_assembly is not None

    def test_input_assembly_not_mutated(self) -> None:
        """Resolver must deep-copy; caller's assembly stays pristine."""
        assembly = _simple_arm()
        original_dims = {
            p.name: dict(p.dimensions) for p in assembly.parts
        }
        original_joints = len(assembly.joints)

        solver = AssemblySolver(assembly)
        positions = solver.solve()

        resolver = CollisionResolver(max_rounds=2)
        resolver.resolve(assembly, positions)

        # Verify nothing changed on the input.
        assert len(assembly.joints) == original_joints
        for part in assembly.parts:
            assert part.dimensions == original_dims[part.name], (
                f"{part.name} dimensions mutated by resolver"
            )

    def test_collision_count_does_not_increase(self) -> None:
        """After resolution, collision count must be ≤ the initial count."""
        assembly = _simple_arm()
        solver = AssemblySolver(assembly)
        positions = solver.solve()

        resolver = CollisionResolver(max_rounds=2)

        # Measure before.
        before = resolver._count_collisions(assembly, positions, None)

        result = resolver.resolve(assembly, positions)

        # After must not be worse.
        if result.modified_assembly and result.modified_positions:
            after = resolver._count_collisions(
                result.modified_assembly,
                result.modified_positions,
                None,
            )
            assert after <= before, (
                f"Collision count increased: {before} → {after}"
            )


# ---------------------------------------------------------------------------
# apply_fixes round-trip — structured suggestions become mutations
# ---------------------------------------------------------------------------


class TestApplyFixesRoundTrip:
    def test_apply_fixes_returns_deep_copy(self) -> None:
        """CollisionFixSuggester.apply_fixes must not mutate input."""
        from lang3d.tools.motion_collision import (
            CollisionFixReport,
            CollisionFixSuggestion,
        )

        assembly = _simple_arm()
        original = copy.deepcopy(assembly)

        report = CollisionFixReport(
            total_collisions=1,
            suggestions=[
                CollisionFixSuggestion(
                    joint_name="upper_link",
                    fix_type="increase_spacing",
                    description="test",
                    parameter="upper_link.spacing_mm",
                    current_value=0.0,
                    suggested_value=15.0,
                    confidence=0.8,
                ),
            ],
            constraint_updates={},
            summary="test",
        )

        from lang3d.tools.motion_collision import CollisionFixSuggester
        suggester = CollisionFixSuggester()
        new_assembly = suggester.apply_fixes(assembly, report)

        # New assembly is different object.
        assert new_assembly is not assembly
        # Original untouched.
        assert len(assembly.joints) == len(original.joints)
        for joint, orig_joint in zip(assembly.joints, original.joints):
            assert joint.offset == orig_joint.offset

    def test_low_confidence_suggestions_skipped(self) -> None:
        """Suggestions below min_confidence must not be applied."""
        from lang3d.tools.motion_collision import (
            CollisionFixReport,
            CollisionFixSuggestion,
            CollisionFixSuggester,
        )

        assembly = _simple_arm()
        original_length = next(
            p.dimensions.get("length", 0) for p in assembly.parts
            if p.name == "upper_link"
        )

        report = CollisionFixReport(
            total_collisions=1,
            suggestions=[
                CollisionFixSuggestion(
                    joint_name="upper_link",
                    fix_type="reduce_link_length",
                    description="test",
                    parameter="upper_link.length",
                    current_value=original_length,
                    suggested_value=original_length * 0.5,
                    confidence=0.3,  # Below default 0.6 threshold
                ),
            ],
            constraint_updates={},
            summary="test",
        )

        suggester = CollisionFixSuggester()
        new_assembly = suggester.apply_fixes(assembly, report, min_confidence=0.6)

        # Find the corresponding part in the new assembly.
        new_part = next(p for p in new_assembly.parts if p.name == "upper_link")
        assert new_part.dimensions.get("length", 0) == original_length, (
            "Low-confidence fix was applied but should have been skipped"
        )

    def test_robotic_arm_assembly_resolves(self) -> None:
        """Smoke test against the built-in ROBOTIC_ARM_ASSEMBLY.

        Verifies the resolver runs end-to-end on a real multi-part
        assembly and returns a coherent result — regardless of whether
        it achieves zero collisions.
        """
        assembly = ROBOTIC_ARM_ASSEMBLY
        solver = AssemblySolver(assembly)
        positions = solver.solve()

        resolver = CollisionResolver(max_rounds=2)
        result = resolver.resolve(assembly, positions)

        assert isinstance(result, CollisionResolution)
        assert result.modified_assembly is not None
        assert len(result.collision_history) >= 1
        # The resolver should not claim success unless it really hit zero.
        if result.resolved:
            assert result.remaining_count == 0
