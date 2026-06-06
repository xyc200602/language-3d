"""Tests for pipeline_context.py — AssemblyContext shared computation."""

from __future__ import annotations

import pytest

from lang3d.knowledge.mechanics import Assembly, Joint, Part
from lang3d.tools.pipeline_context import AssemblyContext


@pytest.fixture
def simple_assembly() -> Assembly:
    parts = [
        Part("base", "structural", "底板", dimensions=dict(length=100, width=80, height=5)),
        Part("pillar", "structural", "立柱", dimensions=dict(diameter=10, height=50)),
    ]
    joints = [
        Joint("fixed", "base", "pillar", parent_anchor="top", child_anchor="bottom"),
    ]
    return Assembly(name="Test Assembly", parts=parts, joints=joints)


class TestAssemblyContextInit:
    def test_default_state(self, simple_assembly):
        ctx = AssemblyContext(assembly=simple_assembly)
        assert ctx.solved is False
        assert ctx.mass_computed is False
        assert ctx.subsystems_built is False
        assert ctx.positions == {}
        assert ctx.mass_result == {}
        assert ctx.subsystems == {}


class TestEnsurePositions:
    def test_solves_positions(self, simple_assembly):
        ctx = AssemblyContext(assembly=simple_assembly)
        positions = ctx.ensure_positions()
        assert ctx.solved is True
        assert len(positions) == 2
        assert "base" in positions
        assert "pillar" in positions

    def test_cached_positions(self, simple_assembly):
        ctx = AssemblyContext(assembly=simple_assembly)
        pos1 = ctx.ensure_positions()
        pos2 = ctx.ensure_positions()
        assert pos1 is pos2  # same object, not recomputed


class TestEnsureMass:
    def test_computes_mass(self, simple_assembly):
        ctx = AssemblyContext(assembly=simple_assembly)
        mass = ctx.ensure_mass()
        assert ctx.mass_computed is True
        assert mass["total_mass_kg"] > 0

    def test_cached_mass(self, simple_assembly):
        ctx = AssemblyContext(assembly=simple_assembly)
        m1 = ctx.ensure_mass()
        m2 = ctx.ensure_mass()
        assert m1 is m2


class TestEnsureSubsystems:
    def test_builds_subsystems(self, simple_assembly):
        ctx = AssemblyContext(assembly=simple_assembly)
        ctx.ensure_positions()  # subsystems depend on positions
        subs = ctx.ensure_subsystems()
        assert ctx.subsystems_built is True
        assert isinstance(subs, dict)
        assert len(subs) > 0

    def test_cached_subsystems(self, simple_assembly):
        ctx = AssemblyContext(assembly=simple_assembly)
        ctx.ensure_positions()
        s1 = ctx.ensure_subsystems()
        s2 = ctx.ensure_subsystems()
        assert s1 is s2


class TestGetCom:
    def test_returns_com(self, simple_assembly):
        ctx = AssemblyContext(assembly=simple_assembly)
        com = ctx.get_com()
        assert len(com) == 3
        assert all(isinstance(v, float) for v in com)
