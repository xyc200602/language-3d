"""Tests for mesh collision detection (Task 63).

Tests cover:
- Bounding mesh creation from Part
- FCL collision detection (no collision, collision cases)
- Skip adjacent parts
- Tool registration and execution
- Assembly collision checking
"""

from __future__ import annotations

import json
import pytest

from lang3d.knowledge.mechanics import Assembly, Joint, Part, ROBOTIC_ARM_ASSEMBLY

# Check if FCL is available for these tests
try:
    import fcl as fcl_mod
    import trimesh
    HAS_FCL = True
except ImportError:
    HAS_FCL = False

pytestmark = pytest.mark.skipif(not HAS_FCL, reason="python-fcl not installed")


def _box(name, l, w, h):
    return Part(name, "structural", name, dimensions=dict(length=l, width=w, height=h))


def _cyl(name, d, h):
    return Part(name, "structural", name, dimensions=dict(diameter=d, height=h))


# ---------------------------------------------------------------------------
# Mesh creation tests
# ---------------------------------------------------------------------------

class TestMeshCreation:
    def test_box_mesh_from_box_part(self):
        from lang3d.tools.mesh_collision import MeshCollisionChecker
        checker = MeshCollisionChecker()
        part = _box("b", 100, 60, 10)
        mesh = checker.create_bounding_mesh(part)
        assert isinstance(mesh, trimesh.Trimesh)
        # Box should have extents approximately matching dimensions
        extents = mesh.bounding_box.extents
        assert abs(extents[0] - 100) < 1
        assert abs(extents[1] - 60) < 1
        assert abs(extents[2] - 10) < 1

    def test_box_mesh_from_cylinder_part(self):
        from lang3d.tools.mesh_collision import MeshCollisionChecker
        checker = MeshCollisionChecker()
        part = _cyl("c", 40, 20)
        mesh = checker.create_bounding_mesh(part)
        assert isinstance(mesh, trimesh.Trimesh)

    def test_box_mesh_from_part_with_diameter(self):
        from lang3d.tools.mesh_collision import MeshCollisionChecker
        checker = MeshCollisionChecker()
        part = Part("p", "s", "p", dimensions={"diameter": 120, "thickness": 8})
        mesh = checker.create_bounding_mesh(part)
        assert isinstance(mesh, trimesh.Trimesh)


# ---------------------------------------------------------------------------
# Collision detection tests
# ---------------------------------------------------------------------------

class TestCollisionDetection:
    def test_no_collision_well_separated(self):
        """Two parts far apart should not collide."""
        from lang3d.tools.mesh_collision import MeshCollisionChecker
        parts = [
            _box("a", 10, 10, 10),
            _box("b", 10, 10, 10),
        ]
        asm = Assembly(name="test", parts=parts, joints=[])
        placements = {
            "a": {"position": [0, 0, 0]},
            "b": {"position": [100, 0, 0]},
        }

        checker = MeshCollisionChecker()
        result = checker.check_assembly_collisions(asm, placements, skip_adjacent=False)
        assert result.collision_free
        assert len(result.pairs) == 1

    def test_collision_overlapping_parts(self):
        """Two overlapping parts should detect collision."""
        from lang3d.tools.mesh_collision import MeshCollisionChecker
        parts = [
            _box("a", 100, 100, 100),
            _box("b", 100, 100, 100),
        ]
        asm = Assembly(name="test", parts=parts, joints=[])
        # Both at same position -> should collide
        placements = {
            "a": {"position": [0, 0, 0]},
            "b": {"position": [0, 0, 0]},
        }

        checker = MeshCollisionChecker()
        result = checker.check_assembly_collisions(asm, placements, skip_adjacent=False)
        assert not result.collision_free
        assert result.pairs[0].is_collision

    def test_skip_adjacent_joints(self):
        """Adjacent parts (connected by joint) should be skipped."""
        from lang3d.tools.mesh_collision import MeshCollisionChecker
        parts = [
            _box("a", 100, 100, 100),
            _box("b", 100, 100, 100),
        ]
        joints = [Joint("fixed", "a", "b")]
        asm = Assembly(name="test", parts=parts, joints=joints)
        # Same position but skip_adjacent=True -> no check
        placements = {
            "a": {"position": [0, 0, 0]},
            "b": {"position": [0, 0, 0]},
        }

        checker = MeshCollisionChecker()
        result = checker.check_assembly_collisions(asm, placements, skip_adjacent=True)
        assert result.collision_free
        assert len(result.pairs) == 0  # pair was skipped


# ---------------------------------------------------------------------------
# Per-part BVH cache tests
#
# The cache (MeshCollisionChecker._bvh_cache) stores each part's FCL BVHModel
# keyed by (name, shrink_mm) so a joint-range search that re-checks the same
# assembly at many angles rebuilds BVHs once per part, not once per pair per
# trial. These tests pin the contract that caching is a PURE performance
# optimisation: it must not change any collision result.
# ---------------------------------------------------------------------------

class TestBVHCache:
    def test_cached_equals_uncached_result(self):
        """A second check (BVH served from cache) must equal the first.

        Proves the cache is correctness-preserving: the same assembly checked
        twice through one checker yields byte-identical collision verdicts,
        penetration depths, and pair counts. If a cache bug ever returned a
        stale/transform-mismatched BVH, this would diverge.
        """
        from lang3d.tools.mesh_collision import MeshCollisionChecker
        parts = [
            _box("base", 100, 80, 10),
            _box("arm", 20, 20, 60),
            _cyl("servo", 24, 35),
        ]
        asm = Assembly(name="cache_test", parts=parts, joints=[])
        placements = {
            "base": {"position": [0, 0, 0]},
            "arm": {"position": [0, 0, 35]},   # sits on the base
            "servo": {"position": [0, 0, 35]}, # overlaps arm -> collision
        }
        checker = MeshCollisionChecker()
        first = checker.check_assembly_collisions(asm, placements, skip_adjacent=False)
        # Second call: every part's BVH is now a cache hit.
        second = checker.check_assembly_collisions(asm, placements, skip_adjacent=False)
        assert len(second.pairs) == len(first.pairs)
        for p1, p2 in zip(first.pairs, second.pairs):
            assert p1.is_collision == p2.is_collision
            assert p1.penetration_depth_mm == p2.penetration_depth_mm

    def test_cache_populated_after_check(self):
        """The cache must actually be populated (not silently bypassed).

        After checking a 3-part assembly, the cache should hold an entry for
        each part touched. This guards against a future refactor that
        accidentally stops routing through _get_bvh (which would silently
        re-introduce the per-pair rebuild that caused the CI timeout).
        """
        from lang3d.tools.mesh_collision import MeshCollisionChecker
        parts = [_box("a", 10, 10, 10), _box("b", 10, 10, 10)]
        asm = Assembly(name="pop_test", parts=parts, joints=[])
        placements = {"a": {"position": [0, 0, 0]}, "b": {"position": [50, 0, 0]}}
        checker = MeshCollisionChecker()
        assert len(checker._bvh_cache) == 0
        checker.check_assembly_collisions(asm, placements, skip_adjacent=False)
        # Both parts should now have a cached BVH at shrink_mm=0.0.
        assert ("a", 0.0) in checker._bvh_cache
        assert ("b", 0.0) in checker._bvh_cache

    def test_cache_keyed_by_shrink_mm(self):
        """Different collision margins must yield separate BVHs.

        The cache key includes shrink_mm so a 0.5mm-margin check does not
        serve a BVH built for 0.0mm (which would under-report clearance).
        """
        from lang3d.tools.mesh_collision import MeshCollisionChecker
        part = _box("x", 40, 40, 40)
        asm = Assembly(name="key_test", parts=[part, _box("y", 40, 40, 40)], joints=[])
        placements = {"x": {"position": [0, 0, 0]}, "y": {"position": [0, 0, 0]}}
        checker = MeshCollisionChecker()
        checker.check_assembly_collisions(asm, placements, min_penetration_mm=0.0)
        checker.check_assembly_collisions(asm, placements, min_penetration_mm=0.5)
        # Both margins cached separately — no aliasing.
        assert ("x", 0.0) in checker._bvh_cache
        assert ("x", 0.5) in checker._bvh_cache


# ---------------------------------------------------------------------------
# Tool registration tests
# ---------------------------------------------------------------------------

class TestMeshCollisionTool:
    def test_tool_registration(self):
        from lang3d.tools.base import ToolRegistry
        from lang3d.tools.mesh_collision import register_mesh_collision_tools
        registry = ToolRegistry()
        register_mesh_collision_tools(registry)
        assert "mesh_collision_check" in registry.list_tools()

    def test_tool_definition(self):
        from lang3d.tools.mesh_collision import MeshCollisionTool
        tool = MeshCollisionTool()
        defn = tool.get_definition()
        assert defn.name == "mesh_collision_check"
        assert "assembly_name" in defn.parameters["properties"]

    def test_tool_execute_builtin(self):
        from lang3d.tools.mesh_collision import MeshCollisionTool
        tool = MeshCollisionTool()
        result = tool.execute(assembly_name="robotic_arm")
        assert "[Mesh Collision Check]" in result
        assert "Assembly: 3-DOF Robotic Arm" in result

    def test_tool_execute_unknown_assembly(self):
        from lang3d.tools.mesh_collision import MeshCollisionTool
        tool = MeshCollisionTool()
        result = tool.execute(assembly_name="nonexistent_xyz")
        assert "Error" in result

    def test_tool_execute_custom_json(self):
        from lang3d.tools.mesh_collision import MeshCollisionTool
        json_str = json.dumps({
            "name": "Custom",
            "parts": [
                {"name": "a", "dimensions": {"length": 10, "width": 10, "height": 10}},
                {"name": "b", "dimensions": {"length": 10, "width": 10, "height": 10}},
            ],
            "joints": [
                {"parent": "a", "child": "b", "type": "fixed",
                 "parent_anchor": "top", "child_anchor": "bottom"},
            ],
        })
        tool = MeshCollisionTool()
        result = tool.execute(assembly_json=json_str)
        assert "[Mesh Collision Check]" in result
        assert "Custom" in result


# ---------------------------------------------------------------------------
# Assembly integration test
# ---------------------------------------------------------------------------

class TestAssemblyIntegration:
    def test_robotic_arm_collision_check_runs(self):
        """Robotic arm assembly collision check completes and checks all parts."""
        from lang3d.tools.assembly_solver import AssemblySolver
        from lang3d.tools.mesh_collision import MeshCollisionChecker

        solver = AssemblySolver(ROBOTIC_ARM_ASSEMBLY)
        placements = solver.solve()

        checker = MeshCollisionChecker()
        result = checker.check_assembly_collisions(
            ROBOTIC_ARM_ASSEMBLY, placements, skip_adjacent=True,
        )
        # Check that the check completed for all parts
        assert result.parts_checked == len(ROBOTIC_ARM_ASSEMBLY.parts)
        # Should have some pairs checked (non-adjacent)
        assert result.pairs_checked >= 1
        # Result is a valid CollisionResult
        assert isinstance(result.collision_free, bool)
