"""Tests for VTK offscreen renderer."""

import json
import os
import tempfile
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_vtk():
    try:
        import vtk  # noqa: F401
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(not _has_vtk(), reason="VTK not installed")


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def renderer():
    from lang3d.tools.vtk_renderer import VTKOffscreenRenderer
    return VTKOffscreenRenderer(width=400, height=300)


# ---------------------------------------------------------------------------
# Test: VTKOffscreenRenderer basic operations
# ---------------------------------------------------------------------------


class TestVTKRendererBasic:
    def test_empty_renderer_raises(self, renderer):
        """Rendering without adding geometry should raise."""
        with pytest.raises(RuntimeError, match="No geometry"):
            renderer.render_to_file("isometric", "/tmp/should_not_exist.png")

    def test_add_box(self, renderer, tmp_dir):
        renderer.add_box(100, 60, 10, color=(0.2, 0.5, 0.85))
        path = renderer.render_to_file("isometric", os.path.join(tmp_dir, "box.png"))
        assert os.path.isfile(path)
        assert os.path.getsize(path) > 1000

    def test_add_cylinder(self, renderer, tmp_dir):
        renderer.add_cylinder(radius=10, height=50, color=(0.85, 0.3, 0.2))
        path = renderer.render_to_file("isometric", os.path.join(tmp_dir, "cyl.png"))
        assert os.path.isfile(path)
        assert os.path.getsize(path) > 1000

    def test_add_box_and_cylinder(self, renderer, tmp_dir):
        renderer.add_box(100, 60, 10, color=(0.2, 0.5, 0.85), position=(0, 0, 0))
        renderer.add_cylinder(5, 50, color=(0.85, 0.3, 0.2), position=(30, 20, 30))
        path = renderer.render_to_file("front", os.path.join(tmp_dir, "combo.png"))
        assert os.path.isfile(path)
        assert os.path.getsize(path) > 1000

    def test_add_axes(self, renderer, tmp_dir):
        renderer.add_box(20, 20, 20)
        renderer.add_axes(length=10)
        path = renderer.render_to_file("isometric", os.path.join(tmp_dir, "axes.png"))
        assert os.path.isfile(path)

    def test_clear(self, renderer, tmp_dir):
        renderer.add_box(20, 20, 20)
        renderer.clear()
        assert not renderer._has_content
        with pytest.raises(RuntimeError):
            renderer.render_to_file("isometric", os.path.join(tmp_dir, "cleared.png"))

    def test_reuse_after_clear(self, renderer, tmp_dir):
        renderer.add_box(20, 20, 20, color=(1, 0, 0))
        renderer.clear()
        renderer.add_cylinder(10, 30, color=(0, 0, 1))
        path = renderer.render_to_file("isometric", os.path.join(tmp_dir, "reuse.png"))
        assert os.path.isfile(path)


# ---------------------------------------------------------------------------
# Test: Multi-angle rendering
# ---------------------------------------------------------------------------


class TestMultiAngle:
    def test_render_all_views_default(self, renderer, tmp_dir):
        renderer.add_box(100, 80, 10)
        paths = renderer.render_all_views(tmp_dir)
        assert len(paths) == 4
        assert all(os.path.isfile(p) for p in paths)
        # Check all views have different file sizes (they should differ)
        sizes = [os.path.getsize(p) for p in paths]
        assert len(set(sizes)) > 1, "All views produced identical images"

    def test_render_specific_views(self, renderer, tmp_dir):
        renderer.add_box(50, 50, 50)
        paths = renderer.render_all_views(tmp_dir, views=["isometric", "top"])
        assert len(paths) == 2
        assert "isometric" in paths[0]
        assert "top" in paths[1]

    def test_render_with_prefix(self, renderer, tmp_dir):
        renderer.add_box(30, 30, 30)
        paths = renderer.render_all_views(tmp_dir, prefix="test_part")
        for p in paths:
            assert "test_part_" in os.path.basename(p)

    def test_all_four_views_produce_valid_pngs(self, renderer, tmp_dir):
        from lang3d.tools.vtk_renderer import VIEW_PRESETS
        renderer.add_box(80, 60, 20)
        for view_name in VIEW_PRESETS:
            path = renderer.render_to_file(view_name, os.path.join(tmp_dir, f"{view_name}.png"))
            assert os.path.isfile(path)
            assert os.path.getsize(path) > 500


# ---------------------------------------------------------------------------
# Test: STL loading
# ---------------------------------------------------------------------------


class TestSTLLoading:
    def test_load_stl_file(self, renderer, tmp_dir):
        # Find any STL in the test data
        stl_candidates = list(Path("data/projects").glob("*.stl"))
        if not stl_candidates:
            pytest.skip("No STL files in data/projects")

        stl_path = str(stl_candidates[0])
        renderer.load_stl(stl_path, color=(0.2, 0.5, 0.85))
        paths = renderer.render_all_views(os.path.join(tmp_dir, "stl"))
        assert len(paths) == 4
        for p in paths:
            assert os.path.getsize(p) > 2000

    def test_load_stl_not_found_raises(self, renderer):
        with pytest.raises(FileNotFoundError):
            renderer.load_stl("/nonexistent/path.stl")

    def test_load_stl_with_position(self, renderer, tmp_dir):
        stl_candidates = list(Path("data/projects").glob("*.stl"))
        if not stl_candidates:
            pytest.skip("No STL files")

        renderer.load_stl(str(stl_candidates[0]), position=(50, 50, 0))
        path = renderer.render_to_file("isometric", os.path.join(tmp_dir, "offset.png"))
        assert os.path.isfile(path)


# ---------------------------------------------------------------------------
# Test: Convenience functions
# ---------------------------------------------------------------------------


class TestConvenienceFunctions:
    def test_render_stl_multi_angle(self, tmp_dir):
        from lang3d.tools.vtk_renderer import render_stl_multi_angle

        stl_candidates = list(Path("data/projects").glob("*.stl"))
        if not stl_candidates:
            pytest.skip("No STL files")

        paths = render_stl_multi_angle(str(stl_candidates[0]), tmp_dir)
        assert len(paths) == 4

    def test_render_assembly_from_positions(self, tmp_dir):
        from lang3d.tools.vtk_renderer import render_assembly_from_positions

        parts = [
            {"name": "base", "dimensions": {"length": 100, "width": 80, "height": 5}, "category": "chassis"},
            {"name": "pillar", "dimensions": {"outer_diameter": 8, "height": 50}},
        ]
        positions = {
            "base": {"position": [0, 0, 0]},
            "pillar": {"position": [35, 25, 27.5]},
        }
        paths = render_assembly_from_positions(parts, positions, tmp_dir)
        assert len(paths) == 4
        for p in paths:
            assert os.path.isfile(p)

    def test_render_assembly_with_stl_dir(self, tmp_dir):
        from lang3d.tools.vtk_renderer import render_assembly_from_positions

        # Use data/projects as STL dir — most parts won't match, that's fine
        parts = [
            {"name": "base_plate", "dimensions": {"length": 300, "width": 200, "height": 5}, "category": "chassis"},
        ]
        positions = {"base_plate": {"position": [0, 0, 0]}}
        paths = render_assembly_from_positions(
            parts, positions, tmp_dir, stl_dir="data/projects"
        )
        assert len(paths) == 4


# ---------------------------------------------------------------------------
# Test: View presets
# ---------------------------------------------------------------------------


class TestViewPresets:
    def test_presets_have_required_keys(self):
        from lang3d.tools.vtk_renderer import VIEW_PRESETS
        for name, preset in VIEW_PRESETS.items():
            assert "position" in preset, f"{name} missing position"
            assert "focal" in preset, f"{name} missing focal"
            assert "up" in preset, f"{name} missing up"
            assert len(preset["position"]) == 3
            assert len(preset["focal"]) == 3
            assert len(preset["up"]) == 3

    def test_four_standard_views(self):
        from lang3d.tools.vtk_renderer import VIEW_PRESETS
        assert set(VIEW_PRESETS.keys()) == {"isometric", "front", "top", "right"}

    def test_isometric_has_positive_z_up(self):
        from lang3d.tools.vtk_renderer import VIEW_PRESETS
        iso = VIEW_PRESETS["isometric"]
        assert iso["up"] == (0, 0, 1), "Z-up coordinate system expected"


# ---------------------------------------------------------------------------
# Test: Color generation
# ---------------------------------------------------------------------------


class TestColorGeneration:
    def test_default_colors_are_valid(self):
        from lang3d.tools.vtk_renderer import _default_color
        for i in range(20):
            c = _default_color(i)
            assert len(c) == 3
            assert all(0 <= v <= 1 for v in c), f"Color {i}: {c} out of range"

    def test_different_indices_give_different_colors(self):
        from lang3d.tools.vtk_renderer import _default_color
        colors = [_default_color(i) for i in range(10)]
        # At least 8 should be distinct
        unique = set(colors)
        assert len(unique) >= 8

    def test_subsystem_colors_valid(self):
        from lang3d.tools.vtk_renderer import SUBSYSTEM_COLORS
        for name, c in SUBSYSTEM_COLORS.items():
            assert len(c) == 3, f"{name}: expected 3-tuple"
            assert all(0 <= v <= 1 for v in c), f"{name}: {c} out of range"


# ---------------------------------------------------------------------------
# Test: Floor grid
# ---------------------------------------------------------------------------


class TestFloorGrid:
    def test_floor_grid_renders(self, renderer, tmp_dir):
        renderer.add_box(50, 50, 10)
        renderer.add_floor_grid(size=200, spacing=50)
        path = renderer.render_to_file("isometric", os.path.join(tmp_dir, "grid.png"))
        assert os.path.isfile(path)
        # Should be bigger than without grid (more geometry)
        assert os.path.getsize(path) > 1000


# ---------------------------------------------------------------------------
# Test: Image dimensions
# ---------------------------------------------------------------------------


class TestImageDimensions:
    def test_custom_resolution(self, tmp_dir):
        from lang3d.tools.vtk_renderer import VTKOffscreenRenderer
        r = VTKOffscreenRenderer(width=800, height=600)
        r.add_box(30, 30, 30)
        path = r.render_to_file("isometric", os.path.join(tmp_dir, "custom.png"))
        assert os.path.isfile(path)

    def test_large_resolution(self, tmp_dir):
        from lang3d.tools.vtk_renderer import VTKOffscreenRenderer
        r = VTKOffscreenRenderer(width=1920, height=1080)
        r.add_box(30, 30, 30)
        path = r.render_to_file("isometric", os.path.join(tmp_dir, "large.png"))
        assert os.path.isfile(path)
        # Larger resolution should produce bigger file
        assert os.path.getsize(path) > 5000
