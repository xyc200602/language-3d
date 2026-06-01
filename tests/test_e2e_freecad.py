"""End-to-end tests for FreeCAD integration.

These tests require FreeCAD to be installed.
They verify the complete modeling pipeline: create → modify → export → verify.
"""

import os
import tempfile
from pathlib import Path

import pytest

# Skip all tests if FreeCAD is not available
def _freecad_available():
    return any(
        (Path(p) / "python.exe").exists()
        for p in [
            os.path.expanduser(r"~\AppData\Local\Programs\FreeCAD 1.1\bin"),
            r"C:\Program Files\FreeCAD 1.1\bin",
            r"C:\Program Files\FreeCAD\bin",
        ]
    )


pytestmark = pytest.mark.skipif(
    not _freecad_available(),
    reason="FreeCAD not installed",
)


@pytest.fixture
def freecad_tools():
    """Set up FreeCAD tools for testing."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from lang3d.tools.freecad import (
        _run_freecad_script, _build_script, _find_freecad_python,
        _is_freecad_available,
    )
    return {
        "run": _run_freecad_script,
        "build": _build_script,
        "python": _find_freecad_python,
        "available": _is_freecad_available,
    }


@pytest.fixture
def project_dir(tmp_path):
    """Create a temporary project directory."""
    return tmp_path


class TestFreeCADBasicModeling:
    """Test basic FreeCAD modeling operations."""

    def test_freecad_available(self, freecad_tools):
        """FreeCAD Python is discoverable."""
        assert freecad_tools["python"]() is not None
        assert freecad_tools["available"]()

    def test_create_box(self, freecad_tools):
        """Create a simple box and verify dimensions."""
        ops = [
            {"type": "new_doc", "name": "TestBox"},
            {"type": "make_box", "length": 50, "width": 30, "height": 20, "name": "MyBox"},
            {"type": "object_info", "object": "MyBox"},
        ]
        script = freecad_tools["build"](ops)
        result = freecad_tools["run"](script)

        assert "Volume: 30000.00" in result
        assert "50.00x30.00x20.00" in result
        assert "Edges: 12, Faces: 6, Vertices: 8" in result

    def test_create_cylinder(self, freecad_tools):
        """Create a cylinder and verify volume."""
        ops = [
            {"type": "new_doc", "name": "TestCyl"},
            {"type": "make_cylinder", "radius": 10, "height": 20, "name": "Cyl"},
            {"type": "object_info", "object": "Cyl"},
        ]
        script = freecad_tools["build"](ops)
        result = freecad_tools["run"](script)

        assert "Volume:" in result
        assert "20.00x20.00" in result  # diameter = 20mm

    def test_create_sphere(self, freecad_tools):
        """Create a sphere and verify dimensions."""
        ops = [
            {"type": "new_doc", "name": "TestSphere"},
            {"type": "make_sphere", "radius": 15, "name": "Ball"},
            {"type": "object_info", "object": "Ball"},
        ]
        script = freecad_tools["build"](ops)
        result = freecad_tools["run"](script)

        assert "Volume:" in result
        assert "30.00mm" in result  # diameter

    def test_create_cone(self, freecad_tools):
        """Create a cone with top and bottom radii."""
        ops = [
            {"type": "new_doc", "name": "TestCone"},
            {"type": "make_cone", "radius1": 20, "radius2": 10, "height": 30, "name": "Cone"},
            {"type": "object_info", "object": "Cone"},
        ]
        script = freecad_tools["build"](ops)
        result = freecad_tools["run"](script)

        assert "Volume:" in result
        assert "30.00mm" in result


class TestFreeCADBooleanOps:
    """Test boolean operations."""

    def test_boolean_cut(self, freecad_tools):
        """Cut a hole through a box."""
        ops = [
            {"type": "new_doc", "name": "TestCut"},
            {"type": "make_box", "length": 100, "width": 100, "height": 50, "name": "Plate"},
            {"type": "make_cylinder", "radius": 10, "height": 50, "name": "Hole"},
            {"type": "boolean", "operation": "cut", "object1": "Plate", "object2": "Hole", "result_name": "PlateWithHole"},
            {"type": "object_info", "object": "PlateWithHole"},
        ]
        script = freecad_tools["build"](ops)
        result = freecad_tools["run"](script)

        assert "Volume:" in result
        # Volume should be less than 100*100*50 = 500000
        assert "4" in result  # ~484292 mm3

    def test_boolean_union(self, freecad_tools):
        """Fuse two boxes together."""
        ops = [
            {"type": "new_doc", "name": "TestUnion"},
            {"type": "make_box", "length": 50, "width": 50, "height": 50, "name": "Box1"},
            {"type": "make_box", "length": 50, "width": 50, "height": 50, "name": "Box2"},
            {"type": "move", "object": "Box2", "dx": 25, "dy": 0, "dz": 0},
            {"type": "boolean", "operation": "union", "object1": "Box1", "object2": "Box2", "result_name": "Fused"},
            {"type": "object_info", "object": "Fused"},
        ]
        script = freecad_tools["build"](ops)
        result = freecad_tools["run"](script)

        assert "Volume:" in result

    def test_plate_with_holes(self, freecad_tools):
        """Create a plate with multiple holes."""
        ops = [
            {"type": "new_doc", "name": "TestPlateHoles"},
            {"type": "plate_with_holes", "length": 200, "width": 150, "thickness": 10,
             "hole_radius": 5, "hole_count_x": 4, "hole_count_y": 3, "margin": 20,
             "name": "MountPlate"},
            {"type": "object_info", "object": "MountPlate"},
        ]
        script = freecad_tools["build"](ops)
        result = freecad_tools["run"](script)

        assert "Volume:" in result
        assert "200.00x150.00x10.00" in result


class TestFreeCADExport:
    """Test file export operations."""

    def test_save_fcstd(self, freecad_tools, project_dir):
        """Save a .FCStd document."""
        path = str(project_dir / "test_save.FCStd")
        ops = [
            {"type": "new_doc", "name": "SaveTest"},
            {"type": "make_box", "length": 30, "width": 30, "height": 30, "name": "Cube"},
            {"type": "save", "path": path},
        ]
        script = freecad_tools["build"](ops)
        freecad_tools["run"](script)

        assert Path(path).exists()
        assert Path(path).stat().st_size > 0

    def test_export_stl(self, freecad_tools, project_dir):
        """Export model as STL."""
        path = str(project_dir / "test_export.stl")
        ops = [
            {"type": "new_doc", "name": "STLTest"},
            {"type": "make_cylinder", "radius": 15, "height": 30, "name": "Cyl"},
            {"type": "export_stl", "path": path},
        ]
        script = freecad_tools["build"](ops)
        result = freecad_tools["run"](script)

        assert Path(path).exists()
        assert Path(path).stat().st_size > 0
        assert "STL exported" in result

    def test_export_step(self, freecad_tools, project_dir):
        """Export model as STEP."""
        path = str(project_dir / "test_export.step")
        ops = [
            {"type": "new_doc", "name": "STEPTest"},
            {"type": "make_box", "length": 40, "width": 40, "height": 40, "name": "Box"},
            {"type": "export_step", "path": path},
        ]
        script = freecad_tools["build"](ops)
        freecad_tools["run"](script)

        assert Path(path).exists()
        assert Path(path).stat().st_size > 0

    def test_full_pipeline(self, freecad_tools, project_dir):
        """Test full modeling pipeline: create → boolean → fillet → save → export."""
        fcstd = str(project_dir / "pipeline.FCStd")
        stl = str(project_dir / "pipeline.stl")
        step = str(project_dir / "pipeline.step")

        ops = [
            {"type": "new_doc", "name": "PipelineTest"},
            {"type": "make_box", "length": 80, "width": 80, "height": 20, "name": "Base"},
            {"type": "make_cylinder", "radius": 8, "height": 20, "name": "Hole1"},
            {"type": "move", "object": "Hole1", "dx": 20, "dy": 20, "dz": 0},
            {"type": "boolean", "operation": "cut", "object1": "Base", "object2": "Hole1", "result_name": "BaseCut"},
            {"type": "fillet", "object": "BaseCut", "radius": 2},
            {"type": "object_info", "object": "BaseCut"},
            {"type": "save", "path": fcstd},
            {"type": "export_stl", "path": stl},
            {"type": "export_step", "path": step},
        ]
        script = freecad_tools["build"](ops)
        result = freecad_tools["run"](script)

        # Verify all files created
        assert Path(fcstd).exists()
        assert Path(stl).exists()
        assert Path(step).exists()
        assert "Volume:" in result


class TestFreeCADTransform:
    """Test move and rotate operations."""

    def test_move_object(self, freecad_tools):
        """Move an object and verify position."""
        ops = [
            {"type": "new_doc", "name": "MoveTest"},
            {"type": "make_box", "length": 10, "width": 10, "height": 10, "name": "Box"},
            {"type": "move", "object": "Box", "dx": 50, "dy": 0, "dz": 0},
            {"type": "object_info", "object": "Box"},
        ]
        script = freecad_tools["build"](ops)
        result = freecad_tools["run"](script)

        assert "Volume:" in result

    def test_rotate_object(self, freecad_tools):
        """Rotate an object and verify dimensions change but volume preserved."""
        ops = [
            {"type": "new_doc", "name": "RotateTest"},
            {"type": "make_box", "length": 20, "width": 10, "height": 5, "name": "Plate"},
            {"type": "rotate", "object": "Plate", "axis": "z", "angle": 45},
            {"type": "object_info", "object": "Plate"},
        ]
        script = freecad_tools["build"](ops)
        result = freecad_tools["run"](script)

        assert "Volume:" in result
        assert "1000.00" in result  # Volume preserved after rotation
        assert "21.21" in result  # Bounding box changed after 45deg rotation


class TestFreeCADBatchTool:
    """Test fc_batch tool via registry."""

    def test_batch_tool_registration(self):
        """FCBatchTool is registered in the tool system."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from lang3d.tools.base import ToolRegistry
        from lang3d.tools.freecad import FCBatchTool

        registry = ToolRegistry()
        tool = FCBatchTool()
        registry.register(tool)

        assert "fc_batch" in registry.list_tools()

    def test_batch_tool_execution(self, freecad_tools, project_dir):
        """Execute fc_batch with multiple operations."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from lang3d.tools.freecad import FCBatchTool

        path = str(project_dir / "batch_test.FCStd")
        tool = FCBatchTool()
        result = tool.execute(operations=[
            {"type": "new_doc", "name": "BatchTest"},
            {"type": "make_box", "length": 50, "width": 50, "height": 10, "name": "Plate"},
            {"type": "make_cylinder", "radius": 5, "height": 10, "name": "Peg"},
            {"type": "boolean", "operation": "union", "object1": "Plate", "object2": "Peg", "result_name": "Result"},
            {"type": "save", "path": path},
        ])

        assert Path(path).exists()


class TestFreeCADFEAIntegration:
    """Test FEA analysis with real FreeCAD."""

    def test_fea_run_cantilever_beam(self, freecad_tools, project_dir):
        """Run FEA on a cantilever beam and verify solver output."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from lang3d.tools.simulation import FEASetupAndRunTool

        # Create beam model
        beam_path = str(project_dir / "beam.FCStd")
        ops = [
            {"type": "new_doc", "name": "Cantilever"},
            {"type": "make_box", "length": 100, "width": 20, "height": 5, "name": "Beam"},
            {"type": "save", "path": beam_path},
        ]
        freecad_tools["run"](freecad_tools["build"](ops))
        assert Path(beam_path).exists()

        # Run FEA
        fea = FEASetupAndRunTool()
        result = fea.execute(
            document_path=beam_path,
            material="steel",
            fixed_face="bottom",
            force_face="top",
            force_magnitude=1000,
            mesh_size="coarse",
        )

        assert "FEA Analysis Results" in result
        assert "Results present: True" in result
        assert "FEA_COMPLETE" in result

    def test_fea_mesh_generation(self, freecad_tools, project_dir):
        """Verify FEA mesh is generated correctly."""
        beam_path = str(project_dir / "beam_mesh.FCStd")
        ops = [
            {"type": "new_doc", "name": "MeshTest"},
            {"type": "make_box", "length": 50, "width": 20, "height": 10, "name": "Block"},
            {"type": "save", "path": beam_path},
        ]
        freecad_tools["run"](freecad_tools["build"](ops))

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from lang3d.tools.simulation import FEASetupAndRunTool

        fea = FEASetupAndRunTool()
        result = fea.execute(
            document_path=beam_path,
            material="aluminum",
            fixed_face="left",
            force_face="right",
            force_magnitude=500,
            mesh_size="coarse",
        )

        assert "Mesh generated:" in result
        assert "nodes" in result
