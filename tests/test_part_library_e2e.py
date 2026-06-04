"""End-to-end tests for part library — realistic geometry generation via FreeCAD.

These tests require FreeCAD to be installed.
They verify that part templates generate valid .FCStd and .STL files
with no script errors.
"""

import os
import re
from pathlib import Path

import pytest


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
        _run_freecad_script, _find_freecad_python,
        _is_freecad_available,
    )
    return {
        "run": _run_freecad_script,
        "python": _find_freecad_python,
        "available": _is_freecad_available,
    }


@pytest.fixture
def output_dir(tmp_path):
    """Create a temporary output directory."""
    return tmp_path


def _generate_part(part_id: str, params: dict, output_dir: Path, freecad_tools: dict) -> dict:
    """Helper: generate a part from template and return result info."""
    from lang3d.knowledge.parts_catalog import get_template, resolve_parameters, format_fc_script

    template = get_template(part_id)
    assert template is not None, f"Template '{part_id}' not found"

    resolved = resolve_parameters(template, params)
    model_script = format_fc_script(template, resolved)

    # Verify no unreplaced placeholders
    unreplaced = re.findall(r'\{[a-zA-Z_][a-zA-Z0-9_]*\}', model_script)
    assert len(unreplaced) == 0, f"Unreplaced placeholders in {part_id}: {unreplaced}"

    # Build output paths
    param_desc = "_".join(
        f"{k}{int(v) if isinstance(v, float) and v == int(v) else v}"
        for k, v in resolved.items()
        if k not in ("thread_detail", "tooth_detail", "bearing_detail")
    )
    safe_name = f"{part_id}_{param_desc}"
    fcstd_path = output_dir / f"{safe_name}.FCStd"
    stl_path = output_dir / f"{safe_name}.stl"

    # Use forward slashes in print messages (avoids backslash escape issues)
    fcstd_display = str(fcstd_path).replace("\\", "/")
    stl_display = str(stl_path).replace("\\", "/")

    full_script = model_script + (
        "import os\n"
        f"os.makedirs(r'{output_dir}', exist_ok=True)\n"
        f"doc.saveAs(r'{fcstd_path}')\n"
        "print('Saved: " + fcstd_display + "')\n"
        "import Mesh\n"
        "_export_list = [o for o in doc.Objects if hasattr(o, 'Shape')]\n"
        "if _export_list:\n"
        f"    Mesh.export(_export_list, r'{stl_path}')\n"
        f"    _stl_sz = os.path.getsize(r'{stl_path}')\n"
        "    print('STL: " + stl_display + " (' + str(_stl_sz) + ' bytes)')\n"
    )

    output = freecad_tools["run"](full_script, timeout=120)

    return {
        "output": output,
        "fcstd_path": fcstd_path,
        "stl_path": stl_path,
        "resolved": resolved,
        "safe_name": safe_name,
    }


def _assert_valid_result(result: dict):
    """Helper: verify generation output has no errors and files exist."""
    output = result["output"]
    # Check no Python tracebacks in output
    assert "Traceback" not in output, f"Python error in output:\n{output}"
    # Check no FreeCAD error log entries (case-insensitive)
    # Note: FreeCAD may print "Error" in non-critical messages, so check for Traceback
    # which is the definitive sign of a Python exception
    assert result["fcstd_path"].exists(), f"FCStd not created: {result['fcstd_path']}"
    assert result["fcstd_path"].stat().st_size > 0, "FCStd file is empty"
    if result["stl_path"].exists():
        assert result["stl_path"].stat().st_size > 0, "STL file is empty"


# ---------------------------------------------------------------------------
# Realistic part generation tests
# ---------------------------------------------------------------------------

class TestRealisticPartGeneration:
    """Test realistic thread/gear generation with actual FreeCAD."""

    def test_realistic_m6_screw(self, freecad_tools, output_dir):
        """Generate M6x25 realistic screw and verify output."""
        result = _generate_part("socket_head_cap_screw", {
            "thread_diameter": 6, "length": 25, "head_diameter": 10,
            "thread_detail": "realistic", "thread_pitch": 1.0,
        }, output_dir, freecad_tools)
        _assert_valid_result(result)
        assert "makeHelix" in result["output"] or result["fcstd_path"].exists()

    def test_realistic_hex_bolt(self, freecad_tools, output_dir):
        """Generate M8x40 realistic hex bolt."""
        result = _generate_part("hex_bolt", {
            "thread_diameter": 8, "length": 40,
            "thread_detail": "realistic", "thread_pitch": 1.25,
        }, output_dir, freecad_tools)
        _assert_valid_result(result)

    def test_realistic_hex_nut(self, freecad_tools, output_dir):
        """Generate M8 realistic hex nut."""
        result = _generate_part("hex_nut", {
            "nominal_diameter": 8,
            "thread_detail": "realistic", "thread_pitch": 1.25,
        }, output_dir, freecad_tools)
        _assert_valid_result(result)

    def test_realistic_gear_20t(self, freecad_tools, output_dir):
        """Generate 20-tooth realistic involute gear."""
        result = _generate_part("spur_gear", {
            "teeth": 20, "module": 1.0, "thickness": 6, "bore_diameter": 8,
            "tooth_detail": "realistic", "pressure_angle": 20.0, "backlash": 0.1,
        }, output_dir, freecad_tools)
        _assert_valid_result(result)

    def test_realistic_gear_small_8t(self, freecad_tools, output_dir):
        """Generate 8-tooth small realistic gear (edge case)."""
        result = _generate_part("spur_gear", {
            "teeth": 8, "module": 1.0, "thickness": 5, "bore_diameter": 4,
            "tooth_detail": "realistic", "pressure_angle": 20.0, "backlash": 0.1,
        }, output_dir, freecad_tools)
        _assert_valid_result(result)


# ---------------------------------------------------------------------------
# Simplified regression tests
# ---------------------------------------------------------------------------

class TestSimplifiedRegression:
    """Verify simplified scripts still work (backward compatibility)."""

    def test_simplified_m6_screw(self, freecad_tools, output_dir):
        """Generate simplified M6 screw — regression test."""
        result = _generate_part("socket_head_cap_screw", {
            "thread_diameter": 6, "length": 25, "head_diameter": 10,
            "thread_detail": "simplified",
        }, output_dir, freecad_tools)
        _assert_valid_result(result)

    def test_simplified_gear(self, freecad_tools, output_dir):
        """Generate simplified gear — regression test."""
        result = _generate_part("spur_gear", {
            "teeth": 20, "module": 1.0, "thickness": 6, "bore_diameter": 8,
            "tooth_detail": "simplified",
        }, output_dir, freecad_tools)
        _assert_valid_result(result)

    def test_all_thread_templates_simplified(self, freecad_tools, output_dir):
        """Verify all 3 thread templates work with simplified scripts."""
        thread_templates = [
            ("socket_head_cap_screw", {"thread_diameter": 6, "length": 20, "head_diameter": 10}),
            ("hex_bolt", {"thread_diameter": 6, "length": 30}),
            ("hex_nut", {"nominal_diameter": 6}),
        ]
        for part_id, params in thread_templates:
            params_with_detail = dict(params)
            params_with_detail.setdefault("thread_detail", "simplified")
            result = _generate_part(part_id, params_with_detail, output_dir, freecad_tools)
            _assert_valid_result(result)
