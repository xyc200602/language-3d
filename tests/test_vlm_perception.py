"""VLM Screen Perception End-to-End Test.

Tests the complete perception pipeline:
1. Screen capture (mss)
2. Window enumeration and capture
3. VLM image analysis (GLM-4V-Flash)
4. Tool registry integration
5. Agent tool-calling with VLM
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pathlib import Path

from PIL import Image, ImageDraw

from lang3d.config import load_config
from lang3d.models.router import ModelRouter
from lang3d.tools.base import ToolRegistry
from lang3d.tools.screen import register_screen_tools
from lang3d.tools.vlm import register_vlm_tools

SCREENSHOT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data", "screenshots", "test"
)
os.makedirs(SCREENSHOT_DIR, exist_ok=True)


def _make_test_images():
    """Create test images for VLM analysis."""
    tmpdir = os.path.join(SCREENSHOT_DIR, "test_inputs")
    os.makedirs(tmpdir, exist_ok=True)
    paths = {}

    # Image 1: Simple red square
    img = Image.new("RGB", (300, 300), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, 250, 250], fill=(255, 0, 0))
    p1 = os.path.join(tmpdir, "red_square.png")
    img.save(p1)
    paths["red_square"] = p1

    # Image 2: Blue circle
    img2 = Image.new("RGB", (300, 300), color=(255, 255, 255))
    draw2 = ImageDraw.Draw(img2)
    draw2.ellipse([30, 30, 270, 270], fill=(0, 0, 255))
    p2 = os.path.join(tmpdir, "blue_circle.png")
    img2.save(p2)
    paths["blue_circle"] = p2

    # Image 3: 3D-like CAD mockup (wireframe box)
    img3 = Image.new("RGB", (800, 600), color=(64, 64, 64))
    draw3 = ImageDraw.Draw(img3)
    # Draw a simple 3D box wireframe
    # Front face
    draw3.polygon([(200, 200), (500, 200), (500, 450), (200, 450)],
                  outline=(255, 255, 255), width=2)
    # Back face
    draw3.polygon([(280, 130), (580, 130), (580, 380), (280, 380)],
                  outline=(200, 200, 200), width=1)
    # Connecting lines
    draw3.line([(200, 200), (280, 130)], fill=(200, 200, 200), width=1)
    draw3.line([(500, 200), (580, 130)], fill=(200, 200, 200), width=1)
    draw3.line([(500, 450), (580, 380)], fill=(200, 200, 200), width=1)
    draw3.line([(200, 450), (280, 380)], fill=(200, 200, 200), width=1)
    # Add "FreeCAD" text label area
    draw3.rectangle([(10, 10), (200, 40)], fill=(80, 80, 80), outline=(150, 150, 150))
    p3 = os.path.join(tmpdir, "cad_mockup.png")
    img3.save(p3)
    paths["cad_mockup"] = p3

    return paths


def test_vlm_color_recognition():
    """Test VLM can identify colors in images."""
    print("\n[1] VLM color recognition...")
    config = load_config()
    router = ModelRouter(config)
    images = _make_test_images()

    result = router.vision(images["red_square"], "What is the main color of the shape in this image? Answer in one word.")
    print(f"  Red square: {result}")
    assert "red" in result.lower()

    result2 = router.vision(images["blue_circle"], "What is the main color of the shape? One word.")
    print(f"  Blue circle: {result2}")
    assert "blue" in result2.lower()

    print("  PASSED")


def test_vlm_shape_recognition():
    """Test VLM can identify shapes."""
    print("\n[2] VLM shape recognition...")
    config = load_config()
    router = ModelRouter(config)
    images = _make_test_images()

    result = router.vision(images["red_square"], "What shape is shown? Answer in one word.")
    print(f"  Red shape: {result}")
    assert any(w in result.lower() for w in ["square", "rectangle", "rectangular"])

    result2 = router.vision(images["blue_circle"], "What shape is shown? Answer in one word.")
    print(f"  Blue shape: {result2}")
    assert "circle" in result2.lower() or "ellip" in result2.lower()

    print("  PASSED")


def test_vlm_3d_scene_understanding():
    """Test VLM can understand 3D wireframe drawings."""
    print("\n[3] VLM 3D scene understanding...")
    config = load_config()
    router = ModelRouter(config)
    images = _make_test_images()

    result = router.vision(
        images["cad_mockup"],
        "Describe the 3D object in this image. Is it a cube, sphere, cylinder, or cone? "
        "Describe the wireframe representation."
    )
    print(f"  3D analysis: {result}")
    assert isinstance(result, str)
    assert len(result) > 10
    # Should identify it as a cube/box-like shape
    assert any(w in result.lower() for w in ["cube", "box", "rectangular", "cuboid", "3d", "wireframe"])

    print("  PASSED")


def test_screen_capture_tool():
    """Test screen_capture tool via registry."""
    print("\n[4] Screen capture tool...")
    registry = ToolRegistry()
    register_screen_tools(registry, screenshot_dir=SCREENSHOT_DIR)

    result = registry.execute("screen_capture", region="fullscreen")
    print(f"  Result: {result}")
    assert "Screenshot saved to:" in result

    # Verify file exists
    path_str = result.split(": ", 1)[1]
    assert Path(path_str).exists()
    assert Path(path_str).stat().st_size > 1000

    print("  PASSED")


def test_list_windows_tool():
    """Test list_windows tool."""
    print("\n[5] List windows tool...")
    registry = ToolRegistry()
    register_screen_tools(registry, screenshot_dir=SCREENSHOT_DIR)

    result = registry.execute("list_windows")
    # Safe print for GBK console
    safe = result.encode("ascii", errors="replace").decode("ascii")
    print(f"  {safe[:200]}...")
    assert "Found" in result
    assert "visible windows" in result

    print("  PASSED")


def test_vlm_analyze_tool():
    """Test vlm_analyze tool via registry."""
    print("\n[6] VLM analyze tool...")
    config = load_config()
    router = ModelRouter(config)

    registry = ToolRegistry()
    register_vlm_tools(registry, router, screenshot_dir=SCREENSHOT_DIR)
    images = _make_test_images()

    result = registry.execute("vlm_analyze", image_path=images["red_square"], prompt="What color is the shape?")
    print(f"  Result: {result}")
    assert "red" in result.lower()

    print("  PASSED")


def test_screen_analyze_tool():
    """Test screen_analyze tool (capture + VLM in one step)."""
    print("\n[7] Screen analyze tool...")
    config = load_config()
    router = ModelRouter(config)

    registry = ToolRegistry()
    register_vlm_tools(registry, router, screenshot_dir=SCREENSHOT_DIR)

    result = registry.execute(
        "screen_analyze",
        prompt="Briefly describe what you see on this screen in 1-2 sentences.",
    )
    print(f"  Result: {result[:300]}...")
    assert "[Screenshot:" in result
    assert "Error" not in result

    print("  PASSED")


def test_full_tool_registry():
    """Test that all tools register together (screen + VLM + FreeCAD + file + bash)."""
    print("\n[8] Full tool registry integration...")
    config = load_config()
    router = ModelRouter(config)

    registry = ToolRegistry()

    # File tools
    from lang3d.tools.file_ops import register_file_tools
    register_file_tools(registry)

    # Bash tools
    from lang3d.tools.bash import register_bash_tools
    register_bash_tools(registry)

    # Screen tools
    register_screen_tools(registry, screenshot_dir=SCREENSHOT_DIR)

    # VLM tools
    register_vlm_tools(registry, router, screenshot_dir=SCREENSHOT_DIR)

    # FreeCAD tools
    try:
        from lang3d.tools.freecad import register_freecad_tools
        register_freecad_tools(registry)
    except Exception:
        pass

    all_tools = registry.list_tools()
    print(f"  Total tools: {len(all_tools)}")

    # Check all expected tool categories
    categories = {
        "file_ops": ["file_read", "file_write", "file_edit"],
        "bash": ["bash", "python_exec"],
        "screen": ["screen_capture", "window_capture", "list_windows"],
        "vlm": ["vlm_analyze", "screen_analyze", "window_analyze", "cad_verify"],
        "freecad": ["fc_new_doc", "fc_make_box", "fc_batch"],
    }

    for cat, expected in categories.items():
        for tool in expected:
            assert tool in all_tools, f"Missing {cat} tool: {tool}"
        print(f"  {cat}: {len(expected)} tools OK")

    # Get all definitions (for sending to LLM)
    defs = registry.get_all_definitions()
    assert len(defs) >= 30  # Should have 30+ tools total
    print(f"  Total tool definitions: {len(defs)}")

    print("  PASSED")


def main():
    print("=" * 60)
    print("  VLM Screen Perception End-to-End Test")
    print("  GLM-4V-Flash + Screen Capture + Tool Registry")
    print("=" * 60)

    tests = [
        ("VLM Color Recognition", test_vlm_color_recognition),
        ("VLM Shape Recognition", test_vlm_shape_recognition),
        ("VLM 3D Scene Understanding", test_vlm_3d_scene_understanding),
        ("Screen Capture Tool", test_screen_capture_tool),
        ("List Windows Tool", test_list_windows_tool),
        ("VLM Analyze Tool", test_vlm_analyze_tool),
        ("Screen Analyze Tool", test_screen_analyze_tool),
        ("Full Tool Registry", test_full_tool_registry),
    ]

    results = []
    for name, test_fn in tests:
        try:
            test_fn()
            results.append((name, True, None))
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append((name, False, str(e)))

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    for name, ok, err in results:
        status = "PASS" if ok else f"FAIL: {err}"
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

    print(f"\n  {passed}/{total} tests passed")

    if passed == total:
        print("\n  ALL VLM PERCEPTION TESTS PASSED")
    else:
        print(f"\n  {total - passed} test(s) failed")


if __name__ == "__main__":
    main()
