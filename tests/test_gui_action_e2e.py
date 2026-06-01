"""Integration test: PyAutoGUI GUI automation tools with FreeCAD.

Tests the full GUI automation pipeline:
  fc_batch (create model) -> fc_open_gui -> gui_mouse_pos -> gui_screenshot ->
  gui_click (menu interaction) -> gui_scroll -> gui_drag -> VLM verify -> fc_close_gui

Run with: python tests/test_gui_action_e2e.py
"""

import sys
import time
from pathlib import Path

# Fix Windows console encoding
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

# Test output directory
TEST_DIR = project_root / "data" / "projects" / "gui_action_test"
TEST_DIR.mkdir(parents=True, exist_ok=True)
SCREENSHOT_DIR = project_root / "data" / "screenshots"
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

FCSTD_PATH = str(TEST_DIR / "gui_action_test.FCStd")


def run_step(name: str, func, *args, **kwargs):
    """Run a test step and print result. Returns (success, result_str)."""
    print(f"\n{'='*60}")
    print(f"STEP: {name}")
    print(f"{'='*60}")
    try:
        result = func(*args, **kwargs)
        display = str(result)[:2000]
        print(f"RESULT:\n{display}")
        if isinstance(result, str) and result.startswith("Error"):
            print(f"FAIL: {name} returned error")
            return False, result
        print(f"PASS: {name}")
        return True, result
    except Exception as e:
        print(f"FAIL: {name} - {e}")
        import traceback
        traceback.print_exc()
        return False, str(e)


def main():
    results = {}

    # ---- Step 1: Create a 3D model using fc_batch ----
    from lang3d.tools.freecad import FCBatchTool

    batch = FCBatchTool()
    step1_ok, _ = run_step(
        "Step 1: Create 3D model (30x30x30 box with R5 center hole)",
        batch.execute,
        operations=[
            {"type": "new_doc", "name": "GuiActionTest"},
            {"type": "make_box", "length": 30, "width": 30, "height": 30, "name": "Box"},
            {"type": "make_cylinder", "radius": 5, "height": 30, "name": "Hole"},
            {"type": "move", "object": "Hole", "dx": 15, "dy": 15, "dz": 0},
            {"type": "boolean", "operation": "cut", "object1": "Box", "object2": "Hole", "result_name": "BoxWithHole"},
            {"type": "save", "path": FCSTD_PATH},
        ],
    )
    results["step1_create_model"] = step1_ok

    if not step1_ok:
        print("\nCannot continue without model. Aborting.")
        return False

    # ---- Step 2: Launch FreeCAD GUI ----
    from lang3d.tools.freecad import FCOpenGUITool

    open_gui = FCOpenGUITool()
    step2_ok, _ = run_step(
        "Step 2: Launch FreeCAD GUI with model",
        open_gui.execute,
        file_path=FCSTD_PATH,
        view="isometric",
        fit_all=True,
        wait_seconds=8,
    )
    results["step2_open_gui"] = step2_ok

    print("\n  Waiting 6s for FreeCAD to fully render...")
    time.sleep(6)

    # ---- Step 3: gui_mouse_pos ----
    from lang3d.tools.gui_action import GUIMousePosTool

    mouse_pos = GUIMousePosTool()
    step3_ok, result3 = run_step(
        "Step 3: Get mouse position",
        mouse_pos.execute,
    )
    results["step3_mouse_pos"] = step3_ok

    # ---- Step 4: gui_screenshot (fullscreen) ----
    from lang3d.tools.gui_action import GUIScreenshotTool

    screenshot = GUIScreenshotTool(screenshot_dir=str(SCREENSHOT_DIR))
    step4_ok, result4 = run_step(
        "Step 4: Screenshot fullscreen via gui_screenshot",
        screenshot.execute,
    )
    results["step4_screenshot"] = step4_ok

    # ---- Step 5-8: Find FreeCAD window for targeted interactions ----
    from lang3d.tools.screen import WindowCaptureTool, _find_windows_by_title
    import ctypes
    import ctypes.wintypes

    window_capture = WindowCaptureTool(screenshot_dir=str(SCREENSHOT_DIR))

    matches = _find_windows_by_title("FreeCAD")
    fc_rect = None
    user32 = ctypes.windll.user32
    screen_w = user32.GetSystemMetrics(0)
    screen_h = user32.GetSystemMetrics(1)

    for hwnd, title in matches:
        rect = ctypes.wintypes.RECT()
        user32.GetWindowRect(hwnd, ctypes.byref(rect))
        left = max(0, rect.left)
        top = max(0, rect.top)
        right = min(screen_w, rect.right)
        bottom = min(screen_h, rect.bottom)
        w = right - left
        h = bottom - top
        if w > 100 and h > 100:
            fc_rect = (left, top, w, h)
            print(f"\n  FreeCAD window found: '{title}' at ({left},{top}) size {w}x{h}")
            break

    if fc_rect:
        center_x = fc_rect[0] + fc_rect[2] // 2
        center_y = fc_rect[1] + fc_rect[3] // 2

        # Step 5: Region screenshot of FreeCAD window
        from lang3d.tools.gui_action import GUIClickTool, GUIScrollTool, GUIDragTool

        region_str = f"{fc_rect[0]},{fc_rect[1]},{fc_rect[2]},{fc_rect[3]}"
        step5_ok, _ = run_step(
            "Step 5: Screenshot FreeCAD region via gui_screenshot",
            screenshot.execute,
            region=region_str,
        )
        results["step5_region_screenshot"] = step5_ok

        # Step 6: gui_click in FreeCAD viewport center
        click = GUIClickTool()
        step6_ok, _ = run_step(
            "Step 6: Click center of FreeCAD viewport",
            click.execute,
            x=center_x,
            y=center_y,
        )
        results["step6_click"] = step6_ok

        # Step 7: gui_scroll (zoom in FreeCAD)
        scroll = GUIScrollTool()
        step7_ok, _ = run_step(
            "Step 7: Scroll up (zoom in) in FreeCAD viewport",
            scroll.execute,
            clicks=5,
            x=center_x,
            y=center_y,
        )
        results["step7_scroll"] = step7_ok
        time.sleep(1)

        # Step 8: gui_drag (rotate view)
        drag = GUIDragTool()
        drag_start_x = fc_rect[0] + fc_rect[2] // 3
        drag_start_y = fc_rect[1] + fc_rect[3] // 3
        drag_end_x = fc_rect[0] + 2 * fc_rect[2] // 3
        drag_end_y = fc_rect[1] + 2 * fc_rect[3] // 3

        step8_ok, _ = run_step(
            "Step 8: Drag to rotate 3D view",
            drag.execute,
            start_x=drag_start_x,
            start_y=drag_start_y,
            end_x=drag_end_x,
            end_y=drag_end_y,
            button="middle",
        )
        results["step8_drag"] = step8_ok
        time.sleep(1)
    else:
        print("\n  SKIP: Could not find FreeCAD window for targeted interactions")
        results["step5_region_screenshot"] = False
        results["step6_click"] = False
        results["step7_scroll"] = False
        results["step8_drag"] = False

    # ---- Step 9: Press Escape to clear any menus ----
    from lang3d.tools.gui_action import GUIPressKeyTool
    press = GUIPressKeyTool()

    step9_ok, _ = run_step(
        "Step 9: Press Escape to clear any menus",
        press.execute,
        key="escape",
    )
    results["step9_press_escape"] = step9_ok

    # ---- Step 10: Capture FreeCAD window after GUI interactions ----
    step10_ok, _ = run_step(
        "Step 10: Capture FreeCAD window after GUI interactions",
        window_capture.execute,
        title="FreeCAD",
    )
    results["step10_capture_after_gui"] = step10_ok

    # ---- Step 11: VLM verify FreeCAD shows the 3D model ----
    window_screenshots = sorted(
        SCREENSHOT_DIR.glob("window_*.png"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if window_screenshots:
        from lang3d.config import load_config
        from lang3d.models.router import ModelRouter
        from lang3d.tools.vlm import VLMAnalyzeTool

        config = load_config()
        router = ModelRouter(config)
        vlm = VLMAnalyzeTool(router)

        latest_screenshot = str(window_screenshots[0])
        print(f"\n  Analyzing screenshot: {latest_screenshot}")

        step11_ok, _ = run_step(
            "Step 11: VLM verify FreeCAD shows 3D model after GUI interactions",
            vlm.execute,
            image_path=latest_screenshot,
            prompt=(
                "This is a FreeCAD window after GUI automation interactions. "
                "1. Is FreeCAD visible with a 3D model? "
                "2. What shape is the model? "
                "3. Was the view rotated or zoomed compared to a standard isometric view?"
            ),
            detail="standard",
        )
        results["step11_vlm_verify"] = step11_ok
    else:
        print("\n  SKIP: No window screenshot available for VLM analysis")
        results["step11_vlm_verify"] = False

    # ---- Step 12: Close FreeCAD ----
    from lang3d.tools.freecad import FCCloseGUITool
    close_gui = FCCloseGUITool()
    step12_ok, _ = run_step(
        "Step 12: Close FreeCAD GUI",
        close_gui.execute,
    )
    results["step12_close_gui"] = step12_ok

    # ---- Summary ----
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")

    passed = 0
    failed = 0
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
        if ok:
            passed += 1
        else:
            failed += 1

    total = passed + failed
    print(f"\nTotal: {passed}/{total} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
