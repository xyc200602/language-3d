"""Integration test: VLM UI element localization.

Tests: fc_batch → fc_open_gui → vlm_locate (find FreeCAD UI elements) → fc_close_gui

Run with: python tests/test_vlm_locate_e2e.py
"""

import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

TEST_DIR = project_root / "data" / "projects" / "vlm_locate_test"
TEST_DIR.mkdir(parents=True, exist_ok=True)
SCREENSHOT_DIR = project_root / "data" / "screenshots"
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

FCSTD_PATH = str(TEST_DIR / "vlm_locate_test.FCStd")


def run_step(name: str, func, *args, **kwargs):
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

    # Step 1: Create a model
    from lang3d.tools.freecad import FCBatchTool
    batch = FCBatchTool()
    step1_ok, _ = run_step(
        "Step 1: Create model",
        batch.execute,
        operations=[
            {"type": "new_doc", "name": "LocateTest"},
            {"type": "make_box", "length": 20, "width": 20, "height": 20, "name": "Box"},
            {"type": "save", "path": FCSTD_PATH},
        ],
    )
    results["step1_create"] = step1_ok

    if not step1_ok:
        print("\nCannot continue. Aborting.")
        return False

    # Step 2: Launch FreeCAD GUI
    from lang3d.tools.freecad import FCOpenGUITool, FCCloseGUITool
    open_gui = FCOpenGUITool()
    close_gui = FCCloseGUITool()

    step2_ok, _ = run_step(
        "Step 2: Launch FreeCAD GUI",
        open_gui.execute,
        file_path=FCSTD_PATH,
        view="isometric",
        fit_all=True,
        wait_seconds=8,
    )
    results["step2_open_gui"] = step2_ok

    print("\n  Waiting 6s for FreeCAD to fully render...")
    time.sleep(6)

    # Step 3: vlm_locate - find all interactive elements
    from lang3d.config import load_config
    from lang3d.models.router import ModelRouter
    from lang3d.tools.vlm import VLMLocateTool

    config = load_config()
    router = ModelRouter(config)
    locate = VLMLocateTool(router, screenshot_dir=str(SCREENSHOT_DIR))

    step3_ok, result3 = run_step(
        "Step 3: vlm_locate all FreeCAD UI elements",
        locate.execute,
        window_title="FreeCAD",
        target="toolbar buttons, menu items, and sidebar panels",
        detail="standard",
    )
    results["step3_vlm_locate_all"] = step3_ok

    # Check if elements were found
    if step3_ok and "Found" in str(result3) and "UI elements" in str(result3):
        # Parse element count
        for line in str(result3).split("\n"):
            if "Found" in line and "UI elements" in line:
                print(f"\n  --> {line.strip()}")
                break
        results["step3_elements_found"] = True
    else:
        results["step3_elements_found"] = False

    # Step 4: vlm_locate - find specific toolbar buttons
    print("\n  Waiting 3s to avoid API rate limit...")
    time.sleep(3)

    step4_ok, result4 = run_step(
        "Step 4: vlm_locate toolbar buttons specifically",
        locate.execute,
        window_title="FreeCAD",
        target="toolbar buttons (icons at the top of the window)",
        detail="standard",
    )
    results["step4_locate_toolbar"] = step4_ok

    # Step 5: Test gui_click on a found element (click in the 3D viewport area)
    if step3_ok:
        from lang3d.tools.gui_action import GUIClickTool, GUIScreenshotTool
        click = GUIClickTool()
        gui_screenshot = GUIScreenshotTool(screenshot_dir=str(SCREENSHOT_DIR))

        # Click somewhere in the center (viewport area)
        step5a_ok, _ = run_step(
            "Step 5a: Click in 3D viewport (center of FreeCAD window)",
            click.execute,
            x=1280,
            y=600,
        )
        results["step5a_click_viewport"] = step5a_ok

        step5b_ok, _ = run_step(
            "Step 5b: Screenshot after click",
            gui_screenshot.execute,
        )
        results["step5b_screenshot"] = step5b_ok
    else:
        results["step5a_click_viewport"] = False
        results["step5b_screenshot"] = False

    # Step 6: Close FreeCAD
    step6_ok, _ = run_step(
        "Step 6: Close FreeCAD",
        close_gui.execute,
    )
    results["step6_close"] = step6_ok

    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")

    passed = failed = 0
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
        passed += ok
        failed += not ok

    total = passed + failed
    print(f"\nTotal: {passed}/{total} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
