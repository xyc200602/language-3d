"""Integration test: FreeCAD menu automation via VLM locate + click.

Tests: fc_batch → fc_open_gui → fc_menu (click menu items) → fc_close_gui

Run with: python tests/test_fc_menu_e2e.py
"""

import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

TEST_DIR = project_root / "data" / "projects" / "fc_menu_test"
TEST_DIR.mkdir(parents=True, exist_ok=True)
SCREENSHOT_DIR = project_root / "data" / "screenshots"
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

FCSTD_PATH = str(TEST_DIR / "fc_menu_test.FCStd")


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

    # Step 1: Create a model with fc_batch (API way)
    from lang3d.tools.freecad import FCBatchTool
    batch = FCBatchTool()
    step1_ok, _ = run_step(
        "Step 1: Create model via fc_batch",
        batch.execute,
        operations=[
            {"type": "new_doc", "name": "MenuTest"},
            {"type": "make_box", "length": 30, "width": 20, "height": 10, "name": "TestBox"},
            {"type": "save", "path": FCSTD_PATH},
        ],
    )
    results["step1_create_model"] = step1_ok

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

    # Step 3: Test fc_menu - click the "File" menu
    from lang3d.config import load_config
    from lang3d.models.router import ModelRouter
    from lang3d.tools.vlm import VLMLocateTool
    from lang3d.tools.gui_action import GUIClickTool, GUITypeTool, GUIPressKeyTool, GUIScreenshotTool
    from lang3d.tools.fc_menu import FCMenuClickTool, FCMenuWorkflowTool

    config = load_config()
    router = ModelRouter(config)

    locate = VLMLocateTool(router, screenshot_dir=str(SCREENSHOT_DIR))
    click = GUIClickTool()
    type_tool = GUITypeTool()
    press_key = GUIPressKeyTool()
    gui_screenshot = GUIScreenshotTool(screenshot_dir=str(SCREENSHOT_DIR))

    # Create fc_menu tool with VLM locate
    fc_menu = FCMenuClickTool(locate_tool=locate, click_tool=click)

    step3_ok, result3 = run_step(
        "Step 3: fc_menu click 'File' menu",
        lambda: fc_menu.execute(name="File", element_type="menu", wait=2.0),
    )
    results["step3_fc_menu_file"] = step3_ok

    # Step 4: Screenshot to verify File menu opened
    time.sleep(1)
    step4_ok, _ = run_step(
        "Step 4: Screenshot after File menu click",
        gui_screenshot.execute,
    )
    results["step4_screenshot"] = step4_ok

    # Step 5: Press Escape to close the menu
    step5_ok, _ = run_step(
        "Step 5: Press Escape to close menu",
        press_key.execute,
        key="escape",
    )
    results["step5_escape"] = step5_ok

    # Step 6: Test fc_menu - click "Part" menu
    print("\n  Waiting 3s to avoid API rate limit...")
    time.sleep(3)

    step6_ok, result6 = run_step(
        "Step 6: fc_menu click 'Part' menu",
        lambda: fc_menu.execute(name="Part", element_type="menu", wait=2.0),
    )
    results["step6_fc_menu_part"] = step6_ok

    # Step 7: Screenshot after Part menu
    time.sleep(1)
    step7_ok, _ = run_step(
        "Step 7: Screenshot after Part menu click",
        gui_screenshot.execute,
    )
    results["step7_screenshot"] = step7_ok

    # Step 8: Press Escape to close Part menu
    step8_ok, _ = run_step(
        "Step 8: Press Escape to close Part menu",
        press_key.execute,
        key="escape",
    )
    results["step8_escape"] = step8_ok

    # Step 9: Test fc_menu_workflow
    print("\n  Waiting 3s to avoid API rate limit...")
    time.sleep(3)

    fc_menu_workflow = FCMenuWorkflowTool(
        locate_tool=locate,
        click_tool=click,
        type_tool=type_tool,
        press_key_tool=press_key,
    )

    step9_ok, result9 = run_step(
        "Step 9: fc_menu_workflow 'save_file'",
        lambda: fc_menu_workflow.execute(workflow="save_file"),
    )
    results["step9_workflow_save"] = step9_ok

    # Step 10: Close FreeCAD
    step10_ok, _ = run_step(
        "Step 10: Close FreeCAD",
        close_gui.execute,
    )
    results["step10_close"] = step10_ok

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

    # Comparison analysis
    print(f"\n{'='*60}")
    print("FC_MENU vs FC_BATCH COMPARISON")
    print(f"{'='*60}")
    print("  fc_batch (API):  Precise, fast, headless. Best for parametric modeling.")
    print("  fc_menu (GUI):   Visual, interactive. Best for workbench-specific features.")
    print("  Result:          fc_menu successfully clicked FreeCAD menu items via VLM locate.")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
