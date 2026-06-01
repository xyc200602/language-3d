"""Integration test: FreeCAD GUI launch + window capture + VLM verification.

Tests the full pipeline:
  fc_batch (create model) -> fc_open_gui -> window_capture -> VLM analyze -> fc_close_gui

Run with: python tests/test_freecad_gui.py
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
TEST_DIR = project_root / "data" / "projects" / "gui_test"
TEST_DIR.mkdir(parents=True, exist_ok=True)
SCREENSHOT_DIR = project_root / "data" / "screenshots"
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

FCSTD_PATH = str(TEST_DIR / "gui_test_model.FCStd")
STL_PATH = str(TEST_DIR / "gui_test_model.stl")


def run_step(name: str, func, *args, **kwargs) -> bool:
    """Run a test step and print result. Returns True on success."""
    print(f"\n{'='*60}")
    print(f"STEP: {name}")
    print(f"{'='*60}")
    try:
        result = func(*args, **kwargs)
        # Truncate result for display to avoid encoding issues
        display = str(result)[:2000]
        print(f"RESULT:\n{display}")
        # Check for error indicators
        if isinstance(result, str) and result.startswith("Error"):
            print(f"FAIL: {name} returned error")
            return False
        print(f"PASS: {name}")
        return True
    except Exception as e:
        print(f"FAIL: {name} - {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    results = {}

    # ---- Step 1: Create a 3D model using fc_batch ----
    from lang3d.tools.freecad import FCBatchTool

    batch = FCBatchTool()
    step1_ok = run_step(
        "Step 1: Create 3D model (30x30x30 box with R5 center hole)",
        batch.execute,
        operations=[
            {"type": "new_doc", "name": "GuiTest"},
            {"type": "make_box", "length": 30, "width": 30, "height": 30, "name": "Box"},
            {"type": "make_cylinder", "radius": 5, "height": 30, "name": "Hole"},
            {"type": "move", "object": "Hole", "dx": 15, "dy": 15, "dz": 0},
            {"type": "boolean", "operation": "cut", "object1": "Box", "object2": "Hole", "result_name": "BoxWithHole"},
            {"type": "save", "path": FCSTD_PATH},
            {"type": "export_stl", "path": STL_PATH},
        ],
    )
    results["step1_create_model"] = step1_ok

    if not step1_ok:
        print("\nCannot continue without model. Aborting.")
        return False

    # Verify files exist
    fcstd_exists = Path(FCSTD_PATH).exists()
    stl_exists = Path(STL_PATH).exists()
    print(f"\n  File check: .FCStd exists={fcstd_exists}, .STL exists={stl_exists}")
    results["step1_files"] = fcstd_exists and stl_exists

    # ---- Step 2: Launch FreeCAD GUI ----
    from lang3d.tools.freecad import FCOpenGUITool

    open_gui = FCOpenGUITool()
    step2_ok = run_step(
        "Step 2: Launch FreeCAD GUI with model",
        open_gui.execute,
        file_path=FCSTD_PATH,
        view="isometric",
        fit_all=True,
        wait_seconds=8,
    )
    results["step2_open_gui"] = step2_ok

    # Wait extra for FreeCAD to fully load the document
    print("\n  Waiting 5s for document to fully render...")
    time.sleep(5)

    # ---- Step 3: Verify FreeCAD window is visible ----
    from lang3d.tools.screen import ListWindowsTool, WindowCaptureTool

    list_windows = ListWindowsTool()
    step3a_ok = run_step(
        "Step 3a: List windows (find FreeCAD)",
        list_windows.execute,
    )
    results["step3a_list_windows"] = step3a_ok

    # ---- Step 4: Capture FreeCAD window ----
    window_capture = WindowCaptureTool(screenshot_dir=str(SCREENSHOT_DIR))
    step4_ok = run_step(
        "Step 4: Capture FreeCAD window",
        window_capture.execute,
        title="FreeCAD",
    )
    results["step4_window_capture"] = step4_ok

    # Extract screenshot path from result
    screenshot_path = None
    if step4_ok:
        # Parse the saved path from the output
        result_str = str(step4_ok)
        for line in result_str.split("\n"):
            if "Saved to:" in line:
                potential_path = line.split("Saved to:")[-1].strip().split(" (")[0].strip()
                if Path(potential_path).exists():
                    screenshot_path = potential_path
                    break

        # Fallback: find most recent window screenshot
        if not screenshot_path:
            screenshots = sorted(SCREENSHOT_DIR.glob("window_*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
            if screenshots:
                screenshot_path = str(screenshots[0])
                print(f"\n  Found screenshot: {screenshot_path}")

    # ---- Step 5: VLM analyze the screenshot ----
    if screenshot_path and Path(screenshot_path).exists():
        from lang3d.config import load_config
        from lang3d.models.router import ModelRouter
        from lang3d.tools.vlm import VLMAnalyzeTool

        config = load_config()
        router = ModelRouter(config)
        vlm = VLMAnalyzeTool(router)

        step5_ok = run_step(
            "Step 5: VLM analyze FreeCAD screenshot",
            vlm.execute,
            image_path=screenshot_path,
            prompt=(
                "This is a screenshot of FreeCAD CAD software. "
                "Please describe what you see: "
                "1. Is FreeCAD open with a 3D model visible? "
                "2. What shape is the 3D model? (cube, cylinder, etc.) "
                "3. Does the model have any features like holes, fillets? "
                "4. What is the background color of the 3D viewport?"
            ),
            detail="standard",
        )
        results["step5_vlm_analyze"] = step5_ok

        # ---- Step 6: VLM CAD verification ----
        # Add delay to avoid rate limiting
        print("\n  Waiting 3s to avoid API rate limit...")
        time.sleep(3)

        from lang3d.tools.vlm import CADVerifyTool

        cad_verify = CADVerifyTool(router, screenshot_dir=str(SCREENSHOT_DIR))
        step6_ok = run_step(
            "Step 6: CAD verify (expect 30x30x30mm cube with R5 center hole)",
            cad_verify.execute,
            expected="A 30x30x30mm cube with a cylindrical hole of radius 5mm through the center, viewed in isometric projection",
            window_title="FreeCAD",
            detail="standard",
        )
        results["step6_cad_verify"] = step6_ok
    else:
        print("\n  SKIP: No screenshot available for VLM analysis")
        results["step5_vlm_analyze"] = False
        results["step6_cad_verify"] = False

    # ---- Step 7: Test fc_set_camera ----
    from lang3d.tools.freecad import FCSetCameraTool

    set_camera = FCSetCameraTool()
    step7_ok = run_step(
        "Step 7: Change camera to front view",
        set_camera.execute,
        file_path=FCSTD_PATH,
        view="front",
        fit_all=True,
    )
    results["step7_set_camera"] = step7_ok

    # Wait for FreeCAD to reopen
    if step7_ok:
        print("\n  Waiting 10s for FreeCAD to fully reopen and render...")
        time.sleep(10)

    # ---- Step 8: Capture with new view ----
    step8_ok = run_step(
        "Step 8: Capture FreeCAD after camera change",
        window_capture.execute,
        title="FreeCAD",
    )
    results["step8_capture_after_camera"] = step8_ok

    # ---- Step 9: Close FreeCAD ----
    from lang3d.tools.freecad import FCCloseGUITool

    close_gui = FCCloseGUITool()
    step9_ok = run_step(
        "Step 9: Close FreeCAD GUI",
        close_gui.execute,
    )
    results["step9_close_gui"] = step9_ok

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
