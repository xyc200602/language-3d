"""Integration test: Multimodal Agent with visual perception loop.

Tests the Agent's ability to use both text reasoning (GLM-5.1) and
visual analysis (GLM-4V) together:

1. Agent creates a 3D model
2. Agent opens FreeCAD GUI and verifies with cad_verify
3. Agent uses gui_* tools to interact with FreeCAD
4. Agent uses VLM to check the result visually
5. Agent adjusts based on visual feedback

Run with: python tests/test_multimodal_agent.py
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

SCREENSHOT_DIR = project_root / "data" / "screenshots"
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    from lang3d.config import load_config
    from lang3d.agent.core import Agent

    config = load_config()
    agent = Agent(config)

    # Track multimodal behavior
    tool_log = []
    vlm_calls = 0
    gui_calls = 0
    fc_calls = 0

    def on_tool_call(name, args):
        arg_str = ", ".join(f"{k}={v!r}" for k, v in list(args.items())[:3])
        if len(args) > 3:
            arg_str += ", ..."
        print(f"  [Tool] {name}({arg_str})")

        tool_log.append({"name": name, "args": args})

        if name in ("vlm_analyze", "screen_analyze", "window_analyze", "cad_verify"):
            nonlocal vlm_calls
            vlm_calls += 1
        elif name.startswith("gui_"):
            nonlocal gui_calls
            gui_calls += 1
        elif name.startswith("fc_"):
            nonlocal fc_calls
            fc_calls += 1

    def on_tool_result(name, result):
        preview = str(result)[:150].replace("\n", " ")
        print(f"  [Result] {preview}")

    def on_thinking(text):
        if text.strip():
            print(f"  [Thinking] {text[:150]}")

    agent.on_tool_call(on_tool_call)
    agent.on_tool_result(on_tool_result)
    agent.on_thinking(on_thinking)

    print(f"Backend: {agent.router.available_backends}")
    print(f"Tools: {len(agent.tools.list_tools())} registered")
    print()

    # ---- Multimodal Task: Create + Verify + Interact ----
    print("=" * 60)
    print("Multimodal Agent Test")
    print("Task: Create a 50x20x5mm plate with a center slot (40x8mm)")
    print("Expected: Agent uses fc_batch + fc_open_gui + cad_verify + gui_*")
    print("=" * 60)

    task = (
        "Create a 50x20x5mm rectangular plate with a center slot (40x8mm, through the thickness). "
        "Save to data/projects/multimodal_test/plate_with_slot.FCStd and .stl. "
        "Then open FreeCAD GUI to view the model, and verify with cad_verify. "
        "Use gui_scroll to zoom in on the model, then take a screenshot with gui_screenshot. "
        "Finally use vlm_analyze to describe what you see in the screenshot."
    )

    start = time.time()
    try:
        result = agent.run_task(task, use_planning=False)
        elapsed = time.time() - start
        print(f"\nResult: {str(result)[:500]}")
        print(f"Time: {elapsed:.1f}s")
        test_ok = True
    except Exception as e:
        elapsed = time.time() - start
        print(f"FAIL: {e}")
        test_ok = False

    # ---- Analysis ----
    print(f"\n{'='*60}")
    print("MULTIMODAL ANALYSIS")
    print(f"{'='*60}")

    tools_used = [t["name"] for t in tool_log]
    unique_tools = set(tools_used)

    print(f"Total tool calls: {len(tool_log)}")
    print(f"Unique tools: {len(unique_tools)}")
    print(f"VLM calls: {vlm_calls}")
    print(f"GUI calls: {gui_calls}")
    print(f"FreeCAD calls: {fc_calls}")
    print(f"Tools: {tools_used}")

    # Verify multimodal behavior
    has_fc = fc_calls > 0
    has_vlm = vlm_calls > 0
    has_gui = gui_calls > 0
    has_batch = "fc_batch" in tools_used
    has_verify = "cad_verify" in tools_used or "vlm_analyze" in tools_used

    # Verify output files
    fcstd = project_root / "data" / "projects" / "multimodal_test" / "plate_with_slot.FCStd"
    stl = project_root / "data" / "projects" / "multimodal_test" / "plate_with_slot.stl"
    has_files = fcstd.exists() and stl.exists()

    # Close FreeCAD if still open
    if "fc_open_gui" in tools_used:
        try:
            from lang3d.tools.freecad import FCCloseGUITool
            close = FCCloseGUITool()
            close.execute()
        except Exception:
            pass

    # ---- Summary ----
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")

    results = {
        "agent_completed": test_ok,
        "used_freecad": has_fc,
        "used_vlm": has_vlm,
        "used_gui": has_gui,
        "used_fc_batch": has_batch,
        "used_verification": has_verify,
        "output_files": has_files,
        "multimodal_integration": has_fc and has_vlm,
    }

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
