"""Integration test: Agent auto-fix loop.

Tests: Agent creates wrong model → cad_verify detects mismatch → auto-fix prompt injected
→ Agent fixes model → re-verify → close FreeCAD

Run with: python tests/test_auto_fix.py
"""

import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

SCREENSHOT_DIR = project_root / "data" / "screenshots"
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    from lang3d.config import load_config
    from lang3d.agent.core import Agent

    config = load_config()
    agent = Agent(config)

    tool_log = []

    def on_tool_call(name, args):
        arg_str = ", ".join(f"{k}={v!r}" for k, v in list(args.items())[:3])
        print(f"  [Tool] {name}({arg_str})")
        tool_log.append({"name": name, "args": args})

    def on_tool_result(name, result):
        preview = str(result)[:150].replace("\n", " ")
        print(f"  [Result] {preview}")

    def on_thinking(text):
        if text.strip():
            print(f"  [Thinking] {text[:150]}")

    agent.on_tool_call(on_tool_call)
    agent.on_tool_result(on_tool_result)
    agent.on_thinking(on_thinking)

    print(f"Tools: {len(agent.tools.list_tools())} registered")
    print()

    # Task: intentionally ask for something with a specific feature,
    # but the agent will create it and verify.
    # The auto-fix triggers when cad_verify returns MATCH: False.
    print("=" * 60)
    print("Auto-Fix Test: Create model → verify → auto-fix if mismatch")
    print("=" * 60)

    task = (
        "Create a 25x25x25mm cube with a center cylindrical hole of radius 4mm. "
        "Save to data/projects/auto_fix_test/cube_with_hole.FCStd and .stl. "
        "Then open FreeCAD GUI and verify with cad_verify that the model has the hole."
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

    # Analysis
    print(f"\n{'='*60}")
    print("AUTO-FIX ANALYSIS")
    print(f"{'='*60}")

    tools_used = [t["name"] for t in tool_log]
    print(f"Tool calls: {len(tool_log)}")
    print(f"Tools: {tools_used}")

    # Check if cad_verify was called
    has_verify = "cad_verify" in tools_used
    has_batch = "fc_batch" in tools_used
    has_open_gui = "fc_open_gui" in tools_used

    # Check if multiple fc_batch calls (indicates auto-fix)
    batch_count = tools_used.count("fc_batch")
    verify_count = tools_used.count("cad_verify")

    print(f"fc_batch calls: {batch_count}")
    print(f"cad_verify calls: {verify_count}")

    # Check output files
    fcstd = project_root / "data" / "projects" / "auto_fix_test" / "cube_with_hole.FCStd"
    stl = project_root / "data" / "projects" / "auto_fix_test" / "cube_with_hole.stl"
    has_files = fcstd.exists() and stl.exists()
    print(f"Output files: {has_files}")

    # Close FreeCAD if open
    if has_open_gui:
        try:
            from lang3d.tools.freecad import FCCloseGUITool
            FCCloseGUITool().execute()
        except Exception:
            pass

    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")

    results = {
        "agent_completed": test_ok,
        "used_fc_batch": has_batch,
        "used_cad_verify": has_verify,
        "output_files": has_files,
        "verification_attempted": verify_count >= 1,
    }

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
