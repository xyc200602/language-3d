"""Integration test: CLI End-to-End test.

Tests the complete user workflow through the Agent:
  User task -> Agent planning -> Tool selection -> Execution -> Verification

This test simulates what a user would do with the CLI:
  /direct "Create a 30x30x10 plate with 4 corner holes of R3"

Run with: python tests/test_cli_e2e.py
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

    # Log tool calls
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

    print(f"Backend: {agent.router.available_backends}")
    print(f"Tools: {len(agent.tools.list_tools())} registered")
    print(f"Workspace: {agent.state.workspace}")
    print()

    # ---- Test 1: Direct mode - create a simple model ----
    print("=" * 60)
    print("Test 1: /direct - Create a simple 3D model")
    print("=" * 60)

    task1 = (
        "Create a 30x30x10mm plate with 4 corner holes of radius 3mm. "
        "Use fc_batch to create and export to STL. "
        "Save to data/projects/cli_test/plate_with_holes.FCStd and .stl"
    )

    start = time.time()
    try:
        result1 = agent.run_task(task1, use_planning=False)
        elapsed1 = time.time() - start
        print(f"\nResult: {result1[:500]}")
        print(f"Time: {elapsed1:.1f}s")
        print(f"Tool calls: {len(tool_log)}")
        test1_ok = True
    except Exception as e:
        print(f"FAIL: {e}")
        elapsed1 = time.time() - start
        test1_ok = False

    # Verify output files exist
    test1_files = False
    if test1_ok:
        fcstd = project_root / "data" / "projects" / "cli_test" / "plate_with_holes.FCStd"
        stl = project_root / "data" / "projects" / "cli_test" / "plate_with_holes.stl"
        fcstd_exists = fcstd.exists()
        stl_exists = stl.exists()
        print(f"\n  .FCStd exists: {fcstd_exists}")
        print(f"  .STL exists: {stl_exists}")
        test1_files = fcstd_exists and stl_exists

    # Check which tools were used
    tools_used = [t["name"] for t in tool_log]
    print(f"  Tools used: {tools_used}")

    # ---- Summary ----
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")

    results = {
        "test1_direct_mode": test1_ok,
        "test1_output_files": test1_files,
        "test1_tool_selection": "fc_batch" in tools_used,
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
