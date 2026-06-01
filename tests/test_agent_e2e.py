"""Agent end-to-end test: let the LLM decide which FreeCAD tools to call.

This tests the full Agent loop:
  User task (natural language)
    -> GLM-5.1 analyzes task, picks tools and parameters
    -> Agent executes tool calls
    -> GLM-5.1 sees results, decides next action
    -> ... loop until done

This is the REAL test of the Language-3D Agent framework.
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lang3d.agent.core import Agent
from lang3d.config import load_config


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "projects", "agent_test")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_agent_task(task: str, task_name: str):
    """Run a single task through the full Agent loop."""
    print(f"\n{'='*60}")
    print(f"  Agent Task: {task_name}")
    print(f"{'='*60}")
    print(f"  Prompt: {task}")
    print(f"{'-'*60}")

    config = load_config()
    agent = Agent(config)

    # Track tool calls
    tool_log = []

    def on_tool_call(name, args):
        tool_log.append((name, args))
        print(f"  [TOOL CALL] {name}({args})")

    def on_tool_result(name, result):
        # Truncate long results
        short = result[:200] + "..." if len(result) > 200 else result
        print(f"  [TOOL RESULT] {name}: {short}")

    agent.on_tool_call(on_tool_call)
    agent.on_tool_result(on_tool_result)

    start = time.time()
    try:
        result = agent.run_task(task, use_planning=False)  # direct mode
    except Exception as e:
        result = f"Agent error: {e}"

    elapsed = time.time() - start
    print(f"{'-'*60}")
    # Safe print - replace non-ascii chars for GBK console
    safe_result = result.encode("ascii", errors="replace").decode("ascii")
    print(f"  Result: {safe_result}")
    print(f"  Tool calls: {len(tool_log)}")
    print(f"  Time: {elapsed:.1f}s")

    return result, tool_log


def main():
    print("=" * 60)
    print("  Language-3D Agent End-to-End Test")
    print("  Using GLM-5.1 + FreeCAD tools")
    print("=" * 60)

    # Task 1: Simple part - a cube with a hole
    task1 = (
        "请用 FreeCAD 创建一个简单的测试零件："
        "一个边长 30mm 的立方体，中心有一个半径 5mm 的贯穿孔。"
        "然后用 fc_export_stl 导出到以下路径："
        f"{os.path.join(OUTPUT_DIR, 'agent_cube_with_hole.stl')}"
    )

    result1, tools1 = run_agent_task(task1, "立方体+贯穿孔")

    # Check output
    stl_path = os.path.join(OUTPUT_DIR, "agent_cube_with_hole.stl")
    if os.path.exists(stl_path) and os.path.getsize(stl_path) > 0:
        print(f"\n  SUCCESS: STL file created ({os.path.getsize(stl_path):,} bytes)")
    else:
        print(f"\n  FAILED: STL file not created")

    # Task 2: A mechanical part - bushing
    task2 = (
        "用 FreeCAD 创建一个套筒零件：外径 30mm，内径 20mm，高度 25mm。"
        f"导出 STL 到 {os.path.join(OUTPUT_DIR, 'agent_bushing.stl')}"
    )

    result2, tools2 = run_agent_task(task2, "套筒零件")

    stl_path2 = os.path.join(OUTPUT_DIR, "agent_bushing.stl")
    if os.path.exists(stl_path2) and os.path.getsize(stl_path2) > 0:
        print(f"\n  SUCCESS: STL file created ({os.path.getsize(stl_path2):,} bytes)")
    else:
        print(f"\n  FAILED: STL file not created")

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  Task 1 (cube+hole): {len(tools1)} tool calls")
    for name, args in tools1:
        print(f"    - {name}({list(args.keys())})")
    print(f"  Task 2 (bushing):   {len(tools2)} tool calls")
    for name, args in tools2:
        print(f"    - {name}({list(args.keys())})")

    # Check files
    files_ok = 0
    for f in ["agent_cube_with_hole.stl", "agent_bushing.stl"]:
        p = os.path.join(OUTPUT_DIR, f)
        if os.path.exists(p) and os.path.getsize(p) > 0:
            files_ok += 1
            print(f"  [{files_ok}/2] {f}: {os.path.getsize(p):,} bytes")

    if files_ok == 2:
        print("\n  ALL AGENT TASKS COMPLETED SUCCESSFULLY")
    else:
        print(f"\n  {2 - files_ok} task(s) did not produce output files")


if __name__ == "__main__":
    main()
