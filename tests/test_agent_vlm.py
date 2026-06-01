"""Agent VLM Integration Test: Agent uses VLM to observe and verify.

This tests the complete loop:
1. Agent creates a 3D model in FreeCAD (headless)
2. Agent captures screen to verify visual result
3. Agent uses VLM to analyze what's on screen

The key innovation: the Agent autonomously decides to use VLM tools
for visual verification after completing a modeling task.
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lang3d.agent.core import Agent
from lang3d.config import load_config

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "projects", "vlm_test")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_agent_task(task: str, task_name: str):
    """Run a single task through the full Agent loop."""
    print(f"\n{'='*60}")
    print(f"  Agent VLM Task: {task_name}")
    print(f"{'='*60}")
    print(f"  Prompt: {task}")
    print(f"{'-'*60}")

    config = load_config()
    agent = Agent(config)

    tool_log = []

    def on_tool_call(name, args):
        tool_log.append((name, args))
        safe_args = str(args)[:100]
        print(f"  [TOOL CALL] {name}({safe_args}...)")

    def on_tool_result(name, result):
        short = result[:200] + "..." if len(result) > 200 else result
        safe = short.encode("ascii", errors="replace").decode("ascii")
        print(f"  [TOOL RESULT] {name}: {safe}")

    agent.on_tool_call(on_tool_call)
    agent.on_tool_result(on_tool_result)

    start = time.time()
    try:
        result = agent.run_task(task, use_planning=False)
    except Exception as e:
        result = f"Agent error: {e}"

    elapsed = time.time() - start
    print(f"{'-'*60}")
    safe_result = result.encode("ascii", errors="replace").decode("ascii")
    print(f"  Result: {safe_result}")
    print(f"  Tool calls: {len(tool_log)}")
    print(f"  Time: {elapsed:.1f}s")

    return result, tool_log


def main():
    print("=" * 60)
    print("  Agent + VLM Screen Perception Test")
    print("  GLM-5.1 (text) + GLM-4V-Flash (vision) + FreeCAD")
    print("=" * 60)

    # Task 1: Create a cylinder and export, then check screen
    task1 = (
        "请完成以下步骤：\n"
        "1. 用 FreeCAD 创建一个圆柱体（半径10mm，高度30mm）\n"
        "2. 导出 STL 到 "
        f"{os.path.join(OUTPUT_DIR, 'vlm_cylinder.stl')}\n"
        "3. 用 screen_capture 截图保存当前屏幕\n"
        "4. 用 vlm_analyze 分析截图，描述你看到了什么"
    )

    result1, tools1 = run_agent_task(task1, "建模 + VLM屏幕观察")

    # Check output
    stl_path = os.path.join(OUTPUT_DIR, "vlm_cylinder.stl")
    if os.path.exists(stl_path) and os.path.getsize(stl_path) > 0:
        print(f"\n  STL file created: {os.path.getsize(stl_path):,} bytes")
    else:
        print(f"\n  STL file not created")

    # Check which tool types were used
    vlm_tools_used = [(n, a) for n, a in tools1 if n in ("screen_capture", "vlm_analyze", "screen_analyze", "window_analyze")]
    fc_tools_used = [(n, a) for n, a in tools1 if n.startswith("fc_") or n == "fc_batch"]

    print(f"\n  FreeCAD tools used: {len(fc_tools_used)}")
    print(f"  VLM/screen tools used: {len(vlm_tools_used)}")
    for name, _ in vlm_tools_used:
        print(f"    - {name}")

    # Task 2: Ask agent to observe screen and describe what it sees
    task2 = (
        "请用 screen_analyze 工具截取当前屏幕，"
        "分析屏幕上正在运行哪些应用程序，并简要描述每个窗口的内容。"
    )

    result2, tools2 = run_agent_task(task2, "VLM屏幕分析")

    vlm_tools2 = [t for t, _ in tools2 if t in ("screen_capture", "vlm_analyze", "screen_analyze", "window_analyze")]
    print(f"\n  VLM/screen tools used: {len(vlm_tools2)}")

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")

    print(f"\n  Task 1 (建模+VLM):")
    for name, args in tools1:
        print(f"    - {name}({list(args.keys())})")

    print(f"\n  Task 2 (VLM屏幕分析):")
    for name, args in tools2:
        print(f"    - {name}({list(args.keys())})")

    # Success criteria:
    # - Task 1: used fc_batch + screen_capture/vlm_analyze
    # - Task 2: used screen_analyze
    t1_has_fc = len(fc_tools_used) > 0
    t1_has_vlm = len(vlm_tools_used) > 0
    t2_has_vlm = len(vlm_tools2) > 0

    print(f"\n  Task 1 CAD tools: {'PASS' if t1_has_fc else 'FAIL'}")
    print(f"  Task 1 VLM tools: {'PASS' if t1_has_vlm else 'FAIL'}")
    print(f"  Task 2 VLM tools: {'PASS' if t2_has_vlm else 'FAIL'}")

    if t1_has_fc and t1_has_vlm and t2_has_vlm:
        print("\n  AGENT + VLM INTEGRATION TEST PASSED")
    else:
        print("\n  PARTIAL PASS - some VLM integration missing")


if __name__ == "__main__":
    main()
