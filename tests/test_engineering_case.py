"""Phase 5: Real engineering case end-to-end test.

Tests the complete engineering workflow:
  1. Natural language task description
  2. Agent creates 3D model with FreeCAD
  3. CAD verification with VLM
  4. Auto-fix if mismatch detected
  5. FEA structural analysis (optional)

This validates the full Language-3D Agent pipeline.
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lang3d.agent.core import Agent
from lang3d.config import load_config


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "projects", "engineering")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _safe_print(text: str, max_len: int = 300) -> None:
    """Print text safely on Windows GBK terminals."""
    try:
        print(text[:max_len])
    except UnicodeEncodeError:
        safe = text[:max_len].encode("gbk", errors="replace").decode("gbk")
        print(safe)


def run_engineering_case(task: str, case_name: str) -> tuple[str, list[tuple]]:
    """Run an engineering case through the full Agent loop."""
    print(f"\n{'='*60}")
    print(f"  Engineering Case: {case_name}")
    print(f"{'='*60}")
    _safe_print(f"  Task: {task}")
    print(f"{'-'*60}")

    config = load_config()
    agent = Agent(config)

    tool_log = []

    def on_tool_call(name, args):
        tool_log.append((name, args))
        _safe_print(f"  [CALL] {name}({list(args.keys())})")

    def on_tool_result(name, result):
        short = result[:200] + "..." if len(result) > 200 else result
        _safe_print(f"  [RESULT] {name}: {short}")

    agent.on_tool_call(on_tool_call)
    agent.on_tool_result(on_tool_result)

    start = time.time()
    try:
        result = agent.run_task(task, use_planning=False)
    except Exception as e:
        result = f"Agent error: {e}"
        _safe_print(f"  [ERROR] {e}")

    elapsed = time.time() - start
    print(f"{'-'*60}")
    _safe_print(f"  Result: {result[:500]}")
    print(f"  Tool calls: {len(tool_log)}")
    print(f"  Time: {elapsed:.1f}s")

    return result, tool_log


def main():
    print("=" * 60)
    print("  Language-3D Phase 5: Real Engineering Cases")
    print("  Testing with GLM-5.1 + FreeCAD + VLM")
    print("=" * 60)

    results = {}

    # Case 1: Cantilever Bracket (L-shaped with mounting holes)
    task1 = (
        "创建一个悬臂支架零件：\n"
        "1. 底板：长120mm x 宽80mm x 厚10mm\n"
        "2. 竖板：宽10mm x 高60mm x 深80mm，与底板垂直相连\n"
        "3. 底板上4个安装孔：直径8mm，位于四角（距边15mm）\n"
        "4. 竖板上有1个轴承孔：直径15mm，中心距底板顶面55mm\n"
        f"保存到 {os.path.join(OUTPUT_DIR, 'cantilever_bracket.FCStd')}\n"
        "建模完成后用 cad_verify 验证。"
    )
    result1, tools1 = run_engineering_case(task1, "悬臂支架")
    results["cantilever_bracket"] = {
        "result": result1,
        "tool_count": len(tools1),
        "tools": [name for name, _ in tools1],
    }

    # Verify file exists
    fcstd = os.path.join(OUTPUT_DIR, "cantilever_bracket.FCStd")
    if os.path.exists(fcstd) and os.path.getsize(fcstd) > 0:
        print(f"\n  SUCCESS: FCStd file created ({os.path.getsize(fcstd):,} bytes)")
    else:
        print(f"\n  FAILED: FCStd file not created")

    # Case 2: Simple stepped shaft
    task2 = (
        "创建一个阶梯轴零件：\n"
        "1. 第一段：直径20mm，长度30mm\n"
        "2. 第二段：直径15mm，长度25mm（与第一段同轴连接）\n"
        "3. 第三段：直径10mm，长度20mm（与第二段同轴连接）\n"
        "4. 在第二段中间有一个键槽：宽4mm x 深2mm x 长12mm\n"
        f"保存到 {os.path.join(OUTPUT_DIR, 'stepped_shaft.FCStd')}\n"
        "建模完成后用 cad_verify 验证。"
    )
    result2, tools2 = run_engineering_case(task2, "阶梯轴")
    results["stepped_shaft"] = {
        "result": result2,
        "tool_count": len(tools2),
        "tools": [name for name, _ in tools2],
    }

    fcstd2 = os.path.join(OUTPUT_DIR, "stepped_shaft.FCStd")
    if os.path.exists(fcstd2) and os.path.getsize(fcstd2) > 0:
        print(f"\n  SUCCESS: FCStd file created ({os.path.getsize(fcstd2):,} bytes)")
    else:
        print(f"\n  FAILED: FCStd file not created")

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for case_name, info in results.items():
        print(f"\n  {case_name}:")
        print(f"    Tool calls: {info['tool_count']}")
        print(f"    Tools used: {', '.join(info['tools'])}")
        _safe_print(f"    Result: {info['result'][:200]}")

    # Check files
    expected_files = {
        "cantilever_bracket": "cantilever_bracket.FCStd",
        "stepped_shaft": "stepped_shaft.FCStd",
    }
    files_ok = 0
    for case, filename in expected_files.items():
        path = os.path.join(OUTPUT_DIR, filename)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            files_ok += 1
            print(f"  [{files_ok}/2] {filename}: {os.path.getsize(path):,} bytes")
        else:
            print(f"  [MISSING] {filename}")

    if files_ok == 2:
        print("\n  ALL ENGINEERING CASES COMPLETED SUCCESSFULLY")
    else:
        print(f"\n  {2 - files_ok} case(s) did not produce output files")

    return results


if __name__ == "__main__":
    main()
