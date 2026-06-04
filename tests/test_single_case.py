"""Quick single-case engineering test for validation."""
from __future__ import annotations

import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lang3d.agent.core import Agent
from lang3d.config import load_config

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "projects", "engineering")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def _safe_print(text: str, max_len: int = 300) -> None:
    try:
        print(text[:max_len])
    except UnicodeEncodeError:
        safe = text[:max_len].encode("gbk", errors="replace").decode("gbk")
        print(safe)

def main():
    task = (
        "创建一个悬臂支架零件：\n"
        "1. 底板：长120mm x 宽80mm x 厚10mm\n"
        "2. 竖板：宽10mm x 高60mm x 深80mm，与底板垂直相连\n"
        "3. 底板上4个安装孔：直径8mm，位于四角（距边15mm）\n"
        "4. 竖板上有1个轴承孔：直径15mm，中心距底板顶面55mm\n"
        f"保存到 {os.path.join(OUTPUT_DIR, 'cantilever_bracket.FCStd')}\n"
        "建模完成后用 cad_verify 验证。"
    )

    print("Running cantilever bracket case...")
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
    elapsed = time.time() - start

    print(f"\n{'='*60}")
    _safe_print(f"Result: {result[:500]}")
    print(f"Tool calls: {len(tool_log)}, Time: {elapsed:.1f}s")

    fcstd = os.path.join(OUTPUT_DIR, "cantilever_bracket.FCStd")
    if os.path.exists(fcstd) and os.path.getsize(fcstd) > 0:
        print(f"SUCCESS: {fcstd} ({os.path.getsize(fcstd):,} bytes)")
    else:
        print("FAILED: file not created")

if __name__ == "__main__":
    main()
