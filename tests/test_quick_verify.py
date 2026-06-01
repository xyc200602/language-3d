"""Quick validation: simple L-bracket with verification."""
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
    # Simpler task - just L-bracket, no complex holes
    task = (
        "用 fc_batch 创建一个 L 型支架："
        "底板 100x60x10mm + 竖板 10x60x50mm（放在底板一端上方）。"
        "然后用 fc_batch 的 save 操作保存到 "
        f"{os.path.join(OUTPUT_DIR, 'l_bracket.FCStd')}。"
        "最后用 fc_open_gui 打开文件，用 cad_verify 验证是一个 L 型支架。"
    )

    print("Running simple L-bracket test...")
    config = load_config()
    agent = Agent(config)

    tool_log = []
    result_log = []
    def on_tool_call(name, args):
        tool_log.append((name, args))
        _safe_print(f"  [CALL] {name}({list(args.keys())})")

    def on_tool_result(name, result):
        result_log.append((name, result))
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

    fcstd = os.path.join(OUTPUT_DIR, "l_bracket.FCStd")
    if os.path.exists(fcstd) and os.path.getsize(fcstd) > 0:
        print(f"SUCCESS: {fcstd} ({os.path.getsize(fcstd):,} bytes)")
    else:
        print("FAILED: file not created")

    # Check if cad_verify returned MATCH: True (check result_log, not call args)
    verify_results = [r for n, r in result_log if n == "cad_verify"]
    for vr in verify_results:
        if "MATCH: True" in vr:
            print("CAD VERIFICATION PASSED!")
            return
    print("CAD VERIFICATION: not passed (may be VLM rate limit or model hidden)")

if __name__ == "__main__":
    main()
